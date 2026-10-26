#include "common/algorithm/mesh/poisson_gpu.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/utils/log.hpp"
#include "gpu/gpu.hpp"

// GPU screened-Poisson CG. The operator, the right-hand side, the Jacobi
// preconditioner and the stopping rule are the CPU solver's (poisson.cpp); only
// where the arithmetic happens changes.
//
// PRECISION. M1 has no fp64 shaders, so x, r, p, q and the stencil are float32.
// Every reduction (p.q, r.r, r.z) is a subgroup shuffle tree plus a
// per-workgroup combine in float, and the on-device CG recurrences (alpha, beta,
// rho) are float as well. The convergence DECISION is made in double on the
// host: the device's per-workgroup partials for r.r are read back and summed
// with Neumaier compensation in double (sum_partials_double), and that double
// norm is compared with the CPU solver's tolerance. float32 has ~7 significant
// digits, so the solve cannot reach a 1e-6 relative residual on its own; it
// converges to the point where each batch stops improving.
//
// SUBMISSIONS. A whole CG iteration is TWO dispatches:
//   1. poisson_spmv  -- p_k = z + beta p_{k-1}, q = A p_k, pq = sum(p_k.q),
//                       alpha = rho / pq         (fused, one 3D dispatch)
//   2. poisson_axpy  -- x += alpha p_k, r -= alpha q, rr = sum(r.r),
//                       rz = sum(r.z), beta = rz / rho, rho <- rz   (fused)
// One CG iteration is one "step" for gpu::run_batched, which times the first
// submission and sizes the rest so one submission lasts about 80 ms: a single
// command buffer for hundreds of iterations would starve a GPU shared with the
// display on a slow machine and can hang the whole system (a 16 GB M1 froze in
// Build Model that way). The residual inspection between batches still happens
// once per batch of up to kMaxIterationsPerBatch iterations -- the batching
// bound is the convergence check, and the submission bound is the display. The
// batch read-back is the 2 * groups float partials (64 KB at 128^3), not a
// vector of the size of the grid.
//
// CONVERGENCE. The accepted solution is judged by a TRUE residual recomputed in
// double on the CPU (rhs - A x), so the returned field satisfies the CPU's
// tolerance regardless of float32 drift. When float32 alone stops above it,
// iterative refinement runs on the device: the CPU forms the double correction
// right-hand side, solves A d = b - A x with the same float CG, and accumulates
// d into x in double. The sphere fixtures need exactly one such round; it is
// logged once per process and counted for the tests (poisson_gpu_last_refinements).

namespace alphashape::meshing {
namespace {

namespace g = alphashape::gpu;

constexpr std::uint32_t kLocalSize = 256;
/// Upper bound on CG iterations assembled between two host-side residual
/// inspections. Within a batch gpu::run_batched splits the work into
/// submissions that each last roughly 50-100 ms, so this bounds the convergence
/// check's granularity, not the length of a GPU job.
constexpr int kMaxIterationsPerBatch = 32;
/// Refinement rounds allowed when float32 stalls above the tolerance.
constexpr int kMaxRefinements = 6;
/// A batch must reduce the residual at least this much, or float32 has hit its
/// lattice and the round ends.
constexpr double kBatchProgress = 0.999;
/// Working vectors: device-only memory on a discrete GPU, host-visible on
/// unified memory (the layer collapses both kinds there).
constexpr g::MemoryKind kWorkingMemory = g::MemoryKind::DeviceLocal;

std::atomic<bool> g_force_failure{false};
std::atomic<int> g_last_submissions{0};
std::atomic<int> g_last_iterations{0};
std::atomic<int> g_last_refinements{0};
std::atomic<bool> g_refinement_reported{false};

/// True when the solve pipeline is expected to run on the device.
bool device_usable() { return g::available() && g::enabled(); }

std::uint32_t groups_for(std::size_t count) {
    return static_cast<std::uint32_t>((count + kLocalSize - 1) / kLocalSize);
}

struct SpmvParams {
    std::uint32_t nx;
    std::uint32_t ny;
    std::uint32_t nz;
    std::uint32_t n;
    std::uint32_t groups;
    float inv_h2;
    float alpha;
};
struct AxpyParams {
    std::uint32_t n;
    std::uint32_t groups;
};
struct VectorParams {
    std::uint32_t n;
    std::uint32_t groups;
};

static_assert(sizeof(SpmvParams) == 28, "SpmvParams must match poisson_spmv.comp");
static_assert(sizeof(AxpyParams) == 8, "AxpyParams must match poisson_axpy.comp");
static_assert(sizeof(VectorParams) == 8, "VectorParams must match poisson_precond.comp");

/// Scalar slots used by the fused reductions (see the GLSL kernels).
enum ScalarSlot : std::uint32_t {
    kRho = 0,
    kPq = 1,
    kAlpha = 2,
    kBeta = 3,
    kRr = 4,
    kRz = 5,
    kScalarCount = 8
};

/// The buffers one solve binds, shared by every dispatch of every batch.
struct DeviceState {
    g::Buffer p;         ///< p_{k-1}: the old search direction
    g::Buffer p_new;     ///< p_k: written by the stencil dispatch
    g::Buffer z;         ///< preconditioned residual r * inv_diag
    g::Buffer q;         ///< A p_k
    g::Buffer x;         ///< solution iterate
    g::Buffer r;         ///< residual
    g::Buffer inv_diag;
    g::Buffer weight;
    g::Buffer partials;
    g::Buffer scalars;
    g::Buffer counters;
    std::uint32_t groups = 0;
    std::array<std::uint32_t, 3> spmv_groups{1, 1, 1};
    std::uint32_t spmv_group_count = 0;
    /// Two prebuilt binding lists: p_old first then p_new, and the reverse.
    std::vector<const g::Buffer*> spmv_old_new;
    std::vector<const g::Buffer*> spmv_new_old;
    std::vector<const g::Buffer*> axpy_old_new;
    std::vector<const g::Buffer*> axpy_new_old;
    std::vector<const g::Buffer*> precond;
};

/// Build one submission's dispatch list: optionally the start-of-solve
/// preconditioner pass, followed by `iterations` CG steps of TWO dispatches each
/// (fused search direction + stencil + p.q, then the x/r/z update + r.r/r.z).
/// `parity` records whether p or p_new is the incoming search direction; the
/// batches alternate it, because each stencil dispatch writes the newer
/// direction into the other buffer.
std::vector<g::Dispatch> build_batch(const detail::PoissonSystem& system, DeviceState& state,
                                     int iterations, bool reset, bool& parity) {
    std::vector<g::Dispatch> dispatches;
    dispatches.reserve(static_cast<std::size_t>(iterations) * 2 + 1);
    const std::uint32_t n = static_cast<std::uint32_t>(system.rhs.size());
    const SpmvParams spmv{static_cast<std::uint32_t>(system.spec.nx),
                          static_cast<std::uint32_t>(system.spec.ny),
                          static_cast<std::uint32_t>(system.spec.nz), n, state.spmv_group_count,
                          static_cast<float>(system.inv_h2), static_cast<float>(system.alpha_s)};
    const AxpyParams axpy{n, state.groups};
    const VectorParams vector{n, state.groups};

    if (reset) {
        g::Dispatch precond;
        precond.kernel = "poisson_precond";
        precond.buffers = state.precond;
        precond.push_constants = g::push(vector);
        precond.groups = {state.groups, 1, 1};
        dispatches.push_back(std::move(precond));
    }
    for (int it = 0; it < iterations; ++it) {
        const bool incoming_is_p = !parity;
        g::Dispatch multiply;
        multiply.kernel = "poisson_spmv";
        multiply.buffers = incoming_is_p ? state.spmv_old_new : state.spmv_new_old;
        multiply.push_constants = g::push(spmv);
        multiply.groups = {state.spmv_groups[0], state.spmv_groups[1], state.spmv_groups[2]};
        dispatches.push_back(std::move(multiply));

        g::Dispatch update;
        update.kernel = "poisson_axpy";
        // The stencil just wrote the NEW direction into the other buffer, and the
        // x/r update must use it.
        update.buffers = incoming_is_p ? state.axpy_old_new : state.axpy_new_old;
        update.push_constants = g::push(axpy);
        update.groups = {state.groups, 1, 1};
        dispatches.push_back(std::move(update));
        parity = !parity;
    }
    return dispatches;
}

/// Sums `count` float partials in double with Neumaier compensation, in fixed
/// index order. The device's own combine is a float sum of the same slots; this
/// host copy is what the convergence decision and the diagnostics use, so a
/// float32 accumulation error cannot make the solve stop early or late. The
/// result is a pure function of the partials (no summation-order dependence).
double sum_partials_double(const std::vector<float>& partials, std::size_t count) {
    double total = 0.0;
    double compensation = 0.0;
    for (std::size_t k = 0; k < count && k < partials.size(); ++k) {
        const double value = static_cast<double>(partials[k]);
        const double next = total + value;
        if (std::fabs(total) >= std::fabs(value)) {
            compensation += (total - next) + value;
        } else {
            compensation += (value - next) + total;
        }
        total = next;
    }
    return total + compensation;
}

/// Double-precision residual rhs - A x.
std::vector<double> true_residual(const detail::PoissonSystem& system,
                                  const std::vector<double>& x) {
    std::vector<double> product;
    system.apply(x, product);
    std::vector<double> residual(system.rhs.size());
    for (std::size_t i = 0; i < residual.size(); ++i) {
        residual[i] = system.rhs[i] - product[i];
    }
    return residual;
}

/// ALPHASHAPE_GPU_SELF_CHECK=1: also run the CPU solver on the first solve and
/// report the difference between the two solution fields.
bool self_check_enabled() {
    static const bool value = [] {
        const char* text = std::getenv("ALPHASHAPE_GPU_SELF_CHECK");
        return text != nullptr && std::string(text) == "1";
    }();
    return value;
}

/// The CPU golden run used by ALPHASHAPE_GPU_SELF_CHECK=1.
void log_self_check(const detail::PoissonSystem& system, const std::vector<double>& gpu,
                    bool need_cpu_solution, const PoissonOptions& options,
                    const detail::PoissonCgStats& gpu_stats) {
    static std::atomic<bool> reported{false};
    if (reported.exchange(true)) {
        return;
    }
    detail::PoissonCgStats cpu_stats;
    const std::vector<double> cpu =
        poisson_cpu_solve(system, options.tolerance, options.max_iterations, cpu_stats);
    const std::vector<float> gpu_field(gpu.begin(), gpu.end());
    const std::vector<float> cpu_field(cpu.begin(), cpu.end());
    const gpu::GoldReport report = gpu::compare(gpu_field, cpu_field);
    double difference = 0.0;
    double cpu_norm = 0.0;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        const double delta = gpu[i] - cpu[i];
        difference += delta * delta;
        cpu_norm += cpu[i] * cpu[i];
    }
    const double relative = cpu_norm > 0.0 ? std::sqrt(difference / cpu_norm) : 0.0;
    log(LogLevel::Info) << "[GPU self-check] Poisson" << (need_cpu_solution ? " (solution was "
                                                                               "the CPU fallback)"
                                                                            : "")
                        << ": mean|d| " << report.mean_abs << ", max|d| " << report.max_abs
                        << ", relative L2 " << relative << "; iterations gpu "
                        << gpu_stats.iterations << " cpu " << cpu_stats.iterations
                        << "; residual gpu " << gpu_stats.relative_residual << " cpu "
                        << cpu_stats.relative_residual;
}

}  // namespace

bool poisson_gpu_supported(const detail::PoissonSystem& system) {
    if (!device_usable()) {
        return false;
    }
    // 32-bit indexing inside the kernels, and every dispatch must stay inside
    // the Vulkan-guaranteed workgroup count per axis (65535). The axpy/precond
    // dispatch is one workgroup per 256 voxels; the stencil dispatch is one
    // workgroup per 64x4 in-plane tile of one z-plane. A larger system falls
    // back to the CPU rather than issuing an out-of-range dispatch.
    constexpr std::uint64_t kMaxGroups = 65535;
    const std::size_t n = system.rhs.size();
    if (n >= std::uint64_t{1} << 31U) {
        return false;
    }
    const std::uint64_t vector_groups = (n + kLocalSize - 1) / kLocalSize;
    const std::uint64_t tiles_x =
        (static_cast<std::uint64_t>(system.spec.nx) + 63U) / 64U;
    const std::uint64_t tiles_y =
        (static_cast<std::uint64_t>(system.spec.ny) + 3U) / 4U;
    const std::uint64_t tiles_z = static_cast<std::uint64_t>(system.spec.nz);
    if (vector_groups > kMaxGroups || tiles_x > kMaxGroups || tiles_y > kMaxGroups ||
        tiles_z > kMaxGroups) {
        return false;
    }
    // The 3D dispatch's linearised group id and the group product must also fit.
    return tiles_x * tiles_y * tiles_z < std::uint64_t{1} << 31U &&
           vector_groups * 2 * sizeof(float) <=
               std::numeric_limits<std::uint32_t>::max();
}

void poisson_gpu_force_failure(bool value) { g_force_failure.store(value); }
int poisson_gpu_last_submissions() { return g_last_submissions.load(); }
int poisson_gpu_last_iterations() { return g_last_iterations.load(); }
int poisson_gpu_last_refinements() { return g_last_refinements.load(); }

std::vector<double> poisson_solve_device(const detail::PoissonSystem& system,
                                         const PoissonOptions& options,
                                         detail::PoissonCgStats& stats) {
    const std::shared_ptr<g::Context> context = g::default_context();
    if (!context) {
        throw g::Error("no GPU context");
    }
    if (g_force_failure.exchange(false)) {
        throw g::Error("forced failure (test hook)");
    }

    const std::size_t n = system.rhs.size();
    const std::uint32_t groups = groups_for(n);
    DeviceState state;
    state.groups = groups;
    // spmv: 64 x 4 threads per workgroup over a 64 x 4 voxel tile of one
    // z-plane, so the six in-plane stencil taps are coalesced plane reads.
    state.spmv_groups = {
        (static_cast<std::uint32_t>(system.spec.nx) + 63U) / 64U,
        (static_cast<std::uint32_t>(system.spec.ny) + 3U) / 4U,
        static_cast<std::uint32_t>(system.spec.nz)};
    state.spmv_group_count = state.spmv_groups[0] * state.spmv_groups[1] * state.spmv_groups[2];
    // The large working vectors live in device-only memory on a discrete GPU
    // (staged host copies); the per-batch reduction partials and the tiny
    // scalar/counter buffers stay host-visible, because they are read back every
    // batch and a staged readback would cost an extra copy each time.
    state.p = g::Buffer::of(*context, std::vector<float>(n, 0.0F), kWorkingMemory);
    state.p_new = g::make_buffer(*context, n * sizeof(float), kWorkingMemory);
    state.z = g::make_buffer(*context, n * sizeof(float), kWorkingMemory);
    state.q = g::make_buffer(*context, n * sizeof(float), kWorkingMemory);
    state.x = g::Buffer::of(*context, std::vector<float>(n, 0.0F), kWorkingMemory);
    state.r = g::Buffer::of(*context, system.rhs_f, kWorkingMemory);
    state.inv_diag = g::Buffer::of(*context, system.inv_diag_f, kWorkingMemory);
    state.weight = g::Buffer::of(*context, system.weight_f, kWorkingMemory);
    state.partials = g::make_buffer(
        *context, 2 * std::max(groups, state.spmv_group_count) * sizeof(float),
        g::MemoryKind::HostVisible);
    state.scalars = g::Buffer::of(*context, std::vector<float>(kScalarCount, 0.0F),
                                  g::MemoryKind::HostVisible);
    state.counters = g::Buffer::of(*context, std::vector<std::uint32_t>(4, 0U),
                                   g::MemoryKind::HostVisible);
    // precond: rho = sum(r * (r * inv_diag)); read-only, no direction written.
    state.precond = {&state.r, &state.inv_diag, &state.partials, &state.scalars,
                     &state.counters, &state.z};
    // The two stencil binding orders: (p_old = p, p_new = p_new) and the reverse.
    // z is not a buffer: the stencil forms r * inv_diag on the fly.
    // Binding order: p_old, p_new, q, weight, partials, scalars, counters, r,
    // inv_diag (the dispatcher binds buffers[i] to binding i).
    // Binding order: p_old, p_new, q, weight, partials, scalars, counters, z.
    state.spmv_old_new = {&state.p,     &state.p_new, &state.q,      &state.weight,
                          &state.partials, &state.scalars, &state.counters, &state.z};
    state.spmv_new_old = {&state.p_new, &state.p,     &state.q,      &state.weight,
                          &state.partials, &state.scalars, &state.counters, &state.z};
    // The update reads the direction the stencil just wrote.
    // Binding order: p, q, x, r, inv_diag, partials, scalars, counters, then the
    // same five vectors re-bound as float4 aliases for the quad path.
    state.axpy_old_new = {&state.p_new, &state.q,     &state.x,      &state.r,
                          &state.inv_diag, &state.partials, &state.scalars, &state.counters,
                          &state.z};
    state.axpy_new_old = {&state.p,     &state.q,     &state.x,      &state.r,
                          &state.inv_diag, &state.partials, &state.scalars, &state.counters,
                          &state.z};

    std::vector<double> accumulation(n, 0.0);
    std::vector<double> residual = system.rhs;
    double residual_norm = system.rhs_norm;
    stats.iterations = 0;
    stats.relative_residual = residual_norm > 0.0 ? 1.0 : 0.0;
    if (residual_norm == 0.0) {
        g_last_iterations.store(0);
        g_last_submissions.store(0);
        return accumulation;
    }
    const double target = options.tolerance * std::max(system.rhs_norm, 1e-300);

    int submissions = 0;
    int refinements = 0;
    bool refined = false;
    // One CG iteration is two dispatches and a submission carries a batch of
    // them. The batch size is timed rather than fixed: the first submission is
    // a single iteration, whose duration then scales the batch towards ~80 ms,
    // so a slow device cannot be handed a multi-second job (on a GPU shared
    // with the display that starves the compositor and can hang the machine).
    // The limiter persists across refinement rounds, so the measured cost of
    // this problem carries over.
    g::BatchLimiter limiter(/*dispatches_per_step=*/2, /*initial_steps=*/1,
                            /*max_steps=*/kMaxIterationsPerBatch);
    for (int round = 0; round <= kMaxRefinements; ++round) {
        // The inner solve always starts from zero with r = current residual.
        state.x.upload(std::vector<float>(n, 0.0F).data(), n * sizeof(float));
        std::vector<float> r0(n);
        for (std::size_t i = 0; i < n; ++i) {
            r0[i] = static_cast<float>(residual[i]);
        }
        state.r.upload(r0.data(), n * sizeof(float));

        // Continuous CG across batches: the Krylov recursion (p, r, rho) is
        // never restarted inside a round, only the residual is inspected.
        int inner = 0;
        bool parity = false;
        double previous = residual_norm;
        while (inner < options.max_iterations - stats.iterations) {
            const int batch = std::min(kMaxIterationsPerBatch,
                                       options.max_iterations - stats.iterations - inner);
            if (batch <= 0) {
                break;
            }
            submissions += static_cast<int>(g::run_batched(
                *context, build_batch(system, state, batch, inner == 0, parity), limiter,
                /*leading_dispatches=*/inner == 0 ? 1 : 0));
            inner += batch;
            // The workgroup partials -- not the float combine the last workgroup
            // folded them into -- are read back and re-summed in double on the
            // host, so the convergence decision uses a compensated double sum of
            // the device's per-workgroup partials.
            const std::vector<float> partials = state.partials.read<float>(2 * groups);
            const double batch_rr = sum_partials_double(partials, groups);
            const double batch_residual = std::sqrt(std::max(batch_rr, 0.0));
            if (!(batch_residual > 0.0) || batch_residual <= target) {
                break;
            }
            // A full batch that cannot improve the residual by 0.1% has hit the
            // float32 lattice; the true residual then decides whether to refine.
            if (!(batch_residual < previous * kBatchProgress)) {
                break;
            }
            previous = batch_residual;
        }
        stats.iterations += inner;

        const std::vector<float> solution = state.x.read<float>(n);
        for (std::size_t i = 0; i < n; ++i) {
            accumulation[i] += static_cast<double>(solution[i]);
        }
        residual = true_residual(system, accumulation);
        residual_norm = std::sqrt(system.dot(residual, residual));
        stats.relative_residual = residual_norm / system.rhs_norm;
        if (stats.relative_residual <= options.tolerance) {
            break;
        }
        if (round + 1 > kMaxRefinements) {
            break;
        }
        ++refinements;
        refined = true;
    }

    g_last_submissions.store(submissions);
    g_last_iterations.store(stats.iterations);
    g_last_refinements.store(refinements);
    if (refined && !g_refinement_reported.exchange(true)) {
        log(LogLevel::Info) << "[GPU] Poisson solve: float32 CG stalled above tolerance; "
                            << refinements << " double-precision refinement round(s) used ("
                            << stats.iterations << " inner iterations, residual "
                            << stats.relative_residual << ")";
    }
    return accumulation;
}

std::vector<double> poisson_solve_with_fallback(const detail::PoissonSystem& system,
                                                const PoissonOptions& options,
                                                detail::PoissonCgStats& stats) {
    if (!poisson_gpu_supported(system)) {
        return poisson_cpu_solve(system, options.tolerance, options.max_iterations, stats);
    }
    const std::uint64_t failures_before = g::failure_count();
    std::vector<double> solution = g::with_fallback<std::vector<double>>(
        "Poisson", [&] { return poisson_solve_device(system, options, stats); },
        [&] { return poisson_cpu_solve(system, options.tolerance, options.max_iterations, stats); });

    if (self_check_enabled()) {
        const bool fell_back = g::failure_count() != failures_before;
        log_self_check(system, solution, fell_back, options, stats);
    }
    return solution;
}

}  // namespace alphashape::meshing
