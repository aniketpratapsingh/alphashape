#include "common/algorithm/mesh/tv_fusion_gpu.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/utils/log.hpp"
#include "gpu/gpu.hpp"

// GPU TV-L1 range-image fusion: the same estimate as tv_l1_fusion(), on the
// gpu:: layer.
//
// Split of work. The histogram integration and the surface initialisation
// (tv_fusion.cpp stages 2-3: the ray-march scatter through relaxed atomics, the
// dilated observed mask and the breadth-first nearest-neighbour extension) stay
// on the CPU. They run once per call, and the profile shows the 200-300
// Chambolle-Pock iterations are the hot part (30% + 7% of BuildModel against a
// fraction of a percent for the scatter): moving the scatter would buy little
// and duplicate a subtle setup. It is measured in the tests.
//
// Device loop. tvl1_dual.comp is the dual ascent + projection, tvl1_primal.comp
// is the primal prox + over-relaxed extrapolation (both transcribed from
// tv_fusion.cpp, including the Neumann boundaries and the exact adjoint
// divergence). One gpu::run carries a whole batch of B iteration pairs, so the
// submission cost is amortised over B*2 dispatches instead of paid per
// iteration. Between batches the host applies the CPU's stopping spirit: the
// mean |u_new - u_old|, accumulated as per-workgroup partials (tvl1_delta +
// tvl1_reduce) and summed on the CPU in double. The batch doubles on every
// settled check (B, 2B, 4B, ... capped at kMaxBatch), which keeps a problem
// that converges in 20 iterations at one round trip and one that needs 300 at
// about seven, not three hundred.
//
// Precision. float32 throughout (M1 has no fp64), while the CPU's divergence,
// prox and step sizes run in double. The histogram table is the CPU's exact
// double->float conversion. The result therefore agrees with the CPU to about
// the float noise of the iteration, not bit for bit (the tests print both).

namespace alphashape::meshing {
namespace {

namespace g = alphashape::gpu;

constexpr int kMaxBins = 16;  // TVL1_MAX_BINS in tvl1_common.glsl
// Workgroup shapes shared with the kernels. The dual kernel unrolls four
// consecutive x voxels per invocation (contiguous float4 accesses, the
// bandwidth-friendly form); the primal and delta kernels handle one voxel per
// invocation with a taller workgroup. Keep each in step with its .comp file's
// layout(local_size_*) and pc stride assumption.
constexpr std::uint32_t kDualLanesX = 32;
// The y extents complete the workgroup shapes but are not needed by the host
// arithmetic (which only divides along x and by the resident row count); they
// stay as documentation of the layout, marked so clang's
// -Wunused-const-variable accepts them.
[[maybe_unused]] constexpr std::uint32_t kDualLanesY = 4;
constexpr std::uint32_t kDualStride = 4;
constexpr std::uint32_t kPointLanesX = 32;
[[maybe_unused]] constexpr std::uint32_t kPointLanesY = 8;
constexpr std::uint32_t kPointStride = 4;
constexpr std::uint32_t kFirstBatch = 8;
constexpr std::uint32_t kMaxBatch = 64;
/// Mean |u_new - u_old| below this stops a batch early. The CPU runs its full
/// iteration budget, so this only triggers once the iterate has effectively
/// stopped moving.
constexpr double kSettleTolerance = 1.0e-15;

std::atomic<bool> g_force_failure{false};
std::mutex g_profile_mutex;
TvFusionGpuProfile g_profile;

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

/// Push-constant block, byte-for-byte Tvl1Params in tvl1_common.glsl. Inert
/// second members are named "_unused" so the shader compiler keeps them.
struct Tvl1Params {
    std::uint32_t nx;
    std::uint32_t ny;
    std::uint32_t nz;
    std::uint32_t bins;
    std::uint32_t n;
    std::uint32_t iterations;
    float tau;
    float sigma;
    float data_weight;
    float tolerance_unused;
    std::uint32_t voxel_count;
    std::uint32_t resident;
    std::uint32_t x_blocks;
    std::uint32_t pad_unused;
};

static_assert(sizeof(Tvl1Params) == 56, "Tvl1Params must match the GLSL push-constant size");
static_assert(offsetof(Tvl1Params, n) == 16, "Tvl1Params layout drifted");
static_assert(offsetof(Tvl1Params, tau) == 24, "Tvl1Params layout drifted");
static_assert(offsetof(Tvl1Params, data_weight) == 32, "Tvl1Params layout drifted");
static_assert(offsetof(Tvl1Params, tolerance_unused) == 36, "Tvl1Params layout drifted");
static_assert(offsetof(Tvl1Params, voxel_count) == 40, "Tvl1Params layout drifted");
static_assert(offsetof(Tvl1Params, resident) == 44, "Tvl1Params layout drifted");
static_assert(offsetof(Tvl1Params, x_blocks) == 48, "Tvl1Params layout drifted");

/// True when ALPHASHAPE_GPU_SELF_CHECK=1: on the first GPU fusion of the
/// process, also run the CPU solver and log gpu::compare stats.
bool self_check_enabled() {
    static const bool value = [] {
        const char* text = std::getenv("ALPHASHAPE_GPU_SELF_CHECK");
        return text != nullptr && std::string(text) == "1";
    }();
    return value;
}

/// Reduction launch: strided segments, groups_x * groups_y under the limit.
std::array<std::uint32_t, 3> reduce_groups(std::uint32_t groups) {
    const std::uint32_t groups_x = std::min(groups, 256U);
    return {groups_x, (groups + groups_x - 1) / groups_x, 1};
}

/// Fixed group count for the reduce kernel (independent of the grid).
std::uint32_t reduce_group_count() { return 1024U; }

/// The per-workgroup partial sums, summed on the CPU in double.
double total_delta(const std::vector<float>& partials) {
    double sum = 0.0;
    for (const float value : partials) {
        sum += static_cast<double>(value);
    }
    return sum;
}

/// Push constants with the per-dispatch fields filled in.
Tvl1Params make_params(const GridSpec& grid, int bins, std::size_t voxels,
                       std::uint32_t resident, std::uint32_t x_blocks, double lambda,
                       std::uint32_t iterations, std::uint32_t reduce_len) {
    const double tau = 1.0 / std::sqrt(12.0);
    Tvl1Params params{};
    params.nx = static_cast<std::uint32_t>(grid.nx);
    params.ny = static_cast<std::uint32_t>(grid.ny);
    params.nz = static_cast<std::uint32_t>(grid.nz);
    params.bins = static_cast<std::uint32_t>(bins);
    params.voxel_count = static_cast<std::uint32_t>(voxels);
    params.n = resident;
    params.x_blocks = x_blocks;
    params.iterations = iterations;
    params.tau = static_cast<float>(tau);
    params.sigma = static_cast<float>(tau);
    params.data_weight = static_cast<float>(tau * lambda);
    params.tolerance_unused = static_cast<float>(kSettleTolerance);
    params.resident = reduce_len;
    return params;
}

/// A dispatch with the given kernel, buffers and push constants.
g::Dispatch dispatch_for(const char* kernel, std::vector<const g::Buffer*> buffers,
                         const Tvl1Params& params, std::array<std::uint32_t, 3> groups) {
    g::Dispatch dispatch;
    dispatch.kernel = kernel;
    dispatch.buffers = std::move(buffers);
    dispatch.push_constants = g::push(params);
    dispatch.groups = groups;
    return dispatch;
}

/// The GPU run. Throws gpu::Error on any device failure; the caller wraps it in
/// with_fallback, which is why setup is read, not consumed: a failed run must
/// leave the CPU fallback its inputs.
VolumeResult run_gpu(const detail::TvFusionSetup& setup, const TvFusionOptions& options) {
    const std::shared_ptr<g::Context> context = g::default_context();
    if (!context) {
        throw g::Error("no GPU context");
    }
    // Unreachable through tv_l1_fusion_device(), which declines a software
    // device before this point. Kept as a backstop because the failure mode is
    // a process abort inside llvmpipe's LLVM JIT, not a recoverable error: a
    // counted failure is vastly preferable to losing the host process.
    if (!tv_l1_fusion_gpu_fp16_available(*context)) {
        throw g::Error("device cannot compile the fp16 dual kernel: " + context->info().name);
    }
    if (g_force_failure.exchange(false)) {
        throw g::Error("forced failure (test hook)");
    }

    const GridSpec& grid = setup.grid;
    const std::size_t voxels = grid.size();
    const int bins = setup.bins;
    if (bins < 2 || bins > kMaxBins) {
        throw g::Error("TV-L1 fusion needs 2.." + std::to_string(kMaxBins) +
                       " histogram bins on the GPU");
    }

    const auto upload_start = std::chrono::steady_clock::now();
    TvFusionGpuProfile profile;
    // -- device buffers ------------------------------------------------------
    // The CPU setup's initial field is the first iterate; the histograms are
    // the CPU's row-normalised weights, uploaded once. The dual fields start at
    // zero, exactly the CPU's px/py/pz.
    g::Buffer g_u = g::Buffer::of(*context, setup.initial_field);
    g::Buffer g_ubar = g::Buffer::of(*context, setup.initial_field);
    // The dual field is packed half precision, two uints per voxel (see
    // tvl1_dual.comp): |p| <= 1, so fp16 resolution is far below the field's
    // scale while the stream shrinks by 6 bytes per voxel. Starts at zero,
    // exactly the CPU's px/py/pz.
    const std::vector<std::uint32_t> zero_dual(voxels * 2, 0U);
    g::Buffer g_p = g::Buffer::of(*context, zero_dual);
    // Compact the dense V x bins histograms to the measured voxels. Unmeasured
    // rows are all zero, which the CPU's prox answers with clamp(v), so the
    // kernel skips their read via kNoRow. In a typical build only 6-11% of
    // voxels carry data, so the dense table would dominate memory traffic.
    constexpr std::uint32_t kNoRow = 0xFFFFFFFFU;
    std::vector<std::uint32_t> row_start(voxels, kNoRow);
    std::vector<float> rows;
    rows.reserve(setup.weights.size() / 4);
    std::size_t measured_rows = 0;
    for (std::size_t v = 0; v < voxels; ++v) {
        const double* source = setup.weights.data() + v * static_cast<std::size_t>(bins);
        bool any = false;
        for (int k = 0; k < bins; ++k) {
            any = any || source[k] > 0.0;
        }
        if (!any) {
            continue;
        }
        row_start[v] = static_cast<std::uint32_t>(measured_rows++);
        for (int k = 0; k < bins; ++k) {
            rows.push_back(static_cast<float>(source[k]));
        }
    }
    g::Buffer g_rows = g::Buffer::of(*context, rows);
    g::Buffer g_row_start = g::Buffer::of(*context, row_start);

    // Every row is dispatched and every 128-voxel x-block of every row runs; the
    // second grid dimension is rounded up per kernel shape (the dual kernel
    // packs 2 rows per group in y, the primal 8), so the kernel guards its tail
    // with kCulled. The launch therefore covers the whole volume.
    //
    // An earlier version culled blocks whose voxels were unmeasured with a
    // constant 3x3x3 neighbourhood, on the claim that such a voxel is a fixed
    // point. It is only a fixed point while its neighbours stay constant: a
    // change two voxels away reaches it after a few iterations, and the culled
    // voxel then never moved, so the GPU field drifted from the CPU's in the
    // unmeasured region (M1: mean 5e-2 at 200 iterations vs a 2e-2 bound).
    // Exactness wins.
    //
    // The block list is the identity over rows (block_list[slot] == slot). Both
    // kernels read the entry as a ROW index (`index = row * nx + x`), so a list
    // of flattened block ids (`row * x_blocks + block`) is only equivalent while
    // `x_blocks == 1`, i.e. `nx <= 128`; on a 192^3 grid (x_blocks == 2) it made
    // the kernels read and write past the volume and Metal raised a GPU address
    // fault. With nothing culled the identity is the correct list.
    std::vector<std::uint32_t> block_list(static_cast<std::size_t>(grid.ny) *
                                          static_cast<std::size_t>(grid.nz));
    for (std::uint32_t slot = 0; slot < block_list.size(); ++slot) {
        block_list[slot] = slot;
    }
    g::Buffer g_blocks = g::Buffer::of(*context, block_list);
    g::Buffer g_previous = g::Buffer::of(*context, setup.initial_field);
    g::Buffer g_delta = g::make_buffer(*context, voxels * sizeof(float));

    // One dispatched slot per (y, z) row.
    const std::uint32_t resident = static_cast<std::uint32_t>(block_list.size());
    const std::array<std::uint32_t, 3> dual_groups = {
        static_cast<std::uint32_t>((grid.nx + static_cast<int>(kDualLanesX) *
                                                     static_cast<int>(kDualStride) - 1) /
                                   (static_cast<int>(kDualLanesX) *
                                    static_cast<int>(kDualStride))),
        (resident + 1) / 2, 1};
    const std::array<std::uint32_t, 3> point_groups = {
        static_cast<std::uint32_t>((grid.nx + static_cast<int>(kPointLanesX) *
                                                     static_cast<int>(kPointStride) - 1) /
                                   (static_cast<int>(kPointLanesX) *
                                    static_cast<int>(kPointStride))),
        (resident + 7) / 8, 1};
    const std::uint32_t group_count = reduce_group_count();
    const std::array<std::uint32_t, 3> reduce_launch = reduce_groups(group_count);
    g::Buffer g_partials = g::make_buffer(*context, group_count * sizeof(float));

    const std::vector<const g::Buffer*> dual_buffers = {&g_ubar, &g_p, &g_blocks};
    const std::vector<const g::Buffer*> primal_buffers = {
        &g_u, &g_ubar, &g_p, &g_rows, &g_row_start, &g_blocks};
    const std::vector<const g::Buffer*> delta_buffers = {&g_u, &g_previous, &g_delta};
    const std::vector<const g::Buffer*> reduce_buffers = {&g_delta, &g_partials};

    // One gpu::run per batch: the batch's dual/primal pairs, then the settle
    // statistic (delta against the pre-batch iterate, reduce to partials). The
    // host reads the partials and sums them in double.
    profile.upload_seconds = seconds_since(upload_start);
    const auto iterate_start = std::chrono::steady_clock::now();
    int remaining = std::max(0, options.iterations);
    std::uint32_t batch = kFirstBatch;
    double movement = 0.0;
    while (remaining > 0) {
        const std::uint32_t count = std::min<std::uint32_t>(
            batch, static_cast<std::uint32_t>(remaining));
        const Tvl1Params params =
            make_params(grid, bins, voxels, resident, static_cast<std::uint32_t>(
                                                    (grid.nx + 127) / 128),
                        options.lambda, count, group_count);
        std::vector<g::Dispatch> dispatches;
        dispatches.reserve(static_cast<std::size_t>(count) * 2 + 2);
        for (std::uint32_t it = 0; it < count; ++it) {
            dispatches.push_back(dispatch_for("tvl1_dual", dual_buffers, params, dual_groups));
            dispatches.push_back(
                dispatch_for("tvl1_primal", primal_buffers, params, point_groups));
        }
        dispatches.push_back(dispatch_for("tvl1_delta", delta_buffers, params, point_groups));
        dispatches.push_back(
            dispatch_for("tvl1_reduce", reduce_buffers, params, reduce_launch));
        g::run(*context, dispatches);
        movement = total_delta(g_partials.read<float>(group_count)) /
                   static_cast<double>(voxels);
        if (Log::enabled(LogLevel::Debug)) {
            log(LogLevel::Debug) << "TV-L1 fusion: batch of " << count
                                 << " iterations, mean |du| " << movement;
        }
        remaining -= static_cast<int>(count);
        ++profile.batches;
        if (remaining <= 0 || movement <= kSettleTolerance) {
            break;
        }
        // The next batch's pre-iterate is this batch's output.
        g_previous.upload(g_u.read<float>(voxels).data(), voxels * sizeof(float));
        batch = std::min<std::uint32_t>(batch * 2U, kMaxBatch);
    }

    profile.iterate_seconds = seconds_since(iterate_start);
    const auto readback_start = std::chrono::steady_clock::now();
    VolumeResult result;
    result.field = VoxelGrid<float>(grid, 0.0F);
    result.field.data = g_u.read<float>(voxels);
    profile.readback_seconds = seconds_since(readback_start);
    {
        const std::lock_guard<std::mutex> lock(g_profile_mutex);
        g_profile = profile;
    }
    result.iso = 0.0F;
    result.observed = setup.observed;
    return result;
}

/// ALPHASHAPE_GPU_SELF_CHECK=1: compare the GPU field against the CPU's on the
/// same setup, once per process.
void log_self_check(const VolumeResult& gpu_result, const detail::TvFusionSetup& setup_cpu_copy,
                    const TvFusionOptions& options) {
    static std::atomic<bool> done{false};
    if (!done.exchange(true)) {
        const VolumeResult cpu_result =
            detail::tv_l1_fusion_solve(setup_cpu_copy, options);
        const g::GoldReport report = g::compare(gpu_result.field.data, cpu_result.field.data);
        log(LogLevel::Info) << "TV-L1 fusion self-check: " << report.count
                            << " voxels, mean |gpu - cpu| " << report.mean_abs
                            << ", max " << report.max_abs;
    }
}

}  // namespace

bool tv_l1_fusion_gpu_fp16_available(const gpu::Context& context) {
    // tvl1_dual.comp stores p as two packed uints per voxel, using
    // packHalf2x16/unpackHalf2x16 (core GLSL, so no 16-bit storage extension is
    // required) and keeps every accumulation in fp32. Mesa's llvmpipe still
    // cannot compile that pattern: LLVM 20 aborts the whole process inside its
    // JIT with "Cannot select: load<(load (s16) ...), anyext from i16>", which
    // is a hard exit rather than a VkResult, so with_fallback cannot absorb it.
    // Every hardware driver the engine supports compiles it, so the capability
    // test is exactly the software-device test.
    return !g::is_software_device(context.info());
}

bool tv_l1_fusion_gpu_supported() {
    if (!g::available() || !g::enabled()) {
        return false;
    }
    // A software device declines here, before any tvl1 pipeline is created (see
    // tv_l1_fusion_gpu_fp16_available for why that ordering is load-bearing).
    const std::shared_ptr<g::Context> context = g::default_context();
    return context != nullptr && tv_l1_fusion_gpu_fp16_available(*context);
}

bool tv_l1_fusion_gpu_would_run() {
    // Same decision tv_l1_fusion_device() makes, so the test hook agrees with
    // what actually happens. The decision depends only on the device, never on
    // the inputs (the kernels have no size limits), hence no arguments.
    return tv_l1_fusion_gpu_supported();
}
TvFusionGpuProfile tv_fusion_gpu_last_profile() {
    const std::lock_guard<std::mutex> lock(g_profile_mutex);
    return g_profile;
}
void tv_fusion_gpu_force_failure(bool value) { g_force_failure.store(value); }

VolumeResult tv_l1_fusion_device(const std::vector<FusionView>& views,
                                 const Eigen::Vector3d& bbox_min,
                                 const Eigen::Vector3d& bbox_max,
                                 const TvFusionOptions& options) {
    // Validate up front with the CPU's own rules, so a bad call throws before
    // any device work and is not absorbed by the fallback (which only catches
    // device failures; a validation error is thrown outside with_fallback).
    const auto setup_start = std::chrono::steady_clock::now();
    detail::TvFusionSetup setup = detail::tv_l1_fusion_setup(views, bbox_min, bbox_max, options);
    const double setup_seconds = seconds_since(setup_start);
    // The dual kernel packs p as fp16. Mesa's llvmpipe cannot compile that
    // pattern: LLVM 20 aborts the whole process with "Cannot select:
    // load<(load (s16) ...), anyext from i16>" inside its JIT (a hard exit, not
    // a VkResult, so the fallback cannot catch it). CPU drivers therefore take
    // the CPU path here -- and on a software device that is no loss at all.
    // The decline happens before any buffer, kernel or pipeline is created, so
    // gpu::failure_count() is untouched: nothing was dispatched.
    if (!tv_l1_fusion_gpu_supported() || setup.bins > kMaxBins) {
        return detail::tv_l1_fusion_solve(std::move(setup), options);
    }
    const bool check = self_check_enabled();
    const detail::TvFusionSetup cpu_copy = check ? setup : detail::TvFusionSetup{};
    VolumeResult result = g::with_fallback<VolumeResult>(
        "TV-L1 fusion",
        [&] { return run_gpu(std::move(setup), options); },
        [&] { return detail::tv_l1_fusion_solve(std::move(setup), options); });
    {
        const std::lock_guard<std::mutex> lock(g_profile_mutex);
        g_profile.setup_seconds = setup_seconds;
    }
    if (check) {
        log_self_check(result, cpu_copy, options);
    }
    return result;
}

}  // namespace alphashape::meshing
