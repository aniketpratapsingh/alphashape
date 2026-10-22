// GPU screened-Poisson tests (M10). Every device test skips with a reason when
// the layer is unavailable, so the suite runs unchanged on a host without a
// Vulkan device.
//
// The CPU reference is the double-precision solver on the SAME assembled system
// (assemble_poisson_system + poisson_cpu_solve), which is exactly what
// screened_poisson() runs when the layer is off: the acceptance bounds are the
// ones test_poisson.cpp asserts, and the end-to-end case repeats the
// point_cloud scenario of test_build_model.cpp with the GPU solve switched on.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "common/algorithm/mesh/build_model_task.hpp"
#include "model/mesh/marching_tetrahedra.hpp"
#include "common/algorithm/mesh/poisson.hpp"
#include "common/algorithm/mesh/poisson_gpu.hpp"
#include "common/algorithm/mesh/poisson_internal.hpp"
#include "common/core/document.hpp"
#include "common/tasks/registry.hpp"
#include "dense_fixture.hpp"
#include "gpu/gpu.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/point_cloud.hpp"

#include "common/core/chunk.hpp"
#include "perf_gate.hpp"

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

namespace alphashape {
namespace {

using meshing::PoissonOptions;
using meshing::PoissonStats;
using meshing::VolumeResult;

#define REQUIRE_GPU_POISSON()                                    \
    do {                                                         \
        if (!gpu::available()) {                                 \
            GTEST_SKIP() << gpu::unavailable_reason();           \
        }                                                        \
    } while (false)

// -- fixture geometry (the same sphere as test_poisson.cpp) -------------------

PointCloud fibonacci_sphere(std::size_t count, double z_min = -1.0) {
    PointCloud cloud;
    const double golden = (1.0 + std::sqrt(5.0)) / 2.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double u = (static_cast<double>(i) + 0.5) / static_cast<double>(count);
        const double z = 1.0 - 2.0 * u;
        if (z < z_min) {
            continue;
        }
        const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double theta = 2.0 * M_PI * static_cast<double>(i) / golden;
        Eigen::Vector3d p(r * std::cos(theta), r * std::sin(theta), z);
        cloud.points.push_back(p);
        cloud.normals.push_back(p.normalized().cast<float>());
    }
    return cloud;
}

const Eigen::Vector3d kBox = Eigen::Vector3d::Constant(1.5);

double sample_field(const VoxelGrid<float>& field, const Eigen::Vector3d& p) {
    const GridSpec& spec = field.spec;
    const Eigen::Vector3d q = (p - spec.origin) / spec.voxel - Eigen::Vector3d::Constant(0.5);
    const Eigen::Vector3d base_d = q.array().floor().matrix();
    const Eigen::Vector3d frac = q - base_d;
    const int bx = static_cast<int>(base_d.x());
    const int by = static_cast<int>(base_d.y());
    const int bz = static_cast<int>(base_d.z());
    const double wx[2] = {1.0 - frac.x(), frac.x()};
    const double wy[2] = {1.0 - frac.y(), frac.y()};
    const double wz[2] = {1.0 - frac.z(), frac.z()};
    double acc = 0.0;
    for (int dz = 0; dz < 2; ++dz) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                const int x = std::min(std::max(bx + dx, 0), spec.nx - 1);
                const int y = std::min(std::max(by + dy, 0), spec.ny - 1);
                const int z = std::min(std::max(bz + dz, 0), spec.nz - 1);
                acc += wx[dx] * wy[dy] * wz[dz] *
                       static_cast<double>(field.data[spec.index(x, y, z)]);
            }
        }
    }
    return acc;
}

double crossing_radius(const VoxelGrid<float>& field, double iso,
                       const Eigen::Vector3d& direction) {
    const double step = 0.25 * field.spec.voxel;
    double previous_t = 0.0;
    double previous_v = sample_field(field, Eigen::Vector3d::Zero()) - iso;
    for (double t = step; t <= 4.0; t += step) {
        const double v = sample_field(field, t * direction) - iso;
        if (previous_v == 0.0) {
            return previous_t;
        }
        if ((previous_v < 0.0) != (v < 0.0)) {
            const double w = previous_v / (previous_v - v);
            return previous_t + w * (t - previous_t);
        }
        previous_t = t;
        previous_v = v;
    }
    return -1.0;
}

struct RadiusError {
    double median = 0.0;
    double p95 = 0.0;
    int misses = 0;
};

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const std::size_t idx = static_cast<std::size_t>(
        fraction * static_cast<double>(values.size() - 1) + 0.5);
    return values[std::min(idx, values.size() - 1)];
}

RadiusError radius_errors(const VoxelGrid<float>& field, double iso) {
    const double golden = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<double> errors;
    RadiusError out;
    for (std::size_t i = 0; i < 200; ++i) {
        const double u = (static_cast<double>(i) + 0.5) / 200.0;
        const double z = 1.0 - 2.0 * u;
        const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double theta = 2.0 * M_PI * static_cast<double>(i) / golden;
        const Eigen::Vector3d direction =
            Eigen::Vector3d(r * std::cos(theta), r * std::sin(theta), z).normalized();
        const double crossing = crossing_radius(field, iso, direction);
        if (crossing < 0.0) {
            ++out.misses;
            continue;
        }
        errors.push_back(std::abs(crossing - 1.0));
    }
    if (errors.empty()) {
        out.median = std::numeric_limits<double>::infinity();
        out.p95 = std::numeric_limits<double>::infinity();
        return out;
    }
    out.median = percentile(errors, 0.5);
    out.p95 = percentile(errors, 0.95);
    return out;
}

/// The CPU and GPU solutions of one assembled system, plus the double residual.
struct SolvePair {
    std::vector<double> cpu;
    std::vector<double> gpu;
    meshing::detail::PoissonCgStats cpu_stats;
    meshing::detail::PoissonCgStats gpu_stats;
    double gpu_true_residual = 0.0;  ///< ||b - A x_gpu|| / ||b||, double, on the CPU
};

SolvePair solve_both_ways(const PointCloud& cloud) {
    PoissonOptions options;
    options.max_resolution = 64;
    const meshing::detail::PoissonSystem system =
        meshing::detail::assemble_poisson_system(cloud, -kBox, kBox, options);
    SolvePair pair;
    pair.cpu = meshing::poisson_cpu_solve(system, options.tolerance, options.max_iterations,
                                          pair.cpu_stats);
    pair.gpu = meshing::poisson_solve_device(system, options, pair.gpu_stats);
    std::vector<double> product;
    system.apply(pair.gpu, product);
    std::vector<double> residual(system.rhs.size());
    for (std::size_t i = 0; i < residual.size(); ++i) {
        residual[i] = system.rhs[i] - product[i];
    }
    pair.gpu_true_residual =
        std::sqrt(system.dot(residual, residual)) / std::max(system.rhs_norm, 1e-300);
    return pair;
}

/// Relative L2 difference of two solution vectors.
double relative_l2(const std::vector<double>& a, const std::vector<double>& b) {
    double difference = 0.0;
    double norm = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const double delta = a[i] - b[i];
        difference += delta * delta;
        norm += b[i] * b[i];
    }
    return norm > 0.0 ? std::sqrt(difference / norm) : 0.0;
}

/// Median distance from every vertex of `a` to the nearest vertex of `b`, in
/// units of `voxel`, through a uniform spatial hash (cell size 2 voxels).
double median_vertex_distance(const Mesh& a, const Mesh& b, double voxel) {
    if (a.vertices.empty() || b.vertices.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    const double cell = 2.0 * voxel;
    std::unordered_map<std::int64_t, std::vector<std::size_t>> buckets;
    auto key = [&](int x, int y, int z) {
        return (static_cast<std::int64_t>(x) * 73856093LL) ^
               (static_cast<std::int64_t>(y) * 19349663LL) ^
               (static_cast<std::int64_t>(z) * 83492791LL);
    };
    auto hash_cell = [&](const Eigen::Vector3d& p) {
        return key(static_cast<int>(std::floor(p.x() / cell)),
                   static_cast<int>(std::floor(p.y() / cell)),
                   static_cast<int>(std::floor(p.z() / cell)));
    };
    for (std::size_t i = 0; i < b.vertices.size(); ++i) {
        buckets[hash_cell(b.vertices[i])].push_back(i);
    }
    std::vector<double> distances;
    distances.reserve(a.vertices.size());
    for (const Eigen::Vector3d& p : a.vertices) {
        const int cx = static_cast<int>(std::floor(p.x() / cell));
        const int cy = static_cast<int>(std::floor(p.y() / cell));
        const int cz = static_cast<int>(std::floor(p.z() / cell));
        double best = std::numeric_limits<double>::infinity();
        // Two rings of cells are enough: misses beyond them cannot be the median.
        for (int dz = -2; dz <= 2; ++dz) {
            for (int dy = -2; dy <= 2; ++dy) {
                for (int dx = -2; dx <= 2; ++dx) {
                    const auto it = buckets.find(key(cx + dx, cy + dy, cz + dz));
                    if (it == buckets.end()) {
                        continue;
                    }
                    for (const std::size_t index : it->second) {
                        best = std::min(best, (b.vertices[index] - p).norm());
                    }
                }
            }
        }
        if (std::isfinite(best)) {
            distances.push_back(best / voxel);
        }
    }
    return distances.empty() ? std::numeric_limits<double>::infinity()
                             : percentile(distances, 0.5);
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Same cases and bounds as test_poisson.cpp, GPU solve
// ---------------------------------------------------------------------------

TEST(PoissonGpu, SphereRadiusAccuracyMatchesCpuBounds) {
    REQUIRE_GPU_POISSON();
    const PointCloud cloud = fibonacci_sphere(20000);
    PoissonOptions options;
    options.max_resolution = 64;
    PoissonStats stats;
    const VolumeResult gpu_result =
        meshing::screened_poisson(cloud, -kBox, kBox, options, &stats);
    const SolvePair pair = solve_both_ways(cloud);
    const RadiusError gpu_error =
        radius_errors(gpu_result.field, static_cast<double>(gpu_result.iso));
    const double voxel = gpu_result.field.spec.voxel;
    VoxelGrid<float> cpu_field;
    cpu_field.spec = gpu_result.field.spec;
    cpu_field.data.assign(pair.cpu.begin(), pair.cpu.end());
    const RadiusError cpu_error =
        radius_errors(cpu_field, static_cast<double>(gpu_result.iso));
    std::printf("[poisson gpu] sphere 64^3: iterations=%d rel_residual=%.3e "
                "median_err=%.4f p95_err=%.4f voxels; cpu median=%.4f p95=%.4f\n",
                stats.iterations, stats.relative_residual, gpu_error.median / voxel,
                gpu_error.p95 / voxel, cpu_error.median / voxel, cpu_error.p95 / voxel);

    EXPECT_EQ(gpu_error.misses, 0);
    EXPECT_LT(gpu_error.median, 0.5 * voxel);
    EXPECT_LT(gpu_error.p95, 1.0 * voxel);
    const double centre = sample_field(gpu_result.field, Eigen::Vector3d::Zero());
    EXPECT_LT(centre, static_cast<double>(gpu_result.iso));
    const Eigen::Vector3d corner = kBox - Eigen::Vector3d::Constant(0.5 * voxel);
    EXPECT_GT(sample_field(gpu_result.field, corner), static_cast<double>(gpu_result.iso));
}

TEST(PoissonGpu, NoisySphereMeetsCpuBound) {
    REQUIRE_GPU_POISSON();
    PointCloud cloud = fibonacci_sphere(20000);
    PoissonOptions options;
    options.max_resolution = 64;
    const GridSpec probe = make_grid(-kBox, kBox, options.max_resolution);
    const double sigma = 0.3 * probe.voxel;
    // A deterministic jitter, as test_poisson.cpp's NormalSampler.
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    auto next_uniform = [&] {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 11) * (1.0 / 9007199254740992.0);
    };
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        const Eigen::Vector3d radial = cloud.points[i].normalized();
        Eigen::Vector3d jitter(next_uniform() - 0.5, next_uniform() - 0.5,
                               next_uniform() - 0.5);
        cloud.points[i] += sigma * jitter;
        Eigen::Vector3d axis = radial.cross(jitter);
        if (axis.norm() < 1e-9) {
            axis = radial.cross(Eigen::Vector3d::UnitX());
        }
        axis.normalize();
        const double angle = (10.0 * M_PI / 180.0) * next_uniform();
        cloud.normals[i] = (Eigen::AngleAxisd(angle, axis) * radial).normalized().cast<float>();
    }
    PoissonStats stats;
    const VolumeResult result = meshing::screened_poisson(cloud, -kBox, kBox, options, &stats);
    const RadiusError error = radius_errors(result.field, static_cast<double>(result.iso));
    const double voxel = result.field.spec.voxel;
    std::printf("[poisson gpu] noisy sphere 64^3: iterations=%d median_err=%.4f voxels\n",
                stats.iterations, error.median / voxel);
    EXPECT_EQ(error.misses, 0);
    EXPECT_LT(error.median, 1.0 * voxel);
}

TEST(PoissonGpu, HemisphereObservedMaskAndScreening) {
    REQUIRE_GPU_POISSON();
    {
        const PointCloud cloud = fibonacci_sphere(20000, 0.0);
        PoissonOptions options;
        options.max_resolution = 64;
        const VolumeResult result = meshing::screened_poisson(cloud, -kBox, kBox, options);
        ASSERT_EQ(result.observed.size(), result.field.spec.size());
        std::size_t unobserved = 0;
        for (const std::uint8_t v : result.observed) {
            if (v == 0U) {
                ++unobserved;
            }
        }
        std::printf("[poisson gpu] hemisphere unobserved voxels=%zu of %zu\n", unobserved,
                    result.observed.size());
        EXPECT_GT(unobserved, result.observed.size() / 4);
    }
    {
        const PointCloud cloud = fibonacci_sphere(15000);
        PoissonOptions screened;
        screened.max_resolution = 48;
        screened.screening = 4.0;
        PoissonOptions unscreened = screened;
        unscreened.screening = 0.0;
        const VolumeResult with = meshing::screened_poisson(cloud, -kBox, kBox, screened);
        const VolumeResult without = meshing::screened_poisson(cloud, -kBox, kBox, unscreened);
        const RadiusError a = radius_errors(with.field, static_cast<double>(with.iso));
        const RadiusError b = radius_errors(without.field, static_cast<double>(without.iso));
        const double voxel = with.field.spec.voxel;
        std::printf("[poisson gpu] screening 4.0 median=%.4f, 0.0 median=%.4f voxels\n",
                    a.median / voxel, b.median / voxel);
        EXPECT_EQ(a.misses, 0);
        EXPECT_EQ(b.misses, 0);
        EXPECT_LE(a.median, b.median + 0.05 * voxel);
    }
}

// ---------------------------------------------------------------------------
// 2. Agreement with the CPU solve and the true residual
// ---------------------------------------------------------------------------

TEST(PoissonGpu, SolutionAgreesWithCpu) {
    REQUIRE_GPU_POISSON();
    const PointCloud cloud = fibonacci_sphere(20000);
    const SolvePair pair = solve_both_ways(cloud);
    const double relative = relative_l2(pair.gpu, pair.cpu);
    const double voxel = make_grid(-kBox, kBox, 64).voxel;

    PoissonOptions options;
    options.max_resolution = 64;
    const meshing::detail::PoissonSystem system =
        meshing::detail::assemble_poisson_system(cloud, -kBox, kBox, options);
    const VolumeResult cpu_result =
        meshing::detail::finalize_poisson_result(system, pair.cpu, cloud, options);
    const VolumeResult gpu_result =
        meshing::detail::finalize_poisson_result(system, pair.gpu, cloud, options);
    const Mesh cpu_mesh = extract_isosurface(cpu_result.field, cpu_result.iso);
    const Mesh gpu_mesh = extract_isosurface(gpu_result.field, gpu_result.iso);
    const double median = median_vertex_distance(gpu_mesh, cpu_mesh, voxel);
    std::printf("[poisson gpu] agreement: relative L2=%.3e, surface median distance=%.4f "
                "voxels (%zu vs %zu vertices)\n",
                relative, median, gpu_mesh.vertices.size(), cpu_mesh.vertices.size());

    EXPECT_LT(relative, 0.05);
    EXPECT_LT(median, 0.1);
    // The true residual of the GPU solution, recomputed in double on the CPU.
    std::printf("[poisson gpu] true residual=%.3e (tolerance %.1e), gpu iterations=%d, cpu %d\n",
                pair.gpu_true_residual, options.tolerance, pair.gpu_stats.iterations,
                pair.cpu_stats.iterations);
    EXPECT_LE(pair.gpu_true_residual, options.tolerance);
}

// ---------------------------------------------------------------------------
// 3. Determinism
// ---------------------------------------------------------------------------

TEST(PoissonGpu, DeterministicRunToRun) {
    REQUIRE_GPU_POISSON();
    const PointCloud cloud = fibonacci_sphere(8000);
    const SolvePair first = solve_both_ways(cloud);
    const SolvePair second = solve_both_ways(cloud);
    EXPECT_EQ(first.gpu, second.gpu);
    EXPECT_EQ(first.gpu_stats.iterations, second.gpu_stats.iterations);
}

// ---------------------------------------------------------------------------
// 4. Fallback semantics
// ---------------------------------------------------------------------------

TEST(PoissonGpu, DisabledMatchesCpuExactly) {
    REQUIRE_GPU_POISSON();
    const PointCloud cloud = fibonacci_sphere(4000);
    PoissonOptions options;
    options.max_resolution = 32;
    const meshing::detail::PoissonSystem system =
        meshing::detail::assemble_poisson_system(cloud, -kBox, kBox, options);

    gpu::set_enabled(false);
    EXPECT_FALSE(meshing::poisson_gpu_supported(system));
    meshing::detail::PoissonCgStats disabled_stats;
    const std::vector<double> disabled =
        meshing::poisson_solve_with_fallback(system, options, disabled_stats);
    gpu::set_enabled(true);
    meshing::detail::PoissonCgStats cpu_stats;
    const std::vector<double> cpu =
        meshing::poisson_cpu_solve(system, options.tolerance, options.max_iterations, cpu_stats);
    EXPECT_EQ(disabled, cpu);
    EXPECT_EQ(disabled_stats.iterations, cpu_stats.iterations);
}

TEST(PoissonGpu, ForcedFailureFallsBackAndCounts) {
    REQUIRE_GPU_POISSON();
    const PointCloud cloud = fibonacci_sphere(4000);
    PoissonOptions options;
    options.max_resolution = 32;
    const meshing::detail::PoissonSystem system =
        meshing::detail::assemble_poisson_system(cloud, -kBox, kBox, options);

    const std::uint64_t before = gpu::failure_count();
    meshing::poisson_gpu_force_failure(true);
    meshing::detail::PoissonCgStats stats;
    const std::vector<double> solution =
        meshing::poisson_solve_with_fallback(system, options, stats);
    EXPECT_EQ(gpu::failure_count(), before + 1);
    meshing::detail::PoissonCgStats cpu_stats;
    EXPECT_EQ(solution, meshing::poisson_cpu_solve(system, options.tolerance,
                                                   options.max_iterations, cpu_stats));
}

// ---------------------------------------------------------------------------
// 5. End to end: BuildModel from the filtered point cloud, GPU solve on
// ---------------------------------------------------------------------------

TEST(PoissonGpu, BuildModelPointCloudMatchesCpuBounds) {
    REQUIRE_GPU_POISSON();
    const std::vector<dense_fixture::TruthImage> truth =
        dense_fixture::parse_truth_images("seq_truth.json");
    ASSERT_EQ(truth.size(), 8U);
    Document document = dense_fixture::aligned_document(dense_fixture::seq_files());
    Chunk& chunk = *document.active_chunk();
    dense_fixture::set_truth_poses(document, truth);
    dense_fixture::set_truth_calibration(document, 480, 360, 500.0);
    ASSERT_GT(dense_fixture::retriangulate_tracks(chunk), 100U);
    dense_fixture::run_build_depth_maps(document, /*downscale=*/2, "mild");

    // The filtered cloud test_build_model.cpp's point_cloud scenario uses.
    Params pc = make_params(builtin_registry().require("BuildPointCloud"));
    pc.set("point_colors", ParamValue{true});
    pc.set("point_confidence", ParamValue{true});
    pc.set("min_confidence", ParamValue{6.0});
    run_task(builtin_registry(), document, "BuildPointCloud", pc);

    Params params = make_params(builtin_registry().require("BuildModel"));
    params.set("source_data", ParamValue{std::string("point_cloud")});
    params.set("surface_type", ParamValue{std::string("arbitrary")});
    params.set("interpolation", ParamValue{std::string("disabled")});
    params.set("face_count", ParamValue{std::string("custom")});
    params.set("face_count_custom", ParamValue{static_cast<std::int64_t>(1000000)});
    params.set("vertex_colors", ParamValue{true});
    params.set("vertex_confidence", ParamValue{true});
    params.set("volume_resolution", ParamValue{static_cast<std::int64_t>(128)});

    const auto measure = [&](bool use_gpu) {
        gpu::set_enabled(use_gpu);
        const auto start = std::chrono::steady_clock::now();
        run_task(builtin_registry(), document, "BuildModel", params);
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        gpu::set_enabled(true);
        const Mesh& mesh = document.active_chunk()->model();
        std::vector<double> distances;
        distances.reserve(mesh.vertices.size());
        for (const Eigen::Vector3d& v : mesh.vertices) {
            distances.push_back(std::min(std::abs(v.z()), std::abs(v.y())));
        }
        std::sort(distances.begin(), distances.end());
        const std::pair<Eigen::Vector3d, Eigen::Vector3d> box =
            build_model_bbox(*document.active_chunk(), "point_cloud");
        const double voxel = (box.second - box.first).maxCoeff() / 128.0;
        struct Measurement {
            std::size_t faces = 0;
            double median = 0.0;
            double p90 = 0.0;
            double seconds = 0.0;
        } m;
        m.faces = mesh.faces.size();
        m.median = distances.empty() ? 0.0 : distances[distances.size() / 2] / voxel;
        m.p90 = distances.empty() ? 0.0 : percentile(distances, 0.9) / voxel;
        m.seconds = seconds;
        return m;
    };

    const auto gpu_run = measure(true);
    const auto cpu_run = measure(false);
    std::printf("[poisson gpu] BuildModel point_cloud: gpu faces=%zu median=%.3f p90=%.3f "
                "voxels (%.2f s) | cpu faces=%zu median=%.3f p90=%.3f voxels (%.2f s)\n",
                gpu_run.faces, gpu_run.median, gpu_run.p90, gpu_run.seconds, cpu_run.faces,
                cpu_run.median, cpu_run.p90, cpu_run.seconds);

    // The bounds test_build_model.cpp asserts for this scenario.
    EXPECT_GT(gpu_run.faces, 0U);
    EXPECT_LT(gpu_run.median, 0.5);
    EXPECT_LT(gpu_run.p90, 1.5);
    EXPECT_NEAR(static_cast<double>(gpu_run.faces), static_cast<double>(cpu_run.faces),
                0.02 * static_cast<double>(cpu_run.faces));
}

// ---------------------------------------------------------------------------
// 6. Timing: 128^3 GPU against the CPU solve on 8 threads
// ---------------------------------------------------------------------------

TEST(PoissonGpu, TimingAgainstCpu) {
    REQUIRE_GPU_POISSON();
    const PointCloud cloud = fibonacci_sphere(60000);
    PoissonOptions options;
    options.max_resolution = 128;
    const meshing::detail::PoissonSystem system =
        meshing::detail::assemble_poisson_system(cloud, -kBox, kBox, options);

    // Warm the device so pipeline compilation is not charged to the benchmark.
    meshing::detail::PoissonCgStats warm;
    meshing::poisson_solve_device(system, options, warm);

    meshing::detail::PoissonCgStats gpu_stats;
    const auto gpu_start = std::chrono::steady_clock::now();
    const std::vector<double> gpu =
        meshing::poisson_solve_device(system, options, gpu_stats);
    const double gpu_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - gpu_start).count();

    meshing::detail::PoissonCgStats cpu_stats;
    const auto cpu_start = std::chrono::steady_clock::now();
    const std::vector<double> cpu =
        meshing::poisson_cpu_solve(system, options.tolerance, options.max_iterations, cpu_stats);
    const double cpu_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - cpu_start).count();

    const double speedup = cpu_seconds / gpu_seconds;
    const int submissions = meshing::poisson_gpu_last_submissions();
    const double submissions_per_iteration =
        gpu_stats.iterations > 0
            ? static_cast<double>(submissions) / static_cast<double>(gpu_stats.iterations)
            : 0.0;
    std::printf("[poisson gpu timing] 128^3: gpu %.2f s (%d iterations, %d submissions), cpu "
                "%.2f s (%d iterations), speed-up %.2fx, refinements %d\n",
                gpu_seconds, gpu_stats.iterations, submissions, cpu_seconds,
                cpu_stats.iterations, speedup, meshing::poisson_gpu_last_refinements());
    std::printf("[poisson gpu timing] submissions per iteration: %.4f (%.1f iterations per "
                "submission, 2 dispatches per iteration)\n",
                submissions_per_iteration,
                submissions_per_iteration > 0.0 ? 1.0 / submissions_per_iteration : 0.0);
    RecordProperty("gpu_seconds", gpu_seconds);
    RecordProperty("cpu_seconds", cpu_seconds);
    RecordProperty("speedup", speedup);
    RecordProperty("submissions_per_iteration", submissions_per_iteration);

    // Batching is a property of the submission scheme, not of the device: a
    // whole CG iteration is two dispatches and a submission carries a batch of
    // them, so this must hold on llvmpipe too.
    EXPECT_GT(gpu_stats.iterations, 0);
    EXPECT_LE(submissions_per_iteration, 1.0 / 20.0)
        << "CG iterations are not being batched into submissions";

    // The solve must agree regardless of where it ran.
    EXPECT_LE(relative_l2(gpu, cpu), 0.05);

    // The same accuracy bounds test_poisson.cpp asserts for its 128^3 case.
    const VolumeResult gpu_result =
        meshing::detail::finalize_poisson_result(system, gpu, cloud, options);
    const RadiusError error = radius_errors(gpu_result.field, static_cast<double>(gpu_result.iso));
    const double voxel = gpu_result.field.spec.voxel;
    std::printf("[poisson gpu timing] 128^3 accuracy: misses=%d median=%.4f voxels\n", error.misses,
                error.median / voxel);
    EXPECT_EQ(error.misses, 0);
    EXPECT_LT(error.median, 0.75 * voxel);

    // A speed-up target is only meaningful on real hardware: on a software
    // device the "GPU" IS the CPU running behind an emulated Vulkan queue, so
    // the comparison measures dispatch overhead, not the port.
    const gpu::DeviceInfo info = gpu::default_context()->info();
    if (gpu::is_software_device(info)) {
        GTEST_SKIP() << "software device (" << info.name
                     << "): the speed-up bound applies to hardware GPUs only; measured "
                     << speedup << "x";
    }
    if (test::perf_asserts_enabled()) {
        EXPECT_GE(speedup, 3.0) << "GPU solve must be at least 3x faster than the CPU solve";
    }
}
}  // namespace alphashape
