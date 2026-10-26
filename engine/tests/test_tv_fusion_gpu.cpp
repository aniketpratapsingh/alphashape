// GPU TV-L1 fusion tests (M10). Every device test skips with a reason when the
// layer is unavailable, so the suite runs unchanged on a machine without a
// Vulkan device.
//
// On a software device (llvmpipe, lavapipe, SwiftShader) the suite still runs:
// the device entry point declines before creating a pipeline -- llvmpipe's LLVM
// backend aborts the process when it compiles the fp16 packing of tvl1_dual --
// and those tests assert the CPU fallback path (bit-exact CPU agreement, no
// counted failure) instead. Only the tests whose numbers describe device
// arithmetic (throughput, the fp16 packing) skip there.
//
// The synthetic cases are the same sphere scenes test_tv_fusion.cpp uses, with
// the SAME bounds: the GPU result must land in the CPU test's acceptance region,
// not merely reproduce a particular run. Field agreement, the zero-crossing
// surface and the full BuildModel path are measured against the CPU on the same
// inputs, and the timing test reports 96^3 and 128^3.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/platform/env.hpp"
#include "base/utils/portable_rng.hpp"
#include "test_rng_helpers.hpp"

#include "dense_fixture.hpp"

#include "base/algorithm/voxel_grid.hpp"
#include "base/utils/error.hpp"
#include "common/algorithm/mesh/build_model_task.hpp"
#include "common/algorithm/mesh/tv_fusion.hpp"
#include "common/algorithm/mesh/tv_fusion_gpu.hpp"
#include "common/core/document.hpp"
#include "common/tasks/registry.hpp"
#include "gpu/gpu.hpp"
#include "model/camera/calibration.hpp"
#include "model/mesh/marching_tetrahedra.hpp"
#include "model/mesh/mesh_ops.hpp"
#include "perf_gate.hpp"

namespace alphashape {
namespace meshing {
namespace {

#define REQUIRE_GPU_TV()                              \
    do {                                              \
        if (!gpu::available()) {                      \
            GTEST_SKIP() << gpu::unavailable_reason(); \
        }                                             \
        gpu::set_enabled(true);                       \
    } while (false)

/// For the tests whose numbers only mean something on a real GPU (throughput,
/// the fp16 packing, the device iterate's accuracy): skip on a software device.
#define REQUIRE_HARDWARE_GPU_TV()                                    \
    do {                                                             \
        REQUIRE_GPU_TV();                                            \
        const std::shared_ptr<gpu::Context> hw_context =             \
            gpu::default_context();                                  \
        if (hw_context != nullptr &&                                 \
            gpu::is_software_device(hw_context->info())) {           \
            GTEST_SKIP() << "needs a hardware GPU, found "           \
                         << hw_context->info().name;                 \
        }                                                            \
    } while (false)

/// True on a software rasteriser, where tv_l1_fusion_device() declines to the
/// CPU before creating any tvl1 pipeline (llvmpipe cannot compile the fp16
/// packing: its LLVM backend aborts the process, not the dispatch).
bool software_device_active() {
    const std::shared_ptr<gpu::Context> context = gpu::default_context();
    return context != nullptr && gpu::is_software_device(context->info());
}

/// Asserts, across the lifetime of a scope, that no device failure was counted
/// on a software device. There the device entry point declines before creating
/// any pipeline, so a counted failure would mean the decline happened too late
/// (the exact bug that crashed llvmpipe: its LLVM JIT aborts compiling the fp16
/// packing). On hardware this is inert -- real device runs may legitimately
/// count a failure.
struct SoftDeclineGuard {
    std::uint64_t before = gpu::failure_count();
    bool software = false;

    SoftDeclineGuard() : software(software_device_active()) {}
    ~SoftDeclineGuard() {
        if (software) {
            EXPECT_EQ(gpu::failure_count(), before)
                << "a software device must decline before dispatching";
        }
    }
};

constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// Synthetic sphere scene, copied from test_tv_fusion.cpp so the CPU and GPU
// tests are independent references of the same contract.
// ---------------------------------------------------------------------------

struct Pose {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Vector3d eye = Eigen::Vector3d::Zero();

    Eigen::Matrix3d world_from_camera() const { return R.transpose(); }
};

Pose look_at(const Eigen::Vector3d& eye, const Eigen::Vector3d& target) {
    Eigen::Vector3d forward = target - eye;
    forward.normalize();
    Eigen::Vector3d up(0.0, 1.0, 0.0);
    if (std::abs(forward.dot(up)) > 0.99) {
        up = Eigen::Vector3d(0.0, 0.0, 1.0);
    }
    const Eigen::Vector3d right = forward.cross(up).normalized();
    const Eigen::Vector3d down = forward.cross(right);
    Pose pose;
    pose.R.row(0) = right;
    pose.R.row(1) = down;
    pose.R.row(2) = forward;
    pose.eye = eye;
    pose.t = -pose.R * eye;
    return pose;
}

DepthMap sphere_depth_map(const Calibration& calibration, const Pose& pose,
                          const Eigen::Vector3d& sphere_centre, double radius) {
    DepthMap map;
    map.camera_id = 0;
    map.width = calibration.width;
    map.height = calibration.height;
    map.depths.assign(static_cast<std::size_t>(map.width) * static_cast<std::size_t>(map.height),
                      -1.0F);
    const Eigen::Vector3d origin = pose.eye - sphere_centre;
    const Eigen::Matrix3d world_from_camera = pose.world_from_camera();
    const double c = origin.squaredNorm() - radius * radius;
    for (int j = 0; j < map.height; ++j) {
        for (int i = 0; i < map.width; ++i) {
            const Eigen::Vector2d pixel(static_cast<double>(i) + 0.5,
                                        static_cast<double>(j) + 0.5);
            const Eigen::Vector3d direction = world_from_camera * calibration.unproject(pixel);
            const double b = origin.dot(direction);
            const double discriminant = b * b - c;
            if (discriminant < 0.0) {
                continue;
            }
            const double near_hit = -b - std::sqrt(discriminant);
            if (near_hit <= 1e-9) {
                continue;
            }
            map.depths[static_cast<std::size_t>(j) * static_cast<std::size_t>(map.width) +
                       static_cast<std::size_t>(i)] = static_cast<float>(near_hit);
        }
    }
    return map;
}

double sample_field(const VoxelGrid<float>& field, const Eigen::Vector3d& point) {
    const GridSpec& spec = field.spec;
    const Eigen::Vector3d local =
        (point - spec.origin) / spec.voxel - Eigen::Vector3d(0.5, 0.5, 0.5);
    const double fx = std::max(0.0, std::min(static_cast<double>(spec.nx - 1), local.x()));
    const double fy = std::max(0.0, std::min(static_cast<double>(spec.ny - 1), local.y()));
    const double fz = std::max(0.0, std::min(static_cast<double>(spec.nz - 1), local.z()));
    const int x0 = static_cast<int>(fx);
    const int y0 = static_cast<int>(fy);
    const int z0 = static_cast<int>(fz);
    const int x1 = std::min(x0 + 1, spec.nx - 1);
    const int y1 = std::min(y0 + 1, spec.ny - 1);
    const int z1 = std::min(z0 + 1, spec.nz - 1);
    const double tx = fx - static_cast<double>(x0);
    const double ty = fy - static_cast<double>(y0);
    const double tz = fz - static_cast<double>(z0);
    const auto v = [&](int x, int y, int z) { return static_cast<double>(field.at(x, y, z)); };
    const double c00 = v(x0, y0, z0) * (1.0 - tx) + v(x1, y0, z0) * tx;
    const double c10 = v(x0, y1, z0) * (1.0 - tx) + v(x1, y1, z0) * tx;
    const double c01 = v(x0, y0, z1) * (1.0 - tx) + v(x1, y0, z1) * tx;
    const double c11 = v(x0, y1, z1) * (1.0 - tx) + v(x1, y1, z1) * tx;
    const double c0 = c00 * (1.0 - ty) + c10 * ty;
    const double c1 = c01 * (1.0 - ty) + c11 * ty;
    return c0 * (1.0 - tz) + c1 * tz;
}

bool point_is_observed(const VolumeResult& result, const Eigen::Vector3d& point) {
    const GridSpec& spec = result.field.spec;
    const Eigen::Vector3d local = (point - spec.origin) / spec.voxel;
    const int x = static_cast<int>(std::floor(local.x()));
    const int y = static_cast<int>(std::floor(local.y()));
    const int z = static_cast<int>(std::floor(local.z()));
    if (!spec.contains(x, y, z)) {
        return false;
    }
    return result.observed[spec.index(x, y, z)] != 0U;
}

struct Crossing {
    bool found = false;
    bool supported = false;
    double t = 0.0;
};

Crossing first_crossing(const VolumeResult& result, const Eigen::Vector3d& origin,
                        const Eigen::Vector3d& direction, double limit) {
    const double step = 0.25 * result.field.spec.voxel;
    double previous_t = 0.0;
    double previous_value = sample_field(result.field, origin);
    for (double t = step; t <= limit; t += step) {
        const double value = sample_field(result.field, origin + t * direction);
        if ((previous_value < 0.0) != (value < 0.0)) {
            Crossing crossing;
            crossing.found = true;
            crossing.t = (value == previous_value)
                             ? t
                             : previous_t + (-previous_value / (value - previous_value)) *
                                                (t - previous_t);
            crossing.supported = point_is_observed(result, origin + previous_t * direction) ||
                                 point_is_observed(result, origin + t * direction);
            return crossing;
        }
        previous_t = t;
        previous_value = value;
    }
    return Crossing{};
}

std::vector<Eigen::Vector3d> random_directions(std::size_t count, std::uint32_t seed) {
    std::mt19937 generator(seed);
    // Portable draw, not std::normal_distribution: that maps engine bits to
    // values differently under libc++ and libstdc++ (TESTING.md).
    const auto normal = [](std::mt19937& g) { return alphashape::rng::normal(g, 0.0, 1.0); };
    std::vector<Eigen::Vector3d> directions;
    directions.reserve(count);
    while (directions.size() < count) {
        // draw3 sequences the three draws in g++'s right-to-left order.
        const Eigen::Vector3d candidate = test_rng::draw3(generator, normal, normal, normal);
        const double norm = candidate.norm();
        if (norm > 1e-6) {
            directions.push_back(candidate / norm);
        }
    }
    return directions;
}

std::vector<Pose> ring_poses() {
    std::vector<Pose> poses;
    for (int k = 0; k < 8; ++k) {
        const double angle = 2.0 * kPi * static_cast<double>(k) / 8.0;
        const Eigen::Vector3d eye(4.0 * std::cos(angle), 2.0, 4.0 * std::sin(angle));
        poses.push_back(look_at(eye, Eigen::Vector3d::Zero()));
    }
    return poses;
}

Calibration sphere_calibration() { return Calibration::initial(CameraType::Frame, 96, 72, 90.0); }

void build_views(const Calibration& calibration, const std::vector<Pose>& poses,
                 const Eigen::Vector3d& sphere_centre, double radius, std::vector<DepthMap>& maps,
                 std::vector<FusionView>& views) {
    maps.assign(poses.size(), DepthMap{});
    views.clear();
    for (std::size_t k = 0; k < poses.size(); ++k) {
        maps[k] = sphere_depth_map(calibration, poses[k], sphere_centre, radius);
        FusionView view;
        view.map = &maps[k];
        view.calibration = calibration;
        view.R = poses[k].R;
        view.t = poses[k].t;
        views.push_back(view);
    }
}

double uniform(std::uint32_t& state) {
    state = state * 1664525U + 1013904223U;
    return static_cast<double>(state >> 8U) / 16777216.0;
}

double gaussian(std::uint32_t& state) {
    const double u1 = std::max(1e-12, uniform(state));
    const double u2 = uniform(state);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
}

struct SphereErrors {
    double median = 0.0;
    double p95 = 0.0;
    std::size_t supported = 0;
};

SphereErrors measure_sphere(const VolumeResult& result, std::size_t count, std::uint32_t seed,
                            const char* tag) {
    const double voxel = result.field.spec.voxel;
    std::vector<double> errors;
    std::size_t supported = 0;
    for (const Eigen::Vector3d& direction : random_directions(count, seed)) {
        const Crossing crossing = first_crossing(result, Eigen::Vector3d::Zero(), direction, 1.5);
        if (!crossing.found || !crossing.supported) {
            continue;
        }
        ++supported;
        errors.push_back(std::abs(crossing.t - 1.0) / voxel);
    }
    std::sort(errors.begin(), errors.end());
    SphereErrors summary;
    summary.supported = supported;
    summary.median = errors.empty() ? 0.0 : errors[errors.size() / 2];
    summary.p95 = errors.empty()
                      ? 0.0
                      : errors[std::min(errors.size() - 1,
                                        static_cast<std::size_t>(
                                            std::llround(0.95 * static_cast<double>(errors.size() - 1))))];
    std::printf("[tv gpu %s] supported %zu / %zu rays, median %.3f voxel, p95 %.3f voxel\n", tag,
                supported, count, summary.median, summary.p95);
    return summary;
}

// ---------------------------------------------------------------------------
// Shared device/CPU comparison helpers.
// ---------------------------------------------------------------------------

struct FieldAgreement {
    double mean_abs = 0.0;
    double max_abs = 0.0;
};

FieldAgreement field_agreement(const std::vector<float>& gpu, const std::vector<float>& cpu) {
    FieldAgreement agreement;
    double sum = 0.0;
    for (std::size_t i = 0; i < gpu.size() && i < cpu.size(); ++i) {
        const double difference = std::abs(static_cast<double>(gpu[i]) -
                                           static_cast<double>(cpu[i]));
        sum += difference;
        agreement.max_abs = std::max(agreement.max_abs, difference);
    }
    agreement.mean_abs = (gpu.empty() || cpu.empty())
                             ? 0.0
                             : sum / static_cast<double>(std::min(gpu.size(), cpu.size()));
    return agreement;
}

/// Median distance from every CPU-surface vertex to its nearest GPU-surface
/// vertex, in voxels: a coarse but robust surface-agreement measure that does
/// not depend on vertex ordering (marching tetrahedra shares the two vertex
/// sets only when the fields agree voxel for voxel).
double surface_distance_voxels(const Mesh& a, const Mesh& b, double voxel) {
    if (a.vertices.empty() || b.vertices.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    // Uniform hash over the GPU surface: bucket every vertex by cell, then
    // search the 3x3x3 neighbourhood of each CPU vertex.
    const double cell = std::max(3.0 * voxel, 1e-9);
    struct Cell {
        int x, y, z;
        bool operator==(const Cell& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };
    struct CellHash {
        std::size_t operator()(const Cell& c) const {
            const std::uint64_t h = static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.x)) *
                                    73856093ULL ^
                                    static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.y)) *
                                        19349663ULL ^
                                    static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.z)) *
                                        83492791ULL;
            return static_cast<std::size_t>(h);
        }
    };
    const auto cell_of = [&](const Eigen::Vector3d& p) {
        return Cell{static_cast<int>(std::floor(p.x() / cell)),
                    static_cast<int>(std::floor(p.y() / cell)),
                    static_cast<int>(std::floor(p.z() / cell))};
    };
    std::unordered_map<Cell, std::vector<std::uint32_t>, CellHash> buckets;
    for (std::uint32_t i = 0; i < b.vertices.size(); ++i) {
        buckets[cell_of(b.vertices[i])].push_back(i);
    }
    std::vector<double> nearest;
    nearest.reserve(a.vertices.size());
    for (const Eigen::Vector3d& p : a.vertices) {
        const Cell c = cell_of(p);
        double best = std::numeric_limits<double>::infinity();
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const auto it = buckets.find(Cell{c.x + dx, c.y + dy, c.z + dz});
                    if (it == buckets.end()) {
                        continue;
                    }
                    for (const std::uint32_t index : it->second) {
                        best = std::min(best, (b.vertices[index] - p).norm());
                    }
                }
            }
        }
        if (best == std::numeric_limits<double>::infinity()) {
            // Widen once to a full scan; only the rare outlier pays.
            for (const Eigen::Vector3d& q : b.vertices) {
                best = std::min(best, (q - p).norm());
            }
        }
        nearest.push_back(best / voxel);
    }
    std::sort(nearest.begin(), nearest.end());
    return nearest[nearest.size() / 2];
}

// Test 3 of the milestone brief: the zero-crossing surface itself, as marching
// tetrahedra extracts it (shared with BuildModel), not just the vertex sets.
// The CPU and GPU fields cross between the same voxels when they agree, so the
// two surfaces must coincide to well under a voxel.
struct SurfaceAgreement {
    double median = 0.0;
    std::size_t cpu_faces = 0;
    std::size_t gpu_faces = 0;
};

SurfaceAgreement surface_agreement(const VolumeResult& gpu, const VolumeResult& cpu) {
    Mesh cpu_mesh = extract_isosurface(cpu.field, cpu.iso, &cpu.observed);
    Mesh gpu_mesh = extract_isosurface(gpu.field, gpu.iso, &gpu.observed);
    SurfaceAgreement agreement;
    agreement.cpu_faces = cpu_mesh.faces.size();
    agreement.gpu_faces = gpu_mesh.faces.size();
    agreement.median =
        surface_distance_voxels(cpu_mesh, gpu_mesh, cpu.field.spec.voxel);
    return agreement;
}

double time_seconds(const std::function<VolumeResult()>& run) {
    const auto start = std::chrono::steady_clock::now();
    run();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

/// The noise-free sphere scene at 64^3 / 16 bins / 200 iterations, shared by the
/// accuracy and agreement tests.
struct SphereScene {
    std::vector<DepthMap> maps;
    std::vector<FusionView> views;
    TvFusionOptions options;
};

SphereScene sphere_scene() {
    SphereScene scene;
    const Calibration calibration = sphere_calibration();
    const std::vector<Pose> poses = ring_poses();
    build_views(calibration, poses, Eigen::Vector3d::Zero(), 1.0, scene.maps, scene.views);
    scene.options.max_resolution = 64;
    scene.options.histogram_bins = 16;
    scene.options.iterations = 200;
    return scene;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. The CPU test's acceptance bounds, on the GPU
// ---------------------------------------------------------------------------

TEST(TvFusionGpu, SphereLevelSetIsAccurate) {
    // Runs on both paths. The bounds are TvFusion.SphereLevelSetIsAccurate's own,
    // so the CPU fallback a software device takes satisfies them as well, and
    // the comparison against the CPU then degenerates to bit equality.
    REQUIRE_GPU_TV();
    SoftDeclineGuard guard;
    SphereScene scene = sphere_scene();
    // On hardware the call must reach the device; on a software device it must
    // decline (see DeclinesSoftwareDeviceWithoutFailure for the strict check).
    ASSERT_EQ(tv_l1_fusion_gpu_would_run(), !software_device_active());
    const Eigen::Vector3d box_min(-1.5, -1.5, -1.5);
    const Eigen::Vector3d box_max(1.5, 1.5, 1.5);

    const auto start = std::chrono::steady_clock::now();
    const VolumeResult gpu_result =
        tv_l1_fusion_device(scene.views, box_min, box_max, scene.options);
    const double gpu_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const VolumeResult cpu_result = tv_l1_fusion(scene.views, box_min, box_max, scene.options);

    const SphereErrors gpu_errors = measure_sphere(gpu_result, 400, 12345U, "gpu sphere");
    const SphereErrors cpu_errors = measure_sphere(cpu_result, 400, 12345U, "cpu sphere");
    const FieldAgreement agreement =
        field_agreement(gpu_result.field.data, cpu_result.field.data);
    std::printf("[tv gpu sphere] gpu %.3f s, cpu field mean |diff| %.2e, max %.2e\n",
                gpu_seconds, agreement.mean_abs, agreement.max_abs);

    // Identical to TvFusion.SphereLevelSetIsAccurate.
    EXPECT_GT(gpu_errors.supported, 150U);
    EXPECT_LT(gpu_errors.median, 0.5);
    EXPECT_LT(gpu_errors.p95, 1.0);
    EXPECT_LT(sample_field(gpu_result.field, Eigen::Vector3d::Zero()), 0.0);
    // ...and not materially worse than the CPU on the same rays.
    EXPECT_LT(gpu_errors.median, cpu_errors.median + 0.05);
}

TEST(TvFusionGpu, SurvivesOutliersAndNoise) {
    // Holds on both paths: on hardware this is the device iterate against the
    // bounds, on a software device it is the CPU fallback (asserted bit-exact
    // against the CPU itself, and the same bounds test_tv_fusion.cpp uses).
    REQUIRE_GPU_TV();
    SoftDeclineGuard guard;
    SphereScene scene = sphere_scene();
    const double voxel = 3.0 / 64.0;
    std::uint32_t state = 20251018U;
    for (DepthMap& map : scene.maps) {
        for (float& depth : map.depths) {
            if (depth <= 0.0F) {
                continue;
            }
            if (uniform(state) < 0.05) {
                depth *= 0.7F;
            } else {
                depth += static_cast<float>(gaussian(state) * 0.3 * voxel);
            }
        }
    }
    const VolumeResult gpu_result =
        tv_l1_fusion_device(scene.views, Eigen::Vector3d(-1.5, -1.5, -1.5),
                            Eigen::Vector3d(1.5, 1.5, 1.5), scene.options);
    const VolumeResult cpu_result = tv_l1_fusion(scene.views, Eigen::Vector3d(-1.5, -1.5, -1.5),
                                                 Eigen::Vector3d(1.5, 1.5, 1.5), scene.options);
    const SphereErrors gpu_errors = measure_sphere(gpu_result, 400, 98765U, "gpu noisy");
    const SphereErrors cpu_errors = measure_sphere(cpu_result, 400, 98765U, "cpu noisy");

    // Identical to TvFusion.SurvivesOutliersAndNoise.
    EXPECT_GT(gpu_errors.supported, 150U);
    EXPECT_LT(gpu_errors.median, 0.75);
    EXPECT_LT(sample_field(gpu_result.field, Eigen::Vector3d::Zero()), 0.0);
    EXPECT_NEAR(gpu_errors.median, cpu_errors.median, 0.05);
}

TEST(TvFusionGpu, HandlesOccludedHoles) {
    // Holds on both paths, like SurvivesOutliersAndNoise.
    REQUIRE_GPU_TV();
    SoftDeclineGuard guard;
    SphereScene scene = sphere_scene();
    for (DepthMap& map : scene.maps) {
        const int x0 = map.width / 2 - 7;
        const int y0 = map.height / 2 - 7;
        for (int j = y0; j < y0 + 15; ++j) {
            for (int i = x0; i < x0 + 15; ++i) {
                map.depths[static_cast<std::size_t>(j) * static_cast<std::size_t>(map.width) +
                           static_cast<std::size_t>(i)] = -1.0F;
            }
        }
    }
    const VolumeResult gpu_result =
        tv_l1_fusion_device(scene.views, Eigen::Vector3d(-1.5, -1.5, -1.5),
                            Eigen::Vector3d(1.5, 1.5, 1.5), scene.options);
    const VolumeResult cpu_result = tv_l1_fusion(scene.views, Eigen::Vector3d(-1.5, -1.5, -1.5),
                                                 Eigen::Vector3d(1.5, 1.5, 1.5), scene.options);
    const SphereErrors gpu_errors = measure_sphere(gpu_result, 400, 24680U, "gpu holes");
    const SphereErrors cpu_errors = measure_sphere(cpu_result, 400, 24680U, "cpu holes");

    // Identical to TvFusion.HandlesOccludedHoles.
    EXPECT_GT(gpu_errors.supported, 150U);
    EXPECT_LT(gpu_errors.median, 0.75);
    EXPECT_NEAR(gpu_errors.median, cpu_errors.median, 0.05);

    // A ray straight through the deleted patch still crosses the surface.
    const std::vector<Pose> poses = ring_poses();
    const Eigen::Vector3d axis = poses[0].eye.normalized();
    bool any_crossing = false;
    for (int k = 0; k < 64; ++k) {
        const double angle = 2.0 * kPi * static_cast<double>(k) / 64.0;
        const Eigen::Vector3d direction =
            axis + 0.05 * Eigen::Vector3d(std::cos(angle), 0.0, std::sin(angle));
        if (first_crossing(gpu_result, Eigen::Vector3d::Zero(), direction.normalized(), 1.2)
                .found) {
            any_crossing = true;
        }
    }
    EXPECT_TRUE(any_crossing);
}

// ---------------------------------------------------------------------------
// 2. Field and surface agreement with the CPU
// ---------------------------------------------------------------------------

TEST(TvFusionGpu, FieldAndSurfaceAgreeWithCpu) {
    // Every bound here is "the device result equals the CPU's within float
    // noise", which the CPU fallback satisfies exactly; on a software device it
    // also proves the decline returned the CPU's own answer.
    REQUIRE_GPU_TV();
    SoftDeclineGuard guard;
    SphereScene scene = sphere_scene();
    const Eigen::Vector3d box_min(-1.5, -1.5, -1.5);
    const Eigen::Vector3d box_max(1.5, 1.5, 1.5);
    const VolumeResult gpu_result = tv_l1_fusion_device(scene.views, box_min, box_max, scene.options);
    const VolumeResult cpu_result = tv_l1_fusion(scene.views, box_min, box_max, scene.options);

    const FieldAgreement agreement =
        field_agreement(gpu_result.field.data, cpu_result.field.data);
    const SurfaceAgreement surface = surface_agreement(gpu_result, cpu_result);
    {
        // A partial solve exercises the arithmetic (float32 device against the
        // CPU's double): at a converged iterate both land on bin centres, but a
        // short solve still carries the accumulated float difference.
        TvFusionOptions partial = scene.options;
        partial.iterations = 5;
        const VolumeResult gpu_partial =
            tv_l1_fusion_device(scene.views, box_min, box_max, partial);
        const VolumeResult cpu_partial = tv_l1_fusion(scene.views, box_min, box_max, partial);
        const FieldAgreement mid =
            field_agreement(gpu_partial.field.data, cpu_partial.field.data);
        std::printf("[tv gpu agreement] after 5 iterations: mean |diff| %.3e max %.3e\n",
                    mid.mean_abs, mid.max_abs);
        EXPECT_LT(mid.mean_abs, 0.02);
    }
    std::printf("[tv gpu agreement] field mean |diff| %.3e, max |diff| %.3e | surface median "
                "%.4f voxel, faces cpu %zu gpu %zu\n",
                agreement.mean_abs, agreement.max_abs, surface.median, surface.cpu_faces,
                surface.gpu_faces);

    // The CPU and GPU solves share their initial field and their data, so they
    // differ only in float32 arithmetic: a field difference far below the 2.0
    // range of u, and a surface that lies within a tenth of a voxel.
    EXPECT_LT(agreement.mean_abs, 0.02);
    EXPECT_LT(agreement.max_abs, 0.5);
    EXPECT_LT(surface.median, 0.1);
    EXPECT_GT(surface.gpu_faces, 0U);
    EXPECT_NEAR(static_cast<double>(surface.gpu_faces),
                static_cast<double>(surface.cpu_faces), 0.05 * static_cast<double>(surface.cpu_faces));
}

// ---------------------------------------------------------------------------
// 3. Determinism
// ---------------------------------------------------------------------------

TEST(TvFusionGpu, DeterministicRunToRun) {
    REQUIRE_GPU_TV();
    SoftDeclineGuard guard;
    SphereScene scene = sphere_scene();
    scene.options.max_resolution = 32;
    scene.options.iterations = 40;
    const Eigen::Vector3d box_min(-1.5, -1.5, -1.5);
    const Eigen::Vector3d box_max(1.5, 1.5, 1.5);
    const VolumeResult first = tv_l1_fusion_device(scene.views, box_min, box_max, scene.options);
    const VolumeResult second = tv_l1_fusion_device(scene.views, box_min, box_max, scene.options);
    ASSERT_EQ(first.field.data.size(), second.field.data.size());
    for (std::size_t i = 0; i < first.field.data.size(); ++i) {
        ASSERT_EQ(first.field.data[i], second.field.data[i]) << "voxel " << i;
    }
    EXPECT_EQ(first.observed, second.observed);
}

TEST(TvFusionGpu, SplitSubmissionsAreBitIdentical) {
    // The iteration loop is split across submissions so no single GPU job can
    // starve a display. The split is a scheduling change: the same dual/primal
    // dispatches in the same order, so the field must be identical bit for bit.
    REQUIRE_GPU_TV();
    SoftDeclineGuard guard;
    SphereScene scene = sphere_scene();
    scene.options.max_resolution = 32;
    scene.options.iterations = 24;
    const Eigen::Vector3d box_min(-1.5, -1.5, -1.5);
    const Eigen::Vector3d box_max(1.5, 1.5, 1.5);

    // A software device declines to the CPU before constructing a limiter, so
    // there is nothing to split; the bounds of the other tests cover that path.
    if (software_device_active()) {
        GTEST_SKIP() << "software device declines to the CPU before any submission";
    }

    // Unsplit reference: the cap is pinned at or above the algorithm's own
    // batch size, so every batch is a single submission.
    std::vector<float> unsplit;
    int unsplit_submissions = 0;
    ASSERT_TRUE(alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BATCH", "64"));
    try {
        unsplit = tv_l1_fusion_device(scene.views, box_min, box_max, scene.options).field.data;
        unsplit_submissions = tv_fusion_gpu_last_profile().submissions;
    } catch (...) {
        alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BATCH", "");
        throw;
    }
    alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BATCH", "");

    std::vector<float> split;
    int split_submissions = 0;
    int split_batches = 0;
    ASSERT_TRUE(alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BATCH", "1"));
    try {
        split = tv_l1_fusion_device(scene.views, box_min, box_max, scene.options).field.data;
        split_submissions = tv_fusion_gpu_last_profile().submissions;
        split_batches = tv_fusion_gpu_last_profile().batches;
    } catch (...) {
        alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BATCH", "");
        throw;
    }
    alphashape::platform::set_env("ALPHASHAPE_GPU_MAX_BATCH", "");

    std::printf("[tv gpu split] unsplit %d submission(s) | one step per submission: %d "
                "submission(s) across %d batch(es)\n",
                unsplit_submissions, split_submissions, split_batches);
    ASSERT_EQ(split.size(), unsplit.size());
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < split.size(); ++i) {
        if (std::memcmp(&split[i], &unsplit[i], sizeof(float)) != 0) {
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u) << "the split changed " << mismatches << " of " << split.size()
                              << " voxels";
    EXPECT_GT(split_submissions, unsplit_submissions)
        << "one step per submission must need more submissions than an unsplit batch";
}

// ---------------------------------------------------------------------------
// 4. Fallback semantics
// ---------------------------------------------------------------------------

TEST(TvFusionGpu, DisabledFallsBackToCpu) {
    REQUIRE_GPU_TV();
    SphereScene scene = sphere_scene();
    scene.options.max_resolution = 32;
    scene.options.iterations = 40;
    const Eigen::Vector3d box_min(-1.5, -1.5, -1.5);
    const Eigen::Vector3d box_max(1.5, 1.5, 1.5);

    gpu::set_enabled(false);
    EXPECT_FALSE(tv_l1_fusion_gpu_would_run());
    const std::uint64_t before = gpu::failure_count();
    const VolumeResult disabled = tv_l1_fusion_device(scene.views, box_min, box_max, scene.options);
    gpu::set_enabled(true);
    const VolumeResult cpu = tv_l1_fusion(scene.views, box_min, box_max, scene.options);

    EXPECT_EQ(disabled.field.data, cpu.field.data);
    EXPECT_EQ(disabled.observed, cpu.observed);
    // The CPU fallback is taken before any device work, so it is not a failure.
    EXPECT_EQ(gpu::failure_count(), before);
    // ...and with the layer enabled the hook agrees with the device decision.
    EXPECT_EQ(tv_l1_fusion_gpu_would_run(), !software_device_active());
}

TEST(TvFusionGpu, ForcedFailureFallsBackAndCounts) {
    REQUIRE_GPU_TV();
    SphereScene scene = sphere_scene();
    scene.options.max_resolution = 32;
    scene.options.iterations = 40;
    const Eigen::Vector3d box_min(-1.5, -1.5, -1.5);
    const Eigen::Vector3d box_max(1.5, 1.5, 1.5);

    // Validation still throws rather than being absorbed by the fallback -- on
    // either path, because it runs before any device work.
    std::vector<FusionView> null_views = {FusionView{}};
    EXPECT_THROW(tv_l1_fusion_device(null_views, box_min, box_max, scene.options), ValueError);

    const std::uint64_t before = gpu::failure_count();
    tv_fusion_gpu_force_failure(true);
    const VolumeResult result = tv_l1_fusion_device(scene.views, box_min, box_max, scene.options);
    if (software_device_active()) {
        // Nothing dispatches: the entry point declines to the CPU before the
        // hook is consulted, so the forced failure is never consumed and no
        // failure is counted. The result is the CPU's, bit for bit.
        EXPECT_EQ(gpu::failure_count(), before);
        // Leave the hook unarmed for the tests that follow.
        tv_fusion_gpu_force_failure(false);
        EXPECT_EQ(result.field.data,
                  tv_l1_fusion(scene.views, box_min, box_max, scene.options).field.data);
        return;
    }
    EXPECT_EQ(gpu::failure_count(), before + 1);
    EXPECT_EQ(result.field.data,
              tv_l1_fusion(scene.views, box_min, box_max, scene.options).field.data);
}

TEST(TvFusionGpu, DeclinesSoftwareDeviceWithoutFailure) {
    REQUIRE_GPU_TV();
    const std::shared_ptr<gpu::Context> context = gpu::default_context();
    ASSERT_NE(context, nullptr);
    const bool software = gpu::is_software_device(context->info());
    // The predicate and the entry point must agree, on both device kinds.
    EXPECT_EQ(tv_l1_fusion_gpu_fp16_available(*context), !software);
    ASSERT_EQ(tv_l1_fusion_gpu_supported(), !software);
    ASSERT_EQ(tv_l1_fusion_gpu_would_run(), !software);

    SphereScene scene = sphere_scene();
    scene.options.max_resolution = 32;
    scene.options.iterations = 40;
    const Eigen::Vector3d box_min(-1.5, -1.5, -1.5);
    const Eigen::Vector3d box_max(1.5, 1.5, 1.5);
    const std::uint64_t before = gpu::failure_count();
    const VolumeResult result =
        tv_l1_fusion_device(scene.views, box_min, box_max, scene.options);
    const VolumeResult cpu = tv_l1_fusion(scene.views, box_min, box_max, scene.options);

    if (software) {
        // Nothing was dispatched, so no failure was recorded: this is the whole
        // point of the pre-dispatch decline (llvmpipe's LLVM JIT aborts the
        // process compiling the fp16 packing, so falling back on a caught
        // device error would be too late).
        EXPECT_EQ(gpu::failure_count(), before) << "the decline must precede dispatch";
        // ...and the result is the CPU solver's own output, bit for bit.
        EXPECT_EQ(result.field.data, cpu.field.data);
        EXPECT_EQ(result.observed, cpu.observed);
        EXPECT_EQ(result.iso, cpu.iso);
        EXPECT_EQ(result.field.spec.nx, cpu.field.spec.nx);
        EXPECT_EQ(result.field.spec.ny, cpu.field.spec.ny);
        EXPECT_EQ(result.field.spec.nz, cpu.field.spec.nz);
        EXPECT_EQ(result.field.spec.voxel, cpu.field.spec.voxel);
    } else {
        // On hardware the same call runs on the device: it may count a failure,
        // but it always produces a field of the CPU's shape and either matches
        // the CPU closely (the device path) or exactly (fallback after a
        // failure).
        EXPECT_EQ(result.field.data.size(), cpu.field.data.size());
        EXPECT_EQ(result.observed.size(), cpu.observed.size());
        if (gpu::failure_count() == before) {
            EXPECT_LT(field_agreement(result.field.data, cpu.field.data).mean_abs, 0.05);
        } else {
            EXPECT_EQ(result.field.data, cpu.field.data);
        }
    }
}

TEST(TvFusionGpu, HardwareDeclaresFp16) {
    REQUIRE_GPU_TV();
    const std::shared_ptr<gpu::Context> context = gpu::default_context();
    ASSERT_NE(context, nullptr);
    if (gpu::is_software_device(context->info())) {
        GTEST_SKIP() << "software device " << context->info().name
                     << " cannot compile the fp16 packing; the entry point declines";
    }
    const bool fp16 = tv_l1_fusion_gpu_fp16_available(*context);
    if (!fp16) {
        // A driver without the packing capability keeps the CPU path; report it
        // rather than fail, so a new device shows up as a log line here.
        std::printf("[tv gpu fp16] %s does not declare the fp16 packing; CPU fallback used\n",
                    context->info().name.c_str());
        EXPECT_FALSE(tv_l1_fusion_gpu_supported());
        return;
    }
    // A device that declares fp16 must also pass the supported() gate, and the
    // dual kernel must then run: a small solve checks the pipeline compiles and
    // dispatches without a counted failure.
    EXPECT_TRUE(tv_l1_fusion_gpu_supported());
    SphereScene scene = sphere_scene();
    scene.options.max_resolution = 24;
    scene.options.iterations = 8;
    const std::uint64_t before = gpu::failure_count();
    const VolumeResult result =
        tv_l1_fusion_device(scene.views, Eigen::Vector3d(-1.5, -1.5, -1.5),
                            Eigen::Vector3d(1.5, 1.5, 1.5), scene.options);
    EXPECT_EQ(gpu::failure_count(), before)
        << "the fp16 dual kernel failed to compile or dispatch on " << context->info().name;
    EXPECT_FALSE(result.field.data.empty());
}

// ---------------------------------------------------------------------------
// 5. End to end: BuildModel on the dense fixture, same bounds as the CPU test
// ---------------------------------------------------------------------------

namespace {

using dense_fixture::percentile;

double plane_distance(const Eigen::Vector3d& p) {
    return std::min(std::abs(p.z()), std::abs(p.y()));
}

struct MeshMeasurement {
    double median_voxels = 0.0;
    double p90_voxels = 0.0;
    std::size_t components = 0;
    std::size_t nonmanifold_edges = 0;
    std::size_t boundary_edges = 0;
    std::size_t faces = 0;
};

MeshMeasurement measure_mesh(Document& document) {
    const Mesh& mesh = document.active_chunk()->model();
    const MeshTopology topology = analyse_topology(mesh);
    std::vector<double> distances;
    distances.reserve(mesh.vertices.size());
    for (const Eigen::Vector3d& v : mesh.vertices) {
        distances.push_back(plane_distance(v));
    }
    std::sort(distances.begin(), distances.end());
    MeshMeasurement m;
    m.faces = mesh.faces.size();
    m.components = topology.components;
    m.nonmanifold_edges = topology.nonmanifold_edges;
    m.boundary_edges = topology.boundary_edges;
    // The voxel size the mesh was extracted from: the same bbox helper and
    // resolution the task and test_build_model use.
    const std::pair<Eigen::Vector3d, Eigen::Vector3d> box =
        build_model_bbox(*document.active_chunk(), "depth_maps");
    const double voxel = (box.second - box.first).maxCoeff() / 192.0;
    std::vector<double> in_voxels;
    in_voxels.reserve(distances.size());
    for (const double d : distances) {
        in_voxels.push_back(d / voxel);
    }
    m.median_voxels = percentile(in_voxels, 0.50);
    m.p90_voxels = percentile(in_voxels, 0.90);
    return m;
}

/// BuildModel(depth_maps, disabled, custom 1e6, 192) on the truth-pose dense
/// document, exactly scenario 1 of test_build_model's end-to-end test.
MeshMeasurement run_build_model(Document& document, const char* tag) {
    Params params = make_params(builtin_registry().require("BuildModel"));
    params.set("source_data", ParamValue{std::string("depth_maps")});
    params.set("surface_type", ParamValue{std::string("arbitrary")});
    params.set("interpolation", ParamValue{std::string("disabled")});
    params.set("face_count", ParamValue{std::string("custom")});
    params.set("face_count_custom", ParamValue{std::int64_t{1000000}});
    params.set("vertex_colors", ParamValue{true});
    params.set("vertex_confidence", ParamValue{true});
    params.set("volume_resolution", ParamValue{std::int64_t{192}});
    const auto start = std::chrono::steady_clock::now();
    run_task(builtin_registry(), document, "BuildModel", params);
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    const MeshMeasurement m = measure_mesh(document);
    std::printf("[tv gpu e2e %s] %.2f s, faces %zu, median %.3f p90 %.3f voxels, components "
                "%zu, non-manifold edges %zu, boundary edges %zu\n",
                tag, seconds, m.faces, m.median_voxels, m.p90_voxels, m.components,
                m.nonmanifold_edges, m.boundary_edges);
    return m;
}

Document depth_document() {
    const std::vector<dense_fixture::TruthImage> truth =
        dense_fixture::parse_truth_images("seq_truth.json");
    Document document = dense_fixture::aligned_document(dense_fixture::seq_files());
    Chunk& chunk = *document.active_chunk();
    dense_fixture::set_truth_poses(document, truth);
    dense_fixture::set_truth_calibration(document, 480, 360, 500.0);
    dense_fixture::retriangulate_tracks(chunk);
    dense_fixture::run_build_depth_maps(document, /*downscale=*/2, "mild");
    return document;
}

}  // namespace

TEST(TvFusionGpu, BuildModelMeetsCpuBounds) {
    // Runs on both paths: on a software device both documents take the CPU
    // solver, so the comparison degenerates to "the same pipeline twice" while
    // the bounds still guard the fused path's integration into BuildModel.
    REQUIRE_GPU_TV();
    SoftDeclineGuard guard;
    Document gpu_document = depth_document();
    gpu::set_enabled(true);
    const MeshMeasurement gpu = run_build_model(gpu_document, "gpu");

    Document cpu_document = depth_document();
    gpu::set_enabled(false);
    const MeshMeasurement cpu = run_build_model(cpu_document, "cpu");
    gpu::set_enabled(true);

    // Same assertions as test_build_model's depth_maps scenario.
    EXPECT_GT(gpu.faces, 0U);
    EXPECT_LT(gpu.median_voxels, 0.5) << "median " << gpu.median_voxels << " voxels";
    EXPECT_LT(gpu.p90_voxels, 1.5) << "p90 " << gpu.p90_voxels << " voxels";
    EXPECT_EQ(gpu.components, 1U);
    EXPECT_EQ(gpu.nonmanifold_edges, 0U);
    // The dense fixture's surface is clipped by the volume border, so the
    // extracted mesh has genuine boundary edges (test_build_model asserts
    // non-manifold edges only). What this guards is "the device mesh is the same
    // surface, with its real open rim, not a degenerate one" - not bit-identical
    // topology, which it cannot be: the two documents' depth maps come from the
    // GPU and the CPU search, and the GPU layer documents those as agreeing in
    // quality and near-agreement, not bit for bit. Mesh topology is a
    // discontinuous function of a depth map - an isovalue crossing one more
    // voxel adds or removes a pocket of boundary edges - so the count moves by a
    // handful of edges when a fraction of a percent of pixels land elsewhere
    // under fused multiply-add. Measured on a Tesla T4 at 93.4% pixel agreement:
    // 8 of 1272, 0.6%. The bound still fails a zero count, a missing rim, or a
    // mesh of some other surface.
    const double boundary_relative =
        std::abs(static_cast<double>(gpu.boundary_edges) -
                 static_cast<double>(cpu.boundary_edges)) /
        static_cast<double>(std::max<std::size_t>(cpu.boundary_edges, 1));
    EXPECT_GT(cpu.boundary_edges, 0U);
    EXPECT_LT(boundary_relative, 0.02)
        << "boundary edges gpu " << gpu.boundary_edges << " cpu " << cpu.boundary_edges;
    // ...and the GPU mesh lands where the CPU's does.
    EXPECT_LT(gpu.median_voxels, cpu.median_voxels + 0.05);
    EXPECT_LT(gpu.p90_voxels, cpu.p90_voxels + 0.1);
}

// ---------------------------------------------------------------------------
// 6. Timing: 96^3 and 128^3, GPU against the 8-thread CPU
// ---------------------------------------------------------------------------

TEST(TvFusionGpu, TimingAgainstCpu) {
    // A speed-up ratio is meaningless on a software device: the "device" is the
    // CPU, so the comparison would time the same code twice.
    REQUIRE_HARDWARE_GPU_TV();
    // The same sphere scene at two volumes; the histogram bins grow with the
    // volume (8 and 16), which is what the profile's BuildModel runs use.
    for (const int resolution : {96, 128}) {
        const Calibration calibration = sphere_calibration();
        const std::vector<Pose> poses = ring_poses();
        std::vector<DepthMap> maps;
        std::vector<FusionView> views;
        build_views(calibration, poses, Eigen::Vector3d::Zero(), 1.0, maps, views);
        TvFusionOptions options;
        options.max_resolution = resolution;
        options.histogram_bins = resolution >= 128 ? 16 : 8;
        options.iterations = 300;
        const Eigen::Vector3d box_min(-1.5, -1.5, -1.5);
        const Eigen::Vector3d box_max(1.5, 1.5, 1.5);

        {
            // Setup cost is on the CPU for both paths (the shared histogram
            // integration); report it so the speed-up's denominator is clear.
            const auto setup_start = std::chrono::steady_clock::now();
            const detail::TvFusionSetup setup =
                detail::tv_l1_fusion_setup(views, box_min, box_max, options);
            const double setup_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - setup_start)
                    .count();
            std::size_t measured = 0;
            for (std::size_t v = 0; v < setup.initial_field.size(); ++v) {
                for (int k = 0; k < setup.bins; ++k) {
                    if (setup.weights[v * static_cast<std::size_t>(setup.bins) +
                                       static_cast<std::size_t>(k)] > 0.0) {
                        ++measured;
                        break;
                    }
                }
            }
            std::printf("[tv gpu stats] %d^3: measured %zu / %zu voxels, setup %.3f s\n",
                        resolution, measured, setup.initial_field.size(), setup_seconds);
        }
        // Warm the pipelines so compilation is not charged to the benchmark.
        (void)tv_l1_fusion_device(views, box_min, box_max, options);
        // Best of three: the sibling test sessions share this machine, and a
        // single sample can be inflated many times over by their load (see
        // docs/m10-profile.md, which records the same effect).
        const auto best_of_three = [](const std::function<VolumeResult()>& run) {
            double best = std::numeric_limits<double>::infinity();
            for (int trial = 0; trial < 3; ++trial) {
                best = std::min(best, time_seconds(run));
            }
            return best;
        };
        const double gpu_seconds = best_of_three(
            [&] { return tv_l1_fusion_device(views, box_min, box_max, options); });
        gpu::set_enabled(false);
        const double cpu_seconds =
            best_of_three([&] { return tv_l1_fusion(views, box_min, box_max, options); });
        gpu::set_enabled(true);
        const TvFusionGpuProfile profile = tv_fusion_gpu_last_profile();
        std::printf("[tv gpu profile] %d^3: setup %.3f s, upload %.3f s, iterate %.3f s "
                    "(%d batches), readback %.3f s\n",
                    resolution, profile.setup_seconds, profile.upload_seconds,
                    profile.iterate_seconds, profile.batches, profile.readback_seconds);
        const double speedup = cpu_seconds / gpu_seconds;
        std::printf("[tv gpu timing] %d^3, %d bins, %d iters: gpu %.3f s cpu %.3f s "
                    "speed-up %.2fx\n",
                    resolution, options.histogram_bins, options.iterations, gpu_seconds,
                    cpu_seconds, speedup);
        RecordProperty("gpu_seconds_" + std::to_string(resolution), gpu_seconds);
        RecordProperty("cpu_seconds_" + std::to_string(resolution), cpu_seconds);
        RecordProperty("speedup_" + std::to_string(resolution), speedup);

        // The target is >= 4x (profile estimate 6-12x). The floor catches a
        // real regression; a smaller measured ratio is reported in
        // LIMITATIONS.md with what limits it.
        if (test::perf_asserts_enabled()) {
            EXPECT_GT(speedup, 3.0) << "target >= 4x at " << resolution << "^3";
        }
    }
}
}  // namespace meshing
}  // namespace alphashape
