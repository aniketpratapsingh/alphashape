// TV-L1 volumetric fusion: multi-view sphere reconstruction accuracy, outlier
// robustness, hole handling, the prox operator against brute force, thread
// determinism and input validation.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/utils/portable_rng.hpp"
#include "test_rng_helpers.hpp"

#include "base/algorithm/voxel_grid.hpp"
#include "base/utils/error.hpp"
#include "common/algorithm/mesh/tv_fusion.hpp"
#include "common/core/chunk.hpp"
#include "model/camera/calibration.hpp"

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

namespace alphashape::meshing {
namespace {

constexpr double kPi = 3.14159265358979323846;

/// A world -> camera pose: looking from `eye` at `target`, world +y up.
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

/// Ray-sphere intersection along the unit bearing of each pixel: the returned
/// depths are distances along that bearing, matching DepthMap's convention.
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
            const Eigen::Vector2d pixel(static_cast<double>(i) + 0.5, static_cast<double>(j) + 0.5);
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
            const std::size_t index =
                static_cast<std::size_t>(j) * static_cast<std::size_t>(map.width) +
                static_cast<std::size_t>(i);
            map.depths[index] = static_cast<float>(near_hit);
        }
    }
    return map;
}

/// Trilinear sample of a scalar grid at a world point (clamped at the border).
double sample(const VoxelGrid<float>& field, const Eigen::Vector3d& point) {
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

/// The first zero crossing along a ray from `origin`, found by marching at a
/// quarter voxel and linearly interpolating the sign change. `supported` is true
/// when the crossing passes through data the field actually measured, i.e. one
/// of the two bracketing voxels is in the observed mask.
struct Crossing {
    bool found = false;
    bool supported = false;
    double t = 0.0;
};

Crossing first_crossing(const VolumeResult& result, const Eigen::Vector3d& origin,
                        const Eigen::Vector3d& direction, double limit) {
    const double step = 0.25 * result.field.spec.voxel;
    double previous_t = 0.0;
    double previous_value = sample(result.field, origin);
    for (double t = step; t <= limit; t += step) {
        const double value = sample(result.field, origin + t * direction);
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

double median_of(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if (n == 0) {
        return 0.0;
    }
    return (n % 2 == 1) ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

double percentile(const std::vector<double>& sorted, double fraction) {
    if (sorted.empty()) {
        return 0.0;
    }
    const std::size_t index = static_cast<std::size_t>(
        std::llround(fraction * static_cast<double>(sorted.size() - 1)));
    return sorted[std::min(index, sorted.size() - 1)];
}

/// Eight cameras around the origin, two units above the plane, looking inward.
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

/// Depth maps for a sphere of radius `radius` from `poses`, plus the matching
/// fusion views. `maps` must outlive the returned views.
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

}  // namespace

// Mould the grain of the field with a hand-written deterministic RNG so the
// outlier and noise tests are reproducible across platforms.
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
    std::size_t sampled = 0;
    std::size_t supported = 0;
};

/// Cast `count` random rays from the origin at a unit sphere and measure how far
/// each supported crossing lands from the true radius, in voxels. Only crossings
/// that pass through observed voxels count; a ray that dies in an unmeasured
/// region carries no information and must not be scored.
SphereErrors measure_sphere(const VolumeResult& result, std::size_t count, std::uint32_t seed,
                            bool verbose) {
    const double voxel = result.field.spec.voxel;
    std::vector<double> errors;
    std::size_t supported = 0;
    std::size_t total = 0;
    for (const Eigen::Vector3d& direction : random_directions(count, seed)) {
        ++total;
        const Crossing crossing = first_crossing(result, Eigen::Vector3d::Zero(), direction, 1.5);
        if (!crossing.found || !crossing.supported) {
            continue;
        }
        ++supported;
        errors.push_back(std::abs(crossing.t - 1.0) / voxel);
    }
    std::sort(errors.begin(), errors.end());
    SphereErrors summary;
    summary.median = median_of(errors);
    summary.p95 = percentile(errors, 0.95);
    summary.sampled = total;
    summary.supported = supported;
    if (verbose) {
        std::cout << "[TvFusion] supported " << supported << " / " << total << " rays, median "
                  << summary.median << " voxel, p95 " << summary.p95 << " voxel\n";
    }
    return summary;
}

// Test 1: a noise-free sphere from eight ring cameras reconstructs to a fraction
// of a voxel, with the centre inside.
TEST(TvFusion, SphereLevelSetIsAccurate) {
    const Calibration calibration = sphere_calibration();
    const std::vector<Pose> poses = ring_poses();
    std::vector<DepthMap> maps;
    std::vector<FusionView> views;
    build_views(calibration, poses, Eigen::Vector3d::Zero(), 1.0, maps, views);

    TvFusionOptions options;
    options.max_resolution = 64;
    options.histogram_bins = 16;
    options.iterations = 200;
    const Eigen::Vector3d box_min(-1.5, -1.5, -1.5);
    const Eigen::Vector3d box_max(1.5, 1.5, 1.5);

    const auto start = std::chrono::steady_clock::now();
    const VolumeResult result = tv_l1_fusion(views, box_min, box_max, options);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    const SphereErrors errors = measure_sphere(result, 400, 12345U, true);
    std::cout << "[TvFusion] sphere fused in " << seconds << " s\n";
    EXPECT_GT(errors.supported, 150U);
    EXPECT_LT(errors.median, 0.5);
    EXPECT_LT(errors.p95, 1.0);
    EXPECT_LT(sample(result.field, Eigen::Vector3d::Zero()), 0.0);
}

// Test 2: five percent of the samples multiply their depth by 0.7 (a gross
// matcher outlier, 30% too close) and the rest carry Gaussian noise of 0.3
// voxel. The L1 data term should absorb both without losing the surface.
TEST(TvFusion, SurvivesOutliersAndNoise) {
    const Calibration calibration = sphere_calibration();
    const std::vector<Pose> poses = ring_poses();
    std::vector<DepthMap> maps;
    std::vector<FusionView> views;
    build_views(calibration, poses, Eigen::Vector3d::Zero(), 1.0, maps, views);

    TvFusionOptions options;
    options.max_resolution = 64;
    options.histogram_bins = 16;
    options.iterations = 200;
    const double voxel = 3.0 / 64.0;

    std::uint32_t state = 20251018U;
    for (DepthMap& map : maps) {
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

    const VolumeResult result =
        tv_l1_fusion(views, Eigen::Vector3d(-1.5, -1.5, -1.5), Eigen::Vector3d(1.5, 1.5, 1.5), options);
    const SphereErrors errors = measure_sphere(result, 400, 98765U, true);
    EXPECT_GT(errors.supported, 150U);
    EXPECT_LT(errors.median, 0.75);
    EXPECT_LT(sample(result.field, Eigen::Vector3d::Zero()), 0.0);
}

// Test 3: a 15x15 block of every depth map is deleted. The surface must still be
// recoverable where the surrounding views have data, and the hole must not turn
// the whole field inside out.
TEST(TvFusion, HandlesOccludedHoles) {
    const Calibration calibration = sphere_calibration();
    const std::vector<Pose> poses = ring_poses();
    std::vector<DepthMap> maps;
    std::vector<FusionView> views;
    build_views(calibration, poses, Eigen::Vector3d::Zero(), 1.0, maps, views);

    for (DepthMap& map : maps) {
        const int x0 = map.width / 2 - 7;
        const int y0 = map.height / 2 - 7;
        for (int j = y0; j < y0 + 15; ++j) {
            for (int i = x0; i < x0 + 15; ++i) {
                map.depths[static_cast<std::size_t>(j) * static_cast<std::size_t>(map.width) +
                           static_cast<std::size_t>(i)] = -1.0F;
            }
        }
    }

    TvFusionOptions options;
    options.max_resolution = 64;
    options.histogram_bins = 16;
    options.iterations = 200;
    const VolumeResult result =
        tv_l1_fusion(views, Eigen::Vector3d(-1.5, -1.5, -1.5), Eigen::Vector3d(1.5, 1.5, 1.5), options);

    // Every camera's hole faces the same spot on the sphere (all look inward with
    // the same image convention), so rays straight at the hole see nothing; rays
    // that clip its rim and rays on the far side must still register a surface.
    const SphereErrors errors = measure_sphere(result, 400, 24680U, true);
    EXPECT_GT(errors.supported, 150U);
    EXPECT_LT(errors.median, 0.75);

    // Routing a ray straight through the deleted patch must still cross the
    // surface somewhere on its way out, because the opposite hemisphere has data.
    const Eigen::Vector3d axis = poses[0].eye.normalized();
    bool any_crossing = false;
    for (int k = 0; k < 64; ++k) {
        const double angle = 2.0 * kPi * static_cast<double>(k) / 64.0;
        const Eigen::Vector3d direction =
            axis + 0.05 * Eigen::Vector3d(std::cos(angle), 0.0, std::sin(angle));
        const Crossing crossing =
            first_crossing(result, Eigen::Vector3d::Zero(), direction.normalized(), 1.2);
        if (crossing.found) {
            any_crossing = true;
        }
    }
    EXPECT_TRUE(any_crossing);
}

// Test 4: the prox operator the solver calls per voxel must match a brute-force
// minimisation of the same objective. The test re-implements the closed form the
// same way the library does, then checks it against a dense 1-D scan.
namespace {

std::vector<double> bin_centres(int bins) {
    std::vector<double> centres(static_cast<std::size_t>(bins));
    for (int k = 0; k < bins; ++k) {
        centres[static_cast<std::size_t>(k)] =
            (bins <= 1) ? 0.0
                        : -1.0 + 2.0 * static_cast<double>(k) / static_cast<double>(bins - 1);
    }
    return centres;
}

/// The same interval sweep the library uses, kept in the test so a regression in
/// the closed form is caught independently of the library's copy.
double reference_prox(double v, double a, const std::vector<double>& h,
                      const std::vector<double>& c) {
    double total = 0.0;
    for (double value : h) {
        total += value;
    }
    if (total <= 0.0 || a <= 0.0) {
        return std::min(1.0, std::max(-1.0, v));
    }
    double left = 0.0;
    for (std::size_t j = 0; j <= c.size(); ++j) {
        const double candidate = v - a * (left - (total - left));
        const double lower = (j == 0) ? -1e300 : c[j - 1];
        const double upper = (j == c.size()) ? 1e300 : c[j];
        if (candidate > lower && candidate < upper) {
            return std::min(1.0, std::max(-1.0, candidate));
        }
        if (j < c.size()) {
            left += h[j];
        }
    }
    double lower_weight = 0.0;
    for (std::size_t k = 0; k < c.size(); ++k) {
        const double upper_weight = total - lower_weight - h[k];
        if (std::abs(v - c[k] - a * (lower_weight - upper_weight)) <= a * h[k]) {
            return std::min(1.0, std::max(-1.0, c[k]));
        }
        lower_weight += h[k];
    }
    return std::min(1.0, std::max(-1.0, v));
}

/// Dense scan of the objective 0.5 (u - v)^2 + a sum h_k |u - c_k| over [-1, 1].
double brute_force_prox(double v, double a, const std::vector<double>& h,
                        const std::vector<double>& c) {
    double best_u = -1.0;
    double best_cost = 1e300;
    const int steps = 200000;
    for (int s = 0; s <= steps; ++s) {
        const double u = -1.0 + 2.0 * static_cast<double>(s) / static_cast<double>(steps);
        double cost = 0.5 * (u - v) * (u - v);
        for (std::size_t k = 0; k < c.size(); ++k) {
            cost += a * h[k] * std::abs(u - c[k]);
        }
        if (cost < best_cost) {
            best_cost = cost;
            best_u = u;
        }
    }
    return best_u;
}

}  // namespace

TEST(TvFusion, ProxMatchesBruteForce) {
    std::uint32_t state = 777U;
    const std::vector<double> centres = bin_centres(8);
    double worst = 0.0;
    for (int trial = 0; trial < 200; ++trial) {
        const double v = -1.0 + 2.0 * uniform(state);
        const double a = 0.01 + 1.5 * uniform(state);
        std::vector<double> h(8U);
        for (double& value : h) {
            value = uniform(state);
        }
        const double reference = reference_prox(v, a, h, centres);
        const double brute = brute_force_prox(v, a, h, centres);
        worst = std::max(worst, std::abs(reference - brute));
    }
    std::cout << "[TvFusion] prox vs brute force worst |diff| = " << worst << "\n";
    EXPECT_LT(worst, 1e-4);
}

// Test 5: the sweeps are pure Jacobi, so the field must be bit-identical no
// matter how many threads OpenMP uses, and bad inputs must be rejected.
TEST(TvFusion, DeterministicAcrossThreadCountsAndValidates) {
    const Calibration calibration = sphere_calibration();
    const std::vector<Pose> poses = ring_poses();
    std::vector<DepthMap> maps;
    std::vector<FusionView> views;
    build_views(calibration, poses, Eigen::Vector3d::Zero(), 1.0, maps, views);

    TvFusionOptions options;
    options.max_resolution = 32;
    options.histogram_bins = 8;
    options.iterations = 40;
    const Eigen::Vector3d box_min(-1.5, -1.5, -1.5);
    const Eigen::Vector3d box_max(1.5, 1.5, 1.5);

#if defined(ALPHASHAPE_HAVE_OPENMP)
    omp_set_num_threads(1);
    const VolumeResult single = tv_l1_fusion(views, box_min, box_max, options);
    omp_set_num_threads(4);
    const VolumeResult parallel = tv_l1_fusion(views, box_min, box_max, options);
    ASSERT_EQ(single.field.data.size(), parallel.field.data.size());
    for (std::size_t k = 0; k < single.field.data.size(); ++k) {
        ASSERT_EQ(single.field.data[k], parallel.field.data[k]) << "voxel " << k;
    }
#else
    const VolumeResult single = tv_l1_fusion(views, box_min, box_max, options);
    const VolumeResult parallel = single;
#endif
    EXPECT_EQ(single.observed.size(), single.field.data.size());

    // Validation.
    const std::vector<FusionView> no_views;
    EXPECT_THROW(tv_l1_fusion(no_views, box_min, box_max, options), ValueError);

    std::vector<FusionView> null_view = {FusionView{}};
    EXPECT_THROW(tv_l1_fusion(null_view, box_min, box_max, options), ValueError);

    TvFusionOptions bad = options;
    bad.histogram_bins = 1;
    EXPECT_THROW(tv_l1_fusion(views, box_min, box_max, bad), ValueError);
    bad = options;
    bad.iterations = -1;
    EXPECT_THROW(tv_l1_fusion(views, box_min, box_max, bad), ValueError);
    bad = options;
    bad.lambda = 0.0;
    EXPECT_THROW(tv_l1_fusion(views, box_min, box_max, bad), ValueError);
    bad = options;
    bad.truncation_voxels = 0.0;
    EXPECT_THROW(tv_l1_fusion(views, box_min, box_max, bad), ValueError);
    bad = options;
    bad.front_band = -1.0;
    EXPECT_THROW(tv_l1_fusion(views, box_min, box_max, bad), ValueError);
    bad = options;
    bad.pixel_step = 0;
    EXPECT_THROW(tv_l1_fusion(views, box_min, box_max, bad), ValueError);

    DepthMap truncated;
    truncated.camera_id = 0;
    truncated.width = 4;
    truncated.height = 4;
    truncated.depths.assign(7U, -1.0F);
    FusionView short_view;
    short_view.map = &truncated;
    short_view.calibration = Calibration::initial(CameraType::Frame, 4, 4, 10.0);
    const std::vector<FusionView> short_views = {short_view};
    EXPECT_THROW(tv_l1_fusion(short_views, box_min, box_max, options), ValueError);

    FusionView mismatched = views.front();
    mismatched.calibration = Calibration::initial(CameraType::Frame, 8, 8, 10.0);
    const std::vector<FusionView> mismatched_views = {mismatched};
    EXPECT_THROW(tv_l1_fusion(mismatched_views, box_min, box_max, options), ValueError);
}

// Test 6: how the data weight trades surface fidelity against regularisation.
// No assertions here — the numbers are printed for the milestone report.
TEST(TvFusion, LambdaSweep) {
    const Calibration calibration = sphere_calibration();
    const std::vector<Pose> poses = ring_poses();
    std::vector<DepthMap> maps;
    std::vector<FusionView> views;
    build_views(calibration, poses, Eigen::Vector3d::Zero(), 1.0, maps, views);

    const Eigen::Vector3d box_min(-1.5, -1.5, -1.5);
    const Eigen::Vector3d box_max(1.5, 1.5, 1.5);
    for (const double lambda : {0.1, 0.3, 1.0, 3.0}) {
        TvFusionOptions options;
        options.max_resolution = 64;
        options.histogram_bins = 16;
        options.iterations = 200;
        options.lambda = lambda;
        const VolumeResult result = tv_l1_fusion(views, box_min, box_max, options);
        const SphereErrors errors = measure_sphere(result, 400, 12345U, false);
        std::cout << "[TvFusion] lambda " << lambda << " -> median " << errors.median
                  << " voxel (p95 " << errors.p95 << ", supported " << errors.supported << "/"
                  << errors.sampled << ")\n";
    }
    SUCCEED();
}

}  // namespace alphashape::meshing
