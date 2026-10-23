// Screened Poisson surface reconstruction (Kazhdan & Hoppe 2013): reconstruction
// accuracy on synthetic spheres, the screening term, the observed mask,
// determinism across thread counts and input validation.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base/algorithm/voxel_grid.hpp"
#include "base/utils/error.hpp"
#include "common/algorithm/mesh/poisson.hpp"
#include "model/mesh/point_cloud.hpp"

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

namespace alphashape {
namespace {

using meshing::PoissonOptions;
using meshing::PoissonStats;
using meshing::VolumeResult;

/// Points spread quasi-uniformly on the unit sphere (the Fibonacci lattice),
/// with the outward radial normal. `z_min` keeps only the upper hemisphere when
/// positive.
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

/// Deterministic standard-normal samples (Box-Muller on a small LCG).
struct NormalSampler {
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    double next_uniform() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 11) * (1.0 / 9007199254740992.0);
    }
    double next_normal() {
        const double u1 = std::max(next_uniform(), 1e-12);
        const double u2 = next_uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    }
};

/// Trilinear sample of a voxel-centre field at a world-space position (same
/// centre convention as the grid: voxel (x,y,z) sits at continuous (x,y,z)).
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

/// Radius at which the field crosses `iso` on the ray from the grid centre in
/// `direction`: samples every quarter voxel and linearly interpolates the first
/// sign change. Returns -1 when no crossing is found.
double crossing_radius(const VoxelGrid<float>& field, double iso, const Eigen::Vector3d& direction) {
    const double step = 0.25 * field.spec.voxel;
    const Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    double previous_t = 0.0;
    double previous_v = sample_field(field, origin) - iso;
    const double limit = 4.0;
    for (double t = step; t <= limit; t += step) {
        const double v = sample_field(field, origin + t * direction) - iso;
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

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const std::size_t idx = static_cast<std::size_t>(
        fraction * static_cast<double>(values.size() - 1) + 0.5);
    return values[std::min(idx, values.size() - 1)];
}

const Eigen::Vector3d kBox = Eigen::Vector3d::Constant(1.5);

/// The 200 directions used to probe the reconstructed radius: a second
/// Fibonacci lattice on the sphere.
std::vector<Eigen::Vector3d> probe_directions(std::size_t count) {
    std::vector<Eigen::Vector3d> dirs;
    const double golden = (1.0 + std::sqrt(5.0)) / 2.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double u = (static_cast<double>(i) + 0.5) / static_cast<double>(count);
        const double z = 1.0 - 2.0 * u;
        const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double theta = 2.0 * M_PI * static_cast<double>(i) / golden;
        dirs.emplace_back(r * std::cos(theta), r * std::sin(theta), z);
    }
    return dirs;
}

/// Median and 95th percentile of |r - 1| over the probe directions.
struct RadiusError {
    double median = 0.0;
    double p95 = 0.0;
    int misses = 0;
};

RadiusError radius_errors(const VoxelGrid<float>& field, double iso) {
    const std::vector<Eigen::Vector3d> dirs = probe_directions(200);
    std::vector<double> errors;
    RadiusError out;
    for (const Eigen::Vector3d& d : dirs) {
        const double r = crossing_radius(field, iso, d.normalized());
        if (r < 0.0) {
            ++out.misses;
            continue;
        }
        errors.push_back(std::abs(r - 1.0));
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

// Test 1: 20k points on the unit sphere, 64^3 grid over [-1.5, 1.5]^3. The
// reconstructed radius should match the true radius to well under a voxel, and
// the field must be negative at the centre and positive at the box corners.
TEST(Poisson, SphereRadiusAccuracy) {
    const PointCloud cloud = fibonacci_sphere(20000);
    PoissonOptions options;
    options.max_resolution = 64;
    const auto start = std::chrono::steady_clock::now();
    PoissonStats stats;
    const VolumeResult result = meshing::screened_poisson(cloud, -kBox, kBox, options, &stats);
    const auto stop = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();

    const RadiusError error = radius_errors(result.field, static_cast<double>(result.iso));
    const double voxel = result.field.spec.voxel;
    std::cout << "[poisson] sphere 64^3: iterations=" << stats.iterations
              << " rel_residual=" << stats.relative_residual
              << " median_err=" << error.median / voxel << " voxels"
              << " p95_err=" << error.p95 / voxel << " voxels"
              << " iso=" << result.iso << " seconds=" << seconds << std::endl;

    EXPECT_EQ(error.misses, 0);
    EXPECT_LT(error.median, 0.5 * voxel);
    EXPECT_LT(error.p95, 1.0 * voxel);

    // Sign convention: inside is below iso, outside above.
    const double centre = sample_field(result.field, Eigen::Vector3d::Zero());
    EXPECT_LT(centre, static_cast<double>(result.iso));
    const Eigen::Vector3d corner = kBox - Eigen::Vector3d::Constant(0.5 * voxel);
    EXPECT_GT(sample_field(result.field, corner), static_cast<double>(result.iso));
    EXPECT_GT(sample_field(result.field, -corner), static_cast<double>(result.iso));
    // A point just inside and just outside the surface.
    EXPECT_LT(sample_field(result.field, Eigen::Vector3d(0.7, 0.0, 0.0)),
              static_cast<double>(result.iso));
    EXPECT_GT(sample_field(result.field, Eigen::Vector3d(1.3, 0.0, 0.0)),
              static_cast<double>(result.iso));
}

// Test 2: positions jittered by sigma = 0.3 voxel and normals perturbed by up to
// 10 degrees. Poisson's global solve should still recover the sphere.
TEST(Poisson, NoisySphereRadiusAccuracy) {
    PointCloud cloud = fibonacci_sphere(20000);
    PoissonOptions options;
    options.max_resolution = 64;
    const GridSpec probe = make_grid(-kBox, kBox, options.max_resolution);
    const double sigma = 0.3 * probe.voxel;

    NormalSampler sampler;
    const double max_angle = 10.0 * M_PI / 180.0;
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        const Eigen::Vector3d radial = cloud.points[i].normalized();
        // Jitter along and across the surface.
        Eigen::Vector3d jitter(sampler.next_normal(), sampler.next_normal(),
                               sampler.next_normal());
        cloud.points[i] += sigma * jitter;
        // Rotate the normal by an angle drawn uniformly in [0, max_angle] about a
        // random axis perpendicular to it.
        Eigen::Vector3d axis = radial.cross(jitter);
        if (axis.norm() < 1e-9) {
            axis = radial.cross(Eigen::Vector3d::UnitX());
        }
        axis.normalize();
        const double angle = max_angle * sampler.next_uniform();
        const Eigen::AngleAxisd rotation(angle, axis);
        cloud.normals[i] = (rotation * radial).normalized().cast<float>();
    }

    PoissonStats stats;
    const VolumeResult result = meshing::screened_poisson(cloud, -kBox, kBox, options, &stats);
    const RadiusError error = radius_errors(result.field, static_cast<double>(result.iso));
    const double voxel = result.field.spec.voxel;
    std::cout << "[poisson] noisy sphere 64^3: iterations=" << stats.iterations
              << " median_err=" << error.median / voxel << " voxels"
              << " p95_err=" << error.p95 / voxel << " voxels" << std::endl;
    EXPECT_EQ(error.misses, 0);
    EXPECT_LT(error.median, 1.0 * voxel);
}

// Test 3: an upper hemisphere only. The observed mask must cover the sampled
// region and stay clear of the far underside of the sphere.
TEST(Poisson, HemisphereObservedMask) {
    const PointCloud cloud = fibonacci_sphere(20000, 0.0);
    PoissonOptions options;
    options.max_resolution = 64;
    const VolumeResult result = meshing::screened_poisson(cloud, -kBox, kBox, options);
    ASSERT_EQ(result.observed.size(), result.field.spec.size());

    const GridSpec& spec = result.field.spec;
    auto observed_at = [&](const Eigen::Vector3d& p) {
        const Eigen::Vector3d q = (p - spec.origin) / spec.voxel - Eigen::Vector3d::Constant(0.5);
        const int x = std::min(std::max(static_cast<int>(std::floor(q.x() + 0.5)), 0), spec.nx - 1);
        const int y = std::min(std::max(static_cast<int>(std::floor(q.y() + 0.5)), 0), spec.ny - 1);
        const int z = std::min(std::max(static_cast<int>(std::floor(q.z() + 0.5)), 0), spec.nz - 1);
        const std::size_t idx = spec.index(x, y, z);
        return idx < result.observed.size() && result.observed[idx] != 0;
    };

    // Voxels near the sampled surface must be observed.
    int observed_hits = 0;
    for (const Eigen::Vector3d& p : cloud.points) {
        if (observed_at(p)) {
            ++observed_hits;
        }
    }
    const double fraction = static_cast<double>(observed_hits) /
                            static_cast<double>(cloud.points.size());
    std::cout << "[poisson] hemisphere observed fraction near points=" << fraction << std::endl;
    EXPECT_GT(fraction, 0.9);

    // A large empty region far below the hemisphere (well beyond the dilation of
    // any point) must be unobserved.
    const int radius = static_cast<int>(std::ceil(options.density_radius_voxels));
    const double clearance = (static_cast<double>(radius) + 1.0) * spec.voxel;
    const Eigen::Vector3d far_below(0.0, 0.0, -1.0 - clearance);
    EXPECT_FALSE(observed_at(far_below));

    // Count unobserved voxels in the box under the sphere for a sanity bound: the
    // hemisphere only observes its own shell, so a good part of the box is clear.
    std::size_t unobserved = 0;
    for (std::uint8_t v : result.observed) {
        if (v == 0) {
            ++unobserved;
        }
    }
    std::cout << "[poisson] hemisphere unobserved voxels=" << unobserved << " of "
              << result.observed.size() << std::endl;
    EXPECT_GT(unobserved, result.observed.size() / 4);
}

// Test 4: the screening term should not degrade accuracy on the clean sphere,
// and the two solves should agree closely.
TEST(Poisson, ScreeningDoesNotDegradeAccuracy) {
    const PointCloud cloud = fibonacci_sphere(15000);
    PoissonOptions screened;
    screened.max_resolution = 48;
    screened.screening = 4.0;
    PoissonOptions unscreened = screened;
    unscreened.screening = 0.0;

    const VolumeResult with = meshing::screened_poisson(cloud, -kBox, kBox, screened);
    const VolumeResult without = meshing::screened_poisson(cloud, -kBox, kBox, unscreened);
    const RadiusError screened_error = radius_errors(with.field, static_cast<double>(with.iso));
    const RadiusError unscreened_error =
        radius_errors(without.field, static_cast<double>(without.iso));
    const double voxel = with.field.spec.voxel;
    std::cout << "[poisson] screening 4.0 median_err=" << screened_error.median / voxel
              << " voxels; screening 0.0 median_err=" << unscreened_error.median / voxel
              << " voxels" << std::endl;
    EXPECT_EQ(screened_error.misses, 0);
    EXPECT_EQ(unscreened_error.misses, 0);
    EXPECT_LE(screened_error.median, unscreened_error.median + 0.05 * voxel);
}

// Test 5: the solve is deterministic across thread counts, and invalid inputs
// are rejected.
TEST(Poisson, DeterminismAndValidation) {
    const PointCloud cloud = fibonacci_sphere(4000);
    PoissonOptions options;
    options.max_resolution = 32;

    // Same inputs, different OpenMP thread counts: the field must be identical.
#if defined(ALPHASHAPE_HAVE_OPENMP)
    omp_set_num_threads(1);
#endif
    const VolumeResult one = meshing::screened_poisson(cloud, -kBox, kBox, options);
#if defined(ALPHASHAPE_HAVE_OPENMP)
    omp_set_num_threads(4);
#endif
    const VolumeResult four = meshing::screened_poisson(cloud, -kBox, kBox, options);
#if defined(ALPHASHAPE_HAVE_OPENMP)
    omp_set_num_threads(omp_get_max_threads());
#endif

    ASSERT_EQ(one.field.data.size(), four.field.data.size());
    EXPECT_EQ(one.iso, four.iso);
    for (std::size_t i = 0; i < one.field.data.size(); ++i) {
        ASSERT_FLOAT_EQ(one.field.data[i], four.field.data[i]) << "voxel " << i;
    }
    EXPECT_EQ(one.observed, four.observed);

    // Validation errors.
    PointCloud empty;
    EXPECT_THROW(meshing::screened_poisson(empty, -kBox, kBox, options), ValueError);

    PointCloud no_normals = cloud;
    no_normals.normals.clear();
    EXPECT_THROW(meshing::screened_poisson(no_normals, -kBox, kBox, options), ValueError);

    PoissonOptions bad = options;
    bad.max_resolution = 2;
    EXPECT_THROW(meshing::screened_poisson(cloud, -kBox, kBox, bad), ValueError);

    bad = options;
    bad.screening = -1.0;
    EXPECT_THROW(meshing::screened_poisson(cloud, -kBox, kBox, bad), ValueError);

    bad = options;
    bad.max_iterations = 0;
    EXPECT_THROW(meshing::screened_poisson(cloud, -kBox, kBox, bad), ValueError);

    bad = options;
    bad.tolerance = 0.0;
    EXPECT_THROW(meshing::screened_poisson(cloud, -kBox, kBox, bad), ValueError);

    bad = options;
    bad.density_radius_voxels = -0.5;
    EXPECT_THROW(meshing::screened_poisson(cloud, -kBox, kBox, bad), ValueError);
}

// Test 6: a 128^3 run on the sphere, to show the solve scales.
TEST(Poisson, Sphere128Performance) {
    const PointCloud cloud = fibonacci_sphere(60000);
    PoissonOptions options;
    options.max_resolution = 128;
    const auto start = std::chrono::steady_clock::now();
    PoissonStats stats;
    const VolumeResult result = meshing::screened_poisson(cloud, -kBox, kBox, options, &stats);
    const auto stop = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();

    const RadiusError error = radius_errors(result.field, static_cast<double>(result.iso));
    const double voxel = result.field.spec.voxel;
    std::cout << "[poisson] sphere 128^3: iterations=" << stats.iterations
              << " rel_residual=" << stats.relative_residual
              << " median_err=" << error.median / voxel << " voxels"
              << " seconds=" << seconds << std::endl;
    EXPECT_GT(result.field.spec.nx, 1);
    EXPECT_EQ(error.misses, 0);
    EXPECT_LT(error.median, 0.75 * voxel);
    // Wall-clock budget, not a correctness bound: 60 s was tuned on the M1
    // development machine. This 4-core x86 host takes ~43 s alone and ~79 s
    // with a second test running beside it, so the budget is scaled for slower
    // hosts (the solve itself is seconds on the device path). The ASan+UBSan
    // build is several times slower again, so it gets that much more headroom.
#ifdef ALPHASHAPE_SANITIZED_BUILD
    EXPECT_LT(seconds, 600.0);
#else
    EXPECT_LT(seconds, 120.0);
#endif
}

}  // namespace
}  // namespace alphashape
