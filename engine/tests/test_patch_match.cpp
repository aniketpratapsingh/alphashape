#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base/utils/error.hpp"
#include "common/algorithm/stereo/patch_match.hpp"
#include "image/image.hpp"
#include "model/camera/calibration.hpp"

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

// Synthetic-rendering tests for the multi-view PatchMatch stereo.
//
// A world plane with a smooth procedural texture is rasterised by intersecting
// each pixel's unprojected ray with the plane; the reference and the neighbours
// are the same plane seen from different positions. The recovered depths are
// compared against the analytic ray/plane intersection, so no fixture files are
// needed and the expected result is known exactly.

namespace {

using alphashape::Calibration;
using alphashape::FloatImage;
using alphashape::stereo::PMView;
using alphashape::stereo::PatchMatchOptions;
using alphashape::stereo::PatchMatchResult;

struct Plane {
    Eigen::Vector3d p0 = Eigen::Vector3d::Zero();
    Eigen::Vector3d n = Eigen::Vector3d(0.0, 0.0, -1.0);
    Eigen::Vector3d u = Eigen::Vector3d(1.0, 0.0, 0.0);  // in-plane basis
    Eigen::Vector3d v = Eigen::Vector3d(0.0, 1.0, 0.0);
};

Plane make_plane(const Eigen::Vector3d& p0, const Eigen::Vector3d& n) {
    Plane plane;
    plane.p0 = p0;
    plane.n = n.normalized();
    plane.u = plane.n.unitOrthogonal();
    plane.v = plane.n.cross(plane.u);
    return plane;
}

// --- procedural texture ----------------------------------------------------

float hash2i(int x, int y) {
    std::uint32_t h = static_cast<std::uint32_t>(x) * 374761393U +
                      static_cast<std::uint32_t>(y) * 668265263U;
    h = (h ^ (h >> 13)) * 1274126177U;
    h ^= h >> 16;
    return static_cast<float>(h & 0xFFFFFFU) / static_cast<float>(0xFFFFFFU);
}

float vnoise(float x, float y) {
    const float fx = std::floor(x);
    const float fy = std::floor(y);
    const int x0 = static_cast<int>(fx);
    const int y0 = static_cast<int>(fy);
    const float ax = x - fx;
    const float ay = y - fy;
    const float sx = ax * ax * (3.0F - 2.0F * ax);
    const float sy = ay * ay * (3.0F - 2.0F * ay);
    const float c00 = hash2i(x0, y0);
    const float c10 = hash2i(x0 + 1, y0);
    const float c01 = hash2i(x0, y0 + 1);
    const float c11 = hash2i(x0 + 1, y0 + 1);
    const float top = c00 + (c10 - c00) * sx;
    const float bottom = c01 + (c11 - c01) * sx;
    return top + (bottom - top) * sy;
}

/// Smooth, non-periodic intensity in [0, 1] of a world point on the plane.
float texture(const Plane& plane, const Eigen::Vector3d& x_world) {
    const Eigen::Vector3d d = x_world - plane.p0;
    const double a = d.dot(plane.u);
    const double b = d.dot(plane.v);
    double value = 0.5 + 0.22 * std::sin(1.7 * a + 0.3) + 0.18 * std::sin(2.3 * b + 1.1) +
                   0.12 * std::sin(1.1 * a + 1.9 * b + 0.7) +
                   0.10 * std::sin(3.1 * a - 0.7 * b + 2.0) +
                   0.14 * static_cast<double>(vnoise(static_cast<float>(a * 1.9),
                                                    static_cast<float>(b * 1.9)) -
                                              0.5F);
    if (value < 0.0) {
        value = 0.0;
    }
    if (value > 1.0) {
        value = 1.0;
    }
    return static_cast<float>(value);
}

/// World-to-camera rotation looking from `c` toward `target` (+X right, +Y down,
/// +Z forward).
Eigen::Matrix3d look_at(const Eigen::Vector3d& c, const Eigen::Vector3d& target) {
    const Eigen::Vector3d forward = (target - c).normalized();
    const Eigen::Vector3d down(0.0, 1.0, 0.0);
    Eigen::Vector3d x = down.cross(forward);
    if (x.norm() < 1e-6) {
        x = Eigen::Vector3d(1.0, 0.0, 0.0);
    }
    x.normalize();
    const Eigen::Vector3d y = forward.cross(x);
    Eigen::Matrix3d r;
    r.row(0) = x;
    r.row(1) = y;
    r.row(2) = forward;
    return r;
}

/// Depth along the pixel's bearing where the ray meets the plane (camera frame).
double ray_plane_depth(const Plane& plane, const Calibration& calib,
                       const Eigen::Matrix3d& r, const Eigen::Vector3d& t, int x, int y) {
    const Eigen::Vector3d b = calib.unproject(
        Eigen::Vector2d(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5));
    const Eigen::Matrix3d rt = r.transpose();
    const Eigen::Vector3d dir = rt * b;
    const Eigen::Vector3d c = -rt * t;
    const double denom = plane.n.dot(dir);
    if (std::abs(denom) < 1e-9) {
        return -1.0;
    }
    return (plane.n.dot(plane.p0) - plane.n.dot(c)) / denom;
}

/// Rasterise `plane` for one camera. `constant` produces a flat image instead.
FloatImage render_view(const Plane& plane, const Calibration& calib, const Eigen::Matrix3d& r,
                       const Eigen::Vector3d& t, bool constant = false) {
    FloatImage image(calib.width, calib.height, 0.0F);
    const Eigen::Matrix3d rt = r.transpose();
    const Eigen::Vector3d c = -rt * t;
    const double dw = plane.n.dot(plane.p0);
    for (int y = 0; y < calib.height; ++y) {
        for (int x = 0; x < calib.width; ++x) {
            if (constant) {
                image.at(x, y) = 0.5F;
                continue;
            }
            double acc = 0.0;
            for (int sy = 0; sy < 2; ++sy) {
                for (int sx = 0; sx < 2; ++sx) {
                    const Eigen::Vector2d pixel(static_cast<double>(x) + 0.25 + 0.5 * sx,
                                                static_cast<double>(y) + 0.25 + 0.5 * sy);
                    const Eigen::Vector3d b = calib.unproject(pixel);
                    const Eigen::Vector3d dir = rt * b;
                    const double denom = plane.n.dot(dir);
                    if (std::abs(denom) < 1e-9) {
                        continue;
                    }
                    const double depth = (dw - plane.n.dot(c)) / denom;
                    if (depth > 0.0) {
                        acc += static_cast<double>(texture(plane, c + depth * dir));
                    }
                }
            }
            image.at(x, y) = static_cast<float>(acc * 0.25);
        }
    }
    return image;
}

/// Owns the images and exposes the PMView list for a scene.
struct Scene {
    Plane plane;
    std::vector<FloatImage> images;
    std::vector<Calibration> calibs;
    std::vector<Eigen::Matrix3d> rotations;
    std::vector<Eigen::Vector3d> translations;

    std::vector<PMView> views() const {
        std::vector<PMView> out;
        for (std::size_t i = 0; i < images.size(); ++i) {
            PMView v;
            v.image = &images[i];
            v.calibration = calibs[i];
            v.R = rotations[i];
            v.t = translations[i];
            out.push_back(v);
        }
        return out;
    }
};

double quantile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t idx = static_cast<std::size_t>(
        std::min<double>(static_cast<double>(values.size() - 1), q * static_cast<double>(values.size() - 1) + 0.5));
    return values[idx];
}

constexpr int kWidth = 160;
constexpr int kHeight = 120;
constexpr double kFocal = 150.0;

Calibration make_calibration(bool distorted) {
    Calibration calib = Calibration::initial(alphashape::CameraType::Frame, kWidth, kHeight, kFocal);
    if (distorted) {
        calib.k1 = -0.08;
        calib.k2 = 0.02;
    }
    return calib;
}

/// Reference at the world origin looking along +Z, four neighbours with
/// baselines of roughly 0.6-1.2 units.
struct SceneSpec {
    std::vector<Eigen::Vector3d> centres;
};

SceneSpec default_scene() {
    SceneSpec spec;
    spec.centres = {
        Eigen::Vector3d(0.0, 0.0, 0.0),      // reference
        Eigen::Vector3d(0.8, 0.0, 0.0),      // neighbour
        Eigen::Vector3d(-0.6, 0.35, 0.0),
        Eigen::Vector3d(0.4, -0.9, 0.25),
        Eigen::Vector3d(-0.5, -0.45, -0.3),
    };
    return spec;
}

Scene build_scene(const Plane& plane, const SceneSpec& spec, bool distort,
                  const Eigen::Vector3d& target, bool constant = false) {
    Scene scene;
    scene.plane = plane;
    const Calibration calib = make_calibration(distort);
    for (const Eigen::Vector3d& c : spec.centres) {
        Eigen::Matrix3d r;
        if (c.norm() < 1e-12) {
            r = Eigen::Matrix3d::Identity();  // reference looks along +Z
        } else {
            r = look_at(c, target);
        }
        const Eigen::Vector3d t = -r * c;
        scene.calibs.push_back(calib);
        scene.rotations.push_back(r);
        scene.translations.push_back(t);
        scene.images.push_back(render_view(plane, calib, r, t, constant));
    }
    return scene;
}

/// Coverage over the interior pixels and the relative / angular error stats
/// against the analytic plane.
struct Accuracy {
    double coverage = 0.0;
    double median_rel = 0.0;
    double p95_rel = 0.0;
    double median_angle_deg = 0.0;
    int interior = 0;
    int covered = 0;
};

Accuracy measure(const Scene& scene, const PatchMatchResult& result, int radius) {
    const Calibration& calib = scene.calibs[0];
    const Eigen::Matrix3d& r = scene.rotations[0];
    const Eigen::Vector3d& t = scene.translations[0];
    std::vector<double> rel;
    std::vector<double> angle;
    Accuracy acc;
    for (int y = radius; y < kHeight - radius; ++y) {
        for (int x = radius; x < kWidth - radius; ++x) {
            const double expected = ray_plane_depth(scene.plane, calib, r, t, x, y);
            if (expected <= 0.0) {
                continue;
            }
            ++acc.interior;
            const std::size_t index =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(kWidth) +
                static_cast<std::size_t>(x);
            const float d = result.depth[index];
            if (d <= 0.0F) {
                continue;
            }
            ++acc.covered;
            const double err = std::abs(static_cast<double>(d) - expected) / expected;
            rel.push_back(err);
            const std::size_t base = index * 3;
            const Eigen::Vector3d n(static_cast<double>(result.normals[base + 0]),
                                    static_cast<double>(result.normals[base + 1]),
                                    static_cast<double>(result.normals[base + 2]));
            if (n.squaredNorm() > 1e-9) {
                // Compare against the plane normal in the reference camera frame.
                const Eigen::Vector3d expected_n = r * scene.plane.n;
                const double cos_a = std::clamp(std::abs(n.normalized().dot(expected_n.normalized())),
                                                -1.0, 1.0);
                angle.push_back(std::acos(cos_a) * 180.0 / 3.14159265358979323846);
            }
        }
    }
    acc.coverage = acc.interior > 0 ? static_cast<double>(acc.covered) / acc.interior : 0.0;
    acc.median_rel = quantile(rel, 0.5);
    acc.p95_rel = quantile(rel, 0.95);
    acc.median_angle_deg = quantile(angle, 0.5);
    return acc;
}

PatchMatchOptions options_for(double dmin, double dmax) {
    PatchMatchOptions options;
    options.depth_min = dmin;
    options.depth_max = dmax;
    options.window_radius = 5;
    options.window_step = 2;
    options.iterations = 6;
    options.max_cost = 0.5;
    options.seed = 12345;
    // One level: these tests pin the single-level search they were written
    // against. The pyramid is covered by test_patch_match_pyramid.cpp.
    options.pyramid_levels = 1;
    return options;
}

/// A cheaper option set for tests that exercise the search structure (thread
/// determinism) rather than the accuracy of the converged result.
PatchMatchOptions reduced_options(double dmin, double dmax) {
    PatchMatchOptions options = options_for(dmin, dmax);
    options.window_radius = 3;
    options.iterations = 2;
    return options;
}

}  // namespace

// ---------------------------------------------------------------------------
// Sample convention
// ---------------------------------------------------------------------------

TEST(PatchMatch, SampleConventionPixelCentre) {
    // Pins the convention the cost loop relies on: FloatImage::sample at the
    // integer coordinate (i, j) is exactly the value of pixel (i, j), i.e. the
    // (i, j) sample sits at the pixel centre (i + 0.5, j + 0.5). This is the
    // same convention Calibration::project returns, so a projection is fed to
    // sample() without a half-pixel shift.
    FloatImage image(8, 6, 0.0F);
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 8; ++x) {
            image.at(x, y) = static_cast<float>(x) + 0.1F * static_cast<float>(y);
        }
    }
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 8; ++x) {
            EXPECT_FLOAT_EQ(image.sample(static_cast<float>(x), static_cast<float>(y)), image.at(x, y));
        }
    }
    // Halfway between the centres of (2, 3) and (3, 3) is the shared edge.
    const float mid = image.sample(2.5F, 3.0F);
    EXPECT_FLOAT_EQ(mid, 0.5F * (image.at(2, 3) + image.at(3, 3)));
}

// ---------------------------------------------------------------------------
// Fronto-parallel plane
// ---------------------------------------------------------------------------

TEST(PatchMatch, FrontoParallelPlane) {
    const Plane plane = make_plane(Eigen::Vector3d(0.0, 0.0, 10.0), Eigen::Vector3d(0.0, 0.0, -1.0));
    const Scene scene = build_scene(plane, default_scene(), /*distort=*/false,
                                    Eigen::Vector3d(0.0, 0.0, 10.0));
    const std::vector<PMView> views = scene.views();
    const auto start = std::chrono::steady_clock::now();
    const PatchMatchResult result =
        alphashape::stereo::run_patch_match(views[0], {views[1], views[2], views[3], views[4]},
                                            options_for(5.0, 20.0));
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const Accuracy acc = measure(scene, result, /*radius=*/5);
    std::printf("[fronto-parallel] coverage=%.4f median_rel=%.5f p95_rel=%.5f runtime=%.3fs\n",
                acc.coverage, acc.median_rel, acc.p95_rel, seconds);
    EXPECT_GT(acc.coverage, 0.85);
    EXPECT_LT(acc.median_rel, 0.005);
    EXPECT_LT(acc.p95_rel, 0.02);
}

// ---------------------------------------------------------------------------
// Slanted plane
// ---------------------------------------------------------------------------

TEST(PatchMatch, SlantedPlane) {
    const Eigen::Vector3d normal =
        Eigen::Vector3d(0.0, std::sin(35.0 * 3.14159265358979323846 / 180.0),
                        -std::cos(35.0 * 3.14159265358979323846 / 180.0));
    const Plane plane = make_plane(Eigen::Vector3d(0.0, 0.0, 10.0), normal);
    const Scene scene =
        build_scene(plane, default_scene(), /*distort=*/false, Eigen::Vector3d(0.0, 0.0, 10.0));
    const std::vector<PMView> views = scene.views();
    const PatchMatchResult result =
        alphashape::stereo::run_patch_match(views[0], {views[1], views[2], views[3], views[4]},
                                            options_for(5.0, 20.0));
    const Accuracy acc = measure(scene, result, /*radius=*/5);
    std::printf("[slanted] coverage=%.4f median_rel=%.5f median_angle_deg=%.4f\n", acc.coverage,
                acc.median_rel, acc.median_angle_deg);
    EXPECT_GT(acc.coverage, 0.85);
    EXPECT_LT(acc.median_rel, 0.01);
    EXPECT_LT(acc.median_angle_deg, 10.0);
}

// ---------------------------------------------------------------------------
// Distorted lenses
// ---------------------------------------------------------------------------

TEST(PatchMatch, DistortedProjection) {
    const Plane plane = make_plane(Eigen::Vector3d(0.0, 0.0, 10.0), Eigen::Vector3d(0.0, 0.0, -1.0));
    const Scene scene = build_scene(plane, default_scene(), /*distort=*/true,
                                    Eigen::Vector3d(0.0, 0.0, 10.0));
    const std::vector<PMView> views = scene.views();
    const PatchMatchResult result =
        alphashape::stereo::run_patch_match(views[0], {views[1], views[2], views[3], views[4]},
                                            options_for(5.0, 20.0));
    const Accuracy acc = measure(scene, result, /*radius=*/5);
    std::printf("[distorted] coverage=%.4f median_rel=%.5f p95_rel=%.5f\n", acc.coverage,
                acc.median_rel, acc.p95_rel);
    EXPECT_GT(acc.coverage, 0.85);
    EXPECT_LT(acc.median_rel, 0.005);
    EXPECT_LT(acc.p95_rel, 0.02);
}

// ---------------------------------------------------------------------------
// Determinism across thread counts
// ---------------------------------------------------------------------------

TEST(PatchMatch, ThreadCountDeterminism) {
    const Plane plane = make_plane(Eigen::Vector3d(0.0, 0.0, 10.0), Eigen::Vector3d(0.0, 0.0, -1.0));
    const Scene scene =
        build_scene(plane, default_scene(), /*distort=*/false, Eigen::Vector3d(0.0, 0.0, 10.0));
    const std::vector<PMView> views = scene.views();
    const std::vector<PMView> neighbours = {views[1], views[2], views[3], views[4]};
    const PatchMatchOptions options = reduced_options(5.0, 20.0);

#if defined(ALPHASHAPE_HAVE_OPENMP)
    omp_set_num_threads(1);
#else
    GTEST_SKIP() << "OpenMP not available";
#endif
    const PatchMatchResult serial = alphashape::stereo::run_patch_match(views[0], neighbours, options);
#if defined(ALPHASHAPE_HAVE_OPENMP)
    omp_set_num_threads(4);
#endif
    const PatchMatchResult parallel =
        alphashape::stereo::run_patch_match(views[0], neighbours, options);

    ASSERT_EQ(serial.depth.size(), parallel.depth.size());
    EXPECT_EQ(serial.depth, parallel.depth);
    EXPECT_EQ(serial.normals, parallel.normals);
    EXPECT_EQ(serial.cost, parallel.cost);
#if defined(ALPHASHAPE_HAVE_OPENMP)
    omp_set_num_threads(omp_get_max_threads());
#endif
}

// ---------------------------------------------------------------------------
// The per-sweep progress hook
// ---------------------------------------------------------------------------

TEST(PatchMatch, SweepHookReportsMonotoneFractions) {
    const Plane plane = make_plane(Eigen::Vector3d(0.0, 0.0, 10.0), Eigen::Vector3d(0.0, 0.0, -1.0));
    const Scene scene =
        build_scene(plane, default_scene(), /*distort=*/false, Eigen::Vector3d(0.0, 0.0, 10.0));
    const std::vector<PMView> views = scene.views();
    const std::vector<PMView> neighbours = {views[1], views[2], views[3], views[4]};

    PatchMatchOptions options = reduced_options(5.0, 20.0);  // 2 iterations -> 4 half-sweeps
    std::vector<double> reported;
    options.on_sweep = [&](double fraction) { reported.push_back(fraction); };

    const PatchMatchResult with_hook =
        alphashape::stereo::run_patch_match(views[0], neighbours, options);

    // One report per half-sweep, strictly increasing, ending at 1.0.
    ASSERT_EQ(reported.size(), 2U * static_cast<std::size_t>(options.iterations));
    for (std::size_t i = 0; i < reported.size(); ++i) {
        EXPECT_GT(reported[i], 0.0) << "report " << i;
        EXPECT_LE(reported[i], 1.0) << "report " << i;
        if (i > 0) {
            EXPECT_GT(reported[i], reported[i - 1]) << "report " << i << " went backwards";
        }
    }
    EXPECT_DOUBLE_EQ(reported.back(), 1.0);

    // The hook changes nothing about the result: it is a report, not an input.
    options.on_sweep = nullptr;
    const PatchMatchResult without_hook =
        alphashape::stereo::run_patch_match(views[0], neighbours, options);
    EXPECT_EQ(with_hook.depth, without_hook.depth);
    EXPECT_EQ(with_hook.normals, without_hook.normals);
    EXPECT_EQ(with_hook.cost, without_hook.cost);
}

TEST(PatchMatch, AThrowingSweepHookStopsTheSearch) {
    // A task's cancellation is a throw from the hook; the search must let it
    // propagate at once rather than finish the remaining sweeps, and no result
    // is produced (so no half-swept map can be mistaken for a finished one).
    const Plane plane = make_plane(Eigen::Vector3d(0.0, 0.0, 10.0), Eigen::Vector3d(0.0, 0.0, -1.0));
    const Scene scene =
        build_scene(plane, default_scene(), /*distort=*/false, Eigen::Vector3d(0.0, 0.0, 10.0));
    const std::vector<PMView> views = scene.views();
    const std::vector<PMView> neighbours = {views[1], views[2], views[3], views[4]};

    PatchMatchOptions options = options_for(5.0, 20.0);  // 6 iterations -> 12 half-sweeps
    std::size_t reports = 0;
    options.on_sweep = [&](double) {
        ++reports;
        if (reports == 1) {
            throw alphashape::CancelledError("stop after one sweep");
        }
    };
    EXPECT_THROW(alphashape::stereo::run_patch_match(views[0], neighbours, options),
                 alphashape::CancelledError);
    EXPECT_EQ(reports, 1U) << "the search kept sweeping after the hook threw";
}

// ---------------------------------------------------------------------------
// Textureless reference
// ---------------------------------------------------------------------------

TEST(PatchMatch, TexturelessReferenceHasNoDepth) {
    const Plane plane = make_plane(Eigen::Vector3d(0.0, 0.0, 10.0), Eigen::Vector3d(0.0, 0.0, -1.0));
    const Scene scene = build_scene(plane, default_scene(), /*distort=*/false,
                                    Eigen::Vector3d(0.0, 0.0, 10.0), /*constant=*/true);
    const std::vector<PMView> views = scene.views();
    const PatchMatchResult result =
        alphashape::stereo::run_patch_match(views[0], {views[1], views[2], views[3], views[4]},
                                            options_for(5.0, 20.0));
    for (float d : result.depth) {
        EXPECT_FLOAT_EQ(d, -1.0F);
    }
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

TEST(PatchMatch, InvalidInputsThrow) {
    const Plane plane = make_plane(Eigen::Vector3d(0.0, 0.0, 10.0), Eigen::Vector3d(0.0, 0.0, -1.0));
    const Scene scene =
        build_scene(plane, default_scene(), /*distort=*/false, Eigen::Vector3d(0.0, 0.0, 10.0));
    const std::vector<PMView> views = scene.views();
    const std::vector<PMView> neighbours = {views[1], views[2]};
    const PatchMatchOptions good = options_for(5.0, 20.0);

    PMView null_ref = views[0];
    null_ref.image = nullptr;
    EXPECT_THROW(
        alphashape::stereo::run_patch_match(null_ref, neighbours, good), alphashape::ValueError);

    PMView bad_calib = views[0];
    bad_calib.calibration.width += 1;
    EXPECT_THROW(
        alphashape::stereo::run_patch_match(bad_calib, neighbours, good), alphashape::ValueError);

    EXPECT_THROW(alphashape::stereo::run_patch_match(views[0], {}, good), alphashape::ValueError);

    PMView null_nb = views[1];
    null_nb.image = nullptr;
    EXPECT_THROW(alphashape::stereo::run_patch_match(views[0], {null_nb}, good),
                 alphashape::ValueError);

    PatchMatchOptions bad_range = good;
    bad_range.depth_min = 20.0;
    bad_range.depth_max = 5.0;
    EXPECT_THROW(alphashape::stereo::run_patch_match(views[0], neighbours, bad_range),
                 alphashape::ValueError);

    PatchMatchOptions zero_min = good;
    zero_min.depth_min = 0.0;
    EXPECT_THROW(alphashape::stereo::run_patch_match(views[0], neighbours, zero_min),
                 alphashape::ValueError);
}
