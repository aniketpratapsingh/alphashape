// Tests for WP-13.5 marker detection: direct ellipse fitting, Hessian-detent
// blob detection with subpixel refinement, polarity handling, quad detection,
// and the 12-megapixel timing case. Synthetic sheets come from
// make_marker_sheet.hpp; no external data is used.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numbers>
#include <random>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/utils/portable_rng.hpp"
#include "common/algorithm/feature/dog_detector.hpp"
#include "common/algorithm/markers/marker_detector.hpp"
#include "image/image.hpp"
#include "make_marker_sheet.hpp"
#include "perf_gate.hpp"

namespace alphashape {
namespace {

constexpr double kPi = std::numbers::pi_v<double>;

double angle_distance_mod_pi(double a, double b) {
    double difference = std::fmod(a - b, kPi);
    if (difference > 0.5 * kPi) difference -= kPi;
    if (difference < -0.5 * kPi) difference += kPi;
    return difference;
}

/// The detection nearest a ground-truth centre, or nullptr if none is close.
const MarkerCandidate* nearest_candidate(const std::vector<MarkerCandidate>& candidates,
                                         const Eigen::Vector2d& truth) {
    const MarkerCandidate* best = nullptr;
    double best_distance = std::numeric_limits<double>::infinity();
    for (const MarkerCandidate& candidate : candidates) {
        const double distance = (candidate.center - truth).norm();
        if (distance < best_distance) {
            best_distance = distance;
            best = &candidate;
        }
    }
    return best;
}

TEST(MarkerDetector, FitEllipseExactOnSamples) {
    const double centre_x = 137.25;
    const double centre_y = 92.5;
    const double semi_major = 53.5;
    const double semi_minor = 26.25;
    const double angle = 0.7;
    std::vector<Eigen::Vector2d> points;
    points.reserve(static_cast<std::size_t>(200));
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    for (int i = 0; i < 200; ++i) {
        const double t = 2.0 * kPi * static_cast<double>(i) / 200.0;
        const double u = semi_major * std::cos(t);
        const double v = semi_minor * std::sin(t);
        points.emplace_back(centre_x + u * c - v * s, centre_y + u * s + v * c);
    }
    const auto fit = fit_ellipse(points);
    ASSERT_TRUE(fit.has_value());
    EXPECT_NEAR(fit->center.x(), centre_x, 1e-9);
    EXPECT_NEAR(fit->center.y(), centre_y, 1e-9);
    EXPECT_NEAR(fit->a, semi_major, 1e-9);
    EXPECT_NEAR(fit->b, semi_minor, 1e-9);
    EXPECT_NEAR(angle_distance_mod_pi(fit->angle_rad, angle), 0.0, 1e-9);
    std::printf("[FitEllipseExact] centre=(%.12f, %.12f) a=%.12f b=%.12f angle=%.12f\n",
                fit->center.x(), fit->center.y(), fit->a, fit->b, fit->angle_rad);

    // Fewer than five points is not enough for a conic.
    const std::vector<Eigen::Vector2d> too_few(points.begin(), points.begin() + 4);
    EXPECT_FALSE(fit_ellipse(too_few).has_value());
    // Collinear samples have no ellipse through them.
    std::vector<Eigen::Vector2d> collinear;
    for (int i = 0; i < 20; ++i) {
        collinear.emplace_back(3.0 + static_cast<double>(i), 7.0 + 2.0 * static_cast<double>(i));
    }
    EXPECT_FALSE(fit_ellipse(collinear).has_value());
}

/// Deterministic ellipses on a rectangular grid of cells, jittered inside each
/// cell so no two targets can overlap.
std::vector<test::EllipseTruth> make_ellipse_truths(int columns, int count, double cell_width,
                                                    double cell_height, double min_radius,
                                                    double max_radius, std::uint32_t seed) {
    std::vector<test::EllipseTruth> truths;
    truths.reserve(static_cast<std::size_t>(count));
    std::mt19937 engine(seed);
    for (int i = 0; i < count; ++i) {
        const int column = i % columns;
        const int row = i / columns;
        const double jitter_x = rng::uniform_real(engine, -5.0, 5.0);
        const double jitter_y = rng::uniform_real(engine, -5.0, 5.0);
        const double radius = rng::uniform_real(engine, min_radius, max_radius);
        const double ratio = rng::uniform_real(engine, 0.5, 1.0);
        const double angle = rng::uniform_real(engine, 0.0, kPi);
        test::EllipseTruth truth;
        truth.cx = (static_cast<double>(column) + 0.5) * cell_width + jitter_x;
        truth.cy = (static_cast<double>(row) + 0.5) * cell_height + jitter_y;
        truth.a = radius;
        truth.b = radius * ratio;
        truth.angle_rad = angle;
        truths.push_back(truth);
    }
    return truths;
}

struct AccuracySummary {
    double rms = 0.0;
    int matched = 0;
    int missed = 0;
};

AccuracySummary centre_accuracy(const FloatImage& image,
                                const std::vector<test::EllipseTruth>& truths,
                                const BlobParams& params) {
    const std::vector<MarkerCandidate> candidates = detect_blobs(image, params);
    AccuracySummary summary;
    double squared = 0.0;
    for (const test::EllipseTruth& truth : truths) {
        const Eigen::Vector2d expected(truth.cx, truth.cy);
        const MarkerCandidate* candidate = nearest_candidate(candidates, expected);
        if (candidate == nullptr || (candidate->center - expected).norm() > 5.0) {
            ++summary.missed;
            continue;
        }
        const auto refined = refine_ellipse_center(image, *candidate, true);
        if (!refined) {
            ++summary.missed;
            continue;
        }
        const double error = (*refined - expected).norm();
        squared += error * error;
        ++summary.matched;
    }
    if (summary.matched > 0) {
        summary.rms = std::sqrt(squared / static_cast<double>(summary.matched));
    }
    return summary;
}

TEST(MarkerDetector, CentersAccurateToSubpixel) {
    const int width = 2048;
    const int height = 1024;
    const std::vector<test::EllipseTruth> truths =
        make_ellipse_truths(20, 200, static_cast<double>(width) / 20.0,
                            static_cast<double>(height) / 10.0, 6.0, 40.0, 20261101U);
    FloatImage image = test::render_ellipses(width, height, truths, 0.95F, 0.05F);
    image = gaussian_blur(image, 1.0F);
    test::add_noise(image, 0.01, 7U);

    const BlobParams params;
    const AccuracySummary summary = centre_accuracy(image, truths, params);
    std::printf("[CentersAccurate] matched=%d missed=%d RMS=%.6f px\n", summary.matched,
                summary.missed, summary.rms);
    EXPECT_EQ(summary.missed, 0);
    EXPECT_LE(summary.rms, 0.05);

    // Baseline JPEG quality 70: block DCT quantisation on top of the noise.
    const FloatImage degraded = test::degrade_like_jpeg(image, 70);
    const AccuracySummary degraded_summary = centre_accuracy(degraded, truths, params);
    std::printf("[CentersAccurateJpeg70] matched=%d missed=%d RMS=%.6f px\n",
                degraded_summary.matched, degraded_summary.missed, degraded_summary.rms);
    EXPECT_EQ(degraded_summary.missed, 0);
    EXPECT_LE(degraded_summary.rms, 0.1);
}

TEST(MarkerDetector, AxisRatioMatchesTruth) {
    const int width = 1024;
    const int height = 1024;
    const std::vector<test::EllipseTruth> truths =
        make_ellipse_truths(8, 64, static_cast<double>(width) / 8.0,
                            static_cast<double>(height) / 8.0, 20.0, 40.0, 20261102U);
    FloatImage image = test::render_ellipses(width, height, truths, 0.95F, 0.05F);
    image = gaussian_blur(image, 1.0F);
    test::add_noise(image, 0.005, 11U);

    const std::vector<MarkerCandidate> candidates = detect_blobs(image);
    double worst_relative = 0.0;
    double sum_relative = 0.0;
    int matched = 0;
    int missed = 0;
    for (const test::EllipseTruth& truth : truths) {
        const Eigen::Vector2d expected(truth.cx, truth.cy);
        const MarkerCandidate* candidate = nearest_candidate(candidates, expected);
        if (candidate == nullptr || (candidate->center - expected).norm() > 5.0) {
            ++missed;
            continue;
        }
        const double truth_ratio = truth.b / truth.a;
        const double relative = std::abs(candidate->axis_ratio - truth_ratio) / truth_ratio;
        worst_relative = std::max(worst_relative, relative);
        sum_relative += relative;
        ++matched;
    }
    const double mean_relative = matched > 0 ? sum_relative / static_cast<double>(matched) : 1.0;
    std::printf("[AxisRatio] matched=%d missed=%d mean=%.5f worst=%.5f\n", matched, missed,
                mean_relative, worst_relative);
    EXPECT_EQ(missed, 0);
    EXPECT_LE(worst_relative, 0.02);
}

TEST(MarkerDetector, RejectsNoise) {
    FloatImage image(1024, 1024, 0.5F);
    test::add_noise(image, 0.05, 20261103U);
    BlobParams params;
    params.max_radius = 32;
    const std::vector<MarkerCandidate> candidates = detect_blobs(image, params);
    std::printf("[RejectsNoise] candidates=%d threshold=%.3f\n",
                static_cast<int>(candidates.size()), static_cast<double>(params.threshold));
    EXPECT_LE(candidates.size(), static_cast<std::size_t>(1));
}

TEST(MarkerDetector, LightOnDarkPolarity) {
    const int size = 256;
    BlobParams dark_on_light;
    BlobParams light_on_dark;
    light_on_dark.dark_on_light = false;

    const std::vector<test::EllipseTruth> dark_dot = {{80.0, 128.0, 12.0, 12.0, 0.0}};
    FloatImage bright = test::render_ellipses(size, size, dark_dot, 0.9F, 0.1F);
    bright = gaussian_blur(bright, 1.0F);
    const std::vector<MarkerCandidate> found_dark = detect_blobs(bright, dark_on_light);
    ASSERT_EQ(found_dark.size(), static_cast<std::size_t>(1));
    EXPECT_NEAR(found_dark.front().center.x(), 80.0, 1.0);
    EXPECT_NEAR(found_dark.front().center.y(), 128.0, 1.0);
    EXPECT_TRUE(detect_blobs(bright, light_on_dark).empty());

    const std::vector<test::EllipseTruth> light_dot = {{176.0, 128.0, 12.0, 12.0, 0.0}};
    FloatImage dark = test::render_ellipses(size, size, light_dot, 0.1F, 0.9F);
    dark = gaussian_blur(dark, 1.0F);
    const std::vector<MarkerCandidate> found_light = detect_blobs(dark, light_on_dark);
    ASSERT_EQ(found_light.size(), static_cast<std::size_t>(1));
    EXPECT_NEAR(found_light.front().center.x(), 176.0, 1.0);
    EXPECT_NEAR(found_light.front().center.y(), 128.0, 1.0);
    EXPECT_TRUE(detect_blobs(dark, dark_on_light).empty());
}

TEST(MarkerDetector, QuadCornersAccurate) {
    const int width = 1024;
    const int height = 1024;
    const int columns = 4;
    const int rows = 3;
    std::mt19937 engine(20261104U);
    std::vector<test::SquareTruth> truths;
    truths.reserve(static_cast<std::size_t>(columns * rows));
    const double cell_width = static_cast<double>(width) / static_cast<double>(columns);
    const double cell_height = static_cast<double>(height) / static_cast<double>(rows);
    for (int i = 0; i < columns * rows; ++i) {
        const int column = i % columns;
        const int row = i / columns;
        const double jitter_x = rng::uniform_real(engine, -8.0, 8.0);
        const double jitter_y = rng::uniform_real(engine, -8.0, 8.0);
        const double side = rng::uniform_real(engine, 20.0, 120.0);
        const double angle = rng::uniform_real(engine, -0.5 * kPi, 0.5 * kPi);
        test::SquareTruth truth;
        truth.cx = (static_cast<double>(column) + 0.5) * cell_width + jitter_x;
        truth.cy = (static_cast<double>(row) + 0.5) * cell_height + jitter_y;
        truth.side = side;
        truth.angle_rad = angle;
        truths.push_back(truth);
    }
    FloatImage image = test::render_squares(width, height, truths, 0.9F, 0.1F);
    image = gaussian_blur(image, 1.0F);
    test::add_noise(image, 0.01, 13U);

    const std::vector<MarkerCandidate> candidates = detect_quads(image, QuadParams{});
    double squared = 0.0;
    int corner_count = 0;
    int missed = 0;
    for (const test::SquareTruth& truth : truths) {
        const Eigen::Vector2d expected(truth.cx, truth.cy);
        const MarkerCandidate* candidate = nearest_candidate(candidates, expected);
        if (candidate == nullptr || (candidate->center - expected).norm() > 5.0 ||
            candidate->corners.size() != static_cast<std::size_t>(4)) {
            ++missed;
            continue;
        }
        const std::array<Eigen::Vector2d, 4> corners = test::square_corners(truth);
        for (const Eigen::Vector2d& corner : corners) {
            double nearest = std::numeric_limits<double>::infinity();
            for (const Eigen::Vector2d& detected : candidate->corners) {
                nearest = std::min(nearest, (detected - corner).norm());
            }
            squared += nearest * nearest;
            ++corner_count;
        }
    }
    const double rms =
        corner_count > 0 ? std::sqrt(squared / static_cast<double>(corner_count)) : 1e9;
    std::printf("[QuadCorners] matched=%d missed=%d corners=%d RMS=%.6f px\n",
                static_cast<int>(truths.size()) - missed, missed, corner_count, rms);
    EXPECT_EQ(missed, 0);
    EXPECT_LE(rms, 0.1);
}

TEST(MarkerDetector, TwelveMegapixelTiming) {
    const int width = 4000;
    const int height = 3000;
    const int columns = 10;
    const int rows = 10;
    const double cell_width = static_cast<double>(width) / static_cast<double>(columns);
    const double cell_height = static_cast<double>(height) / static_cast<double>(rows);
    std::mt19937 engine(20261105U);
    std::vector<test::EllipseTruth> truths;
    truths.reserve(static_cast<std::size_t>(columns * rows));
    for (int i = 0; i < columns * rows; ++i) {
        const int column = i % columns;
        const int row = i / columns;
        const double jitter_x = rng::uniform_real(engine, -30.0, 30.0);
        const double jitter_y = rng::uniform_real(engine, -30.0, 30.0);
        const double radius = rng::uniform_real(engine, 8.0, 16.0);
        const double ratio = rng::uniform_real(engine, 0.7, 1.0);
        const double angle = rng::uniform_real(engine, 0.0, kPi);
        test::EllipseTruth truth;
        truth.cx = (static_cast<double>(column) + 0.5) * cell_width + jitter_x;
        truth.cy = (static_cast<double>(row) + 0.5) * cell_height + jitter_y;
        truth.a = radius;
        truth.b = radius * ratio;
        truth.angle_rad = angle;
        truths.push_back(truth);
    }
    FloatImage image = test::render_ellipses(width, height, truths, 0.95F, 0.05F);
    image = gaussian_blur(image, 1.0F);
    test::add_noise(image, 0.01, 17U);

    BlobParams params;
    params.min_radius = 4;
    params.max_radius = 24;
    const auto start = std::chrono::steady_clock::now();
    const std::vector<MarkerCandidate> candidates = detect_blobs(image, params);
    const auto finish = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(finish - start).count();
    int matched = 0;
    for (const test::EllipseTruth& truth : truths) {
        const Eigen::Vector2d expected(truth.cx, truth.cy);
        const MarkerCandidate* candidate = nearest_candidate(candidates, expected);
        if (candidate != nullptr && (candidate->center - expected).norm() < 3.0) {
            ++matched;
        }
    }
    std::printf("[TwelveMegapixel] seconds=%.3f candidates=%d matched=%d\n", seconds,
                static_cast<int>(candidates.size()), matched);
    EXPECT_GE(matched, 95);
    if (test::perf_asserts_enabled()) {
        EXPECT_LE(seconds, 1.5);
    }
}

}  // namespace
}  // namespace alphashape
