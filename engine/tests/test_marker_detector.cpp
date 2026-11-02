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
    std::printf("[marker fit-ellipse] centre=(%.12f, %.12f) a=%.12f b=%.12f angle=%.12f\n",
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
    // Non-finite samples are filtered before the five-point minimum is counted,
    // so four NaNs plus one usable point is not a five-sample fit.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<Eigen::Vector2d> mostly_nan = {
        {nan, 4.0}, {4.0, nan}, {inf, 9.0}, {9.0, -inf}, {31.0, 12.0}};
    EXPECT_FALSE(fit_ellipse(mostly_nan).has_value());
    std::vector<Eigen::Vector2d> two_valid = {
        {nan, 4.0}, {4.0, nan}, {inf, 9.0}, {9.0, -inf}, {31.0, 12.0}, {34.0, 18.0}};
    EXPECT_FALSE(fit_ellipse(two_valid).has_value());
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
    // One gradient field for every refinement in this sheet.
    const GradientField gradient = compute_gradient_field(image);
    AccuracySummary summary;
    double squared = 0.0;
    for (const test::EllipseTruth& truth : truths) {
        const Eigen::Vector2d expected(truth.cx, truth.cy);
        const MarkerCandidate* candidate = nearest_candidate(candidates, expected);
        if (candidate == nullptr || (candidate->center - expected).norm() > 5.0) {
            ++summary.missed;
            continue;
        }
        const auto refined = refine_ellipse_center(gradient, *candidate);
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
    std::printf("[marker centres] matched=%d missed=%d RMS=%.6f px\n", summary.matched,
                summary.missed, summary.rms);
    EXPECT_EQ(summary.missed, 0);
    EXPECT_LE(summary.rms, 0.05);

    // Baseline JPEG quality 70: block DCT quantisation on top of the noise.
    const FloatImage degraded = test::degrade_like_jpeg(image, 70);
    const AccuracySummary degraded_summary = centre_accuracy(degraded, truths, params);
    std::printf("[marker centres jpeg70] matched=%d missed=%d RMS=%.6f px\n",
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
    std::printf("[marker axis-ratio] matched=%d missed=%d mean=%.5f worst=%.5f\n", matched,
                missed, mean_relative, worst_relative);
    EXPECT_EQ(missed, 0);
    EXPECT_LE(worst_relative, 0.02);
}

/// Straightforward single-threaded version of `gaussian_blur`, used to prove
/// that the parallel passes produce bit-identical values.
FloatImage serial_gaussian_blur(const FloatImage& image, float sigma) {
    if (sigma <= 0.0F || image.empty()) {
        return image;
    }
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0F * sigma)));
    std::vector<float> kernel(static_cast<std::size_t>(2 * radius + 1));
    float sum = 0.0F;
    for (int i = -radius; i <= radius; ++i) {
        const float w = std::exp(-static_cast<float>(i * i) / (2.0F * sigma * sigma));
        kernel[static_cast<std::size_t>(i + radius)] = w;
        sum += w;
    }
    for (float& w : kernel) w /= sum;
    FloatImage tmp(image.width(), image.height());
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            float acc = 0.0F;
            for (int i = -radius; i <= radius; ++i) {
                acc += kernel[static_cast<std::size_t>(i + radius)] * image.at_clamped(x + i, y);
            }
            tmp.at(x, y) = acc;
        }
    }
    FloatImage out(image.width(), image.height());
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            float acc = 0.0F;
            for (int i = -radius; i <= radius; ++i) {
                acc += kernel[static_cast<std::size_t>(i + radius)] * tmp.at_clamped(x, y + i);
            }
            out.at(x, y) = acc;
        }
    }
    return out;
}

TEST(MarkerDetector, RejectsNoise) {
    FloatImage image(1024, 1024, 0.5F);
    test::add_noise(image, 0.05, 20261103U);
    BlobParams params;
    params.max_radius = 32;
    const std::vector<MarkerCandidate> candidates = detect_blobs(image, params);
    std::printf("[marker noise] candidates=%d threshold=%.3f\n",
                static_cast<int>(candidates.size()), static_cast<double>(params.threshold));
    EXPECT_LE(candidates.size(), static_cast<std::size_t>(1));
}

TEST(MarkerDetector, RejectsSaddles) {
    // A checkerboard has a saddle at every interior cell corner: the Hessian
    // determinant is negative there, even where the trace takes the sign the
    // polarity filter wants. The old response multiplication turned those
    // points positive, so this sheet used to yield spurious blobs.
    const int size = 256;
    const int cell = 64;
    FloatImage image(size, size, 0.5F);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const bool dark = ((x / cell) + (y / cell)) % 2 == 0;
            image.at(x, y) = dark ? 0.1F : 0.9F;
        }
    }
    image = gaussian_blur(image, 1.0F);
    const std::vector<MarkerCandidate> candidates = detect_blobs(image);
    std::printf("[marker saddles] candidates=%d\n", static_cast<int>(candidates.size()));
    EXPECT_TRUE(candidates.empty());
}

TEST(MarkerDetector, BorderBlobRejectedOrAccurate) {
    // A target clipped by the image border must not be reported with a centre
    // biased by the clamping of the edge search. Policy: the 60% in-bounds ray
    // requirement drops it; if one is reported anyway, it must still be exact.
    const int size = 256;
    const std::vector<test::EllipseTruth> truth = {{2.0, 128.0, 16.0, 16.0, 0.0}};
    FloatImage image = test::render_ellipses(size, size, truth, 0.95F, 0.05F);
    image = gaussian_blur(image, 1.0F);
    const std::vector<MarkerCandidate> candidates = detect_blobs(image);
    const Eigen::Vector2d expected(truth.front().cx, truth.front().cy);
    std::printf("[marker border] candidates=%d\n", static_cast<int>(candidates.size()));
    EXPECT_LE(candidates.size(), static_cast<std::size_t>(1));
    for (const MarkerCandidate& candidate : candidates) {
        EXPECT_LE((candidate.center - expected).norm(), 0.1);
    }
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
    std::printf("[marker quad-corners] matched=%d missed=%d corners=%d RMS=%.6f px\n",
                static_cast<int>(truths.size()) - missed, missed, corner_count, rms);
    EXPECT_EQ(missed, 0);
    EXPECT_LE(rms, 0.1);
}

TEST(MarkerDetector, ParallelBlurMatchesSerialBits) {
    // The DoG detector consumes the blur outputs, so the row-parallel passes
    // must reproduce the serial arithmetic exactly, not merely closely.
    const int width = 97;
    const int height = 83;
    FloatImage image(width, height, 0.0F);
    std::mt19937 engine(20261106U);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double value = rng::uniform_real(engine, 0.0, 1.0);
            image.at(x, y) = static_cast<float>(value);
        }
    }
    const FloatImage parallel_result = gaussian_blur(image, 2.7F);
    const FloatImage serial_result = serial_gaussian_blur(image, 2.7F);
    ASSERT_EQ(parallel_result.width(), serial_result.width());
    ASSERT_EQ(parallel_result.height(), serial_result.height());
    ASSERT_EQ(parallel_result.pixels().size(), serial_result.pixels().size());
    for (std::size_t i = 0; i < serial_result.pixels().size(); ++i) {
        EXPECT_EQ(parallel_result.pixels()[i], serial_result.pixels()[i]) << "index " << i;
    }
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
    DetectionTimings timings;
    const auto start = std::chrono::steady_clock::now();
    const std::vector<MarkerCandidate> candidates = detect_blobs(image, params, &timings);
    const auto finish = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(finish - start).count();

    DetectionTimings quad_timings;
    const std::vector<MarkerCandidate> quads = detect_quads(image, QuadParams{}, &quad_timings);

    // Greedy one-to-one matching, shortest pair first: a candidate may stand
    // for at most one truth and every unassigned candidate is a false positive.
    struct Pair {
        double distance = 0.0;
        int truth = 0;
        int candidate = 0;
    };
    std::vector<Pair> pairs;
    for (int t = 0; t < static_cast<int>(truths.size()); ++t) {
        for (int c = 0; c < static_cast<int>(candidates.size()); ++c) {
            const Eigen::Vector2d expected(truths[static_cast<std::size_t>(t)].cx,
                                           truths[static_cast<std::size_t>(t)].cy);
            const double distance =
                (candidates[static_cast<std::size_t>(c)].center - expected).norm();
            if (distance < 3.0) pairs.push_back({distance, t, c});
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
        if (a.distance != b.distance) return a.distance < b.distance;
        if (a.truth != b.truth) return a.truth < b.truth;
        return a.candidate < b.candidate;
    });
    std::vector<char> truth_used(truths.size(), 0);
    std::vector<char> candidate_used(candidates.size(), 0);
    int matched = 0;
    for (const Pair& pair : pairs) {
        if (truth_used[static_cast<std::size_t>(pair.truth)] != 0 ||
            candidate_used[static_cast<std::size_t>(pair.candidate)] != 0) {
            continue;
        }
        truth_used[static_cast<std::size_t>(pair.truth)] = 1;
        candidate_used[static_cast<std::size_t>(pair.candidate)] = 1;
        ++matched;
    }
    int false_positives = 0;
    for (char used : candidate_used) {
        if (used == 0) ++false_positives;
    }
    std::printf(
        "[marker 12mp] seconds=%.3f blur=%.3f hessian=%.3f nms=%.3f rays=%.3f quads=%.3f "
        "candidates=%d matched=%d false_positives=%d quad_candidates=%d\n",
        seconds, timings.blur_seconds, timings.hessian_seconds, timings.nms_seconds,
        timings.refine_seconds, quad_timings.quad_seconds, static_cast<int>(candidates.size()),
        matched, false_positives, static_cast<int>(quads.size()));
    EXPECT_GE(matched, 95);
    EXPECT_LE(false_positives, 2);
    if (test::perf_asserts_enabled()) {
        EXPECT_LE(seconds, 1.5);
    }
}

}  // namespace
}  // namespace alphashape
