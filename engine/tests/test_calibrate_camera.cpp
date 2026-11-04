// WP-13.15a: chessboard camera calibration solver tests.
//
// Synthetic corner observations are produced by projecting a 9 x 6 board
// through a known Calibration from random poses, then perturbed by 0.1 px of
// Gaussian noise. A deterministic portable RNG supplies both poses and noise.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"
#include "common/algorithm/calibration/calibrate.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 960;
constexpr int kCols = 9;
constexpr int kRows = 6;
constexpr double kSquare = 0.025;

Calibration truth_calibration() {
    Calibration calibration =
        Calibration::initial(CameraType::Frame, kWidth, kHeight, 1200.0);
    calibration.cx = 8.0;
    calibration.cy = -5.0;
    calibration.k1 = -0.12;
    calibration.k2 = 0.05;
    calibration.p1 = 5e-4;
    return calibration;
}

std::vector<Eigen::Vector2d> physical_board_points() {
    std::vector<Eigen::Vector2d> points;
    points.reserve(static_cast<std::size_t>(kCols * kRows));
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            points.emplace_back(static_cast<double>(col) * kSquare,
                                static_cast<double>(row) * kSquare);
        }
    }
    return points;
}

/// One view's observations. A partial view keeps a subset of corners and an
/// arbitrary right-handed lattice origin / 90-degree rotation, exactly what
/// BoardView reports for an incomplete board.
CalibrationView make_view(std::mt19937& engine, int camera_id, bool partial) {
    const Calibration truth = truth_calibration();
    const double ax = rng::uniform_real(engine, -0.45, 0.45);
    const double ay = rng::uniform_real(engine, -0.35, 0.35);
    const double az = rng::uniform_real(engine, -0.50, 0.50);
    const double depth = rng::uniform_real(engine, 0.45, 0.62);
    const Eigen::Matrix3d rotation =
        (Eigen::AngleAxisd(az, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(ay, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(ax, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    const Eigen::Vector3d centre(
        0.5 * static_cast<double>(kCols) * kSquare,
        0.5 * static_cast<double>(kRows) * kSquare, 0.0);
    Eigen::Vector3d translation =
        -rotation * centre + Eigen::Vector3d(0.0, 0.0, depth);
    translation.x() += rng::uniform_real(engine, -0.05, 0.05);
    translation.y() += rng::uniform_real(engine, -0.03, 0.03);
    const int quarter_turns = partial ? static_cast<int>(rng::uniform_below(engine, 3U))
                                      : 0;
    CalibrationView view;
    view.camera_id = camera_id;
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            if (partial && ((col * 3 + row) % 2) != 0) {
                continue;
            }
            const Eigen::Vector3d world(static_cast<double>(col) * kSquare,
                                        static_cast<double>(row) * kSquare, 0.0);
            const Eigen::Vector3d camera = rotation * world + translation;
            Eigen::Vector2d pixel = truth.project(camera);
            pixel.x() += 0.1 * rng::normal(engine);
            pixel.y() += 0.1 * rng::normal(engine);
            int label_col = col;
            int label_row = row;
            for (int turn = 0; turn < quarter_turns; ++turn) {
                const int old_col = label_col;
                label_col = -label_row;
                label_row = old_col;
            }
            if (partial) {
                label_col += 11;
                label_row += 7;
            }
            view.board_points.emplace_back(
                static_cast<double>(label_col) * kSquare,
                static_cast<double>(label_row) * kSquare);
            view.pixels.push_back(pixel);
        }
    }
    return view;
}

std::vector<CalibrationView> synthetic_views(std::uint32_t seed, int count,
                                             int partial_from = -1) {
    std::mt19937 engine(seed);
    std::vector<CalibrationView> views;
    views.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        views.push_back(make_view(engine, 100 + i,
                                  partial_from >= 0 && i >= partial_from));
    }
    return views;
}

CalibrationResult solve(const std::vector<CalibrationView>& views,
                        const CalibrationFitFlags& fit =
                            default_calibration_fit_flags(),
                        const CalibrateOptions& options = CalibrateOptions{}) {
    const Calibration initial =
        Calibration::initial(CameraType::Frame, kWidth, kHeight, 1000.0);
    return calibrate_camera(kWidth, kHeight, views, initial, fit, options);
}

bool contains(const std::vector<std::string>& values, const std::string& value) {
    for (const std::string& entry : values) {
        if (entry == value) {
            return true;
        }
    }
    return false;
}

}  // namespace

TEST(Calibrate, RecoversSyntheticDistortion) {
    const CalibrationResult result = solve(synthetic_views(13U, 12));
    std::printf("[Calibrate.RecoversSyntheticDistortion] f %.4f cx %.4f cy %.4f "
                "k1 %.6f rms %.4f max %.4f views %d\n",
                result.calibration.f, result.calibration.cx,
                result.calibration.cy, result.calibration.k1, result.rms_px,
                result.max_px, result.views_used);
    EXPECT_NEAR(result.calibration.f, 1200.0, 0.002 * 1200.0);
    EXPECT_NEAR(result.calibration.cx, 8.0, 1.0);
    EXPECT_NEAR(result.calibration.cy, -5.0, 1.0);
    EXPECT_NEAR(result.calibration.k1, -0.12, 0.03 * 0.12);
    EXPECT_LT(result.rms_px, 0.2);
}

TEST(Calibrate, FocalRepeatability) {
    const std::vector<CalibrationView> all = synthetic_views(21U, 12);
    const std::vector<CalibrationView> first(all.begin(), all.begin() + 6);
    const std::vector<CalibrationView> second(all.begin() + 6, all.end());
    const double focal_a = solve(first).calibration.f;
    const double focal_b = solve(second).calibration.f;
    const double relative =
        std::abs(focal_a - focal_b) / (0.5 * (focal_a + focal_b));
    std::printf("[Calibrate.FocalRepeatability] %.4f vs %.4f (%.5f relative)\n",
                focal_a, focal_b, relative);
    EXPECT_LE(relative, 0.005);
}

TEST(Calibrate, PartialViewsContribute) {
    const CalibrationResult result = solve(synthetic_views(31U, 9, 3));
    std::printf("[Calibrate.PartialViewsContribute] f %.4f cx %.4f cy %.4f "
                "k1 %.6f rms %.4f views %d\n",
                result.calibration.f, result.calibration.cx,
                result.calibration.cy, result.calibration.k1, result.rms_px,
                result.views_used);
    EXPECT_NEAR(result.calibration.f, 1200.0, 0.002 * 1200.0);
    EXPECT_NEAR(result.calibration.cx, 8.0, 1.0);
    EXPECT_NEAR(result.calibration.cy, -5.0, 1.0);
    EXPECT_NEAR(result.calibration.k1, -0.12, 0.03 * 0.12);
    EXPECT_LT(result.rms_px, 0.2);
}

TEST(Calibrate, PrunesInsignificantTerms) {
    CalibrateOptions options;
    // A generous threshold keeps the assertion robust against one noisy draw;
    // the default is 1.0. k1's z-score is far above this.
    options.prune_t_ratio = 3.0;
    const CalibrationResult result =
        solve(synthetic_views(41U, 12), default_calibration_fit_flags(), options);
    std::printf("[Calibrate.PrunesInsignificantTerms] k3 %.6f pruned %s\n",
                result.calibration.k3,
                result.pruned.empty() ? "(none)" : result.pruned.front().c_str());
    EXPECT_TRUE(contains(result.pruned, "k3"));
    EXPECT_EQ(result.calibration.k3, 0.0);
    EXPECT_FALSE(contains(result.pruned, "k1"));
}

TEST(Calibrate, CovarianceIsConsistent) {
    CalibrationFitFlags fit{};
    fit[0] = true;  // f
    fit[1] = true;  // cx
    fit[2] = true;  // cy
    fit[5] = true;  // k1
    const CalibrationResult result = solve(synthetic_views(61U, 10), fit);
    for (int k = 0; k < kCalibrationParamCount; ++k) {
        const std::size_t kk = static_cast<std::size_t>(k);
        if (fit[kk]) {
            EXPECT_GT(result.sigma[kk], 0.0) << kCalibrationParamNames[kk];
        } else {
            EXPECT_EQ(result.sigma[kk], 0.0) << kCalibrationParamNames[kk];
        }
    }
    ASSERT_EQ(result.correlation.rows(), kCalibrationParamCount);
    ASSERT_EQ(result.correlation.cols(), kCalibrationParamCount);
    for (int i = 0; i < kCalibrationParamCount; ++i) {
        EXPECT_NEAR(result.correlation(i, i), 1.0, 1e-12);
        for (int j = 0; j < kCalibrationParamCount; ++j) {
            EXPECT_NEAR(result.correlation(i, j), result.correlation(j, i), 1e-12);
            EXPECT_LE(std::abs(result.correlation(i, j)), 1.0 + 1e-12);
        }
    }
}

TEST(Calibrate, NeedsThreeViews) {
    const std::vector<CalibrationView> views = synthetic_views(51U, 2);
    try {
        solve(views);
        FAIL() << "two views must not calibrate";
    } catch (const Error& error) {
        EXPECT_STREQ(error.what(),
                     "calibrate_camera: need at least 3 views with a detected board");
    }
}

}  // namespace alphashape
