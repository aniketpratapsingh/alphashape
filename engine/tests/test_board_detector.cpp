// Tests for WP-13.14 chessboard detection: complete synthetic boards across
// tilts, canonical corner ordering, partial views, clutter rejection, blur and
// noise robustness, option validation and a single-frame timing case.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"
#include "common/algorithm/calibration/board_detector.hpp"
#include "image/image.hpp"
#include "make_chessboard.hpp"
#include "perf_gate.hpp"

namespace alphashape {
namespace {

constexpr double kPi = std::numbers::pi_v<double>;

/// Pose the board centre at (0, 0, distance) with an out-of-plane tilt about
/// an in-plane axis and an in-plane spin, both in degrees.
test::ChessboardPose posed_board(double tilt_deg, double axis_deg, double spin_deg,
                                 double distance, double k1 = -0.08) {
    test::ChessboardPose pose;
    pose.k1 = k1;
    const double axis_rad = axis_deg * kPi / 180.0;
    const Eigen::Vector3d axis(std::cos(axis_rad), std::sin(axis_rad), 0.0);
    const Eigen::Matrix3d tilt =
        Eigen::AngleAxisd(tilt_deg * kPi / 180.0, axis).toRotationMatrix();
    const Eigen::Matrix3d spin =
        Eigen::AngleAxisd(spin_deg * kPi / 180.0, Eigen::Vector3d::UnitZ())
            .toRotationMatrix();
    pose.rotation = spin * tilt;
    const double sq = pose.square_m;
    const Eigen::Vector3d centre(0.5 * static_cast<double>(pose.cols - 1) * sq,
                                 0.5 * static_cast<double>(pose.rows - 1) * sq, 0.0);
    pose.translation = -pose.rotation * centre + Eigen::Vector3d(0.0, 0.0, distance);
    return pose;
}

/// Separable Gaussian blur with clamped edges, mirroring the detector's own.
FloatImage blur_image(const FloatImage& in, double sigma) {
    const int w = in.width();
    const int h = in.height();
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    std::vector<double> kernel(static_cast<std::size_t>(2 * radius + 1), 0.0);
    double sum = 0.0;
    for (int k = -radius; k <= radius; ++k) {
        const double v = std::exp(-0.5 * static_cast<double>(k * k) / (sigma * sigma));
        kernel[static_cast<std::size_t>(k + radius)] = v;
        sum += v;
    }
    for (double& v : kernel) {
        v /= sum;
    }
    FloatImage tmp(w, h, 0.0F);
    FloatImage out(w, h, 0.0F);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double acc = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                acc += kernel[static_cast<std::size_t>(k + radius)] *
                       static_cast<double>(in.at_clamped(x + k, y));
            }
            tmp.at(x, y) = static_cast<float>(acc);
        }
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double acc = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                acc += kernel[static_cast<std::size_t>(k + radius)] *
                       static_cast<double>(tmp.at_clamped(x, y + k));
            }
            out.at(x, y) = static_cast<float>(acc);
        }
    }
    return out;
}

TEST(Board, DetectsChessboard9x6Synthetic) {
    const BoardSpec spec;
    const BoardDetectOptions options;
    std::mt19937 engine(20261031u);
    double sum_squared = 0.0;
    double max_error = 0.0;
    int compared = 0;
    for (int view = 0; view < 30; ++view) {
        const double tilt = rng::uniform_real(engine, 0.0, 45.0);
        const double axis = rng::uniform_real(engine, 0.0, 360.0);
        const double spin = rng::uniform_real(engine, -20.0, 20.0);
        const double distance = rng::uniform_real(engine, 0.55, 0.65);
        const test::ChessboardPose pose = posed_board(tilt, axis, spin, distance);
        const test::ChessboardRender rendered = test::render_chessboard(pose);
        const std::optional<BoardView> result = detect_board(rendered.image, spec, options);
        ASSERT_TRUE(result.has_value()) << "view " << view;
        EXPECT_TRUE(result->complete) << "view " << view;
        ASSERT_EQ(result->corners.size(),
                  static_cast<std::size_t>(spec.cols * spec.rows))
            << "view " << view;
        for (int row = 0; row < spec.rows; ++row) {
            for (int col = 0; col < spec.cols; ++col) {
                const Eigen::Vector2d truth = rendered.corner(col, row);
                double best = std::numeric_limits<double>::infinity();
                for (const BoardCorner& corner : result->corners) {
                    best = std::min(best, (corner.pixel - truth).norm());
                }
                sum_squared += best * best;
                max_error = std::max(max_error, best);
                ++compared;
                EXPECT_LE(best, 0.15) << "view " << view << " corner " << col << ","
                                      << row;
            }
        }
    }
    const double rms = std::sqrt(sum_squared / static_cast<double>(compared));
    std::printf("[board] 9x6 synthetic: RMS %.4f px, max %.4f px over %d corners\n", rms,
                max_error, compared);
    EXPECT_LE(rms, 0.05);
    EXPECT_LE(max_error, 0.15);
}

TEST(Board, CornerOrderingCorrect) {
    const BoardSpec spec;
    const BoardDetectOptions options;
    // A small in-plane spin keeps the origin rule checkable while avoiding
    // perfectly grid-aligned edges: there, a regular 4x4 box supersample has a
    // coherent phase bias (a renderer artefact, not a real camera condition).
    const test::ChessboardPose pose = posed_board(15.0, 0.0, 8.0, 0.6);
    const test::ChessboardRender rendered = test::render_chessboard(pose);
    const std::optional<BoardView> result = detect_board(rendered.image, spec, options);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(result->complete);
    ASSERT_EQ(result->corners.size(), static_cast<std::size_t>(spec.cols * spec.rows));
    for (const BoardCorner& corner : result->corners) {
        ASSERT_GE(corner.col, 0);
        ASSERT_LT(corner.col, spec.cols);
        ASSERT_GE(corner.row, 0);
        ASSERT_LT(corner.row, spec.rows);
        const Eigen::Vector2d truth = rendered.corner(corner.col, corner.row);
        EXPECT_LE((corner.pixel - truth).norm(), 0.1)
            << "label " << corner.col << "," << corner.row;
    }
    Eigen::Vector2d origin = Eigen::Vector2d::Zero();
    Eigen::Vector2d along_col = Eigen::Vector2d::Zero();
    Eigen::Vector2d along_row = Eigen::Vector2d::Zero();
    for (const BoardCorner& corner : result->corners) {
        if (corner.col == 0 && corner.row == 0) {
            origin = corner.pixel;
        } else if (corner.col == 1 && corner.row == 0) {
            along_col = corner.pixel - origin;
        } else if (corner.col == 0 && corner.row == 1) {
            along_row = corner.pixel - origin;
        }
    }
    EXPECT_GT(along_col.x() * along_row.y() - along_col.y() * along_row.x(), 0.0);
}

TEST(Board, LabelsCompleteBoardAtEveryRotation) {
    const BoardSpec spec;
    const BoardDetectOptions options;
    // In-plane spins covering grid-aligned, quarter-turn and oblique boards.
    // A +90 degree spin puts physical corner (0, rows - 1) nearest the image
    // top-left, and 180/270 put the other diagonal corners there, so this also
    // exercises the origin choice when the nearest-top-left corner is not
    // physical (0, 0).
    const double spins[6] = {0.0, 90.0, 180.0, 270.0, 45.0, -90.0};
    for (double spin : spins) {
        const test::ChessboardPose pose = posed_board(12.0, 30.0, spin, 0.62);
        const test::ChessboardRender rendered = test::render_chessboard(pose);
        ASSERT_TRUE(rendered.outside_diagonal_dark(0, 0))
            << "renderer convention, spin " << spin;
        const std::optional<BoardView> result =
            detect_board(rendered.image, spec, options);
        ASSERT_TRUE(result.has_value()) << "spin " << spin;
        ASSERT_TRUE(result->complete) << "spin " << spin;
        ASSERT_EQ(result->corners.size(),
                  static_cast<std::size_t>(spec.cols * spec.rows))
            << "spin " << spin;
        for (const BoardCorner& corner : result->corners) {
            ASSERT_GE(corner.col, 0) << "spin " << spin;
            ASSERT_LT(corner.col, spec.cols) << "spin " << spin;
            ASSERT_GE(corner.row, 0) << "spin " << spin;
            ASSERT_LT(corner.row, spec.rows) << "spin " << spin;
            const Eigen::Vector2d truth = rendered.corner(corner.col, corner.row);
            EXPECT_LE((corner.pixel - truth).norm(), 0.15)
                << "spin " << spin << " label " << corner.col << "," << corner.row;
        }
    }
}

TEST(Board, PartialViewAccepted) {
    const BoardSpec spec;
    const BoardDetectOptions options;
    // A small spin keeps lattice differences comparable and avoids the
    // coherent box-supersample phase of edges exactly parallel to the pixel
    // grid. At 3 degrees the crop boundary still falls in the gap between rows
    // two and three, so no visible corner sits next to the image border.
    const test::ChessboardPose pose = posed_board(20.0, 0.0, 3.0, 0.6);
    const test::ChessboardRender rendered = test::render_chessboard(pose);
    const int crop_height = rendered.image.height() / 2;
    FloatImage cropped(rendered.image.width(), crop_height, 0.0F);
    for (int y = 0; y < crop_height; ++y) {
        for (int x = 0; x < cropped.width(); ++x) {
            cropped.at(x, y) = rendered.image.at(x, y);
        }
    }
    const std::optional<BoardView> result = detect_board(cropped, spec, options);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->complete);
    ASSERT_GE(result->corners.size(), static_cast<std::size_t>(options.min_corners));
    std::vector<int> truth_col(result->corners.size(), -1);
    std::vector<int> truth_row(result->corners.size(), -1);
    for (std::size_t i = 0; i < result->corners.size(); ++i) {
        double best = std::numeric_limits<double>::infinity();
        for (int row = 0; row < spec.rows; ++row) {
            for (int col = 0; col < spec.cols; ++col) {
                const double distance =
                    (result->corners[i].pixel - rendered.corner(col, row)).norm();
                if (distance < best) {
                    best = distance;
                    truth_col[i] = col;
                    truth_row[i] = row;
                }
            }
        }
        EXPECT_LE(best, 0.1) << "corner " << i;
    }
    for (std::size_t i = 0; i < result->corners.size(); ++i) {
        for (std::size_t j = i + 1; j < result->corners.size(); ++j) {
            EXPECT_EQ(result->corners[i].col - result->corners[j].col,
                      truth_col[i] - truth_col[j]);
            EXPECT_EQ(result->corners[i].row - result->corners[j].row,
                      truth_row[i] - truth_row[j]);
        }
    }
}

TEST(Board, RejectsClutter) {
    const BoardSpec spec;
    const BoardDetectOptions options;
    const int width = 400;
    const int height = 300;
    const int grid_w = 24;
    const int grid_h = 18;
    for (int seed = 0; seed < 5; ++seed) {
        std::mt19937 engine(static_cast<unsigned>(9001 + seed));
        std::vector<double> grid(static_cast<std::size_t>(grid_w * grid_h), 0.0);
        for (double& value : grid) {
            value = rng::uniform_real(engine, 0.0, 1.0);
        }
        FloatImage image(width, height, 0.0F);
        for (int y = 0; y < height; ++y) {
            const double fy = (static_cast<double>(y) + 0.5) * grid_h / height;
            for (int x = 0; x < width; ++x) {
                const double fx = (static_cast<double>(x) + 0.5) * grid_w / width;
                const int x0 = std::min(grid_w - 1, static_cast<int>(std::floor(fx)));
                const int y0 = std::min(grid_h - 1, static_cast<int>(std::floor(fy)));
                const int x1 = std::min(grid_w - 1, x0 + 1);
                const int y1 = std::min(grid_h - 1, y0 + 1);
                const double tx = fx - std::floor(fx);
                const double ty = fy - std::floor(fy);
                const auto at = [&](int col, int row) {
                    return grid[static_cast<std::size_t>(row * grid_w + col)];
                };
                const double top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
                const double bottom = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
                image.at(x, y) = static_cast<float>(top * (1.0 - ty) + bottom * ty);
            }
        }
        const std::optional<BoardView> result = detect_board(image, spec, options);
        EXPECT_FALSE(result.has_value()) << "seed " << seed;
    }
}

TEST(Board, HandlesBlurAndNoise) {
    const BoardSpec spec;
    const BoardDetectOptions options;
    const test::ChessboardPose pose = posed_board(20.0, 30.0, 5.0, 0.6);
    const test::ChessboardRender rendered = test::render_chessboard(pose);
    FloatImage blurred = blur_image(rendered.image, 1.5);
    std::mt19937 engine(424242u);
    for (float& value : blurred.pixels()) {
        const double noise = rng::normal(engine, 0.0, 0.02);
        value = static_cast<float>(
            std::clamp(static_cast<double>(value) + noise, 0.0, 1.0));
    }
    const std::optional<BoardView> result = detect_board(blurred, spec, options);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(result->complete);
    ASSERT_EQ(result->corners.size(), static_cast<std::size_t>(spec.cols * spec.rows));
    double sum_squared = 0.0;
    double max_error = 0.0;
    int compared = 0;
    for (int row = 0; row < spec.rows; ++row) {
        for (int col = 0; col < spec.cols; ++col) {
            const Eigen::Vector2d truth = rendered.corner(col, row);
            double best = std::numeric_limits<double>::infinity();
            for (const BoardCorner& corner : result->corners) {
                best = std::min(best, (corner.pixel - truth).norm());
            }
            sum_squared += best * best;
            max_error = std::max(max_error, best);
            ++compared;
        }
    }
    const double rms = std::sqrt(sum_squared / static_cast<double>(compared));
    std::printf("[board] blur+noise: RMS %.4f px, max %.4f px\n", rms, max_error);
    EXPECT_LE(rms, 0.15);
}

TEST(Board, InvalidOptionsRejected) {
    const FloatImage image(64, 64, 0.5F);
    BoardSpec spec;
    BoardSpec small = spec;
    small.cols = 1;
    EXPECT_THROW(detect_board(image, small), ValueError);
    BoardSpec no_size = spec;
    no_size.square_size_m = 0.0;
    EXPECT_THROW(detect_board(image, no_size), ValueError);
    const auto expect_rejected = [&](const BoardDetectOptions& options) {
        EXPECT_THROW(detect_board(image, spec, options), ValueError);
    };
    BoardDetectOptions bad;
    bad.sigma_px = 0.0;
    expect_rejected(bad);
    bad = BoardDetectOptions{};
    bad.min_distance_px = -1.0;
    expect_rejected(bad);
    bad = BoardDetectOptions{};
    bad.min_distance_px = 0.5;
    expect_rejected(bad);
    bad = BoardDetectOptions{};
    bad.response_fraction = 0.0;
    expect_rejected(bad);
    bad = BoardDetectOptions{};
    bad.ring_radius_px = 0.0;
    expect_rejected(bad);
    bad = BoardDetectOptions{};
    bad.angle_tol_deg = 0.0;
    expect_rejected(bad);
    bad = BoardDetectOptions{};
    bad.spacing_tol = 0.0;
    expect_rejected(bad);
    bad = BoardDetectOptions{};
    bad.refine_radius = 0;
    expect_rejected(bad);
    bad = BoardDetectOptions{};
    bad.min_corners = 0;
    expect_rejected(bad);
}

TEST(Board, OneFrameUnderHalfSecond) {
    const test::ChessboardPose pose = posed_board(20.0, 15.0, 5.0, 0.6);
    const test::ChessboardRender rendered = test::render_chessboard(pose);
    const std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    const std::optional<BoardView> result =
        detect_board(rendered.image, BoardSpec{}, BoardDetectOptions{});
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    ASSERT_TRUE(result.has_value());
    const double seconds =
        std::chrono::duration<double>(end - start).count();
    std::printf("[board] one 1280x960 detection: %.4f s\n", seconds);
    if (test::perf_asserts_enabled()) {
        EXPECT_LT(seconds, 0.5);
    }
}


}  // namespace
}  // namespace alphashape
