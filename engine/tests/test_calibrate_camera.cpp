// WP-13.15a: chessboard camera calibration solver tests.
//
// Synthetic corner observations are produced by projecting a 9 x 6 board
// through a known Calibration from random poses, then perturbed by 0.1 px of
// Gaussian noise. A deterministic portable RNG supplies both poses and noise.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"
#include "common/algorithm/calibration/calibrate.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/tasks/registry.hpp"
#include "image/io/image_io.hpp"
#include "make_chessboard.hpp"
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

/// One view's observations. A partial view keeps a subset of corners and an
/// arbitrary right-handed lattice origin / 90-degree rotation, exactly what
/// BoardView reports for an incomplete board. Poses come from `pose_engine`
/// and per-corner noise from `noise_engine`, so a caller can hold the poses
/// fixed while varying the noise draw.
CalibrationView make_view_impl(std::mt19937& pose_engine,
                               std::mt19937& noise_engine, int camera_id,
                               bool partial, const Calibration& truth) {
    const double ax = rng::uniform_real(pose_engine, -0.90, 0.90);
    const double ay = rng::uniform_real(pose_engine, -0.70, 0.70);
    const double az = rng::uniform_real(pose_engine, -0.80, 0.80);
    const double depth = rng::uniform_real(pose_engine, 0.30, 0.42);
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
    translation.x() += rng::uniform_real(pose_engine, -0.07, 0.07);
    translation.y() += rng::uniform_real(pose_engine, -0.045, 0.045);
    const int quarter_turns =
        partial ? static_cast<int>(rng::uniform_below(pose_engine, 3U)) : 0;
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
            pixel.x() += 0.1 * rng::normal(noise_engine);
            pixel.y() += 0.1 * rng::normal(noise_engine);
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

/// Single-engine wrapper: poses and noise share one stream, which keeps the
/// draw order of the original helper unchanged.
CalibrationView make_view(std::mt19937& engine, int camera_id, bool partial) {
    return make_view_impl(engine, engine, camera_id, partial,
                          truth_calibration());
}

/// A full, noise-free board view seen from the given board-to-camera pose.
CalibrationView make_exact_view(int camera_id, const Calibration& truth,
                                const Eigen::Matrix3d& rotation,
                                const Eigen::Vector3d& translation) {
    CalibrationView view;
    view.camera_id = camera_id;
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            view.board_points.emplace_back(static_cast<double>(col) * kSquare,
                                           static_cast<double>(row) * kSquare);
            const Eigen::Vector3d world(static_cast<double>(col) * kSquare,
                                        static_cast<double>(row) * kSquare, 0.0);
            view.pixels.push_back(truth.project(rotation * world + translation));
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
    std::printf("[calibrate recovers] f %.4f cx %.4f cy %.4f "
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
    std::printf("[calibrate focal] %.4f vs %.4f (%.5f relative)\n",
                focal_a, focal_b, relative);
    EXPECT_LE(relative, 0.005);
}

TEST(Calibrate, PartialViewsContribute) {
    const CalibrationResult result = solve(synthetic_views(31U, 9, 3));
    std::printf("[calibrate partial views] f %.4f cx %.4f cy %.4f "
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
    std::printf("[calibrate prunes] k3 %.6f pruned %s\n",
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

TEST(Calibrate, DefaultPruningKeepsSignificantTerms) {
    // Truth k2 is 0.05 and the default fit frees it; one standard error is far
    // below the estimate, so the default threshold must keep it.
    const CalibrationResult result = solve(synthetic_views(41U, 12));
    std::printf("[calibrate default pruning] k2 %.6f sigma %.6f pruned %zu\n",
                result.calibration.k2, result.sigma[6], result.pruned.size());
    EXPECT_FALSE(contains(result.pruned, "k2"));
    EXPECT_LT(std::abs(result.calibration.k2 - 0.05), 0.025);
}

TEST(Calibrate, HomographyFromFourPoints) {
    // Four non-degenerate correspondences give the eight-equation minimum; the
    // padded design must still return the exact homography on those points.
    const std::array<Eigen::Vector2d, 4> board = {
        Eigen::Vector2d(0.02, 0.01), Eigen::Vector2d(0.18, 0.02),
        Eigen::Vector2d(0.20, 0.15), Eigen::Vector2d(0.01, 0.14)};
    Eigen::Matrix3d truth;
    truth << 900.0, 30.0, 640.0,
             -12.0, 870.0, 480.0,
             0.02, -0.01, 1.0;
    const std::vector<Eigen::Vector2d> board_points(board.begin(), board.end());
    std::vector<Eigen::Vector2d> pixels;
    pixels.reserve(board.size());
    for (const Eigen::Vector2d& point : board) {
        const Eigen::Vector3d q =
            truth * Eigen::Vector3d(point.x(), point.y(), 1.0);
        pixels.emplace_back(q.x() / q.z(), q.y() / q.z());
    }
    Eigen::Matrix3d recovered;
    ASSERT_TRUE(homography_dlt(board_points, pixels, recovered));
    // A homography is defined up to scale, and the DLT returns unit Frobenius
    // norm. Dividing by the (2,2) entry is the identity map but keeps the
    // homogeneous third coordinate O(1), so the residual measures the DLT's
    // accuracy instead of amplifying its gauge by the scale factor.
    ASSERT_GT(std::abs(recovered(2, 2)), 0.0);
    recovered *= truth(2, 2) / recovered(2, 2);
    for (std::size_t i = 0; i < board.size(); ++i) {
        const Eigen::Vector3d q =
            recovered * Eigen::Vector3d(board[i].x(), board[i].y(), 1.0);
        ASSERT_GT(std::abs(q.z()), 1e-9);
        const Eigen::Vector2d mapped(q.x() / q.z(), q.y() / q.z());
        EXPECT_LE((mapped - pixels[i]).norm(), 1e-9) << i;
    }
}

TEST(Calibrate, SigmaMatchesEmpiricalSpread) {
    // Well-specified model: the truth carries exactly the terms the fit frees,
    // so the estimator is unbiased and the empirical spread checks the
    // reported covariance rather than model error.
    Calibration truth =
        Calibration::initial(CameraType::Frame, kWidth, kHeight, 1200.0);
    truth.cx = 6.0;
    truth.cy = -4.0;
    truth.k1 = -0.10;
    truth.k2 = 0.04;
    truth.p1 = 4e-4;
    truth.p2 = -2e-4;
    CalibrationFitFlags fit{};
    fit[0] = true;   // f
    fit[1] = true;   // cx
    fit[2] = true;   // cy
    fit[5] = true;   // k1
    fit[6] = true;   // k2
    fit[9] = true;   // p1
    fit[10] = true;  // p2
    CalibrateOptions options;
    options.prune_t_ratio = 0.0;  // fit the same terms in every draw

    constexpr int kDraws = 20;
    constexpr int kViewCount = 6;
    std::array<double, static_cast<std::size_t>(kDraws)> focal{};
    double sigma_sum = 0.0;
    const Calibration initial =
        Calibration::initial(CameraType::Frame, kWidth, kHeight, 1000.0);
    for (int draw = 0; draw < kDraws; ++draw) {
        // Identical poses every draw; only the noise seeds change.
        std::mt19937 pose_engine(900U);
        std::mt19937 noise_engine(1000U + static_cast<std::uint32_t>(draw));
        std::vector<CalibrationView> views;
        views.reserve(static_cast<std::size_t>(kViewCount));
        for (int i = 0; i < kViewCount; ++i) {
            views.push_back(
                make_view_impl(pose_engine, noise_engine, 100 + i, false, truth));
        }
        const CalibrationResult result =
            calibrate_camera(kWidth, kHeight, views, initial, fit, options);
        focal[static_cast<std::size_t>(draw)] = result.calibration.f;
        sigma_sum += result.sigma[0];
    }
    double mean = 0.0;
    for (const double value : focal) {
        mean += value;
    }
    mean /= static_cast<double>(kDraws);
    double spread = 0.0;
    for (const double value : focal) {
        spread += (value - mean) * (value - mean);
    }
    const double empirical =
        std::sqrt(spread / static_cast<double>(kDraws - 1));
    const double reported = sigma_sum / static_cast<double>(kDraws);
    ASSERT_GT(reported, 0.0);
    const double ratio = empirical / reported;
    std::printf("[calibrate sigma] empirical %.5f reported %.5f ratio %.3f\n",
                empirical, reported, ratio);
    EXPECT_GE(ratio, 0.6);
    EXPECT_LE(ratio, 1.6);
}

TEST(Calibrate, NearFrontoParallelFailsCleanly) {
    // Distortion-free on purpose: the closed form works from a homography, so
    // a distorted board breaks the homography model by ~1% and would inject
    // signal that this test is trying to rule out.
    const Calibration truth =
        Calibration::initial(CameraType::Frame, kWidth, kHeight, 1200.0);
    const Eigen::Vector3d centre(
        0.5 * static_cast<double>(kCols) * kSquare,
        0.5 * static_cast<double>(kRows) * kSquare, 0.0);
    std::vector<CalibrationView> views;
    for (int i = 0; i < 6; ++i) {
        // Every board stays well within half a degree of the image plane, where
        // the two Zhang constraints carry no usable signal.
        const double ax = 0.0012 + 0.0004 * static_cast<double>(i);
        const double ay = 0.0009 + 0.0002 * static_cast<double>(i);
        const Eigen::Matrix3d rotation =
            (Eigen::AngleAxisd(ay, Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(ax, Eigen::Vector3d::UnitX()))
                .toRotationMatrix();
        const Eigen::Vector3d translation =
            -rotation * centre +
            Eigen::Vector3d(0.004 * static_cast<double>(i),
                            0.003 * static_cast<double>(i), 0.35);
        views.push_back(make_exact_view(i, truth, rotation, translation));
    }
    std::string message;
    try {
        solve(views);
    } catch (const Error& error) {
        message = error.what();
    }
    std::printf("[calibrate fronto-parallel] %s\n", message.c_str());
    ASSERT_FALSE(message.empty()) << "a fronto-parallel set must be rejected";
    EXPECT_TRUE(message.find("fronto-parallel") != std::string::npos ||
                message.find("out of range") != std::string::npos)
        << message;
}

namespace {

constexpr int kTaskWidth = 640;
constexpr int kTaskHeight = 480;
constexpr double kTaskFocal = 600.0;

void write_gray_png(const std::string& path, const FloatImage& image) {
    Image out(image.width(), image.height(), 1);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const double value = static_cast<double>(image.at(x, y)) * 255.0;
            out.at(x, y, 0) = static_cast<std::uint8_t>(
                std::lround(std::clamp(value, 0.0, 255.0)));
        }
    }
    write_png(path, out);
}

/// A document with one sensor and `count` enabled cameras, each with a rendered
/// chessboard on its frame path.
struct CalibrateScene {
    Document document;
    Chunk* chunk = nullptr;
    std::vector<std::string> paths;

    CalibrateScene() = default;
    CalibrateScene(CalibrateScene&&) = default;
    CalibrateScene& operator=(CalibrateScene&&) = default;
    CalibrateScene(const CalibrateScene&) = delete;
    CalibrateScene& operator=(const CalibrateScene&) = delete;
    ~CalibrateScene() {
        for (const std::string& path : paths) {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    }
};

CalibrateScene make_calibrate_scene(std::uint32_t seed, int count) {
    CalibrateScene scene;
    Chunk& chunk = scene.document.workspace().add_chunk("Chunk 1");
    scene.document.set_active_chunk_id(chunk.id());
    scene.chunk = &chunk;
    Sensor sensor;
    sensor.label = "sensor";
    sensor.type = CameraType::Frame;
    sensor.width = kTaskWidth;
    sensor.height = kTaskHeight;
    sensor.calibration =
        Calibration::initial(CameraType::Frame, kTaskWidth, kTaskHeight, 500.0);
    chunk.sensors().push_back(sensor);
    std::mt19937 engine(seed);
    for (int i = 0; i < count; ++i) {
        Camera camera;
        camera.id = chunk.allocate_id();
        camera.sensor_id = 0;
        camera.label = "cam" + std::to_string(i);
        camera.enabled = true;
        chunk.cameras().push_back(camera);
        const std::string path =
            platform::unique_temp_path("alphashape_calib_", ".png");
        auto frame = std::make_shared<Frame>(path, kTaskWidth, kTaskHeight);
        frame->set_id(chunk.allocate_id());
        frame->set_camera_id(camera.id);
        chunk.frames().push_back(frame);
        test::ChessboardPose pose;
        pose.width = kTaskWidth;
        pose.height = kTaskHeight;
        pose.cols = kCols;
        pose.rows = kRows;
        pose.square_m = kSquare;
        pose.f = kTaskFocal;
        pose.k1 = 0.0;
        const double ax = rng::uniform_real(engine, -0.22, 0.22);
        const double ay = rng::uniform_real(engine, -0.18, 0.18);
        const double az = rng::uniform_real(engine, -0.25, 0.25);
        pose.rotation =
            (Eigen::AngleAxisd(az, Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(ay, Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(ax, Eigen::Vector3d::UnitX()))
                .toRotationMatrix();
        const Eigen::Vector3d centre(
            0.5 * static_cast<double>(kCols) * kSquare,
            0.5 * static_cast<double>(kRows) * kSquare, 0.0);
        pose.translation =
            -pose.rotation * centre + Eigen::Vector3d(0.0, 0.0, 0.45);
        pose.translation.x() += rng::uniform_real(engine, -0.02, 0.02);
        pose.translation.y() += rng::uniform_real(engine, -0.015, 0.015);
        write_gray_png(path, test::render_chessboard(pose).image);
        scene.paths.push_back(path);
    }
    return scene;
}

/// True when the comma-separated report list names this coefficient.
bool pruned_contains(const std::string& list, const std::string& name) {
    std::size_t start = 0;
    while (start <= list.size()) {
        const std::size_t comma = list.find(',', start);
        const std::string token = list.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        if (token == name) {
            return true;
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return false;
}

/// Add one camera with a rendered chessboard on its frame path.
void add_board_camera(CalibrateScene& scene, std::mt19937& engine,
                      int sensor_index, const std::string& label, int width,
                      int height) {
    Chunk& chunk = *scene.chunk;
    Camera camera;
    camera.id = chunk.allocate_id();
    camera.sensor_id = sensor_index;
    camera.label = label;
    camera.enabled = true;
    chunk.cameras().push_back(camera);
    const std::string path =
        platform::unique_temp_path("alphashape_calib_", ".png");
    auto frame = std::make_shared<Frame>(path, width, height);
    frame->set_id(chunk.allocate_id());
    frame->set_camera_id(camera.id);
    chunk.frames().push_back(frame);
    test::ChessboardPose pose;
    pose.width = width;
    pose.height = height;
    pose.cols = kCols;
    pose.rows = kRows;
    pose.square_m = kSquare;
    pose.f = kTaskFocal;
    pose.k1 = 0.0;
    const double ax = rng::uniform_real(engine, -0.22, 0.22);
    const double ay = rng::uniform_real(engine, -0.18, 0.18);
    const double az = rng::uniform_real(engine, -0.25, 0.25);
    pose.rotation =
        (Eigen::AngleAxisd(az, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(ay, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(ax, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    const Eigen::Vector3d centre(
        0.5 * static_cast<double>(kCols) * kSquare,
        0.5 * static_cast<double>(kRows) * kSquare, 0.0);
    pose.translation = -pose.rotation * centre + Eigen::Vector3d(0.0, 0.0, 0.45);
    pose.translation.x() += rng::uniform_real(engine, -0.02, 0.02);
    pose.translation.y() += rng::uniform_real(engine, -0.015, 0.015);
    write_gray_png(path, test::render_chessboard(pose).image);
    scene.paths.push_back(path);
}

/// Two sensors: "good" has `good_count` consistently sized frames, "mixed" has
/// two frames that disagree on size so that sensor alone must be skipped.
CalibrateScene make_mixed_size_scene(std::uint32_t seed, int good_count) {
    CalibrateScene scene;
    Chunk& chunk = scene.document.workspace().add_chunk("Chunk 1");
    scene.document.set_active_chunk_id(chunk.id());
    scene.chunk = &chunk;
    for (int s = 0; s < 2; ++s) {
        Sensor sensor;
        sensor.label = s == 0 ? "good" : "mixed";
        sensor.type = CameraType::Frame;
        sensor.width = kTaskWidth;
        sensor.height = kTaskHeight;
        sensor.calibration =
            Calibration::initial(CameraType::Frame, kTaskWidth, kTaskHeight, 500.0);
        chunk.sensors().push_back(sensor);
    }
    std::mt19937 engine(seed);
    for (int i = 0; i < good_count; ++i) {
        add_board_camera(scene, engine, 0, "good" + std::to_string(i),
                         kTaskWidth, kTaskHeight);
    }
    add_board_camera(scene, engine, 1, "mixed0", kTaskWidth, kTaskHeight);
    add_board_camera(scene, engine, 1, "mixed1", kTaskWidth / 2, kTaskHeight / 2);
    return scene;
}

}  // namespace

TEST(CalibrateCamera, ApplyWritesSensor) {
    CalibrateScene scene = make_calibrate_scene(81U, 5);
    const Calibration before = scene.chunk->sensors()[0].calibration;
    Params params = make_params(builtin_registry().require("CalibrateCamera"));
    run_task(builtin_registry(), scene.document, "CalibrateCamera", params);
    const Calibration& after = scene.chunk->sensors()[0].calibration;
    std::printf("[calibrate camera apply] f %.3f (truth %.1f)\n",
                after.f, kTaskFocal);
    EXPECT_NEAR(after.f, kTaskFocal, 0.01 * kTaskFocal);
    EXPECT_NE(before.f, after.f);
    const std::string* rms =
        scene.chunk->meta().find("calibrate_camera.sensor.rms_px");
    ASSERT_NE(rms, nullptr);
    EXPECT_GT(std::stod(*rms), 0.0);

    // A pruned term is not reported with a sigma: it was zeroed and held for
    // the final solve, so publishing 0 would read as perfect confidence.
    const std::string* pruned =
        scene.chunk->meta().find("calibrate_camera.sensor.pruned");
    ASSERT_NE(pruned, nullptr);
    const std::string pruned_list = *pruned;
    for (int k = 0; k < kCalibrationParamCount; ++k) {
        const std::string name =
            kCalibrationParamNames[static_cast<std::size_t>(k)];
        if (!pruned_contains(pruned_list, name)) {
            continue;
        }
        EXPECT_EQ(
            scene.chunk->meta().find("calibrate_camera.sensor.sigma." + name),
            nullptr)
            << name;
    }

    CalibrateScene retained = make_calibrate_scene(82U, 5);
    const Calibration original = retained.chunk->sensors()[0].calibration;
    Params dry = make_params(builtin_registry().require("CalibrateCamera"));
    dry.set("apply", false);
    run_task(builtin_registry(), retained.document, "CalibrateCamera", dry);
    EXPECT_TRUE(retained.chunk->sensors()[0].calibration == original);
    const std::string* dry_rms =
        retained.chunk->meta().find("calibrate_camera.sensor.rms_px");
    ASSERT_NE(dry_rms, nullptr);
    EXPECT_GT(std::stod(*dry_rms), 0.0);
}

TEST(CalibrateCamera, MixedSizesSkipOnlyThatSensor) {
    CalibrateScene scene = make_mixed_size_scene(91U, 5);
    Params params = make_params(builtin_registry().require("CalibrateCamera"));
    run_task(builtin_registry(), scene.document, "CalibrateCamera", params);
    const Calibration& good = scene.chunk->sensors()[0].calibration;
    const std::string* skipped =
        scene.chunk->meta().find("calibrate_camera.mixed.skipped");
    std::printf("[calibrate camera mixed sizes] good f %.3f skipped \"%s\"\n",
                good.f, skipped != nullptr ? skipped->c_str() : "(none)");
    EXPECT_NEAR(good.f, kTaskFocal, 0.01 * kTaskFocal);
    EXPECT_EQ(scene.chunk->meta().find("calibrate_camera.good.skipped"), nullptr);
    ASSERT_NE(skipped, nullptr);
    EXPECT_TRUE(skipped->find("differing image sizes") != std::string::npos)
        << *skipped;
}

TEST(CalibrateCamera, WritesReport) {
    CalibrateScene scene = make_calibrate_scene(101U, 5);
    const std::string path =
        platform::unique_temp_path("alphashape_calreport_", ".html");
    Params params = make_params(builtin_registry().require("CalibrateCamera"));
    params.set("report_path", ParamValue{path});
    run_task(builtin_registry(), scene.document, "CalibrateCamera", params);
    std::ifstream file(path, std::ios::binary);
    ASSERT_TRUE(file.good()) << path;
    const std::string report((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    file.close();
    std::error_code error;
    std::filesystem::remove(path, error);
    std::printf("[calibrate camera report] %zu bytes\n", report.size());
    EXPECT_NE(report.find("RMS"), std::string::npos);
    EXPECT_NE(report.find("sensor"), std::string::npos);
}

TEST(CalibrateCamera, MissingReportDirectoryFailsBeforeSolve) {
    CalibrateScene scene = make_calibrate_scene(111U, 5);
    const Calibration before = scene.chunk->sensors()[0].calibration;
    // unique_temp_path merely names a path; using one as a directory gives a
    // parent that certainly does not exist.
    const std::filesystem::path missing_dir =
        platform::unique_temp_path("alphashape_calrep_missing_", "");
    const std::string path = (missing_dir / "report.html").string();
    Params params = make_params(builtin_registry().require("CalibrateCamera"));
    params.set("report_path", ParamValue{path});
    try {
        run_task(builtin_registry(), scene.document, "CalibrateCamera", params);
        FAIL() << "a missing report directory must be rejected";
    } catch (const ValueError& error) {
        const std::string message(error.what());
        EXPECT_NE(message.find("report directory does not exist"),
                  std::string::npos)
            << message;
        EXPECT_NE(message.find(missing_dir.string()), std::string::npos)
            << message;
    }
    // The check runs before the solve, so nothing was written to the document.
    EXPECT_TRUE(scene.chunk->sensors()[0].calibration == before);
    EXPECT_EQ(scene.chunk->meta().find("calibrate_camera.sensor.rms_px"),
              nullptr);
}

TEST(CalibrateCamera, ReportListsSkippedSensors) {
    CalibrateScene scene;
    Chunk& chunk = scene.document.workspace().add_chunk("Chunk 1");
    scene.document.set_active_chunk_id(chunk.id());
    scene.chunk = &chunk;
    Sensor sensor;
    sensor.label = "blank";
    sensor.type = CameraType::Frame;
    sensor.width = 64;
    sensor.height = 48;
    sensor.calibration = Calibration::initial(CameraType::Frame, 64, 48, 50.0);
    chunk.sensors().push_back(sensor);
    Camera camera;
    camera.id = chunk.allocate_id();
    camera.sensor_id = 0;
    camera.label = "blank0";
    camera.enabled = true;
    chunk.cameras().push_back(camera);
    // A uniform white image carries no chessboard, so the sole sensor is
    // skipped rather than calibrated.
    const std::string image_path =
        platform::unique_temp_path("alphashape_calib_blank_", ".png");
    Image blank(64, 48, 1);
    for (int y = 0; y < blank.height(); ++y) {
        for (int x = 0; x < blank.width(); ++x) {
            blank.at(x, y, 0) = static_cast<std::uint8_t>(255);
        }
    }
    write_png(image_path, blank);
    scene.paths.push_back(image_path);
    auto frame = std::make_shared<Frame>(image_path, 64, 48);
    frame->set_id(chunk.allocate_id());
    frame->set_camera_id(camera.id);
    chunk.frames().push_back(frame);

    const std::string report_path =
        platform::unique_temp_path("alphashape_calreport_skip_", ".html");
    Params params = make_params(builtin_registry().require("CalibrateCamera"));
    params.set("report_path", ParamValue{report_path});
    try {
        run_task(builtin_registry(), scene.document, "CalibrateCamera", params);
        FAIL() << "a sensor with no detected board must not calibrate";
    } catch (const Error& error) {
        EXPECT_NE(std::string(error.what()).find("at least 3 views"),
                  std::string::npos)
            << error.what();
    }
    std::ifstream file(report_path, std::ios::binary);
    ASSERT_TRUE(file.good()) << report_path;
    const std::string report((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    file.close();
    std::error_code remove_error;
    std::filesystem::remove(report_path, remove_error);
    EXPECT_NE(report.find("Skipped"), std::string::npos);
    EXPECT_NE(report.find("at least 3 views"), std::string::npos);
}

TEST(CalibrateCamera, RejectsUnknownReportFormat) {
    CalibrateScene scene = make_calibrate_scene(102U, 5);
    Params params = make_params(builtin_registry().require("CalibrateCamera"));
    params.set("report_format", ParamValue{std::string{"json"}});
    try {
        run_task(builtin_registry(), scene.document, "CalibrateCamera", params);
        FAIL() << "an unknown report format must be rejected";
    } catch (const ValueError& error) {
        EXPECT_NE(std::string(error.what()).find("json"), std::string::npos)
            << error.what();
    }
}

}  // namespace alphashape
