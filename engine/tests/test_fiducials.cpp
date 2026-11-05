// Tests for WP-13.18a: scanned-film fiducial detection, ordering, the
// per-scan film transform, the derived sensor layout, persistence and the
// undoable film command.

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/io/zip.hpp"
#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "common/algorithm/calibration/calibrate_fiducials_task.hpp"
#include "common/algorithm/calibration/fiducials.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/io/project_io.hpp"
#include "common/tasks/registry.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"
#include "make_film_frame.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

std::vector<Eigen::Vector2d> marks4(double d) {
    return {Eigen::Vector2d(-d, d), Eigen::Vector2d(d, d),
            Eigen::Vector2d(d, -d), Eigen::Vector2d(-d, -d)};
}

/// Four corner marks whose bounding box is centred on the film origin but whose
/// layout has no dihedral symmetry. A symmetric square would make a 90-degree
/// rotation fundamentally ambiguous, so the ordering tests use these.
std::vector<Eigen::Vector2d> centred_marks() {
    return {Eigen::Vector2d(-45.0, 42.0), Eigen::Vector2d(40.0, 47.0),
            Eigen::Vector2d(45.0, -44.0), Eigen::Vector2d(-41.0, -47.0)};
}

/// A smaller, bbox-centred asymmetric layout for the derivation test, where
/// the fixture runs at 50 px/mm so 0.02 mm/px is the matching resolution.
std::vector<Eigen::Vector2d> small_marks() {
    return {Eigen::Vector2d(-13.5, 11.0), Eigen::Vector2d(12.0, 13.0),
            Eigen::Vector2d(13.5, -12.0), Eigen::Vector2d(-11.0, -13.0)};
}

FilmSettings layout_for(const std::vector<Eigen::Vector2d>& marks) {
    FilmSettings settings;
    settings.enabled = true;
    settings.detector = FiducialDetector::Corner;
    settings.search_corners = true;
    settings.search_sides = false;
    for (std::size_t i = 0; i < marks.size(); ++i) {
        settings.fiducials.push_back(
            Fiducial{"m" + std::to_string(i), marks[i]});
    }
    return settings;
}

FilmSettings corner_layout(double d) {
    FilmSettings settings;
    settings.enabled = true;
    settings.detector = FiducialDetector::Corner;
    settings.search_corners = true;
    settings.search_sides = false;
    settings.fiducials = {Fiducial{"tl", Eigen::Vector2d(-d, d)},
                          Fiducial{"tr", Eigen::Vector2d(d, d)},
                          Fiducial{"br", Eigen::Vector2d(d, -d)},
                          Fiducial{"bl", Eigen::Vector2d(-d, -d)}};
    return settings;
}

FiducialDetectOptions options_for(const FilmSettings& settings) {
    FiducialDetectOptions options;
    options.detector = settings.detector;
    options.search_corners = settings.search_corners;
    options.search_sides = settings.search_sides;
    return options;
}

/// 2x2 linear part of a film -> pixel map, normalised by its Frobenius norm.
/// The canonical map diag(1, -1) and its dihedral variants all have norm
/// sqrt(2), so this exposes orientation alone.
Eigen::Matrix2d normalised_linear(const Eigen::Matrix3d& transform) {
    const Eigen::Matrix2d linear = transform.block<2, 2>(0, 0);
    return linear * (std::sqrt(2.0) / linear.norm());
}

void expect_linear_near(const Eigen::Matrix2d& got, const Eigen::Matrix2d& want,
                        double tolerance) {
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 2; ++col) {
            EXPECT_NEAR(got(row, col), want(row, col), tolerance)
                << "entry (" << row << "," << col << ")";
        }
    }
}

test::FilmFrameSpec base_spec() {
    test::FilmFrameSpec spec;
    spec.frame_half_mm = 50.0;
    spec.scale_px_per_mm = 15.0;
    spec.marks_mm = marks4(40.0);
    return spec;
}

/// Print a measured error, matching the board-detector suite's style.
void report(const char* name, double value, const char* unit) {
    std::printf("%s %.4f %s\n", name, value, unit);
}

/// A fresh project path under the system temp directory. The suite is a plain
/// TEST (the ProjectIO suite already has non-fixture cases), so temp handling
/// is a pair of helpers rather than a fixture.
std::string temp_project(const std::string& name) {
    const auto parent = fs::temp_directory_path() /
                        ("alphashape_film_" +
                         std::to_string(platform::process_id()) + "_" +
                         std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
                         "_" + name);
    std::error_code error;
    fs::create_directories(parent, error);
    return (parent / (name + ".psx")).string();
}

void remove_project(const std::string& psx) {
    std::error_code error;
    fs::remove_all(fs::path(psx).parent_path(), error);
}

std::string project_doc_text(const std::string& psx) {
    fs::path archive = psx;
    archive.replace_extension();
    const fs::path zip = archive.string() + ".files/project.zip";
    std::ifstream in(zip, std::ios::binary);
    EXPECT_TRUE(in);
    const std::vector<std::uint8_t> bytes(
        (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return ZipArchive::parse(bytes).get_text("doc.xml");
}

/// Removes one PNG when the test scope ends.
struct ScopedPng {
    std::string path;
    ~ScopedPng() {
        if (!path.empty()) {
            std::error_code error;
            fs::remove(path, error);
        }
    }
};

std::string write_frame_png(const test::FilmFrame& frame, const std::string& name) {
    Image image(frame.image.width(), frame.image.height(), 1);
    for (int y = 0; y < frame.image.height(); ++y) {
        for (int x = 0; x < frame.image.width(); ++x) {
            const double value = std::clamp(
                static_cast<double>(frame.image.at(x, y)), 0.0, 1.0);
            image.at(x, y) = static_cast<std::uint8_t>(std::lround(value * 255.0));
        }
    }
    const std::string path =
        platform::unique_temp_path("alphashape_film_" + name + "_", ".png");
    write_png(path, image);
    return path;
}

/// A chunk with one film sensor, one camera and its frame on disk.
void add_film_camera(Document& document, Chunk& chunk, const FilmSettings& film,
                     const test::FilmFrame& frame, const std::string& path) {
    document.set_active_chunk_id(chunk.id());
    Sensor sensor;
    sensor.label = "film-camera";
    sensor.type = CameraType::Frame;
    sensor.width = frame.image.width();
    sensor.height = frame.image.height();
    sensor.calibration =
        Calibration::initial(CameraType::Frame, sensor.width, sensor.height, 1234.0);
    sensor.film = film;
    chunk.sensors().push_back(sensor);
    Camera camera;
    camera.id = chunk.allocate_id();
    camera.sensor_id = 0;
    camera.label = "scan";
    camera.enabled = true;
    chunk.cameras().push_back(camera);
    auto frame_record =
        std::make_shared<Frame>(path, sensor.width, sensor.height);
    frame_record->set_id(chunk.allocate_id());
    frame_record->set_camera_id(camera.id);
    chunk.frames().push_back(frame_record);
}

TEST(Fiducials, FrameQuadCorners) {
    const test::FilmFrameSpec spec = base_spec();
    const test::FilmFrame frame = test::render_film_frame(spec);
    const std::optional<FrameQuad> quad = detect_frame_quad(frame.image);
    ASSERT_TRUE(quad.has_value());
    const std::array<Eigen::Vector2d, 4> detected = {
        quad->top_left, quad->top_right, quad->bottom_right, quad->bottom_left};
    double maximum = 0.0;
    for (std::size_t i = 0; i < detected.size(); ++i) {
        const double error = (detected[i] - frame.frame_corners[i]).norm();
        maximum = std::max(maximum, error);
        EXPECT_LT(error, 0.5) << "corner " << i;
    }
    report("Fiducials.FrameQuadCorners max error", maximum, "px");
}

/// Large scans must not allocate full-resolution gradient planes or an
/// unbounded Hough accumulator.
TEST(Fiducials, DetectionWorkIsBounded) {
    EXPECT_EQ(fiducial_work_scale(1600, 1200), 1);
    EXPECT_EQ(fiducial_work_scale(4096, 3000), 1);
    EXPECT_EQ(fiducial_work_scale(4097, 100), 2);
    EXPECT_EQ(fiducial_work_scale(12000, 9000), 3);
    EXPECT_EQ(fiducial_radius_step(38, 640, 640), 1);
    const int step = fiducial_radius_step(2000, 4096, 4096);
    EXPECT_GE(step, 2);
    const std::uint64_t radii =
        static_cast<std::uint64_t>((2000 + step - 1) / step);
    const std::uint64_t cells = radii * 4096ULL * 4096ULL;
    EXPECT_LE(cells, static_cast<std::uint64_t>(kFiducialHoughCellCap));
}

TEST(Fiducials, CircleCentres) {
    test::FilmFrameSpec spec = base_spec();
    spec.marks_mm = centred_marks();
    const test::FilmFrame frame = test::render_film_frame(spec);
    FilmSettings settings = layout_for(spec.marks_mm);
    settings.detector = FiducialDetector::Circle;
    const FiducialDetection detection =
        detect_fiducials(frame.image, settings, options_for(settings));
    ASSERT_TRUE(detection.failure.empty()) << detection.failure;
    ASSERT_EQ(detection.by_fiducial.size(), frame.mark_pixels.size());
    double maximum = 0.0;
    for (std::size_t i = 0; i < frame.mark_pixels.size(); ++i) {
        ASSERT_TRUE(detection.by_fiducial[i].has_value()) << "mark " << i;
        const double error =
            (*detection.by_fiducial[i] - frame.mark_pixels[i]).norm();
        maximum = std::max(maximum, error);
        EXPECT_LT(error, 0.3) << "mark " << i;
    }
    report("Fiducials.CircleCentres max error", maximum, "px");
}

TEST(Fiducials, CornerMarks) {
    test::FilmFrameSpec spec = base_spec();
    spec.mark_type = FiducialDetector::Corner;
    spec.marks_mm = centred_marks();
    const test::FilmFrame frame = test::render_film_frame(spec);
    const FilmSettings settings = layout_for(spec.marks_mm);
    const FiducialDetection detection =
        detect_fiducials(frame.image, settings, options_for(settings));
    ASSERT_TRUE(detection.failure.empty()) << detection.failure;
    ASSERT_EQ(detection.by_fiducial.size(), frame.mark_pixels.size());
    double maximum = 0.0;
    for (std::size_t i = 0; i < frame.mark_pixels.size(); ++i) {
        ASSERT_TRUE(detection.by_fiducial[i].has_value()) << "mark " << i;
        const double error =
            (*detection.by_fiducial[i] - frame.mark_pixels[i]).norm();
        maximum = std::max(maximum, error);
        EXPECT_LT(error, 0.5) << "mark " << i;
    }
    report("Fiducials.CornerMarks max error", maximum, "px");
}

TEST(Fiducials, OrdersRotatedScan) {
    const std::vector<Eigen::Vector2d> marks = centred_marks();
    FilmSettings settings = layout_for(marks);
    settings.detector = FiducialDetector::Circle;
    for (int variant = 0; variant < 8; ++variant) {
        test::FilmFrameSpec spec = base_spec();
        spec.marks_mm = marks;
        spec.rotation_deg = 90.0 * static_cast<double>(variant % 4);
        spec.mirror = variant >= 4;
        const test::FilmFrame frame = test::render_film_frame(spec);
        const FiducialDetection detection =
            detect_fiducials(frame.image, settings, options_for(settings));
        ASSERT_TRUE(detection.failure.empty())
            << "variant " << variant << ": " << detection.failure;
        EXPECT_FALSE(detection.ambiguous) << "variant " << variant;
        ASSERT_EQ(detection.by_fiducial.size(), frame.mark_pixels.size());
        double maximum = 0.0;
        for (std::size_t i = 0; i < frame.mark_pixels.size(); ++i) {
            ASSERT_TRUE(detection.by_fiducial[i].has_value())
                << "variant " << variant << " mark " << i;
            maximum = std::max(
                maximum, (*detection.by_fiducial[i] - frame.mark_pixels[i]).norm());
        }
        EXPECT_LT(maximum, 1.0) << "variant " << variant;
        report("Fiducials.OrdersRotatedScan max error", maximum, "px");
    }
}

/// A square layout is dihedrally symmetric, so a 90-degree scan leaves eight
/// equally good assignments. With the matching prior the correct one wins; the
/// default prior (rotation 0) deliberately selects the canonical, unrotated
/// labelling instead. Both report `ambiguous`.
TEST(Fiducials, SymmetricRotatedScanUsesOrientationPrior) {
    test::FilmFrameSpec spec = base_spec();
    spec.marks_mm = marks4(40.0);
    spec.rotation_deg = 90.0;
    const test::FilmFrame frame = test::render_film_frame(spec);
    FilmSettings settings = corner_layout(40.0);
    settings.detector = FiducialDetector::Circle;

    FiducialDetectOptions options = options_for(settings);
    options.scan_orientation = "90";
    const FiducialDetection rotated =
        detect_fiducials(frame.image, settings, options);
    ASSERT_TRUE(rotated.failure.empty()) << rotated.failure;
    EXPECT_TRUE(rotated.ambiguous);
    ASSERT_TRUE(rotated.transform.has_value());
    Eigen::Matrix2d want;
    want << 0.0, 1.0, 1.0, 0.0;  // rotation 90 composed with the y flip
    expect_linear_near(normalised_linear(*rotated.transform), want, 1e-2);
    ASSERT_EQ(rotated.by_fiducial.size(), frame.mark_pixels.size());
    for (std::size_t i = 0; i < frame.mark_pixels.size(); ++i) {
        ASSERT_TRUE(rotated.by_fiducial[i].has_value()) << "mark " << i;
        EXPECT_LT((*rotated.by_fiducial[i] - frame.mark_pixels[i]).norm(), 1.0)
            << "mark " << i;
    }

    options.scan_orientation = "0";
    const FiducialDetection canonical =
        detect_fiducials(frame.image, settings, options);
    ASSERT_TRUE(canonical.failure.empty()) << canonical.failure;
    EXPECT_TRUE(canonical.ambiguous);
    ASSERT_TRUE(canonical.transform.has_value());
    Eigen::Matrix2d canonical_want;
    canonical_want << 1.0, 0.0, 0.0, -1.0;
    expect_linear_near(normalised_linear(*canonical.transform), canonical_want,
                       1e-2);
    // The canonical prior must not claim the rotated marks are the truth.
    bool differs = false;
    for (std::size_t i = 0; i < frame.mark_pixels.size(); ++i) {
        if (!canonical.by_fiducial[i].has_value()) {
            continue;
        }
        if ((*canonical.by_fiducial[i] - frame.mark_pixels[i]).norm() > 1.0) {
            differs = true;
        }
    }
    EXPECT_TRUE(differs);
}

/// Three marks fit an affine exactly, so every relabeling has zero RMS. A
/// sheared expected layout produces an anisotropic affine; the near-similarity
/// gate must reject it rather than silently returning the shear.
TEST(Fiducials, ThreeMarkShearedRelabelingRejected) {
    const std::vector<Eigen::Vector2d> film = {
        Eigen::Vector2d(-45.0, 42.0), Eigen::Vector2d(40.0, 47.0),
        Eigen::Vector2d(45.0, -44.0)};
    test::FilmFrameSpec spec = base_spec();
    spec.marks_mm = film;
    spec.rotation_deg = 0.0;
    const test::FilmFrame frame = test::render_film_frame(spec);
    FilmSettings settings = layout_for(film);
    settings.detector = FiducialDetector::Circle;
    const Eigen::Matrix2d shear = (Eigen::Matrix2d() << 1.0, 0.3, 0.0, 1.0).finished();
    for (Fiducial& fiducial : settings.fiducials) {
        fiducial.film_mm = shear * fiducial.film_mm;
    }
    const FiducialDetection detection =
        detect_fiducials(frame.image, settings, options_for(settings));
    EXPECT_NE(detection.failure.find("fiducial ordering failed"), std::string::npos)
        << detection.failure;
    EXPECT_FALSE(detection.transform.has_value());
    EXPECT_TRUE(detection.by_fiducial.empty());
}

TEST(Fiducials, AffineTransformRecovered) {
    test::FilmFrameSpec spec = base_spec();
    spec.rotation_deg = 0.0;
    spec.marks_mm = centred_marks();
    const test::FilmFrame frame = test::render_film_frame(spec);
    FilmSettings settings = layout_for(spec.marks_mm);
    settings.detector = FiducialDetector::Circle;
    const FiducialDetection detection =
        detect_fiducials(frame.image, settings, options_for(settings));
    ASSERT_TRUE(detection.failure.empty()) << detection.failure;
    ASSERT_TRUE(detection.transform.has_value());
    double maximum_relative = 0.0;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const double truth = frame.transform(row, col);
            const double got = (*detection.transform)(row, col);
            const double relative =
                std::abs(got - truth) / std::max(1.0, std::abs(truth));
            maximum_relative = std::max(maximum_relative, relative);
            EXPECT_LT(relative, 1e-3) << "entry (" << row << "," << col << ")";
        }
    }
    EXPECT_LE(detection.rms_px, 0.3);
    report("Fiducials.AffineTransformRecovered max relative", maximum_relative, "");
    report("Fiducials.AffineTransformRecovered rms", detection.rms_px, "px");
}

TEST(Fiducials, ProjectiveNeedsFour) {
    std::vector<Eigen::Vector2d> film = {
        Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(10.0, 0.0),
        Eigen::Vector2d(0.0, 10.0)};
    std::vector<Eigen::Vector2d> pixels = {
        Eigen::Vector2d(100.0, 100.0), Eigen::Vector2d(220.0, 108.0),
        Eigen::Vector2d(106.0, 230.0)};
    Eigen::Matrix3d transform;
    double rms = 0.0;
    EXPECT_FALSE(fit_film_transform(film, pixels, FilmTransformType::Projective,
                                    transform, rms));
    film.emplace_back(10.0, 10.0);
    pixels.emplace_back(232.0, 240.0);
    EXPECT_TRUE(fit_film_transform(film, pixels, FilmTransformType::Projective,
                                   transform, rms));
    EXPECT_LE(rms, 1e-9);
    report("Fiducials.ProjectiveNeedsFour rms", rms, "px");
}

TEST(Fiducials, OrderingFailureIsReported) {
    const test::FilmFrameSpec spec = base_spec();
    const test::FilmFrame frame = test::render_film_frame(spec);
    FilmSettings settings = corner_layout(40.0);
    settings.detector = FiducialDetector::Circle;
    // Move one expected corner by 10 mm (150 px): no affine or projective
    // assignment fits the detected square within max_residual_px.
    settings.fiducials[2].film_mm.x() += 10.0;
    const FiducialDetection detection =
        detect_fiducials(frame.image, settings, options_for(settings));
    EXPECT_NE(detection.failure.find("fiducial ordering failed"), std::string::npos)
        << detection.failure;
    EXPECT_FALSE(detection.transform.has_value());
    EXPECT_TRUE(detection.by_fiducial.empty());
    report("Fiducials.OrderingFailureIsReported", 1.0, "reported");
}

TEST(Fiducials, CalibrateFiducialsDerivesLayout) {
    test::FilmFrameSpec spec = base_spec();
    spec.rotation_deg = 0.0;
    spec.scale_px_per_mm = 50.0;
    spec.frame_half_mm = 15.0;
    spec.marks_mm = small_marks();
    const test::FilmFrame frame = test::render_film_frame(spec);
    FilmSettings settings = layout_for(spec.marks_mm);
    settings.detector = FiducialDetector::Circle;
    const FiducialDetection detection =
        detect_fiducials(frame.image, settings, options_for(settings));
    ASSERT_TRUE(detection.failure.empty()) << detection.failure;
    std::vector<Eigen::Vector2d> pixels;
    for (const std::optional<Eigen::Vector2d>& pixel : detection.by_fiducial) {
        ASSERT_TRUE(pixel.has_value());
        pixels.push_back(*pixel);
    }
    const std::vector<Fiducial> derived =
        derive_film_layout(settings.fiducials, pixels, 0.02);
    ASSERT_EQ(derived.size(), settings.fiducials.size());
    double maximum = 0.0;
    for (std::size_t i = 0; i < derived.size(); ++i) {
        const double error =
            (derived[i].film_mm - settings.fiducials[i].film_mm).norm();
        maximum = std::max(maximum, error);
        EXPECT_LT(error, 0.01) << "fiducial " << i;
    }
    report("Fiducials.CalibrateFiducialsDerivesLayout max error", maximum, "mm");
}

/// CalibrateFiducials derives only the film layout; the camera's focal length
/// and principal point come from the camera calibration and must not move.
TEST(Fiducials, CalibrateFiducialsLeavesIntrinsics) {
    const std::vector<Eigen::Vector2d> marks = small_marks();
    Document document;
    Chunk& chunk = document.workspace().add_chunk("film");
    document.set_active_chunk_id(chunk.id());
    Sensor sensor;
    sensor.label = "sensor";
    sensor.width = 1600;
    sensor.height = 1600;
    sensor.calibration =
        Calibration::initial(CameraType::Frame, sensor.width, sensor.height, 1000.0);
    sensor.calibration.cx = 12.5;
    sensor.calibration.cy = -7.5;
    sensor.film = layout_for(marks);
    chunk.sensors().push_back(sensor);
    const Calibration before = chunk.sensors()[0].calibration;
    Camera camera;
    camera.id = chunk.allocate_id();
    camera.sensor_id = 0;
    camera.label = "scan";
    for (const Eigen::Vector2d& mm : marks) {
        camera.fiducial_pixels.emplace_back(
            Eigen::Vector2d(800.0 + mm.x() / 0.02, 800.0 - mm.y() / 0.02));
    }
    chunk.cameras().push_back(camera);

    Params params = make_params(builtin_registry().require("CalibrateFiducials"));
    params.set("resolution", 0.02);
    run_task(builtin_registry(), document, "CalibrateFiducials", params);
    const Calibration& after = chunk.sensors()[0].calibration;
    EXPECT_DOUBLE_EQ(after.f, before.f);
    EXPECT_DOUBLE_EQ(after.cx, before.cx);
    EXPECT_DOUBLE_EQ(after.cy, before.cy);
    ASSERT_EQ(chunk.sensors()[0].film.fiducials.size(), marks.size());
    for (std::size_t i = 0; i < marks.size(); ++i) {
        EXPECT_LT((chunk.sensors()[0].film.fiducials[i].film_mm - marks[i]).norm(),
                  1e-9);
    }
}

/// End-to-end task run on the synthetic film fixture: ordered marks and the
/// per-scan transform land on the camera, and the asymmetric layout is not
/// reported as ambiguous.
TEST(Fiducials, DetectFiducialsTaskWritesLayout) {
    test::FilmFrameSpec spec = base_spec();
    spec.marks_mm = centred_marks();
    const test::FilmFrame frame = test::render_film_frame(spec);
    ScopedPng png{write_frame_png(frame, "tasklayout")};
    FilmSettings film = layout_for(spec.marks_mm);
    film.detector = FiducialDetector::Circle;
    Document document;
    Chunk& chunk = document.workspace().add_chunk("film");
    add_film_camera(document, chunk, film, frame, png.path);
    Params params = make_params(builtin_registry().require("DetectFiducials"));
    run_task(builtin_registry(), document, "DetectFiducials", params);

    const Camera& camera = chunk.cameras()[0];
    ASSERT_EQ(camera.fiducial_pixels.size(), frame.mark_pixels.size());
    double maximum = 0.0;
    for (std::size_t i = 0; i < frame.mark_pixels.size(); ++i) {
        ASSERT_TRUE(camera.fiducial_pixels[i].has_value()) << "mark " << i;
        maximum = std::max(
            maximum, (*camera.fiducial_pixels[i] - frame.mark_pixels[i]).norm());
    }
    EXPECT_LT(maximum, 1.0);
    EXPECT_TRUE(camera.film_transform.has_value());
    const std::string prefix = "fiducials/" + std::to_string(camera.id) + "/";
    const std::string* status = chunk.meta().find(prefix + "status");
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(*status, "ok");
    const std::string* ambiguous = chunk.meta().find(prefix + "ambiguous");
    ASSERT_NE(ambiguous, nullptr);
    EXPECT_EQ(*ambiguous, "0");
}

/// Without a calibrated layout detection still reports the raw marks, but they
/// must not be stored in `Camera::fiducial_pixels` (parallel to the layout):
/// saving such a project must reload cleanly.
TEST(Fiducials, PixelsOnlyDetectionRoundTrips) {
    test::FilmFrameSpec spec = base_spec();
    spec.marks_mm = centred_marks();
    const test::FilmFrame frame = test::render_film_frame(spec);
    ScopedPng png{write_frame_png(frame, "pixelsonly")};
    FilmSettings film;
    film.enabled = true;
    film.detector = FiducialDetector::Circle;
    film.search_corners = true;
    Document document;
    Chunk& chunk = document.workspace().add_chunk("film");
    add_film_camera(document, chunk, film, frame, png.path);
    Params params = make_params(builtin_registry().require("DetectFiducials"));
    run_task(builtin_registry(), document, "DetectFiducials", params);

    const std::int32_t camera_id = chunk.cameras()[0].id;
    EXPECT_TRUE(chunk.cameras()[0].fiducial_pixels.empty());
    EXPECT_FALSE(chunk.cameras()[0].film_transform.has_value());
    const std::string prefix = "fiducials/" + std::to_string(camera_id) + "/";
    const std::string* status = chunk.meta().find(prefix + "status");
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(*status, "ok_pixels_only");
    const std::string* marks = chunk.meta().find(prefix + "marks");
    ASSERT_NE(marks, nullptr);
    EXPECT_EQ(std::count(marks->begin(), marks->end(), ';'),
              static_cast<std::ptrdiff_t>(3));

    const std::string psx = temp_project("pixelsonly");
    ProjectIO::save(document, psx);
    const std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    ASSERT_TRUE(loaded != nullptr);
    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    EXPECT_TRUE(
        loaded->workspace().chunks().front()->cameras()[0].fiducial_pixels.empty());
    remove_project(psx);
}

TEST(ProjectIO, FilmRoundTrips) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("film");
    document.set_active_chunk_id(chunk.id());
    Sensor sensor;
    sensor.label = "film-camera";
    sensor.width = 1600;
    sensor.height = 1600;
    sensor.film = film_preset("corners4");
    chunk.sensors().push_back(sensor);
    Camera camera;
    camera.id = 0;
    camera.sensor_id = 0;
    camera.label = "scan";
    camera.enabled = true;
    for (int i = 0; i < 4; ++i) {
        camera.fiducial_pixels.emplace_back(Eigen::Vector2d(
            100.0 + 10.0 * static_cast<double>(i), 200.0 + 5.0 * static_cast<double>(i)));
    }
    camera.film_transform = Eigen::Matrix3d::Identity();
    (*camera.film_transform)(0, 2) = 800.0;
    (*camera.film_transform)(1, 2) = 790.0;
    chunk.cameras().push_back(camera);
    const Chunk original = chunk;
    const std::string psx = temp_project("filmtrip");
    ProjectIO::save(document, psx);
    const std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    ASSERT_TRUE(loaded != nullptr);
    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    EXPECT_TRUE(*loaded->workspace().chunks().front() == original);
    const std::string text = project_doc_text(psx);
    remove_project(psx);
    EXPECT_NE(text.find("<film transform=\"affine\" detector=\"corner\""),
              std::string::npos);
    EXPECT_NE(text.find("<film_transform values="), std::string::npos);
}

TEST(ProjectIO, NoFilmKeepsLayout) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("plain");
    document.set_active_chunk_id(chunk.id());
    Sensor sensor;
    sensor.label = "digital";
    sensor.width = 4000;
    sensor.height = 3000;
    chunk.sensors().push_back(sensor);
    Camera camera;
    camera.id = 0;
    camera.sensor_id = 0;
    camera.label = "photo";
    chunk.cameras().push_back(camera);
    const Chunk original = chunk;
    const std::string psx = temp_project("plain");
    ProjectIO::save(document, psx);
    const std::string text = project_doc_text(psx);
    EXPECT_EQ(text.find("<film"), std::string::npos);
    const std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    ASSERT_TRUE(loaded != nullptr);
    EXPECT_TRUE(*loaded->workspace().chunks().front() == original);
    remove_project(psx);
}

TEST(Commands, SetSensorFilmUndoes) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("film");
    document.set_active_chunk_id(chunk.id());
    Sensor sensor;
    sensor.label = "film-camera";
    chunk.sensors().push_back(sensor);
    const Chunk before = chunk;
    const FilmSettings preset = film_preset("all8");
    auto command =
        make_set_sensor_film_command(chunk.id(), 0, preset);
    command->apply(document);
    Chunk* after = document.workspace().find_chunk(chunk.id());
    ASSERT_TRUE(after != nullptr);
    ASSERT_EQ(after->sensors().size(), 1U);
    EXPECT_TRUE(after->sensors()[0].film == preset);
    command->revert(document);
    EXPECT_TRUE(*after == before);
}

}  // namespace
}  // namespace alphashape
