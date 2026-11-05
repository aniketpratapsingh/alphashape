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
#include "common/io/project_io.hpp"
#include "image/image.hpp"
#include "make_film_frame.hpp"

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

class FilmProject : public ::testing::Test {
protected:
    void SetUp() override {
        const auto base = fs::temp_directory_path() /
                          ("alphashape_film_" + std::to_string(platform::process_id()) +
                           "_" +
                           std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
        dir_ = base.string() + "_" + std::to_string(counter_++);
        std::error_code error;
        fs::remove_all(dir_, error);
        fs::create_directories(dir_, error);
        ASSERT_FALSE(error) << error.message();
    }
    void TearDown() override {
        std::error_code error;
        fs::remove_all(dir_, error);
    }
    std::string path(const std::string& name) const {
        return (fs::path(dir_) / name).string();
    }
    static std::string doc_text(const std::string& psx) {
        fs::path archive = psx;
        archive.replace_extension();
        const fs::path zip = archive.string() + ".files/project.zip";
        std::ifstream in(zip, std::ios::binary);
        EXPECT_TRUE(in);
        const std::vector<std::uint8_t> bytes(
            (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return ZipArchive::parse(bytes).get_text("doc.xml");
    }

private:
    std::string dir_;
    static int counter_;
};

int FilmProject::counter_ = 0;

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
        const Eigen::Vector2d delta =
            *detection.by_fiducial[i] - frame.mark_pixels[i];
        const double error = delta.norm();
        std::printf("  mark %zu detected (%.2f, %.2f) true (%.2f, %.2f) "
                    "delta (%.2f, %.2f)\n",
                    i, detection.by_fiducial[i]->x(),
                    detection.by_fiducial[i]->y(), frame.mark_pixels[i].x(),
                    frame.mark_pixels[i].y(), delta.x(), delta.y());
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
    const FilmSettings settings = layout_for(marks);
    for (int variant = 0; variant < 4; ++variant) {
        test::FilmFrameSpec spec = base_spec();
        spec.marks_mm = marks;
        spec.rotation_deg = (variant % 2 == 0) ? 90.0 : 180.0;
        spec.mirror = variant >= 2;
        const test::FilmFrame frame = test::render_film_frame(spec);
        const FiducialDetection detection =
            detect_fiducials(frame.image, settings, options_for(settings));
        ASSERT_TRUE(detection.failure.empty())
            << "variant " << variant << ": " << detection.failure;
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

TEST_F(FilmProject, ProjectIOFilmRoundTrips) {
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
    const std::string psx = path("film.psx");
    ProjectIO::save(document, psx);
    const std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    ASSERT_TRUE(loaded != nullptr);
    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    EXPECT_TRUE(*loaded->workspace().chunks().front() == original);
    const std::string text = doc_text(psx);
    EXPECT_NE(text.find("<film transform=\"affine\" detector=\"corner\""),
              std::string::npos);
    EXPECT_NE(text.find("<film_transform values="), std::string::npos);
}

TEST_F(FilmProject, ProjectIONoFilmKeepsLayout) {
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
    const std::string psx = path("plain.psx");
    ProjectIO::save(document, psx);
    const std::string text = doc_text(psx);
    EXPECT_EQ(text.find("<film"), std::string::npos);
    const std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    ASSERT_TRUE(loaded != nullptr);
    EXPECT_TRUE(*loaded->workspace().chunks().front() == original);
}

TEST_F(FilmProject, CommandsSetSensorFilmUndoes) {
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
