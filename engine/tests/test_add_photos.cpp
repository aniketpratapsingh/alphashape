// Tests for AddPhotos: importing image files into a chunk as sensors, cameras
// and frames.
//
// The camera fixtures are JPEGs with real EXIF, so every expected value in this
// file is recomputed from read_exif/read_image_info rather than hard-coded —
// that is what keeps the tests honest when the fixtures are regenerated. The
// one exception is the fallback focal-length constant, which is ours.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/algorithm/photos/add_photos_task.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/io/project_io.hpp"
#include "common/tasks/registry.hpp"
#include "image/io/exif.hpp"
#include "image/io/image_io.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

#ifdef ALPHASHAPE_DATA_DIR
const fs::path kDataDir = ALPHASHAPE_DATA_DIR;
#else
// tests/test_add_photos.cpp -> the repository root is two levels up.
const fs::path kDataDir = fs::path(__FILE__).parent_path().parent_path() / "data";
#endif

std::string fixture(const std::string& name) { return (kDataDir / name).string(); }

/// A document with one active chunk, ready for AddPhotos.
Document document_with_chunk() {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    return document;
}

/// Run AddPhotos with the given file names and default parameters.
void add_photos(Document& document, const std::vector<std::string>& filenames) {
    Params params = make_params(builtin_registry().require("AddPhotos"));
    params.set("filenames", ParamValue{filenames});
    run_task(builtin_registry(), document, "AddPhotos", params);
}

/// The focal length in pixels the task must produce, recomputed here from the
/// same inputs so the test is a check of the formula and not of a literal.
double expected_focal_pixels(const std::string& path) {
    const ImageInfo info = read_image_info(path);
    const ExifData exif = read_exif(path);
    std::optional<double> pixel_size = exif.pixel_size_um();
    if (pixel_size.has_value() && exif.pixel_x_dimension.has_value() &&
        *exif.pixel_x_dimension != info.width) {
        *pixel_size *= static_cast<double>(*exif.pixel_x_dimension) /
                       static_cast<double>(info.width);
    }
    if (exif.focal_length_mm.has_value() && *exif.focal_length_mm > 0.0 &&
        pixel_size.has_value() && *pixel_size > 0.0) {
        return *exif.focal_length_mm * 1000.0 / *pixel_size;
    }
    if (exif.focal_length_35mm.has_value() && *exif.focal_length_35mm > 0.0) {
        return *exif.focal_length_35mm *
               std::hypot(static_cast<double>(info.width), static_cast<double>(info.height)) /
               std::hypot(36.0, 24.0);
    }
    return 1.2 * static_cast<double>(std::max(info.width, info.height));
}

/// The position of the camera whose label matches, or npos.
std::size_t camera_index_for_label(const Chunk& chunk, const std::string& label) {
    const auto& cameras = chunk.cameras();
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        if (cameras[i].label == label) {
            return i;
        }
    }
    return std::string::npos;
}

/// Copy a fixture to `dest` byte-for-byte.
void copy_file(const std::string& from, const std::string& dest) {
    std::ifstream in(from, std::ios::binary);
    ASSERT_TRUE(in) << "cannot open " << from;
    std::ofstream out(dest, std::ios::binary);
    ASSERT_TRUE(out) << "cannot write " << dest;
    out << in.rdbuf();
}

TEST(AddPhotos, GroupsThreeCameraFixturesIntoTwoSensors) {
    Document document = document_with_chunk();
    add_photos(document, {fixture("cam_a1.jpg"), fixture("cam_a2.jpg"), fixture("cam_b1.jpg")});

    Chunk& chunk = *document.active_chunk();
    ASSERT_EQ(chunk.sensors().size(), 2U);
    ASSERT_EQ(chunk.cameras().size(), 3U);
    ASSERT_EQ(chunk.frames().size(), 3U);

    const std::size_t a1 = camera_index_for_label(chunk, "cam_a1");
    const std::size_t a2 = camera_index_for_label(chunk, "cam_a2");
    const std::size_t b1 = camera_index_for_label(chunk, "cam_b1");
    ASSERT_NE(a1, std::string::npos);
    ASSERT_NE(a2, std::string::npos);
    ASSERT_NE(b1, std::string::npos);
    EXPECT_EQ(chunk.cameras()[a1].sensor_id, chunk.cameras()[a2].sensor_id);
    EXPECT_NE(chunk.cameras()[a1].sensor_id, chunk.cameras()[b1].sensor_id);
    // Sensor ids are indices into chunk.sensors(), so both must be in range.
    for (const Camera& camera : chunk.cameras()) {
        EXPECT_GE(camera.sensor_id, 0);
        EXPECT_LT(camera.sensor_id, static_cast<std::int32_t>(chunk.sensors().size()));
    }
}

TEST(AddPhotos, SensorKeepsSerialOnlyWhenEveryBodyAgrees) {
    const ExifData a1 = read_exif(fixture("cam_a1.jpg"));
    const ExifData a2 = read_exif(fixture("cam_a2.jpg"));
    ASSERT_TRUE(a1.body_serial.has_value());
    ASSERT_TRUE(a2.body_serial.has_value());
    ASSERT_NE(*a1.body_serial, *a2.body_serial);  // fixture premise: two bodies

    Document single = document_with_chunk();
    add_photos(single, {fixture("cam_a1.jpg")});
    EXPECT_EQ(single.active_chunk()->sensors().front().serial, *a1.body_serial);

    Document mixed = document_with_chunk();
    add_photos(mixed, {fixture("cam_a1.jpg"), fixture("cam_a2.jpg")});
    ASSERT_EQ(mixed.active_chunk()->sensors().size(), 1U);
    EXPECT_TRUE(mixed.active_chunk()->sensors().front().serial.empty());
}

TEST(AddPhotos, FocalPlaneRouteProducesTheExpectedFocalLength) {
    Document document = document_with_chunk();
    add_photos(document, {fixture("cam_a1.jpg")});

    Chunk& chunk = *document.active_chunk();
    ASSERT_EQ(chunk.sensors().size(), 1U);
    const Sensor& sensor = chunk.sensors().front();
    EXPECT_DOUBLE_EQ(sensor.calibration.f, expected_focal_pixels(fixture("cam_a1.jpg")));
    // Sanity: this fixture does have a focal-plane resolution, so the route is
    // the metric one and not the fallback.
    EXPECT_TRUE(read_exif(fixture("cam_a1.jpg")).pixel_size_um().has_value());
}

TEST(AddPhotos, ThirtyFiveMillimetreRouteProducesTheExpectedFocalLength) {
    Document document = document_with_chunk();
    add_photos(document, {fixture("cam_b1.jpg")});

    Chunk& chunk = *document.active_chunk();
    ASSERT_EQ(chunk.sensors().size(), 1U);
    const Sensor& sensor = chunk.sensors().front();
    EXPECT_DOUBLE_EQ(sensor.calibration.f, expected_focal_pixels(fixture("cam_b1.jpg")));
    // The 35 mm route is the one that must fire: no focal-plane resolution.
    EXPECT_FALSE(read_exif(fixture("cam_b1.jpg")).pixel_size_um().has_value());
    EXPECT_TRUE(read_exif(fixture("cam_b1.jpg")).focal_length_35mm.has_value());
}

TEST(AddPhotos, FallbackRouteUsesTheDefaultConstantAndUnknownLabel) {
    Document document = document_with_chunk();
    add_photos(document, {fixture("grey.png")});

    Chunk& chunk = *document.active_chunk();
    ASSERT_EQ(chunk.sensors().size(), 1U);
    const Sensor& sensor = chunk.sensors().front();
    const ImageInfo info = read_image_info(fixture("grey.png"));
    EXPECT_DOUBLE_EQ(sensor.calibration.f,
                     1.2 * static_cast<double>(std::max(info.width, info.height)));
    EXPECT_EQ(sensor.label, "Unknown");
    EXPECT_EQ(sensor.pixel_size_um, 0.0);
    EXPECT_EQ(sensor.focal_length_mm, 0.0);
}

TEST(AddPhotos, GpsIsLoadedWithLongitudeLatitudeAltitude) {
    Document document = document_with_chunk();
    add_photos(document, {fixture("cam_a2.jpg"), fixture("cam_b1.jpg")});

    Chunk& chunk = *document.active_chunk();
    const std::size_t a2 = camera_index_for_label(chunk, "cam_a2");
    const std::size_t b1 = camera_index_for_label(chunk, "cam_b1");
    ASSERT_NE(a2, std::string::npos);
    ASSERT_NE(b1, std::string::npos);

    // The reference slots are (longitude, latitude, altitude): x = lon, y = lat.
    const Camera& camera = chunk.cameras()[a2];
    ASSERT_TRUE(camera.reference.location.has_value());
    const ExifData exif = read_exif(fixture("cam_a2.jpg"));
    ASSERT_TRUE(exif.gps.has_value());
    EXPECT_DOUBLE_EQ((*camera.reference.location).x(), exif.gps->longitude_deg);
    EXPECT_DOUBLE_EQ((*camera.reference.location).y(), exif.gps->latitude_deg);
    EXPECT_LT((*camera.reference.location).x(), 0.0);
    EXPECT_LT((*camera.reference.location).y(), 0.0);
    ASSERT_TRUE(exif.gps->altitude_m.has_value());
    EXPECT_DOUBLE_EQ((*camera.reference.location).z(), *exif.gps->altitude_m);
    EXPECT_TRUE(camera.reference.enabled);

    // cam_b1 has no GPS at all.
    EXPECT_FALSE(chunk.cameras()[b1].reference.location.has_value());
}

TEST(AddPhotos, LoadReferenceFalseLeavesEveryReferenceUnset) {
    Document document = document_with_chunk();
    Params params = make_params(builtin_registry().require("AddPhotos"));
    params.set("filenames", ParamValue{std::vector<std::string>{fixture("cam_a1.jpg")}});
    params.set("load_reference", false);
    run_task(builtin_registry(), document, "AddPhotos", params);

    Chunk& chunk = *document.active_chunk();
    ASSERT_EQ(chunk.cameras().size(), 1U);
    EXPECT_FALSE(chunk.cameras().front().reference.location.has_value());
    // The camera is imported unaligned either way.
    EXPECT_FALSE(chunk.cameras().front().enabled);
}

TEST(AddPhotos, StripExtensionsControlsTheCameraLabel) {
    Document with_strip = document_with_chunk();
    add_photos(with_strip, {fixture("cam_a1.jpg")});
    ASSERT_EQ(with_strip.active_chunk()->cameras().size(), 1U);
    EXPECT_EQ(with_strip.active_chunk()->cameras().front().label, "cam_a1");

    Document without_strip = document_with_chunk();
    Params params = make_params(builtin_registry().require("AddPhotos"));
    params.set("filenames", ParamValue{std::vector<std::string>{fixture("cam_a1.jpg")}});
    params.set("strip_extensions", false);
    run_task(builtin_registry(), without_strip, "AddPhotos", params);
    ASSERT_EQ(without_strip.active_chunk()->cameras().size(), 1U);
    EXPECT_EQ(without_strip.active_chunk()->cameras().front().label, "cam_a1.jpg");
}

TEST(AddPhotos, SecondCallReusesAnExistingSensor) {
    // A copy of cam_a1 under another name is, by construction, the same body.
    const fs::path dir = fs::temp_directory_path() /
                         ("alphashape_addphotos_reuse_" +
                          std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    ASSERT_FALSE(ec);
    const std::string duplicate = (dir / "cam_a1_copy.jpg").string();
    copy_file(fixture("cam_a1.jpg"), duplicate);

    Document document = document_with_chunk();
    add_photos(document, {fixture("cam_a1.jpg")});
    ASSERT_EQ(document.active_chunk()->sensors().size(), 1U);
    add_photos(document, {duplicate});

    Chunk& chunk = *document.active_chunk();
    EXPECT_EQ(chunk.sensors().size(), 1U) << "the second call created a sensor instead of reusing";
    ASSERT_EQ(chunk.cameras().size(), 2U);
    EXPECT_EQ(chunk.cameras()[0].sensor_id, chunk.cameras()[1].sensor_id);

    fs::remove_all(dir, ec);
}

TEST(AddPhotos, DuplicatePathInTheChunkIsRejectedAndLeavesTheChunkUntouched) {
    Document document = document_with_chunk();
    add_photos(document, {fixture("cam_a1.jpg")});
    const Chunk before = *document.active_chunk();

    Params params = make_params(builtin_registry().require("AddPhotos"));
    params.set("filenames", ParamValue{std::vector<std::string>{fixture("cam_a1.jpg")}});
    EXPECT_THROW(run_task(builtin_registry(), document, "AddPhotos", params), ValueError);
    EXPECT_TRUE(*document.active_chunk() == before);
}

TEST(AddPhotos, DuplicatePathInOneCallIsRejected) {
    Document document = document_with_chunk();
    Params params = make_params(builtin_registry().require("AddPhotos"));
    params.set("filenames",
               ParamValue{std::vector<std::string>{fixture("cam_a1.jpg"), fixture("cam_a1.jpg")}});
    EXPECT_THROW(run_task(builtin_registry(), document, "AddPhotos", params), ValueError);
    EXPECT_EQ(document.active_chunk()->sensors().size(), 0U);
    EXPECT_EQ(document.active_chunk()->cameras().size(), 0U);
}

TEST(AddPhotos, MissingFileInTheMiddleOfTheListLeavesTheChunkUntouched) {
    Document document = document_with_chunk();
    const Chunk before = *document.active_chunk();

    Params params = make_params(builtin_registry().require("AddPhotos"));
    params.set("filenames", ParamValue{std::vector<std::string>{
                             fixture("cam_a1.jpg"), fixture("does_not_exist.jpg"),
                             fixture("cam_b1.jpg")}});
    bool threw = false;
    std::string message;
    try {
        run_task(builtin_registry(), document, "AddPhotos", params);
    } catch (const Error& e) {
        threw = true;
        message = e.what();
    }
    EXPECT_TRUE(threw);
    // The error must name the offending file.
    EXPECT_NE(message.find("does_not_exist.jpg"), std::string::npos) << message;
    // Nothing was committed, not even the files that read cleanly.
    EXPECT_TRUE(*document.active_chunk() == before);
    EXPECT_EQ(document.active_chunk()->sensors().size(), 0U);
}

TEST(AddPhotos, EmptyFilenamesListIsRejected) {
    Document document = document_with_chunk();
    Params params = make_params(builtin_registry().require("AddPhotos"));
    params.set("filenames", ParamValue{std::vector<std::string>{}});
    EXPECT_THROW(run_task(builtin_registry(), document, "AddPhotos", params), ValueError);
    EXPECT_EQ(document.active_chunk()->sensors().size(), 0U);
}

TEST(AddPhotos, RequiredFilenamesUnsetIsRejected) {
    Document document = document_with_chunk();
    Params params = make_params(builtin_registry().require("AddPhotos"));
    // filenames is required (no default), so reading it must throw before any
    // file is touched.
    EXPECT_THROW(run_task(builtin_registry(), document, "AddPhotos", params), ValueError);
}

TEST(AddPhotos, UnknownParameterIsInvalidArgumentAndMetaRecordsEveryKey) {
    Document document = document_with_chunk();

    Params bad = make_params(builtin_registry().require("AddPhotos"));
    bad.set("filenames", ParamValue{std::vector<std::string>{fixture("cam_a1.jpg")}});
    EXPECT_THROW(bad.set("no_such_param", std::int64_t{1}), InvalidArgumentError);

    add_photos(document, {fixture("cam_a1.jpg")});
    const Meta& meta = document.active_chunk()->meta();
    EXPECT_NE(meta.find("AddPhotos/filenames"), nullptr);
    EXPECT_NE(meta.find("AddPhotos/load_reference"), nullptr);
    EXPECT_NE(meta.find("AddPhotos/strip_extensions"), nullptr);
}

TEST(AddPhotos, PipelineRoundTripsThroughDetectFeaturesAndSaveLoad) {
    Document document = document_with_chunk();
    document.set_path((fs::temp_directory_path() / "addphotos_pipeline.psx").string());
    add_photos(document, {fixture("cam_a1.jpg"), fixture("cam_a2.jpg"), fixture("cam_b1.jpg")});

    Params detect = make_params(builtin_registry().require("DetectFeatures"));
    run_task(builtin_registry(), document, "DetectFeatures", detect);

    Chunk& chunk = *document.active_chunk();
    for (const KeypointSet& set : chunk.tie_points().keypoint_sets) {
        EXPECT_GT(set.keypoints.size(), 0U);
        EXPECT_TRUE(set.consistent());
    }
    EXPECT_EQ(chunk.tie_points().keypoint_sets.size(), 3U);
    const Chunk before = chunk;

    // Save and reload: the chunk must compare equal field for field.
    const fs::path dir = fs::temp_directory_path() /
                         ("alphashape_addphotos_pipeline_" +
                          std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    ASSERT_FALSE(ec);
    const std::string project = (dir / "proj.psx").string();
    document.set_path(project);
    ProjectIO::save(document, project);
    auto reloaded = ProjectIO::load(project);

    ASSERT_NE(reloaded->active_chunk(), nullptr);
    EXPECT_TRUE(*reloaded->active_chunk() == before);
    for (const KeypointSet& set : reloaded->active_chunk()->tie_points().keypoint_sets) {
        EXPECT_GT(set.keypoints.size(), 0U);
    }
    fs::remove_all(dir, ec);
}

}  // namespace
}  // namespace alphashape
