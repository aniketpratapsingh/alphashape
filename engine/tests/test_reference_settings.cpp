// WP-13.1: chunk reference settings, check points, capture distance and the
// reference table.
//
// Every table assertion uses hand-computed synthetic geometry: an unreferenced
// chunk keeps the numbers in chunk-local units, so the source/estimate/error
// arithmetic is exact and the capture-distance checks reduce to a 10 m height
// difference. The persistence tests pin both directions: non-default settings
// round-trip, and a default chunk writes no <reference> element at all.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/io/zip.hpp"
#include "base/platform/temp.hpp"
#include "common/algorithm/reference/reference_table.hpp"
#include "common/commands/command.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/document.hpp"
#include "common/core/reference_settings.hpp"
#include "common/io/project_io.hpp"
#include "model/crs/coordinate_system.hpp"
#include "perf_gate.hpp"

namespace alphashape::reference {
namespace {

namespace fs = std::filesystem;

Camera make_camera(std::int32_t id, const Eigen::Vector3d& centre) {
    Camera camera;
    camera.id = id;
    camera.label = "cam" + std::to_string(id);
    camera.enabled = true;
    camera.pose = Transform::Identity();
    // X_cam = [I | b] X_world, so the centre is -b.
    camera.pose.block<3, 1>(0, 3) = -centre;
    return camera;
}

Marker make_marker(std::int32_t id, const Eigen::Vector3d& reference,
                   const Eigen::Vector3d& position) {
    Marker marker;
    marker.id = id;
    marker.label = "marker" + std::to_string(id);
    marker.reference = reference;
    marker.position = position;
    marker.valid = true;
    return marker;
}

const ReferenceRow& row_for(const ReferenceTable& table, std::int32_t id) {
    for (const ReferenceRow& row : table.rows) {
        if (row.id == id) {
            return row;
        }
    }
    ADD_FAILURE() << "no reference row for id " << id;
    static const ReferenceRow empty;
    return empty;
}

}  // namespace

TEST(Reference, SettingsDefaults) {
    const ReferenceSettings settings;
    EXPECT_DOUBLE_EQ(settings.camera_location_accuracy_m, 10.0);
    EXPECT_DOUBLE_EQ(settings.camera_rotation_accuracy_deg, 10.0);
    EXPECT_DOUBLE_EQ(settings.marker_location_accuracy_m, 0.005);
    EXPECT_DOUBLE_EQ(settings.marker_projection_accuracy_px, 0.5);
    EXPECT_DOUBLE_EQ(settings.scalebar_accuracy_m, 0.001);
    EXPECT_DOUBLE_EQ(settings.tiepoint_accuracy_px, 1.0);
    EXPECT_TRUE(settings.use_gcp);
    EXPECT_TRUE(settings.is_default());

    const Chunk chunk("defaults");
    EXPECT_TRUE(chunk.reference_settings().is_default());
}

namespace {

/// A project at `psx` keeps its content in a sibling `.files/project.zip`; the
/// doc.xml text is what the persistence assertions inspect.
std::string doc_text(const std::string& psx) {
    fs::path base = psx;
    base.replace_extension();
    std::ifstream in(base.string() + ".files/project.zip", std::ios::binary);
    EXPECT_TRUE(in);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                          std::istreambuf_iterator<char>());
    return ZipArchive::parse(bytes).get_text("doc.xml");
}

}  // namespace

TEST(Reference, SettingsPersistRoundTrip) {
    const fs::path dir =
        fs::temp_directory_path() /
        ("alphashape_reference_" + std::to_string(platform::process_id()));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    ASSERT_FALSE(ec) << ec.message();

    ReferenceSettings changed;
    changed.camera_location_accuracy_m = 0.25;
    changed.camera_rotation_accuracy_deg = 1.5;
    changed.marker_location_accuracy_m = 0.01;
    changed.marker_projection_accuracy_px = 0.75;
    changed.scalebar_accuracy_m = 0.002;
    changed.tiepoint_accuracy_px = 2.0;
    changed.use_gcp = false;

    Document document;
    Chunk& chunk = document.workspace().add_chunk("chunk");
    document.set_active_chunk_id(chunk.id());
    chunk.set_reference_settings(changed);
    const std::string changed_path = (dir / "changed.psx").string();
    ProjectIO::save(document, changed_path);
    const std::shared_ptr<Document> loaded = ProjectIO::load(changed_path);
    ASSERT_NE(loaded->active_chunk(), nullptr);
    EXPECT_EQ(loaded->active_chunk()->reference_settings(), changed);
    EXPECT_NE(doc_text(changed_path).find("<reference "), std::string::npos);

    // Defaults write no element at all.
    Document plain;
    plain.workspace().add_chunk("plain");
    const std::string plain_path = (dir / "plain.psx").string();
    ProjectIO::save(plain, plain_path);
    EXPECT_EQ(doc_text(plain_path).find("<reference"), std::string::npos);
    fs::remove_all(dir, ec);
}

TEST(Reference, TableErrorsAreEstimatedMinusSource) {
    Chunk chunk("errors");
    chunk.set_id(0);
    chunk.cameras().push_back(make_camera(7, Eigen::Vector3d(10.0, 20.0, 30.0)));
    chunk.cameras().back().reference.location = Eigen::Vector3d(0.0, 1.0, 2.0);
    chunk.markers().push_back(
        make_marker(3, Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 2.0, 3.0)));

    const ReferenceTable table = build_reference_table(chunk, crs::CoordinateSystem());
    EXPECT_FALSE(table.georeferenced);

    const ReferenceRow& camera = row_for(table, 7);
    ASSERT_TRUE(camera.source.has_value());
    ASSERT_TRUE(camera.estimated.has_value());
    ASSERT_TRUE(camera.error.has_value());
    EXPECT_NEAR((*camera.estimated - *camera.source - *camera.error).norm(), 0.0, 1e-9);
    EXPECT_NEAR(camera.error->x(), 10.0, 1e-9);
    EXPECT_NEAR(camera.error->y(), 19.0, 1e-9);
    EXPECT_NEAR(camera.error->z(), 28.0, 1e-9);
    // No per-camera accuracy is set: the chunk default (10 m) is the sigma.
    EXPECT_NEAR(camera.sigma, 10.0, 1e-12);

    const ReferenceRow& marker = row_for(table, 3);
    ASSERT_TRUE(marker.error.has_value());
    EXPECT_NEAR(marker.error->x(), 1.0, 1e-9);
    EXPECT_NEAR(marker.error->y(), 2.0, 1e-9);
    EXPECT_NEAR(marker.error->z(), 3.0, 1e-9);
}

TEST(Reference, CheckPointsExcludedFromControl) {
    Chunk chunk("control");
    chunk.set_id(0);
    constexpr int kMarkers = 5;
    for (int i = 0; i < kMarkers; ++i) {
        Marker marker = make_marker(100 + i, Eigen::Vector3d(1.0, 2.0, 3.0),
                                    Eigen::Vector3d(1.0, 2.0, 3.0));
        if (i < 2) {
            set_marker_role(marker, MarkerReferenceRole::CheckPoint);
        }
        chunk.markers().push_back(std::move(marker));
    }

    const ReferenceTable table = build_reference_table(chunk, crs::CoordinateSystem());
    EXPECT_EQ(table.control_count(), static_cast<std::size_t>(kMarkers - 2));
    EXPECT_EQ(row_for(table, 100).kind, ReferenceRowKind::CheckPoint);
    EXPECT_EQ(row_for(table, 101).kind, ReferenceRowKind::CheckPoint);
    EXPECT_EQ(row_for(table, 102).kind, ReferenceRowKind::Marker);
}

TEST(Reference, CaptureDistanceMatchesGeometry) {
    Chunk chunk("capture");
    chunk.set_id(0);
    chunk.cameras().push_back(make_camera(4, Eigen::Vector3d(0.0, 0.0, 10.0)));
    Marker marker = make_marker(9, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    MarkerProjection projection;
    projection.camera_id = 4;
    marker.projections.push_back(projection);
    chunk.markers().push_back(marker);

    const ReferenceTable table = build_reference_table(chunk, crs::CoordinateSystem());
    EXPECT_NEAR(row_for(table, 4).capture_distance_m, 10.0, 1e-6);
    EXPECT_NEAR(row_for(table, 9).capture_distance_m, 10.0, 1e-6);
}

TEST(Reference, UnreferencedChunkIsLocal) {
    Chunk chunk("local");
    chunk.set_id(0);
    chunk.cameras().push_back(make_camera(1, Eigen::Vector3d(2.0, 3.0, 4.0)));
    chunk.cameras().back().reference.location = Eigen::Vector3d(2.0, 3.0, 4.0);
    chunk.markers().push_back(
        make_marker(2, Eigen::Vector3d(1.0, 1.0, 1.0), Eigen::Vector3d(1.0, 1.0, 1.0)));

    const ReferenceTable table = build_reference_table(chunk, crs::CoordinateSystem());
    EXPECT_FALSE(table.georeferenced);
    const ReferenceRow& camera = row_for(table, 1);
    ASSERT_TRUE(camera.source.has_value());
    ASSERT_TRUE(camera.estimated.has_value());
    ASSERT_TRUE(camera.error.has_value());
    EXPECT_NEAR(camera.estimated->x(), 2.0, 1e-12);
    EXPECT_NEAR(camera.error->norm(), 0.0, 1e-12);
}

TEST(Reference, ThousandCamerasAndFiftyMarkersWithinBudget) {
    Chunk chunk("perf");
    chunk.set_id(0);
    constexpr int kCameras = 1000;
    constexpr int kMarkers = 50;
    for (int c = 0; c < kCameras; ++c) {
        Camera camera =
            make_camera(c, Eigen::Vector3d(static_cast<double>(c), 0.0, 100.0));
        camera.reference.location = Eigen::Vector3d(0.0, 0.0, 100.0);
        chunk.cameras().push_back(camera);
    }
    for (int m = 0; m < kMarkers; ++m) {
        Marker marker =
            make_marker(1000 + m, Eigen::Vector3d(0.0, 0.0, 0.0),
                        Eigen::Vector3d(static_cast<double>(m) * 0.1, 1.0, 0.0));
        for (int c = 0; c < kCameras; ++c) {
            MarkerProjection projection;
            projection.camera_id = c;
            marker.projections.push_back(projection);
        }
        chunk.markers().push_back(std::move(marker));
    }

    const auto start = std::chrono::steady_clock::now();
    const ReferenceTable table = build_reference_table(chunk, crs::CoordinateSystem());
    const auto end = std::chrono::steady_clock::now();
    ASSERT_EQ(table.rows.size(), static_cast<std::size_t>(kCameras + kMarkers));

    const double millis =
        std::chrono::duration<double, std::milli>(end - start).count();
    if (test::perf_asserts_enabled()) {
        EXPECT_LE(millis, 200.0) << "reference table took " << millis << " ms";
    }
}

}  // namespace alphashape::reference
