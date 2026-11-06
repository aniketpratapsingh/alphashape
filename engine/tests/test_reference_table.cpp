// WP-14.11: the reference table the Reference pane renders.
//
// The local chunk keeps every coordinate in chunk-local units, so a known
// per-row offset makes both RMSE figures analytic. The geographic chunk
// (EPSG:4979) pins the metre conversion: a 1e-5 degree longitude offset at
// 50 N must come back as metres, not degrees.

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "common/algorithm/reference/reference_table.hpp"
#include "common/core/chunk.hpp"
#include "model/crs/coordinate_system.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape::reference {
namespace {

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

TEST(ReferenceTable, KnownOffsetsAndRmse) {
    Chunk chunk("local");
    chunk.set_id(0);

    constexpr int kCameras = 12;
    double camera_squares = 0.0;
    for (int i = 0; i < kCameras; ++i) {
        const double offset = static_cast<double>(i) - 5.5;
        Camera camera = make_camera(i, Eigen::Vector3d::Zero());
        camera.reference.location = Eigen::Vector3d(-offset, 0.0, 0.0);
        camera.reference.accuracy = Eigen::Vector3d::Constant(0.02);
        chunk.cameras().push_back(camera);
        camera_squares += offset * offset;
    }

    constexpr int kMarkers = 4;
    double marker_squares = 0.0;
    for (int i = 0; i < kMarkers; ++i) {
        const double offset = static_cast<double>(i) + 1.0;
        const Eigen::Vector3d position(10.0, 20.0, 30.0);
        Marker marker = make_marker(
            100 + i, position - Eigen::Vector3d(0.0, 0.0, offset), position);
        marker.pinned = i == 0;
        marker.projections.resize(static_cast<std::size_t>(i + 1));
        chunk.markers().push_back(marker);
        marker_squares += offset * offset;
    }

    const ReferenceTable table = build_reference_table(chunk, crs::CoordinateSystem());
    ASSERT_EQ(table.rows.size(), static_cast<std::size_t>(kCameras + kMarkers));

    const ReferenceRow& camera = row_for(table, 0);
    EXPECT_EQ(camera.kind, ReferenceRowKind::Camera);
    ASSERT_TRUE(camera.error.has_value());
    ASSERT_TRUE(camera.error_m.has_value());
    EXPECT_NEAR(*camera.error_m, 5.5, 1e-9);
    EXPECT_EQ(camera.projection_count, 0U);

    const ReferenceRow& marker = row_for(table, 100);
    EXPECT_EQ(marker.kind, ReferenceRowKind::Marker);
    ASSERT_TRUE(marker.error_m.has_value());
    EXPECT_NEAR(*marker.error_m, 1.0, 1e-9);
    EXPECT_EQ(marker.projection_count, 1U);
    EXPECT_TRUE(marker.pinned);
    EXPECT_FALSE(row_for(table, 101).pinned);

    const ReferenceSummary summary = table.summary();
    EXPECT_EQ(summary.camera_count, static_cast<std::size_t>(kCameras));
    EXPECT_EQ(summary.marker_count, static_cast<std::size_t>(kMarkers));
    const double camera_rmse =
        std::sqrt(camera_squares / static_cast<double>(kCameras));
    const double marker_rmse =
        std::sqrt(marker_squares / static_cast<double>(kMarkers));
    EXPECT_NEAR(summary.camera_rmse_m, camera_rmse, 1e-3);
    EXPECT_NEAR(summary.control_rmse_m, marker_rmse, 1e-3);
    EXPECT_EQ(summary.control_count, static_cast<std::size_t>(kMarkers));
    EXPECT_EQ(summary.check_count, 0U);
    EXPECT_FALSE(std::isfinite(summary.check_rmse_m));
}

TEST(ReferenceTable, CheckPointLeavesTheControlRmse) {
    Chunk chunk("roles");
    chunk.set_id(0);
    const Eigen::Vector3d position(1.0, 2.0, 3.0);
    Marker control = make_marker(10, position, position);
    control.reference = position + Eigen::Vector3d(0.0, 0.0, 1.0);
    chunk.markers().push_back(control);
    Marker check = make_marker(11, position, position);
    check.reference = position + Eigen::Vector3d(0.0, 0.0, 4.0);
    set_marker_role(check, MarkerReferenceRole::CheckPoint);
    chunk.markers().push_back(check);

    const ReferenceTable table = build_reference_table(chunk, crs::CoordinateSystem());
    EXPECT_EQ(row_for(table, 10).kind, ReferenceRowKind::Marker);
    EXPECT_TRUE(row_for(table, 10).enabled);
    EXPECT_EQ(row_for(table, 11).kind, ReferenceRowKind::CheckPoint);
    EXPECT_FALSE(row_for(table, 11).enabled);

    const ReferenceSummary summary = table.summary();
    EXPECT_EQ(summary.marker_count, 2U);
    EXPECT_EQ(summary.control_count, 1U);
    EXPECT_EQ(summary.check_count, 1U);
    EXPECT_NEAR(summary.control_rmse_m, 1.0, 1e-9);
    EXPECT_NEAR(summary.check_rmse_m, 4.0, 1e-9);
    EXPECT_EQ(table.control_count(), 1U);  // check points are not control
}

TEST(ReferenceTable, DisabledRowsLeaveTheRmse) {
    Chunk chunk("disabled");
    chunk.set_id(0);
    chunk.cameras().push_back(make_camera(1, Eigen::Vector3d::Zero()));
    chunk.cameras().back().reference.location = Eigen::Vector3d(1.0, 0.0, 0.0);
    Camera disabled = make_camera(2, Eigen::Vector3d::Zero());
    disabled.enabled = false;
    disabled.reference.location = Eigen::Vector3d(100.0, 0.0, 0.0);
    chunk.cameras().push_back(disabled);

    const ReferenceTable table = build_reference_table(chunk, crs::CoordinateSystem());
    const ReferenceSummary summary = table.summary();
    EXPECT_EQ(summary.camera_count, 2U);
    EXPECT_NEAR(summary.camera_rmse_m, 1.0, 1e-9);
    EXPECT_FALSE(row_for(table, 2).estimated.has_value());
}

TEST(ReferenceTable, CameraCheckboxMeansReferenceEnabled) {
    Chunk chunk("reference flag");
    chunk.set_id(0);
    Camera used = make_camera(1, Eigen::Vector3d::Zero());
    used.reference.location = Eigen::Vector3d(1.0, 0.0, 0.0);
    used.reference.enabled = true;
    chunk.cameras().push_back(used);
    // Aligned but its reference is switched off: the estimate still appears,
    // the row is unchecked and it leaves the RMSE.
    Camera ignored = make_camera(2, Eigen::Vector3d::Zero());
    ignored.reference.location = Eigen::Vector3d(100.0, 0.0, 0.0);
    ignored.reference.enabled = false;
    chunk.cameras().push_back(ignored);
    // Reference on, but not aligned: no estimate, so no error either.
    Camera unaligned = make_camera(3, Eigen::Vector3d::Zero());
    unaligned.enabled = false;
    unaligned.reference.location = Eigen::Vector3d(50.0, 0.0, 0.0);
    chunk.cameras().push_back(unaligned);

    const ReferenceTable table = build_reference_table(chunk, crs::CoordinateSystem());
    const ReferenceRow& off = row_for(table, 2);
    EXPECT_FALSE(off.enabled);
    ASSERT_TRUE(off.estimated.has_value());
    EXPECT_TRUE(off.error_m.has_value());
    const ReferenceRow& aligned_only = row_for(table, 3);
    EXPECT_TRUE(aligned_only.enabled);
    EXPECT_FALSE(aligned_only.estimated.has_value());

    const ReferenceSummary summary = table.summary();
    EXPECT_EQ(summary.camera_count, 3U);
    EXPECT_NEAR(summary.camera_rmse_m, 1.0, 1e-9);
}

TEST(ReferenceTable, AntennaLeverArmMatchesTheEstimate) {
    Chunk chunk("lever arm");
    chunk.set_id(0);
    chunk.set_coordinate_system(crs::CoordinateSystem::from_epsg(4979));
    const geo::Geodetic origin{10.0, 50.0, 100.0};
    const Eigen::Vector3d anchor = geo::geodetic_to_ecef(origin);
    chunk.transform().block<3, 1>(0, 3) = anchor;

    const Eigen::Vector3d lever(1.0, 2.0, 3.0);
    Sensor sensor;
    sensor.antenna.offset_m = lever;
    chunk.sensors().push_back(sensor);

    // Identity pose: R is identity, so the antenna sits at C + lever.
    Camera camera = make_camera(1, Eigen::Vector3d::Zero());
    camera.sensor_id = 0;
    const geo::Geodetic measured = geo::ecef_to_geodetic(anchor + lever);
    camera.reference.location =
        Eigen::Vector3d(measured.longitude_deg, measured.latitude_deg, measured.height_m);
    chunk.cameras().push_back(camera);

    const ReferenceTable table =
        build_reference_table(chunk, chunk.coordinate_system());
    const ReferenceRow& row = row_for(table, 1);
    ASSERT_TRUE(row.estimated.has_value());
    ASSERT_TRUE(row.error_m.has_value());
    // The reference measures the antenna, so the residual vanishes when the
    // lever arm is applied.
    EXPECT_NEAR(*row.error_m, 0.0, 1e-6);
}

TEST(ReferenceTable, GeographicErrorIsInMetres) {
    Chunk chunk("geo");
    chunk.set_id(0);
    chunk.set_coordinate_system(crs::CoordinateSystem::from_epsg(4979));
    const geo::Geodetic origin{10.0, 50.0, 100.0};
    chunk.transform().block<3, 1>(0, 3) = geo::geodetic_to_ecef(origin);

    Camera camera = make_camera(1, Eigen::Vector3d::Zero());
    const Eigen::Vector3d measured(origin.longitude_deg + 1e-5,
                                   origin.latitude_deg, origin.height_m);
    camera.reference.location = measured;
    camera.reference.accuracy = Eigen::Vector3d::Constant(0.02);
    chunk.cameras().push_back(camera);

    const ReferenceTable table =
        build_reference_table(chunk, chunk.coordinate_system());
    EXPECT_TRUE(table.georeferenced);
    const ReferenceRow& row = row_for(table, 1);
    ASSERT_TRUE(row.source.has_value());
    ASSERT_TRUE(row.estimated.has_value());
    // The estimate lands on the anchor in degrees, not on the measured point.
    EXPECT_NEAR(row.estimated->x(), origin.longitude_deg, 1e-9);
    ASSERT_TRUE(row.error_m.has_value());
    const double expected =
        geo::geodetic_to_enu(geo::Geodetic{measured.x(), measured.y(), measured.z()},
                             origin)
            .norm();
    EXPECT_GT(expected, 0.1);
    // The error is metres, so it can never look like a degree difference.
    EXPECT_NEAR(*row.error_m, expected, 0.01);
    EXPECT_GT(*row.error_m, 0.1);
    EXPECT_NEAR(table.summary().camera_rmse_m, expected, 0.01);
}

TEST(ReferenceTable, GeographicErrorIgnoresTheSimilarityScale) {
    Chunk chunk("scaled geo");
    chunk.set_id(0);
    chunk.set_coordinate_system(crs::CoordinateSystem::from_epsg(4979));
    const geo::Geodetic origin{10.0, 50.0, 100.0};
    const Eigen::Vector3d anchor = geo::geodetic_to_ecef(origin);
    // A non-unit similarity: the internal linear part is 2.5*I, not a rotation.
    chunk.transform().setIdentity();
    chunk.transform().block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 2.5;
    chunk.transform().block<3, 1>(0, 3) = anchor;

    Camera camera = make_camera(1, Eigen::Vector3d::Zero());
    const geo::Geodetic measured{origin.longitude_deg + 1e-5, origin.latitude_deg,
                                 origin.height_m};
    camera.reference.location =
        Eigen::Vector3d(measured.longitude_deg, measured.latitude_deg, measured.height_m);
    chunk.cameras().push_back(camera);

    const ReferenceTable table =
        build_reference_table(chunk, chunk.coordinate_system());
    const ReferenceRow& row = row_for(table, 1);
    ASSERT_TRUE(row.error.has_value());
    ASSERT_TRUE(row.error_m.has_value());
    // estimated - source, in ENU at the source: the estimate sits at the
    // anchor, west of the measured point.
    const Eigen::Vector3d expected = -geo::geodetic_to_enu(measured, origin);
    // Orthonormal ENU metres: neither S nor S^2 may leak into the residual.
    EXPECT_NEAR(row.error->x(), expected.x(), 1e-6);
    EXPECT_NEAR(row.error->y(), expected.y(), 1e-6);
    EXPECT_NEAR(row.error->z(), expected.z(), 1e-6);
    EXPECT_NEAR(*row.error_m, expected.norm(), 1e-6);
}

}  // namespace
}  // namespace alphashape::reference
