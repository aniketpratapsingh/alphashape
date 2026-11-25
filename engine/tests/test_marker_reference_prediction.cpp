// test_marker_reference_prediction.cpp -- predicting an imported control point on
// photos from its surveyed reference (P2 GUI: predict from reference).
//
// An imported GCP has a reference but no estimated `position` until it is
// adjusted. Once the chunk is georeferenced (a non-identity local -> ECEF
// transform and a real CRS), the reference can be carried into the local frame
// and projected, which is what lets the Photo tab show a predicted flag and the
// Photos pane list the photos that see the marker.
//
// The round-trip assertions push the helper's local point back through
// `chunk.transform()` and the geodetic conversion, so a wrong axis order, a
// missing inverse or a mislabelled CRS shows up as a kilometre-scale error.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "common/algorithm/markers/project_markers.hpp"
#include "common/core/chunk.hpp"
#include "common/core/marker.hpp"
#include "common/core/types.hpp"
#include "model/camera/calibration.hpp"
#include "model/crs/coordinate_system.hpp"
#include "model/crs/transform.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape {
namespace {

namespace geo = alphashape::geo;

/// A geodetic origin near Adelaide, South Australia (the block the feature was
/// requested for). The exact value does not matter; it only anchors the ENU
/// frame the test transform is built from.
const geo::Geodetic& adelaide_origin() {
    static const geo::Geodetic origin{138.6007, -34.9285, 50.0};
    return origin;
}

/// A similarity local -> ECEF: a scaled, rotated ENU frame at the origin. The
/// scale is not 1, so a helper that assumed a rigid transform would fail the
/// round trip.
Transform test_transform() {
    // local -> ENU: scale 0.5, a rotation about a tilted axis, a shift.
    Eigen::Matrix4d local_to_enu = Eigen::Matrix4d::Identity();
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.4, Eigen::Vector3d(0.3, 0.8, 0.5).normalized())
            .toRotationMatrix();
    local_to_enu.block<3, 3>(0, 0) = 0.5 * rotation;
    local_to_enu.block<3, 1>(0, 3) = Eigen::Vector3d(3.0, -2.0, 1.5);
    // ENU -> ECEF at the origin.
    return geo::enu_to_ecef(adelaide_origin()) * local_to_enu;
}

/// A chunk with a georeferenced transform and a 4326 chunk CRS, no cameras.
Chunk make_georeferenced_chunk() {
    Chunk chunk("predict");
    chunk.transform() = test_transform();
    chunk.crs().epsg = 4326;
    return chunk;
}

/// The geodetic (lon, lat, h) of a local point, through the chunk transform.
Eigen::Vector3d geodetic_of_local(const Chunk& chunk, const Eigen::Vector3d& local) {
    const Eigen::Vector3d ecef = (chunk.transform() * local.homogeneous()).head<3>();
    const geo::Geodetic g = geo::ecef_to_geodetic(ecef);
    return Eigen::Vector3d(g.longitude_deg, g.latitude_deg, g.height_m);
}

// ---------------------------------------------------------------------------
// 1. A geographic (EPSG:4326) reference round-trips through the helper
// ---------------------------------------------------------------------------

TEST(MarkerReferencePrediction, GeographicReferenceRoundTrips) {
    Chunk chunk = make_georeferenced_chunk();
    // A point 40 m east, 25 m north and 3 m up from the origin, expressed as a
    // geodetic reference.
    const Eigen::Vector3d enu(40.0, 25.0, 3.0);
    const geo::Geodetic g = geo::enu_to_geodetic(enu, adelaide_origin());
    const Eigen::Vector3d reference(g.longitude_deg, g.latitude_deg, g.height_m);

    Marker marker;
    marker.id = 0;
    marker.reference = reference;
    chunk.markers().push_back(marker);

    const std::optional<Eigen::Vector3d> local = marker_local_from_reference(chunk, 0);
    ASSERT_TRUE(local.has_value());

    // Pushing the local point back through the transform and the geodetic
    // conversion must reproduce the reference.
    const Eigen::Vector3d recovered = geodetic_of_local(chunk, *local);
    EXPECT_NEAR(recovered.x(), reference.x(), 1e-6) << "longitude";
    EXPECT_NEAR(recovered.y(), reference.y(), 1e-6) << "latitude";
    EXPECT_NEAR(recovered.z(), reference.z(), 1e-3) << "height";
}

// ---------------------------------------------------------------------------
// 2. A projected (EPSG:7854, GDA2020 / MGA zone 54) reference round-trips
// ---------------------------------------------------------------------------

TEST(MarkerReferencePrediction, ProjectedReferenceRoundTrips) {
    Chunk chunk = make_georeferenced_chunk();
    // A real Adelaide coordinate in EPSG:7854.
    const Eigen::Vector3d mga(280221.769, 6154434.822, 11.446);

    Marker marker;
    marker.id = 0;
    marker.reference = mga;
    chunk.markers().push_back(marker);
    CRS slot;
    slot.epsg = 7854;
    chunk.set_marker_crs(slot);

    const std::optional<Eigen::Vector3d> local = marker_local_from_reference(chunk, 0);
    ASSERT_TRUE(local.has_value());

    // The reference in geodetic terms, for the comparison.
    const crs::CrsTransform to_geodetic = crs::CrsTransform::create(
        crs::CoordinateSystem::from_epsg(7854), crs::CoordinateSystem::from_epsg(4979));
    const std::optional<crs::CrsPoint> expected =
        to_geodetic.forward(crs::CrsPoint{mga.x(), mga.y(), mga.z()});
    ASSERT_TRUE(expected.has_value());

    const Eigen::Vector3d recovered = geodetic_of_local(chunk, *local);
    EXPECT_NEAR(recovered.x(), expected->x, 1e-6) << "longitude";
    EXPECT_NEAR(recovered.y(), expected->y, 1e-6) << "latitude";
    EXPECT_NEAR(recovered.z(), expected->z, 1e-3) << "height";
}

// ---------------------------------------------------------------------------
// 3. A chunk that is not georeferenced predicts nothing
// ---------------------------------------------------------------------------

TEST(MarkerReferencePrediction, IdentityTransformPredictsNothing) {
    Chunk chunk("predict");
    chunk.crs().epsg = 4326;  // a CRS with an identity transform is only a label
    Marker marker;
    marker.id = 0;
    marker.reference = Eigen::Vector3d(138.6, -34.9, 50.0);
    chunk.markers().push_back(marker);

    EXPECT_FALSE(marker_local_from_reference(chunk, 0).has_value());
    const std::vector<std::optional<Eigen::Vector3d>> locals =
        marker_locals_from_references(chunk);
    ASSERT_EQ(locals.size(), 1U);
    EXPECT_FALSE(locals[0].has_value());
}

TEST(MarkerReferencePrediction, LocalCrsPredictsNothing) {
    Chunk chunk("predict");
    chunk.transform() = test_transform();
    // The default CRS is the local engineering frame: the transform alone yields
    // ECEF metres, not a chunk frame, so there is nothing to predict.
    Marker marker;
    marker.id = 0;
    marker.reference = Eigen::Vector3d(138.6, -34.9, 50.0);
    chunk.markers().push_back(marker);

    EXPECT_FALSE(marker_local_from_reference(chunk, 0).has_value());
}

TEST(MarkerReferencePrediction, MarkerWithoutReferencePredictsNothing) {
    Chunk chunk = make_georeferenced_chunk();
    Marker marker;
    marker.id = 0;  // no reference, no position
    chunk.markers().push_back(marker);

    EXPECT_FALSE(marker_local_from_reference(chunk, 0).has_value());
    // An out-of-range index is also nullopt, not a crash.
    EXPECT_FALSE(marker_local_from_reference(chunk, 5).has_value());
}

// ---------------------------------------------------------------------------
// 4. The vector form matches the single form
// ---------------------------------------------------------------------------

TEST(MarkerReferencePrediction, VectorFormMatchesSingleForm) {
    Chunk chunk = make_georeferenced_chunk();
    for (int i = 0; i < 3; ++i) {
        Marker marker;
        marker.id = i;
        const geo::Geodetic g =
            geo::enu_to_geodetic(Eigen::Vector3d(10.0 * i, 5.0 * i, 1.0), adelaide_origin());
        marker.reference = Eigen::Vector3d(g.longitude_deg, g.latitude_deg, g.height_m);
        chunk.markers().push_back(marker);
    }
    Marker no_reference;
    no_reference.id = 3;
    chunk.markers().push_back(no_reference);

    const std::vector<std::optional<Eigen::Vector3d>> locals =
        marker_locals_from_references(chunk);
    ASSERT_EQ(locals.size(), 4U);
    for (std::size_t i = 0; i < 3; ++i) {
        const std::optional<Eigen::Vector3d> single = marker_local_from_reference(chunk, i);
        ASSERT_TRUE(single.has_value());
        ASSERT_TRUE(locals[i].has_value());
        EXPECT_LT((*single - *locals[i]).norm(), 1e-9) << "marker " << i;
    }
    EXPECT_FALSE(locals[3].has_value());
}

// ---------------------------------------------------------------------------
// 5. cameras_seeing_marker lists only the cameras whose image contains it
// ---------------------------------------------------------------------------

/// A chunk with one sensor and two cameras at the origin looking down +Z, plus
/// a marker at `local`. The calibration is a 64x48 frame with f = 50 px, so a
/// point on the optical axis lands at the centre (32, 24).
Chunk make_camera_chunk(const Eigen::Vector3d& local) {
    Chunk chunk("cameras");
    Sensor sensor;
    sensor.width = 64;
    sensor.height = 48;
    sensor.calibration = Calibration::initial(CameraType::Frame, 64, 48, 50.0);
    chunk.sensors().push_back(sensor);

    for (std::int32_t i = 0; i < 2; ++i) {
        Camera camera;
        camera.id = i;
        camera.sensor_id = 0;
        camera.enabled = true;
        camera.pose = Transform::Identity();
        chunk.cameras().push_back(camera);
    }

    Marker marker;
    marker.id = 0;
    marker.position = local;
    chunk.markers().push_back(marker);
    return chunk;
}

TEST(MarkerReferencePrediction, CamerasSeeingMarkerListsInImageCameras) {
    // On the optical axis: both cameras see it, at the image centre.
    Chunk chunk = make_camera_chunk(Eigen::Vector3d(0.0, 0.0, 1.0));
    const std::vector<MarkerCameraHit> hits = cameras_seeing_marker(chunk, 0);
    ASSERT_EQ(hits.size(), 2U);
    for (const MarkerCameraHit& hit : hits) {
        EXPECT_NEAR(hit.pixel.x(), 32.0, 1e-9);
        EXPECT_NEAR(hit.pixel.y(), 24.0, 1e-9);
        EXPECT_NEAR(hit.distance_from_centre, 0.0, 1e-9);
    }
}

TEST(MarkerReferencePrediction, CamerasSeeingMarkerExcludesBehindAndOffImage) {
    // Behind the camera: no camera sees it.
    Chunk behind = make_camera_chunk(Eigen::Vector3d(0.0, 0.0, -1.0));
    EXPECT_TRUE(cameras_seeing_marker(behind, 0).empty());

    // Far off to the side: outside the 64x48 image, so clipped away.
    Chunk off = make_camera_chunk(Eigen::Vector3d(100.0, 0.0, 1.0));
    EXPECT_TRUE(cameras_seeing_marker(off, 0).empty());
}

TEST(MarkerReferencePrediction, CamerasSeeingMarkerUsesReferenceWhenNoPosition) {
    // A georeferenced chunk whose marker has only a reference. The reference is
    // placed so it lands on the optical axis of the identity-pose camera.
    Chunk chunk = make_camera_chunk(Eigen::Vector3d(0.0, 0.0, 1.0));
    // Turn the marker into a reference-only one: drop the position, set the
    // reference to the geodetic of the local point through the transform.
    chunk.transform() = test_transform();
    chunk.crs().epsg = 4326;
    const Eigen::Vector3d local(0.0, 0.0, 1.0);
    const Eigen::Vector3d reference = geodetic_of_local(chunk, local);
    chunk.markers()[0].position.reset();
    chunk.markers()[0].reference = reference;

    const std::vector<MarkerCameraHit> hits = cameras_seeing_marker(chunk, 0);
    ASSERT_EQ(hits.size(), 2U);
    for (const MarkerCameraHit& hit : hits) {
        EXPECT_NEAR(hit.pixel.x(), 32.0, 1e-3);
        EXPECT_NEAR(hit.pixel.y(), 24.0, 1e-3);
    }
}

TEST(MarkerReferencePrediction, CamerasSeeingMarkerOrdersByDistanceFromCentre) {
    // Two cameras with different poses so the marker lands off-centre in one.
    Chunk chunk = make_camera_chunk(Eigen::Vector3d(0.0, 0.0, 1.0));
    // Tilt the second camera so the marker projects away from its centre.
    chunk.cameras()[1].pose = Eigen::Matrix4d::Identity();
    chunk.cameras()[1].pose(0, 3) = 0.2;  // shift the camera right: marker moves left

    const std::vector<MarkerCameraHit> hits = cameras_seeing_marker(chunk, 0);
    ASSERT_EQ(hits.size(), 2U);
    // The first camera (identity pose) has the marker dead centre, so it comes first.
    EXPECT_EQ(hits.front().camera_id, 0);
    EXPECT_LE(hits.front().distance_from_centre, hits.back().distance_from_centre);
}

}  // namespace
}  // namespace alphashape
