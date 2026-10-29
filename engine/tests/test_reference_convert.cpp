// WP-12.7: reference data conversion (MK-17).
//
// Three things are checked here, each against an independently computed value:
//
//   * UTM -> geographic and back is a round trip inside 1e-9 degrees, which is
//     where a projected reference has to land for a GNSS-fed import to be
//     usable;
//   * a vertical datum on the target slot applies the geoid: the conversion
//     EGM96 orthometric -> ellipsoidal differs from the horizontal-only answer
//     by exactly the fixture grid's undulation at that point;
//   * a camera heading true north in geographic gets a grid yaw equal to the
//     meridian convergence in UTM, with gamma computed from the textbook
//     formula rather than from the code path under test.
//
// The values going through convert_reference are the model's own: a camera
// reference is (lon, lat, alt) and a marker reference is the same triple
// (CameraReference::location / Marker::reference).

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "common/algorithm/reference/convert_reference.hpp"
#include "common/commands/command.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/marker.hpp"
#include "model/crs/coordinate_system.hpp"
#include "model/crs/euler.hpp"
#include "model/crs/geoid.hpp"
#include "model/crs/transform.hpp"
#include "model/crs/vertical.hpp"
#include "model/raster/geo_raster.hpp"
#include "model/raster/geotiff.hpp"

namespace alphashape::reference {
namespace {

namespace fs = std::filesystem;

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

const fs::path kFixtureDir = fs::path(__FILE__).parent_path() / "fixtures" / "crs";

crs::CoordinateSystem utm(int zone) { return crs::CoordinateSystem::from_epsg(32600 + zone); }

crs::CoordinateSystem geographic() { return crs::CoordinateSystem::from_epsg(4979); }

/// A chunk with `camera_count` cameras at `lon`/`lat` (one-metre z) and
/// `marker_count` markers at the same place, its CRS and both slots set to
/// `source`.
Chunk make_chunk(const crs::CoordinateSystem& source, int camera_count, int marker_count,
                 double lon, double lat) {
    Chunk chunk("conv");
    chunk.set_id(0);
    for (int i = 0; i < camera_count; ++i) {
        Camera camera;
        camera.id = i;
        camera.sensor_id = 0;
        camera.label = "cam" + std::to_string(i);
        camera.enabled = true;
        // Slight offsets so not every reference is the same point.
        camera.reference.location = Eigen::Vector3d(lon + 1e-4 * i, lat - 1e-4 * i, 100.0 + i);
        chunk.cameras().push_back(camera);
    }
    for (int i = 0; i < marker_count; ++i) {
        Marker marker;
        marker.id = 100 + i;
        marker.label = "gcp" + std::to_string(i);
        marker.reference = Eigen::Vector3d(lon - 1e-4 * i, lat + 1e-4 * i, 50.0 + i);
        chunk.markers().push_back(marker);
    }
    chunk.set_coordinate_system(source);
    chunk.set_camera_crs(CRS(source));
    chunk.set_marker_crs(CRS(source));
    return chunk;
}

/// The value a CRS transform produces for one point, as the test's independent
/// expectation of what convert_reference must have written.
crs::CrsPoint transform_point(const crs::CoordinateSystem& from,
                              const crs::CoordinateSystem& to, const Eigen::Vector3d& p) {
    const crs::CrsTransform transform = crs::CrsTransform::create(from, to);
    const std::optional<crs::CrsPoint> out = transform.forward(crs::CrsPoint{p.x(), p.y(), p.z()});
    if (!out) {
        throw Error("test: transform failed");
    }
    return *out;
}

TEST(Crs, ConvertReferenceUtmToGeographic) {
    const crs::CoordinateSystem start = geographic();
    Chunk chunk = make_chunk(start, 2, 2, 11.5820, 48.1351);

    const std::vector<Eigen::Vector3d> before_cameras = [&] {
        std::vector<Eigen::Vector3d> out;
        for (const Camera& c : chunk.cameras()) {
            out.push_back(*c.reference.location);
        }
        return out;
    }();
    const std::vector<Eigen::Vector3d> before_markers = [&] {
        std::vector<Eigen::Vector3d> out;
        for (const Marker& m : chunk.markers()) {
            out.push_back(*m.reference);
        }
        return out;
    }();

    convert_reference(chunk, utm(32), ReferenceKind::Both);
    for (const Camera& camera : chunk.cameras()) {
        EXPECT_NEAR((*camera.reference.location).x(), 680000.0, 50000.0);
        EXPECT_NEAR((*camera.reference.location).y(), 5330000.0, 50000.0);
    }
    // The slot follows, so the numbers and their label agree.
    EXPECT_EQ(chunk.camera_crs().epsg, 32632);
    EXPECT_EQ(chunk.marker_crs().epsg, 32632);
    // Accuracies are weights, carried unchanged.
    for (const Camera& camera : chunk.cameras()) {
        EXPECT_DOUBLE_EQ(camera.reference.accuracy.x(), 10.0);
        EXPECT_DOUBLE_EQ(camera.reference.accuracy.y(), 10.0);
        EXPECT_DOUBLE_EQ(camera.reference.accuracy.z(), 10.0);
    }
    for (const Marker& marker : chunk.markers()) {
        EXPECT_DOUBLE_EQ(marker.reference_accuracy.x(), 0.5);
        EXPECT_DOUBLE_EQ(marker.reference_accuracy.y(), 0.5);
        EXPECT_DOUBLE_EQ(marker.reference_accuracy.z(), 0.5);
    }

    // Round trip back: each value must return inside 1e-9 degrees.
    convert_reference(chunk, start, ReferenceKind::Both);
    ASSERT_EQ(chunk.cameras().size(), before_cameras.size());
    for (std::size_t i = 0; i < before_cameras.size(); ++i) {
        const Eigen::Vector3d& after = *chunk.cameras()[i].reference.location;
        EXPECT_LE(std::abs(after.x() - before_cameras[i].x()), 1e-9) << "camera " << i << " lon";
        EXPECT_LE(std::abs(after.y() - before_cameras[i].y()), 1e-9) << "camera " << i << " lat";
        EXPECT_NEAR(after.z(), before_cameras[i].z(), 1e-6) << "camera " << i << " height";
    }
    for (std::size_t i = 0; i < before_markers.size(); ++i) {
        const Eigen::Vector3d& after = *chunk.markers()[i].reference;
        EXPECT_LE(std::abs(after.x() - before_markers[i].x()), 1e-9) << "marker " << i;
        EXPECT_LE(std::abs(after.y() - before_markers[i].y()), 1e-9) << "marker " << i;
        EXPECT_NEAR(after.z(), before_markers[i].z(), 1e-6) << "marker " << i;
    }
    EXPECT_EQ(chunk.camera_crs().epsg, 4979);
}

TEST(Crs, ConvertReferenceMatchesDirectTransform) {
    const crs::CoordinateSystem start = geographic();
    Chunk chunk = make_chunk(start, 3, 0, 9.0, 50.0);
    const std::vector<Eigen::Vector3d> before = [&] {
        std::vector<Eigen::Vector3d> out;
        for (const Camera& c : chunk.cameras()) {
            out.push_back(*c.reference.location);
        }
        return out;
    }();
    convert_reference(chunk, utm(32), ReferenceKind::Cameras);
    for (std::size_t i = 0; i < before.size(); ++i) {
        const crs::CrsPoint expected = transform_point(start, utm(32), before[i]);
        const Eigen::Vector3d& got = *chunk.cameras()[i].reference.location;
        EXPECT_DOUBLE_EQ(got.x(), expected.x);
        EXPECT_DOUBLE_EQ(got.y(), expected.y);
        EXPECT_DOUBLE_EQ(got.z(), expected.z);
    }
}

// The "no GNSS altitude" state (a NaN z) survives the conversion: the
// horizontal part is converted and the height stays unknown.
TEST(Crs, ConvertReferenceKeepsMissingAltitude) {
    Chunk chunk = make_chunk(geographic(), 1, 0, 9.0, 50.0);
    chunk.cameras()[0].reference.location = Eigen::Vector3d(9.0, 50.0, std::nan(""));
    convert_reference(chunk, utm(32), ReferenceKind::Cameras);
    const Eigen::Vector3d& after = *chunk.cameras()[0].reference.location;
    EXPECT_TRUE(std::isfinite(after.x()));
    EXPECT_TRUE(std::isfinite(after.y()));
    EXPECT_TRUE(std::isnan(after.z()));
    // The horizontal answer is the one the same point gets when a height is
    // supplied: a missing height must not leak into x/y.
    const crs::CrsPoint with_height =
        transform_point(geographic(), utm(32), Eigen::Vector3d(9.0, 50.0, 0.0));
    EXPECT_NEAR(after.x(), with_height.x, 1e-6);
    EXPECT_NEAR(after.y(), with_height.y, 1e-6);
}

// A camera reference with no location at all is left alone, and the slot still
// moves: an unset reference is "not measured", not "in the old CRS".
TEST(Crs, ConvertReferenceLeavesAbsentLocationsAbsent) {
    Chunk chunk = make_chunk(geographic(), 2, 1, 9.0, 50.0);
    chunk.cameras()[1].reference.location.reset();
    convert_reference(chunk, utm(32), ReferenceKind::Both);
    EXPECT_FALSE(chunk.cameras()[1].reference.location.has_value());
    EXPECT_TRUE(chunk.cameras()[0].reference.location.has_value());
    EXPECT_TRUE(chunk.markers()[0].reference.has_value());
}

// --- chunk 2
/// The fixture geoid, opened once. `undulation` is the ground truth the
/// vertical test compares against.
std::shared_ptr<const crs::GeoidGrid> egm96() {
    static const std::shared_ptr<const crs::GeoidGrid> grid =
        crs::GeoidGrid::open((kFixtureDir / "egm96_subset.tif").string());
    return grid;
}

/// EGM96 orthometric heights on top of WGS 84 horizontal. The grid is referred
/// to WGS 84 (EPSG:4979), and the projection is a plain horizontal one; the
/// vertical datum lives beside the WKT (WP-12.5).
crs::CoordinateSystem egm96_geographic() {
    crs::VerticalCrs egm;
    egm.kind = crs::VerticalCrs::Kind::GeoidModel;
    egm.name = "EGM96 height";
    return geographic().with_vertical(egm);
}

crs::GeoidStore store_with_egm96() {
    crs::GeoidStore store;
    store.add("EGM96", "EGM96 height", (kFixtureDir / "egm96_subset.tif").string(), "custom",
              "public domain (NGA)");
    return store;
}

// EGM96 orthometric -> ellipsoidal moves a height by exactly the grid's
// undulation: H = h - N, so the ellipsoidal target is higher by N. The undulation
// is read from the fixture grid directly, not from the code path under test.
TEST(Crs, ConvertReferenceAltitudeUsesVerticalCrs) {
    const double lon = 9.0;
    const double lat = 50.0;
    const std::optional<double> n = egm96()->undulation(lon, lat);
    ASSERT_TRUE(n.has_value());
    ASSERT_GT(std::abs(*n), 1.0) << "fixture undulation is implausibly small";

    Chunk chunk = make_chunk(egm96_geographic(), 1, 1, lon, lat);
    chunk.cameras()[0].reference.location = Eigen::Vector3d(lon, lat, 300.0);
    chunk.markers()[0].reference = Eigen::Vector3d(lon, lat, 300.0);

    // Target: the same horizontal CRS, ellipsoidal vertical. With the datum
    // applied the ellipsoidal height is H + N.
    const crs::GeoidStore store = store_with_egm96();
    convert_reference(chunk, geographic(), ReferenceKind::Both, &store);
    EXPECT_NEAR((*chunk.cameras()[0].reference.location).z(), 300.0 + *n, 1e-3);
    EXPECT_NEAR((*chunk.markers()[0].reference).z(), 300.0 + *n, 1e-3);
    // The horizontal coordinates are untouched by a pure vertical change.
    EXPECT_NEAR((*chunk.cameras()[0].reference.location).x(), lon, 1e-7);
    EXPECT_NEAR((*chunk.cameras()[0].reference.location).y(), lat, 1e-7);
    // The target slot no longer carries a datum.
    EXPECT_FALSE(chunk.camera_crs().vertical.has_value());
}

// The reverse direction: ellipsoidal -> EGM96 orthometric subtracts N.
TEST(Crs, ConvertReferenceAltitudeToOrthometricSubtractsGeoid) {
    const double lon = 9.0;
    const double lat = 50.0;
    const std::optional<double> n = egm96()->undulation(lon, lat);
    ASSERT_TRUE(n.has_value());
    Chunk chunk = make_chunk(geographic(), 1, 0, lon, lat);
    chunk.cameras()[0].reference.location = Eigen::Vector3d(lon, lat, 300.0);
    const crs::GeoidStore store = store_with_egm96();
    convert_reference(chunk, egm96_geographic(), ReferenceKind::Cameras, &store);
    EXPECT_NEAR((*chunk.cameras()[0].reference.location).z(), 300.0 - *n, 1e-3);
}

// A project geoid (not an EPSG vertical CRS) with no store supplying it must
// fail loudly, naming the datum, rather than fall back to ellipsoidal heights.
TEST(Crs, ConvertReferenceNeedsGeoidForCustomDatum) {
    crs::VerticalCrs project;
    project.kind = crs::VerticalCrs::Kind::GeoidModel;
    project.name = "Project geoid 2019";
    project.grid = "project_geoid_2019.tif";
    const crs::CoordinateSystem source = geographic().with_vertical(project);
    Chunk chunk = make_chunk(source, 1, 0, 9.0, 50.0);
    try {
        convert_reference(chunk, utm(32), ReferenceKind::Cameras);
        FAIL() << "a custom geoid datum converted with no store";
    } catch (const Error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("Project geoid 2019"), std::string::npos) << message;
        EXPECT_NE(message.find("project_geoid_2019"), std::string::npos) << message;
    }
    // The chunk is untouched.
    EXPECT_DOUBLE_EQ((*chunk.cameras()[0].reference.location).x(), 9.0);
    EXPECT_EQ(chunk.camera_crs().epsg, 4979);
}

// An EPSG vertical datum always takes PROJ's compound pipeline, even when the
// store carries a grid keyed under the same id/name. The store grid here is a
// deliberately wrong constant (a flat 1000 m over the fixture's area), so the
// two routes give answers ~1000 m apart and cannot be confused.
TEST(Crs, EpsgVerticalDatumIgnoresStoreGridInFavourOfCompound) {
    const double lon = 9.0;
    const double lat = 50.0;
    const std::optional<double> n = egm96()->undulation(lon, lat);
    ASSERT_TRUE(n.has_value());

    crs::VerticalCrs egm;
    egm.kind = crs::VerticalCrs::Kind::GeoidModel;
    egm.name = "EGM96 height";  // resolves to EPSG:5773
    ASSERT_EQ(crs::lookup_vertical_crs(egm).epsg, 5773);
    const crs::CoordinateSystem source = geographic().with_vertical(egm);

    // A store whose "EGM96" entry is a wrong constant, not the real grid.
    const std::string dir = platform::unique_temp_path("alphashape_convref_badgrid_", "");
    std::error_code ec;
    fs::create_directories(dir, ec);
    ASSERT_FALSE(ec) << ec.message();
    const std::string bad = (fs::path(dir) / "bad_egm96.tif").string();
    {
        GeoRaster<float> raster;
        raster.grid.width = 5;
        raster.grid.height = 5;
        raster.grid.origin_x = 8.0;
        raster.grid.origin_y = 51.0;
        raster.grid.pixel_size_x = 1.0;
        raster.grid.pixel_size_y = 1.0;
        raster.channels = 1;
        raster.data.assign(25, 1000.0f);  // would shift heights by 1000 m
        GeoTiffOptions options;
        options.epsg = 4326;
        write_geotiff(bad, raster, options);
    }
    crs::GeoidStore store;
    store.add("EGM96", "EGM96 height", bad, "custom", "test");
    ASSERT_TRUE(store.grid("EGM96") != nullptr) << "the shadowing grid must be present";

    Chunk chunk = make_chunk(source, 1, 0, lon, lat);
    chunk.cameras()[0].reference.location = Eigen::Vector3d(lon, lat, 300.0);
    convert_reference(chunk, geographic(), ReferenceKind::Cameras, &store);
    // Compound route: h = H + N. The bad grid's 1000 m would fail this by ~1000.
    EXPECT_NEAR((*chunk.cameras()[0].reference.location).z(), 300.0 + *n, 0.05);
    fs::remove_all(dir, ec);
}

// The unset/local guard runs before the identity shortcut: LOCAL -> LOCAL must
// throw, not answer "same CRS, so identity".
TEST(Crs, FrameRotationRejectsLocalBeforeIdentityShortcut) {
    const crs::CoordinateSystem local = crs::CoordinateSystem::local();
    const crs::CrsPoint point{0.0, 0.0, 0.0};
    EXPECT_THROW(frame_rotation(local, local, point), ValueError);
    EXPECT_THROW(frame_rotation(geographic(), local, point), ValueError);
    EXPECT_THROW(frame_rotation(local, geographic(), point), ValueError);
    // The identity shortcut still applies to a real, equal CRS.
    const Eigen::Matrix3d identity = frame_rotation(geographic(), geographic(), point);
    EXPECT_TRUE(identity.isApprox(Eigen::Matrix3d::Identity(), 1e-15));
}

// --- chunk 3
/// Textbook meridian convergence in degrees, γ = atan(tan Δλ · sin φ), with Δλ
/// the longitude minus the zone's central meridian and φ the latitude. Positive
/// east of the central meridian: grid north lies γ east of true north, so a
/// direction pointing true north reads an azimuth of -γ in grid terms, and a
/// heading of zero (north) in geographic becomes a yaw of +γ in the grid frame
/// because the yaw turns +north towards -east.
double meridian_convergence(double lon_deg, double lat_deg, int zone) {
    const double central_meridian = 6.0 * zone - 183.0;
    return std::atan(std::tan((lon_deg - central_meridian) * kDeg) * std::sin(lat_deg * kDeg)) /
           kDeg;
}

// The orientation contract: FLU body frame, yaw CCW from north in the CRS's
// ENU basis, so a compass heading maps to yaw = -heading.
//
// Derivation for a nadir camera with compass heading 90 degrees (flying east):
//   body frame (level, yaw -90): x forward = east, y left = north, z up.
//   nadir mount (pitch -90): body_to_camera_nadir() rows give bore (cam +Z) =
//   body -z = down, image down (cam +Y) = body -x = -east = west, image right
//   (cam +X) = body -y = -north = south.
// The header's contract is R_wb = Rz(yaw + 90deg) * Ry(-pitch) * Rx(roll),
// i.e. Rz(90deg) * crs::to_matrix({yaw, -pitch, roll}); camera-to-world is
// R_wb * body_to_camera_nadir()^T.
TEST(Crs, NadirCameraHeadingEastMapsCompassToNegativeYaw) {
    const double heading_deg = 90.0;  // compass: clockwise from north, east
    const crs::EulerAngles angles{/*yaw=*/-heading_deg, /*pitch=*/0.0, /*roll=*/0.0,
                                 crs::RotationOrder::ZYX};
    Eigen::Matrix3d yaw_offset;
    yaw_offset << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;  // Rz(90 deg)
    // The level body at yaw -90 collapses to the ENU basis: Rz(90)*Rz(-90)=I.
    const Eigen::Matrix3d world_to_body = yaw_offset * crs::to_matrix(angles);
    EXPECT_TRUE(world_to_body.isApprox(Eigen::Matrix3d::Identity(), 1e-12))
        << "level body at compass heading east is the ENU basis";

    const Eigen::Matrix3d camera_to_world = world_to_body.transpose() *
                                            body_to_camera_nadir().transpose();
    // Image down (camera +Y) in world coordinates: body -x = -east.
    const Eigen::Vector3d image_down = camera_to_world.col(1);
    EXPECT_NEAR(image_down.x(), -1.0, 1e-12) << "image down points west (-east)";
    EXPECT_NEAR(image_down.y(), 0.0, 1e-12);
    EXPECT_NEAR(image_down.z(), 0.0, 1e-12);
    // Boresight (camera +Z) points straight down; image right is south.
    EXPECT_NEAR(camera_to_world.col(2).z(), -1.0, 1e-12);
    EXPECT_NEAR(camera_to_world.col(0).y(), -1.0, 1e-12) << "image right points south";
}

TEST(Crs, BodyToCameraIsAProperRotationForLevelAndNadir) {
    Eigen::Matrix3d ry_minus_90;
    ry_minus_90 << 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0;
    EXPECT_TRUE(body_to_camera_nadir().isApprox(body_to_camera() * ry_minus_90, 1e-15));
    for (const Eigen::Matrix3d& rotation : {body_to_camera(), body_to_camera_nadir()}) {
        EXPECT_NEAR(rotation.determinant(), 1.0, 1e-15);
        EXPECT_TRUE((rotation * rotation.transpose()).isApprox(Eigen::Matrix3d::Identity(), 1e-15));
    }
    // A level, forward-looking camera's boresight is body forward (+x); a
    // nadir camera's is body down (-z).
    EXPECT_NEAR(body_to_camera().transpose().col(2).x(), 1.0, 1e-15);
    EXPECT_NEAR(body_to_camera_nadir().transpose().col(2).z(), -1.0, 1e-15);
}

TEST(Crs, ConvertReferenceYawGainsMeridianConvergence) {
    const int zone = 32;
    const double lat = 50.0;
    const double lon = 9.5;  // 0.5 degrees east of the zone's central meridian
    const double gamma = meridian_convergence(lon, lat, zone);
    ASSERT_GT(gamma, 0.3) << "the test point must be far enough from the central meridian";

    // Independent observation of the same angle from the transform engine: the
    // grid direction of true north at the point, as an azimuth from grid north.
    const crs::CrsTransform to_grid = crs::CrsTransform::create(geographic(), utm(zone));
    const std::optional<crs::CrsPoint> here = to_grid.forward(crs::CrsPoint{lon, lat, 0.0});
    const std::optional<crs::CrsPoint> north = to_grid.forward(crs::CrsPoint{lon, lat + 1e-6, 0.0});
    const std::optional<crs::CrsPoint> east = to_grid.forward(crs::CrsPoint{lon + 1e-6, lat, 0.0});
    ASSERT_TRUE(here && north && east);
    const double d_east = north->x - here->x;
    const double d_north = north->y - here->y;
    const double metric = std::hypot(east->x - here->x, east->y - here->y);
    ASSERT_GT(metric, 0.0);
    // The grid azimuth of true north is -gamma (grid north lies gamma east of
    // true north). The textbook formula is a spherical approximation, so the
    // tight check is below; this one only pins the sign and the magnitude.
    EXPECT_NEAR(std::atan2(d_east, d_north) / kDeg, -gamma, 1e-2);

    // A camera whose body axes are east/north/up in geographic (identity
    // rotation, so its heading is true north) must read yaw = gamma in UTM.
    crs::EulerAngles heading;
    heading.order = crs::RotationOrder::ZYX;
    const crs::EulerAngles converted =
        convert_orientation(heading, geographic(), utm(zone), crs::CrsPoint{lon, lat, 0.0});
    EXPECT_EQ(converted.order, crs::RotationOrder::ZYX);
    // Independent expectation for the sign from the measured grid direction of
    // true north: it is (sin(az), cos(az)) with az the grid azimuth. A positive
    // ZYX yaw turns +y (north) towards -x (west), i.e. Rz(yaw) e_y is
    // (-sin(yaw), cos(yaw)), so reproducing that direction needs yaw = -az.
    // (Grid north lies gamma east of true north, so az = -gamma and the grid
    // yaw of a true-north heading is +gamma.)
    const double grid_azimuth = std::atan2(d_east, d_north) / kDeg;
    EXPECT_NEAR(converted.yaw, -grid_azimuth, 1e-3) << "grid yaw of a true-north heading";
    // "About up only" is the physical claim, and it is checked on the rotation
    // itself: the frame's up axis must stay up. The pitch/roll magnitudes are
    // the finite-difference residue of that (about 1e-8 deg, from the 1e-5 deg
    // step in `local_basis`), so they are bounded rather than zero.
    const Eigen::Matrix3d frame =
        frame_rotation(geographic(), utm(zone), crs::CrsPoint{lon, lat, 0.0});
    const Eigen::Vector3d up = frame * Eigen::Vector3d::UnitZ();
    EXPECT_NEAR(up.x(), 0.0, 1e-6);
    EXPECT_NEAR(up.y(), 0.0, 1e-6);
    EXPECT_NEAR(up.z(), 1.0, 1e-9);
    EXPECT_NEAR(converted.pitch, 0.0, 1e-6) << "the frame rotation is about up only";
    EXPECT_NEAR(converted.roll, 0.0, 1e-6);

    // Round trip: back to geographic, the heading is north again.
    const crs::CrsPoint there = *to_grid.forward(crs::CrsPoint{lon, lat, 0.0});
    const crs::EulerAngles back = convert_orientation(converted, utm(zone), geographic(), there);
    EXPECT_NEAR(back.yaw, 0.0, 1e-3);
}

// Two projections of the same datum differ by a real rotation of their local
// axes, not only by the convergence: a direction that is pure east in one zone
// is not pure east in the next. The frame rotation must be orthonormal and
// map up to up for a transverse Mercator pair.
TEST(Crs, FrameRotationBetweenUtmZonesIsAboutUp) {
    // The point is given in the SOURCE CRS (zone 32), so it is an easting and
    // a northing, not a longitude and a latitude.
    const Eigen::Matrix3d rotation =
        frame_rotation(utm(32), utm(33), crs::CrsPoint{500000.0, 5540000.0, 0.0});
    EXPECT_LE((rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff(),
              1e-12);
    EXPECT_NEAR(rotation.determinant(), 1.0, 1e-12);
    // Up maps to up, at the finite-difference step's own accuracy (1e-5 deg in
    // `local_basis`, so the direction is good to about 1e-8).
    const Eigen::Vector3d up = rotation * Eigen::Vector3d::UnitZ();
    EXPECT_NEAR(up.x(), 0.0, 1e-6);
    EXPECT_NEAR(up.y(), 0.0, 1e-6);
    EXPECT_NEAR(up.z(), 1.0, 1e-9);
}

TEST(Crs, FrameRotationIdentityForSameCrs) {
    const Eigen::Matrix3d rotation =
        frame_rotation(utm(32), utm(32), crs::CrsPoint{500000.0, 5500000.0, 0.0});
    EXPECT_LE((rotation - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff(), 0.0);
}

// The chunk meta key selects the order; an unknown name is an error rather than
// a silent ZYX, because the numbers would then mean something else.
TEST(Crs, RotationOrderComesFromChunkMeta) {
    Chunk chunk("orders");
    EXPECT_EQ(rotation_order_of(chunk), crs::RotationOrder::ZYX);
    chunk.meta().set(kRotationOrderKey, "XYZ");
    EXPECT_EQ(rotation_order_of(chunk), crs::RotationOrder::XYZ);
    chunk.meta().set(kRotationOrderKey, "zyx");
    EXPECT_EQ(rotation_order_of(chunk), crs::RotationOrder::ZYX);
    chunk.meta().set(kRotationOrderKey, "XYX");
    EXPECT_THROW(rotation_order_of(chunk), ValueError);
}

// --- chunk 4
// The command is one undo entry that restores every reference and both slots.
TEST(Crs, ConvertReferenceCommandUndoes) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("chunk");
    document.set_active_chunk_id(chunk.id());
    Chunk original = make_chunk(geographic(), 2, 2, 9.0, 50.0);
    chunk.cameras() = original.cameras();
    chunk.markers() = original.markers();
    chunk.set_coordinate_system(original.coordinate_system());
    chunk.set_camera_crs(original.camera_crs());
    chunk.set_marker_crs(original.marker_crs());

    const Chunk pre_state = chunk;
    UndoStack stack(&document);
    stack.execute(make_convert_reference_command(chunk.id(), utm(32), ReferenceKind::Both));
    EXPECT_EQ(chunk.camera_crs().epsg, 32632);
    EXPECT_EQ(chunk.marker_crs().epsg, 32632);
    EXPECT_NEAR((*chunk.cameras()[0].reference.location).x(), 500000.0, 20000.0);
    const Chunk after = chunk;

    stack.undo();
    EXPECT_TRUE(chunk == pre_state) << "undo did not restore the pre-state";
    stack.redo();
    EXPECT_TRUE(chunk == after) << "redo differs from the first apply";
}

// The command validates an unset/local target and a missing chunk before
// touching anything.
TEST(Crs, ConvertReferenceCommandRejectsBadTarget) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("chunk");
    document.set_active_chunk_id(chunk.id());
    UndoStack stack(&document);
    EXPECT_THROW(stack.execute(make_convert_reference_command(
                     chunk.id(), crs::CoordinateSystem::local(), ReferenceKind::Both)),
                 ValueError);
    EXPECT_THROW(stack.execute(make_convert_reference_command(999, utm(32), ReferenceKind::Both)),
                 ValueError);
    EXPECT_FALSE(stack.can_undo());
}

// A failure in the second slot must not leave the first one converted: the
// new values are computed for both slots before either is written. (The
// comparison is per field because a NaN position, which is what triggers the
// failure, never compares equal to itself.)
TEST(Crs, ConvertReferenceFailureLeavesChunkUntouched) {
    Chunk chunk = make_chunk(geographic(), 1, 1, 9.0, 50.0);
    const Eigen::Vector3d camera_before = *chunk.cameras()[0].reference.location;
    // A marker with an unmeasurable longitude makes the marker slot throw.
    chunk.markers()[0].reference = Eigen::Vector3d(std::nan(""), 50.0, 0.0);
    EXPECT_THROW(convert_reference(chunk, utm(32), ReferenceKind::Both), ValueError);
    ASSERT_TRUE(chunk.cameras()[0].reference.location.has_value());
    const Eigen::Vector3d& camera_after = *chunk.cameras()[0].reference.location;
    EXPECT_DOUBLE_EQ(camera_after.x(), camera_before.x())
        << "a failed conversion changed the camera slot";
    EXPECT_DOUBLE_EQ(camera_after.y(), camera_before.y());
    EXPECT_DOUBLE_EQ(camera_after.z(), camera_before.z());
    EXPECT_EQ(chunk.camera_crs().epsg, 4979);
    EXPECT_EQ(chunk.marker_crs().epsg, 4979);
}

}  // namespace
}  // namespace alphashape::reference
