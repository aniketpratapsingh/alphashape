// test_surface_frame.cpp -- SurfaceFrame over the CRS engine (M12 §4.9,
// WP-12.8).
//
// The retired test_projection.cpp covered the hand-written Transverse Mercator
// series (TransverseMercator.*) and the geo::Projection selector; both are gone
// (WP-12.8) and their accuracy claims now live in the Crs.* suites
// (test_crs_transform.cpp, test_local_frame.cpp). What remains here is the
// SurfaceFrame contract the DEM/ortho code depends on:
//
//   * auto-UTM at the chunk-local origin, with the height preserved,
//   * exact local -> plane -> local round trips,
//   * the identity frame for an unreferenced chunk,
//   * a non-UTM target (EPSG:3035, LAEA) building and round-tripping,
//   * the metres-per-plane-unit conversion for a geographic plane.

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/algorithm/elevation/surface_frame.hpp"
#include "common/core/chunk.hpp"
#include "model/crs/coordinate_system.hpp"
#include "model/crs/transform.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape {
namespace {

constexpr double kDeg2Rad = 3.14159265358979323846 / 180.0;

/// Build a chunk georeferenced at `origin`: its transform maps local ENU metres
/// at that origin into ECEF (the convention Chunk::transform() documents).
Chunk georeferenced_chunk(const geo::Geodetic& origin) {
    Chunk chunk("geo");
    chunk.transform() = geo::enu_to_ecef(origin);
    return chunk;
}

TEST(SurfaceFrame, AutoUtmAndOriginHeight) {
    const geo::Geodetic origin{9.2, 48.7, 300.0};
    const Chunk chunk = georeferenced_chunk(origin);
    const elevation::SurfaceFrame frame = elevation::SurfaceFrame::for_chunk(chunk);

    // Auto zone at local (0,0,0) -> the UTM zone of the origin (32N here).
    EXPECT_FALSE(frame.is_local());
    EXPECT_EQ(frame.epsg(), 32632);

    // The local origin carries onto its own geodetic position: the plane
    // coordinate is PROJ's UTM easting/northing there and the height is the
    // ellipsoidal height, unchanged.
    const Eigen::Vector3d plane = frame.to_plane(Eigen::Vector3d::Zero());
    const crs::CoordinateSystem utm = crs::CoordinateSystem::from_epsg(32632);
    const crs::CrsTransform to_utm =
        crs::CrsTransform::create(crs::CoordinateSystem::from_epsg(4979), utm);
    const crs::CrsPoint expected =
        *to_utm.forward(crs::CrsPoint{origin.longitude_deg, origin.latitude_deg,
                                      origin.height_m});
    EXPECT_NEAR(plane.x(), expected.x, 1e-4);
    EXPECT_NEAR(plane.y(), expected.y, 1e-4);
    EXPECT_NEAR(plane.z(), 300.0, 1e-6);

    // And back to the local origin.
    const Eigen::Vector3d back = frame.to_local(plane);
    EXPECT_LT(back.norm(), 1e-6);
}

TEST(SurfaceFrame, EastOffsetAndHeightGain) {
    const geo::Geodetic origin{9.2, 48.7, 300.0};
    const Chunk chunk = georeferenced_chunk(origin);
    const elevation::SurfaceFrame frame = elevation::SurfaceFrame::for_chunk(chunk, 32632);

    const Eigen::Vector3d plane = frame.to_plane(Eigen::Vector3d(100.0, 0.0, 0.0));
    const Eigen::Vector3d origin_plane = frame.to_plane(Eigen::Vector3d::Zero());

    // Moving 100 m east on the curved earth lifts the ellipsoidal height by
    // roughly 100^2 / (2 R) and changes easting by close to 100 m.
    const double d_east = plane.x() - origin_plane.x();
    EXPECT_GT(d_east, 99.5);
    EXPECT_LT(d_east, 100.5);

    const double expected = 300.0 + 100.0 * 100.0 / (2.0 * 6371000.0);
    EXPECT_NEAR(plane.z(), expected, 0.05);
}

TEST(SurfaceFrame, RoundTripLocalPlaneExact) {
    const geo::Geodetic origin{9.2, 48.7, 300.0};
    const Chunk chunk = georeferenced_chunk(origin);
    const elevation::SurfaceFrame frame = elevation::SurfaceFrame::for_chunk(chunk, 32632);

    // The spec's SurfaceFrameRoundTripExact bound: to_local(to_plane(p)) within
    // 1e-9 m over the whole grid-scale neighbourhood.
    for (int i = 0; i < 20; ++i) {
        const double a = 137.0 * static_cast<double>(i) * kDeg2Rad;
        const double r = 1500.0 * static_cast<double>(i) / 19.0;
        const Eigen::Vector3d local(r * std::cos(a), r * std::sin(a), 10.0 * std::sin(a));
        const Eigen::Vector3d plane = frame.to_plane(local);
        const Eigen::Vector3d back = frame.to_local(plane);
        EXPECT_LT((back - local).norm(), 1e-9) << "i=" << i;
    }
}

TEST(SurfaceFrame, LaeaTargetIsExact) {
    // EPSG:3035 (ETRS89 / LAEA Europe) is not a UTM zone and could not be named
    // before the CRS engine. It must build and round-trip just as exactly.
    const geo::Geodetic origin{9.2, 48.7, 300.0};
    const Chunk chunk = georeferenced_chunk(origin);
    const elevation::SurfaceFrame frame = elevation::SurfaceFrame::for_chunk(chunk, 3035);
    ASSERT_FALSE(frame.is_local());
    EXPECT_EQ(frame.epsg(), 3035);

    const Eigen::Vector3d plane = frame.to_plane(Eigen::Vector3d(120.0, -80.0, 15.0));
    const Eigen::Vector3d back = frame.to_local(plane);
    EXPECT_LT((back - Eigen::Vector3d(120.0, -80.0, 15.0)).norm(), 1e-9);

    // A projected plane is metric on both axes whatever the projection.
    EXPECT_DOUBLE_EQ(frame.metres_per_plane_unit(plane.y()).x(), 1.0);
    EXPECT_DOUBLE_EQ(frame.metres_per_plane_unit(plane.y()).y(), 1.0);
}

TEST(SurfaceFrame, GeographicPlaneMetresPerDegree) {
    const geo::Geodetic origin{9.2, 48.7, 300.0};
    const Chunk chunk = georeferenced_chunk(origin);
    const elevation::SurfaceFrame frame = elevation::SurfaceFrame::for_chunk(chunk, 4326);
    ASSERT_EQ(frame.epsg(), 4326);

    // One degree of latitude is ~111.1 km; of longitude ~73.5 km at 48.7 N.
    const Eigen::Vector2d per_degree = frame.metres_per_plane_unit(48.7);
    EXPECT_NEAR(per_degree.y(), 111132.0, 500.0);
    EXPECT_NEAR(per_degree.x(), 73500.0, 800.0);

    // The plane itself is lon/lat in degrees.
    const Eigen::Vector3d plane = frame.to_plane(Eigen::Vector3d::Zero());
    EXPECT_NEAR(plane.x(), 9.2, 1e-9);
    EXPECT_NEAR(plane.y(), 48.7, 1e-9);
    EXPECT_NEAR(plane.z(), 300.0, 1e-6);
}

TEST(SurfaceFrame, UnreferencedIsIdentity) {
    Chunk chunk("plain");  // identity transform
    const elevation::SurfaceFrame frame = elevation::SurfaceFrame::for_chunk(chunk);
    EXPECT_TRUE(frame.is_local());
    EXPECT_EQ(frame.epsg(), 0);

    const Eigen::Vector3d p(12.0, -34.0, 56.0);
    EXPECT_LT((frame.to_plane(p) - p).norm(), 1e-12);
    EXPECT_LT((frame.to_local(p) - p).norm(), 1e-12);

    // A real CRS on an unreferenced chunk is meaningless -> ValueError.
    EXPECT_THROW(elevation::SurfaceFrame::for_chunk(chunk, 32632), ValueError);
    EXPECT_THROW(elevation::SurfaceFrame::for_chunk(chunk, crs::CoordinateSystem::from_epsg(4326)),
                 ValueError);
}

TEST(SurfaceFrame, InvalidEpsgRejected) {
    // M12 WP-12.8: an EPSG code PROJ does not know is a ValueError, not a
    // silently ignored request.
    const geo::Geodetic origin{9.2, 48.7, 300.0};
    const Chunk chunk = georeferenced_chunk(origin);
    EXPECT_THROW(elevation::SurfaceFrame::for_chunk(chunk, 999999), ValueError);
    EXPECT_THROW(crs::CoordinateSystem::from_epsg(999999), ValueError);
}

}  // namespace
}  // namespace alphashape
