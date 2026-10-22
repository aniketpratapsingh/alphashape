#include <cmath>
#include <limits>

#include <Eigen/SVD>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/algorithm/elevation/surface_frame.hpp"
#include "common/core/chunk.hpp"
#include "model/geo/geodesy.hpp"
#include "model/geo/projection.hpp"
#include "model/geo/transverse_mercator.hpp"

namespace alphashape::geo {
namespace {

constexpr double kDeg2Rad = 3.14159265358979323846 / 180.0;

// WGS84 derived quantities for the conformality check.
double ell_e2() { return kWgs84F * (2.0 - kWgs84F); }

double prime_vertical_radius(double lat_rad) {
    return kWgs84A / std::sqrt(1.0 - ell_e2() * std::sin(lat_rad) * std::sin(lat_rad));
}

double meridian_radius(double lat_rad) {
    const double w = 1.0 - ell_e2() * std::sin(lat_rad) * std::sin(lat_rad);
    return kWgs84A * (1.0 - ell_e2()) / std::pow(w, 1.5);
}

TEST(TransverseMercator, MeridianArc) {
    // On the central meridian the northing equals k0 times the true meridian
    // arc; easting stays at the false easting.
    double lon0 = 9.0;
    Eigen::Vector2d e30 = tm_forward(lon0, 30.0, lon0, 0.9996, 500000.0, 0.0);
    Eigen::Vector2d e45 = tm_forward(lon0, 45.0, lon0, 0.9996, 500000.0, 0.0);
    Eigen::Vector2d e0 = tm_forward(lon0, 0.0, lon0, 0.9996, 500000.0, 0.0);

    EXPECT_NEAR(e45.x(), 500000.0, 1e-6);
    EXPECT_NEAR(e45.y(), 0.9996 * 4984944.378, 1e-3);
    EXPECT_NEAR(e30.x(), 500000.0, 1e-6);
    EXPECT_NEAR(e0.x(), 500000.0, 1e-6);
    EXPECT_NEAR(e0.y(), 0.0, 1e-9);
    EXPECT_LT(e30.y(), e45.y());
}

TEST(TransverseMercator, RoundTrip) {
    // forward(inverse(E, N)) must recover (E, N) to sub-micrometre and the
    // recovered lon/lat to better than 1e-9 degrees.
    double lon0 = 9.0;
    for (double dlon = -3.5; dlon <= 3.5 + 1e-9; dlon += 0.5) {
        for (double lat = -80.0; lat <= 84.0 + 1e-9; lat += 4.0) {
            const double lon = lon0 + dlon;
            const Eigen::Vector2d en = tm_forward(lon, lat, lon0, 0.9996, 500000.0, 0.0);
            const Eigen::Vector2d ll = tm_inverse(en.x(), en.y(), lon0, 0.9996, 500000.0, 0.0);
            EXPECT_NEAR(ll.x(), lon, 1e-9) << "dlon=" << dlon << " lat=" << lat;
            EXPECT_NEAR(ll.y(), lat, 1e-9) << "dlon=" << dlon << " lat=" << lat;
            const Eigen::Vector2d en2 = tm_forward(ll.x(), ll.y(), lon0, 0.9996, 500000.0, 0.0);
            EXPECT_LT((en2 - en).norm(), 1e-6) << "dlon=" << dlon << " lat=" << lat;
        }
    }
}

TEST(TransverseMercator, Conformality) {
    // The numerical Jacobian of (E, N) w.r.t. ground metres must be a scaled
    // rotation: both singular values equal, and equal to tm_point_scale.
    const double lon0 = 9.0;
    const double k0 = 0.9996;
    // Ground-metre finite-difference step. The coordinates are ~1e6-1e7 m, so
    // a step of 1e-7 m would lose ~0.1% to cancellation in the differencing;
    // 10 m keeps roundoff at ~1e-10 relative while the truncation error of the
    // central difference (~h^2 * k / R^2) stays around 1e-19.
    constexpr double h = 10.0;
    const double lats[5] = {-70.0, -30.0, 0.0, 30.0, 70.0};
    const double dlons[2] = {-2.5, 2.0};
    for (double lat_deg : lats) {
        for (double dlon : dlons) {
            const double lat_rad = lat_deg * kDeg2Rad;
            const double lon = lon0 + dlon;
            const double n_rad = prime_vertical_radius(lat_rad);
            const double m_rad = meridian_radius(lat_rad);
            // Ground-metre perturbations: east = dlam * N cos(lat), north = dphi * M.
            // Convert the radian offsets to degrees before feeding the forward map.
            const double dlon_deg = (h / (n_rad * std::cos(lat_rad))) / kDeg2Rad;
            const double dlat_deg = (h / m_rad) / kDeg2Rad;

            const double base_lon = lon;
            const double base_lat = lat_deg;
            // Central differences in ground metres.
            const Eigen::Vector2d ep = tm_forward(base_lon + dlon_deg, base_lat, lon0, k0, 500000.0, 0.0);
            const Eigen::Vector2d em = tm_forward(base_lon - dlon_deg, base_lat, lon0, k0, 500000.0, 0.0);
            const Eigen::Vector2d np = tm_forward(base_lon, base_lat + dlat_deg, lon0, k0, 500000.0, 0.0);
            const Eigen::Vector2d nm = tm_forward(base_lon, base_lat - dlat_deg, lon0, k0, 500000.0, 0.0);

            // dE/dE_ground etc. Column 0 = east direction, column 1 = north.
            Eigen::Matrix2d jac;
            jac(0, 0) = (ep.x() - em.x()) / (2.0 * h);
            jac(1, 0) = (ep.y() - em.y()) / (2.0 * h);
            jac(0, 1) = (np.x() - nm.x()) / (2.0 * h);
            jac(1, 1) = (np.y() - nm.y()) / (2.0 * h);

            Eigen::JacobiSVD<Eigen::Matrix2d> svd(jac);
            const Eigen::Vector2d s = svd.singularValues();
            EXPECT_NEAR(s(0) / s(1), 1.0, 1e-6)
                << "lat=" << lat_deg << " dlon=" << dlon;
            const double k = tm_point_scale(lon, lat_deg, lon0, k0);
            EXPECT_NEAR(s(0), k, 1e-6) << "lat=" << lat_deg << " dlon=" << dlon;
            EXPECT_NEAR(s(1), k, 1e-6) << "lat=" << lat_deg << " dlon=" << dlon;
        }
    }
}

TEST(TransverseMercator, UtmZones) {
    const UtmZone z32 = utm_zone_for(9.0, 48.0);
    EXPECT_EQ(z32.zone, 32);
    EXPECT_TRUE(z32.north);
    EXPECT_EQ(z32.epsg(), 32632);

    const UtmZone z56 = utm_zone_for(151.0, -33.0);
    EXPECT_EQ(z56.zone, 56);
    EXPECT_FALSE(z56.north);
    EXPECT_EQ(z56.epsg(), 32756);

    // from_epsg round-trips.
    for (int epsg : {32601, 32632, 32660, 32701, 32756, 32760}) {
        const std::optional<UtmZone> z = utm_zone_from_epsg(epsg);
        ASSERT_TRUE(z.has_value()) << epsg;
        EXPECT_EQ(z->epsg(), epsg);
    }
    EXPECT_FALSE(utm_zone_from_epsg(3857).has_value());
    EXPECT_FALSE(utm_zone_from_epsg(0).has_value());
    EXPECT_FALSE(utm_zone_from_epsg(32600).has_value());
    EXPECT_FALSE(utm_zone_from_epsg(32661).has_value());
}

TEST(Projection, FromEpsgAndForward) {
    EXPECT_EQ(Projection::from_epsg(0).epsg(), 0);
    EXPECT_TRUE(Projection::from_epsg(0).kind == ProjectionKind::Local);
    EXPECT_EQ(Projection::from_epsg(4326).epsg(), 4326);
    EXPECT_EQ(Projection::from_epsg(32632).epsg(), 32632);
    EXPECT_EQ(Projection::from_epsg(32756).epsg(), 32756);

    const Projection geo = Projection::from_epsg(4326);
    const Geodetic g{9.2, 48.7, 300.0};
    const Eigen::Vector2d xy = geo.forward(g);
    EXPECT_NEAR(xy.x(), 9.2, 1e-12);
    EXPECT_NEAR(xy.y(), 48.7, 1e-12);
    const Geodetic back = geo.inverse(xy.x(), xy.y(), 300.0);
    EXPECT_NEAR(back.longitude_deg, 9.2, 1e-12);
    EXPECT_NEAR(back.latitude_deg, 48.7, 1e-12);
    EXPECT_NEAR(back.height_m, 300.0, 1e-12);

    const Projection utm = Projection::from_epsg(32632);
    const Eigen::Vector2d en = utm.forward(g);
    const Eigen::Vector2d direct =
        geodetic_to_utm(g.longitude_deg, g.latitude_deg,
                        utm_zone_for(g.longitude_deg, g.latitude_deg));
    EXPECT_LT((en - direct).norm(), 1e-9);
    const Geodetic rt = utm.inverse(en.x(), en.y(), 300.0);
    EXPECT_NEAR(rt.longitude_deg, 9.2, 1e-9);
    EXPECT_NEAR(rt.latitude_deg, 48.7, 1e-9);
    EXPECT_NEAR(rt.height_m, 300.0, 1e-12);

    EXPECT_THROW(Projection::from_epsg(3857), ValueError);
    EXPECT_THROW(Projection::from_epsg(1234), ValueError);
    EXPECT_THROW(Projection::from_epsg(0).forward(g), ValueError);
    EXPECT_THROW(Projection::from_epsg(0).inverse(1.0, 2.0, 3.0), ValueError);
}

// -- SurfaceFrame ---------------------------------------------------------

/// Build a chunk georeferenced at `origin`: its transform maps local ENU metres
/// at that origin into ECEF (the convention Chunk::transform() documents).
Chunk georeferenced_chunk(const Geodetic& origin) {
    Chunk chunk("geo");
    chunk.transform() = enu_to_ecef(origin);
    return chunk;
}

TEST(SurfaceFrame, AutoUtmAndOriginHeight) {
    const Geodetic origin{9.2, 48.7, 300.0};
    const Chunk chunk = georeferenced_chunk(origin);
    const elevation::SurfaceFrame frame = elevation::SurfaceFrame::for_chunk(chunk);

    // Auto zone at local (0,0,0) -> the UTM zone of the origin.
    EXPECT_FALSE(frame.is_local());
    EXPECT_EQ(frame.epsg(), utm_zone_for(9.2, 48.7).epsg());

    // Local origin maps onto geodetic_to_utm(origin), height preserved.
    const Eigen::Vector3d plane = frame.to_plane(Eigen::Vector3d::Zero());
    const Eigen::Vector2d en = geodetic_to_utm(origin.longitude_deg,
                                               origin.latitude_deg,
                                               utm_zone_for(9.2, 48.7));
    EXPECT_NEAR(plane.x(), en.x(), 1e-6);
    EXPECT_NEAR(plane.y(), en.y(), 1e-6);
    EXPECT_NEAR(plane.z(), 300.0, 1e-6);

    // And back to the local origin.
    const Eigen::Vector3d back = frame.to_local(plane);
    EXPECT_LT(back.norm(), 1e-6);
}

TEST(SurfaceFrame, EastOffsetAndHeightGain) {
    const Geodetic origin{9.2, 48.7, 300.0};
    const Chunk chunk = georeferenced_chunk(origin);
    const elevation::SurfaceFrame frame =
        elevation::SurfaceFrame::for_chunk(chunk, 32632);

    const Eigen::Vector3d plane = frame.to_plane(Eigen::Vector3d(100.0, 0.0, 0.0));
    const Eigen::Vector3d origin_plane =
        frame.to_plane(Eigen::Vector3d::Zero());

    // Moving 100 m east on the curved earth lifts the ellipsoidal height by
    // roughly 100^2 / (2 R) and changes easting by close to 100 m.
    const double d_east = plane.x() - origin_plane.x();
    EXPECT_GT(d_east, 99.5);
    EXPECT_LT(d_east, 100.5);

    const double expected = 300.0 + 100.0 * 100.0 / (2.0 * 6371000.0);
    EXPECT_NEAR(plane.z(), expected, 0.05);
}

TEST(SurfaceFrame, RoundTripLocalPlane) {
    const Geodetic origin{9.2, 48.7, 300.0};
    const Chunk chunk = georeferenced_chunk(origin);
    const elevation::SurfaceFrame frame =
        elevation::SurfaceFrame::for_chunk(chunk, 32632);

    for (int i = 0; i < 20; ++i) {
        const double a = 137.0 * static_cast<double>(i) * kDeg2Rad;
        const double r = 1500.0 * static_cast<double>(i) / 19.0;
        const Eigen::Vector3d local(r * std::cos(a), r * std::sin(a),
                                    10.0 * std::sin(a));
        const Eigen::Vector3d plane = frame.to_plane(local);
        const Eigen::Vector3d back = frame.to_local(plane);
        EXPECT_LT((back - local).norm(), 1e-6) << "i=" << i;
    }
}

TEST(SurfaceFrame, UnreferencedIsIdentity) {
    Chunk chunk("plain");  // identity transform
    const elevation::SurfaceFrame frame = elevation::SurfaceFrame::for_chunk(chunk);
    EXPECT_TRUE(frame.is_local());
    EXPECT_EQ(frame.epsg(), 0);

    const Eigen::Vector3d p(12.0, -34.0, 56.0);
    EXPECT_LT((frame.to_plane(p) - p).norm(), 1e-12);
    EXPECT_LT((frame.to_local(p) - p).norm(), 1e-12);

    // A projection on an unreferenced chunk is meaningless -> ValueError.
    EXPECT_THROW(elevation::SurfaceFrame::for_chunk(chunk, 32632), ValueError);
}

}  // namespace
}  // namespace alphashape::geo
