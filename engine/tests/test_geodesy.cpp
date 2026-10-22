// Tests for WGS84 geodesy: geodetic <-> ECEF, and the local ENU tangent frame.
#include "model/geo/geodesy.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace alphashape::geo {
namespace {

TEST(Geodesy, EquatorPrimeMeridian) {
    const Eigen::Vector3d e = geodetic_to_ecef(Geodetic{0.0, 0.0, 0.0});
    EXPECT_NEAR(e.x(), kWgs84A, 1e-6);
    EXPECT_NEAR(e.y(), 0.0, 1e-6);
    EXPECT_NEAR(e.z(), 0.0, 1e-6);
}

TEST(Geodesy, NorthPole) {
    const Eigen::Vector3d e = geodetic_to_ecef(Geodetic{0.0, 90.0, 0.0});
    const double b = kWgs84A * (1.0 - kWgs84F);
    EXPECT_NEAR(e.x(), 0.0, 1e-6);
    EXPECT_NEAR(e.y(), 0.0, 1e-6);
    EXPECT_NEAR(e.z(), b, 1e-6);
}

TEST(Geodesy, NinetyEast) {
    const Eigen::Vector3d e = geodetic_to_ecef(Geodetic{90.0, 0.0, 0.0});
    EXPECT_NEAR(e.x(), 0.0, 1e-6);
    EXPECT_NEAR(e.y(), kWgs84A, 1e-6);
    EXPECT_NEAR(e.z(), 0.0, 1e-6);
}

TEST(Geodesy, RoundTripGrid) {
    double worst_angle = 0.0;
    double worst_height = 0.0;
    for (double lat = -89.9; lat <= 89.9; lat += 7.3) {
        for (double lon = -180.0; lon < 180.0; lon += 23.0) {
            for (double h = -500.0; h <= 9000.0; h += 1373.0) {
                const Geodetic g{lon, lat, h};
                const Eigen::Vector3d e = geodetic_to_ecef(g);
                const Geodetic r = ecef_to_geodetic(e);
                double dlon = std::abs(r.longitude_deg - lon);
                if (dlon > 180.0) {
                    dlon = 360.0 - dlon;
                }
                worst_angle = std::max(worst_angle, dlon);
                worst_angle = std::max(worst_angle, std::abs(r.latitude_deg - lat));
                worst_height = std::max(worst_height, std::abs(r.height_m - h));
            }
        }
    }
    EXPECT_LT(worst_angle, 1e-11) << "worst angle error";
    EXPECT_LT(worst_height, 1e-6) << "worst height error";
}

TEST(Geodesy, EnuOriginMapsToEcefOrigin) {
    const Geodetic origin{11.5, 48.2, 520.0};
    const Eigen::Matrix4d m = enu_to_ecef(origin);
    const Eigen::Vector3d e = (m * Eigen::Vector4d(0.0, 0.0, 0.0, 1.0)).head<3>();
    EXPECT_LT((e - geodetic_to_ecef(origin)).norm(), 1e-9);
}

TEST(Geodesy, EnuRotationIsProper) {
    const Geodetic origin{-70.0, -33.0, 100.0};
    const Eigen::Matrix3d r = enu_to_ecef(origin).topLeftCorner<3, 3>();
    EXPECT_LT((r.transpose() * r - Eigen::Matrix3d::Identity()).norm(), 1e-12);
    EXPECT_NEAR(r.determinant(), 1.0, 1e-12);
}

TEST(Geodesy, HundredMetresNorth) {
    const Geodetic origin{10.0, 45.0, 0.0};
    const Geodetic moved = enu_to_geodetic(Eigen::Vector3d(0.0, 100.0, 0.0), origin);
    const double dlat = moved.latitude_deg - origin.latitude_deg;
    const double expected = 100.0 / 111132.0;
    EXPECT_NEAR(dlat, expected, 1e-3 * expected);
    EXPECT_NEAR(moved.longitude_deg, origin.longitude_deg, 1e-9);
}

TEST(Geodesy, EnuRoundTrip) {
    const Geodetic origin{-122.4, 37.8, 12.0};
    const std::vector<Eigen::Vector3d> pts{
        {0.0, 0.0, 0.0}, {1000.0, -2000.0, 500.0}, {-3000.0, 4000.0, -800.0}, {100.0, 100.0, 0.0}};
    for (const Eigen::Vector3d& p : pts) {
        const Geodetic g = enu_to_geodetic(p, origin);
        const Eigen::Vector3d back = geodetic_to_enu(g, origin);
        EXPECT_LT((back - p).norm(), 1e-6) << "enu " << p.transpose();
    }
}

TEST(Geodesy, UpIsEllipsoidNormal) {
    const Geodetic origin{140.0, -25.0, 30.0};
    const Geodetic above{origin.longitude_deg, origin.latitude_deg, origin.height_m + 50.0};
    const Eigen::Vector3d enu = geodetic_to_enu(above, origin);
    EXPECT_LT((enu - Eigen::Vector3d(0.0, 0.0, 50.0)).norm(), 1e-6);
}

}  // namespace
}  // namespace alphashape::geo
