// geodesy.cpp -- WGS84 geodetic <-> ECEF <-> ENU. Clean-room, from published
// formulas only (TR8350.2 for the closed form and the ENU frame; Bowring 1976,
// iterated, for ECEF -> geodetic). No PROJ and no external geodesy library.

#include "model/geo/geodesy.hpp"

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace alphashape::geo {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg2Rad = kPi / 180.0;
constexpr double kRad2Deg = 180.0 / kPi;

// First eccentricity squared from the flattening: e^2 = 2f - f^2.
constexpr double kE2 = 2.0 * kWgs84F - kWgs84F * kWgs84F;

// The 3x3 ENU -> ECEF rotation about `origin` (columns are the east, north and
// up unit vectors expressed in ECEF).
Eigen::Matrix3d enu_rotation(const Geodetic& origin) {
    const double lon = origin.longitude_deg * kDeg2Rad;
    const double lat = origin.latitude_deg * kDeg2Rad;
    const double slon = std::sin(lon);
    const double clon = std::cos(lon);
    const double slat = std::sin(lat);
    const double clat = std::cos(lat);
    Eigen::Matrix3d r;
    r << -slon, -slat * clon, clat * clon,
          clon, -slat * slon, clat * slon,
           0.0,         clat,        slat;
    return r;
}

}  // namespace

Eigen::Vector3d geodetic_to_ecef(const Geodetic& g) {
    const double lon = g.longitude_deg * kDeg2Rad;
    const double lat = g.latitude_deg * kDeg2Rad;
    const double slat = std::sin(lat);
    const double clat = std::cos(lat);
    // Prime-vertical radius of curvature.
    const double n = kWgs84A / std::sqrt(1.0 - kE2 * slat * slat);
    const double p = (n + g.height_m) * clat;
    return Eigen::Vector3d(p * std::cos(lon), p * std::sin(lon),
                           (n * (1.0 - kE2) + g.height_m) * slat);
}

Geodetic ecef_to_geodetic(const Eigen::Vector3d& ecef) {
    const double x = ecef.x();
    const double y = ecef.y();
    const double z = ecef.z();
    const double p = std::hypot(x, y);

    Geodetic g;
    g.longitude_deg = std::atan2(y, x) * kRad2Deg;

    // Bowring's iteration: a fixed point on the geodetic latitude. The update
    // lat <- atan2(z + N e^2 sin lat, p) follows directly from
    // z = (N(1-e^2) + h) sin lat and p = (N + h) cos lat.
    double lat = std::atan2(z, p * (1.0 - kE2));  // initial guess
    double n = kWgs84A;
    for (int i = 0; i < 100; ++i) {
        const double slat = std::sin(lat);
        n = kWgs84A / std::sqrt(1.0 - kE2 * slat * slat);
        const double lat_next = std::atan2(z + n * kE2 * slat, p);
        if (std::abs(lat_next - lat) < 1e-15) {
            lat = lat_next;
            break;
        }
        lat = lat_next;
    }
    g.latitude_deg = lat * kRad2Deg;

    // Robust height formula (finite and accurate at the poles, where
    // p / cos(lat) is indeterminate): h = p cos lat + z sin lat - a sqrt(1 - e^2 sin^2 lat).
    const double slat = std::sin(lat);
    const double clat = std::cos(lat);
    g.height_m = p * clat + z * slat - kWgs84A * std::sqrt(1.0 - kE2 * slat * slat);
    return g;
}

Eigen::Matrix4d enu_to_ecef(const Geodetic& origin) {
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    m.topLeftCorner<3, 3>() = enu_rotation(origin);
    m.topRightCorner<3, 1>() = geodetic_to_ecef(origin);
    return m;
}

Eigen::Matrix4d ecef_to_enu(const Geodetic& origin) {
    const Eigen::Matrix3d r = enu_rotation(origin);
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    m.topLeftCorner<3, 3>() = r.transpose();
    m.topRightCorner<3, 1>() = -r.transpose() * geodetic_to_ecef(origin);
    return m;
}

Eigen::Vector3d geodetic_to_enu(const Geodetic& point, const Geodetic& origin) {
    const Eigen::Matrix4d m = ecef_to_enu(origin);
    const Eigen::Vector3d ecef = geodetic_to_ecef(point);
    return (m * ecef.homogeneous()).head<3>();
}

Geodetic enu_to_geodetic(const Eigen::Vector3d& enu, const Geodetic& origin) {
    const Eigen::Matrix4d m = enu_to_ecef(origin);
    const Eigen::Vector3d ecef = (m * enu.homogeneous()).head<3>();
    return ecef_to_geodetic(ecef);
}

}  // namespace alphashape::geo
