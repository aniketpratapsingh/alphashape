#pragma once

// geodesy.hpp -- WGS84 geodetic <-> ECEF <-> local ENU conversions.
//
// Clean-room implementation from published formulas only (no PROJ, no external
// geodesy library):
//   * WGS84 defining constants: a = 6378137 m, f = 1 / 298.257223563.
//   * geodetic -> ECEF: the standard closed form (see e.g. the U.S. National
//     Imagery and Mapping Agency TR8350.2, "Department of Defense World
//     Geodetic System 1984", section 2.4.2.1).
//   * ECEF -> geodetic: Bowring (1976), "Transformation from spatial to
//     geographical coordinates", Survey Review 23(181), iterated to
//     convergence -- a fixed-point iteration on the reduced latitude. We
//     iterate rather than using the single non-iterated Bowring step because
//     the low-altitude ellipsoid surface is where the closed form loses
//     accuracy; the iteration converges to machine precision in a few passes.
//   * ENU: the standard east-north-up tangent frame built from the geodetic
//     latitude/longitude of the origin (TR8350.2, section 2.4.2.2).
//
// Angles are degrees, heights are metres (ellipsoidal height, above the WGS84
// ellipsoid -- not the geoid).

#include <Eigen/Core>

namespace alphashape::geo {

/// Geodetic position: longitude and latitude in degrees, ellipsoidal height
/// above the WGS84 ellipsoid in metres.
struct Geodetic {
    double longitude_deg = 0.0;
    double latitude_deg = 0.0;
    double height_m = 0.0;
};

inline constexpr double kWgs84A = 6378137.0;
inline constexpr double kWgs84F = 1.0 / 298.257223563;

/// Geodetic (lon, lat deg; ellipsoidal height m) -> geocentric ECEF (m).
Eigen::Vector3d geodetic_to_ecef(const Geodetic& g);

/// Geocentric ECEF (m) -> geodetic (Bowring's method, iterated to convergence).
Geodetic ecef_to_geodetic(const Eigen::Vector3d& ecef);

/// Rigid 4x4 mapping local ENU coordinates (x east, y north, z up, metres) at
/// `origin` to ECEF: ecef = M * [enu; 1].
Eigen::Matrix4d enu_to_ecef(const Geodetic& origin);

/// Inverse of enu_to_ecef(origin): enu = M * [ecef; 1].
Eigen::Matrix4d ecef_to_enu(const Geodetic& origin);

/// Convenience: ENU coordinates of `point` in the frame centred on `origin`.
Eigen::Vector3d geodetic_to_enu(const Geodetic& point, const Geodetic& origin);

/// Convenience: geodetic position of a local ENU coordinate in `origin`'s frame.
Geodetic enu_to_geodetic(const Eigen::Vector3d& enu, const Geodetic& origin);

}  // namespace alphashape::geo
