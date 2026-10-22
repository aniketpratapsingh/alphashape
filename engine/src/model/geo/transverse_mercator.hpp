#pragma once

/// @file transverse_mercator.hpp
/// @brief Transverse Mercator and UTM projection, Krüger series (order n^3).
///
/// Forward and inverse use the Krüger series expansion carried to order n^3
/// following Karney (2011) as summarised on Wikipedia's
/// "Universal Transverse Mercator coordinate system" page. Accuracy over a
/// UTM zone (|lon - lon0| <= 3 degrees, |lat| <= 84 degrees) is well below
/// one millimetre in the northing; the round-trip forward(inverse(.)) is
/// refined with Newton iterations so it is exact to < 1e-6 m.
///
/// No Norway (zone 32V) or Svalbard (31X/33X/35X/37X) exceptions are applied:
/// zone selection is always the regular 6-degree rule. This is documented
/// deliberately -- callers needing the official zone may clamp themselves.
///
/// Coordinates are always in metres. Latitudes/longitudes are degrees.

#include <optional>

#include <Eigen/Core>

namespace alphashape::geo {

/// A UTM zone: 1..60 plus hemisphere. epsg() yields 32600 + zone (north)
/// or 32700 + zone (south).
struct UtmZone {
    int zone = 0;
    bool north = true;

    /// EPSG code for this zone (32600 + zone north, 32700 + zone south).
    int epsg() const;
};

/// UTM zone covering (longitude_deg, latitude_deg) using the regular
/// 6-degree rule; no Norway/Svalbard exceptions. Longitude is normalised to
/// [-180, 180); the zone is clamped to 1..60. The hemisphere comes from the
/// sign of the latitude.
UtmZone utm_zone_for(double longitude_deg, double latitude_deg);

/// Decode a WGS84 UTM EPSG code (326xx/327xx) into a UtmZone. Returns
/// std::nullopt for any other code.
std::optional<UtmZone> utm_zone_from_epsg(int epsg);

/// Forward Transverse Mercator. Returns (easting, northing) in metres.
Eigen::Vector2d tm_forward(double lon_deg, double lat_deg, double lon0_deg,
                           double k0, double false_e, double false_n);

/// Inverse Transverse Mercator. Returns (longitude_deg, latitude_deg).
Eigen::Vector2d tm_inverse(double easting, double northing, double lon0_deg,
                           double k0, double false_e, double false_n);

/// Point scale factor k at (lon_deg, lat_deg) for a projection with the given
/// central meridian and scale factor.
double tm_point_scale(double lon_deg, double lat_deg, double lon0_deg,
                      double k0);

/// Forward geodetic -> UTM (easting, northing) using k0/false E/N for the
/// zone, including the southern-hemisphere false northing.
Eigen::Vector2d geodetic_to_utm(double lon_deg, double lat_deg,
                                const UtmZone& zone);

/// Inverse UTM -> geodetic (longitude_deg, latitude_deg).
Eigen::Vector2d utm_to_geodetic(double easting, double northing,
                                const UtmZone& zone);

}  // namespace alphashape::geo
