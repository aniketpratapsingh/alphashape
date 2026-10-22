#pragma once

/// @file projection.hpp
/// @brief A map projection selector: local plane, geographic (lon/lat) or UTM.
///
/// Projection wraps the small set of coordinate frames the engine needs for
/// DEM / orthomosaic output:
///   - Local:      an arbitrary chunk-local plane, no geographic meaning.
///   - Geographic: WGS84 longitude/latitude in degrees.
///   - Utm:        a specific WGS84 UTM zone, easting/northing in metres.

#include <Eigen/Core>

#include "model/geo/geodesy.hpp"
#include "model/geo/transverse_mercator.hpp"

namespace alphashape::geo {

enum class ProjectionKind { Local, Geographic, Utm };

struct Projection {
    ProjectionKind kind = ProjectionKind::Local;
    UtmZone zone{};

    /// Build from an EPSG code: 0 -> Local, 4326 -> Geographic,
    /// 326xx/327xx -> Utm. Any other code throws ValueError.
    static Projection from_epsg(int epsg);

    /// The EPSG code describing this projection (0 for Local).
    int epsg() const;

    /// Project a geodetic coordinate to plane (x, y) metres/degrees:
    /// Geographic -> (lon_deg, lat_deg); Utm -> (easting, northing).
    /// Local -> ValueError (no geographic meaning).
    Eigen::Vector2d forward(const Geodetic& g) const;

    /// Inverse of forward(): plane (x, y) + height_m -> Geodetic.
    /// Local -> ValueError.
    Geodetic inverse(double x, double y, double height_m) const;
};

}  // namespace alphashape::geo
