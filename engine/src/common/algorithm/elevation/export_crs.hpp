#pragma once

// Export-time CRS handling for the DEM and orthomosaic writers (M12 §4.10,
// WP-12.9).
//
// An export writes the raster as stored unless `projection`/`crs` names a
// different coordinate system, in which case the raster is reprojected on the
// fly (model/raster/reproject.hpp) and the file is georeferenced in the target
// CRS. `resolution` is the cell size in the TARGET CRS; 0 keeps the source
// ground sample distance.
//
// The parameter rules are shared with BuildDem/BuildOrthomosaic so a script that
// works for one works for the others: `projection` is an Int EPSG code (0 =
// unset), `crs` is "EPSG:<n>", a bare code, a PROJ string or WKT, and setting
// both to disagreeing values is a ValueError.

#include <cstdint>
#include <string>

#include "common/core/types.hpp"
#include "model/raster/geo_raster.hpp"

namespace alphashape::elevation {

/// A resolved export request.
struct ExportCrs {
    /// Requested CRS; unset means "write the raster's own CRS unchanged".
    CRS crs;
    /// Cell size in the TARGET CRS. 0 keeps the source GSD.
    double resolution = 0.0;

    bool requested() const { return crs.is_set(); }
};

/// Parse an export task's `projection` / `crs` / `resolution` parameters.
/// `task` names the task in error messages ("ExportDem").
/// @throws ValueError for a negative EPSG, a negative resolution, or a
///         `projection`/`crs` pair that disagrees.
ExportCrs parse_export_crs(std::int64_t projection, const std::string& crs_text,
                           double resolution, const char* task);

/// The raster in its export form: reprojected into the requested CRS, or the
/// input unchanged (a copy) when the request is unset. `raster_crs` is the
/// container's recorded CRS.
///
/// @throws ValueError when a reprojection is requested but the raster has no
///         CRS to reproject FROM, or when the target CRS has no EPSG code (a
///         custom WKT target is written once WP-12.10 adds WKT GeoKeys), and
///         when the request cannot be satisfied at the requested cell size.
GeoRaster<float> reproject_for_export(const GeoRaster<float>& raster, const CRS& raster_crs,
                                      float nodata, const ExportCrs& target,
                                      const char* task);

GeoRaster<std::uint8_t> reproject_for_export(const GeoRaster<std::uint8_t>& raster,
                                             const CRS& raster_crs,
                                             const ExportCrs& target, const char* task);

/// The EPSG code the written GeoTIFF declares: the target's when reprojecting,
/// else the container's own (0 for the local frame).
int export_epsg(const CRS& raster_crs, const ExportCrs& target);

}  // namespace alphashape::elevation
