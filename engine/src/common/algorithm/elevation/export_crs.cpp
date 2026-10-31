#include "common/algorithm/elevation/export_crs.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

#include "base/utils/error.hpp"
#include "common/algorithm/elevation/surface_frame.hpp"
#include "model/raster/reproject.hpp"

namespace alphashape::elevation {

ExportCrs parse_export_crs(std::int64_t projection, const std::string& crs_text,
                           double resolution, const char* task) {
    ExportCrs request;
    if (projection < 0) {
        throw ValueError(std::string(task) + ": projection must be an EPSG code >= 0");
    }
    if (!std::isfinite(resolution) || resolution < 0.0) {
        throw ValueError(std::string(task) + ": resolution must be a finite value >= 0");
    }
    request.resolution = resolution;
    if (!crs_text.empty()) {
        request.crs = parse_crs_param(crs_text);
        if (projection != 0 && request.crs.is_set() && request.crs.epsg != projection) {
            throw ValueError(std::string(task) + ": projection (" +
                             std::to_string(projection) + ") and crs (\"" + crs_text +
                             "\") name different coordinate systems; pass only one");
        }
    } else if (projection != 0) {
        request.crs = parse_crs_param("EPSG:" + std::to_string(projection));
    }
    return request;
}

namespace {

/// The CRS a container records for its raster, as the engine type. An unset or
/// EPSG-0 container is the local frame.
crs::CoordinateSystem container_crs(const CRS& recorded) {
    if (!recorded.is_set()) {
        return crs::CoordinateSystem::local();
    }
    return static_cast<crs::CoordinateSystem>(recorded);
}

/// Both the source and the target must name a real CRS for a reprojection; a
/// local-frame raster and a real target (or the reverse) cannot be related.
void require_reprojectable(const CRS& raster_crs, const crs::CoordinateSystem& target,
                           const char* task) {
    if (!raster_crs.is_set()) {
        throw ValueError(std::string(task) +
                         ": the raster has no recorded coordinate system, so it cannot be "
                         "reprojected; rebuild it with a CRS or export without one");
    }
    if (target.is_local() || !target.is_set()) {
        throw ValueError(std::string(task) +
                         ": the target coordinate system is empty or local");
    }
    if (target.epsg() == 0) {
        throw ValueError(std::string(task) +
                         ": a custom (non-EPSG) target CRS cannot be written yet: GeoTIFF "
                         "GeoKeys carry authority codes here, and WKT GeoKeys land with "
                         "WP-12.10");
    }
}

/// The EPSG code the written file declares: the target's when reprojecting,
/// else the container's own.
int exported_epsg(const CRS& raster_crs, const ExportCrs& target) {
    if (!target.requested()) {
        return raster_crs.epsg;
    }
    return static_cast<crs::CoordinateSystem>(target.crs).epsg();
}

bool is_geographic(const crs::CoordinateSystem& cs) {
    const crs::CrsKind kind = cs.kind();
    return kind == crs::CrsKind::Geographic2D || kind == crs::CrsKind::Geographic3D;
}

/// Metres per degree of latitude. Only used to phrase the unit error: a value
/// wrong by 111000x is what this catches, so a second-order ellipsoid
/// correction would not change the diagnosis.
constexpr double kMetresPerDegree = 111320.0;

/// The source's ground sample distance in METRES. A projected or local grid is
/// already metric; a geographic one converts its cell with cos(latitude), taken
/// at the raster's centre row.
double source_gsd_metres(const RasterGrid& grid, const crs::CoordinateSystem& recorded) {
    if (recorded.is_set() && is_geographic(recorded)) {
        const double lat = grid.origin_y - 0.5 * grid.height * grid.pixel_size_y;
        const double radians = lat * 3.14159265358979323846 / 180.0;
        const double cos_lat = std::max(1e-6, std::abs(std::cos(radians)));
        return std::min(grid.pixel_size_x * kMetresPerDegree * cos_lat,
                        grid.pixel_size_y * kMetresPerDegree);
    }
    return std::min(grid.pixel_size_x, grid.pixel_size_y);
}

/// One compact spelling for a suggested cell ("%.3g").
std::string format_cell(double value) {
    char buffer[32] = {};
    std::snprintf(buffer, sizeof(buffer), "%.3g", value);
    return buffer;
}

/// Refuse a value that is almost certainly metres handed to a geographic
/// target. A geographic cell is in DEGREES, so a "resolution" of 0.01 or more
/// is about 1.1 km of ground - absurd next to a source that resolves a few
/// metres. Accepting it would silently produce a grid of one or two cells.
void require_resolution_unit(double resolution, const RasterGrid& grid,
                             const crs::CoordinateSystem& from,
                             const crs::CoordinateSystem& to, const char* task) {
    if (!(resolution >= 0.01) || !is_geographic(to)) {
        return;
    }
    const double gsd = source_gsd_metres(grid, from);
    if (!(gsd > 0.0) || gsd >= 10.0) {
        return;  // a genuinely coarse source: the request may be honest
    }
    throw ValueError(std::string(task) + ": resolution " + format_cell(resolution) +
                     " is in DEGREES for a geographic target CRS, so it is about " +
                     format_cell(resolution * kMetresPerDegree) +
                     " m of ground - the source resolves about " + format_cell(gsd) +
                     " m, so this looks like a value in metres. Pass the degree value, "
                     "about " + format_cell(gsd / kMetresPerDegree) +
                     " for the source's own cell.");
}

/// Refuse a result grid with no area to speak of: a 1-cell-wide strip is not a
/// raster a caller can use, and is the symptom of a unit error the checks above
/// could not see (a coarse cell against a tiny extent).
void require_usable_grid(const RasterGrid& grid, const char* task) {
    if (grid.width >= 2 && grid.height >= 2) {
        return;
    }
    throw ValueError(std::string(task) + ": the requested export would produce a " +
                     std::to_string(grid.width) + " x " + std::to_string(grid.height) +
                     " cell grid; the resolution is too coarse for the raster's extent. "
                     "Ask for a smaller cell.");
}

/// The reprojection options for one raster: a hole stays a hole, and carries
/// the caller's nodata when it is written.
ReprojectOptions reproject_options(float nodata_in, float nodata_out) {
    ReprojectOptions options;
    options.nodata_in = nodata_in;
    options.nodata_out = nodata_out;
    return options;
}

/// True when a requested CRS names the SAME coordinate system the raster already
/// records: the identity case, which is a copy (or, at a new cell, an anchored
/// re-cell), never a reprojection. `suggest_grid`'s outward snapping is for
/// mapping between planes; on one plane it would shift a real origin by up to a
/// cell and leave a blank border.
bool same_coordinate_system(const crs::CoordinateSystem& recorded, const ExportCrs& target) {
    if (!target.requested()) {
        return true;
    }
    return recorded.is_set() &&
           recorded.is_equivalent_to(static_cast<crs::CoordinateSystem>(target.crs));
}

/// True when a re-cell request leaves the source grid untouched: an unchanged
/// (or unset) cell size, so the export is a copy.
bool keeps_source_grid(const RasterGrid& grid, double resolution) {
    if (!(resolution > 0.0)) {
        return true;
    }
    return std::abs(resolution / grid.pixel_size_x - 1.0) <= 1e-9 &&
           std::abs(resolution / grid.pixel_size_y - 1.0) <= 1e-9;
}

}  // namespace

GeoRaster<float> reproject_for_export(const GeoRaster<float>& raster, const CRS& raster_crs,
                                      float nodata_in, float nodata_out,
                                      const ExportCrs& target, const char* task) {
    const crs::CoordinateSystem recorded = container_crs(raster_crs);
    if (same_coordinate_system(recorded, target) &&
        keeps_source_grid(raster.grid, target.resolution)) {
        // Same plane, same cell: a bit-identical copy, with no PROJ call at all.
        // This is the common "export as built" path and the same-CRS export the
        // caller asked for, which must not move the origin.
        return raster;
    }
    if (same_coordinate_system(recorded, target)) {
        // Same plane, new cell: anchor the new grid at the source's own corner.
        // A geographic source takes DEGREES here too, so the unit rule applies.
        require_resolution_unit(target.resolution, raster.grid, recorded, recorded, task);
        const RasterGrid grid = resample_grid(raster.grid, target.resolution);
        require_usable_grid(grid, task);
        return reproject(raster, recorded, recorded, grid,
                         reproject_options(nodata_in, nodata_out));
    }
    const crs::CoordinateSystem to = static_cast<crs::CoordinateSystem>(target.crs);
    require_reprojectable(raster_crs, to, task);
    require_resolution_unit(target.resolution, raster.grid, recorded, to, task);
    const RasterGrid grid = suggest_grid(raster.grid, recorded, to, target.resolution);
    require_usable_grid(grid, task);
    return reproject(raster, recorded, to, grid, reproject_options(nodata_in, nodata_out));
}

GeoRaster<std::uint8_t> reproject_for_export(const GeoRaster<std::uint8_t>& raster,
                                             const CRS& raster_crs,
                                             const ExportCrs& target, const char* task) {
    const crs::CoordinateSystem recorded = container_crs(raster_crs);
    if (same_coordinate_system(recorded, target) &&
        keeps_source_grid(raster.grid, target.resolution)) {
        return raster;
    }
    if (same_coordinate_system(recorded, target)) {
        require_resolution_unit(target.resolution, raster.grid, recorded, recorded, task);
        const RasterGrid grid = resample_grid(raster.grid, target.resolution);
        require_usable_grid(grid, task);
        return reproject(raster, recorded, recorded, grid, ReprojectOptions{});
    }
    const crs::CoordinateSystem to = static_cast<crs::CoordinateSystem>(target.crs);
    require_reprojectable(raster_crs, to, task);
    require_resolution_unit(target.resolution, raster.grid, recorded, to, task);
    const RasterGrid grid = suggest_grid(raster.grid, recorded, to, target.resolution);
    require_usable_grid(grid, task);
    return reproject(raster, recorded, to, grid, ReprojectOptions{});
}

int export_epsg(const CRS& raster_crs, const ExportCrs& target) {
    return exported_epsg(raster_crs, target);
}

}  // namespace alphashape::elevation
