#include "common/algorithm/elevation/export_crs.hpp"

#include <cmath>
#include <cstdint>
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
bool same_coordinate_system(const CRS& raster_crs, const crs::CoordinateSystem& recorded,
                            const ExportCrs& target) {
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
    if (same_coordinate_system(raster_crs, recorded, target) &&
        keeps_source_grid(raster.grid, target.resolution)) {
        // Same plane, same cell: a bit-identical copy, with no PROJ call at all.
        // This is the common "export as built" path and the same-CRS export the
        // caller asked for, which must not move the origin.
        return raster;
    }
    if (same_coordinate_system(raster_crs, recorded, target)) {
        // Same plane, new cell: anchor the new grid at the source's own corner.
        const RasterGrid grid = resample_grid(raster.grid, target.resolution);
        return reproject(raster, recorded, recorded, grid,
                         reproject_options(nodata_in, nodata_out));
    }
    const crs::CoordinateSystem to = static_cast<crs::CoordinateSystem>(target.crs);
    require_reprojectable(raster_crs, to, task);
    const RasterGrid grid = suggest_grid(raster.grid, recorded, to, target.resolution);
    return reproject(raster, recorded, to, grid, reproject_options(nodata_in, nodata_out));
}

GeoRaster<std::uint8_t> reproject_for_export(const GeoRaster<std::uint8_t>& raster,
                                             const CRS& raster_crs,
                                             const ExportCrs& target, const char* task) {
    const crs::CoordinateSystem recorded = container_crs(raster_crs);
    if (same_coordinate_system(raster_crs, recorded, target) &&
        keeps_source_grid(raster.grid, target.resolution)) {
        return raster;
    }
    if (same_coordinate_system(raster_crs, recorded, target)) {
        const RasterGrid grid = resample_grid(raster.grid, target.resolution);
        return reproject(raster, recorded, recorded, grid, ReprojectOptions{});
    }
    const crs::CoordinateSystem to = static_cast<crs::CoordinateSystem>(target.crs);
    require_reprojectable(raster_crs, to, task);
    const RasterGrid grid = suggest_grid(raster.grid, recorded, to, target.resolution);
    return reproject(raster, recorded, to, grid, ReprojectOptions{});
}

int export_epsg(const CRS& raster_crs, const ExportCrs& target) {
    return exported_epsg(raster_crs, target);
}

}  // namespace alphashape::elevation
