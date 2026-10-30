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

}  // namespace

GeoRaster<float> reproject_for_export(const GeoRaster<float>& raster, const CRS& raster_crs,
                                      float nodata_in, float nodata_out,
                                      const ExportCrs& target, const char* task) {
    if (!target.requested() && !(target.resolution > 0.0)) {
        return raster;  // unchanged: the common path, no work at all
    }
    if (!target.requested()) {
        // A resolution-only request: a same-CRS resample. Nothing to transform,
        // so the source and target CRSs are the container's own.
        const crs::CoordinateSystem self = container_crs(raster_crs);
        const RasterGrid grid = suggest_grid(raster.grid, self, self, target.resolution);
        return reproject(raster, self, self, grid, reproject_options(nodata_in, nodata_out));
    }
    const crs::CoordinateSystem from = container_crs(raster_crs);
    const crs::CoordinateSystem to = static_cast<crs::CoordinateSystem>(target.crs);
    require_reprojectable(raster_crs, to, task);
    const RasterGrid grid = suggest_grid(raster.grid, from, to, target.resolution);
    return reproject(raster, from, to, grid, reproject_options(nodata_in, nodata_out));
}

GeoRaster<std::uint8_t> reproject_for_export(const GeoRaster<std::uint8_t>& raster,
                                             const CRS& raster_crs,
                                             const ExportCrs& target, const char* task) {
    if (!target.requested() && !(target.resolution > 0.0)) {
        return raster;
    }
    const crs::CoordinateSystem from = container_crs(raster_crs);
    if (!target.requested()) {
        const RasterGrid grid = suggest_grid(raster.grid, from, from, target.resolution);
        return reproject(raster, from, from, grid, ReprojectOptions{});
    }
    const crs::CoordinateSystem to = static_cast<crs::CoordinateSystem>(target.crs);
    require_reprojectable(raster_crs, to, task);
    const RasterGrid grid = suggest_grid(raster.grid, from, to, target.resolution);
    return reproject(raster, from, to, grid, ReprojectOptions{});
}

int export_epsg(const CRS& raster_crs, const ExportCrs& target) {
    return exported_epsg(raster_crs, target);
}

}  // namespace alphashape::elevation
