#include "common/algorithm/elevation/export_dem_task.hpp"

#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/core/assets.hpp"
#include "common/core/document.hpp"
#include "model/raster/geotiff.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

/// The world-file path for a GeoTIFF at `path`: the same directory and stem with
/// a ".tfw" extension.
std::string world_file_path(const std::string& path) {
    const fs::path file(path);
    return (file.parent_path() / (file.stem().string() + ".tfw")).string();
}

}  // namespace

void ExportDemTask::declare_params(Params& params) const {
    params.declare({"path", ParamType::String, std::string{},
                    "Destination GeoTIFF path. Required."});
    params.declare({"nodata", ParamType::Float, -32767.0,
                    "Sentinel written for unmeasured cells (GDAL_NODATA)."});
    params.declare({"tiled", ParamType::Bool, true, "Write a tiled GeoTIFF."});
    params.declare({"tile_size", ParamType::Int, std::int64_t{256},
                    "Tile edge in pixels (a multiple of 16) when tiled."});
    params.declare({"world_file", ParamType::Bool, false,
                    "Also write an ESRI world file next to the GeoTIFF."});
    params.declare({"asset_id", ParamType::Int, std::int64_t{-1},
                    "Id of the elevation to write. -1 selects the active one."});
}

std::vector<Stage> ExportDemTask::stages() const {
    return {Stage{"validate", 0}, Stage{"write", 0}};
}

void ExportDemTask::apply(Document& document, Params& params, const Progress& progress) {
    progress.update("validate", 0.0);
    const std::string path = params.get_string("path");
    if (path.empty()) {
        throw ValueError("path is required");
    }

    const Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("ExportDem needs an active chunk");
    }
    const std::int64_t asset_id = params.get_int("asset_id");
    if (asset_id > std::numeric_limits<std::int32_t>::max()) {
        throw ValueError("ExportDem: asset_id " + std::to_string(asset_id) +
                         " is out of range");
    }
    const std::int32_t wanted =
        asset_id < 0 ? chunk->active_elevation_id() : static_cast<std::int32_t>(asset_id);
    const ElevationAsset* asset = find_asset(chunk->elevations(), wanted);
    if (asset == nullptr && asset_id >= 0) {
        throw ValueError("ExportDem: no elevation asset with id " +
                         std::to_string(asset_id) + " in this chunk");
    }
    if (asset == nullptr || !asset->data) {
        throw Error("No elevation model");
    }
    const Elevation& elevation = *asset->data;
    if (elevation.empty()) {
        throw Error("No elevation model");
    }

    GeoTiffOptions options;
    options.epsg = elevation.epsg;
    options.tiled = params.get_bool("tiled");
    options.tile_size = static_cast<int>(params.get_int("tile_size"));
    options.deflate = true;
    options.nodata = params.get_float("nodata");
    const bool world_file = params.get_bool("world_file");
    progress.update("validate", 1.0);

    progress.update("write", 0.0);
    write_geotiff(path, elevation.raster, options);
    if (world_file) {
        write_world_file(world_file_path(path), elevation.raster.grid);
    }
    progress.update("write", 1.0);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "ExportDem: " << elevation.raster.grid.width << "x"
                            << elevation.raster.grid.height << " cells, epsg "
                            << options.epsg << " (asset " << asset->id << ") -> " << path
                            << (world_file ? " (+ .tfw)" : "");
    }
}

}  // namespace alphashape
