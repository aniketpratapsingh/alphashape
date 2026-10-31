#include "common/algorithm/ortho/export_orthomosaic_task.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/core/assets.hpp"
#include "common/core/document.hpp"
#include "model/raster/geotiff.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

/// The world-file path for a GeoTIFF at `path`: same directory and stem with a
/// ".tfw" extension (the same rule ExportDem uses).
std::string world_file_path(const std::string& path) {
    const fs::path file(path);
    return (file.parent_path() / (file.stem().string() + ".tfw")).string();
}

}  // namespace

void ExportOrthomosaicTask::declare_params(Params& params) const {
    // Names follow Metashape's public exportOrthomosaic() Python API.
    params.declare({"path", ParamType::String, std::string{},
                    "Destination GeoTIFF path. Required."});
    params.declare({"tiled", ParamType::Bool, true, "Write a tiled GeoTIFF."});
    params.declare({"tile_size", ParamType::Int, std::int64_t{256},
                    "Tile edge in pixels (a multiple of 16) when tiled."});
    params.declare({"world_file", ParamType::Bool, false,
                    "Also write an ESRI world file next to the GeoTIFF."});
    params.declare({"alpha", ParamType::Bool, true,
                    "Write RGBA (alpha 0 = no data). False writes RGB with nodata cells "
                    "black."});
    params.declare({"asset_id", ParamType::Int, std::int64_t{-1},
                    "Id of the orthomosaic to write. -1 selects the active one."});
}

std::vector<Stage> ExportOrthomosaicTask::stages() const {
    return {Stage{"validate", 0}, Stage{"write", 0}};
}

void ExportOrthomosaicTask::apply(Document& document, Params& params,
                                  const Progress& progress) {
    progress.update("validate", 0.0);
    const std::string path = params.get_string("path");
    if (path.empty()) {
        throw ValueError("path is required");
    }

    const Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("ExportOrthomosaic needs an active chunk");
    }
    const std::int64_t asset_id = params.get_int("asset_id");
    if (asset_id > std::numeric_limits<std::int32_t>::max()) {
        throw ValueError("ExportOrthomosaic: asset_id " + std::to_string(asset_id) +
                         " is out of range");
    }
    const std::int32_t wanted =
        asset_id < 0 ? chunk->active_orthomosaic_id() : static_cast<std::int32_t>(asset_id);
    const OrthomosaicAsset* asset = find_asset(chunk->orthomosaics(), wanted);
    if (asset == nullptr && asset_id >= 0) {
        throw ValueError("ExportOrthomosaic: no orthomosaic asset with id " +
                         std::to_string(asset_id) + " in this chunk");
    }
    if (asset == nullptr || !asset->data) {
        throw Error("No orthomosaic");
    }
    const Orthomosaic& orthomosaic = *asset->data;
    if (orthomosaic.empty()) {
        throw Error("No orthomosaic");
    }
    if (orthomosaic.raster.channels != 4) {
        throw Error("ExportOrthomosaic: the orthomosaic raster is not RGBA");
    }

    GeoTiffOptions options;
    options.epsg = orthomosaic.epsg();
#if defined(ALPHASHAPE_USE_PROJ)
    // Carry the whole CRS, not just its code: a custom (uncoded) definition
    // then travels in the citation key as WKT2 instead of being lost.
    options.crs.emplace(static_cast<crs::CoordinateSystem>(orthomosaic.crs));
#endif
    options.tiled = params.get_bool("tiled");
    options.tile_size = static_cast<int>(params.get_int("tile_size"));
    options.deflate = true;
    const bool alpha = params.get_bool("alpha");
    const bool world_file = params.get_bool("world_file");
    progress.update("validate", 1.0);

    progress.update("write", 0.0);
    if (alpha) {
        write_geotiff(path, orthomosaic.raster, options);
    } else {
        // Drop the alpha plane. Nodata cells keep their (zero) colour, which
        // is how imagery without an alpha channel expresses "no data".
        GeoRaster<std::uint8_t> rgb;
        rgb.grid = orthomosaic.raster.grid;
        rgb.channels = 3;
        rgb.data.resize(static_cast<std::size_t>(rgb.grid.width) *
                        static_cast<std::size_t>(rgb.grid.height) * 3U);
        for (std::size_t i = 0; i < rgb.data.size() / 3U; ++i) {
            rgb.data[i * 3 + 0] = orthomosaic.raster.data[i * 4 + 0];
            rgb.data[i * 3 + 1] = orthomosaic.raster.data[i * 4 + 1];
            rgb.data[i * 3 + 2] = orthomosaic.raster.data[i * 4 + 2];
        }
        write_geotiff(path, rgb, options);
    }
    if (world_file) {
        write_world_file(world_file_path(path), orthomosaic.raster.grid);
    }
    progress.update("write", 1.0);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "ExportOrthomosaic: " << orthomosaic.raster.grid.width << "x"
                            << orthomosaic.raster.grid.height << " cells, "
                            << (alpha ? "RGBA" : "RGB") << ", epsg "
                            << orthomosaic.epsg() << " (asset " << asset->id << ") -> " << path
                            << (world_file ? " (+ .tfw)" : "");
    }
}

}  // namespace alphashape
