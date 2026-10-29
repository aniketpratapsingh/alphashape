#include "common/algorithm/elevation/build_dem_task.hpp"

#include <string>
#include <utility>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/elevation/dem_builder.hpp"
#include "common/core/assets.hpp"
#include "common/core/document.hpp"

namespace alphashape {
namespace {

using elevation::DemOptions;
using elevation::HoleFill;

/// The validated choice of `fill_method`.
HoleFill parse_fill_method(const std::string& value) {
    if (value == "natural_neighbour") {
        return HoleFill::NaturalNeighbour;
    }
    if (value == "idw") {
        return HoleFill::Idw;
    }
    if (value == "nearest") {
        return HoleFill::Nearest;
    }
    throw ValueError("BuildDem: fill_method must be one of \"natural_neighbour\", \"idw\", "
                     "\"nearest\" (got \"" +
                     value + "\")");
}

DemOptions resolve_options(const Params& params) {
    DemOptions options;
    options.source = params.get_string("source_data");
    options.interpolation = params.get_string("interpolation");
    if (options.source != "depth_maps" && options.source != "point_cloud" &&
        options.source != "model") {
        throw ValueError("BuildDem: source_data must be one of \"depth_maps\", \"point_cloud\", "
                         "\"model\" (got \"" +
                         options.source + "\")");
    }
    if (options.interpolation != "disabled" && options.interpolation != "enabled" &&
        options.interpolation != "extrapolated") {
        throw ValueError("BuildDem: interpolation must be one of \"disabled\", \"enabled\", "
                         "\"extrapolated\" (got \"" +
                         options.interpolation + "\")");
    }
    options.resolution = params.get_float("resolution");
    if (options.resolution < 0.0) {
        throw ValueError("BuildDem: resolution must be >= 0 (0 selects an automatic value)");
    }
    options.epsg = static_cast<int>(params.get_int("projection"));
    options.fill = parse_fill_method(params.get_string("fill_method"));
    options.sibson_exact_max_sites = static_cast<int>(params.get_int("sibson_exact_max_sites"));
    const std::string smoothing = params.get_string("smoothing");
    if (smoothing == "tgv") {
        options.tgv = true;
    } else if (smoothing == "none") {
        options.tgv = false;
    } else {
        throw ValueError("BuildDem: smoothing must be \"tgv\" or \"none\" (got \"" + smoothing +
                         "\")");
    }
    return options;
}

}  // namespace

void BuildDemTask::declare_params(Params& params) const {
    // Names follow Metashape's public Python API (buildDem); the value
    // sets are ours.
    ParamSpec source_data{"source_data", ParamType::String, std::string("depth_maps"),
                          "Elevation source: \"depth_maps\", \"point_cloud\" or \"model\"."};
    source_data.choices = {"depth_maps", "point_cloud", "model"};
    params.declare(source_data);

    ParamSpec interpolation{"interpolation", ParamType::String, std::string("enabled"),
                            "How holes are filled: \"disabled\" (nodata), \"enabled\" (interior "
                            "holes only) or \"extrapolated\" (every hole)."};
    interpolation.choices = {"disabled", "enabled", "extrapolated"};
    params.declare(interpolation);

    ParamSpec resolution{"resolution", ParamType::Float, 0.0,
                         "Cell size in projection units. 0 selects the automatic value from "
                         "the source's sample spacing: the median depth-map ground sample "
                         "distance in metres (converted to degrees for a geographic CRS), the "
                         "median mesh edge for the model source, or the point spacing for a "
                         "point cloud."};
    resolution.minimum = 0.0;
    params.declare(resolution);

    params.declare({"projection", ParamType::Int, std::int64_t{0},
                    "Output EPSG code. 0 selects UTM for a georeferenced chunk, or a "
                    "chunk-local frame when unreferenced."});

    ParamSpec fill_method{"fill_method", ParamType::String, std::string("natural_neighbour"),
                          "Hole-filling interpolation: \"natural_neighbour\", \"idw\" or "
                          "\"nearest\"."};
    fill_method.choices = {"natural_neighbour", "idw", "nearest"};
    params.declare(fill_method);

    ParamSpec exact_sites{"sibson_exact_max_sites", ParamType::Int, std::int64_t{0},
                          "natural_neighbour: holes bordered by at most this many data cells "
                          "use exact Sibson interpolation; larger ones the discrete "
                          "approximation. 0 (default) = discrete everywhere; 20000 is a "
                          "sensible value to enable it."};
    exact_sites.minimum = std::int64_t{0};
    exact_sites.maximum = std::int64_t{1000000};
    params.declare(exact_sites);

    ParamSpec smoothing{"smoothing", ParamType::String, std::string("tgv"),
                        "Surface smoothing: \"tgv\" (TGV^2) or \"none\". The model source is "
                        "never smoothed."};
    smoothing.choices = {"tgv", "none"};
    params.declare(smoothing);

    params.declare({"label", ParamType::String, std::string{},
                    "Label for the new elevation asset. Empty selects \"Elevation <n>\", "
                    "with the smallest unused n. Each run creates a new asset."});
}

std::vector<Stage> BuildDemTask::stages() const {
    return {Stage{"initialize", 0}, Stage{"build_dem", 0}, Stage{"finalize", 0}};
}

void BuildDemTask::apply(Document& document, Params& params, const Progress& progress) {
    progress.update("initialize", 0.0);
    const DemOptions options = resolve_options(params);
    const std::string label = params.get_string("label");

    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("BuildDem needs an active chunk");
    }
    progress.update("initialize", 1.0);

    progress.update("build_dem", 0.0);
    Elevation elevation = elevation::build_dem(
        *chunk, options, [&](double fraction) { progress.update("build_dem", fraction); });
    progress.update("build_dem", 1.0);

    progress.update("finalize", 0.0);
    const std::size_t samples = static_cast<std::size_t>(elevation.raster.grid.width) *
                                static_cast<std::size_t>(elevation.raster.grid.height);
    const int width = elevation.raster.grid.width;
    const int height = elevation.raster.grid.height;
    const int epsg = elevation.epsg;
    // Each run APPENDS a new elevation asset and makes it active; the previous
    // elevation (if any) is kept, so a second BuildDem never overwrites a DEM a
    // earlier run produced.
    ElevationAsset& asset = chunk->add_elevation(label);
    *asset.data = std::move(elevation);
    progress.update("finalize", 1.0);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "BuildDem: " << width << "x" << height << " cells (" << samples
                            << ") from " << options.source << ", interpolation "
                            << options.interpolation << ", epsg " << epsg << " -> asset "
                            << asset.id << " '" << asset.label << "'";
    }
}

}  // namespace alphashape
