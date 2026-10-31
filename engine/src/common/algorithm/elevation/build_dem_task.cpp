#include "common/algorithm/elevation/build_dem_task.hpp"

#include <cmath>
#include <string>
#include <utility>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/elevation/dem_builder.hpp"
#include "common/algorithm/elevation/surface_frame.hpp"
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
    // `projection` (Int EPSG, retained) and `crs` (String "EPSG:n" or WKT, new)
    // are two spellings of the same request; when both are set they must agree.
    const int projection = static_cast<int>(params.get_int("projection"));
    if (projection < 0) {
        throw ValueError("BuildDem: projection must be an EPSG code >= 0");
    }
    const std::string crs_text = params.get_string("crs");
    if (!crs_text.empty()) {
        options.crs = elevation::parse_crs_param(crs_text);
        if (projection != 0 && options.crs.is_set() && options.crs.epsg != projection) {
            throw ValueError(
                "BuildDem: projection (" + std::to_string(projection) +
                ") and crs (\"" + crs_text + "\") name different coordinate systems; "
                "pass only one");
        }
    } else if (projection != 0) {
        options.crs = elevation::parse_crs_param("EPSG:" + std::to_string(projection));
    }
    options.fill = parse_fill_method(params.get_string("fill_method"));
    options.sibson_exact_max_sites = static_cast<int>(params.get_int("sibson_exact_max_sites"));
    options.max_fill_distance = params.get_float("max_fill_distance");
    if (!(options.max_fill_distance >= 0.0) || !std::isfinite(options.max_fill_distance)) {
        throw ValueError("BuildDem: max_fill_distance must be a finite value >= 0 (metres)");
    }
    options.fusion = params.get_string("fusion");
    if (options.fusion != "median" && options.fusion != "robust") {
        throw ValueError("BuildDem: fusion must be \"median\" or \"robust\" (got \"" +
                         options.fusion + "\")");
    }
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
                            "How holes are filled: \"disabled\" (nodata), \"enabled\" (holes inside "
                            "the cameras' ground footprints) or \"extrapolated\" (every hole)."};
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
                    "chunk-local frame when unreferenced. Kept for compatibility; `crs` "
                    "is the general form, and setting both to disagreeing values is an "
                    "error."});

    params.declare({"crs", ParamType::String, std::string{},
                    "Output coordinate system as \"EPSG:<code>\", a PROJ string or WKT. "
                    "Empty uses `projection`; when both are set they must agree."});

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

    ParamSpec max_fill_distance{"max_fill_distance", ParamType::Float, 0.0,
                                "Maximum distance in metres from an interpolated cell to the "
                                "nearest measured cell. Holes farther than this stay nodata. "
                                "0 (default) = unlimited. Applies to every fill_method and to "
                                "both the \"enabled\" and \"extrapolated\" interpolation modes. "
                                "Filled cells sag with distance, so 10 m is a sensible cap for "
                                "a measurement-grade DEM; the orthomosaic benefits from the "
                                "uncapped default."};
    max_fill_distance.minimum = 0.0;
    params.declare(max_fill_distance);

    ParamSpec fusion{"fusion", ParamType::String, std::string("median"),
                     "Per-cell height fusion: \"median\" (default, the plain median of the "
                     "cell's measured heights) or \"robust\" (a confidence-weighted Tukey "
                     "biweight M-estimate over the symmetrically trimmed observations, with "
                     "the cell's data weight reduced when the kept samples disagree)."};
    fusion.choices = {"median", "robust"};
    params.declare(fusion);

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
    const int epsg = elevation.epsg();
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
