#pragma once

// BuildDem: rasterise the chunk's reconstruction into a georeferenced digital
// elevation model (milestone 9).
//
// The adapter between the chunk's data model and the elevation primitives
// (surface_frame.hpp, interpolate.hpp, tgv_denoise.hpp). Depending on
// `source_data` the samples are
//
//   * "point_cloud": every dense-cloud point mapped into the plane frame,
//   * "model":       the mesh triangles rasterised top-down onto the cell
//                    centres (MAX height per cell, i.e. a DSM),
//   * "depth_maps":  every valid depth pixel back-projected into chunk-local
//                    space (exactly as stereo/fusion.cpp documents) and mapped
//                    into the plane frame, carrying its confidence.
//
// Each cell then holds the median of the heights of the samples falling in it
// (the max height for the model) with the summed confidence, or, when
// `fusion` is "robust", a confidence-weighted Tukey biweight estimate over the
// cell's symmetrically trimmed observations (cell_fusion.hpp). Holes are filled
// per `interpolation` and an optional TGV^2 pass smooths the surface before it
// is written into Chunk::elevation().

#include <functional>
#include <string>

#include "common/algorithm/elevation/interpolate.hpp"
#include "common/algorithm/elevation/tgv_denoise.hpp"
#include "common/core/chunk.hpp"

namespace alphashape::elevation {

/// The TGV^2 weights this builder uses by default.
///
/// The generic TgvOptions defaults (alpha1 = 1, alpha0 = 2) are calibrated for
/// images normalised to [0, 1]; a DEM carries metre-scale steps (an 8 m roof
/// over one cell) that those weights flatten. build_dem passes heights scaled by
/// 1/resolution, so one cell step is one unit and a typical stereo DEM's noise
/// sits at ~0.1 unit while a real building edge is ~an order of magnitude
/// larger. alpha1 = 0.2 (alpha0 = 0.4, ratio 2 for piecewise-affine plateaus and
/// ramps) removes sub-decimetre noise without touching genuine edges.
TgvOptions dem_smoothing_options();

/// The resolved inputs of one build_dem() call. Names of the string options
/// follow Metashape's public Python API (buildDem); the value sets
/// are ours.
struct DemOptions {
    /// "point_cloud", "model" or "depth_maps".
    std::string source = "depth_maps";
    /// "disabled", "enabled" (fill holes inside the cameras' ground footprints)
    /// or "extrapolated" (fill every hole).
    std::string interpolation = "enabled";
    /// Cell size in plane units. <= 0 selects an automatic resolution from the
    /// source's sample spacing.
    double resolution = 0.0;
    /// Requested CRS for the output grid (M12 §4.9), as the project DTO the
    /// task layer produces. Unset = auto UTM for a georeferenced chunk, or the
    /// chunk-local frame when unreferenced.
    CRS crs;
    /// Hole-filling method used for the cells interpolation fills.
    HoleFill fill = HoleFill::NaturalNeighbour;
    /// Holes bordered by at most this many data cells are filled with exact
    /// Sibson interpolation (natural_neighbour only); 0 = discrete everywhere.
    int sibson_exact_max_sites = 0;
    /// Maximum distance in METRES from an interpolated cell to the nearest
    /// measured cell; farther holes stay nodata. 0 = unlimited. The builder
    /// converts this to cells with the DEM's metres-per-cell before filling.
    double max_fill_distance = 0.0;
    /// Per-cell height fusion: "median" (default, the plain per-cell median) or
    /// "robust" (confidence-weighted Tukey biweight M-estimate over the cell's
    /// symmetrically trimmed observations; see cell_fusion.hpp).
    std::string fusion = "median";
    /// Run the TGV^2 smoothing / inpainting pass (never for the model source).
    bool tgv = true;
    TgvOptions tgv_options = dem_smoothing_options();
};

/// Build the DEM and return it (the caller commits it to the chunk). `progress`
/// receives fractions in [0, 1]. Throws ValueError for an unknown source or
/// interpolation, an empty source, or a source that cannot yield a resolution.
Elevation build_dem(const Chunk& chunk, const DemOptions& options,
                    const std::function<void(double)>& progress = {});

}  // namespace alphashape::elevation
