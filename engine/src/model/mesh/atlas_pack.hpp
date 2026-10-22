#pragma once

// Atlas packing: fit a set of flattened charts onto one square texture page.
//
// Takes the per-chart world-unit coordinates produced by build_charts() and
// places them into [0,1]^2 with one shared uniform scale, rotating charts as
// needed to pack them tightly, then writes the resulting texture coordinates
// back onto the mesh.
//
// This header declares the interface only. pack_atlas() is a stub until the
// texturing milestone lands its implementation.

#include <vector>

#include "model/mesh/mesh.hpp"
#include "model/mesh/uv_charts.hpp"

namespace alphashape {

/// Parameters for the atlas packer.
struct PackOptions {
    /// Side of the (square) texture page, in pixels.
    int texture_size = 4096;
    /// Minimum gap, in texels, that must remain between any two charts and
    /// between a chart and the page border. Absorbs bilinear filtering bleed.
    int padding_px = 4;
};

/// Summary of a completed pack.
struct PackResult {
    /// The uniform scale the packer chose, in texels per world unit.
    double texels_per_unit = 0.0;
    /// Fraction of the page's texels that ended up covered by chart interiors.
    double fill_ratio = 0.0;
};

/// Rotate, place and uniformly scale `charts` into one [0,1]^2 page. On success
/// writes `mesh.uvs` (the coordinate pool) and `mesh.uv_faces` (one coordinate
/// per face corner) — corners of the same chart may share a pool entry. Charts
/// never overlap and keep at least `options.padding_px` texels between each
/// other. Throws Error("pack_atlas: not implemented") until implemented.
PackResult pack_atlas(Mesh& mesh, const std::vector<Chart>& charts, const PackOptions& options);

}  // namespace alphashape
