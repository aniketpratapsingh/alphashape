#pragma once

// Orthomosaic blending: turn a set of rectified tiles plus a label per cell
// into a single RGB raster.
//
//   * Disabled   copy the labelled tile's colour; the seams of the labelling
//                are visible as-is.
//   * Average    score-weighted mean of every valid tile at the cell.
//   * Mosaic     Burt & Adelson (1983) "A multiresolution spline with
//                application to image mosaics": a Laplacian pyramid of each
//                tile that labels some cell, a Gaussian pyramid of that tile's
//                binary label mask, and a level-by-level weighted blend whose
//                weights are restricted to cells where the tile is valid (and
//                renormalised so a cell with no valid contributor stays empty).
//
// The output carries the label field and the visibility mask.

#include <cstdint>
#include <string>
#include <vector>

#include "common/algorithm/ortho/orthorectify.hpp"

namespace alphashape::ortho {

/// Metashape's public Blending names: MosaicBlending / AverageBlending /
/// DisabledBlending.
enum class Blending { Mosaic, Average, Disabled };

struct OrthoMosaic {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;     ///< 3 * W * H
    std::vector<std::uint8_t> valid;   ///< W * H
    std::vector<std::int32_t> labels;  ///< W * H, index into tiles or -1
};

/// Blend `tiles` according to `labels` (one label per cell, -1 = none).
OrthoMosaic blend(int width, int height, const std::vector<OrthoTile>& tiles,
                  const std::vector<std::int32_t>& labels, Blending mode);

/// Parse a blending mode from its lower-case name: "mosaic", "average",
/// "disabled". Throws ValueError on any other text.
Blending parse_blending(const std::string& text);

/// The lower-case name of `mode`, the inverse of parse_blending.
std::string to_string(Blending mode);

}  // namespace alphashape::ortho
