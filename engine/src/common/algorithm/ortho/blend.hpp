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
///
/// `pyramid_levels` pins the mosaic blend's pyramid depth: -1 (the default) picks
/// the depth from the grid, as before. Block-wise callers pass one value derived
/// from the WHOLE output grid so every block applies the same multiresolution
/// operator instead of a shallower one of its own.
/// `avoid` (optional, ghosting.hpp) is one tile-local deviant mask per tile.
/// Average blending leaves a deviant camera out of the mean unless it is the
/// only valid one (so a ghost cannot bleed into an averaged cell); Mosaic
/// blending needs no change, because a tile's blend weight is its label mask
/// and a non-winning tile already contributes nothing. Nullptr (the default)
/// preserves the previous behaviour exactly.
OrthoMosaic blend(int width, int height, const std::vector<OrthoTile>& tiles,
                  const std::vector<std::int32_t>& labels, Blending mode,
                  int pyramid_levels = -1,
                  const std::vector<std::vector<std::uint8_t>>* avoid = nullptr);

/// The pyramid depth blend() would pick for a `width` x `height` grid. A block
/// plan uses this to pin the depth at the whole grid's value.
int pyramid_level_count(int width, int height);

/// How far a blend at `levels` can read, in cells, from an output cell: the
/// accumulated reach of the 5-tap blurs down the pyramid (level k reaches
/// 2 * 2^k cells, so the sum is 2 * (2^levels - 1)). A block processed with
/// `levels` needs at least this much margin for its interior to be blended with
/// exactly the operator the whole grid would apply.
int pyramid_support_cells(int levels);

/// Parse a blending mode from its lower-case name: "mosaic", "average",
/// "disabled". Throws ValueError on any other text.
Blending parse_blending(const std::string& text);

/// The lower-case name of `mode`, the inverse of parse_blending.
std::string to_string(Blending mode);

}  // namespace alphashape::ortho
