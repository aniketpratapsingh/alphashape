#pragma once

// Opt-in step-edge refinement for a DEM height field.
//
// Stereo matching plus smoothing turns a sharp roof/building outline into a
// ramp several cells wide; orthorectifying with that DEM then smears the
// outline. This pass finds long straight height discontinuities and rebuilds a
// sharp step across them:
//
//   1. Sobel gradient magnitude of the height field, in metres per cell.
//   2. Candidate cells: gradient above min_height / (2 cells) and a height
//      step >= min_height between the samples +-2..3 cells along it.
//   3. Orientation region growing (+-15 degrees) groups the candidates; a
//      total-least-squares line (PCA of the cell centres) is fitted to each.
//   4. One plane per side of the line, fitted by Huber IRLS over the band's
//      non-transition cells (Huber 1964, Ann. Math. Stat. 35(1)).
//   5. Accepted segments snap each band cell to its side's plane, blended over
//      the last cell toward the original surface and clamped to the band's
//      original height range. Only measured cells are evidence or written.
//
// The caller runs this pass on the final DEM surface, i.e. after hole filling
// and TGV smoothing, so the heights it sharpens are the ones that get written.
//
// The grouping follows the LSD idea of chaining aligned gradient evidence
// (von Gioi et al. 2010, "LSD: a fast line segment detector", IEEE TPAMI
// 32(4)) before the robust two-plane model.
//
// Every option and the return value are in metres: heights are metres,
// `cell_size_x` and `cell_size_y` are metres per cell along the grid axes,
// and `min_height` and `min_length` are metres.
//
// Memory: peak is about 38 bytes per cell plus the caller's inputs -- an
// 8-byte double height copy, a 1-byte usability mask, three float gradient
// planes (12), a 1-byte candidate mask, one int32 label and one int32 CSR
// cell array (8), and float best-height/best-distance arrays (8). The
// gradient, candidate and label working sets are released after grouping, so
// the plane stage holds about 21 bytes per cell plus the inputs.

#include <cstdint>
#include <vector>

namespace alphashape::elevation {

struct StepEdgeOptions {
    double min_height = 1.5;   ///< metres; smallest accepted height step
    double min_length = 3.0;   ///< metres; shortest accepted segment
    int band_half_width = 5;   ///< cells fitted/snapped on each side of a line
};

/// Refine `heights` (row-major, width*height, metres) in place. `valid[i] == 0`
/// marks cells that are not measurements (nodata, hole-filled, or repaired-only
/// height guesses), which are never read as evidence nor written. Returns the
/// number of accepted straight segments.
///
/// Throws ValueError when the sizes disagree, width or height is < 3, or a
/// cell size is not positive and finite.
int refine_step_edges(std::vector<float>& heights, const std::vector<std::uint8_t>& valid,
                      int width, int height, double cell_size_x, double cell_size_y,
                      const StepEdgeOptions& options);

}  // namespace alphashape::elevation
