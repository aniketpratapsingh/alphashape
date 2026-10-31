#pragma once

// Multi-view consensus ghosting filter for the orthomosaic (Q4).
//
// A moving car, a walking person or a transient shadow is seen by one
// photograph and not by the others that cover the same ground. The seam
// labeller, given only a per-camera preference score, has no reason to prefer
// the majority: keeping a 200-cell object costs a few units of data term,
// while cutting it out costs its whole perimeter in seam edges. The filter
// therefore builds an explicit DEVIANT mask before the seams are chosen.
//
// Basis: robust multi-view consensus for deghosting (Jacobs, Loscos & Ward
// 2008; Gallagher 2005) and moving-object removal in mosaics (Davis 1998,
// "Mosaics of scenes with moving objects"; Uyttendaele, Eden & Szeliski 2001,
// "Eliminating ghosting and exposure artifacts in image mosaics"). Our own
// formulation, stated in our own terms:
//
//   * at several scales (1x, 4x, 16x box-downsampled by default) a cell is
//     compared against the per-channel MEDIAN of every camera valid there;
//   * the spread is the robust MAD of those values (1.4826 * MAD), floored at
//     min_abs_diff / k so an unusually consistent set does not mark texture
//     noise;
//   * a camera is deviant when some channel departs from the median by more
//     than both k spreads and min_abs_diff. Fewer than min_cameras views means
//     there is no consensus, so nothing is marked there;
//   * coarse masks are upsampled and OR-ed with the finer ones, so a large
//     transient (a shadow) is caught even when its interior is nearly flat; a
//     camera takes part in a coarse cell only when it covers at least half of
//     it, so a sliver of coverage is never compared as if it were a full view;
//   * per tile the mask is closed then opened (radius 1) and components below
//     min_component_cells are dropped, which removes isolated sensor noise
//     without eroding a genuine object;
//   * the separate speckle pass only relabels small label islands that touch a
//     deviant region (dilated by 2 cells), and prefers a neighbour the
//     consensus does not call deviant there.
//
// Everything is deterministic and OpenMP-safe: rows are parallel, each cell
// writes only its own mask entries, and every tie is broken by index.

#include <cstdint>
#include <vector>

#include "common/algorithm/ortho/orthorectify.hpp"

namespace alphashape::ortho {

struct GhostOptions {
    /// Scales examined: 1x, 4x, 16x (and so on, x4 per extra level).
    int levels = 3;
    /// Robust z threshold on the MAD-measured spread.
    double k = 4.0;
    /// Absolute grey-level floor: a difference this large is always suspicious.
    double min_abs_diff = 18.0;
    /// Minimum cameras covering a cell for a consensus to exist.
    int min_cameras = 3;
    /// Radius of the square structuring element used to close, then open.
    int morph_radius = 1;
    /// Marked components smaller than this are dropped.
    int min_component_cells = 16;
};

/// One deviant mask per tile, tile-local (width * height bytes, row-major).
/// Mismatched or empty tiles get an empty mask.
std::vector<std::vector<std::uint8_t>> detect_ghosts(
    int width, int height, const std::vector<OrthoTile>& tiles,
    const GhostOptions& options = {});

/// Relabel small 4-connected label islands (speckles) to the most common
/// neighbouring label that is valid at every cell of the island. `pinned`
/// (nullable) marks cells that may never change; an island containing a pinned
/// cell is left alone. `deviant` (nullable, one tile-local mask per tile) scopes
/// the pass to the ghosting filter: only islands that touch a deviant region
/// (dilated by 2 cells) are relabelled, and a neighbouring label that is itself
/// deviant at the island is used only when no clean neighbour exists. Returns
/// the number of components removed. Deterministic.
int filter_label_speckles(int width, int height, const std::vector<OrthoTile>& tiles,
                          std::vector<std::int32_t>& labels,
                          const std::vector<std::uint8_t>* pinned, int min_component_cells,
                          const std::vector<std::vector<std::uint8_t>>* deviant = nullptr);

}  // namespace alphashape::ortho
