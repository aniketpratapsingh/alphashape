#pragma once

// Seamline labelling: choose, for every output cell, which rectified tile's
// colour it takes, then move the seams between tiles to where the two images
// look alike.
//
// The labelling is a Markov random field over the output cells. A cell's label
// is a tile index, or the special label -1 where no tile covers it, and the
// objective is
//
//     E(l) = data_weight * sum_p ( best(p) - score_{l(p)}(p) )
//          + sum_{p ~ q} w( l(p), l(q) )
//
// with, over the four-neighbour grid p ~ q,
//
//     best(p)   = the largest score of any tile valid at p,
//     w(a, b)   = 0                                            if a == b, or a < 0, or b < 0,
//               = seam_weight * ( d_{a,b}(p) + d_{a,b}(q) ) + smoothness   otherwise,
//     d_{a,b}(x) = ( 1/3 * sum_c | I_a(x, c) - I_b(x, c) | ) / 255  in [0, 1],
//
// the mean per-channel grey-level disagreement, over the 255-level range, that
// the two tiles show at x. A label that does not cover a cell carries an
// infinite data cost, so every coverable cell stays labelled and no label ever
// escapes the region where its tile is valid.
//
// The data term is a *preference*, not the dominant cost: the best tile at a
// cell is free and any other costs only its score shortfall against `best`,
// which is small in an overlap where two views are near-equally good. With the
// defaults (seam_weight 2, data_weight 0.02) the seam term dominates: one edge
// whose two ends disagree by 30 grey levels costs 2 * 2 * 30 / 255 = 0.47, i.e.
// about 23x the full data budget of a cell (data_weight * 1.0). A seam therefore
// detours many cells rather than cross a 30-grey disagreement, while a seam over
// agreeing views (disagreement near 0) still costs only `smoothness`. The
// `smoothness` constant (0.30) is paid per label-change edge, so label regions
// stay compact instead of sprouting single-cell islands. Note that a seam can be
// the *cheapest* option even where one camera covers a whole feature: enclosing
// that feature in its own region costs its perimeter in seam edges, which can
// exceed the length of a straight boundary crossing it.
//
// Optimiser: multi-scale alpha-expansion (Boykov, Veksler & Zabih 2001) over a
// 2x block-mean image pyramid (default: a 2x and a 4x level above the full
// grid), with an ICM polish. The coarsest level is solved from its own argmax,
// its labels are upsampled to the next level and refined, and so on down to the
// full grid. Each level runs a few alpha-expansion cycles -- for every label
// alpha a min-cut (push-relabel max-flow) decides the exact set of cells that
// should switch to alpha, which flips whole regions at once where plain ICM
// cannot, and is the reason seams move off the flat spots instead of sitting
// where argmax happens to switch. Submodularity holds because the seam term is a
// metric (a sum of absolute colour differences plus a constant). The full grid
// is finished by the ICM polish alone: the coarse levels carry the global
// structure, and a full-resolution min-cut buys almost nothing for most of the
// runtime. The flow is single-threaded with a fixed edge order and FIFO queue,
// and the ICM sweep runs in checkerboard order, so the labelling is bit-identical
// for 1 and 4 threads.
//
// `iterations == 0` returns the plain per-cell argmax of the score, with no
// smoothing (useful as the "before" labelling).

#include <cstdint>
#include <vector>

#include "common/algorithm/ortho/orthorectify.hpp"

namespace alphashape::ortho {

struct SeamOptions {
    int iterations = 2;         ///< ICM polish sweeps per level (0 = argmax only)
    double seam_weight = 2.0;   ///< weight of the colour-agreement seam term
    double data_weight = 0.02;  ///< weight of the score-shortfall preference term
    double smoothness = 0.30;   ///< constant cost of one label-change edge
    int coarse_levels = 2;      ///< 2x block-mean levels solved before the full grid
    int expansion_cycles = 2;   ///< alpha-expansion passes per pyramid level
};

/// Label per output cell: the index of the winning tile, or -1 where no tile is
/// valid. Deterministic; independent of the thread count.
///
/// `pinned_labels` / `pinned` (both or neither) fix part of the labelling: where
/// `pinned[i]` is non-zero, cell `i` keeps `pinned_labels[i]` and the optimiser
/// may not change it (the label carries an infinite data cost against every
/// other label, and the pair terms against it are still charged). Block-wise
/// callers pass the margin cells a previously processed block already decided,
/// which is what keeps seams continuous across block borders. A pinned label
/// that this tile set does not cover falls back to the per-cell argmax.
std::vector<std::int32_t> assign_seams(
    int width, int height, const std::vector<OrthoTile>& tiles, const SeamOptions& options,
    const std::vector<std::int32_t>* pinned_labels = nullptr,
    const std::vector<std::uint8_t>* pinned = nullptr);

/// The energy of a labelling under `options` (used by tests and callers to
/// compare two labellings). Invalid labels contribute nothing where no tile is
/// valid and an infinite cost where one is.
double energy(int width, int height, const std::vector<OrthoTile>& tiles,
              const std::vector<std::int32_t>& labels, const SeamOptions& options);

}  // namespace alphashape::ortho
