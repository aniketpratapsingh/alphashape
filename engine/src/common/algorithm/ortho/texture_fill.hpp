// texture_fill.hpp -- exemplar-based completion of interior orthomosaic holes.
//
// Natural-neighbour interpolation fills a hole with a smooth gradient, which
// is right for terrain but obviously wrong inside a textured surface (grass,
// gravel, a roof). This module instead copies texture from the valid pixels
// around the hole, following the iterative search-and-vote scheme of Wexler,
// Shechtman & Irani (2007) with the nearest-neighbour field found by
// PatchMatch (Barnes et al. 2009). It is an alternative to the smooth fill,
// not a replacement: the caller runs the smooth fill first and hands the
// resulting cells here as the initial image, so a component with no usable
// source still degrades gracefully to the smooth result.
//
// The function is a pure, deterministic transform: it owns no global state and
// the output depends only on the arguments. Both properties are tested.

#ifndef ALPHASHAPE_COMMON_ALGORITHM_ORTHO_TEXTURE_FILL_HPP
#define ALPHASHAPE_COMMON_ALGORITHM_ORTHO_TEXTURE_FILL_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace alphashape {
namespace ortho {

/// Tunables for texture_fill(). The defaults are sized for a typical ortho
/// hole of a few hundred cells, where a 7x7 patch still contains several
/// periods of common ground texture.
struct TextureFillOptions {
    /// Patch half-size: the patch is (2*radius+1)^2 cells. 3 gives 7x7.
    int patch_radius = 3;
    /// Components larger than this many cells keep the smooth fill. Completion
    /// cost grows with the hole area times the source search, so a big hole is
    /// both slow and prone to inventing structure that is not there.
    int max_component_cells = 20000;
    /// The source search is restricted to the component's bounding box dilated
    /// by this many cells. Capping the window keeps the per-component memory
    /// and work local; valid cells outside the window are ignored.
    int source_margin = 48;
    /// Number of 2x box-downsample levels (including the finest). The build
    /// stops early when a level would be smaller than 4 patches across.
    int pyramid_levels = 3;
    /// Search + vote rounds per pyramid level ("EM" iterations in the paper).
    int em_iterations = 4;
    /// PatchMatch propagation / random-search sweeps per EM iteration.
    int patchmatch_iterations = 4;
    /// Base seed. Each component reseeds deterministically from this and the
    /// raster position of its first cell, so the result is independent of the
    /// order in which components -- or threads -- are processed.
    std::uint32_t seed = 1;
};

/// What texture_fill() did, for logging and tests.
struct TextureFillReport {
    /// Number of 4-connected components of `fill_mask` seen.
    std::size_t components = 0;
    /// Number of cells actually completed (component total minus skipped).
    std::size_t filled_cells = 0;
    /// Components left as the caller wrote them because they were too large,
    /// or because their window held no full valid source patch.
    std::size_t skipped_large = 0;
};

/// Complete the cells marked in `fill_mask` by copying texture from the valid
/// cells around them, in place.
///
/// `rgba` is width*height*4 bytes. A cell whose alpha is 0 is nodata; every
/// other cell is a valid source. `fill_mask[i] != 0` names the cells to
/// complete; they must already hold a reasonable starting value (the smooth
/// fill) because it initialises the pyramid and is the fallback when a
/// component is skipped. Completed cells get alpha 255; cells with
/// `fill_mask[i] == 0` are never modified.
///
/// Returns the per-run counters. The cost is per component, bounded by
/// `max_component_cells` and `source_margin`, and all scratch buffers are
/// released before the next component starts.
TextureFillReport texture_fill(int width, int height, std::vector<std::uint8_t>& rgba,
                               const std::vector<std::uint8_t>& fill_mask,
                               const TextureFillOptions& options = {});

}  // namespace ortho
}  // namespace alphashape

#endif  // ALPHASHAPE_COMMON_ALGORITHM_ORTHO_TEXTURE_FILL_HPP
