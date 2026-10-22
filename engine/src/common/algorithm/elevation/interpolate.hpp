#pragma once

// Elevation raster hole interpolation.
//
// Two pieces:
//   * nearest_sites() — an exact Euclidean distance transform (EDT) over a
//     validity mask, using the Felzenszwalb & Huttenlocher (2012) algorithm
//     ("Distance Transforms of Sampled Functions", Theory of Computing 8),
//     i.e. the lower envelope of parabolas evaluated separably along rows and
//     columns, extended to carry the arg-min site index.
//   * fill_holes() — nearest-neighbour, inverse-distance-weighting, or
//     natural-neighbour (discrete Sibson, Park et al. 2006, IEEE TVCG 12(2))
//     interpolation of the invalid cells of an elevation raster.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace alphashape::elevation {

/// Result of an exact Euclidean distance transform over a validity mask.
struct NearestSites {
    int width = 0;
    int height = 0;
    /// Euclidean distance in cells to the nearest valid cell. 0 at valid
    /// cells, +inf when the raster contains no valid cell at all.
    std::vector<float> distance;
    /// Linear index (y * width + x) of that nearest valid cell; -1 when there
    /// is none. Ties are broken deterministically (lowest linear index wins).
    std::vector<std::int32_t> site;
};

/// Exact EDT (Felzenszwalb-Huttenlocher), separable, O(W*H). `valid[i] != 0`
/// marks a data cell. Throws ValueError on non-positive dimensions or a mask
/// whose size does not equal width*height.
NearestSites nearest_sites(int width, int height,
                           const std::vector<std::uint8_t>& valid);

/// Hole-filling methods.
enum class HoleFill {
    /// Nearest valid cell's value (Voronoi / piecewise-constant).
    Nearest,
    /// Inverse distance weighting over the k nearest valid cells.
    Idw,
    /// Discrete Sibson (natural neighbour) interpolation.
    NaturalNeighbour,
};

struct HoleFillOptions {
    HoleFill method = HoleFill::NaturalNeighbour;
    /// Cells: only holes whose EDT distance to the nearest valid cell is
    /// <= max_distance are filled; farther holes are left invalid.
    double max_distance = std::numeric_limits<double>::infinity();
    /// IDW exponent.
    double idw_power = 2.0;
    /// IDW: number of nearest valid cells to blend (found by expanding rings,
    /// bounded by idw_search_radius).
    int idw_neighbours = 12;
    /// IDW: maximum radius (cells) for the nearest-neighbour search.
    double idw_search_radius = 64.0;
    /// If true, fill only holes not connected (4-connectivity) to the raster
    /// border through invalid cells.
    bool interior_only = false;
};

struct HoleFillReport {
    /// Number of previously-invalid cells that received a value.
    std::size_t filled = 0;
    /// Number of cells still invalid after the call.
    std::size_t remaining = 0;
};

/// Fills the invalid cells of `values` in place and sets `valid[i] = 1` for
/// every filled cell. Valid cells are never changed, and their flags stay 1.
/// Throws ValueError on bad dimensions, mismatched buffer sizes, or invalid
/// options (negative power, non-positive idw_neighbours/search radius,
/// negative max_distance).
HoleFillReport fill_holes(int width, int height, std::vector<float>& values,
                          std::vector<std::uint8_t>& valid,
                          const HoleFillOptions& options = {});

}  // namespace alphashape::elevation
