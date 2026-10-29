#pragma once

// Exact natural-neighbour (Sibson) coordinates on a Delaunay triangulation.
//
// Sibson (1981) defines the weight of site s at a query q as the share of q's
// Voronoi cell (after inserting q) that is taken from s's original cell. The
// computation here follows the "stolen area" construction (Watson 1992,
// "Contouring: A Guide to the Analysis and Display of Spatial Data"): the
// triangles whose circumcircle strictly contains q are exactly the ones q would
// replace; their boundary vertices are q's natural neighbours, and the region
// q steals from neighbour a is the polygon bounded by
//   - the circumcentre of (q, previous neighbour, a),
//   - the circumcentres of the replaced triangles around a, in fan order,
//   - the circumcentre of (q, a, next neighbour).
// Its area divided by the total is a's weight. The result reproduces linear
// functions exactly (up to rounding), and is C1 away from the sites.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "base/algorithm/delaunay2d.hpp"

namespace alphashape::elevation {

struct SibsonWeight {
    int site = -1;      ///< representative site index in the triangulation
    double weight = 0;  ///< in (0, 1]; the weights of one query sum to 1
};

/// Natural-neighbour coordinates of q. Returns false (and an empty `out`) when
/// q is outside the convex hull of the sites or the triangulation is degenerate.
/// A query exactly on a site returns that site with weight 1; a query on the
/// hull boundary returns the linear weights along its edge (the limit of the
/// Sibson coordinates there). `hint` is a triangle to start the point location
/// from (-1 for none) and is updated to the containing triangle, so scanning
/// neighbouring queries reuses it.
bool sibson_coordinates(const Delaunay2D& dt, const Eigen::Vector2d& q,
                        std::vector<SibsonWeight>& out, int& hint);

/// Exact Sibson pass for fill_holes(): for every 4-connected component of cells
/// with fill[i] != 0, the data cells 8-adjacent to it are the sites; when there
/// are between 3 and max_sites of them, every component cell inside their
/// convex hull gets result[i] = sum of weight * value over its natural
/// neighbours. Cells outside the hull, and components with too many sites, keep
/// the result they already have. Returns the number of cells computed exactly.
std::size_t exact_sibson_fill(int width, int height, const std::vector<float>& values,
                              const std::vector<std::uint8_t>& valid,
                              const std::vector<std::uint8_t>& fill,
                              std::vector<float>& result, int max_sites);

}  // namespace alphashape::elevation
