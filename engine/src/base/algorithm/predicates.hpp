#pragma once

// Robust geometric predicates for 2-D triangulation.
//
// Both predicates return the exact sign of a determinant of the *input* double
// coordinates: +1, -1, or 0. They first evaluate the determinant in ordinary
// double arithmetic and accept that sign when the rounding error bound cannot
// have changed it (an error bound of the form Shewchuk 1997 derives); only when
// the filter is inconclusive is the determinant re-evaluated exactly, with
// floating-point expansion (multi-component) arithmetic built from error-free
// transformations (two-sum, two-difference, two-product).
//
// Reference: J. R. Shewchuk, "Adaptive Precision Floating-Point Arithmetic and
// Fast Robust Geometric Predicates", Discrete & Computational Geometry 18(3),
// 1997. The two-stage structure and the error constants follow that paper; the
// expansion routines here are written for this engine (grow-by-one summation
// rather than the paper's merged two-expansion walk), which is slower in the
// exact path but simpler to audit and exact for every input.
//
// The signs are exact for coordinates whose squared magnitudes do not
// overflow a double, and whose magnitudes span no more than about 400 binary
// exponents (the exact path rescales the whole input by a power of two before
// forming the expansions, so a mesh carried at 1e-9 or 1e9 metres is as exact
// as one at unit scale). Coordinates that straddle a wider range, or that are
// subnormal, are outside what two-component expansions can represent.

#include <Eigen/Core>

namespace alphashape {

/// Sign of twice the signed area of the triangle (a, b, c):
/// +1 when a -> b -> c turns counter-clockwise, -1 when it turns clockwise,
/// and 0 when the three points are exactly collinear.
///
/// Exact for every finite input.
int orient2d(double ax, double ay, double bx, double by, double cx, double cy);

/// Convenience overload.
int orient2d(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
             const Eigen::Vector2d& c);

/// Sign of the InCircle determinant of (a, b, c, d), which is positive when d
/// lies strictly inside the circle through a, b, c *provided* a, b, c are in
/// counter-clockwise order. Zero when the four points are exactly cocircular
/// (and when a, b, c are collinear).
///
/// Exact for every finite input.
int incircle(double ax, double ay, double bx, double by, double cx, double cy,
             double dx, double dy);

/// Convenience overload; same convention (a, b, c counter-clockwise).
int incircle(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
             const Eigen::Vector2d& c, const Eigen::Vector2d& d);

/// Orientation of three *indexed* vertices. Returns orient2d() when the
/// coordinates are not exactly collinear. When they are collinear the caller
/// needs a nonzero answer, so the vertices are symbolically perturbed and the
/// sign is the parity of the permutation that sorts (ia, ib, ic): +1 for an
/// even permutation, -1 for an odd one. The result is antisymmetric under
/// swapping any two arguments and depends only on the indices, so repeated
/// calls agree (Simulation of Simplicity, Edelsbrunner and Muecke, ACM
/// Transactions on Programming Languages and Systems 12(1), 1990).
///
/// Returns 0 when two indices coincide: a degenerate triangle stays
/// degenerate under the perturbation.
int orient2d_perturbed(double ax, double ay, double bx, double by, double cx,
                       double cy, int ia, int ib, int ic);

/// Incircle with symbolic perturbation, same convention as incircle(). When
/// the exact determinant is zero (cocircular or collinear), the sign is the
/// parity of the permutation that sorts (ia, ib, ic, id); 0 when two indices
/// coincide.
int incircle_perturbed(double ax, double ay, double bx, double by, double cx,
                       double cy, double dx, double dy, int ia, int ib, int ic,
                       int id);

}  // namespace alphashape
