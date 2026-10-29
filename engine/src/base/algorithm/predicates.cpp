#include "base/algorithm/predicates.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace alphashape {
namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

// Shewchuk 1997, section 4: the forward error bounds of the double-precision
// filters, "ccwerrboundA" and "iccerrboundA" in that paper.
constexpr double kOrientBound = (3.0 + 16.0 * kEps) * kEps;
constexpr double kIncircleBound = (10.0 + 96.0 * kEps) * kEps;

int sign_of(double x) { return (x > 0.0) ? 1 : ((x < 0.0) ? -1 : 0); }

// An expansion is a sum of nonoverlapping doubles: no two components carry the
// same bit, so every partial sum below is exact in double and the expansion
// represents its value exactly.
using Expansion = std::vector<double>;

// a + b == x + y exactly.
void two_sum(double a, double b, double& x, double& y) {
    x = a + b;
    const double b_virtual = x - a;
    const double a_virtual = x - b_virtual;
    y = (a - a_virtual) + (b - b_virtual);
}

// a - b == x + y exactly.
void two_diff(double a, double b, double& x, double& y) {
    x = a - b;
    const double b_virtual = a - x;
    const double a_virtual = x + b_virtual;
    y = (a - a_virtual) + (b_virtual - b);
}

// a * b == x + y exactly: x is the rounded product, y its rounding error.
void two_product(double a, double b, double& x, double& y) {
    x = a * b;
    y = std::fma(a, b, -x);
}

// Exact value of a - b as a zero-eliminated expansion.
Expansion exact_diff(double a, double b) {
    double x = 0.0;
    double y = 0.0;
    two_diff(a, b, x, y);
    Expansion e;
    if (y != 0.0) e.push_back(y);
    if (x != 0.0 || e.empty()) e.push_back(x);
    return e;
}

// Add the scalar b to a nonoverlapping expansion, smallest components first.
// The result is again zero-eliminated and nonoverlapping.
Expansion grow_expansion_zeroelim(const Expansion& e, double b) {
    Expansion out;
    out.reserve(e.size() + 1);
    double q = b;
    for (double ei : e) {
        double sum = 0.0;
        double err = 0.0;
        two_sum(q, ei, sum, err);
        if (err != 0.0) out.push_back(err);
        q = sum;
    }
    if (q != 0.0 || out.empty()) out.push_back(q);
    return out;
}

// Sum of two expansions. Shewchuk's merged two-expansion walk would be faster;
// growing one expansion by the components of the other is exact for the same
// inputs and much easier to audit. The precondition is that the components of
// `f` are visited smallest first, which holds for every expansion built above.
Expansion fast_expansion_sum_zeroelim(const Expansion& e, const Expansion& f) {
    Expansion out = e;
    for (double fi : f) out = grow_expansion_zeroelim(out, fi);
    return out;
}

// Exact product by a single double. Each two_product yields a two-component
// nonoverlapping pair, added low-then-high so the increasing-magnitude order
// the following grow_expansion_zeroelim expects is preserved.
Expansion scale_expansion_zeroelim(const Expansion& e, double b) {
    if (b == 0.0 || e.empty()) return Expansion{0.0};
    Expansion out;
    for (double ei : e) {
        double hi = 0.0;
        double lo = 0.0;
        two_product(ei, b, hi, lo);
        out = grow_expansion_zeroelim(out, lo);
        out = grow_expansion_zeroelim(out, hi);
    }
    return out;
}

Expansion expansion_product(const Expansion& e, const Expansion& f) {
    Expansion out{0.0};
    for (double fi : f) {
        out = fast_expansion_sum_zeroelim(
            out, scale_expansion_zeroelim(e, fi));
    }
    return out;
}

Expansion negation(const Expansion& e) {
    Expansion out = e;
    for (double& v : out) v = -v;
    return out;
}

// Both determinants are homogeneous in the coordinates (orient2d of degree 2,
// incircle of degree 4) and a power-of-two factor is exact, so rescaling every
// coordinate by the same 2^k cannot change a sign. It is what keeps a mesh at
// 1e-9 units or at 1e9 units as accurate as one at unit scale: without it a
// product of two coordinates can overflow or underflow and the expansion loses
// the terms that decide the sign.
//
// Coordinates whose exponents span more than kMaxSpan bits cannot be held
// exactly by a two-component expansion at any common scale, so they are used as
// they are (and are outside what these predicates claim).
constexpr int kMaxSpan = 400;

struct Rescale {
    int exponent = 0;
    double operator()(double v) const { return std::ldexp(v, exponent); }
};

Rescale rescale_for(double max_abs, double min_abs) {
    if (max_abs == 0.0) return Rescale{0};
    int max_exp = 0;
    std::frexp(max_abs, &max_exp);
    int min_exp = 0;
    std::frexp(min_abs, &min_exp);
    if (max_exp - min_exp > kMaxSpan) return Rescale{0};
    return Rescale{-max_exp};
}

// Largest and smallest nonzero magnitude over a set of coordinates.
void extent(double v, double& max_abs, double& min_abs) {
    const double a = std::fabs(v);
    if (a == 0.0) return;
    if (a > max_abs) max_abs = a;
    if (a < min_abs) min_abs = a;
}

// Every expansion this file produces is nonoverlapping and its components are
// in increasing order of magnitude, so the largest component strictly dominates
// the sum of all the others and its sign is the sign of the whole. The maximum
// is taken rather than the last entry so that the answer does not quietly
// depend on that ordering invariant being preserved by every helper.
int sign_of_expansion(const Expansion& e) {
    if (e.empty()) return 0;
    std::size_t largest = 0;
    for (std::size_t i = 1; i < e.size(); ++i) {
        if (std::fabs(e[i]) > std::fabs(e[largest])) largest = i;
    }
    return sign_of(e[largest]);
}

// det = (ax-cx)(by-cy) - (ay-cy)(bx-cx), the classic two-by-two form of twice
// the signed area. The differences are each exact (an expansion), so the
// expansion product is the exact determinant even when the doubles cancel.
// The coordinates are first rescaled by a power of two, which leaves the sign
// intact and keeps every expansion component a normal double.
Expansion orient2d_exact(double ax, double ay, double bx, double by, double cx,
                         double cy) {
    double max_abs = 0.0;
    double min_abs = std::numeric_limits<double>::infinity();
    extent(ax, max_abs, min_abs);
    extent(ay, max_abs, min_abs);
    extent(bx, max_abs, min_abs);
    extent(by, max_abs, min_abs);
    extent(cx, max_abs, min_abs);
    extent(cy, max_abs, min_abs);
    const Rescale r = rescale_for(max_abs, min_abs);
    const Expansion acx = exact_diff(r(ax), r(cx));
    const Expansion bcy = exact_diff(r(by), r(cy));
    const Expansion acy = exact_diff(r(ay), r(cy));
    const Expansion bcx = exact_diff(r(bx), r(cx));
    return fast_expansion_sum_zeroelim(expansion_product(acx, bcy),
                                       negation(expansion_product(acy, bcx)));
}

}  // namespace

int orient2d(double ax, double ay, double bx, double by, double cx, double cy) {
    const double detleft = (ax - cx) * (by - cy);
    const double detright = (ay - cy) * (bx - cx);
    const double det = detleft - detright;
    // Shewchuk's A-level filter: when |det| exceeds the accumulated rounding
    // error of the two products and the difference, that rounding cannot have
    // flipped the sign. |detleft| + |detright| bounds the error of both.
    const double errbound = kOrientBound * (std::fabs(detleft) + std::fabs(detright));
    if (det > errbound) return 1;
    if (-det > errbound) return -1;
    return sign_of_expansion(orient2d_exact(ax, ay, bx, by, cx, cy));
}

int orient2d(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
             const Eigen::Vector2d& c) {
    return orient2d(a.x(), a.y(), b.x(), b.y(), c.x(), c.y());
}

namespace {

// +1 for an even number of inversions among the distinct indices, -1 for an odd
// number, 0 when two indices repeat. Sorting a sequence costs exactly as many
// swaps as it has inversions, so the inversion count is the permutation parity.
int index_parity(int a, int b) { return (a < b) ? 1 : -1; }

int parity3(int a, int b, int c) {
    if (a == b || b == c || a == c) return 0;
    const int p = index_parity(a, b) * index_parity(a, c) * index_parity(b, c);
    return (p > 0) ? 1 : -1;
}

int parity4(int a, int b, int c, int d) {
    if (a == b || a == c || a == d || b == c || b == d || c == d) return 0;
    const int p = index_parity(a, b) * index_parity(a, c) * index_parity(a, d) *
                  index_parity(b, c) * index_parity(b, d) * index_parity(c, d);
    return (p > 0) ? 1 : -1;
}

}  // namespace

int orient2d_perturbed(double ax, double ay, double bx, double by, double cx,
                       double cy, int ia, int ib, int ic) {
    const int exact = orient2d(ax, ay, bx, by, cx, cy);
    if (exact != 0) return exact;
    return parity3(ia, ib, ic);
}

namespace {

Expansion expansion_square_sum(const Expansion& x, const Expansion& y) {
    return fast_expansion_sum_zeroelim(expansion_product(x, x),
                                       expansion_product(y, y));
}

// The InCircle determinant, written with the differences from d as Shewchuk
// does, evaluated exactly. Every difference is exact as an expansion, so the
// lifts and the three two-by-two determinants below are exact too.
Expansion incircle_exact(double ax, double ay, double bx, double by, double cx,
                         double cy, double dx, double dy) {
    double max_abs = 0.0;
    double min_abs = std::numeric_limits<double>::infinity();
    extent(ax, max_abs, min_abs);
    extent(ay, max_abs, min_abs);
    extent(bx, max_abs, min_abs);
    extent(by, max_abs, min_abs);
    extent(cx, max_abs, min_abs);
    extent(cy, max_abs, min_abs);
    extent(dx, max_abs, min_abs);
    extent(dy, max_abs, min_abs);
    const Rescale r = rescale_for(max_abs, min_abs);
    const Expansion adx = exact_diff(r(ax), r(dx));
    const Expansion ady = exact_diff(r(ay), r(dy));
    const Expansion bdx = exact_diff(r(bx), r(dx));
    const Expansion bdy = exact_diff(r(by), r(dy));
    const Expansion cdx = exact_diff(r(cx), r(dx));
    const Expansion cdy = exact_diff(r(cy), r(dy));

    const Expansion abdet = fast_expansion_sum_zeroelim(
        expansion_product(adx, bdy), negation(expansion_product(bdx, ady)));
    const Expansion bcdet = fast_expansion_sum_zeroelim(
        expansion_product(bdx, cdy), negation(expansion_product(cdx, bdy)));
    const Expansion cadet = fast_expansion_sum_zeroelim(
        expansion_product(cdx, ady), negation(expansion_product(adx, cdy)));

    const Expansion alift = expansion_square_sum(adx, ady);
    const Expansion blift = expansion_square_sum(bdx, bdy);
    const Expansion clift = expansion_square_sum(cdx, cdy);

    Expansion det = expansion_product(alift, bcdet);
    det = fast_expansion_sum_zeroelim(det, expansion_product(blift, cadet));
    det = fast_expansion_sum_zeroelim(det, expansion_product(clift, abdet));
    return det;
}

}  // namespace

int incircle(double ax, double ay, double bx, double by, double cx, double cy,
             double dx, double dy) {
    const double adx = ax - dx;
    const double ady = ay - dy;
    const double bdx = bx - dx;
    const double bdy = by - dy;
    const double cdx = cx - dx;
    const double cdy = cy - dy;

    const double bdxcdy = bdx * cdy;
    const double cdxbdy = cdx * bdy;
    const double cdxady = cdx * ady;
    const double adxcdy = adx * cdy;
    const double adxbdy = adx * bdy;
    const double bdxady = bdx * ady;

    const double alift = adx * adx + ady * ady;
    const double blift = bdx * bdx + bdy * bdy;
    const double clift = cdx * cdx + cdy * cdy;

    const double det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) +
                       clift * (adxbdy - bdxady);

    // The "permanent" is the same expression with every term made positive, so
    // it bounds the absolute rounding error accumulated above.
    const double permanent =
        (std::fabs(bdxcdy) + std::fabs(cdxbdy)) * alift +
        (std::fabs(cdxady) + std::fabs(adxcdy)) * blift +
        (std::fabs(adxbdy) + std::fabs(bdxady)) * clift;
    const double errbound = kIncircleBound * permanent;
    if (det > errbound) return 1;
    if (-det > errbound) return -1;
    return sign_of_expansion(incircle_exact(ax, ay, bx, by, cx, cy, dx, dy));
}

int incircle(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
             const Eigen::Vector2d& c, const Eigen::Vector2d& d) {
    return incircle(a.x(), a.y(), b.x(), b.y(), c.x(), c.y(), d.x(), d.y());
}

int incircle_perturbed(double ax, double ay, double bx, double by, double cx,
                       double cy, double dx, double dy, int ia, int ib, int ic,
                       int id) {
    const int exact = incircle(ax, ay, bx, by, cx, cy, dx, dy);
    if (exact != 0) return exact;
    return parity4(ia, ib, ic, id);
}

}  // namespace alphashape
