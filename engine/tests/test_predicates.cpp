// Robust orient2d / incircle: exact signs on degenerate and adversarial inputs.

#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "base/algorithm/predicates.hpp"

namespace alphashape {
namespace {

// Exact reference for integer coordinates. The products stay below 2^62 for
// the ranges used here, so int64 arithmetic is the true determinant and its
// sign is the answer orient2d has to reproduce.
int exact_orient(std::int64_t ax, std::int64_t ay, std::int64_t bx,
                 std::int64_t by, std::int64_t cx, std::int64_t cy) {
    const std::int64_t det = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx);
    return (det > 0) ? 1 : ((det < 0) ? -1 : 0);
}

TEST(Predicates, Orient2dSigns) {
    EXPECT_EQ(orient2d(0.0, 0.0, 1.0, 0.0, 0.0, 1.0), 1);
    EXPECT_EQ(orient2d(0.0, 0.0, 0.0, 1.0, 1.0, 0.0), -1);
    EXPECT_EQ(orient2d(0.0, 0.0, 1.0, 1.0, 2.0, 2.0), 0);
    // Two coincident vertices: no triangle, so no orientation.
    EXPECT_EQ(orient2d(1.0, 1.0, 1.0, 1.0, 2.0, 2.0), 0);

    const Eigen::Vector2d a(0.25, -3.5);
    const Eigen::Vector2d b(7.0, 2.0);
    const Eigen::Vector2d c(-4.0, 9.75);
    const int s = orient2d(a, b, c);
    EXPECT_NE(s, 0);
    // Antisymmetric under any transposition, invariant under a cyclic shift.
    EXPECT_EQ(orient2d(b, a, c), -s);
    EXPECT_EQ(orient2d(a, c, b), -s);
    EXPECT_EQ(orient2d(b, c, a), s);
    // Mirroring the plane flips every orientation; translating it does not.
    EXPECT_EQ(orient2d(Eigen::Vector2d(a.x(), -a.y()), Eigen::Vector2d(b.x(), -b.y()),
                       Eigen::Vector2d(c.x(), -c.y())),
              -s);
    EXPECT_EQ(orient2d(a + Eigen::Vector2d(1e6, -2e5),
                       b + Eigen::Vector2d(1e6, -2e5),
                       c + Eigen::Vector2d(1e6, -2e5)),
              s);
}

TEST(Predicates, NearDegenerate) {
    // Integer-scaled collinear triples with a one-unit bend: the double filter
    // has to agree with the exact int64 determinant at every scale.
    const std::int64_t scales[] = {1, 1024, 1000000};
    for (std::int64_t t : scales) {
        const double s = static_cast<double>(t);
        EXPECT_EQ(orient2d(0.0, 0.0, s, s, 2.0 * s, 2.0 * s),
                  exact_orient(0, 0, t, t, 2 * t, 2 * t));
        EXPECT_EQ(orient2d(0.0, 0.0, s, s, 2.0 * s, 2.0 * s + 1.0),
                  exact_orient(0, 0, t, t, 2 * t, 2 * t + 1));
        EXPECT_EQ(orient2d(0.0, 0.0, s, s, 2.0 * s, 2.0 * s - 1.0),
                  exact_orient(0, 0, t, t, 2 * t, 2 * t - 1));
        EXPECT_EQ(orient2d(0.0, 0.0, s, s, 2.0 * s, 2.0 * s + 1.0), 1);
        EXPECT_EQ(orient2d(0.0, 0.0, s, s, 2.0 * s, 2.0 * s - 1.0), -1);
    }

    // An offset of 1e-17 is not representable next to 2.0, so the doubles are
    // exactly collinear and the sign is 0 - not the tiny positive number a
    // rounding-blind implementation would report.
    EXPECT_EQ(2.0 + 1e-17, 2.0);
    EXPECT_EQ(orient2d(0.0, 0.0, 1.0, 1.0, 2.0, 2.0 + 1e-17), 0);

    // One ulp above 2.0 is representable and does bend the triple upward.
    const double bent = std::nextafter(2.0, 4.0);
    ASSERT_GT(bent, 2.0);
    EXPECT_EQ(orient2d(0.0, 0.0, 1.0, 1.0, 2.0, bent), 1);

    // Filter-defeating case: both rounded products equal 5.4e16, so the naive
    // determinant is exactly 0 while the exact determinant of the inputs is
    // -8 (verified with rational arithmetic). Only the expansion path finds it.
    const double p52 = std::ldexp(1.0, 52);
    const double p53 = std::ldexp(1.0, 53);
    EXPECT_EQ(orient2d(2.0, -2.0, p52, p52, p53, p53 + 4.0), -1);

    // Symbolic perturbation: collinear vertices are ordered by index, and the
    // answer is the parity of that permutation, so it is antisymmetric.
    EXPECT_EQ(orient2d_perturbed(0, 0, 1, 1, 2, 2, 0, 1, 2), 1);
    EXPECT_EQ(orient2d_perturbed(0, 0, 1, 1, 2, 2, 1, 0, 2), -1);
    EXPECT_EQ(orient2d_perturbed(0, 0, 1, 1, 2, 2, 2, 0, 1), 1);
    // A non-degenerate triple keeps the geometric answer, indices or not.
    EXPECT_EQ(orient2d_perturbed(0, 0, 1, 0, 0, 1, 5, 4, 3), 1);
    // Repeated indices describe a degenerate triangle: still 0.
    EXPECT_EQ(orient2d_perturbed(0, 0, 1, 1, 2, 2, 0, 1, 1), 0);
}

TEST(Predicates, IncircleCocircular) {
    // Four points of the integer circle x^2 + y^2 = 25, in counter-clockwise
    // order, so d = (4, -3) is exactly cocircular with a, b, c.
    const Eigen::Vector2d a(0.0, 5.0);
    const Eigen::Vector2d b(5.0, 0.0);
    const Eigen::Vector2d c(3.0, 4.0);
    const Eigen::Vector2d cocircular(4.0, -3.0);
    ASSERT_EQ(orient2d(a, b, c), 1);
    EXPECT_EQ(incircle(a, b, c, cocircular), 0);

    // Inside and outside, as integer determinants of +250 and -1750.
    EXPECT_EQ(incircle(a, b, c, Eigen::Vector2d(0.0, 0.0)), 1);
    EXPECT_EQ(incircle(a, b, c, Eigen::Vector2d(10.0, 10.0)), -1);

    // Same four cocircular points scaled by 1e10: the naive determinant is
    // -3.09e26, comfortably inside the -4.8e28 error bound, so only the exact
    // expansion path can report 0.
    const double s = 1e10;
    EXPECT_EQ(incircle(Eigen::Vector2d(0.0, 5.0 * s), Eigen::Vector2d(5.0 * s, 0.0),
                       Eigen::Vector2d(3.0 * s, 4.0 * s),
                       Eigen::Vector2d(4.0 * s, -3.0 * s)),
              0);
    // Moving d by one ulp breaks the tie in the direction the very small exact
    // residual says (rational arithmetic gives -7.5e30 / 32768): downward is
    // outside, one ulp the other way is inside.
    const Eigen::Vector2d a_big(0.0, 5.0 * s);
    const Eigen::Vector2d b_big(5.0 * s, 0.0);
    const Eigen::Vector2d c_big(3.0 * s, 4.0 * s);
    const Eigen::Vector2d lower(4.0 * s, std::nextafter(-3.0 * s, -1e300));
    const Eigen::Vector2d upper(4.0 * s, std::nextafter(-3.0 * s, 1e300));
    ASSERT_LT(lower.y(), upper.y());
    EXPECT_EQ(incircle(a_big, b_big, c_big, lower), -1);
    EXPECT_EQ(incircle(a_big, b_big, c_big, upper), 1);

    // Symbolic perturbation: cocircular vertices fall back to index parity,
    // antisymmetrically, and stay 0 when two indices coincide.
    EXPECT_EQ(incircle_perturbed(0, 5, 5, 0, 3, 4, 4, -3, 0, 1, 2, 3), 1);
    EXPECT_EQ(incircle_perturbed(0, 5, 5, 0, 3, 4, 4, -3, 1, 0, 2, 3), -1);
    EXPECT_EQ(incircle_perturbed(0, 5, 5, 0, 3, 4, 4, -3, 3, 0, 1, 2), -1);
    EXPECT_EQ(incircle_perturbed(0, 5, 5, 0, 3, 4, 4, -3, 0, 1, 2, 2), 0);
    // A genuinely interior point keeps the geometric sign.
    EXPECT_EQ(incircle_perturbed(0, 5, 5, 0, 3, 4, 0, 0, 7, 8, 9, 10), 1);
}

TEST(Predicates, Deterministic) {
    // A fixed lattice of nearly-degenerate points. Recomputing must give the
    // same signs in the same order: no dependence on uninitialised state, on
    // evaluation order, or on anything that varies between runs.
    const int n = 24;
    std::vector<int> first;
    first.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const double x = 0.5 * static_cast<double>(i);
            const double y = 0.5 * static_cast<double>(j);
            first.push_back(orient2d_perturbed(0.0, 0.0, 3.0, 3.0, x, y, 0, 1,
                                               i * n + j));
        }
    }
    std::vector<int> second;
    second.reserve(first.size());
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const double x = 0.5 * static_cast<double>(i);
            const double y = 0.5 * static_cast<double>(j);
            second.push_back(orient2d_perturbed(0.0, 0.0, 3.0, 3.0, x, y, 0, 1,
                                                i * n + j));
        }
    }
    EXPECT_EQ(first, second);

    // Every point on the line x == y is exactly collinear with the first two,
    // so the perturbation decides each one and the sign tracks index parity.
    for (int k = 0; k < n; ++k) {
        const double v = 0.25 * static_cast<double>(k);
        // (0,0), (3,3), (v,v): indices (0, 1, 2 + k) - always an even
        // permutation, so every one of these returns +1.
        EXPECT_EQ(orient2d_perturbed(0.0, 0.0, 3.0, 3.0, v, v, 0, 1, 2 + k), 1);
        // Swapping the last two flips it.
        EXPECT_EQ(orient2d_perturbed(0.0, 0.0, 3.0, 3.0, v, v, 0, 2 + k, 1), -1);
    }

    // Swapping two arguments must negate the sign of a perturbed call too, for
    // degenerate and non-degenerate inputs alike.
    const int stable = orient2d_perturbed(1.0, 2.0, 4.0, 2.0, 2.0, 5.0, 11, 3, 7);
    EXPECT_EQ(orient2d_perturbed(4.0, 2.0, 1.0, 2.0, 2.0, 5.0, 3, 11, 7), -stable);
}

}  // namespace
}  // namespace alphashape
