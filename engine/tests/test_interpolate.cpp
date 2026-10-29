// Tests for elevation raster hole interpolation: exact Euclidean distance
// transform (Felzenszwalb-Huttenlocher) and nearest / IDW / discrete-Sibson
// hole filling.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"
#include "common/algorithm/elevation/interpolate.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using alphashape::ValueError;
using alphashape::elevation::fill_holes;
using alphashape::elevation::HoleFill;
using alphashape::elevation::HoleFillOptions;
using alphashape::elevation::HoleFillReport;
using alphashape::elevation::nearest_sites;
using alphashape::elevation::NearestSites;

/// Number of cells in a `w` x `h` raster.
std::size_t area(int w, int h) {
    return static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
}

/// Linear index of (x, y) in a `w`-wide raster.
std::size_t lin(int x, int y, int w) {
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
           static_cast<std::size_t>(x);
}

/// Brute-force Euclidean distance in cells to the nearest valid cell.
float brute_distance(int width, int height,
                     const std::vector<std::uint8_t>& valid, int x, int y) {
    const float inf = std::numeric_limits<float>::infinity();
    float best = inf;
    for (int yy = 0; yy < height; ++yy) {
        for (int xx = 0; xx < width; ++xx) {
            if (valid[lin(xx, yy, width)] == 0) continue;
            const float dx = static_cast<float>(xx - x);
            const float dy = static_cast<float>(yy - y);
            best = std::min(best, std::sqrt(dx * dx + dy * dy));
        }
    }
    return best;
}

std::vector<std::uint8_t> random_mask(int width, int height, std::uint32_t seed,
                                      double valid_prob) {
    std::mt19937 rng(seed);
    // Portable draw: std::uniform_real_distribution is implementation-defined
    // across libc++ and libstdc++ (TESTING.md, "Numeric consistency").
    const auto dist = [](std::mt19937& g) { return alphashape::rng::uniform_real(g, 0.0, 1.0); };
    std::vector<std::uint8_t> m(static_cast<std::size_t>(width) *
                                static_cast<std::size_t>(height));
    for (auto& v : m) v = dist(rng) < valid_prob ? 1 : 0;
    return m;
}

}  // namespace

TEST(DistanceTransform, MatchesBruteForceOnRandomMasks) {
    const int W = 37;
    const int H = 53;
    for (std::uint32_t seed : {1u, 2u, 3u}) {
        const std::vector<std::uint8_t> valid = random_mask(W, H, seed, 0.65);
        const NearestSites ns = nearest_sites(W, H, valid);
        ASSERT_EQ(ns.distance.size(), static_cast<std::size_t>(W) * H);
        ASSERT_EQ(ns.site.size(), static_cast<std::size_t>(W) * H);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const std::size_t idx =
                    lin(x, y, W);
                const float expect = brute_distance(W, H, valid, x, y);
                // Exact float equality: both paths sum integer squares.
                EXPECT_EQ(ns.distance[idx], expect)
                    << "seed " << seed << " at (" << x << "," << y << ")";
                if (valid[idx] != 0) {
                    EXPECT_EQ(ns.distance[idx], 0.0f);
                    EXPECT_EQ(ns.site[idx], static_cast<std::int32_t>(idx));
                } else if (std::isfinite(expect)) {
                    const std::int32_t s = ns.site[idx];
                    ASSERT_GE(s, 0);
                    ASSERT_LT(s, W * H);
                    ASSERT_NE(valid[static_cast<std::size_t>(s)], 0);
                    // The reported site really is at the reported distance.
                    const int sx = s % W;
                    const int sy = s / W;
                    const float dx = static_cast<float>(sx - x);
                    const float dy = static_cast<float>(sy - y);
                    EXPECT_EQ(ns.distance[idx],
                              std::sqrt(dx * dx + dy * dy));
                } else {
                    EXPECT_EQ(ns.site[idx], -1);
                }
            }
        }
    }
}

TEST(DistanceTransform, AllInvalidGivesInfinity) {
    const int W = 8;
    const int H = 6;
    std::vector<std::uint8_t> valid(static_cast<std::size_t>(W) * H, 0);
    const NearestSites ns = nearest_sites(W, H, valid);
    for (std::size_t i = 0; i < ns.distance.size(); ++i) {
        EXPECT_FALSE(std::isfinite(ns.distance[i]));
        EXPECT_EQ(ns.site[i], -1);
    }
}

TEST(DistanceTransform, SingleSite) {
    const int W = 9;
    const int H = 7;
    std::vector<std::uint8_t> valid(static_cast<std::size_t>(W) * H, 0);
    const int sx = 3;
    const int sy = 5;
    valid[lin(sx, sy, W)] = 1;
    const NearestSites ns = nearest_sites(W, H, valid);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const std::size_t idx = lin(x, y, W);
            EXPECT_EQ(ns.site[idx], static_cast<std::int32_t>(sy) * W + sx);
            const float dx = static_cast<float>(x - sx);
            const float dy = static_cast<float>(y - sy);
            EXPECT_EQ(ns.distance[idx], std::sqrt(dx * dx + dy * dy));
        }
    }
}

TEST(DistanceTransform, InvalidSizesThrow) {
    EXPECT_THROW(nearest_sites(0, 4, {}), ValueError);
    EXPECT_THROW(nearest_sites(4, -1, {}), ValueError);
    EXPECT_THROW(nearest_sites(4, 4, std::vector<std::uint8_t>(3, 1)),
                 ValueError);
}

namespace {

/// Plane z = 0.3x - 0.2y + 5 over a W x H grid.
float plane_value(int x, int y) {
    return static_cast<float>(0.3 * static_cast<double>(x) -
                              0.2 * static_cast<double>(y) + 5.0);
}

/// Fills `values` with plane_value and clears `valid` inside a disc of the
/// given radius centred at (cx, cy).
void make_plane_with_disc_hole(int w, int h, int cx, int cy, int radius,
                               std::vector<float>& values,
                               std::vector<std::uint8_t>& valid) {
    const std::int64_t r2 = static_cast<std::int64_t>(radius) * radius;
    values.assign(area(w, h), 0.0f);
    valid.assign(area(w, h), 1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t i = lin(x, y, w);
            values[i] = plane_value(x, y);
            const std::int64_t dx = x - cx;
            const std::int64_t dy = y - cy;
            if (dx * dx + dy * dy <= r2) valid[i] = 0;
        }
    }
}

/// Mean absolute error of the originally-hole cells of `values` against the
/// plane. `was_valid` marks the cells that held data before filling.
double mean_hole_error(int w, int h, const std::vector<float>& values,
                       const std::vector<std::uint8_t>& was_valid) {
    double sum = 0.0;
    std::size_t count = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t i = lin(x, y, w);
            if (was_valid[i] != 0) continue;
            sum += std::fabs(static_cast<double>(values[i]) -
                             static_cast<double>(plane_value(x, y)));
            ++count;
        }
    }
    return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

std::size_t count_invalid(const std::vector<std::uint8_t>& valid) {
    std::size_t c = 0;
    for (std::uint8_t v : valid) {
        if (v == 0) ++c;
    }
    return c;
}

}  // namespace

/// Quadratic bowl z = 0.01 * ((x-cx)^2 + (y-cy)^2).
float bowl_value(int x, int y, int cx, int cy) {
    const double dx = static_cast<double>(x - cx);
    const double dy = static_cast<double>(y - cy);
    return static_cast<float>(0.01 * (dx * dx + dy * dy));
}

/// Fills `values` with bowl_value(vx, vy) and clears `valid` inside a disc of
/// the given radius centred at (hx, hy). The bowl vertex and the hole centre
/// are separate so the hole can sit on a slope of the bowl.
void make_bowl_with_hole(int w, int h, int vx, int vy, int hx, int hy,
                         int radius, std::vector<float>& values,
                         std::vector<std::uint8_t>& valid) {
    const std::int64_t r2 = static_cast<std::int64_t>(radius) * radius;
    values.assign(area(w, h), 0.0f);
    valid.assign(area(w, h), 1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t i = lin(x, y, w);
            values[i] = bowl_value(x, y, vx, vy);
            const std::int64_t dx = x - hx;
            const std::int64_t dy = y - hy;
            if (dx * dx + dy * dy <= r2) valid[i] = 0;
        }
    }
}

/// Mean absolute error of the originally-hole cells of `values` against the
/// bowl. `was_valid` marks the cells that held data before filling.
double mean_hole_error_bowl(int w, int h, int cx, int cy,
                            const std::vector<float>& values,
                            const std::vector<std::uint8_t>& was_valid) {
    double sum = 0.0;
    std::size_t count = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t i = lin(x, y, w);
            if (was_valid[i] != 0) continue;
            sum += std::fabs(static_cast<double>(values[i]) -
                             static_cast<double>(bowl_value(x, y, cx, cy)));
            ++count;
        }
    }
    return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

TEST(HoleFill, NaturalNeighbourBeatsNearestOnPlane) {
    const int W = 160;
    const int H = 160;
    std::vector<float> values;
    std::vector<std::uint8_t> valid;
    make_plane_with_disc_hole(W, H, W / 2, H / 2, 20, values, valid);
    const std::vector<std::uint8_t> was_valid = valid;

    double vmin = std::numeric_limits<double>::infinity();
    double vmax = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (was_valid[i] == 0) continue;
        vmin = std::min(vmin, static_cast<double>(values[i]));
        vmax = std::max(vmax, static_cast<double>(values[i]));
    }
    const double range = vmax - vmin;

    std::vector<float> nn = values;
    std::vector<std::uint8_t> nn_valid = valid;
    HoleFillOptions opt;
    opt.method = HoleFill::NaturalNeighbour;
    fill_holes(W, H, nn, nn_valid, opt);

    std::vector<float> near = values;
    std::vector<std::uint8_t> near_valid = valid;
    HoleFillOptions nopt;
    nopt.method = HoleFill::Nearest;
    fill_holes(W, H, near, near_valid, nopt);

    const double nn_err = mean_hole_error(W, H, nn, was_valid);
    const double near_err = mean_hole_error(W, H, near, was_valid);
    std::cout << "[ NN plane ] mean err = " << nn_err << " ("
              << 100.0 * nn_err / range << "% of range), nearest = "
              << near_err << " (" << 100.0 * near_err / range << "% of range)\n";
    // Discrete Sibson has linear precision, so on a plane the error is pure
    // discretisation. It must be far below nearest and a fraction of a percent
    // of the data range.
    EXPECT_LT(nn_err, 0.0025 * range);
    EXPECT_LT(nn_err, 0.3 * near_err);
}

TEST(HoleFill, NaturalNeighbourBeatsIdwAndNearestOnBowl) {
    // Hole on the flank of the bowl, so every rim datum differs strongly from
    // the hole interior. Nearest and IDW then extrapolate the rim values
    // inward, while natural neighbour reproduces the bowl from the stolen-area
    // weights and stays close to the truth.
    const int W = 240;
    const int H = 240;
    const int vx = 120;
    const int vy = 120;
    const int hx = 170;
    const int hy = 120;
    std::vector<float> values;
    std::vector<std::uint8_t> valid;
    make_bowl_with_hole(W, H, vx, vy, hx, hy, 20, values, valid);
    const std::vector<std::uint8_t> was_valid = valid;

    std::vector<float> nn = values;
    std::vector<std::uint8_t> nn_valid = valid;
    HoleFillOptions opt;
    opt.method = HoleFill::NaturalNeighbour;
    fill_holes(W, H, nn, nn_valid, opt);

    std::vector<float> idw = values;
    std::vector<std::uint8_t> idw_valid = valid;
    HoleFillOptions iopt;
    iopt.method = HoleFill::Idw;
    fill_holes(W, H, idw, idw_valid, iopt);

    std::vector<float> near = values;
    std::vector<std::uint8_t> near_valid = valid;
    HoleFillOptions nopt;
    nopt.method = HoleFill::Nearest;
    fill_holes(W, H, near, near_valid, nopt);

    const double nn_err =
        mean_hole_error_bowl(W, H, vx, vy, nn, was_valid);
    const double idw_err =
        mean_hole_error_bowl(W, H, vx, vy, idw, was_valid);
    const double near_err =
        mean_hole_error_bowl(W, H, vx, vy, near, was_valid);
    std::cout << "[ NN bowl ] nn = " << nn_err << ", idw = " << idw_err
              << ", nearest = " << near_err << "\n";
    EXPECT_LT(nn_err, idw_err);
    EXPECT_LT(nn_err, near_err);
    // Natural neighbour should be decisively better, not marginally.
    EXPECT_LT(nn_err, 0.6 * near_err);
}

TEST(HoleFill, NaturalNeighbourMatchesReferenceScatter) {
    // Independent reference implementation of the discrete Sibson scatter:
    // every fillable hole cell p votes f(s(p)) into every q with
    // |q - p| < r(p), and value(q) is the mean. This validates the VOTE SET
    // (which cells contribute, and the strict-interior test) against a from-
    // scratch implementation; the comparison is up to float rounding because
    // the production code accumulates per source-row band and merges the bands,
    // not in flat raster order. Bit-exact determinism across thread counts is
    // covered separately by HoleFill.ThreadCountDoesNotChangeResult.
    const int W = 29;
    const int H = 23;
    const int cx = 14;
    const int cy = 11;
    const int radius = 7;
    std::mt19937 rng(20261019u);
    // Portable draw: see the comment on the other dist above.
    const auto dist = [](std::mt19937& g) { return alphashape::rng::uniform_real(g, -3.0, 7.0); };
    std::vector<float> values(area(W, H), 0.0f);
    std::vector<std::uint8_t> valid(area(W, H), 1);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            values[lin(x, y, W)] = static_cast<float>(dist(rng));
            const int dx = x - cx;
            const int dy = y - cy;
            if (dx * dx + dy * dy <= radius * radius) valid[lin(x, y, W)] = 0;
        }
    }
    const std::vector<std::uint8_t> was_valid = valid;
    const NearestSites edt = nearest_sites(W, H, was_valid);

    // Reference: raster-order scatter over the hole cells.
    std::vector<double> rsum(area(W, H), 0.0);
    std::vector<std::int64_t> rcnt(area(W, H), 0);
    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            const std::size_t pi = lin(px, py, W);
            if (was_valid[pi] != 0) continue;
            const std::int32_t s = edt.site[pi];
            ASSERT_GE(s, 0);
            const double d = static_cast<double>(edt.distance[pi]);
            const std::int64_t r2 =
                static_cast<std::int64_t>(std::llround(d * d));
            const int ri = static_cast<int>(d);
            const double f = static_cast<double>(
                values[static_cast<std::size_t>(s)]);
            for (int qy = std::max(0, py - ri);
                 qy <= std::min(H - 1, py + ri); ++qy) {
                for (int qx = std::max(0, px - ri);
                     qx <= std::min(W - 1, px + ri); ++qx) {
                    const std::int64_t dx = qx - px;
                    const std::int64_t dy = qy - py;
                    if (dx * dx + dy * dy >= r2) continue;
                    const std::size_t qi = lin(qx, qy, W);
                    rsum[qi] += f;
                    ++rcnt[qi];
                }
            }
        }
    }

    HoleFillOptions opt;
    opt.method = HoleFill::NaturalNeighbour;
    std::vector<std::uint8_t> filled_valid = was_valid;
    fill_holes(W, H, values, filled_valid, opt);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const std::size_t i = lin(x, y, W);
            if (was_valid[i] != 0) continue;
            ASSERT_GT(rcnt[i], 0);
            const double expect =
                rsum[i] / static_cast<double>(rcnt[i]);
            EXPECT_NEAR(static_cast<double>(values[i]), expect, 1e-6)
                << "at (" << x << "," << y << ")";
            EXPECT_EQ(filled_valid[i], 1);
        }
    }
    // Data cells must be untouched.
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (was_valid[i] != 0) {
            EXPECT_EQ(filled_valid[i], 1);
        }
    }
}

TEST(HoleFill, NaturalNeighbourIsC0AcrossHoleBoundary) {
    const int W = 128;
    const int H = 128;
    std::vector<float> values;
    std::vector<std::uint8_t> valid;
    make_plane_with_disc_hole(W, H, W / 2, H / 2, 24, values, valid);

    // Slope estimate: the largest adjacent-cell difference among data cells.
    double slope = 0.0;
    double vmin = std::numeric_limits<double>::infinity();
    double vmax = -std::numeric_limits<double>::infinity();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const std::size_t i = lin(x, y, W);
            if (valid[i] == 0) continue;
            vmin = std::min(vmin, static_cast<double>(values[i]));
            vmax = std::max(vmax, static_cast<double>(values[i]));
            if (x + 1 < W && valid[i + 1] != 0) {
                slope = std::max(slope, std::fabs(static_cast<double>(
                                             values[i + 1] - values[i])));
            }
            if (y + 1 < H) {
                const std::size_t d = i + static_cast<std::size_t>(W);
                if (valid[d] != 0) {
                    slope = std::max(slope, std::fabs(static_cast<double>(
                                                 values[d] - values[i])));
                }
            }
        }
    }
    ASSERT_GT(slope, 0.0);

    const std::vector<std::uint8_t> was_valid = valid;
    // Radius of each cell in the pre-fill mask (before filling flips flags).
    const NearestSites edt = nearest_sites(W, H, was_valid);
    HoleFillOptions opt;
    opt.method = HoleFill::NaturalNeighbour;
    fill_holes(W, H, values, valid, opt);

    // Continuity across the hole boundary. The discrete weight of a site at a
    // hole cell q is the number of source cells p with |p-q| < r(p) whose
    // nearest site is that site. For a cell with r(q) = 1 this count is tiny
    // (the disc of radius 1 holds 1 or 5 cells), so the weight of a rim site
    // is quantised to 1/count of the cell's votes rather than its continuous
    // area fraction. At the few rim cells where the 1-deep hole boundary meets
    // a straight run of constant EDT radius, that quantisation lets a single
    // cell carry nearly the whole mean, and the boundary difference rises to a
    // few times the data slope — still only ~1.5% of the data range, against
    // 100% for the rejected gather. Cells at depth r >= 2 have enough votes
    // that their weights are smooth: their boundary difference is 0 to within
    // float rounding.
    const double limit = 4.0 * slope;
    const double range_limit = 0.02 * (vmax - vmin);
    double max_jump = 0.0;
    double deep_jump = 0.0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const std::size_t i = lin(x, y, W);
            if (was_valid[i] != 0) continue;  // only filled cells
            const int r = static_cast<int>(edt.distance[i]);
            const int dxs[4] = {1, -1, 0, 0};
            const int dys[4] = {0, 0, 1, -1};
            for (int k = 0; k < 4; ++k) {
                const int nx = x + dxs[k];
                const int ny = y + dys[k];
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                const std::size_t j = lin(nx, ny, W);
                if (was_valid[j] == 0) continue;
                const double jmp = std::fabs(
                    static_cast<double>(values[i]) -
                    static_cast<double>(values[j]));
                max_jump = std::max(max_jump, jmp);
                if (r >= 2) {
                    deep_jump = std::max(deep_jump, jmp);
                }
                EXPECT_LE(jmp, limit + 1e-4)
                    << "at (" << x << "," << y << ") r=" << r;
                EXPECT_LE(jmp, range_limit + 1e-4)
                    << "at (" << x << "," << y << ") r=" << r;
            }
        }
    }
    std::cout << "[ NN C0 ] max boundary jump = " << max_jump << " ("
              << max_jump / slope << " x slope), r>=2 max = " << deep_jump
              << " (" << deep_jump / slope << " x slope)\n";
    // Depth r >= 2 is smooth to float rounding.
    EXPECT_LT(deep_jump, 1e-4);
}

TEST(HoleFill, IdwStaysWithinDataRange) {
    const int W = 96;
    const int H = 96;
    std::vector<float> values;
    std::vector<std::uint8_t> valid;
    make_plane_with_disc_hole(W, H, 40, 55, 15, values, valid);

    double vmin = std::numeric_limits<double>::infinity();
    double vmax = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (valid[i] == 0) continue;
        vmin = std::min(vmin, static_cast<double>(values[i]));
        vmax = std::max(vmax, static_cast<double>(values[i]));
    }

    HoleFillOptions opt;
    opt.method = HoleFill::Idw;
    const HoleFillReport rep = fill_holes(W, H, values, valid, opt);
    EXPECT_GT(rep.filled, 0u);
    EXPECT_EQ(rep.remaining, 0u);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double v = static_cast<double>(values[i]);
        EXPECT_GE(v, vmin - 1e-3);
        EXPECT_LE(v, vmax + 1e-3);
    }
}

TEST(HoleFill, MaxDistanceLeavesFarCells) {
    const int W = 64;
    const int H = 64;
    std::vector<float> values(static_cast<std::size_t>(W) * H, 2.0f);
    std::vector<std::uint8_t> valid(static_cast<std::size_t>(W) * H, 1);
    // Blank the right half entirely: an unbounded hole.
    for (int y = 0; y < H; ++y) {
        for (int x = W / 2; x < W; ++x) {
            valid[lin(x, y, W)] = 0;
        }
    }
    const NearestSites edt = nearest_sites(W, H, valid);
    const double max_distance = 6.0;

    HoleFillOptions opt;
    opt.method = HoleFill::NaturalNeighbour;
    opt.max_distance = max_distance;
    const std::vector<std::uint8_t> before = valid;
    const HoleFillReport rep = fill_holes(W, H, values, valid, opt);

    std::size_t expected_filled = 0;
    std::size_t holes = 0;
    for (std::size_t i = 0; i < valid.size(); ++i) {
        if (before[i] != 0) continue;
        ++holes;
        if (static_cast<double>(edt.distance[i]) <= max_distance) {
            ++expected_filled;
        }
    }
    EXPECT_EQ(rep.filled, expected_filled);
    EXPECT_EQ(rep.remaining, holes - expected_filled);
    EXPECT_GT(rep.remaining, 0u);
    // Cells beyond max_distance are untouched and still invalid.
    for (std::size_t i = 0; i < valid.size(); ++i) {
        if (before[i] != 0) continue;
        if (static_cast<double>(edt.distance[i]) > max_distance) {
            EXPECT_EQ(valid[i], 0);
        } else {
            EXPECT_EQ(valid[i], 1);
        }
    }
}

TEST(HoleFill, InteriorOnlySkipsBorderConnectedHole) {
    const int W = 48;
    const int H = 48;
    std::vector<float> values(static_cast<std::size_t>(W) * H, 1.0f);
    std::vector<std::uint8_t> valid(static_cast<std::size_t>(W) * H, 1);
    // Enclosed hole: 6x6 at (24, 24).
    for (int y = 22; y <= 27; ++y) {
        for (int x = 22; x <= 27; ++x) {
            valid[lin(x, y, W)] = 0;
        }
    }
    // Border-connected notch: 6x6 touching the top edge.
    for (int y = 0; y <= 5; ++y) {
        for (int x = 8; x <= 13; ++x) {
            valid[lin(x, y, W)] = 0;
        }
    }
    const std::vector<std::uint8_t> before = valid;

    HoleFillOptions opt;
    opt.method = HoleFill::NaturalNeighbour;
    opt.interior_only = true;
    const HoleFillReport rep = fill_holes(W, H, values, valid, opt);

    EXPECT_EQ(rep.filled, 36u);
    EXPECT_EQ(rep.remaining, count_invalid(valid));
    EXPECT_EQ(rep.remaining, 36u);
    // Enclosed hole filled.
    EXPECT_EQ(valid[static_cast<std::size_t>(24) * W + 24], 1);
    // Border-connected notch untouched (top-left corner of the notch and its
    // interior).
    EXPECT_EQ(valid[lin(10, 0, W)], 0);
    EXPECT_EQ(valid[lin(10, 3, W)], 0);
    // Data cells unchanged.
    EXPECT_EQ(valid[static_cast<std::size_t>(40) * W + 40], 1);
    EXPECT_FLOAT_EQ(values[static_cast<std::size_t>(40) * W + 40], 1.0f);
}

TEST(HoleFill, RegionFillsBorderConnectedHoleInsideIt) {
    // The failure this guards: the nodata rim OUTSIDE the survey surrounds the
    // data, so a real hole inside the cover reaches the raster border through
    // invalid cells. A fill region states the cover explicitly, so the hole is
    // filled even though `interior_only` would have skipped it.
    const int W = 48;
    const int H = 48;
    std::vector<float> values(area(W, H), 2.0f);
    std::vector<std::uint8_t> valid(area(W, H), 1);
    // A 6-wide column of holes from the top border down to the middle: connected
    // to the raster border, so `interior_only` alone would leave it nodata.
    for (int y = 0; y <= 23; ++y) {
        for (int x = 20; x <= 25; ++x) {
            valid[lin(x, y, W)] = 0;
        }
    }
    // The region: two overlapping "footprints" covering x in [8, 39] -- the
    // hole is inside their union, the rest of the raster is outside it.
    HoleFillOptions opt;
    opt.method = HoleFill::Nearest;
    opt.interior_only = true;  // ignored because a region is given
    opt.region.assign(area(W, H), 0);
    for (int y = 0; y < H; ++y) {
        for (int x = 8; x <= 39; ++x) {
            opt.region[lin(x, y, W)] = 1;
        }
    }
    const HoleFillReport rep = fill_holes(W, H, values, valid, opt);

    // Every hole cell is inside the region: all 24 * 6 of them are filled.
    EXPECT_EQ(rep.filled, 24u * 6u);
    EXPECT_EQ(rep.remaining, 0u);
    EXPECT_EQ(valid[lin(20, 0, W)], 1);
    EXPECT_EQ(valid[lin(22, 23, W)], 1);
    EXPECT_FLOAT_EQ(values[lin(22, 10, W)], 2.0f);  // nearest cell's value
}

TEST(HoleFill, RegionLeavesCellsOutsideItInvalid) {
    // A hole spanning both the region and the ground beyond it: only the cells
    // where region[i] != 0 may be filled, the rest stay nodata.
    const int W = 40;
    const int H = 40;
    std::vector<float> values(area(W, H), 3.0f);
    std::vector<std::uint8_t> valid(area(W, H), 1);
    for (int y = 10; y <= 29; ++y) {
        for (int x = 10; x <= 29; ++x) {
            valid[lin(x, y, W)] = 0;
        }
    }
    HoleFillOptions opt;
    opt.method = HoleFill::Nearest;
    opt.region.assign(area(W, H), 0);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < 20; ++x) {  // the region: the western half only
            opt.region[lin(x, y, W)] = 1;
        }
    }
    const HoleFillReport rep = fill_holes(W, H, values, valid, opt);

    EXPECT_EQ(rep.filled, 20u * 10u);  // hole cells with x in [10, 19]
    EXPECT_EQ(valid[lin(15, 20, W)], 1);
    EXPECT_EQ(valid[lin(25, 20, W)], 0);  // inside the hole, outside the region
    EXPECT_GT(rep.remaining, 0u);
}

TEST(HoleFill, RegionKeepsAConcaveGapInvalid) {
    // Two separated footprints with a gap between them. The union of two
    // rectangles is concave; a convex hull of the region would fill the gap.
    const int W = 64;
    const int H = 32;
    std::vector<float> values(area(W, H), 4.0f);
    std::vector<std::uint8_t> valid(area(W, H), 1);
    for (int y = 8; y <= 23; ++y) {
        for (int x = 8; x <= 55; ++x) {
            valid[lin(x, y, W)] = 0;
        }
    }
    HoleFillOptions opt;
    opt.method = HoleFill::Nearest;
    opt.region.assign(area(W, H), 0);
    for (int y = 8; y <= 23; ++y) {
        for (int x = 8; x <= 23; ++x) {  // west strip
            opt.region[lin(x, y, W)] = 1;
        }
        for (int x = 40; x <= 55; ++x) {  // east strip
            opt.region[lin(x, y, W)] = 1;
        }
    }
    const HoleFillReport rep = fill_holes(W, H, values, valid, opt);

    EXPECT_EQ(rep.filled, 2u * 16u * 16u);
    EXPECT_EQ(valid[lin(15, 15, W)], 1);  // inside the west strip
    EXPECT_EQ(valid[lin(47, 15, W)], 1);  // inside the east strip
    EXPECT_EQ(valid[lin(31, 15, W)], 0);  // the gap: neither footprint
    EXPECT_EQ(valid[lin(32, 8, W)], 0);
}

TEST(HoleFill, RegionSizeMismatchThrows) {
    std::vector<float> values(16, 0.0f);
    std::vector<std::uint8_t> valid(16, 1);
    HoleFillOptions opt;
    opt.region.assign(15, 1);
    EXPECT_THROW(fill_holes(4, 4, values, valid, opt), ValueError);
}

TEST(HoleFill, ThreadCountDoesNotChangeResult) {
    const int W = 128;
    const int H = 128;
    std::vector<float> values;
    std::vector<std::uint8_t> valid;
    make_plane_with_disc_hole(W, H, 70, 60, 20, values, valid);

    for (HoleFill m : {HoleFill::Nearest, HoleFill::Idw,
                       HoleFill::NaturalNeighbour}) {
        HoleFillOptions opt;
        opt.method = m;

        std::vector<float> one = values;
        std::vector<std::uint8_t> one_valid = valid;
        std::vector<float> four = values;
        std::vector<std::uint8_t> four_valid = valid;

#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        fill_holes(W, H, one, one_valid, opt);
#ifdef _OPENMP
        omp_set_num_threads(4);
#endif
        fill_holes(W, H, four, four_valid, opt);
#ifdef _OPENMP
        omp_set_num_threads(omp_get_num_procs());
#endif

        ASSERT_EQ(one_valid, four_valid) << "method " << static_cast<int>(m);
        for (std::size_t i = 0; i < one.size(); ++i) {
            // Bit-identical.
            ASSERT_EQ(one[i], four[i])
                << "method " << static_cast<int>(m) << " at " << i;
        }
    }
}

TEST(HoleFill, InvalidArgumentsThrow) {
    std::vector<float> values(16, 0.0f);
    std::vector<std::uint8_t> valid(16, 1);
    HoleFillOptions opt;

    EXPECT_THROW(fill_holes(0, 4, values, valid, opt), ValueError);
    EXPECT_THROW(fill_holes(4, 0, values, valid, opt), ValueError);
    EXPECT_THROW(fill_holes(5, 5, values, valid, opt), ValueError);
    {  // size mismatch
        std::vector<std::uint8_t> bad(3, 1);
        EXPECT_THROW(fill_holes(4, 4, values, bad, opt), ValueError);
    }
    opt.idw_power = -1.0;
    EXPECT_THROW(fill_holes(4, 4, values, valid, opt), ValueError);
    opt = HoleFillOptions{};
    opt.idw_neighbours = 0;
    EXPECT_THROW(fill_holes(4, 4, values, valid, opt), ValueError);
    opt = HoleFillOptions{};
    opt.idw_search_radius = 0.0;
    EXPECT_THROW(fill_holes(4, 4, values, valid, opt), ValueError);
    opt = HoleFillOptions{};
    opt.max_distance = -1.0;
    EXPECT_THROW(fill_holes(4, 4, values, valid, opt), ValueError);
}

TEST(HoleFill, LargeRasterNaturalNeighbourIsFast) {
    const int W = 1024;
    const int H = 1024;
    std::vector<float> values;
    std::vector<std::uint8_t> valid;
    make_plane_with_disc_hole(W, H, 512, 512, 100, values, valid);

    HoleFillOptions opt;
    opt.method = HoleFill::NaturalNeighbour;
    const auto t0 = std::chrono::steady_clock::now();
    const HoleFillReport rep = fill_holes(W, H, values, valid, opt);
    const auto t1 = std::chrono::steady_clock::now();
    const double secs =
        std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[ NN 1024x1024 r=100 ] " << secs << " s, filled "
              << rep.filled << "\n";
    EXPECT_GT(rep.filled, 0u);
    EXPECT_LT(secs, 3.0);
}
