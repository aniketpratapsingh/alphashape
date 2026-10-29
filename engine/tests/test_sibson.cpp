// Exact Sibson coordinates against a brute-force Voronoi clipping oracle, plus
// linear reproduction, hull handling and the fill_holes() integration.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/algorithm/delaunay2d.hpp"
#include "common/algorithm/elevation/interpolate.hpp"
#include "common/algorithm/elevation/sibson.hpp"
#include "perf_gate.hpp"

namespace alphashape::elevation {
namespace {

using V = Eigen::Vector2d;
using Polygon = std::vector<V>;

std::vector<V> unit_sites(int n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<V> out = {V(0, 0), V(1, 0), V(1, 1), V(0, 1)};
    for (int i = 0; i < n; ++i) {
        out.emplace_back(static_cast<double>(rng()) / 4294967296.0,
                         static_cast<double>(rng()) / 4294967296.0);
    }
    return out;
}

V uniform_point(std::mt19937& rng, double lo, double hi) {
    const double a = static_cast<double>(rng()) / 4294967296.0;
    const double b = static_cast<double>(rng()) / 4294967296.0;
    return {lo + (hi - lo) * a, lo + (hi - lo) * b};
}

// Keep the part of `poly` where (x - m) . n <= 0 (Sutherland-Hodgman).
Polygon clip(const Polygon& poly, const V& m, const V& n) {
    Polygon out;
    for (std::size_t i = 0; i < poly.size(); ++i) {
        const V& a = poly[i];
        const V& b = poly[(i + 1) % poly.size()];
        const double da = (a - m).dot(n);
        const double db = (b - m).dot(n);
        if (da <= 0.0) out.push_back(a);
        if ((da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0)) {
            out.push_back(a + (b - a) * (da / (da - db)));
        }
    }
    return out;
}

double area(const Polygon& p) {
    double twice = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const V& a = p[i];
        const V& b = p[(i + 1) % p.size()];
        twice += a.x() * b.y() - b.x() * a.y();
    }
    return 0.5 * std::abs(twice);
}

// Sibson weights by definition: the share of q's new Voronoi cell that lies in
// each site's old cell.
std::vector<double> brute_force(const std::vector<V>& s, const V& q) {
    Polygon cell = {V(-10, -10), V(11, -10), V(11, 11), V(-10, 11)};
    for (const V& p : s) {
        cell = clip(cell, 0.5 * (p + q), p - q);
    }
    const double total = area(cell);
    std::vector<double> w(s.size(), 0.0);
    for (std::size_t i = 0; i < s.size(); ++i) {
        Polygon part = cell;
        for (std::size_t j = 0; j < s.size() && !part.empty(); ++j) {
            if (j != i) part = clip(part, 0.5 * (s[i] + s[j]), s[j] - s[i]);
        }
        w[i] = part.size() >= 3 ? area(part) / total : 0.0;
    }
    return w;
}

TEST(DemSibson, ExactAgainstBruteForce) {
    const auto sites = unit_sites(196, 5);
    const Delaunay2D dt(sites);
    std::mt19937 rng(9);
    std::vector<SibsonWeight> w;
    int hint = -1;
    for (int k = 0; k < 500; ++k) {
        const V q = uniform_point(rng, 0.02, 0.98);
        ASSERT_TRUE(sibson_coordinates(dt, q, w, hint));
        std::vector<double> ours(sites.size(), 0.0);
        for (const SibsonWeight& sw : w) ours[static_cast<std::size_t>(sw.site)] = sw.weight;
        const auto ref = brute_force(sites, q);
        for (std::size_t i = 0; i < sites.size(); ++i) {
            ASSERT_NEAR(ours[i], ref[i], 1e-9) << "query " << k << " site " << i;
        }
    }
}

TEST(DemSibson, LinearReproduction) {
    const auto sites = unit_sites(300, 21);
    const Delaunay2D dt(sites);
    const auto f = [](const V& p) { return 2.0 * p.x() - 3.0 * p.y() + 0.5; };
    std::mt19937 rng(4);
    std::vector<SibsonWeight> w;
    int hint = -1;
    for (int k = 0; k < 2000; ++k) {
        const V q = uniform_point(rng, 0.0, 1.0);
        ASSERT_TRUE(sibson_coordinates(dt, q, w, hint));
        double v = 0.0;
        double sum = 0.0;
        V centroid = V::Zero();
        for (const SibsonWeight& sw : w) {
            const V& p = sites[static_cast<std::size_t>(sw.site)];
            v += sw.weight * f(p);
            sum += sw.weight;
            centroid += sw.weight * p;
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
        EXPECT_NEAR(v, f(q), 1e-9);
        EXPECT_NEAR((centroid - q).norm(), 0.0, 1e-9);
    }
}

TEST(DemSibson, OutOfHullAndBoundary) {
    const Delaunay2D dt(unit_sites(50, 2));
    std::vector<SibsonWeight> w;
    int hint = -1;
    EXPECT_FALSE(sibson_coordinates(dt, V(1.5, 0.5), w, hint));
    EXPECT_TRUE(w.empty());
    EXPECT_FALSE(sibson_coordinates(dt, V(-1e-9, 0.5), w, hint));

    // On the hull edge between corners 0 and 1: linear weights along it.
    ASSERT_TRUE(sibson_coordinates(dt, V(0.25, 0.0), w, hint));
    ASSERT_EQ(w.size(), 2U);
    for (const SibsonWeight& sw : w) {
        EXPECT_NEAR(sw.weight, sw.site == 0 ? 0.75 : 0.25, 1e-12);
    }
    // Exactly on a site.
    ASSERT_TRUE(sibson_coordinates(dt, V(1, 1), w, hint));
    ASSERT_EQ(w.size(), 1U);
    EXPECT_EQ(w[0].site, 2);
    EXPECT_EQ(w[0].weight, 1.0);
}

TEST(DemSibson, Deterministic) {
    const auto sites = unit_sites(400, 13);
    const Delaunay2D a(sites);
    const Delaunay2D b(sites);
    std::mt19937 rng(77);
    std::vector<SibsonWeight> wa;
    std::vector<SibsonWeight> wb;
    int ha = -1;
    int hb = -1;
    for (int k = 0; k < 200; ++k) {
        const V q = uniform_point(rng, 0.0, 1.0);
        ASSERT_TRUE(sibson_coordinates(a, q, wa, ha));
        ASSERT_TRUE(sibson_coordinates(b, q, wb, hb));
        ASSERT_EQ(wa.size(), wb.size());
        for (std::size_t i = 0; i < wa.size(); ++i) {
            EXPECT_EQ(wa[i].site, wb[i].site);
            EXPECT_EQ(wa[i].weight, wb[i].weight);
        }
    }
}

TEST(DemSibson, FillHolesExactOnPlane) {
    const int W = 64;
    const int H = 48;
    std::vector<float> values(static_cast<std::size_t>(W * H));
    std::vector<std::uint8_t> valid(values.size(), 1);
    const auto plane = [](int x, int y) { return 0.25 * x - 0.5 * y + 100.0; };
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const std::size_t i = static_cast<std::size_t>(y * W + x);
            values[i] = static_cast<float>(plane(x, y));
            const double dx = x - 30.5;
            const double dy = y - 20.0;
            if (dx * dx + dy * dy < 12.0 * 12.0 || (x > 50 && x < 60 && y > 5 && y < 40)) {
                valid[i] = 0;
                values[i] = 0.0f;
            }
        }
    }
    HoleFillOptions options;
    options.sibson_exact_max_sites = 20000;
    const HoleFillReport report = fill_holes(W, H, values, valid, options);
    EXPECT_EQ(report.remaining, 0U);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const std::size_t i = static_cast<std::size_t>(y * W + x);
            ASSERT_EQ(valid[i], 1);
            ASSERT_NEAR(static_cast<double>(values[i]), plane(x, y), 1e-3) << x << "," << y;
        }
    }
    std::vector<std::uint8_t> bad(4, 0);
    std::vector<float> v4(4, 0.0f);
    HoleFillOptions negative;
    negative.sibson_exact_max_sites = -1;
    EXPECT_ANY_THROW(fill_holes(2, 2, v4, bad, negative));
}

TEST(DemSibson, Perf20kSites100kQueries) {
    std::mt19937 rng(31);
    std::vector<V> sites = {V(0, 0), V(1, 0), V(1, 1), V(0, 1)};
    for (int i = 0; i < 20000; ++i) sites.push_back(uniform_point(rng, 0.0, 1.0));
    const auto t0 = std::chrono::steady_clock::now();
    const Delaunay2D dt(sites);
    std::vector<SibsonWeight> w;
    int hint = -1;
    // Queries along rows, the order fill_holes visits a hole in.
    for (int r = 0; r < 250; ++r) {
        for (int c = 0; c < 400; ++c) {
            const V q((c + 0.5) / 400.0, (r + 0.5) / 250.0);
            ASSERT_TRUE(sibson_coordinates(dt, q, w, hint));
        }
    }
    const double s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    RecordProperty("seconds", std::to_string(s));
    if (test::perf_asserts_enabled()) {
        EXPECT_LE(s, 10.0);
    }
}

}  // namespace
}  // namespace alphashape::elevation
