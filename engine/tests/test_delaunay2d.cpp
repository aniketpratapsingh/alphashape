// Delaunay2D: empty-circumcircle property, topology, degenerate inputs.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "base/algorithm/delaunay2d.hpp"
#include "base/algorithm/predicates.hpp"
#include "perf_gate.hpp"

namespace alphashape {
namespace {

using V = Eigen::Vector2d;

const V& at(const std::vector<V>& p, int i) { return p[static_cast<std::size_t>(i)]; }

std::vector<V> random_sites(int n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<V> out;
    out.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        // Raw generator output keeps the sequence identical on every standard library.
        out.emplace_back(static_cast<double>(rng()) / 4294967296.0,
                         static_cast<double>(rng()) / 4294967296.0);
    }
    return out;
}

// Every triangle is counter-clockwise, no site is strictly inside any
// circumcircle, and every undirected edge is shared by at most two triangles.
void expect_valid_delaunay(const Delaunay2D& dt) {
    const auto& p = dt.sites();
    const auto tris = dt.triangles();
    ASSERT_FALSE(tris.empty());
    std::set<std::pair<int, int>> directed;
    for (const auto& t : tris) {
        ASSERT_EQ(orient2d(at(p, t[0]), at(p, t[1]), at(p, t[2])), 1);
        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_TRUE(directed.insert({t[i], t[(i + 1) % 3]}).second)
                << "directed edge used twice";
        }
    }
    for (const auto& t : tris) {
        for (std::size_t s = 0; s < p.size(); ++s) {
            if (dt.site_alias(static_cast<int>(s)) != static_cast<int>(s)) {
                continue;
            }
            ASSERT_LE(incircle(at(p, t[0]), at(p, t[1]), at(p, t[2]), p[s]), 0)
                << "site " << s << " inside a circumcircle";
        }
    }
}

int representative_count(const Delaunay2D& dt) {
    int n = 0;
    for (std::size_t i = 0; i < dt.sites().size(); ++i) {
        n += dt.site_alias(static_cast<int>(i)) == static_cast<int>(i) ? 1 : 0;
    }
    return n;
}

TEST(Delaunay2D, EmptyCircumcircle) {
    const Delaunay2D dt(random_sites(2000, 7));
    ASSERT_FALSE(dt.degenerate());
    expect_valid_delaunay(dt);
}

TEST(Delaunay2D, EulerAndHull) {
    const Delaunay2D dt(random_sites(500, 11));
    const auto hull = dt.convex_hull();
    const int n = representative_count(dt);
    const int h = static_cast<int>(hull.size());
    EXPECT_EQ(static_cast<int>(dt.triangles().size()), 2 * n - h - 2);
    const auto& p = dt.sites();
    for (std::size_t i = 0; i < hull.size(); ++i) {
        const V& a = at(p, hull[i]);
        const V& b = at(p, hull[(i + 1) % hull.size()]);
        for (std::size_t s = 0; s < p.size(); ++s) {
            EXPECT_GE(orient2d(a, b, p[s]), 0) << "site outside the hull";
        }
        EXPECT_TRUE(dt.is_on_hull(hull[i]));
    }
}

TEST(Delaunay2D, CocircularRingAndGrid) {
    std::vector<V> ring;
    for (int k = 0; k < 32; ++k) {
        const double a = 6.283185307179586 * k / 32.0;
        ring.emplace_back(std::cos(a), std::sin(a));
    }
    ring.emplace_back(0.0, 0.0);
    const Delaunay2D r1(ring);
    const Delaunay2D r2(ring);
    expect_valid_delaunay(r1);
    EXPECT_EQ(r1.triangles(), r2.triangles());

    std::vector<V> grid;
    for (int y = 0; y < 20; ++y) {
        for (int x = 0; x < 20; ++x) {
            grid.emplace_back(x, y);
        }
    }
    const Delaunay2D g1(grid);
    const Delaunay2D g2(grid);
    expect_valid_delaunay(g1);
    EXPECT_EQ(g1.triangles(), g2.triangles());
    // A 20 x 20 grid has 76 hull sites (collinear ones included) and 2*400-76-2.
    EXPECT_EQ(g1.convex_hull().size(), 76U);
    EXPECT_EQ(g1.triangles().size(), 2U * 400U - 76U - 2U);
}

TEST(Delaunay2D, Duplicates) {
    auto sites = random_sites(200, 3);
    sites.push_back(sites[5]);
    sites.push_back(sites[17]);
    sites.push_back(sites[5]);
    const Delaunay2D dt(sites);
    EXPECT_EQ(dt.site_alias(200), 5);
    EXPECT_EQ(dt.site_alias(201), 17);
    EXPECT_EQ(dt.site_alias(202), 5);
    EXPECT_EQ(dt.site_alias(5), 5);
    expect_valid_delaunay(dt);
    for (const auto& t : dt.triangles()) {
        for (int v : t) {
            EXPECT_LT(v, 200);
        }
    }
}

TEST(Delaunay2D, Collinear) {
    std::vector<V> line;
    for (int i = 0; i < 10; ++i) {
        line.emplace_back(0.1 * i, 0.3 * i);
    }
    const Delaunay2D dt(line);
    EXPECT_TRUE(dt.degenerate());
    EXPECT_TRUE(dt.triangles().empty());
    EXPECT_EQ(dt.locate(V(0.0, 0.0)), -1);
    EXPECT_TRUE(Delaunay2D({V(0, 0), V(1, 1)}).degenerate());

    // Collinear start followed by an off-line site still triangulates.
    line.emplace_back(5.0, -2.0);
    const Delaunay2D fan(line);
    EXPECT_FALSE(fan.degenerate());
    expect_valid_delaunay(fan);
}

TEST(Delaunay2D, LocateAndNeighbours) {
    const std::vector<V> sq = {V(0, 0), V(1, 0), V(1, 1), V(0, 1), V(0.5, 0.4)};
    const Delaunay2D dt(sq);
    expect_valid_delaunay(dt);
    const int t = dt.locate(V(0.5, 0.1));
    ASSERT_GE(t, 0);
    const auto& tri = dt.triangle(t);
    const auto& p = dt.sites();
    EXPECT_GE(orient2d(at(p, tri.v[0]), at(p, tri.v[1]), V(0.5, 0.1)), 0);
    EXPECT_GE(orient2d(at(p, tri.v[1]), at(p, tri.v[2]), V(0.5, 0.1)), 0);
    EXPECT_GE(orient2d(at(p, tri.v[2]), at(p, tri.v[0]), V(0.5, 0.1)), 0);
    EXPECT_EQ(dt.locate(V(2.0, 0.5)), -1);
    EXPECT_GE(dt.locate(V(1.0, 0.5)), 0) << "a point on the hull is inside";

    // The centre site sees all four corners, counter-clockwise.
    const auto nb = dt.vertex_neighbours(4);
    ASSERT_EQ(nb.size(), 4U);
    for (std::size_t i = 0; i < nb.size(); ++i) {
        EXPECT_EQ(orient2d(p[4], at(p, nb[i]), at(p, nb[(i + 1) % nb.size()])), 1);
    }
    EXPECT_FALSE(dt.is_on_hull(4));
    EXPECT_TRUE(dt.is_on_hull(0));
}

TEST(Delaunay2D, Perf100k) {
    const auto sites = random_sites(100000, 42);
    const auto t0 = std::chrono::steady_clock::now();
    const Delaunay2D dt(sites);
    const double s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    RecordProperty("build_seconds", std::to_string(s));
    EXPECT_EQ(static_cast<int>(dt.triangles().size()),
              2 * 100000 - static_cast<int>(dt.convex_hull().size()) - 2);
    if (test::perf_asserts_enabled()) {
        EXPECT_LE(s, 2.0);
    }
}

}  // namespace
}  // namespace alphashape
