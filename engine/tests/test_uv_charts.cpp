// UV chart segmentation + flattening.
//
// Covers: the partition invariant (every face in exactly one chart, corner_uv
// sized 3*f), chart counts on hand-built inputs (box -> 6 sides, a corner -> 2,
// a flat grid -> 1), isometry of planar charts, orientation, sphere sanity
// (cone adherence, overlap-freedom, area distortion), max_chart_faces capping,
// argument validation, determinism and a timing bound.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/uv_charts.hpp"

namespace alphashape {

namespace {

// ---------------------------------------------------------------------------
// Mesh generators (all built by hand: CCW wound, outward normals)
// ---------------------------------------------------------------------------

/// Icosphere: subdivide an icosahedron and project new vertices onto the unit
/// sphere. Shared midpoints keep it a closed manifold.
Mesh make_icosphere(int subdivisions) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<Eigen::Vector3d> verts = {
        {-1, t, 0}, {1, t, 0},  {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
        {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1},  {-t, 0, -1}, {-t, 0, 1},
    };
    for (Eigen::Vector3d& v : verts) {
        v.normalize();
    }
    std::vector<Eigen::Vector3i> faces = {
        {0, 11, 5}, {0, 5, 1},   {0, 1, 7},  {0, 7, 10}, {0, 10, 11},
        {1, 5, 9},  {5, 11, 4},  {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4},  {3, 4, 2},   {3, 2, 6},  {3, 6, 8},  {3, 8, 9},
        {4, 9, 5},  {2, 4, 11},  {6, 2, 10}, {8, 6, 7},  {9, 8, 1},
    };
    for (int s = 0; s < subdivisions; ++s) {
        std::unordered_map<std::uint64_t, int> midpoints;
        std::vector<Eigen::Vector3i> next;
        next.reserve(faces.size() * 4);
        const auto midpoint = [&](int a, int b) -> int {
            const int lo = std::min(a, b);
            const int hi = std::max(a, b);
            const std::uint64_t key =
                (static_cast<std::uint64_t>(lo) << 32U) | static_cast<std::uint64_t>(hi);
            const auto it = midpoints.find(key);
            if (it != midpoints.end()) {
                return it->second;
            }
            Eigen::Vector3d mid = (verts[static_cast<std::size_t>(a)] +
                                   verts[static_cast<std::size_t>(b)]) *
                                  0.5;
            mid.normalize();
            verts.push_back(mid);
            const int index = static_cast<int>(verts.size()) - 1;
            midpoints.emplace(key, index);
            return index;
        };
        for (const Eigen::Vector3i& f : faces) {
            const int a = midpoint(f[0], f[1]);
            const int b = midpoint(f[1], f[2]);
            const int c = midpoint(f[2], f[0]);
            next.push_back({f[0], a, c});
            next.push_back({a, f[1], b});
            next.push_back({c, b, f[2]});
            next.push_back({a, b, c});
        }
        faces.swap(next);
    }
    Mesh mesh;
    mesh.vertices = std::move(verts);
    mesh.faces = std::move(faces);
    return mesh;
}

/// Unit cube [0,1]^3, twelve triangles, six planar sides.
Mesh make_box() {
    Mesh mesh;
    mesh.vertices = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                     {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    mesh.faces = {
        {0, 2, 1}, {0, 3, 2},  // z = 0, outward -z
        {4, 5, 6}, {4, 6, 7},  // z = 1, outward +z
        {0, 1, 5}, {0, 5, 4},  // y = 0, outward -y
        {3, 7, 6}, {3, 6, 2},  // y = 1, outward +y
        {0, 4, 7}, {0, 7, 3},  // x = 0, outward -x
        {1, 2, 6}, {1, 6, 5},  // x = 1, outward +x
    };
    return mesh;
}

/// n x n triangles-per-side grid on z = 0 (edge lengths 1/n), outward +z.
Mesh make_flat_grid(int n) {
    Mesh mesh;
    const int stride = n + 1;
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            mesh.vertices.push_back(Eigen::Vector3d(static_cast<double>(i) / n,
                                                    static_cast<double>(j) / n, 0.0));
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const int a = j * stride + i;
            const int b = a + 1;
            const int c = a + stride + 1;
            const int d = a + stride;
            mesh.faces.push_back({a, b, c});
            mesh.faces.push_back({a, c, d});
        }
    }
    return mesh;
}

/// An "L" corner: a floor patch (z = 0, outward +z) meeting a wall (y = 0,
/// outward -y) along the line y = 0, z = 0. n = cells per side.
Mesh make_corner(int n) {
    Mesh mesh;
    // Floor: x, y in [0,1], z = 0, outward +z.
    const int stride = n + 1;
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            mesh.vertices.push_back(Eigen::Vector3d(static_cast<double>(i) / n,
                                                    static_cast<double>(j) / n, 0.0));
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const int a = j * stride + i;
            const int b = a + 1;
            const int c = a + stride + 1;
            const int d = a + stride;
            mesh.faces.push_back({a, b, c});
            mesh.faces.push_back({a, c, d});
        }
    }
    // Wall: x in [0,1], z in [0,1], y = 0, outward -y. Wound so that
    // (v1-v0) x (v2-v0) points along -y.
    const int base = static_cast<int>(mesh.vertices.size());
    for (int j = 0; j <= n; ++j) {  // j = z index
        for (int i = 0; i <= n; ++i) {  // i = x index
            mesh.vertices.push_back(Eigen::Vector3d(static_cast<double>(i) / n, 0.0,
                                                    static_cast<double>(j) / n));
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const int a = base + j * stride + i;
            const int b = a + 1;
            const int c = a + stride + 1;
            const int d = a + stride;
            // Reversed winding relative to the floor: normal flips to -y.
            mesh.faces.push_back({a, c, b});
            mesh.faces.push_back({a, d, c});
        }
    }
    return mesh;
}

/// A flat ribbon coiled into a spiral lying in z = 0 (outward +z). Every face is
/// coplanar, so the cone test can never separate the turns: the ring spacing is
/// far smaller than the ribbon width, so the flattened ribbon overlaps itself
/// and only the overlap machinery can split it.
Mesh make_spiral_ribbon(int segments) {
    Mesh mesh;
    const double dtheta = 0.5;
    const double r0 = 1.0;
    const double dr = 0.005;  // radial advance per segment; a full turn advances
                              // 2*pi/dtheta * dr ~= 0.063 << width 0.3
    const double half_width = 0.15;
    for (int i = 0; i <= segments; ++i) {
        const double theta = static_cast<double>(i) * dtheta;
        const double r = r0 - static_cast<double>(i) * dr;
        const double cx = std::cos(theta);
        const double cy = std::sin(theta);
        mesh.vertices.push_back(Eigen::Vector3d((r - half_width) * cx, (r - half_width) * cy, 0.0));
        mesh.vertices.push_back(Eigen::Vector3d((r + half_width) * cx, (r + half_width) * cy, 0.0));
    }
    for (int i = 0; i < segments; ++i) {
        const int a = i * 2;      // inner rail, i
        const int b = i * 2 + 1;  // outer rail, i
        const int c = a + 2;      // inner rail, i+1
        const int d = a + 3;      // outer rail, i+1
        mesh.faces.push_back({a, b, d});
        mesh.faces.push_back({a, d, c});
    }
    return mesh;
}

// ---------------------------------------------------------------------------
// Verification helpers
// ---------------------------------------------------------------------------
/// Signed area of a projected triangle.
double signed_area(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& c) {
    return 0.5 * ((b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x()));
}

/// Every face appears in exactly one chart and each chart's corner_uv is 3*f.
void expect_partition(const Mesh& mesh, const std::vector<Chart>& charts) {
    std::vector<int> seen(mesh.faces.size(), 0);
    for (const Chart& chart : charts) {
        ASSERT_EQ(chart.corner_uv.size(), chart.faces.size() * 3U);
        for (int f : chart.faces) {
            ASSERT_GE(f, 0);
            ASSERT_LT(f, static_cast<int>(mesh.faces.size()));
            ++seen[static_cast<std::size_t>(f)];
        }
    }
    for (int f = 0; f < static_cast<int>(mesh.faces.size()); ++f) {
        EXPECT_EQ(seen[static_cast<std::size_t>(f)], 1) << "face " << f;
    }
}

/// Every projected triangle is CCW (positive area).
void expect_orientation_positive(const std::vector<Chart>& charts) {
    for (const Chart& chart : charts) {
        for (std::size_t i = 0; i < chart.faces.size(); ++i) {
            const double area = signed_area(chart.corner_uv[i * 3], chart.corner_uv[i * 3 + 1],
                                            chart.corner_uv[i * 3 + 2]);
            EXPECT_GT(area, 0.0) << "chart face " << chart.faces[i];
        }
    }
}

/// Projected edge lengths equal their 3-D lengths on a planar chart.
void expect_isometry(const Mesh& mesh, const std::vector<Chart>& charts, double tol) {
    for (const Chart& chart : charts) {
        for (std::size_t i = 0; i < chart.faces.size(); ++i) {
            const Eigen::Vector3i& tri = mesh.faces[static_cast<std::size_t>(chart.faces[i])];
            for (int e = 0; e < 3; ++e) {
                const int a = tri[e];
                const int b = tri[(e + 1) % 3];
                const double len3d = (mesh.vertices[static_cast<std::size_t>(a)] -
                                      mesh.vertices[static_cast<std::size_t>(b)])
                                         .norm();
                const double len2d = (chart.corner_uv[i * 3 + static_cast<std::size_t>(e)] -
                                      chart.corner_uv[i * 3 + static_cast<std::size_t>((e + 1) % 3)])
                                         .norm();
                EXPECT_NEAR(len2d, len3d, tol * std::max(1.0, len3d));
            }
        }
    }
}

/// 3-D area of mesh face f.
double face_area_3d(const Mesh& mesh, int f) {
    const Eigen::Vector3i& tri = mesh.faces[static_cast<std::size_t>(f)];
    const Eigen::Vector3d& p0 = mesh.vertices[static_cast<std::size_t>(tri[0])];
    const Eigen::Vector3d& p1 = mesh.vertices[static_cast<std::size_t>(tri[1])];
    const Eigen::Vector3d& p2 = mesh.vertices[static_cast<std::size_t>(tri[2])];
    return 0.5 * (p1 - p0).cross(p2 - p0).norm();
}

/// Independent SAT overlap test (no shared vertices assumed). Mirrors intent,
/// not the implementation: the test re-derives the predicate from scratch.
bool uv_triangles_overlap(const Eigen::Vector2d* a, const Eigen::Vector2d* b) {
    Eigen::Vector2d axes[6];
    for (int k = 0; k < 3; ++k) {
        const Eigen::Vector2d ea = a[(k + 1) % 3] - a[k];
        const Eigen::Vector2d eb = b[(k + 1) % 3] - b[k];
        axes[k] = Eigen::Vector2d(-ea.y(), ea.x());
        axes[k + 3] = Eigen::Vector2d(-eb.y(), eb.x());
    }
    for (const Eigen::Vector2d& raw : axes) {
        const double len = raw.norm();
        if (!(len > 0.0)) {
            continue;
        }
        const Eigen::Vector2d axis = raw / len;
        double a_lo = a[0].dot(axis);
        double a_hi = a_lo;
        double b_lo = b[0].dot(axis);
        double b_hi = b_lo;
        for (int k = 1; k < 3; ++k) {
            a_lo = std::min(a_lo, a[k].dot(axis));
            a_hi = std::max(a_hi, a[k].dot(axis));
            b_lo = std::min(b_lo, b[k].dot(axis));
            b_hi = std::max(b_hi, b[k].dot(axis));
        }
        if (std::min(a_hi, b_hi) - std::max(a_lo, b_lo) <= 0.0) {
            return false;
        }
    }
    return true;
}

bool charts_share_vertex(const Mesh& mesh, int fa, int fb) {
    const Eigen::Vector3i& a = mesh.faces[static_cast<std::size_t>(fa)];
    const Eigen::Vector3i& b = mesh.faces[static_cast<std::size_t>(fb)];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (a[i] == b[j]) {
                return true;
            }
        }
    }
    return false;
}

/// Number of charts holding exactly `size` faces.
std::size_t chart_of_size(const std::vector<Chart>& charts, std::size_t size) {
    std::size_t count = 0;
    for (const Chart& chart : charts) {
        if (chart.faces.size() == size) {
            ++count;
        }
    }
    return count;
}

/// Counts overlapping face pairs across all charts (shared-vertex pairs skip).
std::size_t count_uv_overlaps(const Mesh& mesh, const std::vector<Chart>& charts) {
    std::size_t overlaps = 0;
    for (const Chart& chart : charts) {
        for (std::size_t i = 0; i + 1 < chart.faces.size(); ++i) {
            for (std::size_t j = i + 1; j < chart.faces.size(); ++j) {
                if (charts_share_vertex(mesh, chart.faces[i], chart.faces[j])) {
                    continue;
                }
                if (uv_triangles_overlap(&chart.corner_uv[i * 3], &chart.corner_uv[j * 3])) {
                    ++overlaps;
                }
            }
        }
    }
    return overlaps;
}

// ---------------------------------------------------------------------------
// 1. Partition invariant
// ---------------------------------------------------------------------------

TEST(UvCharts, PartitionInvariant) {
    ChartOptions options;
    for (const Mesh& mesh :
         {make_box(), make_corner(4), make_flat_grid(6), make_icosphere(2)}) {
        const std::vector<Chart> charts = build_charts(mesh, options);
        ASSERT_FALSE(charts.empty());
        expect_partition(mesh, charts);
    }
}

// ---------------------------------------------------------------------------
// 2. Chart counts on hand-built inputs
// ---------------------------------------------------------------------------

TEST(UvCharts, BoxHasSixCharts) {
    ChartOptions options;
    options.max_normal_angle_deg = 40.0;
    const Mesh box = make_box();
    const std::vector<Chart> charts = build_charts(box, options);
    ASSERT_EQ(charts.size(), 6U);
    // Every side contributes its two coplanar triangles to one chart.
    for (const Chart& chart : charts) {
        EXPECT_EQ(chart.faces.size(), 2U);
    }
    expect_partition(box, charts);
}

TEST(UvCharts, CornerHasTwoCharts) {
    ChartOptions options;
    options.max_normal_angle_deg = 40.0;
    const Mesh corner = make_corner(4);
    const std::vector<Chart> charts = build_charts(corner, options);
    ASSERT_EQ(charts.size(), 2U);
    // The corner is two 32-face grids meeting at 90 deg, so both charts hold 32.
    std::size_t total = 0;
    for (const Chart& chart : charts) {
        EXPECT_EQ(chart.faces.size(), 32U);
        total += chart.faces.size();
    }
    EXPECT_EQ(total, corner.faces.size());
    expect_partition(corner, charts);
}

TEST(UvCharts, FlatGridIsOneChart) {
    ChartOptions options;
    const Mesh grid = make_flat_grid(8);
    const std::vector<Chart> charts = build_charts(grid, options);
    ASSERT_EQ(charts.size(), 1U);
    EXPECT_EQ(charts[0].faces.size(), grid.faces.size());
    expect_partition(grid, charts);
}

// ---------------------------------------------------------------------------
// 3. Isometry on planar charts
// ---------------------------------------------------------------------------

TEST(UvCharts, PlanarChartsAreIsometric) {
    ChartOptions options;
    const Mesh box = make_box();
    const std::vector<Chart> box_charts = build_charts(box, options);
    expect_isometry(box, box_charts, 1e-9);

    const Mesh grid = make_flat_grid(6);
    const std::vector<Chart> grid_charts = build_charts(grid, options);
    expect_isometry(grid, grid_charts, 1e-9);
}

// ---------------------------------------------------------------------------
// 4. Orientation
// ---------------------------------------------------------------------------

TEST(UvCharts, ProjectedTrianglesKeepOrientation) {
    ChartOptions options;
    for (const Mesh& mesh :
         {make_box(), make_corner(4), make_flat_grid(5), make_icosphere(3)}) {
        const std::vector<Chart> charts = build_charts(mesh, options);
        expect_orientation_positive(charts);
    }
}

// ---------------------------------------------------------------------------
// 5. Sphere: cone adherence, overlap-freedom, area distortion
// ---------------------------------------------------------------------------

TEST(UvCharts, SphereChartsAreClean) {
    ChartOptions options;
    options.max_normal_angle_deg = 40.0;
    const Mesh sphere = make_icosphere(4);
    ASSERT_EQ(sphere.faces.size(), 5120U);
    ChartStats stats;
    const std::vector<Chart> charts = build_charts(sphere, options, &stats);

    std::cout << "[UvCharts] icosphere(4) charts=" << charts.size()
              << " overlap_splits=" << stats.overlap_splits
              << " max_overlaps=" << count_uv_overlaps(sphere, charts) << std::endl;

    ASSERT_GE(charts.size(), 6U);
    ASSERT_LE(charts.size(), 400U);

    const double cos_cone = std::cos(40.0 * 3.14159265358979323846 / 180.0);
    const auto face_normal = [&sphere](int f) {
        const Eigen::Vector3i& tri = sphere.faces[static_cast<std::size_t>(f)];
        const Eigen::Vector3d& p0 = sphere.vertices[static_cast<std::size_t>(tri[0])];
        const Eigen::Vector3d& p1 = sphere.vertices[static_cast<std::size_t>(tri[1])];
        const Eigen::Vector3d& p2 = sphere.vertices[static_cast<std::size_t>(tri[2])];
        return (p1 - p0).cross(p2 - p0).normalized();
    };
    // Per chart: every face normal within the cone of the chart's *seed* normal
    // (the fixed reference growth is anchored to), and the projection preserves
    // at least cos(cone) of the area.
    for (const Chart& chart : charts) {
        ASSERT_FALSE(chart.faces.empty());
        double area3d = 0.0;
        double area2d = 0.0;
        const Eigen::Vector3d seed_normal = face_normal(chart.faces[0]);
        for (std::size_t i = 0; i < chart.faces.size(); ++i) {
            const int f = chart.faces[i];
            EXPECT_GE(face_normal(f).dot(seed_normal), cos_cone - 1e-9)
                << "face " << f << " outside chart cone";
            area3d += face_area_3d(sphere, f);
            area2d += std::abs(signed_area(chart.corner_uv[i * 3], chart.corner_uv[i * 3 + 1],
                                           chart.corner_uv[i * 3 + 2]));
        }
        ASSERT_GT(area3d, 0.0);
        const double ratio = area2d / area3d;
        EXPECT_GE(ratio, cos_cone - 1e-9) << "area distortion below cos(cone)";
    }

    // No chart folds over itself.
    EXPECT_EQ(count_uv_overlaps(sphere, charts), 0U);
    expect_partition(sphere, charts);
}

// ---------------------------------------------------------------------------
// 6. Face cap, validation, determinism, timing
// ---------------------------------------------------------------------------

/// A self-overlapping coplanar ribbon cannot be separated by the cone test (all
/// normals are identical), so the flatten step must detect the overlap, re-grow
/// with a tighter cone, and bottom out at one face per chart.
TEST(UvCharts, OverlappingChartIsSplit) {
    ChartOptions options;
    const Mesh ribbon = make_spiral_ribbon(400);
    ChartStats stats;
    const std::vector<Chart> charts = build_charts(ribbon, options, &stats);
    std::cout << "[UvCharts] spiral ribbon charts=" << charts.size()
              << " overlap_splits=" << stats.overlap_splits << std::endl;
    EXPECT_GT(stats.overlap_splits, 0U);
    EXPECT_EQ(count_uv_overlaps(ribbon, charts), 0U);
    expect_partition(ribbon, charts);
}

TEST(UvCharts, MaxChartFacesCapsCharts) {
    ChartOptions options;
    options.max_chart_faces = 10;
    const Mesh grid = make_flat_grid(10);
    const std::vector<Chart> charts = build_charts(grid, options);
    ASSERT_GT(charts.size(), 1U);
    for (const Chart& chart : charts) {
        EXPECT_LE(chart.faces.size(), 10U);
    }
    expect_partition(grid, charts);
}

TEST(UvCharts, RejectsBadOptions) {
    const Mesh box = make_box();
    ChartOptions options;

    options.max_normal_angle_deg = 0.0;
    EXPECT_THROW(build_charts(box, options), ValueError);
    options.max_normal_angle_deg = 90.0;
    EXPECT_THROW(build_charts(box, options), ValueError);
    options.max_normal_angle_deg = -5.0;
    EXPECT_THROW(build_charts(box, options), ValueError);

    options.max_normal_angle_deg = 40.0;
    options.max_chart_faces = 0;
    EXPECT_THROW(build_charts(box, options), ValueError);
}

TEST(UvCharts, RejectsBadFaceIndex) {
    Mesh mesh = make_box();
    mesh.faces[2][1] = 999;  // out of range
    ChartOptions options;
    EXPECT_THROW(build_charts(mesh, options), ValueError);

    Mesh negative = make_box();
    negative.faces[0][0] = -1;
    EXPECT_THROW(build_charts(negative, options), ValueError);
}

TEST(UvCharts, IsDeterministic) {
    ChartOptions options;
    const Mesh sphere = make_icosphere(3);
    const std::vector<Chart> first = build_charts(sphere, options);
    const std::vector<Chart> second = build_charts(sphere, options);
    ASSERT_EQ(first.size(), second.size());
    for (std::size_t c = 0; c < first.size(); ++c) {
        EXPECT_EQ(first[c].faces, second[c].faces);
        ASSERT_EQ(first[c].corner_uv.size(), second[c].corner_uv.size());
        for (std::size_t k = 0; k < first[c].corner_uv.size(); ++k) {
            EXPECT_EQ(first[c].corner_uv[k], second[c].corner_uv[k]);
        }
    }
}

TEST(UvCharts, HandlesDegenerateFaces) {
    // A triangle with two coincident corners is a zero-area face: it cannot join
    // a chart and must come out as its own single-face chart.
    Mesh mesh = make_flat_grid(2);
    const int dup = static_cast<int>(mesh.vertices.size());
    mesh.vertices.push_back(mesh.vertices[0]);
    mesh.faces.push_back({0, 1, dup});  // collinear/zero-area
    ChartOptions options;
    const std::vector<Chart> charts = build_charts(mesh, options);
    expect_partition(mesh, charts);
    EXPECT_EQ(chart_of_size(charts, 1), 1U) << "zero-area face should be a lone chart";
    // The lone chart still carries three (finite) coordinates.
    for (const Chart& chart : charts) {
        if (chart.faces.size() == 1U) {
            EXPECT_EQ(chart.corner_uv.size(), 3U);
            for (const Eigen::Vector2d& uv : chart.corner_uv) {
                EXPECT_TRUE(uv.allFinite());
            }
        }
    }
}

TEST(UvCharts, LargeSphereWithinTimeBudget) {
    ChartOptions options;
    const Mesh sphere = make_icosphere(6);  // 81920 faces
    ASSERT_EQ(sphere.faces.size(), 81920U);
    const auto start = std::chrono::steady_clock::now();
    const std::vector<Chart> charts = build_charts(sphere, options);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
    std::cout << "[UvCharts] icosphere(6) charts=" << charts.size()
              << " time=" << elapsed.count() << "s" << std::endl;
    EXPECT_LT(elapsed.count(), 2.0);
    expect_partition(sphere, charts);
}

}  // namespace
}  // namespace alphashape
