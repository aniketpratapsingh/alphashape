// test_atlas_pack.cpp -- tests for the texture-atlas packer (milestone 8).
//
// What these tests pin:
//   * every uv lands inside [0,1] and every uv_faces entry is a valid index,
//   * charts never overlap in texture space, and different charts keep at least
//     padding_px - 1 texels between their nearest texels,
//   * one shared scale: every face's uv-to-world edge ratio equals
//     texels_per_unit, so no chart is stretched relative to another,
//   * orientation: the signed area of a uv triangle keeps the sign of the
//     chart's own corner_uv signed area (the packer only rotates, never flips),
//   * fill quality on many charts, and that a thin rotated chart comes back
//     axis-aligned,
//   * the validation errors, determinism, and throughput at scale.
//
// Charts are built by hand here on purpose: build_charts() is a separate pass
// with its own tests, and the packer must be checkable without it.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include "base/utils/portable_rng.hpp"

#include "base/utils/error.hpp"
#include "model/mesh/atlas_pack.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/uv_charts.hpp"

namespace alphashape {
namespace {

using Vec2 = Eigen::Vector2d;

/// A mesh plus a hand-built chart partition of it.
struct ChartedMesh {
    Mesh mesh;
    std::vector<Chart> charts;
};

/// Half of the (v0, v1, v2) corner coordinates of a unit square split along its
/// diagonal, in the same winding as the two faces added below.
std::array<Vec2, 6> unit_square_corners() {
    return {Vec2(0.0, 0.0), Vec2(1.0, 0.0), Vec2(1.0, 1.0),
            Vec2(0.0, 0.0), Vec2(1.0, 1.0), Vec2(0.0, 1.0)};
}

/// Append one rectangle as a chart: two faces over four fresh vertices, lying in
/// the z = 0 plane, rotated by `angle` radians about the origin, and with the
/// rectangle's own corner positions as its world-unit chart coordinates. A
/// rectangle is the cleanest chart the packer can be handed — its minimum-area
/// rectangle is itself, so any departure from that shape is the packer's doing.
void add_rect_chart(ChartedMesh& out, double w, double h, double angle) {
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const auto rot = [c, s](double x, double y) { return Vec2(c * x - s * y, s * x + c * y); };
    const Vec2 p0 = rot(0.0, 0.0);
    const Vec2 p1 = rot(w, 0.0);
    const Vec2 p2 = rot(w, h);
    const Vec2 p3 = rot(0.0, h);

    const int base = static_cast<int>(out.mesh.vertices.size());
    const int first_face = static_cast<int>(out.mesh.faces.size());
    out.mesh.vertices.emplace_back(p0.x(), p0.y(), 0.0);
    out.mesh.vertices.emplace_back(p1.x(), p1.y(), 0.0);
    out.mesh.vertices.emplace_back(p2.x(), p2.y(), 0.0);
    out.mesh.vertices.emplace_back(p3.x(), p3.y(), 0.0);
    out.mesh.faces.emplace_back(base + 0, base + 1, base + 2);
    out.mesh.faces.emplace_back(base + 0, base + 2, base + 3);

    Chart chart;
    chart.faces = {first_face, first_face + 1};
    chart.corner_uv = {p0, p1, p2, p0, p2, p3};
    out.charts.push_back(std::move(chart));
}

/// `count` unit squares in a row, each square its own chart. The squares are
/// `cell` units apart in x so the mesh has no accidental shared vertices.
ChartedMesh make_unit_squares(int count, double cell = 2.0) {
    ChartedMesh out;
    const std::array<Vec2, 6> corners = unit_square_corners();
    for (int i = 0; i < count; ++i) {
        const double x = cell * static_cast<double>(i);
        const int base = static_cast<int>(out.mesh.vertices.size());
        out.mesh.vertices.emplace_back(x + 0.0, 0.0, 0.0);
        out.mesh.vertices.emplace_back(x + 1.0, 0.0, 0.0);
        out.mesh.vertices.emplace_back(x + 1.0, 1.0, 0.0);
        out.mesh.vertices.emplace_back(x + 0.0, 1.0, 0.0);
        out.mesh.faces.emplace_back(base + 0, base + 1, base + 2);
        out.mesh.faces.emplace_back(base + 0, base + 2, base + 3);

        Chart chart;
        chart.faces = {2 * i, 2 * i + 1};
        chart.corner_uv.assign(corners.begin(), corners.end());
        out.charts.push_back(std::move(chart));
    }
    return out;
}

/// `count` rectangles of random size in [min_side, max_side], each its own
/// chart, all axis-aligned. Seeded, so the pack is reproducible.
ChartedMesh make_random_rects(int count, double min_side, double max_side, unsigned seed) {
    ChartedMesh out;
    std::mt19937 rng(seed);
    // Portable draw: std::uniform_real_distribution is implementation-defined
    // across libc++ and libstdc++ (TESTING.md, "Numeric consistency").
    const auto side = [min_side, max_side](std::mt19937& g) {
        return alphashape::rng::uniform_real(g, min_side, max_side);
    };
    for (int i = 0; i < count; ++i) {
        const double w = side(rng);
        const double h = side(rng);
        add_rect_chart(out, w, h, 0.0);
        // Space the rectangles out so no two charts share a mesh vertex.
        const int last = static_cast<int>(out.mesh.vertices.size()) - 4;
        for (int v = 0; v < 4; ++v) {
            out.mesh.vertices[static_cast<std::size_t>(last + v)].x() +=
                3.0 * static_cast<double>(i);
        }
    }
    return out;
}

/// The texture coordinate of `face`'s `corner` (0..2), read through the pool.
Vec2 face_uv(const Mesh& mesh, int face, int corner) {
    const Eigen::Vector3i& tri = mesh.uv_faces[static_cast<std::size_t>(face)];
    const Eigen::Vector2f& uv = mesh.uvs[static_cast<std::size_t>(tri[corner])];
    return Vec2(static_cast<double>(uv.x()), static_cast<double>(uv.y()));
}

/// Rasterise every textured triangle into a `size x size` texel grid, storing
/// the chart id that owns each covered texel (-1 where empty, kConflict where
/// two different charts claimed the same texel). A texel is covered when its
/// CENTRE lies inside the uv triangle, matching the engine's texel-centre
/// convention (texel (i, j) sits at ((i + 0.5) / size, (j + 0.5) / size) with v
/// growing down). `*conflict` is set when any texel is claimed twice.
constexpr int kConflict = -2;

std::vector<int> rasterise_charts(const Mesh& mesh, const std::vector<Chart>& charts,
                                  int size, bool* conflict) {
    std::vector<int> grid(static_cast<std::size_t>(size) * static_cast<std::size_t>(size), -1);
    *conflict = false;
    std::vector<int> face_chart(mesh.faces.size(), -1);
    for (std::size_t ci = 0; ci < charts.size(); ++ci) {
        for (const int f : charts[ci].faces) {
            face_chart[static_cast<std::size_t>(f)] = static_cast<int>(ci);
        }
    }
    const double inv = 1.0 / static_cast<double>(size);
    for (std::size_t f = 0; f < mesh.faces.size(); ++f) {
        const Vec2 a = face_uv(mesh, static_cast<int>(f), 0);
        const Vec2 b = face_uv(mesh, static_cast<int>(f), 1);
        const Vec2 c = face_uv(mesh, static_cast<int>(f), 2);
        const double min_u = std::min(a.x(), std::min(b.x(), c.x()));
        const double max_u = std::max(a.x(), std::max(b.x(), c.x()));
        const double min_v = std::min(a.y(), std::min(b.y(), c.y()));
        const double max_v = std::max(a.y(), std::max(b.y(), c.y()));
        const int i0 = std::max(0, static_cast<int>(std::floor(min_u * size - 0.5)));
        const int i1 = std::min(size - 1, static_cast<int>(std::ceil(max_u * size - 0.5)));
        const int j0 = std::max(0, static_cast<int>(std::floor(min_v * size - 0.5)));
        const int j1 = std::min(size - 1, static_cast<int>(std::ceil(max_v * size - 0.5)));
        const double twice_area = (b - a).x() * (c - a).y() - (b - a).y() * (c - a).x();
        if (std::abs(twice_area) <= 0.0) continue;
        const int owner = face_chart[f];
        for (int j = j0; j <= j1; ++j) {
            for (int i = i0; i <= i1; ++i) {
                const Vec2 p((static_cast<double>(i) + 0.5) * inv,
                             (static_cast<double>(j) + 0.5) * inv);
                const double w0 = ((b.x() - p.x()) * (c.y() - p.y()) -
                                   (b.y() - p.y()) * (c.x() - p.x())) / twice_area;
                const double w1 = ((c.x() - p.x()) * (a.y() - p.y()) -
                                   (c.y() - p.y()) * (a.x() - p.x())) / twice_area;
                const double w2 = 1.0 - w0 - w1;
                if (w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9) continue;
                int& slot = grid[static_cast<std::size_t>(j) * static_cast<std::size_t>(size) +
                                 static_cast<std::size_t>(i)];
                if (slot < 0) {
                    slot = owner;
                } else if (slot != owner && slot != kConflict) {
                    // Two charts over one texel: the atlas would bleed.
                    slot = kConflict;
                    *conflict = true;
                }
            }
        }
    }
    return grid;
}

/// Signed area of a uv triangle, positive when its corners wind counter-
/// clockwise on the page (v down).
double signed_uv_area(const Vec2& a, const Vec2& b, const Vec2& c) {
    return 0.5 * ((b - a).x() * (c - a).y() - (b - a).y() * (c - a).x());
}

/// Chart-relative signed area of the triangle at `face_offset` inside a chart,
/// in the chart's own corner_uv coordinates.
double signed_chart_area(const Chart& chart, std::size_t face_offset) {
    const Vec2& a = chart.corner_uv[3 * face_offset];
    const Vec2& b = chart.corner_uv[3 * face_offset + 1];
    const Vec2& c = chart.corner_uv[3 * face_offset + 2];
    return signed_uv_area(a, b, c);
}

TEST(AtlasPack, UvsStayInsideThePageAndIndicesAreValid) {
    ChartedMesh scene = make_unit_squares(7);
    PackOptions options;
    options.texture_size = 256;
    options.padding_px = 4;
    const PackResult result = pack_atlas(scene.mesh, scene.charts, options);

    ASSERT_EQ(scene.mesh.uv_faces.size(), scene.mesh.faces.size());
    ASSERT_FALSE(scene.mesh.uvs.empty());
    for (const Eigen::Vector3i& tri : scene.mesh.uv_faces) {
        for (int corner = 0; corner < 3; ++corner) {
            ASSERT_GE(tri[corner], 0);
            ASSERT_LT(tri[corner], static_cast<int>(scene.mesh.uvs.size()));
        }
    }
    for (const Eigen::Vector2f& uv : scene.mesh.uvs) {
        EXPECT_GE(uv.x(), 0.0f);
        EXPECT_LE(uv.x(), 1.0f);
        EXPECT_GE(uv.y(), 0.0f);
        EXPECT_LE(uv.y(), 1.0f);
    }
    EXPECT_GT(result.texels_per_unit, 0.0);
    EXPECT_GT(result.fill_ratio, 0.0);
    EXPECT_LE(result.fill_ratio, 1.0);
}

TEST(AtlasPack, ChartsDoNotOverlapAndKeepTheirPadding) {
    // Right-sized charts on a small page: the interesting case is charts
    // touching, not charts sparse in a sea of empty texels.
    ChartedMesh scene = make_random_rects(30, 0.5, 3.0, 1234u);
    PackOptions options;
    options.texture_size = 256;
    options.padding_px = 4;

    const PackResult result = pack_atlas(scene.mesh, scene.charts, options);
    ASSERT_GT(result.texels_per_unit, 0.0);

    bool conflict = false;
    const std::vector<int> grid =
        rasterise_charts(scene.mesh, scene.charts, options.texture_size, &conflict);
    EXPECT_FALSE(conflict) << "two charts claimed the same texel";

    // The padding contract: no texel of one chart may lie within
    // padding_px - 1 texels of a texel of a different chart. Dilation is the
    // clean way to state it — grow chart c's texels into a square of half-width
    // `radius`, and require every texel that growth touches to be either empty
    // or chart c's own. (Comparing two charts' halos against each other would
    // false-alarm: the empty midway texel is legitimately close to both.)
    const int radius = options.padding_px - 1;
    const int size = options.texture_size;
    const std::size_t n = static_cast<std::size_t>(size);
    int violations = 0;
    int worst_gap = size;  // smallest Chebyshev texel gap between two charts
    // Measure the true gap over a slightly wider window than the contract
    // needs, so the printed number is the real separation rather than the
    // contract's bound.
    const int probe = options.padding_px + 3;
    for (std::size_t at = 0; at < n * n; ++at) {
        const int chart = grid[at];
        if (chart < 0) continue;
        const int ci = static_cast<int>(at % n);
        const int cj = static_cast<int>(at / n);
        for (int dj = -probe; dj <= probe; ++dj) {
            const int jj = cj + dj;
            if (jj < 0 || jj >= size) continue;
            for (int di = -probe; di <= probe; ++di) {
                const int ii = ci + di;
                if (ii < 0 || ii >= size) continue;
                const int other = grid[static_cast<std::size_t>(jj) * n +
                                       static_cast<std::size_t>(ii)];
                if (other < 0 || other == chart) continue;
                const int gap = std::max(std::abs(di), std::abs(dj));
                worst_gap = std::min(worst_gap, gap);
                if (gap <= radius) ++violations;
            }
        }
    }
    EXPECT_EQ(violations, 0) << "charts came closer than padding_px texels";
    std::printf("[AtlasPack] min inter-chart texel gap = %d (padding_px = %d, fill = %.3f)\n",
                worst_gap, options.padding_px, result.fill_ratio);
}

TEST(AtlasPack, OneUniformScaleForEveryChart) {
    // Charts of very different sizes and rotations: if the packer rescaled any
    // chart on its own, a face's uv edge length no longer matches its world
    // edge length by the shared factor.
    ChartedMesh scene;
    add_rect_chart(scene, 4.0, 1.0, 0.0);
    add_rect_chart(scene, 0.5, 0.25, 0.6);
    add_rect_chart(scene, 2.5, 2.5, -1.1);
    add_rect_chart(scene, 1.0, 3.0, 0.35);
    PackOptions options;
    options.texture_size = 512;
    options.padding_px = 6;

    const PackResult result = pack_atlas(scene.mesh, scene.charts, options);
    ASSERT_GT(result.texels_per_unit, 0.0);

    // World edge length of a face comes from the mesh; the uv edge length times
    // the page size is in texels, so the ratio is texels per world unit.
    for (std::size_t f = 0; f < scene.mesh.faces.size(); ++f) {
        const Eigen::Vector3i& tri = scene.mesh.faces[f];
        for (int corner = 0; corner < 3; ++corner) {
            const int a = corner;
            const int b = (corner + 1) % 3;
            const Eigen::Vector3d& pa = scene.mesh.vertices[static_cast<std::size_t>(tri[a])];
            const Eigen::Vector3d& pb = scene.mesh.vertices[static_cast<std::size_t>(tri[b])];
            const double world = (pb - pa).norm();
            const Vec2 ua = face_uv(scene.mesh, static_cast<int>(f), a);
            const Vec2 ub = face_uv(scene.mesh, static_cast<int>(f), b);
            const double texels = (ub - ua).norm() * options.texture_size;
            ASSERT_GT(world, 0.0);
            EXPECT_NEAR(texels / world, result.texels_per_unit,
                        result.texels_per_unit * 1e-6)
                << "face " << f << " corner " << corner << " is rescaled on its own";
        }
    }
}

TEST(AtlasPack, OrientationIsPreserved) {
    // v grows down and the packer only rotates, so a chart's signed corner area
    // and the signed area of its uv triangles must keep the same sign. The sign
    // itself is whatever the input had: these hand-built squares give +0.5 for
    // their first face under the (b-a)x(c-a) formula, and that must survive
    // packing unchanged.
    ChartedMesh scene = make_random_rects(12, 0.4, 2.0, 99u);
    PackOptions options;
    options.texture_size = 512;
    options.padding_px = 4;
    pack_atlas(scene.mesh, scene.charts, options);

    // Pin the convention explicitly for one known square as well, so a change
    // to the frame handedness in the packer cannot silently pass this test.
    EXPECT_GT(signed_chart_area(scene.charts[0], 0), 0.0);

    int checked = 0;
    for (std::size_t ci = 0; ci < scene.charts.size(); ++ci) {
        const Chart& chart = scene.charts[ci];
        for (std::size_t k = 0; k < chart.faces.size(); ++k) {
            const int f = chart.faces[k];
            const double chart_area = signed_chart_area(chart, k);
            const Vec2 a = face_uv(scene.mesh, f, 0);
            const Vec2 b = face_uv(scene.mesh, f, 1);
            const Vec2 c = face_uv(scene.mesh, f, 2);
            const double uv_area = signed_uv_area(a, b, c);
            // Compare signs, not values: the packer rotates and scales the
            // chart, so only the winding is invariant.
            EXPECT_EQ(chart_area < 0.0, uv_area < 0.0)
                << "chart " << ci << " face offset " << k << " flipped winding";
            ++checked;
        }
    }
    EXPECT_GT(checked, 0);
}

TEST(AtlasPack, ManyChartsPackDensely) {
    ChartedMesh scene = make_random_rects(200, 0.4, 2.5, 20261018u);
    PackOptions options;
    options.texture_size = 1024;
    options.padding_px = 2;

    const PackResult result = pack_atlas(scene.mesh, scene.charts, options);
    std::printf("[AtlasPack] 200 charts: fill = %.4f, texels_per_unit = %.3f\n",
                result.fill_ratio, result.texels_per_unit);
    EXPECT_GE(result.fill_ratio, 0.5);

    bool conflict = false;
    const std::vector<int> grid =
        rasterise_charts(scene.mesh, scene.charts, options.texture_size, &conflict);
    EXPECT_FALSE(conflict);
}

TEST(AtlasPack, ThinRotatedChartComesBackAxisAligned) {
    // One long thin chart, rotated 30 degrees. Its minimum-area rectangle is
    // the rectangle itself, so after packing its uv bounding box must have the
    // true aspect ratio and be axis-aligned.
    const double w = 12.0;
    const double h = 0.75;
    const double angle = 30.0 * 3.14159265358979323846 / 180.0;
    ChartedMesh scene;
    add_rect_chart(scene, w, h, angle);
    PackOptions options;
    options.texture_size = 512;
    options.padding_px = 4;

    const PackResult result = pack_atlas(scene.mesh, scene.charts, options);
    ASSERT_GT(result.texels_per_unit, 0.0);

    double min_u = 1.0;
    double max_u = 0.0;
    double min_v = 1.0;
    double max_v = 0.0;
    for (std::size_t f = 0; f < scene.mesh.faces.size(); ++f) {
        for (int corner = 0; corner < 3; ++corner) {
            const Vec2 uv = face_uv(scene.mesh, static_cast<int>(f), corner);
            min_u = std::min(min_u, uv.x());
            max_u = std::max(max_u, uv.x());
            min_v = std::min(min_v, uv.y());
            max_v = std::max(max_v, uv.y());
        }
    }
    const double box_w = (max_u - min_u) * options.texture_size;
    const double box_h = (max_v - min_v) * options.texture_size;
    std::printf("[AtlasPack] thin chart bbox = %.1f x %.1f texels\n", box_w, box_h);
    // The rectangle is landscape, so it must be the wide way round, its aspect
    // must match the true one, and the box must actually hold that rectangle.
    EXPECT_GT(box_w, box_h);
    EXPECT_NEAR(box_w / box_h, w / h, 0.05);
    EXPECT_GT(box_w, w * result.texels_per_unit * 0.99);
    EXPECT_GT(box_h, h * result.texels_per_unit * 0.99);
}

TEST(AtlasPack, RejectsIncompleteAndInconsistentPartitions) {
    PackOptions options;
    options.texture_size = 128;
    options.padding_px = 2;

    // A face left out of every chart.
    {
        ChartedMesh scene = make_unit_squares(3);
        scene.charts[1].faces.pop_back();
        scene.charts[1].corner_uv.resize(3);
        EXPECT_THROW(pack_atlas(scene.mesh, scene.charts, options), Error);
    }
    // A face claimed by two charts.
    {
        ChartedMesh scene = make_unit_squares(3);
        scene.charts[2].faces.push_back(scene.charts[0].faces[0]);
        scene.charts[2].corner_uv.push_back(Vec2(0.0, 0.0));
        scene.charts[2].corner_uv.push_back(Vec2(1.0, 0.0));
        scene.charts[2].corner_uv.push_back(Vec2(1.0, 1.0));
        EXPECT_THROW(pack_atlas(scene.mesh, scene.charts, options), Error);
    }
    // Corner count out of step with the face count.
    {
        ChartedMesh scene = make_unit_squares(2);
        scene.charts[0].corner_uv.pop_back();
        EXPECT_THROW(pack_atlas(scene.mesh, scene.charts, options), Error);
    }
    // A face id past the end of the mesh.
    {
        ChartedMesh scene = make_unit_squares(2);
        scene.charts[0].faces[0] = 999;
        EXPECT_THROW(pack_atlas(scene.mesh, scene.charts, options), Error);
    }
    // Sanity: the same scene with a consistent partition does not throw.
    {
        ChartedMesh scene = make_unit_squares(3);
        EXPECT_NO_THROW(pack_atlas(scene.mesh, scene.charts, options));
    }
}

TEST(AtlasPack, RejectsBadOptionsAndImpossibleCharts) {
    // A page too small to texture at all.
    {
        ChartedMesh scene = make_unit_squares(2);
        PackOptions options;
        options.texture_size = 8;
        options.padding_px = 0;
        EXPECT_THROW(pack_atlas(scene.mesh, scene.charts, options), Error);
    }
    // Negative padding.
    {
        ChartedMesh scene = make_unit_squares(2);
        PackOptions options;
        options.texture_size = 128;
        options.padding_px = -1;
        EXPECT_THROW(pack_atlas(scene.mesh, scene.charts, options), Error);
    }
    // Padding that eats the whole page leaves no room for any chart: the
    // smallest a padded rectangle can ever be is (1 + padding) texels, because
    // rounding up any positive content already costs a texel.
    {
        ChartedMesh scene = make_unit_squares(2);
        PackOptions options;
        options.texture_size = 64;
        options.padding_px = 64;
        EXPECT_THROW(pack_atlas(scene.mesh, scene.charts, options), Error);
    }
    {
        ChartedMesh scene = make_unit_squares(2);
        PackOptions options;
        options.texture_size = 128;
        options.padding_px = 128;
        EXPECT_THROW(pack_atlas(scene.mesh, scene.charts, options), Error);
    }
    // Two oversized charts are *not* impossible: the shared scale simply drops
    // until both fit, which is what a shared scale exists to do. The ceiling is
    // the padding: each chart's rectangle is ceil(1000 s) + 4 texels, and two
    // of them must fit across the 128-texel page, so ceil(1000 s) <= 60 and the
    // best achievable fill is 2 * 60^2 / 128^2 = 0.439.
    {
        ChartedMesh scene;
        add_rect_chart(scene, 1000.0, 1000.0, 0.0);
        add_rect_chart(scene, 1000.0, 1000.0, 0.0);
        PackOptions options;
        options.texture_size = 128;
        options.padding_px = 4;
        const PackResult result = pack_atlas(scene.mesh, scene.charts, options);
        const double best_fill = 2.0 * 60.0 * 60.0 / (128.0 * 128.0);
        EXPECT_NEAR(result.fill_ratio, best_fill, 0.01);
        EXPECT_NEAR(result.texels_per_unit, 0.06, 0.001);
    }
    // A single oversized chart is legitimate too: shrinking it to fill the page
    // is exactly what the shared scale is for.
    {
        ChartedMesh scene;
        add_rect_chart(scene, 1000.0, 1000.0, 0.0);
        PackOptions options;
        options.texture_size = 128;
        options.padding_px = 0;
        const PackResult result = pack_atlas(scene.mesh, scene.charts, options);
        EXPECT_GT(result.texels_per_unit, 0.0);
        EXPECT_GT(result.fill_ratio, 0.9);
    }
}

TEST(AtlasPack, EmptyMeshIsANoOp) {
    Mesh mesh;
    PackOptions options;
    const PackResult result = pack_atlas(mesh, {}, options);
    EXPECT_TRUE(mesh.uvs.empty());
    EXPECT_TRUE(mesh.uv_faces.empty());
    EXPECT_EQ(result.texels_per_unit, 0.0);
    EXPECT_EQ(result.fill_ratio, 0.0);
}

TEST(AtlasPack, IsDeterministic) {
    ChartedMesh first = make_random_rects(120, 0.3, 3.0, 7u);
    ChartedMesh second = make_random_rects(120, 0.3, 3.0, 7u);
    PackOptions options;
    options.texture_size = 512;
    options.padding_px = 3;

    const PackResult ra = pack_atlas(first.mesh, first.charts, options);
    const PackResult rb = pack_atlas(second.mesh, second.charts, options);
    EXPECT_DOUBLE_EQ(ra.texels_per_unit, rb.texels_per_unit);
    EXPECT_DOUBLE_EQ(ra.fill_ratio, rb.fill_ratio);
    ASSERT_EQ(first.mesh.uvs.size(), second.mesh.uvs.size());
    ASSERT_EQ(first.mesh.uv_faces.size(), second.mesh.uv_faces.size());
    for (std::size_t i = 0; i < first.mesh.uvs.size(); ++i) {
        EXPECT_EQ(first.mesh.uvs[i], second.mesh.uvs[i]) << "uv " << i << " differs";
    }
    for (std::size_t i = 0; i < first.mesh.uv_faces.size(); ++i) {
        EXPECT_EQ(first.mesh.uv_faces[i], second.mesh.uv_faces[i]) << "uv_faces " << i;
    }
}

TEST(AtlasPack, PacksFiveThousandChartsQuickly) {
    ChartedMesh scene = make_random_rects(5000, 0.2, 1.5, 424242u);
    PackOptions options;
    options.texture_size = 4096;
    options.padding_px = 2;

    const auto start = std::chrono::steady_clock::now();
    const PackResult result = pack_atlas(scene.mesh, scene.charts, options);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
    std::printf("[AtlasPack] 5000 charts: %.3f s, fill = %.4f, texels_per_unit = %.3f\n",
                elapsed.count(), result.fill_ratio, result.texels_per_unit);
    EXPECT_LT(elapsed.count(), 2.0);
    EXPECT_GT(result.fill_ratio, 0.0);
}


}  // namespace
}  // namespace alphashape
