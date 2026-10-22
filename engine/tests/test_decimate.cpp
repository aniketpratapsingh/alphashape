// Quadric decimation: topology preservation on a sphere, planarity on a plane,
// error capping, attribute bookkeeping, argument validation and determinism.

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
#include "model/mesh/decimate.hpp"
#include "model/mesh/mesh.hpp"

namespace alphashape {
namespace {

using Index = std::size_t;

// ---------------------------------------------------------------------------
// Mesh generators
// ---------------------------------------------------------------------------

/// Icosphere: subdivide an icosahedron, projecting each new vertex onto the unit
/// sphere. Vertices are shared through a midpoint cache so the result is a
/// closed manifold with V - E + F == 2.
Mesh make_icosphere(int subdivisions) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<Eigen::Vector3d> verts = {
        {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
        {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
        {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1},
    };
    for (Eigen::Vector3d& v : verts) {
        v.normalize();
    }
    std::vector<Eigen::Vector3i> faces = {
        {0, 11, 5},  {0, 5, 1},   {0, 1, 7},   {0, 7, 10},  {0, 10, 11},
        {1, 5, 9},   {5, 11, 4},  {11, 10, 2}, {10, 7, 6},  {7, 1, 8},
        {3, 9, 4},   {3, 4, 2},   {3, 2, 6},   {3, 6, 8},   {3, 8, 9},
        {4, 9, 5},   {2, 4, 11},  {6, 2, 10},  {8, 6, 7},   {9, 8, 1},
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
            Eigen::Vector3d m = 0.5 * (verts[static_cast<Index>(a)] + verts[static_cast<Index>(b)]);
            m.normalize();
            const int index = static_cast<int>(verts.size());
            verts.push_back(m);
            midpoints.emplace(key, index);
            return index;
        };
        for (const Eigen::Vector3i& f : faces) {
            const int ab = midpoint(f[0], f[1]);
            const int bc = midpoint(f[1], f[2]);
            const int ca = midpoint(f[2], f[0]);
            next.push_back({f[0], ab, ca});
            next.push_back({f[1], bc, ab});
            next.push_back({f[2], ca, bc});
            next.push_back({ab, bc, ca});
        }
        faces.swap(next);
    }

    Mesh mesh;
    mesh.vertices = std::move(verts);
    mesh.faces = std::move(faces);
    return mesh;
}

/// Axis-aligned triangle grid in the z = 0 plane: `n` x `n` quads, each split
/// into two CCW-from-+z triangles.
Mesh make_plane(int n) {
    Mesh mesh;
    const int stride = n + 1;
    mesh.vertices.reserve(static_cast<Index>(stride) * static_cast<Index>(stride));
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            mesh.vertices.push_back(Eigen::Vector3d(static_cast<double>(i), static_cast<double>(j), 0.0));
        }
    }
    mesh.faces.reserve(static_cast<Index>(2) * static_cast<Index>(n) * static_cast<Index>(n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const int a = j * stride + i;
            const int b = a + 1;
            const int c = a + stride;
            const int d = c + 1;
            mesh.faces.push_back({a, b, c});
            mesh.faces.push_back({b, d, c});
        }
    }
    return mesh;
}

// ---------------------------------------------------------------------------
// Topology helpers
// ---------------------------------------------------------------------------

struct EdgeStats {
    std::size_t boundary = 0;
    std::size_t non_manifold = 0;
    std::size_t unique = 0;
};

EdgeStats edge_stats(const Mesh& mesh) {
    std::unordered_map<std::uint64_t, int> counts;
    for (const Eigen::Vector3i& f : mesh.faces) {
        for (int k = 0; k < 3; ++k) {
            const std::uint64_t a = static_cast<std::uint64_t>(f[k]);
            const std::uint64_t b = static_cast<std::uint64_t>(f[(k + 1) % 3]);
            const std::uint64_t lo = std::min(a, b);
            const std::uint64_t hi = std::max(a, b);
            ++counts[(lo << 32U) | hi];
        }
    }
    EdgeStats stats;
    stats.unique = counts.size();
    for (const auto& entry : counts) {
        if (entry.second == 1) {
            ++stats.boundary;
        } else if (entry.second > 2) {
            ++stats.non_manifold;
        }
    }
    return stats;
}

/// V - E + F. A closed genus-0 surface gives 2.
long euler_characteristic(const Mesh& mesh) {
    const EdgeStats stats = edge_stats(mesh);
    return static_cast<long>(mesh.vertices.size()) - static_cast<long>(stats.unique) +
           static_cast<long>(mesh.faces.size());
}

/// Largest deviation of any vertex from the unit sphere.
double max_radius_error(const Mesh& mesh) {
    double worst = 0.0;
    for (const Eigen::Vector3d& v : mesh.vertices) {
        worst = std::max(worst, std::abs(v.norm() - 1.0));
    }
    return worst;
}

Eigen::Vector3d triangle_normal(const Mesh& mesh, const Eigen::Vector3i& f) {
    const Eigen::Vector3d& p0 = mesh.vertices[static_cast<Index>(f[0])];
    const Eigen::Vector3d& p1 = mesh.vertices[static_cast<Index>(f[1])];
    const Eigen::Vector3d& p2 = mesh.vertices[static_cast<Index>(f[2])];
    return (p1 - p0).cross(p2 - p0);
}

// ---------------------------------------------------------------------------
// 1. Closed surface: topology, shape and orientation all survive decimation.
// ---------------------------------------------------------------------------

TEST(Decimate, IcosphereKeepsTopologyAndShape) {
    Mesh mesh = make_icosphere(6);
    ASSERT_EQ(mesh.faces.size(), 81920U);
    const double before_error = max_radius_error(mesh);

    DecimateOptions options;
    options.target_faces = 8000;
    options.preserve_boundary = true;
    const std::size_t result = decimate_quadric(mesh, options);

    EXPECT_EQ(result, mesh.faces.size());
    EXPECT_LE(mesh.faces.size(), 8000U);
    EXPECT_GE(mesh.faces.size(), 7600U);

    const EdgeStats stats = edge_stats(mesh);
    EXPECT_EQ(stats.boundary, 0U);
    EXPECT_EQ(stats.non_manifold, 0U);
    EXPECT_EQ(euler_characteristic(mesh), 2L);

    const double after_error = max_radius_error(mesh);
    EXPECT_LT(after_error, 0.01) << "input error " << before_error;
    EXPECT_LT(after_error, before_error + 0.01);

    // Every face must still be wound outward.
    for (const Eigen::Vector3i& f : mesh.faces) {
        const Eigen::Vector3d n = triangle_normal(mesh, f);
        ASSERT_GT(n.norm(), 0.0);
        const Eigen::Vector3d centroid =
            (mesh.vertices[static_cast<Index>(f[0])] +
             mesh.vertices[static_cast<Index>(f[1])] +
             mesh.vertices[static_cast<Index>(f[2])]) /
            3.0;
        EXPECT_GT(n.dot(centroid), 0.0);
    }
}

// ---------------------------------------------------------------------------
// 2. Open planar patch: stays exactly planar, keeps its outline.
// ---------------------------------------------------------------------------

TEST(Decimate, PlaneStaysPlanarAndKeepsCorner) {
    Mesh mesh = make_plane(100);
    ASSERT_EQ(mesh.faces.size(), 20000U);

    Mesh reference = mesh;
    Eigen::Vector3d lo = reference.vertices[0];
    Eigen::Vector3d hi = reference.vertices[0];
    for (const Eigen::Vector3d& v : reference.vertices) {
        lo = lo.cwiseMin(v);
        hi = hi.cwiseMax(v);
    }

    DecimateOptions options;
    options.target_faces = 500;
    options.preserve_boundary = true;
    const std::size_t result = decimate_quadric(mesh, options);

    EXPECT_EQ(result, mesh.faces.size());
    EXPECT_LE(mesh.faces.size(), 500U);

    for (const Eigen::Vector3d& v : mesh.vertices) {
        ASSERT_LT(std::abs(v.z()), 1e-9);
    }

    Eigen::Vector3d new_lo = mesh.vertices[0];
    Eigen::Vector3d new_hi = mesh.vertices[0];
    for (const Eigen::Vector3d& v : mesh.vertices) {
        new_lo = new_lo.cwiseMin(v);
        new_hi = new_hi.cwiseMax(v);
    }
    EXPECT_LT((new_lo - lo).cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_LT((new_hi - hi).cwiseAbs().maxCoeff(), 1e-9);

    const EdgeStats stats = edge_stats(mesh);
    EXPECT_EQ(stats.non_manifold, 0U);
    EXPECT_EQ(stats.non_manifold, 0U);
    // An open patch has exactly one boundary loop, and its corners are pinned by
    // two heavily weighted constraint planes each, so the loop survives.
    EXPECT_GT(stats.boundary, 0U);
    EXPECT_LE(stats.boundary, 400U);
    EXPECT_EQ(euler_characteristic(mesh), 1L);
}

// ---------------------------------------------------------------------------
// 3. Error capping, in both directions.
// ---------------------------------------------------------------------------

TEST(Decimate, MaxErrorStopsSphereEarly) {
    Mesh mesh = make_icosphere(5);
    ASSERT_EQ(mesh.faces.size(), 20480U);

    // A 1e-12 cap is below the error of every collapse a curved surface admits,
    // so the run refuses all of them and the mesh comes back untouched.
    DecimateOptions tiny;
    tiny.target_faces = 100;
    tiny.max_error = 1e-12;
    tiny.preserve_boundary = true;
    EXPECT_EQ(decimate_quadric(mesh, tiny), 20480U);
    EXPECT_EQ(mesh.faces.size(), 20480U);

    // A cap in the middle of the cost range halts the run before the target.
    Mesh mid = make_icosphere(5);
    DecimateOptions capped;
    capped.target_faces = 100;
    capped.max_error = 1e-4;
    capped.preserve_boundary = true;
    const std::size_t result = decimate_quadric(mid, capped);

    EXPECT_EQ(result, mid.faces.size());
    EXPECT_GT(mid.faces.size(), 100U) << "cap must stop the run early";
    EXPECT_LT(mid.faces.size(), 20480U) << "cap must allow some collapses";
    EXPECT_EQ(edge_stats(mid).non_manifold, 0U);

    // An uncapped run reaches the target, so the cap is what stopped it.
    Mesh uncapped = make_icosphere(5);
    DecimateOptions open;
    open.target_faces = 100;
    open.preserve_boundary = true;
    EXPECT_EQ(decimate_quadric(uncapped, open), 100U);
}

TEST(Decimate, PlaneDecimatesWithZeroErrorCap) {
    Mesh mesh = make_plane(40);
    ASSERT_EQ(mesh.faces.size(), 3200U);

    DecimateOptions options;
    options.target_faces = 500;
    options.max_error = 0.0;
    options.preserve_boundary = true;
    const std::size_t result = decimate_quadric(mesh, options);

    EXPECT_EQ(result, mesh.faces.size());
    // A plane's collapses are exactly free, so a zero cap still reduces it.
    EXPECT_LT(mesh.faces.size(), 2000U);
    EXPECT_LE(mesh.faces.size(), 500U);
    for (const Eigen::Vector3d& v : mesh.vertices) {
        ASSERT_LT(std::abs(v.z()), 1e-9);
    }
}

// ---------------------------------------------------------------------------
// 4. Attribute arrays stay sized to the vertex list.
// ---------------------------------------------------------------------------

TEST(Decimate, AttributesStayConsistent) {
    Mesh mesh = make_icosphere(4);
    const std::size_t vertex_count = mesh.vertices.size();
    mesh.colours.resize(vertex_count * 3);
    mesh.confidence.resize(vertex_count);
    for (std::size_t i = 0; i < vertex_count; ++i) {
        mesh.colours[i * 3] = static_cast<std::uint8_t>(i % 256);
        mesh.colours[i * 3 + 1] = static_cast<std::uint8_t>((i * 3) % 256);
        mesh.colours[i * 3 + 2] = static_cast<std::uint8_t>((i * 7) % 256);
        mesh.confidence[i] = 0.25F + 0.5F * static_cast<float>(i % 4) / 3.0F;
    }

    DecimateOptions options;
    options.target_faces = 400;
    options.preserve_boundary = true;
    decimate_quadric(mesh, options);

    EXPECT_EQ(mesh.colours.size(), mesh.vertices.size() * 3);
    EXPECT_EQ(mesh.confidence.size(), mesh.vertices.size());
    // Decimation refuses textured meshes, so no uv pool can appear or survive.
    EXPECT_TRUE(mesh.uvs.empty());
    EXPECT_TRUE(mesh.uv_faces.empty());
    for (const std::uint8_t c : mesh.colours) {
        static_cast<void>(c);  // in range by construction of uint8_t
    }
    for (const float c : mesh.confidence) {
        ASSERT_GE(c, 0.0F);
        ASSERT_LE(c, 1.0F);
    }
}

// ---------------------------------------------------------------------------
// 5. Contract: no-op paths, validation and determinism.
// ---------------------------------------------------------------------------

TEST(Decimate, NoOpTargetsReturnInputCount) {
    Mesh mesh = make_icosphere(3);
    const std::size_t before = mesh.faces.size();
    Mesh original = mesh;

    DecimateOptions zero;
    zero.target_faces = 0;
    EXPECT_EQ(decimate_quadric(mesh, zero), before);
    EXPECT_EQ(mesh.faces.size(), original.faces.size());

    DecimateOptions larger;
    larger.target_faces = before + 10;
    EXPECT_EQ(decimate_quadric(mesh, larger), before);
    EXPECT_EQ(mesh.faces.size(), original.faces.size());

    DecimateOptions equal;
    equal.target_faces = before;
    EXPECT_EQ(decimate_quadric(mesh, equal), before);
    EXPECT_TRUE(mesh == original);
}

TEST(Decimate, RejectsMalformedInput) {
    Mesh mesh = make_icosphere(2);

    Mesh bad_colours = mesh;
    bad_colours.colours.assign(3, 0);
    DecimateOptions options;
    options.target_faces = 10;
    EXPECT_THROW(decimate_quadric(bad_colours, options), ValueError);

    Mesh bad_confidence = mesh;
    bad_confidence.confidence.assign(2, 0.0F);
    EXPECT_THROW(decimate_quadric(bad_confidence, options), ValueError);

    Mesh bad_uvs = mesh;
    bad_uvs.uv_faces.assign(1, Eigen::Vector3i(0, 0, 0));
    EXPECT_THROW(decimate_quadric(bad_uvs, options), ValueError);

    // A textured mesh (uv pool but no per-corner map) is refused as well.
    Mesh uv_pool_only = mesh;
    uv_pool_only.uvs.assign(1, Eigen::Vector2f(0.0F, 0.0F));
    EXPECT_THROW(decimate_quadric(uv_pool_only, options), ValueError);

    Mesh bad_index = mesh;
    bad_index.faces[0][2] = static_cast<int>(mesh.vertices.size());
    EXPECT_THROW(decimate_quadric(bad_index, options), ValueError);

    Mesh negative_index = mesh;
    negative_index.faces[0][1] = -1;
    EXPECT_THROW(decimate_quadric(negative_index, options), ValueError);
}

TEST(Decimate, IsDeterministic) {
    Mesh first = make_icosphere(4);
    Mesh second = make_icosphere(4);

    DecimateOptions options;
    options.target_faces = 500;
    options.preserve_boundary = true;

    const std::size_t a = decimate_quadric(first, options);
    const std::size_t b = decimate_quadric(second, options);

    EXPECT_EQ(a, b);
    EXPECT_EQ(first.faces.size(), second.faces.size());
    EXPECT_EQ(first.vertices.size(), second.vertices.size());
    ASSERT_EQ(first.faces.size(), second.faces.size());
    for (std::size_t i = 0; i < first.faces.size(); ++i) {
        EXPECT_TRUE(first.faces[i] == second.faces[i]) << "face " << i;
    }
    for (std::size_t i = 0; i < first.vertices.size(); ++i) {
        EXPECT_TRUE((first.vertices[i] - second.vertices[i]).norm() == 0.0)
            << "vertex " << i;
    }
}

// ---------------------------------------------------------------------------
// 6. Throughput: a 327k-face sphere down to 20k in well under ten seconds.
// ---------------------------------------------------------------------------

TEST(Decimate, LargeMeshTiming) {
    Mesh mesh = make_icosphere(7);
    ASSERT_EQ(mesh.faces.size(), 327680U);

    DecimateOptions options;
    options.target_faces = 20000;
    options.preserve_boundary = true;

    const auto start = std::chrono::steady_clock::now();
    const std::size_t result = decimate_quadric(mesh, options);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double ms =
        std::chrono::duration<double, std::milli>(elapsed).count();

    std::cout << "decimate 327680 -> " << result << " faces in " << ms << " ms"
              << std::endl;

    EXPECT_LE(result, 20000U);
    EXPECT_EQ(result, mesh.faces.size());
    EXPECT_LT(ms, 10000.0);
    EXPECT_EQ(edge_stats(mesh).non_manifold, 0U);
}

}  // namespace
}  // namespace alphashape
