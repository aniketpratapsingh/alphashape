// Marching tetrahedra: iso-surface extraction from a voxel-centre scalar field.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base/algorithm/voxel_grid.hpp"
#include "base/utils/error.hpp"
#include "model/mesh/marching_tetrahedra.hpp"
#include "model/mesh/mesh_ops.hpp"

namespace alphashape {
namespace {

GridSpec cube_grid(int n, double voxel = 1.0, const Eigen::Vector3d& origin = Eigen::Vector3d::Zero()) {
    GridSpec spec;
    spec.origin = origin;
    spec.voxel = voxel;
    spec.nx = n;
    spec.ny = n;
    spec.nz = n;
    return spec;
}

/// Fill a grid from an arbitrary function of the world-space point.
template <typename Fn>
VoxelGrid<float> field_from(const GridSpec& spec, Fn fn) {
    VoxelGrid<float> grid(spec);
    for (int z = 0; z < spec.nz; ++z) {
        for (int y = 0; y < spec.ny; ++y) {
            for (int x = 0; x < spec.nx; ++x) {
                grid.at(x, y, z) = static_cast<float>(fn(spec.centre(x, y, z)));
            }
        }
    }
    return grid;
}

Eigen::Vector3d face_normal(const Mesh& mesh, const Eigen::Vector3i& face) {
    const Eigen::Vector3d& a = mesh.vertices[static_cast<std::size_t>(face[0])];
    const Eigen::Vector3d& b = mesh.vertices[static_cast<std::size_t>(face[1])];
    const Eigen::Vector3d& c = mesh.vertices[static_cast<std::size_t>(face[2])];
    return (b - a).cross(c - a);
}

Eigen::Vector3d face_centroid(const Mesh& mesh, const Eigen::Vector3i& face) {
    return (mesh.vertices[static_cast<std::size_t>(face[0])] +
            mesh.vertices[static_cast<std::size_t>(face[1])] +
            mesh.vertices[static_cast<std::size_t>(face[2])]) /
           3.0;
}

/// Number of distinct positions, i.e. vertices after merging exact duplicates.
std::size_t distinct_positions(const Mesh& mesh) {
    std::vector<Eigen::Vector3d> sorted = mesh.vertices;
    std::sort(sorted.begin(), sorted.end(), [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
        if (a.x() != b.x()) return a.x() < b.x();
        if (a.y() != b.y()) return a.y() < b.y();
        return a.z() < b.z();
    });
    sorted.erase(std::unique(sorted.begin(), sorted.end(),
                             [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) { return a == b; }),
                 sorted.end());
    return sorted.size();
}

const Eigen::Vector3d kSphereCentre(24.0, 24.0, 24.0);
constexpr double kSphereRadius = 15.0;

VoxelGrid<float> sphere_field(const GridSpec& spec, const Eigen::Vector3d& centre, double radius) {
    return field_from(spec, [&](const Eigen::Vector3d& p) { return (p - centre).norm() - radius; });
}

// ---------------------------------------------------------------------------
// 1. Sphere: geometric accuracy, closed genus-0 topology, outward orientation.
// ---------------------------------------------------------------------------
TEST(MarchingTetrahedra, SphereVerticesLieOnTheSurface) {
    const GridSpec spec = cube_grid(48);
    const VoxelGrid<float> grid = sphere_field(spec, kSphereCentre, kSphereRadius);

    const Mesh mesh = extract_isosurface(grid, 0.0f);
    ASSERT_FALSE(mesh.empty());

    double max_error = 0.0;
    for (const Eigen::Vector3d& v : mesh.vertices) {
        max_error = std::max(max_error, std::abs((v - kSphereCentre).norm() - kSphereRadius));
    }
    std::cout << "[sphere] vertices=" << mesh.vertices.size() << " faces=" << mesh.faces.size()
              << " max vertex error=" << max_error << " voxels\n";
    EXPECT_LT(max_error, 0.05);
}

TEST(MarchingTetrahedra, SphereTopologyIsClosedSphere) {
    const GridSpec spec = cube_grid(48);
    const VoxelGrid<float> grid = sphere_field(spec, kSphereCentre, kSphereRadius);
    const Mesh mesh = extract_isosurface(grid, 0.0f);

    const MeshTopology topo = analyse_topology(mesh);
    EXPECT_EQ(topo.boundary_edges, 0U);
    EXPECT_EQ(topo.nonmanifold_edges, 0U);
    EXPECT_EQ(topo.components, 1U);
    EXPECT_EQ(topo.euler, 2L);
}

TEST(MarchingTetrahedra, SphereNormalsPointOutward) {
    const GridSpec spec = cube_grid(48);
    const VoxelGrid<float> grid = sphere_field(spec, kSphereCentre, kSphereRadius);
    const Mesh mesh = extract_isosurface(grid, 0.0f);

    for (const Eigen::Vector3i& face : mesh.faces) {
        const Eigen::Vector3d outward = face_centroid(mesh, face) - kSphereCentre;
        EXPECT_GT(face_normal(mesh, face).dot(outward), 0.0);
    }
}

TEST(MarchingTetrahedra, NoDuplicateVertexPositions) {
    const GridSpec spec = cube_grid(48);
    const VoxelGrid<float> grid = sphere_field(spec, kSphereCentre, kSphereRadius);
    const Mesh mesh = extract_isosurface(grid, 0.0f);
    EXPECT_EQ(distinct_positions(mesh), mesh.vertices.size());
}

// ---------------------------------------------------------------------------
// 2. Inverting the field flips every normal.
// ---------------------------------------------------------------------------
TEST(MarchingTetrahedra, InvertedFieldFlipsOrientations) {
    const GridSpec spec = cube_grid(48);
    VoxelGrid<float> grid = sphere_field(spec, kSphereCentre, kSphereRadius);
    for (float& value : grid.data) value = -value;

    const Mesh mesh = extract_isosurface(grid, 0.0f);
    ASSERT_FALSE(mesh.empty());
    for (const Eigen::Vector3i& face : mesh.faces) {
        const Eigen::Vector3d outward = face_centroid(mesh, face) - kSphereCentre;
        EXPECT_LT(face_normal(mesh, face).dot(outward), 0.0);
    }
}

// ---------------------------------------------------------------------------
// 3. Plane: vertices exactly on the plane, boundary only at the grid border.
// ---------------------------------------------------------------------------
TEST(MarchingTetrahedra, PlaneVerticesLieExactlyOnThePlane) {
    const GridSpec spec = cube_grid(20);
    const double level = 10.3;
    const VoxelGrid<float> grid =
        field_from(spec, [&](const Eigen::Vector3d& p) { return p.z() - level; });

    const Mesh mesh = extract_isosurface(grid, 0.0f);
    ASSERT_FALSE(mesh.empty());
    for (const Eigen::Vector3d& v : mesh.vertices) {
        EXPECT_NEAR(v.z(), level, 1e-9);
    }
}

TEST(MarchingTetrahedra, PlaneBoundaryEdgesOnlyOnGridBorder) {
    const GridSpec spec = cube_grid(20);
    const VoxelGrid<float> grid =
        field_from(spec, [](const Eigen::Vector3d& p) { return p.z() - 10.3; });
    const Mesh mesh = extract_isosurface(grid, 0.0f);
    ASSERT_FALSE(mesh.empty());

    // Count edge uses; the surface is a rectangle, so a one-face edge is a
    // border edge and both of its endpoints must sit on the grid border.
    std::vector<std::size_t> uses;
    std::unordered_map<std::uint64_t, std::size_t> index_of;
    std::vector<std::pair<int, int>> edges;
    for (const Eigen::Vector3i& face : mesh.faces) {
        for (int e = 0; e < 3; ++e) {
            const int a = face[e];
            const int b = face[(e + 1) % 3];
            const int lo = std::min(a, b);
            const int hi = std::max(a, b);
            const std::uint64_t key = (static_cast<std::uint64_t>(lo) << 32) |
                                      static_cast<std::uint64_t>(hi);
            const auto inserted = index_of.emplace(key, edges.size());
            if (inserted.second) {
                edges.emplace_back(lo, hi);
                uses.push_back(1);
            } else {
                ++uses[inserted.first->second];
            }
        }
    }

    const double border_lo = 0.5;
    const double border_hi = static_cast<double>(spec.nx) - 0.5;
    bool saw_boundary = false;
    for (std::size_t i = 0; i < edges.size(); ++i) {
        if (uses[i] != 1) {
            EXPECT_LE(uses[i], 2U);  // a manifold sheet
            continue;
        }
        saw_boundary = true;
        for (int endpoint : {edges[i].first, edges[i].second}) {
            const Eigen::Vector3d& p = mesh.vertices[static_cast<std::size_t>(endpoint)];
            EXPECT_TRUE(std::abs(p.x() - border_lo) < 1e-9 || std::abs(p.x() - border_hi) < 1e-9 ||
                        std::abs(p.y() - border_lo) < 1e-9 || std::abs(p.y() - border_hi) < 1e-9);
        }
    }
    EXPECT_TRUE(saw_boundary);
}

// ---------------------------------------------------------------------------
// 4. Mask: trimming, and the size check.
// ---------------------------------------------------------------------------
TEST(MarchingTetrahedra, MaskTrimsToTheUnmaskedHalf) {
    const GridSpec spec = cube_grid(48);
    const VoxelGrid<float> grid = sphere_field(spec, kSphereCentre, kSphereRadius);

    const int half = spec.nx / 2;
    std::vector<std::uint8_t> mask(spec.size(), 0);
    for (int z = 0; z < spec.nz; ++z)
        for (int y = 0; y < spec.ny; ++y)
            for (int x = half; x < spec.nx; ++x) mask[spec.index(x, y, z)] = 1;

    const Mesh masked = extract_isosurface(grid, 0.0f, &mask);
    ASSERT_FALSE(masked.empty());
    EXPECT_GT(analyse_topology(masked).boundary_edges, 0U);  // the cut is open
    for (const Eigen::Vector3d& v : masked.vertices) {
        EXPECT_GE(v.x(), static_cast<double>(half) - 1e-9);
    }

    // The full surface is larger, and confined to the same half of space.
    const Mesh full = extract_isosurface(grid, 0.0f);
    EXPECT_LT(masked.faces.size(), full.faces.size());
}

TEST(MarchingTetrahedra, MaskOfWrongSizeThrows) {
    const GridSpec spec = cube_grid(16);
    const VoxelGrid<float> grid = sphere_field(spec, Eigen::Vector3d(8.0, 8.0, 8.0), 5.0);
    std::vector<std::uint8_t> mask(spec.size() - 1, 1);
    EXPECT_THROW(extract_isosurface(grid, 0.0f, &mask), ValueError);
}

// ---------------------------------------------------------------------------
// 5. Empty results and multiple components.
// ---------------------------------------------------------------------------
TEST(MarchingTetrahedra, FieldEntirelyAboveIsoYieldsEmptyMesh) {
    const GridSpec spec = cube_grid(16);
    VoxelGrid<float> grid(spec, 1.0f);  // every sample above iso = 0
    const Mesh mesh = extract_isosurface(grid, 0.0f);
    EXPECT_TRUE(mesh.empty());
    EXPECT_TRUE(mesh.vertices.empty());
}

TEST(MarchingTetrahedra, TwoDisjointSpheresGiveTwoComponents) {
    const GridSpec spec = cube_grid(48);
    const Eigen::Vector3d a(12.0, 24.0, 24.0);
    const Eigen::Vector3d b(36.0, 24.0, 24.0);
    const VoxelGrid<float> grid =
        field_from(spec, [&](const Eigen::Vector3d& p) {
            return std::min((p - a).norm() - 5.0, (p - b).norm() - 5.0);
        });

    const Mesh mesh = extract_isosurface(grid, 0.0f);
    ASSERT_FALSE(mesh.empty());
    const MeshTopology topo = analyse_topology(mesh);
    EXPECT_EQ(topo.components, 2U);
    EXPECT_EQ(topo.boundary_edges, 0U);
    EXPECT_EQ(topo.euler, 4L);
}

// ---------------------------------------------------------------------------
// 6. Timing at production resolution (no assertion beyond non-emptiness).
// ---------------------------------------------------------------------------
TEST(MarchingTetrahedra, Extracts128CubedSphere) {
    const GridSpec spec = cube_grid(128);
    const VoxelGrid<float> grid =
        sphere_field(spec, Eigen::Vector3d(64.0, 64.0, 64.0), 40.0);
    const auto start = std::chrono::steady_clock::now();
    const Mesh mesh = extract_isosurface(grid, 0.0f);
    const auto stop = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "[sphere 128^3] vertices=" << mesh.vertices.size()
              << " faces=" << mesh.faces.size() << " time=" << ms << " ms\n";
    EXPECT_FALSE(mesh.empty());
    EXPECT_EQ(analyse_topology(mesh).euler, 2L);
}

}  // namespace
}  // namespace alphashape
