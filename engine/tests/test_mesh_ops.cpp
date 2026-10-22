// Mesh utilities: vertex normals, topology analysis, component pruning and
// vertex compaction.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base/algorithm/voxel_grid.hpp"
#include "base/utils/error.hpp"
#include "model/mesh/marching_tetrahedra.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/mesh_ops.hpp"

namespace alphashape {
namespace {

/// Axis-aligned unit cube as 12 CCW-outward triangles, shared vertices. Used as
/// the hand-built closed manifold reference for the topology checks.
Mesh unit_cube() {
    Mesh mesh;
    mesh.vertices = {
        {0.0, 0.0, 0.0},  // 0
        {1.0, 0.0, 0.0},  // 1
        {1.0, 1.0, 0.0},  // 2
        {0.0, 1.0, 0.0},  // 3
        {0.0, 0.0, 1.0},  // 4
        {1.0, 0.0, 1.0},  // 5
        {1.0, 1.0, 1.0},  // 6
        {0.0, 1.0, 1.0},  // 7
    };
    mesh.faces = {
        {0, 3, 2}, {0, 2, 1},  // z = 0, normal -z
        {4, 5, 6}, {4, 6, 7},  // z = 1, normal +z
        {0, 1, 5}, {0, 5, 4},  // y = 0, normal -y
        {2, 3, 7}, {2, 7, 6},  // y = 1, normal +y
        {0, 4, 7}, {0, 7, 3},  // x = 0, normal -x
        {1, 2, 6}, {1, 6, 5},  // x = 1, normal +x
    };
    return mesh;
}

Mesh sphere_mesh(const Eigen::Vector3d& centre, double radius, int n = 40) {
    GridSpec spec;
    spec.origin = Eigen::Vector3d::Zero();
    spec.voxel = 1.0;
    spec.nx = n;
    spec.ny = n;
    spec.nz = n;
    VoxelGrid<float> grid(spec);
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x)
                grid.at(x, y, z) =
                    static_cast<float>((spec.centre(x, y, z) - centre).norm() - radius);
    return extract_isosurface(grid, 0.0f);
}

/// Attach synthetic attributes that encode each vertex's index, so a remap that
/// loses sync shows up as a mismatch. Also attaches a per-corner uv layout whose
/// first pool index is derived from the face index, so a face/uv_faces desync
/// (e.g. remove_small_components dropping one but not the other) is detectable.
void tag_attributes(Mesh& mesh) {
    mesh.colours.resize(3 * mesh.vertices.size());
    mesh.confidence.resize(mesh.vertices.size());
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        const auto byte = static_cast<std::uint8_t>(i % 251);
        mesh.colours[3 * i + 0] = byte;
        mesh.colours[3 * i + 1] = static_cast<std::uint8_t>(byte + 1U);
        mesh.colours[3 * i + 2] = static_cast<std::uint8_t>(byte + 2U);
        mesh.confidence[i] = static_cast<float>(i);
    }
    // One uv per face corner, each tagged with its face index in the pool entry
    // it uses. The pool is three entries per face, so entry 3*f carries face f.
    mesh.uvs.resize(3 * mesh.faces.size());
    mesh.uv_faces.resize(mesh.faces.size());
    for (std::size_t f = 0; f < mesh.faces.size(); ++f) {
        for (int k = 0; k < 3; ++k) {
            mesh.uvs[3 * f + static_cast<std::size_t>(k)] =
                Eigen::Vector2f(static_cast<float>(f), static_cast<float>(k));
            mesh.uv_faces[f][k] = static_cast<int>(3 * f) + k;
        }
    }
}

// ---------------------------------------------------------------------------
// vertex_normals
// ---------------------------------------------------------------------------
TEST(MeshOps, CubeVertexNormalIsUnitAndFacesOutward) {
    const Mesh cube = unit_cube();
    const std::vector<Eigen::Vector3f> normals = vertex_normals(cube);
    ASSERT_EQ(normals.size(), cube.vertices.size());

    // Every cube corner is on the outward diagonal, so each vertex normal must
    // point the same way (component signs equal to the corner's offset from the
    // cube centre).
    for (std::size_t i = 0; i < normals.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(normals[i].norm()), 1.0, 1e-6);
        const Eigen::Vector3d dir =
            (cube.vertices[i] - Eigen::Vector3d(0.5, 0.5, 0.5)).normalized();
        EXPECT_GT(normals[i].cast<double>().dot(dir), 0.0);
    }
}

TEST(MeshOps, SphereNormalsPointOutwardAndAreUnit) {
    const Eigen::Vector3d centre(20.0, 20.0, 20.0);
    const double radius = 12.0;
    const Mesh mesh = sphere_mesh(centre, radius);
    ASSERT_FALSE(mesh.empty());

    const std::vector<Eigen::Vector3f> normals = vertex_normals(mesh);
    ASSERT_EQ(normals.size(), mesh.vertices.size());
    for (std::size_t i = 0; i < normals.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(normals[i].norm()), 1.0, 1e-4);
        const Eigen::Vector3d outward = (mesh.vertices[i] - centre).normalized();
        EXPECT_GT(normals[i].cast<double>().dot(outward), 0.9);
    }
}

TEST(MeshOps, IsolatedVertexGetsZeroNormal) {
    Mesh cube = unit_cube();
    cube.vertices.emplace_back(9.0, 9.0, 9.0);  // referenced by no face
    const std::vector<Eigen::Vector3f> normals = vertex_normals(cube);
    ASSERT_EQ(normals.size(), cube.vertices.size());
    EXPECT_EQ(normals.back(), Eigen::Vector3f::Zero());
}

// ---------------------------------------------------------------------------
// analyse_topology
// ---------------------------------------------------------------------------
TEST(MeshOps, CubeTopologyIsClosedEulerTwo) {
    const MeshTopology topo = analyse_topology(unit_cube());
    EXPECT_EQ(topo.boundary_edges, 0U);
    EXPECT_EQ(topo.nonmanifold_edges, 0U);
    EXPECT_EQ(topo.components, 1U);
    EXPECT_EQ(topo.euler, 2L);  // V8 - E12 + F12
}

TEST(MeshOps, EmptyMeshHasZeroTopology) {
    const MeshTopology topo = analyse_topology(Mesh{});
    EXPECT_EQ(topo.boundary_edges, 0U);
    EXPECT_EQ(topo.nonmanifold_edges, 0U);
    EXPECT_EQ(topo.components, 0U);
    EXPECT_EQ(topo.euler, 0L);
}

TEST(MeshOps, SingleTriangleHasThreeBoundaryEdges) {
    Mesh triangle;
    triangle.vertices = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    triangle.faces = {{0, 1, 2}};
    const MeshTopology topo = analyse_topology(triangle);
    EXPECT_EQ(topo.boundary_edges, 3U);
    EXPECT_EQ(topo.components, 1U);
    EXPECT_EQ(topo.euler, 1L);  // 3 - 3 + 1
}

TEST(MeshOps, ThreeFacesSharingAnEdgeAreNonManifold) {
    Mesh fin;
    fin.vertices = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                    {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, -1.0, 0.0}};
    fin.faces = {{0, 1, 2}, {1, 0, 3}, {0, 1, 4}};  // edge (0,1) used three times
    const MeshTopology topo = analyse_topology(fin);
    EXPECT_EQ(topo.nonmanifold_edges, 1U);
    EXPECT_EQ(topo.boundary_edges, 6U);
    EXPECT_EQ(topo.components, 1U);
}

TEST(MeshOps, DisjointTrianglesAreTwoComponents) {
    Mesh mesh;
    mesh.vertices = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
                     {5.0, 0.0, 0.0}, {6.0, 0.0, 0.0}, {5.0, 1.0, 0.0}};
    mesh.faces = {{0, 1, 2}, {3, 4, 5}};
    const MeshTopology topo = analyse_topology(mesh);
    EXPECT_EQ(topo.components, 2U);
    EXPECT_EQ(topo.euler, 2L);
}

// ---------------------------------------------------------------------------
// remove_small_components
// ---------------------------------------------------------------------------
TEST(MeshOps, RemoveSmallComponentsKeepsTheLargeOneAndSyncsAttributes) {
    Mesh big = sphere_mesh(Eigen::Vector3d(14.0, 14.0, 14.0), 8.0);
    const Mesh small = sphere_mesh(Eigen::Vector3d(32.0, 32.0, 32.0), 3.0, 40);
    ASSERT_FALSE(big.empty());
    ASSERT_FALSE(small.empty());

    const std::size_t big_vertices = big.vertices.size();
    const std::size_t big_faces = big.faces.size();
    const std::size_t small_faces = small.faces.size();

    // Concatenate with a per-vertex tag that lets us spot desyncs.
    for (std::size_t i = 0; i < small.vertices.size(); ++i) big.vertices.push_back(small.vertices[i]);
    for (const Eigen::Vector3i& f : small.faces) {
        big.faces.emplace_back(f[0] + static_cast<int>(big_vertices),
                               f[1] + static_cast<int>(big_vertices),
                               f[2] + static_cast<int>(big_vertices));
    }
    // Small component is far away: tag every vertex with its own index.
    tag_attributes(big);

    const std::size_t removed = remove_small_components(big, 0.5);
    EXPECT_EQ(removed, small_faces);
    EXPECT_EQ(big.vertices.size(), big_vertices);
    EXPECT_EQ(big.faces.size(), big_faces);

    // Attributes stayed in sync: vertex i still carries the tag written for it.
    ASSERT_EQ(big.colours.size(), 3 * big.vertices.size());
    ASSERT_EQ(big.confidence.size(), big.vertices.size());
    for (std::size_t i = 0; i < big.vertices.size(); ++i) {
        const auto byte = static_cast<std::uint8_t>(i % 251);
        EXPECT_EQ(big.colours[3 * i + 0], byte);
        EXPECT_EQ(big.colours[3 * i + 1], static_cast<std::uint8_t>(byte + 1U));
        EXPECT_EQ(big.colours[3 * i + 2], static_cast<std::uint8_t>(byte + 2U));
        EXPECT_FLOAT_EQ(big.confidence[i], static_cast<float>(i));
    }
    // Dropping faces must drop their uv_faces entries in lockstep, while the uv
    // pool is untouched (it is indexed by uv_faces, not by face position).
    ASSERT_EQ(big.uv_faces.size(), big.faces.size());
    EXPECT_EQ(big.uvs.size(), 3 * (big_faces + small_faces));
    // The surviving faces are the original 0..big_faces-1, so their uv triples
    // still index the same pool entries.
    for (std::size_t f = 0; f < big.faces.size(); ++f) {
        EXPECT_EQ(big.uv_faces[f], Eigen::Vector3i(static_cast<int>(3 * f) + 0,
                                                   static_cast<int>(3 * f) + 1,
                                                   static_cast<int>(3 * f) + 2));
    }
    EXPECT_EQ(analyse_topology(big).components, 1U);
}

TEST(MeshOps, RemoveSmallComponentsZeroFractionRemovesNothing) {
    Mesh cube = unit_cube();
    EXPECT_EQ(remove_small_components(cube, 0.0), 0U);
    EXPECT_EQ(cube.faces.size(), 12U);
}

TEST(MeshOps, RemoveSmallComponentsFractionOneKeepsOnlyLargest) {
    Mesh big = sphere_mesh(Eigen::Vector3d(20.0, 20.0, 20.0), 10.0);
    const Mesh small = sphere_mesh(Eigen::Vector3d(32.0, 32.0, 32.0), 2.0);
    for (std::size_t i = 0; i < small.vertices.size(); ++i) big.vertices.push_back(small.vertices[i]);
    const int offset = static_cast<int>(big.vertices.size() - small.vertices.size());
    for (const Eigen::Vector3i& f : small.faces)
        big.faces.emplace_back(f[0] + offset, f[1] + offset, f[2] + offset);

    // min_fraction = 1 keeps only components exactly as large as the largest.
    const std::size_t removed = remove_small_components(big, 1.0);
    EXPECT_EQ(removed, small.faces.size());
    EXPECT_EQ(analyse_topology(big).components, 1U);
}

TEST(MeshOps, RemoveSmallComponentsRejectsOutOfRangeFraction) {
    Mesh cube = unit_cube();
    EXPECT_THROW(remove_small_components(cube, -0.1), ValueError);
    EXPECT_THROW(remove_small_components(cube, 1.5), ValueError);
}

// ---------------------------------------------------------------------------
// remove_unreferenced_vertices
// ---------------------------------------------------------------------------
TEST(MeshOps, RemoveUnreferencedVerticesRemapsFacesAndKeepsOrder) {
    Mesh mesh;
    // Vertex 1 and 3 are loose; keep 0 and 5, and reference a fresh one.
    mesh.vertices = {{0.0, 0.0, 0.0},   // 0 kept
                     {9.0, 9.0, 9.0},   // 1 dropped
                     {1.0, 0.0, 0.0},   // 2 dropped
                     {8.0, 8.0, 8.0},   // 3 dropped
                     {0.0, 1.0, 0.0}};  // 4 kept
    mesh.faces = {{0, 4, 0}};           // degenerate topologically but valid indices
    tag_attributes(mesh);

    remove_unreferenced_vertices(mesh);
    ASSERT_EQ(mesh.vertices.size(), 2U);
    EXPECT_EQ(mesh.vertices[0], Eigen::Vector3d(0.0, 0.0, 0.0));
    EXPECT_EQ(mesh.vertices[1], Eigen::Vector3d(0.0, 1.0, 0.0));
    EXPECT_EQ(mesh.faces.size(), 1U);
    EXPECT_EQ(mesh.faces[0], Eigen::Vector3i(0, 1, 0));

    // Attribute of kept vertex 0 (old index 0) and kept vertex 1 (old index 4).
    ASSERT_EQ(mesh.colours.size(), 6U);
    EXPECT_EQ(mesh.colours[0], 0U);
    EXPECT_EQ(mesh.colours[3], 4U);
    EXPECT_FLOAT_EQ(mesh.confidence[0], 0.0f);
    EXPECT_FLOAT_EQ(mesh.confidence[1], 4.0f);
    // The uv pool and uv_faces are indexed independently of `vertices`, so
    // compaction leaves both exactly as they were: one face's worth of corners.
    ASSERT_EQ(mesh.uvs.size(), 3U);
    ASSERT_EQ(mesh.uv_faces.size(), 1U);
    EXPECT_EQ(mesh.uv_faces[0], Eigen::Vector3i(0, 1, 2));
    EXPECT_FLOAT_EQ(mesh.uvs[1].y(), 1.0f);
}

TEST(MeshOps, RemoveUnreferencedVerticesIsNoOpWhenAlreadyCompact) {
    Mesh cube = unit_cube();
    const Mesh before = cube;
    remove_unreferenced_vertices(cube);
    EXPECT_TRUE(cube == before);
}

TEST(MeshOps, RemoveUnreferencedVerticesDropsLooseTailOnly) {
    Mesh cube = unit_cube();
    const std::size_t original = cube.vertices.size();
    tag_attributes(cube);
    cube.vertices.emplace_back(4.0, 4.0, 4.0);  // loose
    cube.colours.push_back(7U);
    cube.colours.push_back(7U);
    cube.colours.push_back(7U);
    cube.confidence.push_back(7.0f);
    // A loose vertex adds no face, so the uv pool and uv_faces are unchanged.
    cube.uvs.emplace_back(7.0f, 7.0f);

    const std::size_t uv_pool = cube.uvs.size();
    const std::size_t uv_face_count = cube.uv_faces.size();
    remove_unreferenced_vertices(cube);
    EXPECT_EQ(cube.vertices.size(), original);
    EXPECT_EQ(cube.colours.size(), 3 * original);
    EXPECT_EQ(cube.confidence.size(), original);
    EXPECT_EQ(cube.uvs.size(), uv_pool);
    EXPECT_EQ(cube.uv_faces.size(), uv_face_count);
}

}  // namespace
}  // namespace alphashape
