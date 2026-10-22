#pragma once

// Mesh post-processing: per-vertex normals, topology statistics and the two
// container-cleanup operations the meshing pipeline runs after extraction.
//
// These operate on plain `Mesh` values and carry no state, so they live in
// `alphashape_model` alongside the mesh type itself.

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "model/mesh/mesh.hpp"

namespace alphashape {

/// One unit normal per vertex, area-weighted: each incident face contributes its
/// geometric normal scaled by its area (the unnormalised cross product), and the
/// sum is normalised. A vertex with no incident face, or cancellation to a
/// zero-length sum, gets a zero normal.
std::vector<Eigen::Vector3f> vertex_normals(const Mesh& mesh);

/// Structural summary of a mesh's connectivity.
struct MeshTopology {
    /// Edges used by exactly one face.
    std::size_t boundary_edges = 0;
    /// Edges used by three or more faces.
    std::size_t nonmanifold_edges = 0;
    /// Connected components under shared-edge face adjacency.
    std::size_t components = 0;
    /// V - E + F.
    long euler = 0;
};

MeshTopology analyse_topology(const Mesh& mesh);

/// Remove connected components (face adjacency via shared edges) with fewer
/// faces than `min_fraction` times the largest component's face count. Returns
/// the number of faces removed. Colours and confidence of surviving vertices are
/// kept in sync, and the per-face uv_faces entries of surviving faces are dropped
/// or kept in lockstep (the uv pool itself is untouched); see
/// remove_unreferenced_vertices.
std::size_t remove_small_components(Mesh& mesh, double min_fraction);

/// Drop vertices no face references, remap the face indices, and keep the
/// per-vertex attribute vectors (colours / confidence) in sync. The texture pool
/// (uvs) and the per-face corner map (uv_faces) index their own arrays rather
/// than the vertices, so both are left untouched. A no-op on an already-compact
/// mesh.
void remove_unreferenced_vertices(Mesh& mesh);

}  // namespace alphashape
