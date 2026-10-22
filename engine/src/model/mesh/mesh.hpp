#pragma once

// Indexed triangle mesh — the milestone-7 surface product.
//
// Library layering: `alphashape_model` sits below `alphashape_core` and must
// not include anything from common/. The struct therefore lives here, and
// common/core/chunk.hpp includes this header rather than declaring Mesh itself
// (it used to, up to milestone 6).

#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace alphashape {

/// An indexed triangle mesh in chunk-local coordinates.
///
/// Faces are wound counter-clockwise as seen from *outside* the surface, so the
/// geometric normal of a face (v0, v1, v2)
///
///     n = (v1 - v0) x (v2 - v0)
///
/// points OUT of the surface. Meshing stages that produce a mesh (marching
/// tetrahedra, Poisson) are responsible for this orientation; every later stage
/// (decimation, texturing, export) relies on it.
struct Mesh {
    /// Positions in chunk-local coordinates.
    std::vector<Eigen::Vector3d> vertices;
    /// Triangle indices into `vertices`; CCW seen from outside.
    std::vector<Eigen::Vector3i> faces;

    // -- texture coordinates (milestone 8) ---------------------------------
    //
    // UV CONVENTION. u grows to the right, v grows DOWN; (0, 0) is the
    // top-left corner of the texture image and (1, 1) is the bottom-right.
    // Texel (i, j) therefore has its centre at ((i + 0.5) / W, (j + 0.5) / H),
    // and image row 0 is v = 0.
    //
    // Both arrays are empty on a mesh that has not been textured. When uvs (the
    // pool) is non-empty it holds texture coordinates that corners index into;
    // it is NOT one entry per vertex, because an OBJ-style layout lets a single
    // vertex carry several coordinates where the surface is seamed (the two
    // sides of a seam share a position but not a texture coordinate). uv_faces
    // is either empty or holds exactly one Vector3i per face: the three pool
    // indices for that face's corners, in the same (v0, v1, v2) order as faces.
    /// Texture-coordinate pool; empty, or arbitrary size, indexed by uv_faces.
    std::vector<Eigen::Vector2f> uvs;
    /// Empty, or one entry per face: pool indices for the face's 3 corners.
    std::vector<Eigen::Vector3i> uv_faces;

    /// Empty, or 3 bytes (RGB) per vertex.
    std::vector<std::uint8_t> colours;
    /// Empty, or one confidence value per vertex.
    std::vector<float> confidence;

    /// A mesh counts as empty when it carries no faces — the meshing stages
    /// never emit loose vertices.
    bool empty() const { return faces.empty(); }

    bool operator==(const Mesh& other) const {
        return vertices == other.vertices && faces == other.faces && uvs == other.uvs &&
               uv_faces == other.uv_faces && colours == other.colours &&
               confidence == other.confidence;
    }
};

}  // namespace alphashape
