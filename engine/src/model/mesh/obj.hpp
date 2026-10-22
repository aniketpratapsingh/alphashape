#pragma once

// OBJ export: write a Mesh, plus an optional companion material library, as a
// Wavefront .obj file.
//
// LAYERING: this writer lives in `alphashape_model` next to the Mesh it
// consumes, exactly as ply.hpp does. It needs no document/zip machinery, and
// chunk-level export (chunk-local -> world) is done by the caller passing
// `transform`.
//
// The file written matches the OBJ subset that Blender, MeshLab and Meshlab's
// siblings all read:
//
//   v  x y z            positions, printed round-trippably (%.17g)
//   v  x y z r g b      the same, with vertex colours in [0,1] when the mesh
//                       carries them (a widely supported non-standard extension)
//   vt u v              texture coordinates, with v flipped on the way out
//   f  a/ta b/tb c/tc   faces, 1-based, with per-corner texture indices
//   f  a b c            faces without texture coordinates
//
// UV CONVENTION. The Mesh stores v growing DOWN with (0,0) at the image's
// top-left. OBJ's texture origin is bottom-left, so v is flipped to 1 - v on
// the way out; u is unchanged.

#include <string>

#include <Eigen/Core>

#include "model/mesh/mesh.hpp"

namespace alphashape {

/// Write `mesh` as a Wavefront OBJ at `obj_path`.
///
/// When `texture_file` is non-empty AND the mesh has uv_faces, a companion
/// material library is written next to the OBJ (`<obj_path>'s stem + ".mtl"`)
/// and referenced with an `mtllib` line; the single material's `map_Kd` names
/// `texture_file`. When either is missing, no material library is written and
/// faces are emitted without texture indices.
///
/// When `transform` is non-null it is applied to every vertex position (as a
/// point, w = 1) — used to lift a chunk-local mesh into world coordinates.
/// Throws ProjectIOError on an IO failure, ValueError on an inconsistent mesh.
void write_obj(const std::string& obj_path, const Mesh& mesh, const std::string& texture_file,
               const Eigen::Matrix4d* transform = nullptr);

}  // namespace alphashape
