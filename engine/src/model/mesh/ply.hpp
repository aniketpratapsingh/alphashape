#pragma once

// PLY export: write a Mesh or a PointCloud as a binary little-endian PLY file.
//
// LAYERING: both writers live in `alphashape_model`, next to the Mesh and
// PointCloud types they consume (model/mesh/). This is the layering-clean
// choice: `alphashape_model` must not depend on common/, and PLY needs no
// document/zip machinery, so putting PointCloud in model/ (rather than
// marshalling it through common/io) lets one writer pair serve both geometry
// types where they are declared. Chunk-level export (chunk-local -> world via
// the chunk transform) is done by the caller passing `transform`.
//
// The property set written is fixed and matches what later stages produce:
//
//   vertex:  double x, y, z
//            float nx, ny, nz           (only if normals are present)
//            uchar red, green, blue     (only if colours are present)
//            float confidence           (only if confidence is present)
//   face:    list uchar int vertex_indices   (mesh only)

#include <string>
#include <vector>

#include <Eigen/Core>

#include "model/mesh/mesh.hpp"
#include "model/mesh/point_cloud.hpp"

namespace alphashape {

/// Write `mesh` as a binary little-endian PLY. When `transform` is non-null it
/// is applied to every vertex position (as a point, w = 1) — used to lift a
/// chunk-local mesh into world coordinates. Throws ProjectIOError on an IO
/// failure.
void write_ply(const std::string& path, const Mesh& mesh,
               const Eigen::Matrix4d* transform = nullptr);

/// Write `cloud` as a binary little-endian PLY. `transform`, when non-null, maps
/// positions (w = 1) and the rotation part maps normals. Throws ProjectIOError
/// on an IO failure.
void write_ply(const std::string& path, const PointCloud& cloud,
               const Eigen::Matrix4d* transform = nullptr);

}  // namespace alphashape
