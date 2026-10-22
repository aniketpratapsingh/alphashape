#pragma once

// Z-buffer rasterisation of the mesh into a camera. Produces the per-pixel
// depth (and owning face id) that the texturing stage reads to decide which
// faces a camera actually sees.
//
// This header declares the interface only. render_depth() is a stub until the
// texturing milestone lands its implementation.

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "model/camera/calibration.hpp"
#include "model/mesh/mesh.hpp"

namespace alphashape {

/// Depth and face-id buffers for one camera view.
///
/// DEPTH SEMANTICS match the dense-stereo DepthMap exactly: `depth[i]` is the
/// distance from the camera centre along the pixel's UNIT bearing, in
/// chunk-local units, not a camera-frame Z coordinate. The pixel at
/// (i, j) is sampled at its centre (i + 0.5, j + 0.5). An empty pixel holds
/// +infinity in `depth` and -1 in `face`, so a "no surface here" test is a
/// comparison against +inf rather than a sentinel that could be a real depth.
struct DepthBuffer {
    int width = 0;
    int height = 0;
    /// width * height, row-major; distance along the unit bearing, +inf = empty.
    std::vector<float> depth;
    /// width * height, row-major; face id into Mesh::faces, -1 = empty.
    std::vector<std::int32_t> face;
};

/// Z-buffer render of `mesh` into a camera. Vertices are projected with the
/// full calibration (distortion included), triangles are rasterised with linear
/// interpolation in image space, and the pixel centre (i + 0.5, j + 0.5)
/// convention is used throughout. Triangles any of whose vertices project
/// behind the camera are skipped. Throws Error("render_depth: not implemented")
/// until implemented.
DepthBuffer render_depth(const Mesh& mesh, const Calibration& calibration, const Eigen::Matrix3d& R,
                         const Eigen::Vector3d& t);

}  // namespace alphashape
