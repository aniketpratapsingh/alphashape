#pragma once

// Dense point cloud — the fused output of the depth maps (milestone 6).
//
// Library layering: like Mesh, this lives in `alphashape_model` so the PLY
// export writer (and any other geometry tool in model/) can consume a cloud
// without reaching up into common/. common/core/chunk.hpp includes this header.

#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace alphashape {

/// A set of points in chunk-local coordinates with optional per-point
/// attributes. Every attribute vector is either empty or one entry per point
/// (three bytes per point for colours).
struct PointCloud {
    /// Positions in chunk-local coordinates.
    std::vector<Eigen::Vector3d> points;
    /// Empty, or one unit normal per point, in the chunk-local frame.
    std::vector<Eigen::Vector3f> normals;
    /// Empty, or 3 bytes (RGB) per point.
    std::vector<std::uint8_t> colours;
    /// Empty, or one confidence value per point.
    std::vector<float> confidence;

    bool operator==(const PointCloud& other) const {
        return points == other.points && normals == other.normals &&
               colours == other.colours && confidence == other.confidence;
    }
};

}  // namespace alphashape
