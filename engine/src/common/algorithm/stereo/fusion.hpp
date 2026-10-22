#pragma once

// Depth-map fusion: back-project every valid depth into chunk-local space and
// merge the points (optionally onto a voxel grid, keeping one representative
// per cell) into a single dense point cloud, with per-point normals, RGB and
// confidence carried through where available.

#include <vector>

#include <Eigen/Core>

#include "common/core/chunk.hpp"
#include "image/image.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::stereo {

struct FusionInput {
    const DepthMap* map = nullptr;
    Calibration calibration;      // scaled to the map's resolution
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    const Image* image = nullptr; // FULL-resolution image for colour, or nullptr
};

struct FusionOptions {
    double voxel_size = 0.0;      // 0 = automatic
    bool colours = true;
    float min_confidence = 0.0F;
};

PointCloud fuse_depth_maps(const std::vector<FusionInput>& inputs, const FusionOptions& options);

}  // namespace alphashape::stereo
