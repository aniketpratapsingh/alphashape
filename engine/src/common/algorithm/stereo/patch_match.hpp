#pragma once

// Multi-view PatchMatch depth estimation. For one reference view, iterate over
// depth and normal candidates (propagate + random search, red-black sweeps),
// aggregating a photometric cost over the selected neighbour views. The result
// is a per-pixel depth (distance along the bearing, see DepthMap), an optional
// normal, and the final cost.

#include <cstdint>
#include <functional>
#include <vector>

#include <Eigen/Core>

#include "image/image.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::stereo {

struct PMView {
    const FloatImage* image = nullptr;   // luminance at depth-map resolution
    Calibration calibration;             // matches image resolution (scaled_calibration)
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();  // world->camera
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

/// Called at the end of every refinement sweep with the fraction of the search
/// finished (0..1), so a front end can move a progress bar inside one camera
/// and a caller can cancel between sweeps. The hook may throw (a task's
/// CancelledError); both the CPU and the GPU implementation let it propagate
/// unchanged, which discards the half-swept map.
using SweepProgress = std::function<void(double fraction)>;

struct PatchMatchOptions {
    double depth_min = 0.0;
    double depth_max = 0.0;
    int window_radius = 5;       // window is (2r+1)^2, sampled every window_step px
    int window_step = 2;
    int iterations = 6;          // red-black sweeps
    int best_k = 0;              // neighbours averaged per pixel; 0 = ceil(n/2)
    double max_cost = 0.5;       // final pixels with cost above this get depth -1
    double min_texture_stddev = 1.0 / 255.0;  // skip windows flatter than this (intensity 0..1)
    std::uint64_t seed = 1;
    /// Optional per-sweep progress. Not part of a map's identity and never
    /// persisted; empty means "report nothing".
    SweepProgress on_sweep;
};

struct PatchMatchResult {
    int width = 0, height = 0;
    std::vector<float> depth;    // -1 = none
    std::vector<float> normals;  // 3*N, camera frame of the reference
    std::vector<float> cost;     // N
};

PatchMatchResult run_patch_match(const PMView& reference, const std::vector<PMView>& neighbours,
                                 const PatchMatchOptions& options);

}  // namespace alphashape::stereo
