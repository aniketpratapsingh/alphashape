#pragma once

// AlignCameras: the structure-from-motion stage.
//
// AlignCameras runs the incremental SfM pipeline over the
// chunk's tie points and writes a pose onto every registered camera plus a
// triangulated position onto every tie point. Shape: one task, a small parameter set, two stages, and an atomic
// commit that leaves the chunk untouched on failure or cancellation.
//
// The algorithm itself is build_incremental (sfm/structure_builder.hpp): choose
// an initial pair, seed it, resect cameras one at a time, triangulate, and refine
// with bundle adjustment. This task is the adapter between the chunk's data
// model and the builder's inputs/outputs:
//
//   * chunk  -> BuilderInput:  one BuilderCamera per camera with a keypoint set,
//     its sensor's Calibration, its keypoints converted to the Calibration pixel
//     convention (keypoint_to_pixel), and (when continuing) its current pose.
//   * BuilderResult -> chunk:  each registered camera's pose (world -> camera),
//     each track's point, and the chunk's bounding region.
//
// Conventions and ownership notes:
//   * Camera::pose is the 4x4 world -> camera transform; the builder's
//     AbsolutePose is (R, t) with X_cam = R X_world + t, so the 4x4 is exactly
//     [R t; 0 1] and the conversion is exact in both directions.
//   * On a commit every camera in the chunk gets a value: registered cameras are
//     enabled with their pose, unregistered ones are disabled with the identity
//     pose. A camera that could not be aligned is therefore never left with a
//     stale pose.
//   * The chunk transform is left alone: georeferencing it is the next milestone.

#include "common/tasks/task.hpp"

namespace alphashape {

class AlignCamerasTask : public AbstractTask {
public:
    std::string name() const override { return "AlignCameras"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
