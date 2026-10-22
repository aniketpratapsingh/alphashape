#pragma once

// OptimizeCameras: refine intrinsics, poses and points with bundle adjustment.
//
// This is the self-calibration stage. It is the adapter between the chunk's data
// model and sfm::bundle_adjust (bundle_adjustment.hpp):
//
//   * chunk -> BAProblem: one pose per aligned (enabled) camera, one calibration
//     per sensor used by those cameras (shared by every camera on the sensor),
//     one point per triangulated track, and one observation per track
//     observation in an aligned camera (keypoint_to_pixel).
//   * free mask: for each sensor, coefficient X is free iff the task's fit_X is
//     true AND X is not in that sensor's fixed_params.
//   * GNSS priors: on a georeferenced chunk (transform != identity) every aligned
//     camera with an enabled reference gets a centre prior in the local ENU frame.
//   * BAProblem -> chunk: sensor calibrations, camera poses and track points,
//     committed atomically through a staged copy.
//
// Gauge. The objective is invariant under a global similarity. When at least
// three GNSS priors have both horizontal axes constrained and their centres are
// not collinear, the priors pin all 7 DoF, so no pose is fixed and no scale
// anchor is set. Otherwise the first aligned camera is held fully constant and
// the scale is anchored on the aligned camera with the largest baseline to it
// (choose_scale_anchor_axis).

#include "common/tasks/task.hpp"

namespace alphashape {

class OptimizeCamerasTask : public AbstractTask {
public:
    std::string name() const override { return "OptimizeCameras"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
