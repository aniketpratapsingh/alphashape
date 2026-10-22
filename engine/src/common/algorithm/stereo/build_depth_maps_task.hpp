#pragma once

// BuildDepthMaps: the dense depth-map stage of the stereo pipeline.
//
// This is the adapter between the chunk's data model and the stereo:: primitives
// (view_selection.hpp, patch_match.hpp, depth_filter.hpp):
//
//   * chunk -> neighbourhoods: stereo::select_views picks, for every aligned
//     camera, the neighbours that make useful references, together with a
//     depth range (along the unit bearing) covering the visible surface.
//   * per reference camera: the camera's luminance image is box-downsampled to
//     the working resolution (image/operators/downsample.hpp) and its reduced
//     calibration derived (scaled_calibration); stereo::run_patch_match then
//     estimates a per-pixel depth (distance along the bearing) plus normals.
//   * stereo::filter_depth_map cross-checks each map against its neighbourhood
//     and turns the retained count into the per-pixel confidence.
//
// Every image is decoded once and cached for the task's lifetime, keyed by
// camera id. A camera with no neighbourhood gets no depth map.
//
// Commit. A NEW depth-map asset is appended and made active, so an earlier set
// is never overwritten; every point cloud in the chunk is dropped, because each
// was derived from maps that are no longer the active ones. Cameras left
// without a neighbourhood simply have no map. `label` names the new set.

#include "common/tasks/task.hpp"

namespace alphashape {

class BuildDepthMapsTask : public AbstractTask {
public:
    std::string name() const override { return "BuildDepthMaps"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
