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
// Cancellation. A finished camera's map is staged immediately; when the task is
// cancelled, the maps finished so far are committed as a new (unfiltered)
// depth-map set before CancelledError is rethrown. Those maps are complete per
// camera and carry their alignment fingerprint, so the next run with
// `reuse_depth` (on by default) skips their search and filters them. A cancel
// inside one camera takes effect within one PatchMatch sweep, because the sweep
// hook polls the cancellation flag.
//
// Reuse. A map is reused only when its downscale AND its alignment fingerprint
// match the chunk's current geometry (camera pose + sensor calibration +
// downscale), so re-aligning cameras or re-calibrating a sensor rebuilds the
// stale maps instead of consuming them.
//
// Commit. A NEW depth-map asset is appended and made active, so an earlier set
// is never overwritten; every point cloud in the chunk is dropped, because each
// was derived from maps that are no longer the active ones. Cameras left
// without a neighbourhood simply have no map. `label` names the new set.

#include <cstdint>

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

/// Test hook: how many PatchMatch searches BuildDepthMaps has STARTED in this
/// process. A reused map is not counted, so the difference across two runs is
/// exactly the number of cameras whose search was repeated — the timing-free way
/// to assert that reuse works.
std::uint64_t depth_map_search_count();

/// Test hook: reset depth_map_search_count() to 0.
void reset_depth_map_search_count();

}  // namespace alphashape
