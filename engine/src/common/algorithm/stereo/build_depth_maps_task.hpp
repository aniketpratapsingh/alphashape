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
// cancelled, the maps finished so far are committed before CancelledError is
// rethrown. Those maps are complete per camera and carry their alignment
// fingerprint, so the next run with `reuse_depth` (on by default) skips their
// search and filters them. A cancel inside one camera takes effect within one
// PatchMatch sweep, because the sweep hook polls the cancellation flag.
//
// A cancel commits ONLY depth maps: every other product in the chunk (point
// clouds, models, DEMs, orthomosaics, earlier depth-map sets) is left exactly
// as it was. Repeated cancels do not accumulate sets: the first writes one
// marked "partial", later ones merge into it, and a run that finally completes
// replaces it in place.
//
// Reuse. A map is reused only when its downscale AND its alignment fingerprint
// match the chunk's current geometry AND NEIGHBOURHOOD: the reference camera's
// pose and the sensor calibration, the neighbour cameras' ids, poses and
// calibrations, the depth range, the reference image file and the PatchMatch
// settings. Moving one camera therefore rebuilds every map that had it as a
// neighbour, not only its own, so a re-alignment or re-calibration never reuses
// a map the change invalidated.
//
// Commit. A new depth-map asset is appended and made active (or the partial set
// this run resumed from is completed in place), so an earlier completed set is
// never overwritten. On SUCCESS every point cloud in the chunk is dropped,
// because each was derived from maps that are no longer the active ones; a
// cancellation leaves them alone. Cameras left without a neighbourhood simply
// have no map. `label` names the new set.

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
