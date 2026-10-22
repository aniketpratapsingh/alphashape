#pragma once

// BuildPointCloud: fuse the chunk's depth maps into one dense point cloud.
//
// This is the adapter between the chunk's data model and stereo::fuse_depth_maps
// (fusion.hpp). Every depth map whose camera still exists and is enabled is
// back-projected into chunk-local space and merged (voxel-decimated) into a
// single cloud. Full-resolution images are read only when point_colors is set,
// and are used to colour each fused point.
//
// Commit. A NEW point-cloud asset is appended and made active, so an earlier
// cloud (for example one fused from a previous depth-map set) is never
// overwritten. `label` names the new asset.

#include "common/tasks/task.hpp"

namespace alphashape {

class BuildPointCloudTask : public AbstractTask {
public:
    std::string name() const override { return "BuildPointCloud"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
