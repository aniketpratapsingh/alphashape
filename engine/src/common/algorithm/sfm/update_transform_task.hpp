#pragma once

// UpdateTransform: fit ONE 7-parameter similarity from the chunk's current local
// frame to the reference frame and apply it uniformly (WP-13.12).
//
// This differs from OptimizeCameras: no bundle adjustment, no change to the
// internal shape of the block. It uses the control markers and camera GNSS
// references as weighted correspondences, robustly rejects blunders, holds check
// points out for reporting, and rewrites poses, tie points, the region and every
// marker position through the same similarity. The chunk's transform and CRS
// slots are set exactly as georeference_chunk does.

#include "common/tasks/task.hpp"

namespace alphashape {

class UpdateTransformTask : public AbstractTask {
public:
    std::string name() const override { return "UpdateTransform"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
