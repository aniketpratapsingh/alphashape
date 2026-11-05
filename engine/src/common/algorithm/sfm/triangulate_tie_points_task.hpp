#pragma once

// TriangulateTiePoints: tie points from matches, with the camera poses held.
//
// Intended to run after AlignFromReference, which places the cameras from their
// navigation metadata. The task rebuilds tracks from the stored matches over
// the aligned cameras only, triangulates every track with the poses fixed
// (sfm::triangulate_multiview), and replaces the chunk's tie points:
//
//   * observations whose reprojection error exceeds the threshold are dropped;
//     the point survives when at least `min_views` remain;
//   * a track below `min_views`, below the parallax floor, or with no valid
//     triangulation (behind a camera / degenerate) is rejected and counted;
//   * accepted tracks keep their observation list (minus rejected views) and
//     their triangulated point.
//
// The edit is staged and committed as one undoable command.

#include "common/tasks/task.hpp"

namespace alphashape {

class TriangulateTiePointsTask : public AbstractTask {
public:
    std::string name() const override { return "TriangulateTiePoints"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
