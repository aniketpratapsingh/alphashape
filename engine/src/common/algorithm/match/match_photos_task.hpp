#pragma once

// MatchPhotos: the second real pipeline stage: detect -> preselect -> match ->
// build tracks. Its parameter names mirror Metashape's public Python API so
// scripts read the same.
//
// The algorithm composes the pieces already built and reviewed:
//   * detect_keypoints_for_frame (feature/detect_features_task.hpp) — the same
//     detection DetectFeatures runs, applied only where a camera has none yet;
//   * preselection (match/preselection.hpp) — exhaustive / sequential /
//     reference / generic candidate pairs;
//   * match_descriptors (match/hamming_matcher.hpp) — 2-NN Hamming + ratio;
//   * estimate_essential_ransac (sfm/essential.hpp) — geometric verification;
//   * build_tracks / limit_tracks (match/tracks.hpp) — tie points.
//
// Matching needs no pixel geometry beyond the calibration: a keypoint becomes a
// bearing through Calibration::unproject(keypoint_to_pixel(kp)). The +0.5 in
// keypoint_to_pixel is the whole pixel-convention question, settled empirically
// in tests/test_detector.cpp.

#include "common/tasks/task.hpp"

namespace alphashape {

class MatchPhotosTask : public AbstractTask {
public:
    std::string name() const override { return "MatchPhotos"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
