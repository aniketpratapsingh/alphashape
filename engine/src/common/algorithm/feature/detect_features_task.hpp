#pragma once

// DetectFeatures: the first real pipeline stage, wired through the task layer.
//
// Detection is the first stage of MatchPhotos. We expose it as its own task in v0 so the vertical slice — registry, schema,
// work items, project round-trip — is proven on one real algorithm before
// matching is built on top of it.
//
// The per-frame "load image -> detect -> describe -> cap" body is factored out
// as detect_keypoints_for_frame() so MatchPhotos runs the identical detection
// when a camera has no keypoints yet (MatchPhotos' detect stage is this task). Both tasks share one implementation, so the two cannot diverge.

#include <cstdint>

#include <Eigen/Core>

#include "common/tasks/task.hpp"

namespace alphashape {

class Frame;

class DetectFeaturesTask : public AbstractTask {
public:
    std::string name() const override { return "DetectFeatures"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

/// Detect, describe and cap the keypoints of one frame, returning a KeypointSet
/// keyed by the frame's camera id. This is the exact body DetectFeatures runs
/// per frame, exposed so MatchPhotos can reuse it. Throws if the frame is not
/// linked to a camera, if a limit is negative, or if the image cannot be read.
///
/// keypoint_limit and keypoint_limit_per_mpx carry the DetectFeatures semantics:
/// a value <= 0 means "no limit". The caller resolves them from its own params.
KeypointSet detect_keypoints_for_frame(const Document& document, const Frame& frame,
                                       std::int64_t keypoint_limit,
                                       std::int64_t keypoint_limit_per_mpx);

/// Convert a detector keypoint to the Calibration pixel convention.
///
/// The detector emits keypoints in FloatImage::sample's convention, where the
/// centre of the top-left pixel is (0,0) and integer coordinate k is the centre
/// of pixel k (see dog_detector.cpp: keypoint.x = octave_x, and the descriptor
/// samples at octave_x through FloatImage::sample). Calibration::unproject
/// instead puts the centre of the top-left pixel at (0.5, 0.5), so its pixel
/// index k has centre k + 0.5. The two conventions therefore differ by a
/// constant +0.5 in both axes, measured empirically by the Detector test
/// "KeypointConventionMatchesSampleCoordinates" (tests/test_detector.cpp).
inline Eigen::Vector2d keypoint_to_pixel(const Keypoint& keypoint) {
    return Eigen::Vector2d(static_cast<double>(keypoint.x) + 0.5,
                           static_cast<double>(keypoint.y) + 0.5);
}

}  // namespace alphashape
