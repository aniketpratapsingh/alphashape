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
#include <vector>

#include <Eigen/Core>

#include "common/tasks/task.hpp"

namespace alphashape {

class Chunk;
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
///
/// A film camera is detected on its film-normalised image (WP-13.18b). When
/// `fiducial_exclusion_mm` is positive, keypoints within that distance of a
/// layout fiducial or of the normalised image border are dropped: fiducial
/// marks sit at identical positions in every frame and would otherwise match
/// with zero parallax and poison the adjustment.
KeypointSet detect_keypoints_for_frame(const Document& document, const Chunk& chunk,
                                       const Frame& frame,
                                       std::int64_t keypoint_limit,
                                       std::int64_t keypoint_limit_per_mpx,
                                       double fiducial_exclusion_mm = 0.0);

/// Peak bytes one image's detection holds per image pixel, from the buffers
/// detect_keypoints_for_frame keeps alive at once: the float luminance (4), the
/// blurred base and one blur scratch (4 + 4), and the scale space's six
/// Gaussian and five DoG float levels per octave (11 x 4 = 44 at full size,
/// x 4/3 over the halving octaves = 58.7). 70.7, rounded up to 72. The decoded
/// 8-bit RGB is released before the scale space is built.
inline constexpr double kDetectionBytesPerPixel = 72.0;

/// The share of the task memory budget (platform::memory_budget_bytes) that
/// concurrent detections may take together. The rest is the keypoint sets,
/// the document and the matching that follows.
inline constexpr double kDetectionBudgetShare = 0.5;

/// How many images detection may work on at once: `threads`, unless that many
/// working sets of `image_pixels` each would not fit kDetectionBudgetShare of
/// `budget_bytes`. 0 for `budget_bytes` or `image_pixels` means unknown (no
/// cap). Always at least 1, so an image larger than the budget still runs.
int detection_concurrency(int threads, double image_pixels, std::uint64_t budget_bytes);

/// detection_concurrency for detecting `frames` on this machine: the OpenMP
/// thread count, the largest frame, and platform::memory_budget_bytes().
int detection_concurrency_for(const std::vector<const Frame*>& frames);

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
