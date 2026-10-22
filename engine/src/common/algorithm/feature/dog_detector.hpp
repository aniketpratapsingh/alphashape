#pragma once

// Difference-of-Gaussians scale-space keypoint detector.
//
// The classic scale-space detector: a Gaussian pyramid,
// DoG/LoG extremum localisation, and SIFT-style dominant orientation. This is
// our own implementation of the published method — Lowe, "Distinctive Image
// Features from Scale-Invariant Keypoints", IJCV 2004 — with the constants
// that paper recommends.

#include <vector>

#include "common/core/types.hpp"
#include "image/image.hpp"

namespace alphashape {

struct DogParams {
    int scales_per_octave = 3;           ///< Lowe 2004 §3.2: 3 gives the best repeatability
    float base_sigma = 1.6F;             ///< Lowe 2004 §3.3
    float assumed_blur = 0.5F;           ///< Lowe 2004 §3.3: blur already present in the input
    float contrast_threshold = 0.04F;    ///< on [0,1] intensities, divided by scales_per_octave
                                         ///< (Lowe's 0.03 for S=3, as in common implementations)
    float edge_ratio = 10.0F;            ///< Lowe 2004 §4.1: principal-curvature ratio r
    int max_octaves = 8;
    int border = 5;                      ///< pixels ignored at each octave's edge
    int refine_iterations = 5;           ///< Lowe 2004 §4: subpixel fit attempts
    int orientation_bins = 36;           ///< Lowe 2004 §5
    float orientation_peak_ratio = 0.8F; ///< Lowe 2004 §5: secondary peaks become keypoints
};

/// The Gaussian and DoG pyramids. Built once per image and shared by the
/// detector and the descriptor, which samples the Gaussian levels.
struct ScaleSpace {
    int scales_per_octave = 3;
    float base_sigma = 1.6F;
    std::vector<std::vector<FloatImage>> gaussians;  ///< [octave][S + 3]
    std::vector<std::vector<FloatImage>> dogs;       ///< [octave][S + 2]

    int octaves() const { return static_cast<int>(gaussians.size()); }
};

/// One detection, with what the descriptor needs to find its patch again.
struct Detection {
    Keypoint keypoint;        ///< in input-image pixels; scale is sigma in input pixels
    float response = 0.0F;    ///< |DoG| at the refined extremum; used for ranking
    int octave = 0;
    int layer = 0;            ///< Gaussian level nearest the refined scale
    float octave_x = 0.0F;    ///< position in the octave's pixel grid
    float octave_y = 0.0F;
    float octave_sigma = 0.0F;
};

class DogDetector {
public:
    explicit DogDetector(DogParams params = {});

    const DogParams& params() const { return params_; }

    ScaleSpace build_scale_space(const FloatImage& luminance) const;

    /// Detections, unsorted. Deterministic for a given image.
    std::vector<Detection> detect(const ScaleSpace& scale_space) const;

private:
    DogParams params_;
};

/// Separable Gaussian blur with edge clamping. Exposed for tests and reuse.
FloatImage gaussian_blur(const FloatImage& image, float sigma);

}  // namespace alphashape
