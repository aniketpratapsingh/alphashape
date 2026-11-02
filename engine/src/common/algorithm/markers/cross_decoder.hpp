#pragma once

/// @file cross_decoder.hpp
/// @brief Cross-target detection (WP-13.8).
///
/// Two printed styles are supported, and both are searched for in one pass:
///
///   * **Checker cross** (also called a bowtie or four-quadrant GCP target):
///     two black and two white quadrants meeting at a saddle point, with the
///     equal colours diagonally opposite. This is the main mode. A saddle is
///     a pixel whose Hessian determinant is negative; the response used is
///     `-det(H)`. The saddle is refined by a weighted least-squares quadratic
///     fit over a small window and the fitted critical point (the zero of the
///     fitted gradient) is taken as the centre (Lucchese & Mancera 2002,
///     "A robust algorithm for detecting and locating image features").
///
///   * **Bar cross**: two orthogonal bars of one polarity on a plain
///     background. Ridge pixels are found from the Hessian (Steger 1998,
///     "An unbiased detector of curvilinear structures"): the larger
///     eigenvalue must dominate the smaller one and the gradient across the
///     ridge must vanish. Ridge points are grouped spatially, split into two
///     orientation clusters, fitted with total least squares, and the two
///     near-orthogonal lines are intersected.
///
/// ## Rejected look-alikes
///
/// Four patterns can resemble a cross without being one:
///
///   * **AprilTag cell corners** and **dense checkerboards** produce many
///     adjacent saddles. A saddle is kept only when no other saddle lies
///     within `3 * max(radius)`, so a lattice rejects itself while an isolated
///     printed cross, whose neighbours are far away, survives.
///   * **Square outlines, bordered squares and window rectangles** turn their
///     corners into two perpendicular bar arms. Requiring the intersection to
///     lie between 20 % and 80 % of both arms rejects an endpoint (a corner).
///   * **Checkerboard crossings** can still form two perpendicular ridge
///     lines. A bar cross additionally requires the four diagonal quadrants to
///     be a uniform background; a checkerboard's quadrants alternate and fail
///     that test.
///
/// The image is expected in [0, 1]. `inverted` selects light targets on a dark
/// background; the detector inverts the image internally, so callers never
/// have to do it. `detect_crosses` returns `CodedTarget` values with
/// `code = -1` (crosses are non-coded); `center` uses the calibration
/// convention of `MarkerCandidate` and `CodedTarget`, where the centre of the
/// top-left pixel is (0.5, 0.5), while `FloatImage::sample` addresses pixel
/// centres at integer coordinates.

#include <optional>
#include <vector>

#include <Eigen/Core>

#include "common/algorithm/markers/circular_decoder.hpp"
#include "common/algorithm/markers/marker_detector.hpp"
#include "image/image.hpp"

namespace alphashape {

struct CrossParams {
    /// Light target on a dark background (the image is inverted internally).
    bool inverted = false;
    /// Minimum target half-size in pixels. 0 applies the internal floor of
    /// 6 px, which keeps small fiducial cell corners out of the result.
    double minimum_size = 0.0;
    /// Non-maximum-suppression distance between detections, in pixels.
    double minimum_dist = 5.0;
    /// Absolute saddle-response floor; 0 selects the adaptive threshold.
    double min_response = 0.0;
};

/// How a cross was found, which decides its full-resolution refinement.
enum class CrossStyle {
    Saddle,  ///< Checker/bowtie cross, refined by a quadratic saddle fit.
    Bar,     ///< Two fitted ridge lines, refined by re-fitting them locally.
};

struct CrossDetection {
    CodedTarget target;
    CrossStyle style = CrossStyle::Saddle;
};

/// Detect every cross target in `gray`, tagging each with its style. Results
/// are sorted by centre y then x.
std::vector<CrossDetection> detect_crosses_styled(
    const FloatImage& gray, const CrossParams& params = {});

/// Same detections without the style tags.
std::vector<CodedTarget> detect_crosses(const FloatImage& gray,
                                        const CrossParams& params = {});

/// Re-refine one saddle at `approximate` on the full-resolution image. Used by
/// the DetectMarkers task after detecting on a downscaled copy. `radius_hint`
/// is the target half-size in full-resolution pixels. Returns nullopt when the
/// local quadratic fit is not a saddle.
std::optional<Eigen::Vector2d> refine_cross_center(const FloatImage& gray,
                                                    const Eigen::Vector2d& approximate,
                                                    double radius_hint);

/// Re-refine one bar cross at `approximate`. The local window is blurred,
/// Steger ridge points are re-found near each arm, the two lines are fitted
/// and intersected. Returns nullopt when either line cannot be recovered.
std::optional<Eigen::Vector2d> refine_bar_cross_center(
    const FloatImage& gray, const Eigen::Vector2d& approximate,
    double radius_hint);

}  // namespace alphashape
