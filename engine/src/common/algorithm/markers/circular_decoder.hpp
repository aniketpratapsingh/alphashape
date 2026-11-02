#pragma once

/// @file circular_decoder.hpp
/// @brief Decoding of circular coded targets (WP-13.6).
///
/// The pattern is documented in target_patterns.hpp. `decode_circular` takes a
/// MarkerCandidate from the WP-13.5 detector and resolves the outer-border
/// ellipse, samples the N coding sectors, and looks the ring word up in the
/// table for `bits`.
///
/// Pixel conventions match the detector: `MarkerCandidate::center` and
/// `CodedTarget::center` use the calibration convention where the centre of the
/// top-left pixel is (0.5, 0.5), while `FloatImage::sample` addresses pixel
/// centres at integer coordinates.
///
/// ## Error control
///
/// A single misread sector plus an unknown rotation must still decode. The
/// tables have minimum Hamming distance 3 between all rotations of all words,
/// so the decoder accepts only when the best match is at most `max_hamming`
/// away and the runner-up *ring code* is at least two bits farther away.
/// Because distance 3 corrects one error, `max_hamming` is clamped to 1 here
/// regardless of the caller's value; the marker-parameter `tolerance` mapped by
/// a later work package is clamped to the same bound.
///
/// The centre comes from the outer-border ellipse. When the target has a black
/// centre dot (parity 1) the refinement runs from that dot, which is the
/// precise fiducial and is preferred; a target without a dot (parity 0) keeps
/// the border-ellipse centre. The ellipse-centre bias of a strongly tilted
/// target is a known limitation and is not corrected (the centre is the ellipse
/// centre, not a perspective-corrected projection of the target centre).
///
/// Ring validation uses the wide inner white gap (0.3..0.45 R) as the light
/// reference and the dark outer border as the dark reference. The design's
/// outer white gap is only 0.05 R wide, which any realistic blur fills in, so
/// it is not used as an accept/reject reference; this keeps decoding working
/// under the blur and tilt required by the WP-13.6 tests.

#include <optional>

#include <Eigen/Core>

#include "common/algorithm/markers/marker_detector.hpp"
#include "image/image.hpp"

namespace alphashape {

struct CodedTarget {
    int code = -1;                        ///< 2 * ring table index + centre parity.
    Eigen::Vector2d center = Eigen::Vector2d::Zero();  ///< Calibration convention.
    int bits = 0;                         ///< Ring width N.
    int hamming = 0;                      ///< Distance to the accepted ring code.
    double radius_px = 0.0;               ///< Outer-border semi-major radius.
};

/// Decode one candidate. The candidate may be the centre dot or the outer
/// border; the decoder locates the outer-border ellipse from the candidate's
/// centre. Targets below 6 px outer radius for N >= 16, or 4 px otherwise, are
/// rejected. Returns nullopt when the candidate is not a valid coded target.
std::optional<CodedTarget> decode_circular(const FloatImage& gray,
                                           const MarkerCandidate& candidate, int bits,
                                           int max_hamming);

}  // namespace alphashape
