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
/// A ring word is accepted when its Hamming distance to the nearest rotation of
/// a table word is at most `max_hamming`, with `max_hamming` clamped to 1. The
/// tables keep every rotation of every word at least three bits apart, so the
/// nearest code is unique: a word within one bit of a code is at least two bits
/// from every other code, and no runner-up margin is needed.
///
/// The default is an exact match (`max_hamming = 0`). One-sector correction is
/// opt-in because it is expensive in false positives: a uniformly random N-bit
/// word is within one bit of some rotation of some code for 45.7% of the words
/// at N=12, 49.0% at N=14, 48.8% at N=16 and 45.9% at N=20. An exact match
/// accepts only 3.81%, 3.50%, 3.05% and 2.29% respectively (the fraction of
/// words that are themselves a rotation of a table word). A caller that enables
/// correction therefore needs a clean sector read.
///
/// Ring validation also rejects structurally invalid candidates: sectors that
/// sit within 20% of the light-minus-dark contrast of the threshold, a ring
/// with only one intensity class, and annuli whose eight angular bins are not
/// uniformly light (inner gap) or dark (border).
///
/// The centre comes from the outer-border ellipse. When the target has a black
/// centre dot (parity 1) the refinement runs from that dot, which is the
/// precise fiducial and is preferred within 0.35 R. A target without a dot
/// (parity 0) keeps the border-ellipse centre unless the refinement moves it by
/// at most 0.10 R: its first radial edges mix dark sectors at 0.5 R with light
/// sectors at 0.85 R. The ellipse-centre bias of a strongly tilted target is a
/// known limitation and is not corrected (the centre is the ellipse centre, not
/// a perspective-corrected projection).
///
/// Ring validation uses the wide inner white gap (0.3..0.45 R) as the light
/// reference and the dark outer border as the dark reference. The design's
/// outer white gap is only 0.05 R wide, which any realistic blur fills in, so
/// it is not used as an accept/reject reference; this keeps decoding working
/// under the blur and tilt required by the WP-13.6 tests.

#include <cstdint>
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

/// Result of looking a ring word up in the table.
struct WordMatch {
    int index = -1;    ///< Table index, so code = 2 * index + parity.
    int distance = 0;  ///< Hamming distance to the nearest rotated table word.
};

/// Look up `word` in the table for `bits`. Returns nullopt for an unsupported
/// width, a negative `max_hamming`, or when the nearest rotation is farther
/// than `min(max_hamming, 1)`. Exposed so the distance rules can be tested
/// without rendering a target.
std::optional<WordMatch> match_circular_word(std::uint32_t word, int bits,
                                             int max_hamming);

/// Decode one candidate. The candidate may be the centre dot or the outer
/// border; the decoder locates the outer-border ellipse from the candidate's
/// centre. Targets below 6 px outer radius for N >= 16, or 4 px otherwise, are
/// rejected. Returns nullopt when the candidate is not a valid coded target.
/// `max_hamming = 0` (the default) requires an exact ring-word match; 1 enables
/// one-sector correction (larger values are clamped to 1).
std::optional<CodedTarget> decode_circular(const FloatImage& gray,
                                           const MarkerCandidate& candidate, int bits,
                                           int max_hamming = 0);

}  // namespace alphashape
