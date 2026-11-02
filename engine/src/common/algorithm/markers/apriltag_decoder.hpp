#pragma once

/// @file apriltag_decoder.hpp
/// @brief Decoding of the published AprilTag families (WP-13.7).
///
/// Follows Olson 2011, "AprilTag: A robust and flexible visual fiducial
/// system", and Wang & Olson 2016, "AprilTag 2: Efficient and robust fiducial
/// detection". The tables and bit layout come from the published
/// BSD-2-Clause AprilRobotics/apriltag sources (see tag_families.hpp); the
/// decoder itself is original.
///
/// A quad from `detect_quads` is refined to subpixel corners by fitting a
/// weighted line to the gradient-magnitude ridge along each side and
/// intersecting adjacent lines. A normalised direct linear transform then maps
/// the tag frame to the image, and each payload cell is sampled at its
/// published `bit_x/bit_y` centre. The threshold is a bilinear model fitted to
/// the sampled border ring (the inner reference frame) and to the quiet zone
/// one cell outside it, so illumination gradients across the tag do not flip
/// cells. The four 90-degree rotations are tried on the sampled word, and the
/// corner order is reported so that corner 0 is the published corner.
///
/// ## Pixel convention
///
/// `MarkerCandidate::corners` and `CodedTarget::center` use the calibration
/// convention: the centre of the top-left pixel is (0.5, 0.5). `FloatImage`
/// addresses pixel centres at integer coordinates, so sampling subtracts half
/// a pixel. Corners are returned counter-clockwise (negative shoelace area in
/// image coordinates, where Y points down), starting at the published corner
/// 0, which is the image of tag-frame `(-1, +1)` (the bottom-left corner of an
/// upright tag). This is the same ordering as `detect_quads`, rotated to start
/// at the published corner.
///
/// ## Reversed-border families
///
/// `Circle21h7`, `Standard41h12` and `Standard52h13` have a white reference
/// frame instead of a black outer border, and their payload extends beyond
/// that frame. The quad that reaches the decoder is the outer silhouette for
/// these families (the dark region's outer boundary), so the frame is the
/// central `width_at_border / total_width` square of the candidate. The
/// decoder tries both that mapping and the direct one, so a caller that
/// passes the reference square still works. `Circle21h7` is table-only: its
/// published tags are circular and `detect_quads` cannot produce a quad for
/// them (see LIMITATIONS.md).
///
/// ## Error control
///
/// A word is accepted when its Hamming distance to the nearest code, over the
/// four rotations, is at most `min(max_hamming, (min_hamming-1)/2)`. The
/// tables keep all rotated words at least `min_hamming` apart, so the balls
/// around codes do not overlap within that radius and the nearest code is
/// unique; no runner-up margin is needed. `default_max_hamming()` is the
/// largest radius whose random-word acceptance stays under 1e-3:
///
/// | family        | bits | codes  | rotated | d=0      | d=1      | d=2      | default |
/// |---------------|------|--------|---------|----------|----------|----------|---------|
/// | 16h5          | 16   | 30     | 120     | 1.83e-3  | 3.11e-2  | 2.51e-1  | 0       |
/// | 25h9          | 25   | 35     | 140     | 4.17e-6  | 1.09e-4  | 1.36e-3  | 1       |
/// | 36h10         | 36   | 2320   | 9280    | 1.35e-7  | 5.00e-6  | 9.01e-5  | 2       |
/// | 36h11         | 36   | 587    | 2348    | 3.42e-8  | 1.26e-6  | 2.28e-5  | 3       |
/// | Circle21h7    | 21   | 38     | 152     | 7.25e-5  | 1.60e-3  | 1.68e-2  | 0       |
/// | Standard41h12 | 41   | 2115   | 8460    | 3.85e-9  | 1.62e-7  | 3.32e-6  | 4       |
/// | Standard52h13 | 52   | 48714  | 194856  | 4.33e-11 | 2.29e-9  | 5.97e-8  | 5       |
///
/// The values are exact counts: distinct rotated codes times the size of the
/// Hamming ball, divided by 2^bits. Balls of radius at most `(h-1)/2` cannot
/// overlap because the code distance is at least `h = 2d+1`. For the families
/// with a larger correction budget the chosen radius is the edge of 1e-3:
/// 36h11 d=3 is 2.67e-4 (d=4 is 2.28e-3), Standard41h12 d=4 is 4.34e-4 (d=5 is
/// 3.32e-3), Standard52h13 d=5 is 1.25e-4 (d=6 is 1.01e-3). 16h5 cannot reach
/// 1e-3 even at d=0 (120 of 65536 words are a rotation of a code), so its
/// default is 0 as the least permissive value. A Monte Carlo run of 10^6
/// random words reproduces these fractions.

#include <cstdint>
#include <optional>
#include <vector>

#include "common/algorithm/markers/circular_decoder.hpp"
#include "common/algorithm/markers/marker_detector.hpp"
#include "common/algorithm/markers/tag_families.hpp"
#include "image/image.hpp"

namespace alphashape {

/// Result of matching one sampled word in layout bit order (bit 0 is the
/// first published layout entry, i.e. the most significant code bit).
struct TagMatch {
    int id = -1;        ///< Index into `family.codes`.
    int distance = 0;   ///< Hamming distance to that code.
    int rotation = 0;   ///< 90-degree steps needed to undo the tag's rotation.
};

/// Match `word` against every code and all four rotations. Returns nullopt for
/// an empty family, a negative `max_hamming`, or when the best distance is
/// larger than `min(max_hamming, (min_hamming-1)/2)`. Exposed so the distance
/// rules can be tested without rendering a tag.
std::optional<TagMatch> match_tag_word(std::uint64_t word,
                                       const TagFamily& family,
                                       int max_hamming);

/// Decode one square candidate. Returns nullopt when the candidate is not a
/// structurally valid tag of `family`, when a payload cell is too close to the
/// local threshold, or when no rotation matches within the allowed distance.
std::optional<CodedTarget> decode_tag(const FloatImage& gray,
                                      const MarkerCandidate& quad,
                                      const TagFamily& family,
                                      int max_hamming = 1);

/// Detect and decode every tag of `family` in `gray`. This is the convenience
/// path a task uses: it runs `detect_quads` on the image, and for a
/// reversed-border family also on the inverted image (where the white
/// reference frame is a solid dark ring), then decodes each candidate. If
/// nothing decodes, a reversed-border family retries on a bilinear upsampling:
/// 2x when the shorter image side is under 200 px, and 3x when it is under
/// 128 px. At the doubled or tripled scale the detector's subpixel line fits
/// are accurate for a small printed tag; the reported centre, corners and
/// radius are mapped back to the original image.
std::vector<CodedTarget> decode_tags(const FloatImage& gray,
                                     AprilTagFamily family,
                                     int max_hamming = 1,
                                     const QuadParams& params = {});

}  // namespace alphashape
