#pragma once

// Census transform (DP-03).
//
// The census transform of Zabih & Woodfill (1994), "Non-parametric local
// transforms for computing visual correspondence": for every pixel, record the
// sign of the intensity difference between each supported neighbour and the
// centre, packed into a bit mask. The descriptor is invariant to any strictly
// monotone change of intensity (a gain, a bias, a gamma), which is what makes it
// tolerate exposure and white-balance differences between overlapping images and
// stay meaningful on low-contrast ground where a ZNCC window is mostly noise.
//
// This is the "wide census" of Mei et al. (2011), "On Building an Accurate
// Stereo Matching System on Graphics Hardware": the 16 bits are the horizontal
// arm and the vertical arm of the window compared against the centre, 8 samples
// each at unit spacing. Two arms rather than the full (2r+1)^2 grid is what fits
// a 32-bit register while keeping a support wide enough to see structure at the
// 9x9 scale the search uses.
//
// Bit layout, for radius r = 4 and the centre at (x, y) (unit spacing inside the
// census itself; PatchMatch's own window subsampling is unrelated):
//
//   bit 0..3   I(x - 4 + k, y) < I(x, y)   for k = 0..3   (left arm, outward-in)
//   bit 4..7   I(x + 1 + k, y) < I(x, y)   for k = 0..3   (right arm, inside-out)
//   bit 8..11  I(x, y - 4 + k) < I(x, y)   for k = 0..3   (top arm)
//   bit 12..15 I(x, y + 1 + k) < I(x, y)   for k = 0..3   (bottom arm)
//
// A tie (equal intensities) sets the bit to 0: "strictly darker" sets it, so a
// flat region is all zeros rather than all ones and a saturated flat region
// cannot masquerade as a strongly textured one.
//
// The comparison is the census *distance* of the published cost (Mei et al.
// 2011; Hirschmuller & Scharstein 2009, "Evaluation of Stereo Matching Costs on
// New Radiometric Calibration Data"): the Hamming distance of the two bit masks,
// normalised by the number of bits, so every census cost lives in [0, 1] and 0.5
// is exactly chance-level agreement (half the bits differ). The [0, 1] scale is
// the same interval the intensity cost uses (1 - ZNCC) / 2, so one `max_cost`
// threshold keeps its meaning ("the match must be better than chance") whichever
// cost a run selects.
//
// Why Hamming rather than a correlation of the packed bytes: the numeric value
// of a packed bit mask is not a metric on the pattern -- flipping the top bit of
// 0x7FFF moves the value by 0.5 while flipping the bottom bit moves it by
// 1/65535, though both are one disagreeing comparison. Hamming weights every
// comparison equally, which is the property the census's noise tolerance rests
// on (and is the distance both cited papers evaluate).

#include <cstdint>
#include <deque>
#include <vector>

#include "image/image.hpp"

namespace alphashape::stereo {

/// Census transform parameters.
struct CensusOptions {
    /// Support radius, in pixels. The wide census uses `2 * window_radius`
    /// comparisons (8 horizontal + 8 vertical at radius 4), so 16 bits are exact
    /// for radius 4 and the radius is capped at 8 by the 16-bit descriptor.
    int window_radius = 4;
    /// True: the 16-bit two-arm wide census described above. False: a 3x3
    /// census (the eight immediate neighbours, packed in the low byte), kept for
    /// a cheap descriptor and for experiments; the search's own cost type uses
    /// the wide form.
    bool wide = true;
    /// Census the neighbour views as well as the reference. The search needs
    /// both; the flag exists so a caller computing a reference-only descriptor
    /// (a preview, a mask) can skip the neighbours it does not have.
    bool transform_neighbours = true;
};

/// One view's census descriptor field, row-major, one 16-bit mask per pixel.
struct CensusImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint16_t> values;

    /// Descriptor at (x, y), clamping to the edge pixel outside the image. The
    /// transform already stored edge-clamped values on every pixel, so the
    /// accessor is for callers sampling at continuous coordinates and never
    /// needs to repeat the clamping themselves.
    std::uint16_t at(int x, int y) const;
};

/// Transforms one image. Every pixel gets a descriptor; a sample that falls
/// outside the image is read from the nearest edge pixel, so the border pixels
/// are defined rather than left as a hole (a depth map's interior is what the
/// search uses, but a border descriptor is still read when a window reaches it).
CensusImage census_transform(const FloatImage& image, const CensusOptions& options = {});

/// Normalised Hamming distance of two descriptors, in [0, 1]: the number of
/// differing bits over the 16 the wide census fills.
inline double census_distance(std::uint16_t a, std::uint16_t b) {
    // Portable popcount: the width is fixed at 16, so the loop is unrolled by
    // any optimiser and no <bit> / builtin dependency is introduced.
    std::uint32_t v = static_cast<std::uint32_t>(a ^ b);
    int bits = 0;
    while (v != 0) {
        bits += static_cast<int>(v & 1U);
        v >>= 1U;
    }
    return static_cast<double>(bits) / 16.0;
}

/// Per-view descriptor cache. A run transforms each view once and reuses it for
/// every candidate plane, so the O(pixels * support) transform is paid once per
/// view per pyramid level rather than once per cost evaluation. Keyed by the
/// image's address: the caller owns the images and keeps them alive for the
/// cache's lifetime (the search builds one cache per level and drops it with the
/// level's reduced images). References returned by get() stay valid for the
/// cache's lifetime, including across later get() calls for other images.
class CensusCache {
public:
    explicit CensusCache(CensusOptions options = {}) : options_(options) {}

    /// The descriptor field for `image`, computed on first request.
    const CensusImage& get(const FloatImage& image);

    /// Number of views transformed so far (diagnostics and tests).
    std::size_t size() const { return entries_.size(); }

    const CensusOptions& options() const { return options_; }

private:
    struct Entry {
        const FloatImage* image = nullptr;
        CensusImage census;
    };

    CensusOptions options_;
    /// A deque so that push_back never moves an already-returned descriptor.
    std::deque<Entry> entries_;
};

}  // namespace alphashape::stereo
