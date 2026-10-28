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
// Two ways to compare a pair of descriptors live here. `census_distance` is the
// Hamming distance of two masks (the distance the cited papers evaluate, and the
// cheapest). The search itself uses `census_values`: it reads the mask as a
// sample in [0, 1] and lets the existing ZNCC aggregate over the window, which
// is the census-and-ZNCC cost of the plan's algorithm section (the resulting cost
// is (1 - ZNCC), the same scale as the intensity cost). The distinction
// that mattered when choosing: Hamming treats each comparison equally (right for
// "how many ranks disagree") but is 16 discrete levels with no sub-pixel
// interpolation possible on a bit mask, while correlating the mask values keeps
// the ZNCC's bilinear sampling and variance floor, and the descriptor has already
// made the values invariant to any monotone intensity change, so the invariance
// survives the correlation. `census_distance` stays as the descriptor-level
// primitive (tests, future use); the search does not call it.

#include <cstdint>
#include <deque>
#include <vector>

#include "image/image.hpp"

namespace alphashape::stereo {

/// Largest support radius the 16-bit wide descriptor can hold: its four arms
/// make 4 * radius comparisons, so radius 4 fills the word exactly and anything
/// larger would silently drop comparisons.
constexpr int kCensusMaxRadius = 4;

/// Census transform parameters.
struct CensusOptions {
    /// Support radius, in pixels. The wide census makes `4 * window_radius`
    /// comparisons (four arms of `window_radius` samples each: left, right, top,
    /// bottom), so 16 bits are exact for radius 4 and the radius is capped at
    /// kCensusMaxRadius (4) by the 16-bit descriptor. A larger radius would shift
    /// comparisons out of the word and silently drop them.
    int window_radius = 4;
    /// True: the 16-bit four-arm wide census described above. False: a 3x3
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
///
/// The normaliser is 16 even for the 3x3 (`wide = false`) descriptor, whose
/// eight bits only reach 0.5: that keeps one scale across both forms and is
/// harmless because the search always uses the wide form (the narrow one is a
/// cheap descriptor for previews and experiments, see CensusOptions::wide).
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

/// The descriptor field as intensities in [0, 1] (the mask over 65535).
///
/// The search correlates these values rather than the raw masks: the census
/// descriptor is a bit mask, but correlating its numeric value as a sample in
/// [0, 1] is what lets the existing ZNCC machinery (and its bilinear sampling,
/// i.e. sub-pixel accuracy) work on census data unchanged. The rank bits, not
/// the mask's numeric magnitude, are what survive an intensity change, so the
/// quotient preserves the invariance (a monotone map leaves every bit fixed,
/// hence every value fixed). This is the census-and-ZNCC cost the plan's
/// algorithm section describes. Normalising to [0, 1] keeps it on the same
/// scale as the intensity cost.
std::vector<float> census_values(const CensusImage& census);

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
