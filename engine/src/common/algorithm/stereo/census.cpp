#include "common/algorithm/stereo/census.hpp"

#include <algorithm>
#include <cstddef>

#include "base/utils/error.hpp"
#include "image/image.hpp"

namespace alphashape::stereo {
namespace {

inline std::size_t si(int v) { return static_cast<std::size_t>(v); }

/// Edge-clamped read of one pixel, as FloatImage::at_clamped does but on the raw
/// row-major array: the transform walks the image once and should not pay a
/// function call per supported sample for every pixel.
inline float clamped(const std::vector<float>& pixels, int width, int height, int x, int y) {
    const int cx = std::clamp(x, 0, width - 1);
    const int cy = std::clamp(y, 0, height - 1);
    return pixels[si(cy) * si(width) + si(cx)];
}

/// The wide census: four arms of `radius` samples each, 8 bits per axis.
///
/// Bit order chosen so that a horizontal pair is adjacent in the word and the
/// two axes occupy one byte each -- pure convenience for anyone reading the
/// value; the distance only depends on the SET of comparisons.
std::uint16_t wide_census(const std::vector<float>& pixels, int width, int height, int x, int y,
                          int radius) {
    const float centre = pixels[si(y) * si(width) + si(x)];
    std::uint16_t mask = 0;
    int bit = 0;
    // Horizontal arm: the pixels strictly left of the centre, then strictly
    // right. Sampling outward-in from the centre on the right and inward-out on
    // the left would be equally valid; a fixed order keeps the mask stable.
    for (int k = 0; k < radius; ++k) {
        const float v = clamped(pixels, width, height, x - radius + k, y);
        if (v < centre) {
            mask = static_cast<std::uint16_t>(mask | (1U << bit));
        }
        ++bit;
    }
    for (int k = 0; k < radius; ++k) {
        const float v = clamped(pixels, width, height, x + 1 + k, y);
        if (v < centre) {
            mask = static_cast<std::uint16_t>(mask | (1U << bit));
        }
        ++bit;
    }
    // Vertical arm.
    for (int k = 0; k < radius; ++k) {
        const float v = clamped(pixels, width, height, x, y - radius + k);
        if (v < centre) {
            mask = static_cast<std::uint16_t>(mask | (1U << bit));
        }
        ++bit;
    }
    for (int k = 0; k < radius; ++k) {
        const float v = clamped(pixels, width, height, x, y + 1 + k);
        if (v < centre) {
            mask = static_cast<std::uint16_t>(mask | (1U << bit));
        }
        ++bit;
    }
    return mask;
}

/// The 3x3 census: the eight immediate neighbours, packed into the low byte in
/// the same reading order as the wide form (left-out, left-in, left/below, ...).
/// Values occupy bits 0..7; the upper byte is zero, so the descriptor still
/// compares with the same Hamming rule and normaliser.
std::uint16_t narrow_census(const std::vector<float>& pixels, int width, int height, int x, int y) {
    const float centre = pixels[si(y) * si(width) + si(x)];
    static constexpr int kOffsets[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                                           {0, 1},   {1, -1}, {1, 0},  {1, 1}};
    std::uint16_t mask = 0;
    for (int k = 0; k < 8; ++k) {
        const float v =
            clamped(pixels, width, height, x + kOffsets[k][0], y + kOffsets[k][1]);
        if (v < centre) {
            mask = static_cast<std::uint16_t>(mask | (1U << k));
        }
    }
    return mask;
}

}  // namespace

std::uint16_t CensusImage::at(int x, int y) const {
    if (values.empty()) {
        return 0;
    }
    const int cx = std::clamp(x, 0, width - 1);
    const int cy = std::clamp(y, 0, height - 1);
    return values[si(cy) * si(width) + si(cx)];
}

CensusImage census_transform(const FloatImage& image, const CensusOptions& options) {
    if (options.window_radius < 1) {
        throw ValueError("stereo::census_transform: window_radius must be positive");
    }
    if (options.wide && options.window_radius > 8) {
        // 2 arms * 2 * radius comparisons must fit the 16-bit descriptor.
        throw ValueError("stereo::census_transform: a wide census needs window_radius <= 8");
    }
    if (!options.wide && options.window_radius != 1) {
        throw ValueError("stereo::census_transform: the 3x3 census requires window_radius == 1");
    }
    const int width = image.width();
    const int height = image.height();
    CensusImage out;
    out.width = width;
    out.height = height;
    if (width <= 0 || height <= 0) {
        return out;
    }
    out.values.assign(si(width) * si(height), 0);
    const std::vector<float>& pixels = image.pixels();
    const int radius = options.window_radius;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            out.values[si(y) * si(width) + si(x)] =
                options.wide ? wide_census(pixels, width, height, x, y, radius)
                             : narrow_census(pixels, width, height, x, y);
        }
    }
    return out;
}

std::vector<float> census_values(const CensusImage& census) {
    std::vector<float> out(census.values.size(), 0.0F);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<float>(census.values[i]) / 65535.0F;
    }
    return out;
}

const CensusImage& CensusCache::get(const FloatImage& image) {
    for (const Entry& entry : entries_) {
        if (entry.image == &image) {
            return entry.census;
        }
    }
    entries_.push_back(Entry{&image, census_transform(image, options_)});
    return entries_.back().census;
}
}  // namespace alphashape::stereo
