// fill_holes_detail.hpp -- the smooth RGBA hole fill that BuildOrthomosaic runs
// on the assembled raster, exposed so a unit test can drive it directly.

#ifndef ALPHASHAPE_COMMON_ALGORITHM_ORTHO_FILL_HOLES_DETAIL_HPP
#define ALPHASHAPE_COMMON_ALGORITHM_ORTHO_FILL_HOLES_DETAIL_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "base/utils/error.hpp"
#include "common/algorithm/elevation/interpolate.hpp"

namespace alphashape {
namespace ortho {
namespace detail {

/// Complete every invalid (alpha == 0) cell that `options` selects, per colour
/// channel, then make the completed cells opaque.
///
/// The validity mask is snapshotted ONCE before any channel runs. Rebuilding it
/// from alpha per channel (as this used to) reads the alpha the previous channel
/// itself set, so after channel 0 every filled cell looked valid and its G and B
/// were never written: filled holes came out red.
inline void fill_smooth_rgba_holes(int width, int height, std::vector<std::uint8_t>& rgba,
                                   const elevation::HoleFillOptions& options) {
    if (width <= 0 || height <= 0) {
        throw ValueError("fill_smooth_rgba_holes: width and height must be positive");
    }
    const std::size_t cells =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (rgba.size() != cells * 4U) {
        throw ValueError("fill_smooth_rgba_holes: buffer size does not match width*height");
    }
    std::vector<std::uint8_t> was_valid(cells, 0);
    for (std::size_t i = 0; i < cells; ++i) {
        was_valid[i] = rgba[i * 4U + 3U] != 0 ? 1 : 0;
    }
    std::vector<float> values(cells, 0.0F);
    std::vector<std::uint8_t> channel_valid(cells, 0);
    for (int channel = 0; channel < 3; ++channel) {
        const std::size_t offset = static_cast<std::size_t>(channel);
        for (std::size_t i = 0; i < cells; ++i) {
            values[i] = static_cast<float>(rgba[i * 4U + offset]);
        }
        // fill_holes() sets valid[i] = 1 for the cells it filled; give it a
        // fresh copy so channel 0's result cannot leak into channel 1.
        channel_valid = was_valid;
        elevation::fill_holes(width, height, values, channel_valid, options);
        for (std::size_t i = 0; i < cells; ++i) {
            if (channel_valid[i] != 0) {
                const float c = std::clamp(values[i], 0.0F, 255.0F);
                rgba[i * 4U + offset] = static_cast<std::uint8_t>(std::lround(c));
            }
        }
    }
    // Only a cell the fill actually completed becomes opaque; a hole outside
    // `options` (exterior, or beyond max_distance) stays transparent.
    for (std::size_t i = 0; i < cells; ++i) {
        if (was_valid[i] == 0 && channel_valid[i] != 0) {
            rgba[i * 4U + 3U] = 255;
        }
    }
}

}  // namespace detail
}  // namespace ortho
}  // namespace alphashape

#endif  // ALPHASHAPE_COMMON_ALGORITHM_ORTHO_FILL_HOLES_DETAIL_HPP
