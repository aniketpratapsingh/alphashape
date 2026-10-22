#pragma once

// Per-camera pixel masks (WP-11.1, M11 §4.6). A mask marks pixels that must be
// ignored by feature detection and dense matching — sky, moving objects,
// water, lens obstructions. Stored as a single-channel u8 raster of the same
// resolution as the camera image.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace alphashape {

/// A mask for one camera. `camera_id` names a camera in `Chunk::cameras()`.
/// When `invert` is false a non-zero mask byte means "use this pixel"; when
/// true the meaning flips (the mask marks the usable region instead).
struct Mask {
    int camera_id = -1;
    int width = 0;
    int height = 0;
    bool invert = false;
    std::vector<std::uint8_t> data;

    /// True when the dimensions are positive and `data` holds exactly
    /// width * height bytes.
    bool is_valid() const {
        if (width <= 0 || height <= 0) {
            return false;
        }
        const std::size_t expected =
            static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        return data.size() == expected;
    }

    bool operator==(const Mask&) const = default;
};

}  // namespace alphashape
