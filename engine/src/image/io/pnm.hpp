#pragma once

// Binary PGM (P5) and PPM (P6), 8-bit. Enough to load fixtures and real
// photos converted with any tool, without pulling a codec into v0.

#include <string>

#include "image/image.hpp"

namespace alphashape {

/// Throws ProjectIOError on a missing or malformed file, or maxval > 255.
Image read_pnm(const std::string& path);

/// Writes P5 for 1 channel, P6 for 3.
void write_pnm(const Image& image, const std::string& path);

}  // namespace alphashape
