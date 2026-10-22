#pragma once

// Image file IO: format detection, header probing and full decode.
//
// This is the single entry point every other module uses to turn a file on
// disk into an Image. Codec libraries (libjpeg-turbo, libpng, libtiff) are an
// implementation detail: their headers are linked PRIVATE to alphashape_image
// and never reach a consumer of this header.
//
// ---------------------------------------------------------------------------
// Decode contract
// ---------------------------------------------------------------------------
// The decoded Image always has 1 (greyscale) or 3 (RGB) interleaved 8-bit
// channels. Specifically:
//
//   * Grey stays 1 channel, RGB stays 3.
//   * Alpha is dropped: grey+alpha becomes 1 channel, RGBA becomes 3.
//   * Palette images (PNG colour type 3, TIFF photometric 3) are expanded to RGB.
//   * 16-bit samples are reduced to 8-bit by *rounding*,
//     (v * 255 + 32767) / 65535, never by truncation.
//   * CMYK JPEG/TIFF, YCbCr TIFF, planar-separate TIFF, bilevel 1-bit and
//     other exotic layouts throw an Error. They are deliberately not
//     half-supported.
//   * EXIF orientation is NOT applied. Pixels are returned exactly as stored
//     in the file; the orientation tag is reported by the EXIF reader and it
//     is the caller's decision whether and how to apply it.
//
// All I/O and decode failures throw alphashape::Error (a ProjectIOError
// subclass). No codec is ever allowed to print to stderr or call exit().

#include <cstdint>
#include <string>
#include <vector>

#include "image/image.hpp"

namespace alphashape {

enum class ImageFormat { Unknown, Jpeg, Png, Tiff, Pnm };

/// What the file's header says. `channels` and `bit_depth` describe the file as
/// stored, not the decoded Image.
struct ImageInfo {
    ImageFormat format = ImageFormat::Unknown;
    int width = 0;
    int height = 0;
    /// Stored samples per pixel, with alpha counted only where the file stores
    /// a full RGB + alpha tuple: grey, grey+alpha and palette all report 1,
    /// RGB reports 3 and RGBA reports 4.
    int channels = 0;
    /// Bits per sample as stored: 8 or 16.
    int bit_depth = 0;
};

/// Sniffs the file's magic bytes. The extension is ignored — a PNG named
/// "photo.jpg" is reported as Png. Returns Unknown for a file that cannot be
/// opened or whose signature is not recognised; detection never throws.
ImageFormat detect_image_format(const std::string& path);

/// Reads the header only: no pixel data is decoded, so this is cheap. Throws
/// Error if the file cannot be opened or its header cannot be understood.
ImageInfo read_image_info(const std::string& path);

/// Full decode, honouring the contract documented at the top of this file.
Image read_image(const std::string& path);

/// ---------------------------------------------------------------------------
/// Encode contract
/// ---------------------------------------------------------------------------
/// Encoding writes 8-bit PNG only, because every Image the pipeline produces is
/// 8-bit. The channel count selects the PNG colour type: 1 -> greyscale,
/// 3 -> RGB, 4 -> RGBA. Any other count throws Error. The output is a complete,
/// self-contained PNG file with no ancillary chunks beyond what libpng emits by
/// default, so encode-then-decode is byte-exact for the pixel data.
/// ---------------------------------------------------------------------------

/// Encode `image` as a PNG byte stream. Throws Error on an unsupported channel
/// count or an internal libpng failure.
std::vector<std::uint8_t> encode_png(const Image& image);

/// Encode `image` and write it to `path`. Throws ProjectIOError if the file
/// cannot be opened or written; the pixel encoding failures of encode_png
/// propagate unchanged.
void write_png(const std::string& path, const Image& image);

}  // namespace alphashape
