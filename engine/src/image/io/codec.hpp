#pragma once

// Internal codec interface. Not installed and not included by anything outside
// src/image/: it exists so that each codec translation unit can pull in its own
// C library header (jpeglib.h / png.h / tiffio.h) without leaking those headers
// into image_io.hpp's consumers.
//
// Every function here obeys the decode contract documented in image_io.hpp:
// the result is 1- or 3-channel 8-bit data, alpha dropped, palette expanded,
// 16-bit rounded down to 8, and unsupported layouts rejected with Error.
//
// JPEG and PNG are handed the bytes of the whole file because both libraries
// expose a memory source (jpeg_mem_src, png_set_read_fn). TIFF is handed a path
// because libtiff's per-handle error handler is only reachable through
// TIFFOpenExt, and threading its client-callback form through a byte span buys
// nothing here.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "base/utils/error.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"

namespace alphashape::image_detail {

/// The largest dimension, in pixels, a decode will accept. A corrupt header can
/// easily claim 2^32-16 in a 4-byte width field; refusing anything beyond a
/// million per side keeps the pixel-buffer arithmetic away from overflow and
/// stops a hostile file from driving the process into the OOM killer.
inline constexpr std::uint32_t kMaxDimension = 1U << 20;  // 1,048,576
/// The largest total decoded buffer, in bytes. 4 GiB is far beyond any image
/// the pipeline will meet while still being small enough that a bogus size is
/// rejected instead of allocated.
inline constexpr std::uint64_t kMaxDecodeBytes = 4ULL << 30;

/// Validates dimensions and the size of a tightly packed buffer of
/// `bytes_per_pixel` each. Throws Error with `what` on any violation, so a
/// codec can call this before it allocates anything.
void check_decodable(std::uint64_t width, std::uint64_t height, std::uint64_t bytes_per_pixel,
                     const std::string& what);

/// Reads an entire file. Throws ProjectIOError if it cannot be opened or read.
std::vector<std::uint8_t> read_file_bytes(const std::string& path);

/// Reads at most `max_bytes` from the head of a file. Used for format sniffing
/// and header probes, so a 100 MB photo is never pulled into memory just to
/// learn it is a PNG. Throws ProjectIOError if the file cannot be opened.
std::vector<std::uint8_t> read_file_prefix(const std::string& path, std::size_t max_bytes);

ImageInfo jpeg_info(const std::uint8_t* data, std::size_t size);
Image jpeg_decode(const std::uint8_t* data, std::size_t size);

ImageInfo png_info(const std::uint8_t* data, std::size_t size);
Image png_decode(const std::uint8_t* data, std::size_t size);

/// Encode an 8-bit Image as a PNG byte stream. Channel count selects the colour
/// type: 1 -> grey, 3 -> RGB, 4 -> RGBA (an Image only ever holds 1 or 3, the
/// 4-channel case exists so the encoder can serve a future RGBA page). The
/// returned bytes are a complete PNG file, no alignment or padding. Throws
/// Error on an unsupported channel count or a libpng failure.
std::vector<std::uint8_t> png_encode(const Image& image);

ImageInfo tiff_info(const std::string& path);
Image tiff_decode(const std::string& path);

/// The PNM header probe reused by read_image_info; the pixels are still read by
/// the existing read_pnm().
ImageInfo pnm_info(const std::uint8_t* data, std::size_t size);

}  // namespace alphashape::image_detail
