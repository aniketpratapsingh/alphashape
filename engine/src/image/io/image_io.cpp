#include "image/io/image_io.hpp"

#include <cstdio>
#include <cstddef>
#include <string>
#include <limits>
#include <vector>

#include "base/utils/error.hpp"
#include "image/io/codec.hpp"
#include "image/io/pnm.hpp"

namespace alphashape {

namespace {

bool starts_with(const std::vector<std::uint8_t>& bytes, const char* magic, std::size_t len) {
    if (bytes.size() < len) {
        return false;
    }
    for (std::size_t i = 0; i < len; ++i) {
        if (bytes[i] != static_cast<std::uint8_t>(magic[i])) {
            return false;
        }
    }
    return true;
}

/// The magic-byte sniff, shared by detect_image_format and the two readers so
/// the file is only ever opened once per public call.
ImageFormat detect_from_bytes(const std::vector<std::uint8_t>& bytes) {
    if (starts_with(bytes, "\xff\xd8\xff", 3)) {
        return ImageFormat::Jpeg;
    }
    static const std::uint8_t png_sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    if (bytes.size() >= 8) {
        bool png = true;
        for (std::size_t i = 0; i < 8; ++i) {
            if (bytes[i] != png_sig[i]) {
                png = false;
                break;
            }
        }
        if (png) {
            return ImageFormat::Png;
        }
    }
    if (starts_with(bytes, "II*\0", 4) || starts_with(bytes, "MM\0*", 4)) {
        return ImageFormat::Tiff;
    }
    if (starts_with(bytes, "P5", 2) || starts_with(bytes, "P6", 2)) {
        return ImageFormat::Pnm;
    }
    return ImageFormat::Unknown;
}

/// The number of leading bytes every signature needs. 8 covers PNG's eight-byte
/// signature and the six-byte "Exif\0\0" a JPEG APP1 would start with; nothing
/// here needs more.
constexpr std::size_t kSignatureBytes = 8;

}  // namespace

ImageFormat detect_image_format(const std::string& path) {
    try {
        return detect_from_bytes(image_detail::read_file_prefix(path, kSignatureBytes));
    } catch (const Error&) {
        // A file that cannot be opened has no format; detection never throws.
        return ImageFormat::Unknown;
    }
}

ImageInfo read_image_info(const std::string& path) {
    const std::vector<std::uint8_t> bytes = image_detail::read_file_bytes(path);
    switch (detect_from_bytes(bytes)) {
        case ImageFormat::Jpeg:
            return image_detail::jpeg_info(bytes.data(), bytes.size());
        case ImageFormat::Png:
            return image_detail::png_info(bytes.data(), bytes.size());
        case ImageFormat::Tiff:
            return image_detail::tiff_info(path);
        case ImageFormat::Pnm:
            return image_detail::pnm_info(bytes.data(), bytes.size());
        case ImageFormat::Unknown:
            break;
    }
    throw ProjectIOError("'" + path + "': unrecognised image format");
}

Image read_image(const std::string& path) {
    const std::vector<std::uint8_t> bytes = image_detail::read_file_bytes(path);
    switch (detect_from_bytes(bytes)) {
        case ImageFormat::Jpeg:
            return image_detail::jpeg_decode(bytes.data(), bytes.size());
        case ImageFormat::Png:
            return image_detail::png_decode(bytes.data(), bytes.size());
        case ImageFormat::Tiff:
            return image_detail::tiff_decode(path);
        case ImageFormat::Pnm:
            // PNM pixel decoding already exists; reuse it rather than growing a
            // second implementation that could drift.
            return read_pnm(path);
        case ImageFormat::Unknown:
            break;
    }
    throw ProjectIOError("'" + path + "': unrecognised image format");
}

std::vector<std::uint8_t> encode_png(const Image& image) {
    return image_detail::png_encode(image);
}

void write_png(const std::string& path, const Image& image) {
    const std::vector<std::uint8_t> bytes = image_detail::png_encode(image);
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        throw ProjectIOError("write_png: cannot open '" + path + "' for writing");
    }
    std::size_t written = 0;
    if (!bytes.empty()) {
        written = std::fwrite(bytes.data(), 1, bytes.size(), file);
    }
    const int close_result = std::fclose(file);
    if (written != bytes.size() || close_result != 0) {
        throw ProjectIOError("write_png: failed to write '" + path + "'");
    }
}

}  // namespace alphashape
