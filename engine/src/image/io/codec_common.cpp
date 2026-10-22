#include "image/io/codec.hpp"

#include <cctype>
#include <fstream>
#include <limits>
#include <optional>

#include "base/utils/error.hpp"

namespace alphashape::image_detail {

void check_decodable(std::uint64_t width, std::uint64_t height, std::uint64_t bytes_per_pixel,
                     const std::string& what) {
    if (width == 0 || height == 0) {
        throw ProjectIOError(what + ": zero image size");
    }
    if (width > kMaxDimension || height > kMaxDimension) {
        throw ProjectIOError(what + ": dimensions " + std::to_string(width) + "x" +
                             std::to_string(height) + " exceed the supported maximum");
    }
    if (bytes_per_pixel == 0) {
        throw ProjectIOError(what + ": image has no channels");
    }
    const std::uint64_t bytes = width * height * bytes_per_pixel;
    if (bytes > kMaxDecodeBytes) {
        throw ProjectIOError(what + ": decoded image would exceed the size limit");
    }
}

std::vector<std::uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw ProjectIOError("cannot open image '" + path + "'");
    }
    const std::streamoff end = in.tellg();
    if (end < 0) {
        throw ProjectIOError("cannot determine size of '" + path + "'");
    }
    // A pathological stream length would overflow size_t on 32-bit hosts; the
    // check is cheap and keeps the resize honest.
    if (static_cast<std::uintmax_t>(end) > std::numeric_limits<std::size_t>::max()) {
        throw ProjectIOError("'" + path + "' is too large to read");
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    in.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (static_cast<std::size_t>(in.gcount()) != bytes.size()) {
            throw ProjectIOError("short read from '" + path + "'");
        }
    }
    return bytes;
}

std::vector<std::uint8_t> read_file_prefix(const std::string& path, std::size_t max_bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw ProjectIOError("cannot open image '" + path + "'");
    }
    std::vector<std::uint8_t> bytes(max_bytes);
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(max_bytes));
    // A short read is the normal case for a small file; gcount() is the truth.
    const std::streamsize got = in.gcount();
    if (got < 0) {
        throw ProjectIOError("cannot read '" + path + "'");
    }
    bytes.resize(static_cast<std::size_t>(got));
    return bytes;
}

namespace {

/// Walks a PNM header (P5/P6) without decoding the pixels. Returns nullopt on
/// anything that is not a well-formed 8-bit binary PGM/PPM, so the caller can
/// treat it as "unknown" rather than a hard error.
std::optional<ImageInfo> probe_pnm(const std::uint8_t* data, std::size_t size) {
    // Minimal header scanner: magic, then three whitespace-separated integers,
    // skipping '#' comment lines exactly as the reader does.
    std::size_t pos = 0;
    const auto skip_space_and_comments = [&]() {
        for (;;) {
            while (pos < size && std::isspace(static_cast<unsigned char>(data[pos])) != 0) {
                ++pos;
            }
            if (pos < size && data[pos] == '#') {
                while (pos < size && data[pos] != '\n') {
                    ++pos;
                }
            } else {
                return;
            }
        }
    };
    const auto read_int = [&](int& out) -> bool {
        skip_space_and_comments();
        if (pos >= size || std::isdigit(static_cast<unsigned char>(data[pos])) == 0) {
            return false;
        }
        long value = 0;
        while (pos < size && std::isdigit(static_cast<unsigned char>(data[pos])) != 0) {
            value = value * 10 + (data[pos] - '0');
            if (value > 1000000) {  // far beyond any real PNM dimension
                return false;
            }
            ++pos;
        }
        out = static_cast<int>(value);
        return true;
    };

    if (size < 2 || data[0] != 'P' || (data[1] != '5' && data[1] != '6')) {
        return std::nullopt;
    }
    pos = 2;
    int width = 0;
    int height = 0;
    int maxval = 0;
    if (!read_int(width) || !read_int(height) || !read_int(maxval)) {
        return std::nullopt;
    }
    if (width <= 0 || height <= 0 || maxval <= 0 || maxval > 255) {
        return std::nullopt;
    }
    ImageInfo info;
    info.format = ImageFormat::Pnm;
    info.width = width;
    info.height = height;
    info.channels = data[1] == '5' ? 1 : 3;
    info.bit_depth = 8;
    return info;
}

}  // namespace

ImageInfo pnm_info(const std::uint8_t* data, std::size_t size) {
    const std::optional<ImageInfo> info = probe_pnm(data, size);
    if (!info.has_value()) {
        throw ProjectIOError("not a well-formed 8-bit binary PGM/PPM");
    }
    return *info;
}

}  // namespace alphashape::image_detail
