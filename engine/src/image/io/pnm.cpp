#include "image/io/pnm.hpp"

#include <cctype>
#include <fstream>

#include "base/utils/error.hpp"

namespace alphashape {

namespace {

/// Read one header integer, skipping whitespace and '#' comments.
int read_header_int(std::istream& in, const std::string& path) {
    for (;;) {
        const int c = in.peek();
        if (c == '#') {
            std::string ignored;
            std::getline(in, ignored);
        } else if (c != EOF && std::isspace(c) != 0) {
            in.get();
        } else {
            break;
        }
    }
    int value = -1;
    if (!(in >> value) || value < 0) {
        throw ProjectIOError("'" + path + "': malformed PNM header");
    }
    return value;
}

}  // namespace

Image read_pnm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw ProjectIOError("cannot open image '" + path + "'");
    }
    char magic[2] = {};
    in.read(magic, 2);
    if (!in || magic[0] != 'P' || (magic[1] != '5' && magic[1] != '6')) {
        throw ProjectIOError("'" + path + "' is not a binary PGM/PPM (P5/P6)");
    }
    const int channels = magic[1] == '5' ? 1 : 3;
    const int width = read_header_int(in, path);
    const int height = read_header_int(in, path);
    const int maxval = read_header_int(in, path);
    if (width == 0 || height == 0) {
        throw ProjectIOError("'" + path + "' has zero size");
    }
    if (maxval == 0 || maxval > 255) {
        throw ProjectIOError("'" + path + "': only 8-bit PNM is supported (maxval " +
                             std::to_string(maxval) + ")");
    }
    in.get();  // the single whitespace byte that ends the header

    Image image(width, height, channels);
    in.read(reinterpret_cast<char*>(image.data()), static_cast<std::streamsize>(image.size_bytes()));
    if (static_cast<std::size_t>(in.gcount()) != image.size_bytes()) {
        throw ProjectIOError("'" + path + "': truncated pixel data");
    }
    return image;
}

void write_pnm(const Image& image, const std::string& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw ProjectIOError("cannot write image '" + path + "'");
    }
    out << (image.channels() == 1 ? "P5" : "P6") << '\n'
        << image.width() << ' ' << image.height() << "\n255\n";
    out.write(reinterpret_cast<const char*>(image.data()), static_cast<std::streamsize>(image.size_bytes()));
    if (!out) {
        throw ProjectIOError("short write to '" + path + "'");
    }
}

}  // namespace alphashape
