#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <tiffio.h>

#include "base/utils/error.hpp"
#include "image/io/codec.hpp"

namespace alphashape::image_detail {

namespace {

/// A path-based open that installs silent per-handle callbacks, so a corrupt
/// TIFF produces an exception and not a wall of stderr text. Using
/// TIFFOpenExt with a TIFFOpenOptions keeps this local to the handle instead of
/// mutating libtiff's process-global handlers, which would be a data race if
/// two threads decoded at once.
struct TiffOpener {
    TIFFOpenOptions* options = nullptr;

    TiffOpener() : options(TIFFOpenOptionsAlloc()) {}
    ~TiffOpener() {
        if (options != nullptr) {
            TIFFOpenOptionsFree(options);
        }
    }

    static int silent_handler(TIFF* /*tif*/, void* /*user*/, const char* /*module*/,
                              const char* /*fmt*/, va_list /*ap*/) {
        // Return non-zero to say "handled"; nothing is printed.
        return 1;
    }

    /// Opens for reading, or throws if the file is not a TIFF. `path` is
    /// borrowed.
    TIFF* open(const std::string& path) {
        if (options == nullptr) {
            throw ProjectIOError("TIFF: cannot allocate open options");
        }
        TIFFOpenOptionsSetErrorHandlerExtR(options, silent_handler, nullptr);
        TIFFOpenOptionsSetWarningHandlerExtR(options, silent_handler, nullptr);
        TIFF* tif = TIFFOpenExt(path.c_str(), "r", options);
        if (tif == nullptr) {
            throw ProjectIOError("cannot open TIFF '" + path + "'");
        }
        return tif;
    }
};

/// The subset of the first IFD that governs how the pixels are laid out.
struct TiffLayout {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint16_t bits_per_sample = 1;
    std::uint16_t samples_per_pixel = 1;
    std::uint16_t photometric = 0;
    std::uint16_t planar_config = 0;
    std::uint16_t sample_format = 0;
    /// Grey, RGB or palette: the colour models we decode. Anything else throws.
    enum class Model { Grey, Rgb, Palette } model = Model::Grey;
};

TiffLayout read_layout(TIFF* tif, const std::string& path) {
    TiffLayout layout;
    if (TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &layout.width) == 0 ||
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &layout.height) == 0) {
        throw ProjectIOError("'" + path + "': TIFF has no image dimensions");
    }
    if (layout.width == 0 || layout.height == 0) {
        throw ProjectIOError("'" + path + "': TIFF has zero size");
    }
    // These three are defaulted by the spec, so a missing tag is not an error.
    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &layout.bits_per_sample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &layout.samples_per_pixel);
    TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &layout.planar_config);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &layout.sample_format);
    if (TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &layout.photometric) == 0) {
        throw ProjectIOError("'" + path + "': TIFF has no photometric tag");
    }

    // Reject an implausible geometry before any buffer is sized from it. The
    // worst-case bytes-per-pixel here is 4 samples of 2 bytes.
    check_decodable(layout.width, layout.height,
                    static_cast<std::uint64_t>(layout.samples_per_pixel) * 2, path);

    if (layout.bits_per_sample != 8 && layout.bits_per_sample != 16) {
        throw ProjectIOError("'" + path + "': only 8- and 16-bit samples are supported (file has " +
                             std::to_string(layout.bits_per_sample) + ")");
    }
    if (layout.planar_config == PLANARCONFIG_SEPARATE) {
        throw ProjectIOError("'" + path + "': planar-separate TIFF is not supported");
    }
    if (layout.planar_config != PLANARCONFIG_CONTIG) {
        throw ProjectIOError("'" + path + "': unknown planar configuration");
    }
    if (layout.sample_format != SAMPLEFORMAT_UINT && layout.sample_format != SAMPLEFORMAT_VOID) {
        throw ProjectIOError("'" + path + "': only unsigned integer samples are supported");
    }
    if (layout.samples_per_pixel < 1 || layout.samples_per_pixel > 4) {
        throw ProjectIOError("'" + path + "': unsupported sample count (" +
                             std::to_string(layout.samples_per_pixel) + ")");
    }
    switch (layout.photometric) {
        case PHOTOMETRIC_MINISWHITE:
        case PHOTOMETRIC_MINISBLACK:
            if (layout.samples_per_pixel != 1 && layout.samples_per_pixel != 2) {
                throw ProjectIOError("'" + path + "': grey TIFF with " +
                                     std::to_string(layout.samples_per_pixel) + " samples");
            }
            layout.model = TiffLayout::Model::Grey;
            break;
        case PHOTOMETRIC_RGB:
            if (layout.samples_per_pixel != 3 && layout.samples_per_pixel != 4) {
                throw ProjectIOError("'" + path + "': RGB TIFF with " +
                                     std::to_string(layout.samples_per_pixel) + " samples");
            }
            layout.model = TiffLayout::Model::Rgb;
            break;
        case PHOTOMETRIC_PALETTE:
            if (layout.samples_per_pixel != 1) {
                throw ProjectIOError("'" + path + "': palette TIFF with more than one sample");
            }
            layout.model = TiffLayout::Model::Palette;
            break;
        case PHOTOMETRIC_SEPARATED:
            throw ProjectIOError("'" + path + "': CMYK/separated TIFF is not supported");
        case PHOTOMETRIC_YCBCR:
            throw ProjectIOError("'" + path + "': YCbCr TIFF is not supported");
        default:
            throw ProjectIOError("'" + path + "': unsupported TIFF photometric (" +
                                 std::to_string(layout.photometric) + ")");
    }
    return layout;
}

/// Channels as stored: grey and grey+alpha are one, RGB is three, RGBA is four.
int stored_channels(const TiffLayout& layout) {
    switch (layout.model) {
        case TiffLayout::Model::Rgb:
            return layout.samples_per_pixel == 4 ? 4 : 3;
        case TiffLayout::Model::Grey:
        case TiffLayout::Model::Palette:
            return 1;
    }
    return 1;
}

/// Reads every pixel into `raw` in the file's native sample width, regardless
/// of the storage organisation. Handles stripped and tiled files; restores the
/// MINISWHITE inversion later. The buffer is packed tightly at the file's own
/// bytes-per-sample, so a 1-byte-per-sample layout is not padded to 16 bits.
void read_raw_pixels(TIFF* tif, const TiffLayout& layout, const std::string& path,
                     std::vector<std::uint8_t>& raw) {
    const std::size_t sample_bytes = layout.bits_per_sample / 8;
    const std::size_t pixel_bytes =
        static_cast<std::size_t>(layout.samples_per_pixel) * sample_bytes;
    raw.assign(static_cast<std::size_t>(layout.width) * static_cast<std::size_t>(layout.height) *
                   pixel_bytes,
               0);
    const std::size_t row_stride = static_cast<std::size_t>(layout.width) * pixel_bytes;

    if (TIFFIsTiled(tif) != 0) {
        std::uint32_t tile_width = 0;
        std::uint32_t tile_height = 0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tile_width);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tile_height);
        if (tile_width == 0 || tile_height == 0) {
            throw ProjectIOError("'" + path + "': tiled TIFF with zero tile size");
        }
        const tmsize_t tile_size = TIFFTileSize(tif);
        const tmsize_t tile_row = TIFFTileRowSize(tif);
        if (tile_size <= 0 || tile_row <= 0) {
            throw ProjectIOError("'" + path + "': tiled TIFF with unusable tile size");
        }
        std::vector<std::uint8_t> tile(static_cast<std::size_t>(tile_size));
        for (std::uint32_t y = 0; y < layout.height; y += tile_height) {
            for (std::uint32_t x = 0; x < layout.width; x += tile_width) {
                if (TIFFReadTile(tif, tile.data(), x, y, 0, 0) < 0) {
                    throw ProjectIOError("'" + path + "': failed to read a tile");
                }
                const std::uint32_t copy_w = std::min(tile_width, layout.width - x);
                const std::uint32_t copy_h = std::min(tile_height, layout.height - y);
                const std::size_t copy_bytes = static_cast<std::size_t>(copy_w) * pixel_bytes;
                if (copy_bytes > static_cast<std::size_t>(tile_row)) {
                    throw ProjectIOError("'" + path + "': tile row is smaller than its pixels");
                }
                for (std::uint32_t ty = 0; ty < copy_h; ++ty) {
                    const std::uint8_t* src = tile.data() + static_cast<std::size_t>(ty) *
                                                                  static_cast<std::size_t>(tile_row);
                    std::uint8_t* dst = raw.data() +
                                        static_cast<std::size_t>(y + ty) * row_stride +
                                        static_cast<std::size_t>(x) * pixel_bytes;
                    std::memcpy(dst, src, copy_bytes);
                }
            }
        }
        return;
    }

    const tmsize_t scanline = TIFFScanlineSize(tif);
    if (scanline <= 0 || static_cast<std::size_t>(scanline) < row_stride) {
        throw ProjectIOError("'" + path + "': unusable scanline size");
    }
    std::vector<std::uint8_t> row(static_cast<std::size_t>(scanline));
    for (std::uint32_t y = 0; y < layout.height; ++y) {
        if (TIFFReadScanline(tif, row.data(), y, 0) < 0) {
            throw ProjectIOError("'" + path + "': failed to read scanline " + std::to_string(y));
        }
        std::memcpy(raw.data() + static_cast<std::size_t>(y) * row_stride, row.data(),
                    row_stride);
    }
}

/// Composes one 16-bit sample from the raw buffer. libtiff returns samples in
/// host order, so a plain load is correct; we still spell it out to document
/// the assumption.
std::uint16_t load_sample(const std::uint8_t* p, bool sixteen) {
    if (!sixteen) {
        return p[0];
    }
    std::uint16_t value = 0;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

}  // namespace

ImageInfo tiff_info(const std::string& path) {
    TiffOpener opener;
    TIFF* tif = opener.open(path);
    struct Close {
        TIFF* t;
        ~Close() { TIFFClose(t); }
    } close{tif};

    // First directory only, per the interface contract.
    const TiffLayout layout = read_layout(tif, path);
    ImageInfo info;
    info.format = ImageFormat::Tiff;
    info.width = static_cast<int>(layout.width);
    info.height = static_cast<int>(layout.height);
    info.channels = stored_channels(layout);
    info.bit_depth = static_cast<int>(layout.bits_per_sample);
    return info;
}

Image tiff_decode(const std::string& path) {
    TiffOpener opener;
    TIFF* tif = opener.open(path);
    struct Close {
        TIFF* t;
        ~Close() { TIFFClose(t); }
    } close{tif};

    const TiffLayout layout = read_layout(tif, path);
    std::vector<std::uint8_t> raw;
    read_raw_pixels(tif, layout, path, raw);

    // The palette lookup is the only colour model that needs tag state beyond
    // the layout, so fetch it here before the handle closes.
    const std::uint16_t* palette_r = nullptr;
    const std::uint16_t* palette_g = nullptr;
    const std::uint16_t* palette_b = nullptr;
    if (layout.model == TiffLayout::Model::Palette) {
        std::uint16_t* r = nullptr;
        std::uint16_t* g = nullptr;
        std::uint16_t* b = nullptr;
        if (TIFFGetFieldDefaulted(tif, TIFFTAG_COLORMAP, &r, &g, &b) == 0 || r == nullptr) {
            throw ProjectIOError("'" + path + "': palette TIFF has no colour map");
        }
        palette_r = r;
        palette_g = g;
        palette_b = b;
    }

    const bool sixteen = layout.bits_per_sample == 16;
    const int out_components = layout.model == TiffLayout::Model::Grey ? 1 : 3;
    const std::size_t sample_bytes = layout.bits_per_sample / 8;
    const std::size_t pixel_bytes =
        static_cast<std::size_t>(layout.samples_per_pixel) * sample_bytes;
    const bool invert = layout.photometric == PHOTOMETRIC_MINISWHITE;
    // A TIFF colour map always holds 1 << bits_per_sample entries.
    const std::uint32_t palette_entries = 1U << layout.bits_per_sample;

    Image image(static_cast<int>(layout.width), static_cast<int>(layout.height), out_components);
    for (std::uint32_t y = 0; y < layout.height; ++y) {
        const std::uint8_t* row =
            raw.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(layout.width) *
                             pixel_bytes;
        for (std::uint32_t x = 0; x < layout.width; ++x) {
            const std::uint8_t* pixel = row + static_cast<std::size_t>(x) * pixel_bytes;
            const auto sample = [&](int c) {
                return load_sample(pixel + static_cast<std::size_t>(c) * sample_bytes, sixteen);
            };
            std::uint16_t out[3] = {0, 0, 0};
            if (layout.model == TiffLayout::Model::Palette) {
                const std::uint16_t index = sample(0);
                if (index >= palette_entries) {
                    throw ProjectIOError("'" + path + "': palette index out of range");
                }
                // Colormap entries are always 16-bit, whatever the pixel bit
                // depth, and the established convention (and how writers scale
                // them) is that the 8-bit colour is the high byte. Rounding
                // instead would turn a written 255 (= 0xFF00) into 254.
                out[0] = static_cast<std::uint16_t>(palette_r[index] >> 8);
                out[1] = static_cast<std::uint16_t>(palette_g[index] >> 8);
                out[2] = static_cast<std::uint16_t>(palette_b[index] >> 8);
            } else {
                for (int c = 0; c < out_components; ++c) {
                    out[c] = sample(c);
                }
            }
            for (int c = 0; c < out_components; ++c) {
                std::uint32_t value = out[c];
                if (invert) {
                    value = sixteen ? (65535U - value) : (255U - value);
                }
                // Palette values are already 8-bit after the high-byte shift
                // above; only a 16-bit *sample* needs the rounding reduction.
                if (sixteen && layout.model != TiffLayout::Model::Palette) {
                    value = (value * 255U + 32767U) / 65535U;  // round, not truncate
                }
                image.at(static_cast<int>(x), static_cast<int>(y), c) =
                    static_cast<std::uint8_t>(value);
            }
        }
    }
    return image;
}

}  // namespace alphashape::image_detail
