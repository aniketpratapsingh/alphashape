// GeoTIFF 1.1 writer/reader implemented directly against TIFF 6.0 via libtiff.
// See geotiff.hpp for the interface. The tag numbers and their process-wide
// libtiff registration live in base/io/geotiff_tags.hpp.
#include "model/raster/geotiff.hpp"

#include "base/io/geotiff_tags.hpp"

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <tiffio.h>

#include "base/utils/error.hpp"

namespace alphashape {
namespace {

/// The GeoTIFF tag numbers and the process-wide registration live in
/// base/io/geotiff_tags.hpp, shared with the CRS layer's geoid decoder: libtiff
/// keeps exactly one tag extender per process, and two of them would chain and
/// merge overlapping field sets. This TU only consumes the reader/writer side.

// --- GeoKey IDs ---------------------------------------------------------
constexpr std::uint16_t kKeyGtModelType = 1024;
constexpr std::uint16_t kKeyGtRasterType = 1025;
/// GTCitationGeoKey: a general citation some writers (GDAL, ArcGIS) put a WKT
/// or an "ESRI PE String = <WKT>" prefix into when no CRS-specific key does.
constexpr std::uint16_t kKeyGtCitation = 1026;
constexpr std::uint16_t kKeyGeographicType = 2048;
constexpr std::uint16_t kKeyGeogCitation = 2049;
constexpr std::uint16_t kKeyProjectedCstType = 3072;
constexpr std::uint16_t kKeyPcsCitation = 3073;

constexpr std::uint16_t kModelProjected = 1;
constexpr std::uint16_t kModelGeographic = 2;
constexpr std::uint16_t kModelUserDefined = 32767;
constexpr std::uint16_t kRasterTypePixelIsArea = 1;

/// A 4xxx EPSG code is a geographic CRS; the special-cased 4326 always is.
bool is_geographic_epsg(int epsg) { return epsg >= 4000 && epsg <= 4999; }

// --- Silence libtiff, collecting the message into the thrown exception ---

/// Per-handle handlers so a corrupt file produces an exception and not a wall
/// of stderr text, without mutating libtiff's process-global handlers (which
/// would race with another thread decoding at the same time). Mirrors the
/// approach in image/io/tiff_codec.cpp.
struct TiffOpener {
    TIFFOpenOptions* options = nullptr;
    std::string last_message;

    TiffOpener() : options(TIFFOpenOptionsAlloc()) {}
    ~TiffOpener() {
        if (options != nullptr) {
            TIFFOpenOptionsFree(options);
        }
    }
    TiffOpener(const TiffOpener&) = delete;
    TiffOpener& operator=(const TiffOpener&) = delete;

    static int handler(TIFF* /*tif*/, void* user, const char* module, const char* fmt,
                       va_list ap) {
        if (user != nullptr) {
            auto* self = static_cast<TiffOpener*>(user);
            char buffer[512];
            // libtiff hands us a printf format plus a va_list; formatting it is
            // exactly the point, so the non-literal warning is silenced here.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif
            const int n = std::vsnprintf(buffer, sizeof(buffer), fmt, ap);
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
            self->last_message.assign(buffer, n > 0 ? static_cast<std::size_t>(n) : 0U);
            if (module != nullptr) {
                self->last_message += " (";
                self->last_message += module;
                self->last_message += ")";
            }
        }
        return 1;  // handled: print nothing
    }

    TIFF* open(const std::string& path, const char* mode) {
        if (options == nullptr) {
            throw ProjectIOError("GeoTIFF: cannot allocate TIFF open options");
        }
        // The extender must be installed before the handle is created: libtiff
        // invokes it from inside the open path to add our custom fields.
        install_geotiff_tags();
        TIFFOpenOptionsSetErrorHandlerExtR(options, handler, this);
        TIFFOpenOptionsSetWarningHandlerExtR(options, handler, this);
        TIFF* tif = TIFFOpenExt(path.c_str(), mode, options);
        if (tif == nullptr) {
            throw ProjectIOError("GeoTIFF: cannot open '" + path + "'" +
                                 (last_message.empty() ? "" : " (" + last_message + ")"));
        }
        return tif;
    }
};

/// RAII close.
struct TiffCloser {
    TIFF* tif = nullptr;
    TiffCloser() = default;
    ~TiffCloser() {
        if (tif != nullptr) {
            TIFFClose(tif);
        }
    }
    TiffCloser(const TiffCloser&) = delete;
    TiffCloser& operator=(const TiffCloser&) = delete;
};

void validate_grid(const RasterGrid& g, const char* what) {
    if (g.width <= 0 || g.height <= 0) {
        throw ValueError(std::string("GeoTIFF: ") + what + " has non-positive dimensions");
    }
    if (!(g.pixel_size_x > 0.0) || !(g.pixel_size_y > 0.0)) {
        throw ValueError(std::string("GeoTIFF: ") + what + " has non-positive pixel size");
    }
}

/// A GeoKey whose value is stored inline in the directory.
void push_inline_key(std::vector<std::uint16_t>& keys, std::uint16_t id,
                     std::uint16_t value) {
    keys.push_back(id);
    keys.push_back(0);  // TIFFTagLocation: the value is inline
    keys.push_back(1);  // Count
    keys.push_back(value);
}

/// A GeoKey whose value is a string in GeoAsciiParams: the entry points at
/// `offset`, `count` characters including the `|` terminator (libgeotiff's and
/// GDAL's convention).
void push_ascii_key(std::vector<std::uint16_t>& keys, std::string& ascii,
                    std::uint16_t id, const std::string& text) {
    if (text.size() + 1 > 0xFFFFU || ascii.size() + text.size() + 1 > 0xFFFFU) {
        throw ValueError("GeoTIFF: CRS definition is too long for GeoAsciiParams");
    }
    const auto offset = static_cast<std::uint16_t>(ascii.size());
    const auto count = static_cast<std::uint16_t>(text.size() + 1);
    ascii.append(text);
    ascii.push_back('|');
    keys.push_back(id);
    keys.push_back(static_cast<std::uint16_t>(kTagGeoAsciiParams));
    keys.push_back(count);
    keys.push_back(offset);
}

/// The GeoKeyDirectory and the GeoAsciiParams pool it points into.
struct GeoKeyEncoding {
    std::vector<std::uint16_t> keys;
    std::string ascii;
};

/// How the writer chooses the model type, the type key and its citation.
struct GeoKeyPlan {
    std::uint16_t model_type = kModelUserDefined;
    std::uint16_t type_key = 0;      // 2048/3072; 0 = no CRS key
    std::uint16_t type_value = 0;    // EPSG code, or 32767 when user-defined
    std::uint16_t citation_key = 0;  // 2049/3073; 0 = no citation
    std::string citation;            // CRS name, or full WKT2 for a custom CRS
};

GeoKeyPlan plan_geo_keys(const GeoTiffOptions& options) {
    GeoKeyPlan plan;
    const crs::CoordinateSystem* cs = nullptr;
    bool projected = false;
    bool geographic = false;
    int epsg = options.epsg;
#if defined(ALPHASHAPE_USE_PROJ)
    if (options.crs.has_value() && options.crs->is_set()) {
        cs = &*options.crs;
        epsg = cs->epsg();
        const crs::CrsKind kind = cs->kind();
        projected = kind == crs::CrsKind::Projected;
        geographic =
            kind == crs::CrsKind::Geographic2D || kind == crs::CrsKind::Geographic3D;
    }
#endif
    if (cs == nullptr) {
        if (options.epsg == 0) {
            return plan;  // local: model type 32767 and no CRS key
        }
        geographic = is_geographic_epsg(options.epsg);
        projected = !geographic;
    }
    if (projected) {
        plan.model_type = kModelProjected;
        plan.type_key = kKeyProjectedCstType;
        plan.citation_key = kKeyPcsCitation;
    } else if (geographic) {
        plan.model_type = kModelGeographic;
        plan.type_key = kKeyGeographicType;
        plan.citation_key = kKeyGeogCitation;
    } else {
        // Vertical, compound or geocentric: no GeoTIFF model type describes it,
        // and the CRS engine does not expose a compound CRS's horizontal part,
        // so the file stays user-defined and carries the full WKT2 below.
        plan.type_key = kKeyGeographicType;
        plan.citation_key = kKeyGeogCitation;
    }
    // Only a horizontal code that fits the 16-bit key becomes the type value.
    // A custom CRS, a vertical/compound code (which must never masquerade as a
    // horizontal 2048/3072 code) and a code above 65535 are user-defined and
    // carry their WKT2 definition in the citation (GDAL's convention).
    const bool horizontal = projected || geographic;
    const bool code_fits = epsg > 0 && epsg <= 0xFFFF;
    if (horizontal && code_fits) {
        plan.type_value = static_cast<std::uint16_t>(epsg);
#if defined(ALPHASHAPE_USE_PROJ)
        if (cs != nullptr) {
            plan.citation = cs->name();
        }
#endif
    } else {
        plan.type_value = kModelUserDefined;
#if defined(ALPHASHAPE_USE_PROJ)
        if (cs != nullptr) {
            plan.citation = cs->to_wkt();
        }
#endif
    }
    return plan;
}

/// Builds the GeoKeyDirectory (sorted by ascending key id) and the
/// GeoAsciiParams pool its citation entries point into.
GeoKeyEncoding build_geo_keys(const GeoTiffOptions& options) {
    const GeoKeyPlan plan = plan_geo_keys(options);
    const bool has_type = plan.type_key != 0;
    const bool has_citation = has_type && !plan.citation.empty();
    GeoKeyEncoding out;
    out.keys = {1, 1, 0,
                static_cast<std::uint16_t>(2 + (has_type ? 1 : 0) +
                                           (has_citation ? 1 : 0))};
    push_inline_key(out.keys, kKeyGtModelType, plan.model_type);
    push_inline_key(out.keys, kKeyGtRasterType, kRasterTypePixelIsArea);
    if (has_type) {
        push_inline_key(out.keys, plan.type_key, plan.type_value);
        if (has_citation) {
            push_ascii_key(out.keys, out.ascii, plan.citation_key, plan.citation);
        }
    }
    return out;
}

/// Writes the georeferencing tags shared by all variants.
void write_geo_tags(TIFF* tif, const RasterGrid& g, const GeoTiffOptions& options) {
    const double pixel_scale[3] = {g.pixel_size_x, g.pixel_size_y, 0.0};
    TIFFSetField(tif, kTagModelPixelScale, 3, pixel_scale);
    // PixelIsArea: the tiepoint is the outer top-left corner of pixel (0,0).
    const double tiepoint[6] = {0.0, 0.0, 0.0, g.origin_x, g.origin_y, 0.0};
    TIFFSetField(tif, kTagModelTiepoint, 6, tiepoint);

    GeoKeyEncoding keys = build_geo_keys(options);
    TIFFSetField(tif, kTagGeoKeyDirectory, static_cast<std::uint32_t>(keys.keys.size()),
                 keys.keys.data());
    if (!keys.ascii.empty()) {
        TIFFSetField(tif, kTagGeoAsciiParams, keys.ascii.data());
    }

    if (options.nodata.has_value()) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%.17g", *options.nodata);
        TIFFSetField(tif, kTagGdalNodata, buffer);
    }
}

/// The layout choices (tiling, compression, predictor) common to both writers.
void apply_common_tags(TIFF* tif, int width, int height, const GeoTiffOptions& options,
                       int samples_per_pixel, int bits_per_sample) {
    if (options.tiled) {
        TIFFSetField(tif, TIFFTAG_TILEWIDTH, options.tile_size);
        TIFFSetField(tif, TIFFTAG_TILELENGTH, options.tile_size);
    }
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, static_cast<std::uint32_t>(width));
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, static_cast<std::uint32_t>(height));
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bits_per_sample);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    if (options.deflate) {
        TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_ADOBE_DEFLATE);
    } else {
        TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    }
}

/// Opens for writing, choosing BigTIFF when the raw payload would overflow a
/// classic 4 GB TIFF. Registering the tags happens inside.
TIFF* open_for_write(const std::string& path, std::uint64_t raw_bytes, TiffOpener& opener) {
    const char* mode = raw_bytes > (std::uint64_t{7} << 29) ? "w8" : "w";
    return opener.open(path, mode);
}

}  // namespace

void write_geotiff(const std::string& path, const GeoRaster<float>& raster,
                   const GeoTiffOptions& options) {
    validate_grid(raster.grid, "float raster");
    if (raster.channels != 1) {
        throw ValueError("GeoTIFF: float GeoTIFF must have exactly one band (a DEM)");
    }
    const std::size_t expected = static_cast<std::size_t>(raster.grid.width) *
                                 static_cast<std::size_t>(raster.grid.height) *
                                 static_cast<std::size_t>(raster.channels);
    if (raster.data.size() != expected) {
        throw ValueError("GeoTIFF: float raster data size does not match width*height*channels");
    }
    if (options.tiled && (options.tile_size <= 0 || options.tile_size % 16 != 0)) {
        throw ValueError("GeoTIFF: tile_size must be a positive multiple of 16");
    }

    TiffOpener opener;
    const std::uint64_t raw_bytes = static_cast<std::uint64_t>(raster.data.size()) * sizeof(float);
    TiffCloser close;
    close.tif = open_for_write(path, raw_bytes, opener);
    TIFF* tif = close.tif;

    apply_common_tags(tif, raster.grid.width, raster.grid.height, options, 1, 32);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    if (options.deflate) {
        // Predictor 3 (floating-point horizontal differencing) needs the
        // samples to be written as raw IEEE bytes, which the default byte
        // order gives us.
        TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_FLOATINGPOINT);
    }
    write_geo_tags(tif, raster.grid, options);

    const int width = raster.grid.width;
    const int height = raster.grid.height;
    if (options.tiled) {
        const int ts = options.tile_size;
        std::vector<float> tile(static_cast<std::size_t>(ts) * static_cast<std::size_t>(ts), 0.0F);
        for (int y = 0; y < height; y += ts) {
            for (int x = 0; x < width; x += ts) {
                const int copy_w = std::min(ts, width - x);
                const int copy_h = std::min(ts, height - y);
                std::fill(tile.begin(), tile.end(),
                          options.nodata.has_value() ? static_cast<float>(*options.nodata) : 0.0F);
                for (int ty = 0; ty < copy_h; ++ty) {
                    const float* src = raster.data.data() +
                                       (static_cast<std::size_t>(y + ty) *
                                            static_cast<std::size_t>(width) +
                                        static_cast<std::size_t>(x));
                    std::copy(src, src + copy_w,
                              tile.data() + static_cast<std::size_t>(ty) * static_cast<std::size_t>(ts));
                }
                if (TIFFWriteTile(tif, tile.data(), static_cast<std::uint32_t>(x),
                                  static_cast<std::uint32_t>(y), 0, 0) < 0) {
                    throw ProjectIOError("GeoTIFF: failed to write a tile to '" + path + "'");
                }
            }
        }
    } else {
        // libtiff's predictors encode *in place* in the buffer handed to
        // TIFFWriteScanline (both the horizontal and the floating-point one
        // accumulate into it), so the caller's raster must not be passed
        // directly: doing so silently corrupts it. One scratch row is reused.
        std::vector<float> row(static_cast<std::size_t>(width));
        for (int y = 0; y < height; ++y) {
            const float* src = raster.data.data() + static_cast<std::size_t>(y) *
                                                       static_cast<std::size_t>(width);
            std::copy(src, src + width, row.begin());
            if (TIFFWriteScanline(tif, row.data(), static_cast<std::uint32_t>(y), 0) < 0) {
                throw ProjectIOError("GeoTIFF: failed to write a scanline to '" + path + "'");
            }
        }
    }
    if (TIFFFlush(tif) != 1) {
        throw ProjectIOError("GeoTIFF: failed to flush '" + path + "'");
    }
}

void write_geotiff(const std::string& path, const GeoRaster<std::uint8_t>& raster,
                   const GeoTiffOptions& options) {
    validate_grid(raster.grid, "8-bit raster");
    if (raster.channels != 1 && raster.channels != 3 && raster.channels != 4) {
        throw ValueError("GeoTIFF: 8-bit GeoTIFF must have 1, 3 or 4 channels");
    }
    const std::size_t expected = static_cast<std::size_t>(raster.grid.width) *
                                 static_cast<std::size_t>(raster.grid.height) *
                                 static_cast<std::size_t>(raster.channels);
    if (raster.data.size() != expected) {
        throw ValueError("GeoTIFF: 8-bit raster data size does not match width*height*channels");
    }
    if (options.tiled && (options.tile_size <= 0 || options.tile_size % 16 != 0)) {
        throw ValueError("GeoTIFF: tile_size must be a positive multiple of 16");
    }

    TiffOpener opener;
    TiffCloser close;
    close.tif = open_for_write(path, static_cast<std::uint64_t>(raster.data.size()), opener);
    TIFF* tif = close.tif;

    apply_common_tags(tif, raster.grid.width, raster.grid.height, options, raster.channels, 8);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,
                 raster.channels == 1 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB);
    if (raster.channels == 4) {
        const std::uint16_t extra = EXTRASAMPLE_UNASSALPHA;
        TIFFSetField(tif, TIFFTAG_EXTRASAMPLES, 1, &extra);
    }
    if (options.deflate) {
        TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
    }
    write_geo_tags(tif, raster.grid, options);

    const int width = raster.grid.width;
    const int height = raster.grid.height;
    const int ch = raster.channels;
    const std::uint8_t pad =
        options.nodata.has_value() ? static_cast<std::uint8_t>(*options.nodata) : std::uint8_t{0};
    if (options.tiled) {
        const int ts = options.tile_size;
        std::vector<std::uint8_t> tile(static_cast<std::size_t>(ts) * static_cast<std::size_t>(ts) *
                                           static_cast<std::size_t>(ch),
                                       pad);
        for (int y = 0; y < height; y += ts) {
            for (int x = 0; x < width; x += ts) {
                const int copy_w = std::min(ts, width - x);
                const int copy_h = std::min(ts, height - y);
                std::fill(tile.begin(), tile.end(), pad);
                const std::size_t tile_row = static_cast<std::size_t>(ts) * static_cast<std::size_t>(ch);
                for (int ty = 0; ty < copy_h; ++ty) {
                    const std::uint8_t* src =
                        raster.data.data() +
                        (static_cast<std::size_t>(y + ty) * static_cast<std::size_t>(width) +
                         static_cast<std::size_t>(x)) *
                            static_cast<std::size_t>(ch);
                    std::copy(src, src + static_cast<std::size_t>(copy_w) * static_cast<std::size_t>(ch),
                              tile.data() + static_cast<std::size_t>(ty) * tile_row);
                }
                if (TIFFWriteTile(tif, tile.data(), static_cast<std::uint32_t>(x),
                                  static_cast<std::uint32_t>(y), 0, 0) < 0) {
                    throw ProjectIOError("GeoTIFF: failed to write a tile to '" + path + "'");
                }
            }
        }
    } else {
        // Same in-place predictor hazard as the float path above: encode into a
        // scratch row rather than the caller's raster.
        std::vector<std::uint8_t> row(static_cast<std::size_t>(width) *
                                      static_cast<std::size_t>(ch));
        for (int y = 0; y < height; ++y) {
            const std::uint8_t* src =
                raster.data.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(width) *
                                         static_cast<std::size_t>(ch);
            std::copy(src, src + row.size(), row.begin());
            if (TIFFWriteScanline(tif, row.data(), static_cast<std::uint32_t>(y), 0) < 0) {
                throw ProjectIOError("GeoTIFF: failed to write a scanline to '" + path + "'");
            }
        }
    }
    if (TIFFFlush(tif) != 1) {
        throw ProjectIOError("GeoTIFF: failed to flush '" + path + "'");
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

namespace {

/// Enough of the first IFD to lay out the pixel buffer and the GeoTIFF tags.
struct ReadLayout {
    int width = 0;
    int height = 0;
    int channels = 1;
    int bits_per_sample = 0;
    bool is_float = false;
    bool tiled = false;
    int tile_width = 0;
    int tile_height = 0;
};

ReadLayout read_layout(TIFF* tif, const std::string& path) {
    ReadLayout out;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    if (TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width) == 0 ||
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height) == 0) {
        throw ProjectIOError("'" + path + "': TIFF has no image dimensions");
    }
    if (width == 0 || height == 0) {
        throw ProjectIOError("'" + path + "': TIFF has zero size");
    }
    std::uint16_t bits = 1;
    std::uint16_t samples = 1;
    std::uint16_t planar = PLANARCONFIG_CONTIG;
    std::uint16_t sample_format = SAMPLEFORMAT_UINT;
    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bits);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);
    TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &planar);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &sample_format);

    if (planar != PLANARCONFIG_CONTIG) {
        throw ProjectIOError("'" + path + "': only planar-contiguous TIFF is supported");
    }
    if (bits != 8 && bits != 32) {
        throw ProjectIOError("'" + path + "': only 8- and 32-bit samples are supported (got " +
                             std::to_string(bits) + ")");
    }
    if (bits == 32 && sample_format != SAMPLEFORMAT_IEEEFP) {
        throw ProjectIOError("'" + path + "': 32-bit samples must be IEEE float");
    }
    if (bits == 8 && sample_format != SAMPLEFORMAT_UINT && sample_format != SAMPLEFORMAT_VOID) {
        throw ProjectIOError("'" + path + "': 8-bit samples must be unsigned integer");
    }
    if (samples != 1 && samples != 3 && samples != 4) {
        throw ProjectIOError("'" + path + "': unsupported sample count (" +
                             std::to_string(samples) + ")");
    }
    // Guard against a malicious header before any buffer is sized from it.
    // A sidecar entry is bounded by the archive, but a GeoTIFF's dimensions
    // live in fixed-width tags, so a 40-byte file can still claim 2^32-1 per
    // side. The per-side limit keeps width*height*channels*bytes far from
    // overflow and from any allocation the process could not honour; the byte
    // limit then bounds the whole raster.
    constexpr std::uint64_t kMaxSide = 1U << 20U;
    if (width > kMaxSide || height > kMaxSide) {
        throw ProjectIOError("'" + path + "': TIFF declares an implausible size");
    }
    const std::uint64_t pixels = static_cast<std::uint64_t>(width) *
                                 static_cast<std::uint64_t>(height);
    const std::uint64_t bytes = pixels * samples * (bits / 8U);
    if (bytes > (std::uint64_t{4} << 30)) {
        throw ProjectIOError("'" + path + "': TIFF declares an implausible size");
    }
    // A GeoTIFF payload cannot be much larger than the file that carries it:
    // even at Deflate's theoretical best, storage expands by at most 1032:1
    // (the same bound the zip layer enforces). Reject a geometry the file is far
    // too small to hold, rather than allocating it. A size we cannot read is not
    // evidence of corruption, so that passes.
    constexpr std::uint64_t kMaxDeflateRatio = 1032;
    std::error_code size_ec;
    const std::uintmax_t file_bytes = std::filesystem::file_size(path, size_ec);
    if (!size_ec && static_cast<std::uint64_t>(file_bytes) < bytes / kMaxDeflateRatio) {
        throw ProjectIOError("'" + path +
                             "': TIFF declares a raster larger than its file can hold");
    }

    out.width = static_cast<int>(width);
    out.height = static_cast<int>(height);
    out.channels = samples;
    out.bits_per_sample = bits;
    out.is_float = bits == 32;
    if (TIFFIsTiled(tif) != 0) {
        out.tiled = true;
        std::uint32_t tw = 0;
        std::uint32_t th = 0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tw);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &th);
        if (tw == 0 || th == 0) {
            throw ProjectIOError("'" + path + "': tiled TIFF with zero tile size");
        }
        out.tile_width = static_cast<int>(tw);
        out.tile_height = static_cast<int>(th);
    }
    return out;
}

RasterGrid read_grid(TIFF* tif, const std::string& path, int width, int height) {
    // ModelTiepointTag: {i,j,k, x,y,z}; with PixelIsArea the (0,0,0) tie maps
    // to the outer top-left corner, which is exactly RasterGrid::origin_*.
    std::uint16_t tp_count = 0;
    double* tp = nullptr;
    double origin_x = 0.0;
    double origin_y = 0.0;
    if (TIFFGetField(tif, kTagModelTiepoint, &tp_count, &tp) != 0 && tp != nullptr && tp_count >= 6) {
        origin_x = tp[3];
        origin_y = tp[4];
    }
    double psx = 1.0;
    double psy = 1.0;
    std::uint16_t ps_count = 0;
    double* ps = nullptr;
    if (TIFFGetField(tif, kTagModelPixelScale, &ps_count, &ps) != 0 && ps != nullptr && ps_count >= 2) {
        psx = ps[0];
        psy = ps[1];
    }
    if (!(psx > 0.0) || !(psy > 0.0)) {
        throw ProjectIOError("'" + path + "': pixel scale must be positive");
    }
    RasterGrid grid;
    grid.width = width;
    grid.height = height;
    grid.origin_x = origin_x;
    grid.origin_y = origin_y;
    grid.pixel_size_x = psx;
    grid.pixel_size_y = psy;
    return grid;
}

/// The GeoKeys the reader cares about, decoded from the GeoKeyDirectory.
struct GeoKeys {
    std::uint16_t model_type = kModelUserDefined;
    int epsg = 0;
    std::optional<std::string> citation;  // CRS name or, for a custom CRS, WKT
};

/// The `count` characters at `offset` of a GeoAsciiParams string, minus the
/// `|` separator and any NUL padding. Empty when the key does not point there
/// or the range is out of bounds.
std::string ascii_value(const char* ascii, std::size_t ascii_len,
                        std::uint16_t location, std::uint16_t count,
                        std::uint16_t offset) {
    if (location != kTagGeoAsciiParams || ascii == nullptr ||
        static_cast<std::size_t>(offset) >= ascii_len) {
        return {};
    }
    const std::size_t n = std::min<std::size_t>(count, ascii_len - offset);
    std::string value(ascii + offset, n);
    while (!value.empty() && (value.back() == '|' || value.back() == '\0')) {
        value.pop_back();
    }
    return value;
}

/// Reads the directories and resolves the primary type key. An EPSG key wins;
/// a user-defined (32767) or absent one leaves `epsg` 0 and exposes the
/// citation so the caller can try to parse WKT from it.
GeoKeys read_geo_keys(TIFF* tif) {
    GeoKeys out;
    std::uint16_t count = 0;
    std::uint16_t* keys = nullptr;
    if (TIFFGetField(tif, kTagGeoKeyDirectory, &count, &keys) == 0 || keys == nullptr ||
        count < 4) {
        return out;
    }
    const std::uint16_t entries = keys[3];
    // Each entry is 4 SHORTs, following the 4-SHORT header.
    if (static_cast<std::uint32_t>(4) + static_cast<std::uint32_t>(entries) * 4U > count) {
        return out;
    }
    const char* ascii = nullptr;
    TIFFGetField(tif, kTagGeoAsciiParams, &ascii);
    const std::size_t ascii_len = ascii != nullptr ? std::strlen(ascii) : 0U;
    std::uint16_t geographic_type = 0;
    std::uint16_t projected_type = 0;
    std::optional<std::string> geog_citation;
    std::optional<std::string> pcs_citation;
    std::optional<std::string> gt_citation;
    for (std::uint16_t i = 0; i < entries; ++i) {
        const std::size_t base = 4 + static_cast<std::size_t>(i) * 4;
        const std::uint16_t key_id = keys[base];
        const std::uint16_t location = keys[base + 1];
        const std::uint16_t n = keys[base + 2];
        const std::uint16_t value = keys[base + 3];
        switch (key_id) {
            case kKeyGtModelType:
                if (location == 0 && n >= 1) {
                    out.model_type = value;
                }
                break;
            case kKeyGeographicType:
                if (location == 0 && n >= 1) {
                    geographic_type = value;
                }
                break;
            case kKeyProjectedCstType:
                if (location == 0 && n >= 1) {
                    projected_type = value;
                }
                break;
            case kKeyGeogCitation: {
                std::string s = ascii_value(ascii, ascii_len, location, n, value);
                if (!s.empty()) {
                    geog_citation = std::move(s);
                }
                break;
            }
            case kKeyPcsCitation: {
                std::string s = ascii_value(ascii, ascii_len, location, n, value);
                if (!s.empty()) {
                    pcs_citation = std::move(s);
                }
                break;
            }
            case kKeyGtCitation: {
                std::string s = ascii_value(ascii, ascii_len, location, n, value);
                if (!s.empty()) {
                    gt_citation = std::move(s);
                }
                break;
            }
            default:
                break;
        }
    }
    // 32767 is "user-defined", not an EPSG code. Prefer a real (non-user-
    // defined) projected code, then a real geographic one: a file may carry
    // both, and a vertical code must never be mistaken for either.
    const auto real = [](std::uint16_t code) {
        return code != 0 && code != kModelUserDefined ? code : std::uint16_t{0};
    };
    const std::uint16_t projected_code = real(projected_type);
    const std::uint16_t geographic_code = real(geographic_type);
    const std::uint16_t type = projected_code != 0 ? projected_code : geographic_code;
    out.epsg = static_cast<int>(type);
    // Citation priority: PCS, then geographic, then the general GTCitation.
    if (pcs_citation.has_value()) {
        out.citation = std::move(pcs_citation);
    } else if (geog_citation.has_value()) {
        out.citation = std::move(geog_citation);
    } else if (gt_citation.has_value()) {
        out.citation = std::move(gt_citation);
    }
    return out;
}

/// The CRS a file records: EPSG key first, then WKT in the citation, then the
/// local frame. Always populated in a PROJ-enabled build; a build without the
/// CRS engine returns the EPSG number only.
struct ReadCrsInfo {
    int epsg = 0;
    std::optional<crs::CoordinateSystem> crs;
};

ReadCrsInfo read_crs(TIFF* tif) {
    const GeoKeys keys = read_geo_keys(tif);
    ReadCrsInfo out;
    out.epsg = keys.epsg;
#if defined(ALPHASHAPE_USE_PROJ)
    if (keys.epsg != 0) {
        try {
            out.crs = crs::CoordinateSystem::from_epsg(keys.epsg);
            return out;
        } catch (const ValueError&) {
            // A code this PROJ build does not know: try the citation instead.
        }
    }
    if (keys.citation.has_value() && !keys.citation->empty()) {
        // ArcGIS writes its WKT as "ESRI PE String = <WKT>" in GTCitation;
        // strip the marker before handing the text to PROJ.
        constexpr const char* kEsriPrefix = "ESRI PE String = ";
        std::string wkt = *keys.citation;
        if (wkt.rfind(kEsriPrefix, 0) == 0) {
            wkt.erase(0, std::strlen(kEsriPrefix));
        }
        try {
            out.crs = crs::CoordinateSystem::from_wkt(wkt);
            return out;
        } catch (const ValueError&) {
            // The citation is a plain name (or malformed WKT): fall back local.
        }
    }
    out.crs = crs::CoordinateSystem::local();
#else
    (void)tif;
#endif
    return out;
}

std::optional<double> read_nodata(TIFF* tif) {
    char* text = nullptr;
    if (TIFFGetField(tif, kTagGdalNodata, &text) == 0 || text == nullptr) {
        return std::nullopt;
    }
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text) {
        return std::nullopt;  // not parseable
    }
    return value;
}

/// Reads every pixel into `out`, handling tiled and stripped files. The
/// destination is sized and zero/nodata-padded by the caller.
void read_pixels(TIFF* tif, const ReadLayout& layout, const std::string& path,
                 std::uint8_t* out, std::size_t out_row_bytes) {
    const std::size_t sample_bytes = static_cast<std::size_t>(layout.bits_per_sample) / 8U;
    const std::size_t pixel_bytes =
        static_cast<std::size_t>(layout.channels) * sample_bytes;
    if (layout.tiled) {
        const tmsize_t tile_size = TIFFTileSize(tif);
        const tmsize_t tile_row = TIFFTileRowSize(tif);
        if (tile_size <= 0 || tile_row <= 0) {
            throw ProjectIOError("'" + path + "': tiled TIFF with unusable tile size");
        }
        if (static_cast<std::size_t>(tile_row) <
            static_cast<std::size_t>(layout.tile_width) * pixel_bytes) {
            throw ProjectIOError("'" + path + "': tile row is smaller than its pixels");
        }
        std::vector<std::uint8_t> tile(static_cast<std::size_t>(tile_size));
        for (int y = 0; y < layout.height; y += layout.tile_height) {
            for (int x = 0; x < layout.width; x += layout.tile_width) {
                if (TIFFReadTile(tif, tile.data(), static_cast<std::uint32_t>(x),
                                 static_cast<std::uint32_t>(y), 0, 0) < 0) {
                    throw ProjectIOError("'" + path + "': failed to read a tile");
                }
                const int copy_w = std::min(layout.tile_width, layout.width - x);
                const int copy_h = std::min(layout.tile_height, layout.height - y);
                const std::size_t copy_bytes =
                    static_cast<std::size_t>(copy_w) * pixel_bytes;
                for (int ty = 0; ty < copy_h; ++ty) {
                    const std::uint8_t* src =
                        tile.data() + static_cast<std::size_t>(ty) *
                                          static_cast<std::size_t>(tile_row);
                    std::uint8_t* dst = out +
                                        static_cast<std::size_t>(y + ty) * out_row_bytes +
                                        static_cast<std::size_t>(x) * pixel_bytes;
                    std::memcpy(dst, src, copy_bytes);
                }
            }
        }
        return;
    }

    const tmsize_t scanline = TIFFScanlineSize(tif);
    if (scanline <= 0 || static_cast<std::size_t>(scanline) > out_row_bytes) {
        throw ProjectIOError("'" + path + "': unusable scanline size");
    }
    for (int y = 0; y < layout.height; ++y) {
        std::uint8_t* dst = out + static_cast<std::size_t>(y) * out_row_bytes;
        if (TIFFReadScanline(tif, dst, static_cast<std::uint32_t>(y), 0) < 0) {
            throw ProjectIOError("'" + path + "': failed to read scanline " + std::to_string(y));
        }
    }
}

}  // namespace

GeoTiffInfo read_geotiff_info(const std::string& path) {
    TiffOpener opener;
    TiffCloser close;
    close.tif = opener.open(path, "r");
    TIFF* tif = close.tif;

    const ReadLayout layout = read_layout(tif, path);
    GeoTiffInfo info;
    info.grid = read_grid(tif, path, layout.width, layout.height);
    info.channels = layout.channels;
    info.bits_per_sample = layout.bits_per_sample;
    info.is_float = layout.is_float;
    ReadCrsInfo crs_info = read_crs(tif);
    info.epsg = crs_info.epsg;
    info.crs = std::move(crs_info.crs);
    info.tiled = layout.tiled;
    info.nodata = read_nodata(tif);
    return info;
}

GeoRaster<float> read_geotiff_float(const std::string& path) {
    TiffOpener opener;
    TiffCloser close;
    close.tif = opener.open(path, "r");
    TIFF* tif = close.tif;

    const ReadLayout layout = read_layout(tif, path);
    if (!layout.is_float) {
        throw ProjectIOError("'" + path + "': not a 32-bit float GeoTIFF (bits=" +
                             std::to_string(layout.bits_per_sample) + ")");
    }
    if (layout.channels != 1) {
        throw ProjectIOError("'" + path + "': float GeoTIFF must have exactly one band");
    }

    GeoRaster<float> raster;
    raster.grid = read_grid(tif, path, layout.width, layout.height);
    raster.channels = 1;
    raster.data.assign(static_cast<std::size_t>(layout.width) *
                           static_cast<std::size_t>(layout.height),
                       0.0F);
    const std::size_t row_bytes = static_cast<std::size_t>(layout.width) * sizeof(float);
    read_pixels(tif, layout, path, reinterpret_cast<std::uint8_t*>(raster.data.data()), row_bytes);
    return raster;
}

GeoRaster<std::uint8_t> read_geotiff_u8(const std::string& path) {
    TiffOpener opener;
    TiffCloser close;
    close.tif = opener.open(path, "r");
    TIFF* tif = close.tif;

    const ReadLayout layout = read_layout(tif, path);
    if (layout.is_float || layout.bits_per_sample != 8) {
        throw ProjectIOError("'" + path + "': not an 8-bit GeoTIFF");
    }

    GeoRaster<std::uint8_t> raster;
    raster.grid = read_grid(tif, path, layout.width, layout.height);
    raster.channels = layout.channels;
    raster.data.assign(static_cast<std::size_t>(layout.width) *
                           static_cast<std::size_t>(layout.height) *
                           static_cast<std::size_t>(layout.channels),
                       0);
    const std::size_t row_bytes = static_cast<std::size_t>(layout.width) *
                                  static_cast<std::size_t>(layout.channels);
    read_pixels(tif, layout, path, raster.data.data(), row_bytes);
    return raster;
}

void write_world_file(const std::string& path, const RasterGrid& grid) {
    if (grid.width <= 0 || grid.height <= 0) {
        throw ValueError("world file: grid has non-positive dimensions");
    }
    if (!(grid.pixel_size_x > 0.0) || !(grid.pixel_size_y > 0.0)) {
        throw ValueError("world file: grid has non-positive pixel size");
    }
    const Eigen::Vector2d top_left = grid.pixel_center(0, 0);
    std::ofstream out(path);
    if (!out) {
        throw ProjectIOError("world file: cannot open '" + path + "' for writing");
    }
    // Six lines: A, D, B, E, C, F in the ESRI convention. "%.17g" round-trips
    // an IEEE double exactly.
    out.precision(17);
    out << grid.pixel_size_x << '\n'
        << 0.0 << '\n'
        << 0.0 << '\n'
        << -grid.pixel_size_y << '\n'
        << top_left.x() << '\n'
        << top_left.y() << '\n';
    if (!out) {
        throw ProjectIOError("world file: failed to write '" + path + "'");
    }
}

}  // namespace alphashape
