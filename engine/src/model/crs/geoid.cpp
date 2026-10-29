#include "model/crs/geoid.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <proj.h>

#include <tiffio.h>

#include "base/io/geotiff_tags.hpp"
#include "base/utils/error.hpp"
#include "model/crs/proj_context.hpp"

namespace alphashape::crs {

namespace {

struct PjDeleter {
    void operator()(PJ* pj) const {
        if (pj != nullptr) {
            proj_destroy(pj);
        }
    }
};
using PjPtr = std::unique_ptr<PJ, PjDeleter>;

struct ContextDeleter {
    void operator()(PJ_CONTEXT* ctx) const {
        if (ctx != nullptr) {
            proj_context_destroy(ctx);
        }
    }
};
using ContextPtr = std::unique_ptr<PJ_CONTEXT, ContextDeleter>;

/// Catmull-Rom (the interpolating cardinal spline): value at t in [0,1] between
/// p1 and p2. At t = 0 it returns p1 exactly, which is why a grid node
/// evaluates to its stored sample and therefore agrees with PROJ at that node.
double catmull_rom(double p0, double p1, double p2, double p3, double t) {
    return p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 +
                                          t * (3.0 * (p1 - p2) + p3 - p0)));
}

double clamp(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

/// One thread's PROJ state for one grid: a private context whose search path
/// is the grid's directory, and the vgridshift pipeline built on it.
///
/// PROJ forbids sharing a PJ_CONTEXT between threads, and these grids are
/// shared (GeoidStore hands the same GeoidGrid to every caller), so the state
/// is per thread. Construction order matters: `ctx` is declared first so it is
/// destroyed last, after `pj` which belongs to it.
struct ProjGridSession {
    ContextPtr ctx;
    PjPtr pj;
};

/// Per-thread sessions, keyed by grid path. A path identifies a grid
/// completely, so an entry never needs invalidating: it lives until the thread
/// exits, holding at most one context per distinct grid the thread evaluated.
struct ThreadGeoidProj {
    std::unordered_map<std::string, ProjGridSession> grids;
};

thread_local ThreadGeoidProj tl_geoid_proj;

/// The outcome of asking PROJ. The three cases must stay distinguishable: a
/// grid PROJ cannot open must fall back to the GeoTIFF reader, while a point
/// PROJ *refused* (outside the grid's own coverage, or nodata) must not be
/// answered by a reader that might invent a value.
struct ProjAnswer {
    enum class Kind { Unavailable, Refused, Value };
    Kind kind = Kind::Unavailable;
    double value = 0.0;
};

/// A grid evaluated through PROJ. PROJ resolves `+grids=` by name only, and the
/// thread context must not be mutated (it is shared with every other CRS
/// operation on that thread), so a private context per thread carries the
/// grid's directory as its search path and the bare file name is passed.
class ProjGridShift {
public:
    explicit ProjGridShift(std::string grid_path) {
        const std::filesystem::path path(grid_path);
        dir_ = path.parent_path().empty() ? std::string(".") : path.parent_path().string();
        name_ = path.filename().string();
        // PROJ splits +grids= on whitespace and has no quoting rule, so a file
        // name containing a space cannot be handed over at all. That is not an
        // error: proj_undulation() stays unavailable and the GeoTIFF reader
        // answers instead.
        available_ = !name_.empty() && name_.find_first_of(" \t") == std::string::npos;
        key_ = dir_ + "/" + name_;
    }

    /// True when PROJ could be asked for this grid at all (path shape only; a
    /// grid PROJ cannot open is discovered on first evaluation).
    bool available() const { return available_; }

    ProjAnswer at(double lon_deg, double lat_deg) const {
        if (!available_) {
            return {};
        }
        ProjGridSession& session = session_for_thread();
        if (!session.pj) {
            return {};
        }
        proj_errno_reset(session.pj.get());
        // Visualisation order (x = lon, y = lat) in, the same order out.
        const PJ_COORD in = proj_coord(lon_deg, lat_deg, 0.0, 0.0);
        const PJ_COORD out = proj_trans(session.pj.get(), PJ_FWD, in);
        if (proj_errno(session.pj.get()) != 0 || !std::isfinite(out.xyz.z)) {
            return ProjAnswer{ProjAnswer::Kind::Refused, 0.0};
        }
        return ProjAnswer{ProjAnswer::Kind::Value, out.xyz.z};
    }

private:
    /// This thread's session for this grid, created on first use. A grid the
    /// session could not build stays cached as "no pipeline" so a 77 MB grid
    /// PROJ refuses is not re-tried on every point.
    ProjGridSession& session_for_thread() const {
        auto found = tl_geoid_proj.grids.find(key_);
        if (found != tl_geoid_proj.grids.end()) {
            return found->second;
        }
        ProjGridSession session;
        session.ctx.reset(proj_context_create());
        if (session.ctx) {
            // No downloads, no shared grid cache: the file is already on disk
            // and is the only grid this context will ever look for.
            proj_context_set_enable_network(session.ctx.get(), 0);
            proj_grid_cache_set_enable(session.ctx.get(), 0);
            const char* paths[2] = {dir_.c_str(), nullptr};
            proj_context_set_search_paths(session.ctx.get(), 1, paths);
            // vgridshift works in radians like every PROJ step, so the degrees
            // the engine speaks are converted in and back out. The vertical
            // value it produces is already metres.
            const std::string pipeline = "+proj=pipeline"
                                         " +step +proj=unitconvert +xy_in=deg +xy_out=rad"
                                         " +step +proj=vgridshift +grids=" +
                                         name_ +
                                         " +multiplier=1"
                                         " +step +proj=unitconvert +xy_in=rad +xy_out=deg";
            session.pj.reset(proj_create(session.ctx.get(), pipeline.c_str()));
        }
        return tl_geoid_proj.grids.emplace(key_, std::move(session)).first->second;
    }

    std::string dir_;
    std::string name_;
    std::string key_;
    bool available_ = false;
};

// ---------------------------------------------------------------------------
// The GeoTIFF carrier. A geoid grid is the NGA layout: one band, 32-bit IEEE
// float, ModelPixelScale (33550) and ModelTiepoint (33922) describing a north-up
// grid, plus GDAL_NODATA (42113) when cells are missing. This is decoded here
// rather than through model/raster/geotiff.cpp because that library depends on
// this one, so calling it back would be a cycle; the subset needed is a header
// read and one copy loop. The tag registration itself is shared, though: see
// base/io/geotiff_tags.hpp.
// ---------------------------------------------------------------------------

/// GeoKey 1025, GTRasterTypeGeoKey: 1 = PixelIsArea, 2 = PixelIsPoint.
constexpr std::uint16_t kKeyRasterType = 1025;

struct DecodedGrid {
    GeoidGridInfo info;
    std::vector<float> samples;
    /// GDAL_NODATA (42113) as a number; `has_nodata` is false when the tag is
    /// absent or is not a finite number.
    double nodata = 0.0;
    bool has_nodata = false;
};

/// Reads one geoid GeoTIFF. Every failure is a ProjectIOError naming the file,
/// because a store with an unreadable grid must say which one. With
/// `decode_samples` false only the header and tags are read, so asking for a
/// bounding box (add/save) does not pull a 77 MB grid through memory.
DecodedGrid decode_grid(const std::string& path, bool decode_samples = true) {
    install_geotiff_tags();
    TIFF* tif = TIFFOpen(path.c_str(), "r");
    if (tif == nullptr) {
        throw ProjectIOError("'" + path + "': cannot open the geoid grid");
    }
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint16_t bits = 0;
    std::uint16_t samples_per_pixel = 0;
    std::uint16_t sample_format = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);
    TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sample_format);

    DecodedGrid out;
    auto fail = [&](const std::string& why) {
        TIFFClose(tif);
        throw ProjectIOError("'" + path + "': " + why);
    };
    if (width < 2 || height < 2) {
        fail("a geoid grid needs at least 2x2 nodes");
    }
    // The dimensions are attacker-controlled (a crafted header can claim
    // 0xFFFFFFFF per side), and the sample buffer is allocated from them, so
    // they are bounded BEFORE any size arithmetic. The limits are far above the
    // 1440x721 EGM2008 grid: 1e6 nodes per side, 4e8 samples (1.6 GB of float32)
    // in total. Width and height are uint32, so the cap also keeps every
    // conversion to int (GeoidGridInfo::width) in range.
    constexpr std::uint32_t kMaxSide = 1'000'000U;
    constexpr std::uint64_t kMaxSamples = 400'000'000ULL;
    if (width > kMaxSide || height > kMaxSide) {
        fail("a geoid grid may not exceed " + std::to_string(kMaxSide) + " nodes per side");
    }
    if (static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) > kMaxSamples) {
        fail("a geoid grid may not exceed " + std::to_string(kMaxSamples) + " samples");
    }
    if (bits != 32 || sample_format != SAMPLEFORMAT_IEEEFP) {
        fail("a geoid grid must be 32-bit IEEE float");
    }
    if (samples_per_pixel != 1) {
        fail("a geoid grid must have exactly one band");
    }

    double* scale = nullptr;
    std::uint16_t scale_count = 0;
    double* tie = nullptr;
    std::uint16_t tie_count = 0;
    if (TIFFGetField(tif, kTagModelPixelScale, &scale_count, &scale) == 0 || scale == nullptr ||
        scale_count < 2) {
        fail("missing ModelPixelScale (33550)");
    }
    if (TIFFGetField(tif, kTagModelTiepoint, &tie_count, &tie) == 0 || tie == nullptr ||
        tie_count < 6) {
        fail("missing ModelTiepoint (33922)");
    }
    const double pixel_x = scale[0];
    const double pixel_y = scale[1];
    if (!(pixel_x > 0.0) || !(pixel_y > 0.0)) {
        fail("pixel scale must be positive");
    }
    // The tiepoint is the node of sample (0,0) only under PixelIsPoint, which
    // is what the NGA grids declare. A PixelIsArea grid puts the tiepoint on
    // the outer corner of that sample, half a cell away, so the node moves to
    // the sample centre. Without this a grid written by our own GeoTIFF writer
    // would read half a cell off while a PROJ grid read correctly.
    double node_x = tie[3];
    double node_y = tie[4];
    // GDAL_NODATA marks missing cells (the NGA grids declare -32767). Read here
    // because both evaluators honour it: PROJ refuses the value, and the
    // bicubic reader must refuse a stencil that touches one.
    char* nodata_text = nullptr;
    if (TIFFGetField(tif, kTagGdalNodata, &nodata_text) != 0 && nodata_text != nullptr) {
        char* end = nullptr;
        out.nodata = std::strtod(nodata_text, &end);
        out.has_nodata = end != nodata_text && std::isfinite(out.nodata);
    }
    std::uint16_t* keys = nullptr;
    std::uint16_t key_count = 0;
    if (TIFFGetField(tif, kTagGeoKeyDirectory, &key_count, &keys) != 0 && keys != nullptr &&
        key_count >= 4) {
        const std::uint16_t entries = keys[3];
        for (std::uint16_t i = 0; i < entries; ++i) {
            const std::size_t base = 4 + static_cast<std::size_t>(i) * 4;
            if (base + 3 >= key_count) {
                break;
            }
            if (keys[base] != kKeyRasterType || keys[base + 1] != 0) {
                continue;
            }
            if (keys[base + 3] == 1) {  // PixelIsArea
                node_x += 0.5 * pixel_x;
                node_y -= 0.5 * pixel_y;
            }
        }
    }

    if (!decode_samples) {
        TIFFClose(tif);
        out.info.width = static_cast<int>(width);
        out.info.height = static_cast<int>(height);
        out.info.min_lon_deg = node_x;
        out.info.max_lat_deg = node_y;
        out.info.max_lon_deg = node_x + static_cast<double>(width - 1) * pixel_x;
        out.info.min_lat_deg = node_y - static_cast<double>(height - 1) * pixel_y;
        return out;
    }

    out.samples.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    const std::size_t row_values = static_cast<std::size_t>(width);
    auto* base = reinterpret_cast<float*>(out.samples.data());
    if (TIFFIsTiled(tif) != 0) {
        // The NGA grids are tiled and Deflate-compressed; libtiff refuses
        // TIFFReadScanline on a tiled image, so read tile by tile.
        std::uint32_t tile_width = 0;
        std::uint32_t tile_height = 0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tile_width);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tile_height);
        const tmsize_t tile_bytes = TIFFTileSize(tif);
        if (tile_width == 0 || tile_height == 0 || tile_bytes <= 0) {
            fail("tiled geoid grid with an unusable tile size");
        }
        std::vector<float> tile(static_cast<std::size_t>(tile_bytes) / sizeof(float));
        for (std::uint32_t row = 0; row < height; row += tile_height) {
            for (std::uint32_t col = 0; col < width; col += tile_width) {
                if (TIFFReadTile(tif, tile.data(), col, row, 0, 0) < 0) {
                    fail("cannot read a tile at " + std::to_string(col) + "," +
                         std::to_string(row));
                }
                const std::uint32_t copy_w = std::min(tile_width, width - col);
                const std::uint32_t copy_h = std::min(tile_height, height - row);
                for (std::uint32_t ty = 0; ty < copy_h; ++ty) {
                    std::copy_n(tile.data() + static_cast<std::size_t>(ty) *
                                                  static_cast<std::size_t>(tile_width),
                                copy_w,
                                base + static_cast<std::size_t>(row + ty) * row_values + col);
                }
            }
        }
    } else {
        // Scanline at a time; the full EGM2008 grid is 77 MB and read-only
        // here, so there is no reason to ask libtiff for one giant strip.
        for (std::uint32_t row = 0; row < height; ++row) {
            if (TIFFReadScanline(tif, base + static_cast<std::size_t>(row) * row_values, row, 0) <
                0) {
                fail("cannot read scanline " + std::to_string(row));
            }
        }
    }
    TIFFClose(tif);

    out.info.width = static_cast<int>(width);
    out.info.height = static_cast<int>(height);
    out.info.min_lon_deg = node_x;
    out.info.max_lat_deg = node_y;
    out.info.max_lon_deg = node_x + static_cast<double>(width - 1) * pixel_x;
    out.info.min_lat_deg = node_y - static_cast<double>(height - 1) * pixel_y;
    return out;
}

/// A GeoTIFF decode that keeps only the grid geometry.
DecodedGrid decode_grid_header(const std::string& path) {
    return decode_grid(path, /*decode_samples=*/false);
}

}  // namespace

// ---------------------------------------------------------------------------
// GeoidGrid: the decoded GeoTIFF plus its two evaluators.
// ---------------------------------------------------------------------------

struct GeoidGrid::Impl {
    GeoidGridInfo info;
    /// Row-major from the north-west node, row 0 = max_lat.
    ///
    /// Decoded on the first bicubic evaluation, not at open(). The full EGM2008
    /// grid is 77 MB and PROJ reads it through its own memory map, so a grid
    /// opened only to be evaluated through PROJ (the normal path) never pays
    /// for a resident copy. A decode that fails leaves `samples` empty and
    /// records it, so `grid_undulation()` reports "no value" instead of
    /// re-attempting a 77 MB read on every point.
    ///
    /// A GeoidGrid is shared (GeoidStore hands one out per entry) and callers
    /// evaluate from several threads, so the lazy decode is guarded by
    /// call_once: the samples are written exactly once and afterwards only read.
    mutable std::vector<float> samples;
    mutable std::once_flag decode_once;
    mutable bool decode_ok = false;
    /// GDAL_NODATA when the grid declares one; bicubic support that touches a
    /// nodata cell has no value rather than a 32 768 m one.
    mutable double nodata = 0.0;
    mutable bool has_nodata = false;
    std::string path;
    ProjGridShift proj;

    explicit Impl(std::string file);

    /// The samples, decoded on first use. Null when they cannot be read.
    const std::vector<float>* decoded() const {
        std::call_once(decode_once, [this] {
            try {
                DecodedGrid grid = decode_grid(path);
                samples = std::move(grid.samples);
                nodata = grid.nodata;
                has_nodata = grid.has_nodata;
                decode_ok = true;
            } catch (const std::exception&) {
                decode_ok = false;
            }
        });
        return decode_ok ? &samples : nullptr;
    }

    /// True when the sample at (col,row) carries a value: finite, and not the
    /// declared nodata sentinel.
    bool valid(int col, int row) const {
        const float value = samples[static_cast<std::size_t>(row) *
                                        static_cast<std::size_t>(info.width) +
                                    static_cast<std::size_t>(col)];
        // Compare in the stored precision: a sentinel such as -32767 is only
        // meaningful as the float32 that GDAL wrote and the file holds.
        return std::isfinite(value) && !(has_nodata && value == static_cast<float>(nodata));
    }

    double sample(int col, int row) const {
        return static_cast<double>(
            samples[static_cast<std::size_t>(row) * static_cast<std::size_t>(info.width) +
                    static_cast<std::size_t>(col)]);
    }
};

GeoidGrid::GeoidGrid(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

GeoidGrid::Impl::Impl(std::string file) : path(std::move(file)), proj(path) {}

std::shared_ptr<const GeoidGrid> GeoidGrid::open(const std::string& path) {
    auto impl = std::make_shared<Impl>(path);
    impl->info = decode_grid_header(path).info;
    // Node (0,0) is the tiepoint. The NGA grids declare pixel-is-point, so a
    // sample is the undulation AT that node, not a cell average — which is why
    // the bounding box is of nodes and a point exactly on a node interpolates to
    // that node's value.
    return std::shared_ptr<const GeoidGrid>(new GeoidGrid(std::move(impl)));
}

const GeoidGridInfo& GeoidGrid::info() const {
    static const GeoidGridInfo kEmpty;
    return impl_ ? impl_->info : kEmpty;
}

const std::string& GeoidGrid::path() const {
    static const std::string kEmpty;
    return impl_ ? impl_->path : kEmpty;
}

namespace {

/// `lon` moved into [min_lon, min_lon + 360): the grid's own convention.
///
/// A geoid grid may be indexed by either convention (a global grid west-first,
/// -180..180, or one at 0..360), and a caller may hand over either spelling.
/// Comparing raw degrees would then declare a covered point uncovered (lon 190
/// on a -180..180 grid, or lon -170 on a 0..360 grid), so every lookup and
/// coverage test normalises first.
double normalize_lon(double lon, double min_lon) {
    if (!std::isfinite(lon)) {
        return lon;
    }
    double wrapped = std::fmod(lon - min_lon, 360.0);
    if (wrapped < 0.0) {
        wrapped += 360.0;
    }
    return min_lon + wrapped;
}

}  // namespace

bool GeoidGrid::covers(double lon_deg, double lat_deg) const {
    if (!impl_ || !std::isfinite(lat_deg)) {
        return false;
    }
    const GeoidGridInfo& g = impl_->info;
    // A grid spanning 360 degrees is closed in longitude; every other grid is
    // a closed box, so the normalised longitude is compared against its east
    // edge too. Shifting by min_lon makes both conventions work: lon 190 on a
    // -180..180 grid normalises to -170.
    const double lon = normalize_lon(lon_deg, g.min_lon_deg);
    return lon >= g.min_lon_deg && lon <= g.max_lon_deg && lat_deg >= g.min_lat_deg &&
           lat_deg <= g.max_lat_deg;
}

std::optional<double> GeoidGrid::proj_undulation(double lon_deg, double lat_deg) const {
    if (!impl_ || !covers(lon_deg, lat_deg) || !impl_->proj.available()) {
        return std::nullopt;
    }
    const ProjAnswer answer =
        impl_->proj.at(normalize_lon(lon_deg, impl_->info.min_lon_deg), lat_deg);
    if (answer.kind != ProjAnswer::Kind::Value) {
        return std::nullopt;
    }
    return answer.value;
}

std::optional<double> GeoidGrid::grid_undulation(double lon_deg, double lat_deg) const {
    if (!impl_ || !covers(lon_deg, lat_deg) || impl_->decoded() == nullptr) {
        return std::nullopt;
    }
    const GeoidGridInfo& g = impl_->info;
    const double lon = normalize_lon(lon_deg, g.min_lon_deg);
    // Continuous node coordinates: node (col,row) sits at (col,row), so a
    // point at the bounding box's edges lands exactly on the outer nodes and
    // the 4x4 stencil's clamped indices keep it in range there.
    const double x = (lon - g.min_lon_deg) / (g.max_lon_deg - g.min_lon_deg) *
                     static_cast<double>(g.width - 1);
    const double y = (g.max_lat_deg - lat_deg) / (g.max_lat_deg - g.min_lat_deg) *
                     static_cast<double>(g.height - 1);
    const int col = static_cast<int>(std::floor(clamp(x, 0.0, static_cast<double>(g.width - 1))));
    const int row = static_cast<int>(std::floor(clamp(y, 0.0, static_cast<double>(g.height - 1))));
    const double fx = x - static_cast<double>(col);
    const double fy = y - static_cast<double>(row);

    const auto sx = [&](int c) { return c < 0 ? 0 : (c >= g.width ? g.width - 1 : c); };
    const auto sy = [&](int r) { return r < 0 ? 0 : (r >= g.height ? g.height - 1 : r); };

    // Catmull-Rom along x for each of the four rows, then along y. Clamping the
    // stencil indices makes the edge rows/columns a clamped extension: the
    // curve becomes linear there, which is the documented edge fallback.
    double rows[4];
    for (int d = 0; d < 4; ++d) {
        const int rr = sy(row + d - 1);
        // The interpolation is a weighted average with negative lobes, so one
        // nodata or NaN support sample would smear a sentinel into its whole
        // neighbourhood. A nodata cell therefore makes the point valueless.
        const int c0 = sx(col - 1);
        const int c1 = sx(col);
        const int c2 = sx(col + 1);
        const int c3 = sx(col + 2);
        if (!impl_->valid(c0, rr) || !impl_->valid(c1, rr) || !impl_->valid(c2, rr) ||
            !impl_->valid(c3, rr)) {
            return std::nullopt;
        }
        rows[d] = catmull_rom(impl_->sample(c0, rr), impl_->sample(c1, rr), impl_->sample(c2, rr),
                              impl_->sample(c3, rr), fx);
    }
    if (!std::isfinite(rows[0]) || !std::isfinite(rows[1]) || !std::isfinite(rows[2]) ||
        !std::isfinite(rows[3])) {
        return std::nullopt;
    }
    const double value = catmull_rom(rows[0], rows[1], rows[2], rows[3], fy);
    return std::isfinite(value) ? std::optional<double>(value) : std::nullopt;
}

std::optional<double> GeoidGrid::undulation(double lon_deg, double lat_deg) const {
    if (!impl_) {
        return std::nullopt;
    }
    if (impl_->proj.available()) {
        // PROJ is the evaluator whenever it can be asked at all, so its answer
        // stands — including "no value". A point PROJ refused (outside the
        // grid's own coverage, or on a nodata cell) must not be answered by the
        // GeoTIFF reader: that would replace a refusal with an interpolated
        // value. Only a grid PROJ cannot even open falls through.
        if (!covers(lon_deg, lat_deg)) {
            return std::nullopt;
        }
        const ProjAnswer answer =
            impl_->proj.at(normalize_lon(lon_deg, impl_->info.min_lon_deg), lat_deg);
        if (answer.kind == ProjAnswer::Kind::Value) {
            return answer.value;
        }
        if (answer.kind == ProjAnswer::Kind::Refused) {
            return std::nullopt;
        }
        // Unavailable: the context or the pipeline could not be built on this
        // thread. That is "PROJ is unavailable for this grid", so the reader
        // answers below.
    }
    return grid_undulation(lon_deg, lat_deg);
}

// ---------------------------------------------------------------------------
// GeoidStore. The index is a tiny, self-describing XML file next to the grids:
//
//   <geoids>
//     <geoid id="EGM96" name="EGM96 height" file="EGM96.tif"
//            source="proj-data" licence="public domain (NGA)"
//            min_lon="-180" min_lat="-90" max_lon="180" max_lat="90"/>
//   </geoids>
//
// It is read and written here rather than through common/io's XmlNode because
// that lives in alphashape_core, which already depends on this library. The
// grammar is a closed set of attributes on one element kind, so a scanner is
// enough and there is no second XML parser to keep in step.
// ---------------------------------------------------------------------------

namespace {

std::string xml_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (const char ch : in) {
        switch (ch) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            default: out.push_back(ch); break;
        }
    }
    return out;
}

std::string xml_unescape(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        if (in[i] != '&') {
            out.push_back(in[i]);
            continue;
        }
        const std::size_t semi = in.find(';', i);
        if (semi == std::string::npos) {
            out.push_back(in[i]);
            continue;
        }
        const std::string entity = in.substr(i + 1, semi - i - 1);
        if (entity == "amp") {
            out.push_back('&');
        } else if (entity == "lt") {
            out.push_back('<');
        } else if (entity == "gt") {
            out.push_back('>');
        } else if (entity == "quot") {
            out.push_back('"');
        } else if (entity == "apos") {
            out.push_back('\'');
        } else {
            out.append(in, i, semi - i + 1);
        }
        i = semi;
    }
    return out;
}

/// One `<geoid .../>` element's attributes. Throws ProjectIOError with the file
/// name when the text is not that grammar, or when a required attribute is
/// absent.
std::vector<std::pair<std::string, std::string>> parse_element(const std::string& element,
                                                               const std::filesystem::path& file) {
    std::vector<std::pair<std::string, std::string>> attrs;
    std::size_t i = 0;
    const auto skip_space = [&]() {
        while (i < element.size() && (element[i] == ' ' || element[i] == '\t' ||
                                      element[i] == '\n' || element[i] == '\r')) {
            ++i;
        }
    };
    skip_space();
    while (i < element.size()) {
        const std::size_t name_start = i;
        while (i < element.size() && element[i] != '=' && element[i] != '/' &&
               element[i] != '>') {
            ++i;
        }
        const std::string name = element.substr(name_start, i - name_start);
        skip_space();
        // "/>" and ">" end the attribute list; a stray "/" here is that marker,
        // not the start of another attribute.
        if (i >= element.size() || element[i] == '/' || element[i] == '>') {
            break;
        }
        if (element[i] != '=') {
            throw ProjectIOError("'" + file.string() + "': <geoid> attribute '" + name +
                                 "' has no value");
        }
        ++i;
        skip_space();
        if (i >= element.size() || element[i] != '"') {
            throw ProjectIOError("'" + file.string() + "': <geoid> attribute '" + name +
                                 "' is not double-quoted");
        }
        ++i;
        const std::size_t value_start = i;
        while (i < element.size() && element[i] != '"') {
            ++i;
        }
        if (i >= element.size()) {
            throw ProjectIOError("'" + file.string() + "': unterminated value for attribute '" +
                                 name + "'");
        }
        attrs.emplace_back(name, xml_unescape(element.substr(value_start, i - value_start)));
        ++i;
        skip_space();
    }
    return attrs;
}

std::string attr_of(const std::vector<std::pair<std::string, std::string>>& attrs,
                    const std::string& name, const std::filesystem::path& file, bool required) {
    for (const auto& [key, value] : attrs) {
        if (key == name) {
            return value;
        }
    }
    if (required) {
        throw ProjectIOError("'" + file.string() + "': <geoid> is missing attribute '" + name +
                             "'");
    }
    return std::string();
}

double number_of(const std::vector<std::pair<std::string, std::string>>& attrs,
                 const std::string& name, const std::filesystem::path& file) {
    const std::string text = attr_of(attrs, name, file, true);
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0' || !std::isfinite(value)) {
        throw ProjectIOError("'" + file.string() + "': attribute '" + name + "' is not a number");
    }
    return value;
}

/// Why `value` is not a single plain path component, or nullopt when it is.
///
/// Both the `file` attribute of an `index.xml` and the `id` a grid is stored
/// under end up as `dir / name`, and the index is untrusted input: a project
/// may be opened from anywhere. Anything that could escape `dir` or address a
/// different volume is rejected, so the join can never leave the store.
std::optional<std::string> path_component_error(const std::string& value) {
    if (value.empty()) {
        return "is empty";
    }
    if (value.size() > 128) {
        return "is longer than 128 characters";
    }
    if (value == "." || value == "..") {
        return "is a relative directory reference";
    }
    for (const char ch : value) {
        const bool plain = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
                           (ch >= '0' && ch <= '9') || ch == '.' || ch == '_' || ch == '-';
        if (!plain) {
            // Covers '/', '\', ':' (drive letters, ADS), NUL and everything
            // else; the message names the rule rather than echoing bytes.
            return "must match [A-Za-z0-9._-]+";
        }
    }
    return std::nullopt;
}

/// True when `candidate` is `base` or sits below it, component by component.
bool is_within(const std::filesystem::path& candidate, const std::filesystem::path& base) {
    auto b = base.begin();
    auto c = candidate.begin();
    for (; b != base.end(); ++b, ++c) {
        if (c == candidate.end() || *c != *b) {
            return false;
        }
    }
    return true;
}

/// `dir / name` after proving that the result resolves inside `dir`.
///
/// The component check above already excludes every escape we know of; this is
/// the belt-and-braces check the join itself needs, so a platform quirk (a
/// symlinked directory, a translated device path) cannot turn a plain name into
/// a path outside the store.
std::filesystem::path join_inside(const std::filesystem::path& dir, const std::string& name,
                                  const std::string& what, const std::string& where) {
    const std::filesystem::path joined = dir / name;
    std::error_code ec;
    const std::filesystem::path base = std::filesystem::weakly_canonical(dir, ec);
    if (ec) {
        throw Error("'" + where + "': cannot resolve the geoid store directory '" + dir.string() +
                    "': " + ec.message());
    }
    const std::filesystem::path full = std::filesystem::weakly_canonical(joined, ec);
    if (ec || !is_within(full, base)) {
        throw Error("'" + where + "': geoid " + what + " '" + name +
                    "' resolves outside the geoid store directory");
    }
    return joined;
}

}  // namespace

GeoidStore::GeoidStore() = default;

namespace {

/// Id of a built-in geoid: the PROJ grid's stem with the authority prefix
/// stripped, e.g. "us_nga_egm96_15" -> "EGM96", "us_nga_egm08_25" -> "EGM2008".
/// The ids are the ones M12 §4.5 names in its API sketch.
std::string proj_data_id(const std::string& grid_name) {
    if (grid_name == "us_nga_egm96_15.tif") {
        return "EGM96";
    }
    if (grid_name == "us_nga_egm08_25.tif") {
        return "EGM2008";
    }
    return std::filesystem::path(grid_name).stem().string();
}

const char* const kIndexFile = "index.xml";

}  // namespace

namespace {

/// Geometry from a grid's GeoTIFF header without decoding its samples; nullopt
/// when the file is not a readable grid. `GeoidGrid::open()` is the full read;
/// this is for `save()`/`add()`, which must not pull 77 MB through memory to
/// learn a bounding box.
std::optional<GeoidGridInfo> peek_grid_info(const std::string& path) {
    try {
        return decode_grid_header(path).info;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

}  // namespace

GeoidStore GeoidStore::from_proj_data() {
    GeoidStore store;
    struct Known {
        const char* file;
        const char* name;
    };
    static constexpr Known kKnown[] = {
        {"us_nga_egm96_15.tif", "EGM96 height"},
        {"us_nga_egm08_25.tif", "EGM2008 height"},
    };
    for (const Known& known : kKnown) {
        const std::filesystem::path path = find_data_file(known.file);
        if (path.empty()) {
            continue;  // not installed; nothing is downloaded
        }
        GeoidStore::Entry entry;
        entry.model.id = proj_data_id(known.file);
        entry.model.name = known.name;
        entry.model.path = path.string();
        entry.model.source = "proj-data";
        entry.model.licence = "public domain (NGA)";
        // A grid whose header cannot be read is still listed: the coverage
        // stays the default world box and `undulation()` will report it as
        // uncovered rather than the store silently dropping the model.
        if (const std::optional<GeoidGridInfo> info = peek_grid_info(path.string())) {
            entry.model.min_lon_deg = info->min_lon_deg;
            entry.model.min_lat_deg = info->min_lat_deg;
            entry.model.max_lon_deg = info->max_lon_deg;
            entry.model.max_lat_deg = info->max_lat_deg;
        }
        store.entries_.push_back(std::move(entry));
    }
    return store;
}

bool GeoidStore::add(const std::string& id, const std::string& name,
                     const std::string& grid_path, const std::string& source,
                     const std::string& licence) {
    if (id.empty()) {
        throw Error("a geoid needs an id");
    }
    // The id names the stored file (`id + ".tif"`), so it must be one plain
    // component: `add("../x", ...)` must not write outside the store.
    if (const std::optional<std::string> why = path_component_error(id)) {
        throw Error("geoid id '" + id + "' " + *why);
    }
    if (find(id)) {
        throw Error("geoid '" + id + "' is already in the store");
    }
    const std::filesystem::path source_path(grid_path);
    if (!std::filesystem::exists(source_path)) {
        throw Error("geoid grid '" + grid_path + "' does not exist");
    }
    // A store with a directory copies the grid in, so the project is
    // self-contained; a store without one (the in-memory ProjData case) just
    // references the file where it is.
    std::string stored = grid_path;
    if (!directory_.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(directory_, ec);
        if (ec) {
            throw Error("cannot create the geoid store directory '" + directory_.string() +
                        "': " + ec.message());
        }
        const std::filesystem::path target =
            join_inside(directory_, id + ".tif", "id", "GeoidStore::add");
        if (std::filesystem::weakly_canonical(source_path, ec) !=
            std::filesystem::weakly_canonical(target, ec)) {
            std::filesystem::copy_file(source_path, target,
                                       std::filesystem::copy_options::overwrite_existing, ec);
            if (ec) {
                throw Error("cannot copy geoid grid '" + grid_path + "' into '" +
                            target.string() + "': " + ec.message());
            }
        }
        stored = target.string();
    }
    Entry entry;
    entry.model.id = id;
    entry.model.name = name.empty() ? id : name;
    entry.model.path = stored;
    entry.model.source = source.empty() ? "custom" : source;
    entry.model.licence = licence;
    if (const std::optional<GeoidGridInfo> info = peek_grid_info(stored)) {
        entry.model.min_lon_deg = info->min_lon_deg;
        entry.model.min_lat_deg = info->min_lat_deg;
        entry.model.max_lon_deg = info->max_lon_deg;
        entry.model.max_lat_deg = info->max_lat_deg;
    }
    entries_.push_back(std::move(entry));
    return true;
}

std::optional<GeoidModel> GeoidStore::find(const std::string& id) const {
    for (const Entry& entry : entries_) {
        if (entry.model.id == id) {
            return entry.model;
        }
    }
    return std::nullopt;
}

std::vector<GeoidModel> GeoidStore::list() const {
    std::vector<GeoidModel> models;
    models.reserve(entries_.size());
    for (const Entry& entry : entries_) {
        models.push_back(entry.model);
    }
    return models;
}

const std::filesystem::path& GeoidStore::directory() const { return directory_; }

void GeoidStore::set_directory(std::filesystem::path dir) { directory_ = std::move(dir); }

std::shared_ptr<const GeoidGrid> GeoidStore::grid_for(const Entry& entry) const {
    if (!entry.grid) {
        try {
            entry.grid = GeoidGrid::open(entry.model.path);
        } catch (const std::exception&) {
            // Reported as "not covered" by the callers: a grid that cannot be
            // opened has no undulation, and load() is where a *missing* file is
            // an error.
            return nullptr;
        }
    }
    return entry.grid;
}

std::shared_ptr<const GeoidGrid> GeoidStore::grid(const std::string& id) const {
    for (const Entry& entry : entries_) {
        if (entry.model.id == id) {
            return grid_for(entry);
        }
    }
    return nullptr;
}

bool GeoidStore::covers(const std::string& id, double lon_deg, double lat_deg) const {
    for (const Entry& entry : entries_) {
        if (entry.model.id != id) {
            continue;
        }
        if (const std::shared_ptr<const GeoidGrid> grid = grid_for(entry)) {
            return grid->covers(lon_deg, lat_deg);
        }
        return lon_deg >= entry.model.min_lon_deg && lon_deg <= entry.model.max_lon_deg &&
               lat_deg >= entry.model.min_lat_deg && lat_deg <= entry.model.max_lat_deg;
    }
    return false;
}

std::optional<double> GeoidStore::undulation(const std::string& id, double lon_deg,
                                            double lat_deg) const {
    for (const Entry& entry : entries_) {
        if (entry.model.id != id) {
            continue;
        }
        const std::shared_ptr<const GeoidGrid> grid = grid_for(entry);
        if (!grid) {
            return std::nullopt;
        }
        return grid->undulation(lon_deg, lat_deg);
    }
    return std::nullopt;
}

void GeoidStore::save(const std::filesystem::path& dir) const {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        throw Error("cannot create the geoid store directory '" + dir.string() + "': " +
                    ec.message());
    }
    std::string xml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<geoids>\n";
    for (const Entry& entry : entries_) {
        const std::filesystem::path source(entry.model.path);
        // The store is copied next to the index under the model's id, so the
        // project keeps working when the machine that made it is gone.
        const std::string file_name = entry.model.id + ".tif";
        // The id was validated when it entered the store (add()/load()), and is
        // validated again here because a store can be populated by other paths.
        if (const std::optional<std::string> why = path_component_error(entry.model.id)) {
            throw Error("geoid id '" + entry.model.id + "' " + *why +
                        "; refusing to write grid '" + file_name + "'");
        }
        const std::filesystem::path target =
            join_inside(dir, file_name, "id", "GeoidStore::save");
        if (!std::filesystem::exists(source)) {
            throw Error("geoid '" + entry.model.id + "' is missing its grid '" +
                        entry.model.path + "'");
        }
        if (std::filesystem::weakly_canonical(source, ec) !=
            std::filesystem::weakly_canonical(target, ec)) {
            std::filesystem::copy_file(source, target,
                                       std::filesystem::copy_options::overwrite_existing, ec);
            if (ec) {
                throw Error("cannot copy geoid grid '" + entry.model.path + "' into '" +
                            target.string() + "': " + ec.message());
            }
        }
        const GeoidModel& m = entry.model;
        auto num = [](double v) {
            // 12 significant digits: enough for a 0.25-degree node to be exact,
            // short enough to stay readable.
            char buffer[64];
            std::snprintf(buffer, sizeof(buffer), "%.12g", v);
            return std::string(buffer);
        };
        xml += "  <geoid id=\"" + xml_escape(m.id) + "\" name=\"" + xml_escape(m.name) +
               "\" file=\"" + xml_escape(file_name) + "\" source=\"" + xml_escape(m.source) +
               "\" licence=\"" + xml_escape(m.licence) + "\" min_lon=\"" + num(m.min_lon_deg) +
               "\" min_lat=\"" + num(m.min_lat_deg) + "\" max_lon=\"" + num(m.max_lon_deg) +
               "\" max_lat=\"" + num(m.max_lat_deg) + "\"/>\n";
    }
    xml += "</geoids>\n";
    const std::filesystem::path index = dir / kIndexFile;
    const std::filesystem::path temp = dir / (std::string(kIndexFile) + ".tmp");
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw Error("cannot write '" + temp.string() + "'");
        }
        out << xml;
        if (!out.flush()) {
            throw Error("cannot write '" + temp.string() + "'");
        }
    }
    std::filesystem::rename(temp, index, ec);
    if (ec) {
        throw Error("cannot publish '" + index.string() + "': " + ec.message());
    }
}

GeoidStore GeoidStore::load(const std::filesystem::path& dir) {
    const std::filesystem::path index = dir / kIndexFile;
    std::ifstream in(index, std::ios::binary);
    if (!in) {
        throw Error("'" + index.string() + "': a geoid store needs its index");
    }
    const std::string text((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    GeoidStore store;
    store.directory_ = dir;
    // The index is our own single-element grammar: find each <geoid .../> and
    // read its attributes. Anything else in the file is not a geoid entry.
    std::size_t pos = 0;
    while ((pos = text.find("<geoid ", pos)) != std::string::npos) {
        const std::size_t end = text.find('>', pos);
        if (end == std::string::npos) {
            throw Error("'" + index.string() + "': unterminated <geoid> element");
        }
        const auto attrs = parse_element(text.substr(pos + 7, end - pos - 7), index);
        Entry entry;
        entry.model.id = attr_of(attrs, "id", index, true);
        // The id becomes the stored file name on the next save(), so it is held
        // to the same rule as `file` here rather than only when it is written.
        if (const std::optional<std::string> why = path_component_error(entry.model.id)) {
            throw ProjectIOError("'" + index.string() + "': geoid id '" + entry.model.id + "' " +
                                 *why);
        }
        entry.model.name = attr_of(attrs, "name", index, false);
        const std::string file = attr_of(attrs, "file", index, true);
        entry.model.source = attr_of(attrs, "source", index, false);
        entry.model.licence = attr_of(attrs, "licence", index, false);
        entry.model.min_lon_deg = number_of(attrs, "min_lon", index);
        entry.model.min_lat_deg = number_of(attrs, "min_lat", index);
        entry.model.max_lon_deg = number_of(attrs, "max_lon", index);
        entry.model.max_lat_deg = number_of(attrs, "max_lat", index);
        // The index is untrusted: `file` is joined onto `dir`, so it must be one
        // plain component and the result must stay inside the store.
        if (const std::optional<std::string> why = path_component_error(file)) {
            throw ProjectIOError("'" + index.string() + "': geoid '" + entry.model.id +
                                 "' has an unusable file name: '" + file + "' " + *why);
        }
        const std::filesystem::path grid = join_inside(dir, file, "file", index.string());
        if (!std::filesystem::exists(grid)) {
            // A project that names a geoid it does not carry must not load as if
            // the heights were ellipsoidal: that would shift every orthometric
            // height by tens of metres, silently.
            throw Error("geoid '" + entry.model.id + "' is not available: '" + grid.string() +
                        "' is missing; add it with CoordinateSystem.addGeoid");
        }
        entry.model.path = grid.string();
        entry.grid = GeoidGrid::open(grid.string());
        store.entries_.push_back(std::move(entry));
        pos = end;
    }
    return store;
}

}  // namespace alphashape::crs
