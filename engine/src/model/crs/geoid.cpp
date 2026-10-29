#include "model/crs/geoid.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <proj.h>

#include <tiffio.h>

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

/// A grid opened on a private PROJ context whose grid search path is the
/// grid's directory. PROJ resolves `+grids=` by name only, and the thread
/// context must not be mutated (it is shared with every other CRS operation on
/// that thread), so the path is provided here and the bare file name is passed.
///
/// Both the horizontal extent check and the evaluation go through this object;
/// `proj_undulation` returns nullopt as soon as PROJ reports the grid outside
/// its coverage, so a caller never sees an extrapolated value.
class ProjGridShift {
public:
    explicit ProjGridShift(const std::string& grid_path) {
        const std::filesystem::path path(grid_path);
        // PROJ splits +grids= on whitespace and has no quoting rule, so a file
        // name containing a space cannot be handed over at all. That is not an
        // error: proj_undulation() stays unavailable and the GeoTIFF reader
        // answers instead.
        if (path.filename().string().find_first_of(" \t") != std::string::npos) {
            return;
        }
        ctx_.reset(proj_context_create());
        if (!ctx_) {
            return;
        }
        // No downloads, no shared grid cache: the file is already on disk and is
        // the only grid this context will ever look for.
        proj_context_set_enable_network(ctx_.get(), 0);
        proj_grid_cache_set_enable(ctx_.get(), 0);
        const std::string dir = path.parent_path().empty() ? std::string(".")
                                                           : path.parent_path().string();
        const char* paths[2] = {dir.c_str(), nullptr};
        proj_context_set_search_paths(ctx_.get(), 1, paths);

        const std::string pipeline = "+proj=vgridshift +grids=" + path.filename().string() +
                                     " +multiplier=1";
        pj_.reset(proj_create(ctx_.get(), pipeline.c_str()));
        usable_ = pj_ != nullptr;
    }

    bool usable() const { return usable_; }

    /// N in metres at (lon, lat), with the horizontal axes switched into PROJ's
    /// pipeline order (lon, lat) and the vertical input at 0.
    std::optional<double> at(double lon_deg, double lat_deg) const {
        if (!usable_) {
            return std::nullopt;
        }
        proj_errno_reset(pj_.get());
        const PJ_COORD in = proj_coord(lon_deg, lat_deg, 0.0, 0.0);
        const PJ_COORD out = proj_trans(pj_.get(), PJ_FWD, in);
        if (proj_errno(pj_.get()) != 0 || !std::isfinite(out.xyz.z)) {
            return std::nullopt;
        }
        return out.xyz.z;
    }

private:
    ContextPtr ctx_;
    PjPtr pj_;  ///< the vgridshift step, not a pipeline: input is lon/lat degrees
    bool usable_ = false;
};

// ---------------------------------------------------------------------------
// The GeoTIFF carrier. A geoid grid is the NGA layout: one band, 32-bit IEEE
// float, ModelPixelScale (33550) and ModelTiepoint (33922) describing a north-up
// grid. This is decoded here rather than through model/raster/geotiff.cpp
// because that library depends on this one, so calling it back would be a
// cycle; the subset needed is a header read and one copy loop.
// ---------------------------------------------------------------------------

constexpr std::uint32_t kTagPixelScale = 33550;
constexpr std::uint32_t kTagTiepoint = 33922;
constexpr std::uint32_t kTagGeoKeyDirectory = 34735;
/// GeoKey 1025, GTRasterTypeGeoKey: 1 = PixelIsArea, 2 = PixelIsPoint.
constexpr std::uint16_t kKeyRasterType = 1025;

/// The extender libtiff had installed before ours, so the hook can chain.
TIFFExtendProc g_parent_tag_extender = nullptr;

void merge_geotiff_fields(TIFF* tif) {
    // Field names must have static storage: they outlive every handle.
    static char kPixelScale[] = "ModelPixelScaleTag";
    static char kTiepoint[] = "ModelTiepointTag";
    static char kGeoKeys[] = "GeoKeyDirectoryTag";
    static const TIFFFieldInfo fields[] = {
        {kTagPixelScale, TIFF_VARIABLE, TIFF_VARIABLE, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1,
         kPixelScale},
        {kTagTiepoint, TIFF_VARIABLE, TIFF_VARIABLE, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1, kTiepoint},
        {kTagGeoKeyDirectory, TIFF_VARIABLE, TIFF_VARIABLE, TIFF_SHORT, FIELD_CUSTOM, 1, 1,
         kGeoKeys},
    };
    TIFFMergeFieldInfo(tif, fields, static_cast<std::uint32_t>(std::size(fields)));
}

/// libtiff has no process-global slot for custom fields: they are per handle,
/// and this extender is the only supported way to add them to every file this
/// process opens. A one-shot TIFFMergeFieldInfo would work for the first grid
/// and silently lose the tags for the second.
void geotiff_tag_extender(TIFF* tif) {
    merge_geotiff_fields(tif);
    if (g_parent_tag_extender != nullptr) {
        g_parent_tag_extender(tif);
    }
}

void install_geotiff_tags() {
    // A plain function-local static: the initialiser runs once, guarded by the
    // compiler's own thread-safe initialisation, so no <mutex> is needed here.
    static const bool installed = [] {
        g_parent_tag_extender = TIFFSetTagExtender(geotiff_tag_extender);
        return true;
    }();
    static_cast<void>(installed);
}

struct DecodedGrid {
    GeoidGridInfo info;
    std::vector<float> samples;
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
    if (TIFFGetField(tif, kTagPixelScale, &scale_count, &scale) == 0 || scale == nullptr ||
        scale_count < 2) {
        fail("missing ModelPixelScale (33550)");
    }
    if (TIFFGetField(tif, kTagTiepoint, &tie_count, &tie) == 0 || tie == nullptr ||
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
    // Scanline at a time: a full EGM2008 grid is 77 MB and read-only here, so
    // there is no reason to ask libtiff for one giant strip.
    for (std::uint32_t row = 0; row < height; ++row) {
        auto* dst = reinterpret_cast<float*>(out.samples.data() +
                                             static_cast<std::size_t>(row) *
                                                 static_cast<std::size_t>(width));
        if (TIFFReadScanline(tif, dst, row, 0) < 0) {
            fail("cannot read scanline " + std::to_string(row));
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

/// A GeoTIFF decode that keeps only the grid geometry. Used by add()/save(),
/// which must not pull a 77 MB grid through memory to learn a bounding box.
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
    std::vector<float> samples;
    std::string path;
    ProjGridShift proj;

    explicit Impl(std::string file);

    double sample(int col, int row) const {
        return static_cast<double>(
            samples[static_cast<std::size_t>(row) * static_cast<std::size_t>(info.width) +
                    static_cast<std::size_t>(col)]);
    }
};

GeoidGrid::GeoidGrid(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

GeoidGrid::Impl::Impl(std::string file) : path(std::move(file)), proj(path) {}

std::shared_ptr<const GeoidGrid> GeoidGrid::open(const std::string& path) {
    DecodedGrid decoded = decode_grid(path);
    auto impl = std::make_shared<Impl>(path);
    impl->info = decoded.info;
    impl->samples = std::move(decoded.samples);
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

bool GeoidGrid::covers(double lon_deg, double lat_deg) const {
    if (!impl_) {
        return false;
    }
    const GeoidGridInfo& g = impl_->info;
    return lon_deg >= g.min_lon_deg && lon_deg <= g.max_lon_deg && lat_deg >= g.min_lat_deg &&
           lat_deg <= g.max_lat_deg;
}

std::optional<double> GeoidGrid::proj_undulation(double lon_deg, double lat_deg) const {
    if (!impl_ || !covers(lon_deg, lat_deg)) {
        return std::nullopt;
    }
    return impl_->proj.at(lon_deg, lat_deg);
}

std::optional<double> GeoidGrid::grid_undulation(double lon_deg, double lat_deg) const {
    if (!impl_ || !covers(lon_deg, lat_deg)) {
        return std::nullopt;
    }
    const GeoidGridInfo& g = impl_->info;
    // Continuous node coordinates: node (col,row) sits at (col,row), so a
    // point at the bounding box's edges lands exactly on the outer nodes and
    // the 4x4 stencil's clamped indices keep it in range there.
    const double x = (lon_deg - g.min_lon_deg) / (g.max_lon_deg - g.min_lon_deg) *
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
        rows[d] = catmull_rom(impl_->sample(sx(col - 1), rr), impl_->sample(sx(col), rr),
                              impl_->sample(sx(col + 1), rr), impl_->sample(sx(col + 2), rr), fx);
    }
    return catmull_rom(rows[0], rows[1], rows[2], rows[3], fy);
}

std::optional<double> GeoidGrid::undulation(double lon_deg, double lat_deg) const {
    if (const auto via_proj = proj_undulation(lon_deg, lat_deg)) {
        return via_proj;
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
        if (i >= element.size() || element[i] != '=') {
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
        const std::filesystem::path target = directory_ / (id + ".tif");
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
        const std::filesystem::path target = dir / file_name;
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
        entry.model.name = attr_of(attrs, "name", index, false);
        const std::string file = attr_of(attrs, "file", index, true);
        entry.model.source = attr_of(attrs, "source", index, false);
        entry.model.licence = attr_of(attrs, "licence", index, false);
        entry.model.min_lon_deg = number_of(attrs, "min_lon", index);
        entry.model.min_lat_deg = number_of(attrs, "min_lat", index);
        entry.model.max_lon_deg = number_of(attrs, "max_lon", index);
        entry.model.max_lat_deg = number_of(attrs, "max_lat", index);
        const std::filesystem::path grid = dir / file;
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
