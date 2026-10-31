#include "common/algorithm/ortho/build_orthomosaic_task.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "base/platform/memory.hpp"
#include "common/algorithm/elevation/dem_builder.hpp"
#include "common/algorithm/elevation/interpolate.hpp"
#include "model/crs/coordinate_system.hpp"
#include "model/raster/reproject.hpp"
#include "common/algorithm/ortho/blend.hpp"
#include "common/algorithm/ortho/colour_model.hpp"
#include "common/algorithm/ortho/gain.hpp"
#include "common/algorithm/ortho/orthorectify.hpp"
#include "common/algorithm/ortho/seamlines.hpp"
#include "common/core/assets.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "image/io/image_io.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

/// The tolerance (metres) the DEM must stand above the marching ray before the
/// cell counts as occluded. A quarter DEM cell is the scale of the interpolant's
/// own error over one cell, so it does not hide genuine walls.
constexpr double kOcclusionToleranceCells = 0.25;

/// Round to three significant figures, matching BuildDem's automatic-resolution
/// contract (0.101 -> 0.101, 0.12334 -> 0.123).
double round_sig3(double v) {
    if (!(v > 0.0) || !std::isfinite(v)) {
        return v;
    }
    const double magnitude = std::floor(std::log10(v));
    const double scale = std::pow(10.0, 2.0 - magnitude);
    return std::round(v * scale) / scale;
}

/// Round UP to three significant figures (0.12334 -> 0.124). Used when a value
/// is raised to bring an estimate DOWN: rounding to nearest could land back on
/// the far side of the target and make the caller loop.
double ceil_sig3(double v) {
    if (!(v > 0.0) || !std::isfinite(v)) {
        return v;
    }
    const double magnitude = std::floor(std::log10(v));
    const double scale = std::pow(10.0, 2.0 - magnitude);
    // The epsilon absorbs the float round trip in v * scale: an exact 17 that
    // lands on 17.000000000000004 must not become 18.
    return std::ceil(v * scale - 1e-9 * std::abs(v * scale)) / scale;
}

/// A short decimal for a message ("0.019", "12.3"), never scientific notation.
std::string format_number(double v) {
    std::ostringstream out;
    out.precision(3);
    out << v;
    return out.str();
}

double median_of(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid),
                     values.end());
    double median = values[mid];
    if (values.size() % 2 == 0) {
        const double lo = *std::max_element(
            values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid));
        median = 0.5 * (lo + median);
    }
    return median;
}

/// Bilinear, nodata-aware height sample on a raster grid (the same rule the
/// orthorectifier uses); returns `nodata` when nothing valid contributes or the
/// point is off the grid.
double sample_grid(const GeoRaster<float>& raster, float nodata, double x, double y) {
    const RasterGrid& grid = raster.grid;
    const Eigen::Vector2d c = grid.to_pixel(x, y);
    const double cf = c.x() - 0.5;
    const double rf = c.y() - 0.5;
    const int c0 = static_cast<int>(std::floor(cf));
    const int r0 = static_cast<int>(std::floor(rf));
    const double fx = cf - static_cast<double>(c0);
    const double fy = rf - static_cast<double>(r0);
    double sum = 0.0;
    double weight = 0.0;
    for (int dr = 0; dr <= 1; ++dr) {
        for (int dc = 0; dc <= 1; ++dc) {
            const int col = c0 + dc;
            const int row = r0 + dr;
            if (!grid.contains_pixel(col, row)) {
                continue;
            }
            const float h = raster.at(col, row);
            if (h == nodata || std::isnan(h)) {
                continue;
            }
            const double w = (dc == 0 ? 1.0 - fx : fx) * (dr == 0 ? 1.0 - fy : fy);
            sum += w * static_cast<double>(h);
            weight += w;
        }
    }
    if (weight <= 0.0) {
        return static_cast<double>(nodata);
    }
    return sum / weight;
}

/// Median of the valid heights on `raster`.
double median_height(const GeoRaster<float>& raster, float nodata) {
    std::vector<double> heights;
    heights.reserve(raster.data.size());
    for (const float h : raster.data) {
        if (h != nodata && std::isfinite(h)) {
            heights.push_back(static_cast<double>(h));
        }
    }
    return median_of(std::move(heights));
}

/// The largest cell count an output grid may have. RasterGrid holds int
/// dimensions, so a single side above this (or a product above it) would
/// overflow the cast, silently producing a negative size that later code sizes
/// buffers and loops with. A resolution that asks for more cells than this is a
/// mistake by orders of magnitude, not a run anyone wants.
constexpr double kMaxGridCells = 2147483647.0;  // INT_MAX

/// A compact human-readable cell count: "1e+16", not "10000000000000000".
std::string format_cells(double cells) {
    std::ostringstream out;
    out.precision(3);
    out << cells;
    return out.str();
}

/// The north-up grid covering `extent` at `resolution`, its origin snapped
/// outward to a multiple of the resolution (row 0 northernmost).
///
/// The cell counts are derived in double and validated BEFORE the cast to int:
/// at a micrometre resolution over a 100 m extent the quotient is 1e8 per side
/// and 1e16 cells, and casting an unbounded double is undefined behaviour that
/// sails straight into a negative RasterGrid size. Anything above kMaxGridCells
/// is refused here, with the same actionable wording the memory guard uses.
RasterGrid ortho_grid_for(const RasterGrid& extent, double resolution) {
    if (!(resolution > 0.0) || !std::isfinite(resolution)) {
        throw ValueError("BuildOrthomosaic: resolution must be a positive finite number "
                         "(got " +
                         std::to_string(resolution) + " m)");
    }
    const double east =
        extent.origin_x + static_cast<double>(extent.width) * extent.pixel_size_x;
    const double south =
        extent.origin_y - static_cast<double>(extent.height) * extent.pixel_size_y;
    const double origin_x = std::floor(extent.origin_x / resolution) * resolution;
    const double origin_y = std::ceil(extent.origin_y / resolution) * resolution;
    const double span_x = east - origin_x;
    const double span_y = origin_y - south;
    const double cols = std::ceil(span_x / resolution);
    const double rows = std::ceil(span_y / resolution);
    const double cells = cols * rows;

    // NaN fails every comparison, so each quotient is checked for finiteness
    // explicitly: a non-finite extent must not reach the cast either.
    if (!std::isfinite(span_x) || !std::isfinite(span_y) || !std::isfinite(cells) ||
        !(cells <= kMaxGridCells) || !(cols <= kMaxGridCells) || !(rows <= kMaxGridCells)) {
        throw ValueError(
            "BuildOrthomosaic: a resolution of " + std::to_string(resolution) + " m over a " +
            format_cells(span_x) + " x " + format_cells(span_y) + " m extent would need " +
            format_cells(cells) + " cells (the limit is " + format_cells(kMaxGridCells) +
            "); choose a coarser resolution.");
    }

    RasterGrid grid;
    grid.pixel_size_x = resolution;
    grid.pixel_size_y = resolution;
    grid.origin_x = origin_x;
    grid.origin_y = origin_y;
    grid.width = std::max(1, static_cast<int>(cols));
    grid.height = std::max(1, static_cast<int>(rows));
    return grid;
}

/// A plane -> local map sampled on the grid's corner nodes and evaluated by
/// bilinear interpolation, with the vertical carried exactly.
///
/// The projection inverse is a geodesy round trip per point (tens of
/// microseconds); the orthorectifier calls it for every valid cell of every
/// camera. The map is near-rigid (UTM scale, convergence, curvature), so its
/// horizontal part varies almost linearly over a DEM cell and bilinear
/// interpolation of the node values is accurate far below a millimetre at cell
/// centres. The height component is not interpolated: the node records
/// d(local z)/dh, and the query's own height (the DEM sample the orthorectifier
/// just read) is applied exactly, so no vertical error is introduced at all.
///
/// MEMORY. The node count is the grid's corner lattice, i.e. about one node per
/// output cell: at 28 M cells a Vector3d per node alone is 670 MB. The local
/// offsets are therefore stored as float relative to `anchor` (the first node's
/// exact position), which keeps them within the grid's own extent where a
/// float's 24-bit mantissa is far finer than the millimetre this map promises.
struct PlaneToLocalMap {
    int cols = 0;  ///< node columns (grid.width + 1)
    int rows = 0;  ///< node rows (grid.height + 1)
    double x0 = 0.0;
    double y0 = 0.0;
    double dx = 1.0;
    double dy = 1.0;
    Eigen::Vector3d anchor = Eigen::Vector3d::Zero();  ///< exact first-node local
    std::vector<Eigen::Vector3f> local;  ///< cols * rows, relative to `anchor`
    std::vector<float> height;           ///< plane height each node was mapped at
    std::vector<float> vertical;         ///< d(local z)/dh at each node

    Eigen::Vector3d operator()(const Eigen::Vector3d& p) const {
        const double cf =
            std::clamp((p.x() - x0) / dx, 0.0, static_cast<double>(cols - 1));
        const double rf =
            std::clamp((y0 - p.y()) / dy, 0.0, static_cast<double>(rows - 1));
        const int c0 = std::min(static_cast<int>(std::floor(cf)), cols - 2);
        const int r0 = std::min(static_cast<int>(std::floor(rf)), rows - 2);
        const double fx = cf - static_cast<double>(c0);
        const double fy = rf - static_cast<double>(r0);
        const auto index = [this](int col, int row) {
            return static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
                   static_cast<std::size_t>(col);
        };
        const std::size_t i00 = index(c0, r0);
        const std::size_t i10 = index(c0 + 1, r0);
        const std::size_t i01 = index(c0, r0 + 1);
        const std::size_t i11 = index(c0 + 1, r0 + 1);
        const double w00 = (1.0 - fx) * (1.0 - fy);
        const double w10 = fx * (1.0 - fy);
        const double w01 = (1.0 - fx) * fy;
        const double w11 = fx * fy;
        const Eigen::Vector3f relative =
            static_cast<float>(w00) * local[i00] + static_cast<float>(w10) * local[i10] +
            static_cast<float>(w01) * local[i01] + static_cast<float>(w11) * local[i11];
        Eigen::Vector3d out = (relative.cast<double>() + anchor);
        // height/vertical are stored as float (see the struct comment): cast them
        // up explicitly rather than let -Wdouble-promotion reject the sum.
        const double node_height = w00 * static_cast<double>(height[i00]) +
                                   w10 * static_cast<double>(height[i10]) +
                                   w01 * static_cast<double>(height[i01]) +
                                   w11 * static_cast<double>(height[i11]);
        const double scale = w00 * static_cast<double>(vertical[i00]) +
                             w10 * static_cast<double>(vertical[i10]) +
                             w01 * static_cast<double>(vertical[i01]) +
                             w11 * static_cast<double>(vertical[i11]);
        out.z() += (p.z() - node_height) * scale;
        return out;
    }
};

}  // namespace

Eigen::Vector3d plane_to_local_anchor(const elevation::SurfaceFrame& frame,
                                     const RasterGrid& grid,
                                     const GeoRaster<float>& heights, float nodata,
                                     double fallback_height) {
    double h = sample_grid(heights, nodata, grid.origin_x, grid.origin_y);
    if (!std::isfinite(h) || h == static_cast<double>(nodata)) {
        h = std::isfinite(fallback_height) ? fallback_height : median_height(heights, nodata);
    }
    if (!std::isfinite(h)) {
        h = 0.0;
    }
    return frame.to_local(Eigen::Vector3d(grid.origin_x, grid.origin_y, h));
}

std::function<Eigen::Vector3d(const Eigen::Vector3d&)> make_plane_to_local(
    const elevation::SurfaceFrame& frame, const RasterGrid& grid,
    const GeoRaster<float>& heights, float nodata, double fallback_height,
    const PlaneToLocalLattice* lattice) {
    if (frame.is_local()) {
        return {};  // identity: the pose already lives in the plane frame
    }

    auto map = std::make_shared<PlaneToLocalMap>();
    map->cols = grid.width + 1;
    map->rows = grid.height + 1;
    map->x0 = grid.origin_x;
    map->y0 = grid.origin_y;
    map->dx = grid.pixel_size_x;
    map->dy = grid.pixel_size_y;
    const std::size_t nodes =
        static_cast<std::size_t>(map->cols) * static_cast<std::size_t>(map->rows);
    map->local.resize(nodes);
    map->height.resize(nodes);
    map->vertical.resize(nodes);
    // Nodes whose cell is nodata fall back to the raster's median valid height:
    // the node is only a control point for the interpolation, and the query
    // height is applied exactly on top.
    double fallback = fallback_height;
    if (!std::isfinite(fallback)) {
        fallback = median_height(heights, nodata);
    }
    if (!std::isfinite(fallback)) {
        fallback = 0.0;
    }

    // The float offsets are relative to this exact node, so their magnitude is
    // the grid's own extent (metres) rather than a UTM ordinate in the millions.
    // Set the anchor before the loop so the first node stores an exact zero.
    //
    // With a shared lattice the anchor is the GLOBAL first node, so a block's
    // offsets are the same numbers the whole-grid map would store (a per-block
    // anchor would differ in the last float bit and make a block's rectification
    // differ from the whole grid's).
    map->anchor = lattice != nullptr
                      ? lattice->anchor
                      : plane_to_local_anchor(frame, grid, heights, nodata, fallback);
    const Eigen::Vector3d anchor = map->anchor;

    for (int row = 0; row < map->rows; ++row) {
        for (int col = 0; col < map->cols; ++col) {
            // Global node coordinates in the shared-lattice case: nodes are named
            // by their integer position on the whole grid, so every block samples
            // exactly the point the whole-grid map sampled there.
            double x = 0.0;
            double y = 0.0;
            if (lattice != nullptr) {
                x = lattice->origin_x +
                    static_cast<double>(lattice->col_offset + col) * map->dx;
                y = lattice->origin_y -
                    static_cast<double>(lattice->row_offset + row) * map->dy;
            } else {
                x = map->x0 + static_cast<double>(col) * map->dx;
                y = map->y0 - static_cast<double>(row) * map->dy;
            }
            double h = sample_grid(heights, nodata, x, y);
            if (!std::isfinite(h) || h == static_cast<double>(nodata)) {
                h = fallback;
            }
            const std::size_t i =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(map->cols) +
                static_cast<std::size_t>(col);
            map->height[i] = static_cast<float>(h);
            const Eigen::Vector3d local = frame.to_local(Eigen::Vector3d(x, y, h));
            map->local[i] = (local - anchor).cast<float>();
            const Eigen::Vector3d up = frame.to_local(Eigen::Vector3d(x, y, h + 1.0));
            map->vertical[i] = static_cast<float>(up.z() - local.z());
        }
    }

    return [map](const Eigen::Vector3d& p) { return (*map)(p); };
}

namespace {

/// How the per-camera colour model is built, and applied before the seams are
/// chosen. "none" leaves every stored pixel alone; "gain" is one multiplier per
/// camera per channel (gain.hpp); "affine" is a multiplier and an offset per
/// camera per channel (colour_model.hpp), which also removes a contrast
/// difference rather than only a brightness one.
enum class ColourMode { None, Gain, Affine };

/// The validated parameter bundle.
struct ResolvedOptions {
    std::string surface = "elevation";
    ortho::Blending blending = ortho::Blending::Mosaic;
    bool fill_holes = true;
    bool refine_seamlines = true;
    ColourMode colour = ColourMode::None;
    double resolution = 0.0;  ///< 0 = automatic
    int projection = 0;
    /// The requested output CRS (M12 §4.9), as the DTO the parameter produced.
    /// Unset = the DEM's own CRS.
    CRS crs;
};

/// Parse the `color_correction` choice. The parameter used to be a Bool; a
/// project saved by that version records "true"/"false", so those two spellings
/// are still accepted here and mean what they meant then (an old `true` ran the
/// gain solve, and now names it explicitly). Returns None and does not throw on
/// an unknown word.
ColourMode parse_colour_mode(const std::string& text) {
    if (text == "none" || text == "false" || text.empty()) {
        return ColourMode::None;
    }
    if (text == "gain" || text == "true") {
        return ColourMode::Gain;
    }
    if (text == "affine") {
        return ColourMode::Affine;
    }
    return ColourMode::None;
}

ResolvedOptions resolve_options(const Params& params) {
    ResolvedOptions options;
    options.surface = params.get_string("surface_data");
    if (options.surface != "elevation" && options.surface != "model") {
        throw ValueError("BuildOrthomosaic: surface_data must be \"elevation\" or \"model\" "
                         "(got \"" +
                         options.surface + "\")");
    }
    // parse_blending() raises ValueError for any other text, which is the
    // contract the task inherits.
    options.blending = ortho::parse_blending(params.get_string("blending_mode"));
    options.fill_holes = params.get_bool("fill_holes");
    options.refine_seamlines = params.get_bool("refine_seamlines");
    const std::string colour = params.get_string("color_correction");
    options.colour = parse_colour_mode(colour);
    if (options.colour == ColourMode::None && colour != "none" && colour != "false" &&
        !colour.empty()) {
        throw ValueError("BuildOrthomosaic: color_correction must be \"none\", \"gain\" or "
                         "\"affine\" (got \"" +
                         colour + "\")");
    }
    options.resolution = params.get_float("resolution");
    if (options.resolution < 0.0) {
        throw ValueError(
            "BuildOrthomosaic: resolution must be >= 0 (0 selects an automatic value)");
    }
    options.projection = static_cast<int>(params.get_int("projection"));
    if (options.projection < 0) {
        throw ValueError("BuildOrthomosaic: projection must be an EPSG code >= 0");
    }
    // `projection` (Int EPSG, retained) and `crs` (String, new) must agree when
    // both are set. An unset request builds in the DEM's CRS; a DIFFERENT one
    // reprojects the surface once and builds the mosaic in the requested CRS
    // (WP-12.9).
    const std::string crs_text = params.get_string("crs");
    if (!crs_text.empty()) {
        options.crs = elevation::parse_crs_param(crs_text);
        if (options.projection != 0 && options.crs.is_set() &&
            options.crs.epsg != options.projection) {
            throw ValueError(
                "BuildOrthomosaic: projection (" + std::to_string(options.projection) +
                ") and crs (\"" + crs_text +
                "\") name different coordinate systems; pass only one");
        }
    } else if (options.projection != 0) {
        options.crs = elevation::parse_crs_param("EPSG:" + std::to_string(options.projection));
    }
    return options;
}

/// Resolve a frame's recorded path against the document directory, the same
/// rule AddPhotos / BuildPointCloud / BuildModel / BuildTexture use.
fs::path resolve_frame_path(const Document& document, const std::string& path) {
    const fs::path recorded(path);
    if (recorded.is_absolute() || document.path().empty()) {
        return recorded;
    }
    return fs::path(document.path()).parent_path() / recorded;
}

/// One enabled camera with a pose and a linked photograph. The photograph
/// itself is NOT held here: decoding every frame up front costs
/// cameras x image bytes (36 photos of 60 MP are ~6 GB), so the task decodes one
/// inside the rectify loop and releases it before the next.
struct CameraEntry {
    std::int32_t camera_id = -1;
    Calibration calibration;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Vector3d centre_local = Eigen::Vector3d::Zero();
    std::string image_path;  ///< resolved absolute path to the photograph
};

/// Collect every enabled camera that has a pose and a linked image (the paths
/// only: see CameraEntry). A camera without a linked frame is skipped (nothing
/// to sample); one that advertises a pose but has no sensor is a hard error, as
/// in BuildTexture.
std::vector<CameraEntry> collect_cameras(const Document& document, const Chunk& chunk) {
    std::unordered_map<std::int32_t, const Frame*> frame_of;
    for (const auto& frame : chunk.frames()) {
        if (frame->camera_id() >= 0) {
            frame_of.emplace(frame->camera_id(), frame.get());
        }
    }

    std::vector<CameraEntry> entries;
    for (const Camera& camera : chunk.cameras()) {
        if (!camera.enabled) {
            continue;
        }
        const auto frame_it = frame_of.find(camera.id);
        if (frame_it == frame_of.end()) {
            continue;  // no photograph: nothing to rectify
        }
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk.sensors().size())) {
            throw ValueError("BuildOrthomosaic: camera " + std::to_string(camera.id) +
                             " has no sensor");
        }
        CameraEntry entry;
        entry.camera_id = camera.id;
        entry.calibration =
            chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration;
        entry.R = camera.pose.block<3, 3>(0, 0);
        entry.t = camera.pose.block<3, 1>(0, 3);
        entry.centre_local = -entry.R.transpose() * entry.t;
        entry.image_path = resolve_frame_path(document, frame_it->second->path()).string();
        entries.push_back(std::move(entry));
    }
    return entries;
}

// -- per-stage memory model ------------------------------------------------
//
// Everything below is bytes PER OUTPUT CELL. The stages run one after another,
// so the peak is the largest stage, never the sum of unrelated stages. Only
// what is genuinely resident in each stage is counted there.

/// Resident throughout every stage once `initialize` finishes:
///   * the output grid's seam labels, int32            = 4
///   * the plane->local node map, (grid+1)^2 nodes of float3 + 2 floats,
///     i.e. 20 B/node ~ 20 B/cell (georeferenced chunks only)
constexpr double kBytesPerCellGridLabels = 4.0;
constexpr double kBytesPerNodeMap = 20.0;

/// Rectify stage. Tiles are cropped to their footprints, so this stage's cost
/// is a footprint area plus one decoded photograph; only the grid labels are
/// full-grid (and they are not allocated yet at that point).
///
/// Seam stage, full-grid part (`assign_seams` fine level):
///   * labels int32 (the caller's, counted above)                = -
///   * `best` double                                             = 8
///   * `any_valid` u8                                            = 1
///   * the two checkerboard parity lists, size_t per cell        = 8
///   * the alpha-expansion graph (see kSeamGraphBytesPerNode)
constexpr double kBytesPerCellSeamField = 8.0 + 1.0 + 8.0;

/// Blend stage, full-grid part (`ortho::blend`, Mosaic):
///   * out.rgb u8x3                                              = 3
///   * out.valid u8                                              = 1
///   * out.labels int32 (a copy of the caller's labels)           = 4
///   * the three per-channel accumulator pyramids, sum(1/4^k) of
///     a float plane, i.e. 3 * 4 / (1 - 1/4)                      = 16
///   * one tile's Gaussian/Laplacian/mask planes at the same
///     geometric series, about 3 planes at ~1.33 * 4 B            = 16
///   * the collapse's resize/up sources, another ~2 planes        = 11
///   * `scratch_mask` u8                                          = 1
constexpr double kBytesPerCellBlend = 3.0 + 1.0 + 4.0 + 16.0 + 16.0 + 11.0 + 1.0;

/// Finalize stage, full-grid part. The old code built `result.raster` (RGBA
/// u8 = 4 B/cell) while the blend mosaic (rgb+valid+labels = 8 B/cell) was
/// still alive, then `fill_holes` ran one channel at a time:
///   * result RGBA u8                                            = 4
///   * mosaic rgb u8x3 + valid u8 + labels int32                 = 8
///   * fill_holes per channel: float values (4) + u8 mask (1)
///     + NearestSites distance float (4) + site int32 (4)
///     + blocked u8 (1) + fill u8 (1) + result float (4)
///     + is_source u8 (1) + one band of Sibson accumulators
///       (bounded by kSibsonBandBudgetBytes, well under 1 B/cell
///       for any grid large enough to be interesting)              ~21
constexpr double kBytesPerCellFinalize = 4.0 + 8.0 + 21.0;

/// Seam stage's extra: the push-relabel graph `expand_alpha` builds for ONE
/// alpha move (`MaxFlow` in seamlines.cpp).
///
/// Nodes are the cells not already carrying alpha, i.e. up to one node per
/// cell of the level being solved, plus source and sink. Per node, from the
/// members' actual element types:
///   head_    int     4
///   height_  int     4
///   current_ int     4
///   excess_  double  8
///   seen_    char    1
///                                                          = 21
/// Edges: `add_edge` pushes two directed entries, so one graph edge is 2 x
/// (to_ int 4 + cap_ double 8 + next_ int 4) = 32 B. Each cell contributes two
/// unary edges (to alpha / keep) plus at most two pair edges shared with its
/// right and lower neighbours; the pair term emits one directed edge for the
/// shifted constant, one for `q keeps` and one for `q -> p`, so 3 + at most 2 =
/// 5 edges per cell in the worst case. 5 x 32 = 160.
///
/// `node` (int per cell, the cell -> node map) is another 4.
///
/// 21 + 160 + 4 = 185 B per NODE. The edge vectors are NOT reserved (their final
/// size depends on how many cells already carry alpha), so a 1.5x factor covers
/// the reallocation transient while the old and the new buffer are both alive:
/// 277.5 B per node.
///
/// WHERE IT RUNS. `assign_seams` calls alpha_expansion only on its COARSE
/// pyramid levels (default coarse_levels = 2, i.e. 2x and 4x, so 1/4 and 1/16 of
/// the fine cells); the full-resolution grid is finished by the ICM polish,
/// which allocates no graph. The seam stage's graph therefore peaks at the 2x
/// level. Should a future caller solve the fine grid with alpha-expansion
/// (coarse_levels = 0), the graph would be 4x this figure -- which is exactly
/// why the estimate keeps the coarse-level assumption visible here.
constexpr double kSeamGraphBytesPerNode = 1.5 * (21.0 + 5.0 * 32.0 + 4.0);
/// The coarsest-so-fine level alpha_expansion actually solves: the 2x level.
constexpr double kSeamGraphCellFraction = 0.25;

/// Bytes per camera footprint cell in a rectified tile (rgb u8x3 + valid u8 +
/// score float).
constexpr double kBytesPerTileCell = 8.0;/// A coarse seamline pyramid level holds its tiles at 1/factor^2 of the fine
/// area; the two configured levels (2x and 4x) therefore add 1/4 + 1/16 of the
/// footprint. The coarse levels also build their own Field (best/any_valid/
/// parity at the coarse cell count) and their own label vector.
constexpr double kCoarseFootprintFraction = 0.25 + 0.0625;
constexpr double kCoarseFieldBytesPerCell = kBytesPerCellSeamField + kBytesPerCellGridLabels;

/// Per-stage full-grid byte totals.
struct StageBytes {
    double rectify = 0.0;
    double seams = 0.0;
    double blend = 0.0;
    double finalize = 0.0;
};

/// The peak memory BuildOrthomosaic needs, in bytes.
///
/// `footprint_cells` is the summed area of every camera's tile crop box (the
/// rectified-tile cost); it is predicted from camera_footprint_bounds() before
/// any tile is allocated. `has_node_map` adds the plane->local map, which is
/// built first and lives until the task ends.
///
/// The formula, stage by stage (each stage's residency is what is listed in
/// its comment above; the peak is their maximum, not their sum):
///
///   rectify  = tiles (footprint, +coarse levels) + ONE decoded photograph
///   seams    = labels + node map + seam field + seam graph
///              + tiles (resident: the coarse levels read them) + coarse field
///   blend    = labels + node map + blend planes + tiles (read per cell)
///   finalize = labels + node map + result RGBA + live mosaic + fill_holes
///
/// The decoded photograph exists only inside the rectify loop, so it is counted
/// there and nowhere else: it is released before the seam stage starts.
double estimate_peak_bytes(std::uint64_t cells, double footprint_cells, bool has_node_map,
                           double photo_cells) {
    const double grid = static_cast<double>(cells);
    const double node_map = has_node_map ? grid * kBytesPerNodeMap : 0.0;
    // The tile set is resident through rectify, seams and blend.
    const double tiles =
        footprint_cells * kBytesPerTileCell * (1.0 + kCoarseFootprintFraction);
    // One decoded photograph, 3 channels of u8, is resident while its tile is
    // built. The image is dropped before the next camera decodes.
    const double photo = photo_cells * 3.0;

    StageBytes stages;
    stages.rectify = tiles + photo;
    // The seam stage carries the tiles (its coarse levels read them), the fine
    // Field (best/any_valid/parity) and the push-relabel graph for the 2x
    // level's alpha moves. Both coarse levels add their own Field and labels at
    // 1/4 and 1/16 of the fine cell count.
    const double coarse = grid * kCoarseFieldBytesPerCell * (1.0 / 4.0 + 1.0 / 16.0);
    const double seam_graph = grid * kSeamGraphBytesPerNode * kSeamGraphCellFraction;
    stages.seams = grid * kBytesPerCellGridLabels + node_map +
                   grid * kBytesPerCellSeamField + seam_graph + tiles + coarse;
    stages.blend =
        grid * kBytesPerCellGridLabels + node_map + grid * kBytesPerCellBlend + tiles;
    stages.finalize =
        grid * kBytesPerCellGridLabels + node_map + grid * kBytesPerCellFinalize;
    return std::max(std::max(stages.rectify, stages.seams),
                    std::max(stages.blend, stages.finalize));
}

/// The largest block side the plan considers (cells).
constexpr int kMaxBlockCells = 4096;
/// The smallest block side the plan considers (cells). Below this the
/// per-block overhead (margin, one photograph, the seam field's fixed cost)
/// dominates and further shrinking buys almost nothing.
constexpr int kMinBlockCells = 512;
/// Extra margin cells beyond the blend pyramid's support. The pyramid's edge
/// blurs clamp at the window border, so a cell exactly at the support distance
/// still sees a slightly different neighbourhood; a small slack takes the
/// interior out of that region.
constexpr int kBlockMarginSlack = 16;
/// The margin is rounded up to this multiple. The blend's pyramid halves per
/// level, so a window origin that is not a multiple of 2^levels shifts every
/// coarse level by part of a cell against the whole grid's.
constexpr int kBlockMarginAlignment = 64;
/// The fraction of what the block plan leaves over that decoded photographs may
/// use. Applied to the REMAINDER, not to the whole budget: the blocks come first
/// (a smaller cache buys a bigger block, which is the better trade, since a
/// block's margin overhead grows with its perimeter while a cache hit only saves
/// one decode). Half the remainder is the point where the cache stops competing
/// with the block's own peak and still holds a real flight's worth of frames.
constexpr double kPhotoCacheBudgetFraction = 0.5;
/// The absolute ceiling on the decoded-photograph cache, whatever the machine's
/// budget: beyond a couple of gigabytes the reuse stops paying for the bytes.
constexpr std::uint64_t kPhotoCacheMaxBytes = 2048ULL * 1024ULL * 1024ULL;

/// fill_holes()'s per-cell residency on the ASSEMBLED grid, on top of the
/// assembled labels (4 B) and RGBA (4 B) that are already alive: the same figure
/// kBytesPerCellFinalize's third component already models for the single-piece
/// path, i.e. one channel's float values + validity mask + EDT distance/sites +
/// blocked/fill/is_source flags + result float plane. The Sibson accumulators are
/// banded under kSibsonBandBudgetBytes, so they are a fixed term, not per cell.
constexpr double kBytesPerCellFillHoles = 21.0;
/// The banded Sibson accumulator ceiling (64 MiB of sums + counts in
/// fill_holes) plus room for its band bookkeeping and the EDT's own scratch.
constexpr double kFillHolesFixedBytes = 96.0 * 1024.0 * 1024.0;

/// The lowest and highest valid sample of a surface raster. Scanned once by the
/// caller and passed down: the auto-coarsening search calls the estimator tens
/// of times, and rescanning a 25 M-cell DEM on each one is a pointless cost.
struct HeightRange {
    double lowest = std::numeric_limits<double>::infinity();
    double highest = -std::numeric_limits<double>::infinity();

    bool valid() const { return std::isfinite(lowest) && std::isfinite(highest); }
};

HeightRange surface_height_range(const GeoRaster<float>& surface, float surface_nodata) {
    HeightRange range;
    for (const float h : surface.data) {
        if (h == surface_nodata || std::isnan(h)) {
            continue;
        }
        range.lowest = std::min(range.lowest, static_cast<double>(h));
        range.highest = std::max(range.highest, static_cast<double>(h));
    }
    return range;
}

/// The footprint area every camera's tile crop box will cover, padded the way
/// the rectifier pads it (two cells, and the cell rounding on each edge). This
/// is the same bound orthorectify() allocates against, so the guard's estimate
/// cannot be smaller than the allocation that follows.
///
/// It must run BEFORE the plane->local node map is allocated (that map is
/// itself `cells * 20 B`, the largest single grid-proportional object here), so
/// it takes no map: it evaluates the exact frame transform at the grid's mid
/// column and the surface's lowest/highest height and uses those two local
/// heights for `camera_footprint_bounds`. The node map it stands in for differs
/// from the exact transform by far less than the footprint's own 1% inflation and
/// three-cell margin.
double predicted_footprint_cells(const std::vector<CameraEntry>& cameras,
                                 const RasterGrid& grid, const HeightRange& heights,
                                 const elevation::SurfaceFrame& frame) {
    if (!heights.valid()) {
        return static_cast<double>(grid.width) * static_cast<double>(grid.height);
    }
    const double lowest = heights.lowest;
    const double highest = heights.highest;

    const Eigen::Vector2d metres_per_unit =
        frame.metres_per_plane_unit(grid.pixel_center(0, grid.height / 2).y());
    const bool mapped = !frame.is_local();
    // `frame` is captured by reference; this lambda allocates nothing.
    const auto to_local = [&frame](const Eigen::Vector3d& p) { return frame.to_local(p); };
    double total = 0.0;
    for (const CameraEntry& entry : cameras) {
        ortho::OrthoCamera camera;
        camera.calibration = entry.calibration;
        camera.R = entry.R;
        camera.t = entry.t;
        camera.plane_metres_per_unit = metres_per_unit;
        double local_low = lowest;
        double local_high = highest;
        if (mapped) {
            camera.plane_to_local = to_local;
            camera.local_to_plane = [&frame](const Eigen::Vector3d& p) {
                return frame.to_plane(p);
            };
            const Eigen::Vector2d mid = grid.pixel_center(grid.width / 2, grid.height / 2);
            const Eigen::Vector3d lo =
                to_local(Eigen::Vector3d(mid.x(), mid.y(), lowest));
            const Eigen::Vector3d hi =
                to_local(Eigen::Vector3d(mid.x(), mid.y(), highest));
            local_low = std::min(lo.z(), hi.z());
            local_high = std::max(lo.z(), hi.z());
        }
        const ortho::TileBounds bounds =
            ortho::camera_footprint_bounds(grid, camera, local_low, local_high, 2);
        total += static_cast<double>(bounds.width) * static_cast<double>(bounds.height);
    }
    return total;
}

/// The source pixel count of the largest photograph, i.e. what ONE decoded
/// frame costs (3 bytes per pixel). Taken from the calibration, which carries
/// the sensor dimensions, so no image has to be decoded to know it; a camera
/// with no recorded sensor size falls back to 1 so the term never vanishes.
double largest_photograph_cells(const std::vector<CameraEntry>& cameras) {
    double largest = 1.0;
    for (const CameraEntry& entry : cameras) {
        const double pixels = static_cast<double>(entry.calibration.width) *
                              static_cast<double>(entry.calibration.height);
        if (std::isfinite(pixels) && pixels > 0.0) {
            largest = std::max(largest, pixels);
        }
    }
    return largest;
}

/// An LRU cache of decoded photographs, bounded by BYTES.
///
/// A block decodes every camera that reaches it; consecutive blocks share most
/// of those cameras, so reusing the decode is the difference between one decode
/// per (block, camera) pair and one per camera. The cache is bounded by the
/// plan's share of the memory budget, so it can never be the reason a run does
/// not fit. Least-recently-used entries are dropped first, and the entry just
/// handed out is always kept (a single photograph larger than the whole cache
/// still works: it stays the only entry).
class PhotoCache {
public:
    explicit PhotoCache(std::uint64_t budget_bytes) : budget_(budget_bytes) {}

    /// The decoded photograph for `slot`, decoding (and caching) it on a miss.
    /// Returns an empty image when the file cannot be read; the caller turns
    /// that into the same ValueError the uncached path raises.
    const Image& get(std::size_t slot, const std::string& path) {
        const auto found = index_.find(slot);
        if (found != index_.end()) {
            ++hits_;
            entries_[found->second].last = ++clock_;
            return entries_[found->second].image;
        }
        ++misses_;
        Image image = read_image(path);
        if (image.empty()) {
            static const Image kEmpty;
            return kEmpty;
        }
        const std::uint64_t bytes = static_cast<std::uint64_t>(image.size_bytes());
        evict_for(bytes);
        entries_.push_back(Entry{slot, bytes, ++clock_, std::move(image)});
        index_.emplace(slot, entries_.size() - 1);
        used_ += bytes;
        return entries_.back().image;
    }

    /// Photographs actually decoded, i.e. the decodes a run paid for.
    std::size_t misses() const { return misses_; }
    /// Cache lookups served without a decode.
    std::size_t hits() const { return hits_; }
    /// The byte capacity this cache was sized to.
    std::uint64_t budget_bytes() const { return budget_; }

    /// Mark `slot` most-recently-used when it is already cached, so the frames a
    /// following block will need survive the decodes this one does not share.
    /// A miss stays a miss: touching must not decode anything early, or a
    /// prefetch would pay for frames the run may never reach.
    void touch(std::size_t slot) {
        const auto found = index_.find(slot);
        if (found == index_.end()) {
            return;
        }
        entries_[found->second].last = ++clock_;
    }

private:
    struct Entry {
        std::size_t slot = 0;
        std::uint64_t bytes = 0;
        std::uint64_t last = 0;
        Image image;
    };

    /// Drop least-recently-used entries until `incoming` more bytes fit. The
    /// entry being inserted is not in `entries_` yet, so a cache of exactly one
    /// frame's bytes always holds the frame the caller is using.
    void evict_for(std::uint64_t incoming) {
        while (budget_ > 0 && !entries_.empty() && used_ + incoming > budget_) {
            std::size_t victim = 0;
            for (std::size_t i = 1; i < entries_.size(); ++i) {
                if (entries_[i].last < entries_[victim].last) {
                    victim = i;
                }
            }
            used_ -= entries_[victim].bytes;
            index_.erase(entries_[victim].slot);
            entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(victim));
            for (std::size_t i = victim; i < entries_.size(); ++i) {
                index_[entries_[i].slot] = i;
            }
        }
    }

    std::uint64_t budget_ = 0;  ///< 0 = unbounded (the whole-grid path)
    std::uint64_t used_ = 0;
    std::uint64_t clock_ = 0;
    std::size_t hits_ = 0;
    std::size_t misses_ = 0;
    std::vector<Entry> entries_;
    std::unordered_map<std::size_t, std::size_t> index_;
};

/// Estimate one gain per camera per channel over the WHOLE output grid, read on
/// a coarse lattice.
///
/// The coarse grid is the output grid coarsened by `stride`; the photographs are
/// rectified onto it exactly as the block loop rectifies them onto the full grid
/// (same plane frame, same occlusion test), and every coarse cell that two or
/// more cameras cover contributes one sample to each of their overlap pairs. The
/// solve then gives gains that are a property of the FLIGHT, not of a block, so
/// every block applies the same multiplier to the same camera and no block border
/// can disagree with its neighbour.
///
/// MEMORY. The coarse tiles are cropped to their footprints (the same crop the
/// full pass uses) and a coarse tile is 1/stride^2 of the corresponding
/// full-resolution one, so the pass costs a small fraction of a single extra
/// ortho. They are kept only for the duration of this call and released before
/// the block loop allocates anything.
///
/// Returns one {r, g, b} gain per camera, in the entry order of `cameras`.
std::vector<std::array<double, 3>> estimate_camera_gains(
    const std::vector<CameraEntry>& cameras, const elevation::SurfaceFrame& frame,
    const GeoRaster<float>& surface, float surface_nodata,
    const ortho::OrthoSurfaceRange& surface_range, double occlusion_tolerance,
    const RasterGrid& grid, int stride, double fallback_height, PhotoCache& photos,
    const Progress& progress, const std::string& stage) {
    std::vector<std::array<double, 3>> identity(
        cameras.size(), std::array<double, 3>{1.0, 1.0, 1.0});
    if (cameras.size() < 2 || grid.width <= 0 || grid.height <= 0 || stride <= 0) {
        return identity;
    }

    RasterGrid coarse;
    coarse.pixel_size_x = grid.pixel_size_x * static_cast<double>(stride);
    coarse.pixel_size_y = grid.pixel_size_y * static_cast<double>(stride);
    coarse.origin_x = grid.origin_x;
    coarse.origin_y = grid.origin_y;
    coarse.width = (grid.width + stride - 1) / stride;
    coarse.height = (grid.height + stride - 1) / stride;

    const std::function<Eigen::Vector3d(const Eigen::Vector3d&)> to_local =
        make_plane_to_local(frame, coarse, surface, surface_nodata, fallback_height);
    const Eigen::Vector2d metres_per_unit =
        frame.metres_per_plane_unit(coarse.pixel_center(0, coarse.height / 2).y());

    std::vector<ortho::OrthoTile> tiles;
    tiles.reserve(cameras.size());
    for (std::size_t k = 0; k < cameras.size(); ++k) {
        if (progress.is_cancelled()) {
            throw Error("BuildOrthomosaic cancelled");
        }
        const CameraEntry& entry = cameras[k];
        const Image& image = photos.get(k, entry.image_path);
        if (image.empty()) {
            throw ValueError("BuildOrthomosaic: camera " + std::to_string(entry.camera_id) +
                             " has an unreadable image");
        }
        ortho::OrthoCamera camera;
        camera.calibration = entry.calibration;
        camera.R = entry.R;
        camera.t = entry.t;
        camera.image = &image;
        camera.plane_metres_per_unit = metres_per_unit;
        if (to_local) {
            camera.plane_to_local = to_local;
            camera.local_to_plane = [&frame](const Eigen::Vector3d& p) {
                return frame.to_plane(p);
            };
            camera.plane_centre = frame.to_plane(entry.centre_local);
        }
        tiles.push_back(ortho::orthorectify(surface, surface_nodata, coarse, camera,
                                           static_cast<int>(k), occlusion_tolerance, true,
                                           &surface_range));
        progress.update(stage, static_cast<double>(k + 1) / static_cast<double>(cameras.size()));
    }

    // Pair every coarse cell's covering cameras. Iterating the later camera
    // against the earlier ones (j < i) records each pair exactly once; the
    // accumulator itself also normalises the order, so this is only for locality.
    ortho::OverlapAccumulator overlaps;
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        const ortho::OrthoTile& later = tiles[i];
        if (later.empty()) {
            continue;
        }
        for (std::size_t j = 0; j < i; ++j) {
            const ortho::OrthoTile& earlier = tiles[j];
            if (earlier.empty()) {
                continue;
            }
            const int x0 = std::max(earlier.x0, later.x0);
            const int x1 = std::min(earlier.x0 + earlier.width, later.x0 + later.width);
            const int y0 = std::max(earlier.y0, later.y0);
            const int y1 = std::min(earlier.y0 + earlier.height, later.y0 + later.height);
            for (int row = y0; row < y1; ++row) {
                for (int col = x0; col < x1; ++col) {
                    const std::size_t ec = earlier.index(col, row);
                    const std::size_t lc = later.index(col, row);
                    if (earlier.valid[ec] == 0 || later.valid[lc] == 0) {
                        continue;
                    }
                    overlaps.add(static_cast<int>(j), static_cast<int>(i),
                                 &earlier.rgb[ec * 3], &later.rgb[lc * 3]);
                }
            }
        }
    }

    if (overlaps.empty()) {
        if (Log::enabled(LogLevel::Info)) {
            log(LogLevel::Info) << "BuildOrthomosaic: gain compensation found no overlapping "
                                   "camera pairs on the coarse grid; leaving gains at 1";
        }
        return identity;
    }
    const std::vector<ortho::OverlapPair> pairs = overlaps.finish();
    if (Log::enabled(LogLevel::Info)) {
        // The RAW gain range before mean-normalisation is the useful diagnostic:
        // a range hugging 1 means the photographs already agree, while a wide
        // range that normalises flat would mean the solve found no real signal.
        double raw_lo = 1.0;
        double raw_hi = 1.0;
        for (const ortho::OverlapPair& pair : pairs) {
            for (int c = 0; c < 3; ++c) {
                const double a = pair.mean_a[static_cast<std::size_t>(c)];
                const double b = pair.mean_b[static_cast<std::size_t>(c)];
                if (a > 1e-6 && b > 1e-6) {
                    const double ratio = a / b;
                    raw_lo = std::min(raw_lo, ratio);
                    raw_hi = std::max(raw_hi, ratio);
                }
            }
        }
        log(LogLevel::Info) << "BuildOrthomosaic: gain estimate over " << pairs.size()
                            << " overlapping camera pairs, overlap brightness ratio "
                            << raw_lo << " .. " << raw_hi;
    }
    const std::vector<std::array<double, 3>> gains =
        ortho::solve_gains(static_cast<int>(cameras.size()), pairs);

    // The solved gains themselves, per channel, at INFO: the pair/ratio log above
    // says what went IN, this says what came OUT. A range of 1 .. 1 here while the
    // ratios above are wide is the signature of a solve that failed to carry its
    // signal through (the bug this diagnostic was added to catch), so the two
    // lines are meant to be read together.
    if (Log::enabled(LogLevel::Info)) {
        double samples = 0.0;
        for (const ortho::OverlapPair& pair : pairs) {
            samples += pair.count;
        }
        std::ostringstream text;
        text << "BuildOrthomosaic: gain solve used " << pairs.size() << " pairs, "
             << static_cast<std::uint64_t>(samples) << " samples; gains per channel";
        static const char* const kNames[3] = {"r", "g", "b"};
        for (int c = 0; c < 3; ++c) {
            double lo = gains.empty() ? 1.0 : gains.front()[static_cast<std::size_t>(c)];
            double hi = lo;
            double sum = 0.0;
            for (const auto& gain : gains) {
                const double g = gain[static_cast<std::size_t>(c)];
                lo = std::min(lo, g);
                hi = std::max(hi, g);
                sum += g;
            }
            text << ' ' << kNames[c] << "[" << format_number(lo) << " .. "
                 << format_number(hi) << " mean "
                 << format_number(gains.empty() ? 1.0
                                                : sum / static_cast<double>(gains.size()))
                 << ']';
        }
        log(LogLevel::Info) << text.str();
    }
    return gains;
}

constexpr int kGainSampleStride = 8;

// -- the affine colour model's sampling budget -------------------------------
//
// The affine model is fitted on 3-D surface points, not on an output-grid
// raster, so its budget is stated in POINTS rather than in cells. One point
// sampled in C cameras yields up to C(C-1)/2 observations, which for the
// 12-camera aerial fixture is 66 -- and for a 40-camera flight would be 780, so
// the observation list is capped separately rather than letting the point cap
// imply it.
//
// 200 000 points is the brief's figure, and it is also where the returns stop:
// every camera's a and b are already pinned by tens of thousands of
// observations, and a denser lattice changes the fit by far less than the
// 8-bit quantisation the result is applied at. The solve's own memory is 2
// unknowns per camera per channel and nothing per observation (the pair
// statistics are accumulated as they are met).
constexpr std::uint64_t kAffineMaxPoints = 200000;
/// The observation cap: 32 B each, so 256 MB at the very worst, and only
/// reachable by a very large flight seen by very many cameras at once. Reaching
/// it truncates the sample rather than failing -- 8 M disagreement measurements
/// still pin a two-parameter model.
constexpr std::size_t kAffineMaxObservations = 8000000;
/// The smallest rectifier score a sample may carry. The rectifier's score is
/// cos(view) * border taper, in (0, 1]; 0.15 rejects the taper's outer 15% of
/// the image border and views more oblique than about 81 degrees. A border pixel
/// is a bilinear read across a compressed gradient, so its stored value is a
/// poor measurement of the surface's colour -- exactly what the model must not
/// be fitted to.
constexpr double kAffineMinSampleScore = 0.15;

/// Sample the paired observations the AFFINE colour model is fitted to, and
/// solve it.
///
/// Sampling is on 3-D SURFACE points, not on output-grid overlap means. The two
/// are not equivalent: a mean over an overlap region throws away the spread
/// (dark vs bright parts of the shared area), and that spread is the only thing
/// that separates the gain from the offset. One surface point sampled in several
/// cameras gives the pair (I_i, I_j) the affine model needs.
///
/// The lattice is the DEM's own cell grid thinned by `stride`, capped at
/// `kAffineMaxPoints` points overall. Each point is projected into every camera
/// with orthorectify()'s visibility test (project_surface_point), and its colour
/// is read bilinearly from the photograph at its working resolution. Points seen
/// by three or more cameras contribute one observation per camera PAIR; points
/// seen by fewer than two contribute nothing.
///
/// MEMORY. Nothing here is grid-sized. Each camera holds one `SurfaceSample`
/// (24 B: two doubles and a score) per lattice point it actually sees, plus one
/// three-float colour per such point, and both are released as soon as the
/// observations are formed; the observations themselves are 32 B each and capped
/// at kAffineMaxObservations. Photographs go through the caller's LRU cache,
/// exactly like the rectify loop, so the sampler is not a second full decode of
/// the flight.
std::vector<ortho::CameraColour> estimate_camera_colours(
    const std::vector<CameraEntry>& cameras, const elevation::SurfaceFrame& frame,
    const GeoRaster<float>& surface, float surface_nodata,
    const ortho::OrthoSurfaceRange& surface_range, double occlusion_tolerance,
    const RasterGrid& grid, int stride, PhotoCache& photos, const Progress& progress,
    const std::string& stage) {
    std::vector<ortho::CameraColour> identity(cameras.size());
    if (cameras.size() < 2 || grid.width <= 0 || grid.height <= 0 || stride <= 0 ||
        surface.empty()) {
        return identity;
    }

    // The lattice lives on the DEM, and only its VALID samples are used: a
    // nodata cell has no surface point to sample the photographs at.
    const RasterGrid& dem_grid = surface.grid;
    int dem_stride =
        std::max(stride, 1) * std::max(1, static_cast<int>(std::lround(
                                              grid.pixel_size_x / dem_grid.pixel_size_x)));

    // Size the sampling stride from the DEM: the requested stride, then doubled
    // until the total stays under the cap. Doubling keeps the lattice aligned to
    // the DEM's own cells (which is what makes the sample set independent of the
    // output grid's resolution) and converges in a handful of steps.
    std::uint64_t columns = static_cast<std::uint64_t>((dem_grid.width + dem_stride - 1) / dem_stride);
    std::uint64_t rows = static_cast<std::uint64_t>((dem_grid.height + dem_stride - 1) / dem_stride);
    while (columns * rows > static_cast<std::uint64_t>(kAffineMaxPoints) && dem_stride < 4096) {
        dem_stride *= 2;
        columns = static_cast<std::uint64_t>((dem_grid.width + dem_stride - 1) / dem_stride);
        rows = static_cast<std::uint64_t>((dem_grid.height + dem_stride - 1) / dem_stride);
    }

    const std::size_t camera_count = cameras.size();
    std::vector<ortho::OrthoCamera> ortho_cameras(camera_count);
    const auto to_local = [&frame](const Eigen::Vector3d& p) { return frame.to_local(p); };
    const Eigen::Vector2d metres_per_unit =
        frame.metres_per_plane_unit(grid.pixel_center(0, grid.height / 2).y());
    for (std::size_t k = 0; k < camera_count; ++k) {
        const CameraEntry& entry = cameras[k];
        ortho::OrthoCamera& camera = ortho_cameras[k];
        camera.calibration = entry.calibration;
        camera.R = entry.R;
        camera.t = entry.t;
        camera.plane_metres_per_unit = metres_per_unit;
        if (!frame.is_local()) {
            camera.plane_to_local = to_local;
            camera.local_to_plane = [&frame](const Eigen::Vector3d& p) { return frame.to_plane(p); };
            camera.plane_centre = frame.to_plane(entry.centre_local);
        }
    }

    // -- pass 1: where does each lattice point land in each camera? -----------
    // The photographs are not needed yet, so this pass costs a projection and an
    // occlusion march per (point, camera) and nothing else. A point no camera
    // sees is dropped before any image is decoded.
    //
    // Storage is per camera and COMPACT: each camera's `pixels[k]` grows in
    // lockstep with its `points_by_camera[k]` entry, so camera k's n-th colour
    // vector belongs to the n-th point in its list. A dense C x points lattice
    // would be mostly empty -- most cameras see only part of the site -- and
    // would make the sampler's memory follow the camera count rather than the
    // number of actual observations.
    std::vector<std::vector<ortho::SurfaceSample>> pixels(camera_count);
    std::vector<std::pair<int, ortho::SurfaceSample>> at_point;
    std::vector<std::vector<int>> points_by_camera(camera_count);
    std::size_t point_count = 0;
    for (std::uint64_t y = 0; y < rows; ++y) {
        if (progress.is_cancelled()) {
            throw Error("BuildOrthomosaic cancelled");
        }
        for (std::uint64_t x = 0; x < columns; ++x) {
            const int col = static_cast<int>(x) * dem_stride;
            const int row = static_cast<int>(y) * dem_stride;
            if (!dem_grid.contains_pixel(col, row)) {
                continue;
            }
            const float h = surface.at(col, row);
            if (h == surface_nodata || std::isnan(h)) {
                continue;
            }
            const Eigen::Vector2d xy = dem_grid.pixel_center(col, row);
            const Eigen::Vector3d p(xy.x(), xy.y(), static_cast<double>(h));
            at_point.clear();
            for (std::size_t k = 0; k < camera_count; ++k) {
                ortho::SurfaceSample sample;
                if (!ortho::project_surface_point(surface, surface_nodata, ortho_cameras[k],
                                                  occlusion_tolerance, p, surface_range,
                                                  sample)) {
                    continue;
                }
                // A pixel the rectifier would score near zero is at the image
                // border (stretched) or seen at a grazing angle: poor evidence
                // for a colour model, so it is not offered as one.
                if (sample.score < kAffineMinSampleScore) {
                    continue;
                }
                at_point.emplace_back(static_cast<int>(k), sample);
            }
            if (at_point.size() < 2) {
                continue;  // seen by one camera: no pair, no observation
            }
            // Commit the point. Ordering inside `at_point` is by camera index,
            // so every camera's list stays ascending in point id and the pair
            // merge below is a straight walk.
            const int local = static_cast<int>(point_count);
            for (const auto& [k, sample] : at_point) {
                points_by_camera[static_cast<std::size_t>(k)].push_back(local);
                pixels[static_cast<std::size_t>(k)].push_back(sample);
            }
            ++point_count;
        }
        progress.update(stage, 0.5 * (static_cast<double>(y + 1) / static_cast<double>(rows)));
    }
    ortho_cameras.clear();
    ortho_cameras.shrink_to_fit();

    if (point_count == 0) {
        if (Log::enabled(LogLevel::Info)) {
            log(LogLevel::Info) << "BuildOrthomosaic: affine colour model found no surface point "
                                   "seen by two or more cameras; leaving every camera at the "
                                   "identity";
        }
        return identity;
    }

    // -- pass 2: read each camera's colour at its own lattice points ----------
    // One decode per camera (through the LRU cache), then every point that
    // camera saw. `pixels[k][i]` and `colour[k][i]` both belong to
    // `points_by_camera[k][i]`, so every vector shares one index.
    std::vector<std::vector<std::array<float, 3>>> colour(camera_count);
    for (std::size_t k = 0; k < camera_count; ++k) {
        if (progress.is_cancelled()) {
            throw Error("BuildOrthomosaic cancelled");
        }
        if (points_by_camera[k].empty()) {
            continue;
        }
        const CameraEntry& entry = cameras[k];
        const Image& image = photos.get(k, entry.image_path);
        if (image.empty()) {
            throw ValueError("BuildOrthomosaic: camera " + std::to_string(entry.camera_id) +
                             " has an unreadable image");
        }
        colour[k].resize(pixels[k].size());
        for (std::size_t i = 0; i < pixels[k].size(); ++i) {
            std::uint8_t rgb[3] = {0, 0, 0};
            pixels[k][i].sample_colour(image, rgb);
            colour[k][i] = {static_cast<float>(rgb[0]) * (1.0F / 255.0F),
                            static_cast<float>(rgb[1]) * (1.0F / 255.0F),
                            static_cast<float>(rgb[2]) * (1.0F / 255.0F)};
        }
        // The samples are spent once the colour is read, and the photograph is
        // about to be handed back to the cache: release both now, so the peak
        // while the observations are being formed is the observations alone.
        pixels[k].clear();
        pixels[k].shrink_to_fit();
        progress.update(stage, 0.5 + 0.4 * (static_cast<double>(k + 1) /
                                           static_cast<double>(colour.size())));
    }
    pixels.clear();
    pixels.shrink_to_fit();

    // -- pair extraction and solve --------------------------------------------
    // Reserve for what this flight can produce, capped: the estimate is
    // points x pairs-per-point, and reserving the hard ceiling would cost 256 MB
    // on a run that will never reach it.
    const std::size_t pairs_per_point =
        camera_count < 8 ? camera_count * (camera_count - 1) / 2 : 28;
    const std::size_t estimate = point_count * std::max<std::size_t>(pairs_per_point, 1);
    std::vector<ortho::ColourObservation> observations;
    observations.reserve(std::min<std::size_t>(estimate, kAffineMaxObservations));
    for (std::size_t k = 0; k < colour.size() && observations.size() < kAffineMaxObservations; ++k) {
        if (points_by_camera[k].empty()) {
            continue;
        }
        for (std::size_t other = k + 1; other < colour.size(); ++other) {
            if (points_by_camera[other].empty()) {
                continue;
            }
            // Walk both lists in lockstep: both are ascending in point id, so a
            // merge is linear and needs no bitmap. i and j also index the
            // per-camera colour vectors, which share the lists' order.
            std::size_t i = 0;
            std::size_t j = 0;
            while (i < points_by_camera[k].size() && j < points_by_camera[other].size() &&
                   observations.size() < kAffineMaxObservations) {
                const int a = points_by_camera[k][i];
                const int b = points_by_camera[other][j];
                if (a < b) {
                    ++i;
                    continue;
                }
                if (b < a) {
                    ++j;
                    continue;
                }
                ortho::ColourObservation observation;
                observation.a = static_cast<std::int32_t>(k);
                observation.b = static_cast<std::int32_t>(other);
                observation.value_a = colour[k][i];
                observation.value_b = colour[other][j];
                observations.push_back(observation);
                ++i;
                ++j;
            }
        }
        // Release the camera's colour as its pairs are consumed: the model is
        // solved from the pair list, not from the per-camera arrays.
        colour[k].clear();
        colour[k].shrink_to_fit();
    }
    colour.clear();
    colour.shrink_to_fit();

    if (observations.empty()) {
        return identity;
    }

    const std::vector<ortho::CameraColour> solved =
        ortho::solve_affine_colours(static_cast<int>(cameras.size()), observations);

    if (Log::enabled(LogLevel::Info)) {
        std::ostringstream text;
        text << "BuildOrthomosaic: affine colour model over " << point_count
             << " surface points, " << observations.size() << " camera-pair observations;";
        static const char* const kNames[3] = {"r", "g", "b"};
        for (std::size_t c = 0; c < 3; ++c) {
            double g_lo = 1.0;
            double g_hi = 1.0;
            double o_lo = 0.0;
            double o_hi = 0.0;
            for (const ortho::CameraColour& colour_of : solved) {
                const double a = colour_of.gain[c];
                const double b = colour_of.offset[c];
                g_lo = std::min(g_lo, a);
                g_hi = std::max(g_hi, a);
                o_lo = std::min(o_lo, b);
                o_hi = std::max(o_hi, b);
            }
            text << ' ' << kNames[c] << "[gain " << format_number(g_lo) << " .. "
                 << format_number(g_hi) << ", offset " << format_number(o_lo) << " .. "
                 << format_number(o_hi) << " grey]";
        }
        log(LogLevel::Info) << text.str();
    }
    return solved;
}

}  // namespace

// -- per-camera colour correction -------------------------------------------
//
// The model must be the SAME for every block (a block that solved its own would
// disagree with its neighbour at the border), so it is estimated ONCE over the
// whole flight before the block loop runs, and applied per camera inside each
// block. The gain estimate never needs every cell: a coarse lattice of sample
// cells gives each overlapping pair hundreds of colour samples, which pins a
// per-channel mean far more precisely than a gain needs, at a memory cost of a
// few megabytes rather than a whole extra ortho. The affine estimate samples
// surface points instead (see estimate_camera_colours), because a gain and an
// offset need the SPREAD across a shared area, not just its mean.

int block_margin_cells(int block_cells, int pyramid_levels) {
    // The margin exists to make a block's interior identical to the whole grid's
    // there, so its size is set by the operators that read across cells, not by
    // the block size:
    //   * the mosaic blend at `pyramid_levels` reads pyramid_support_cells()
    //     cells (126 at the 6-level cap), plus kBlockMarginSlack for the
    //     edge-clamped blurs;
    //   * the seam labeller's own multi-scale pass bottoms out at its 4x level,
    //     whose 5-tap blurs and alpha-expansion stay well inside that.
    // A block must be at least twice the margin for the interior to be the
    // majority of the window; below that the caller should use a bigger block.
    if (block_cells <= 0) {
        return 0;
    }
    const int levels = std::max(0, pyramid_levels);
    // TWO supports, not one. The margin must cover (a) the pyramid READ: a level-0
    // cell the pyramid touches reads up to `support` cells away, and (b) the
    // FILL growth: those cells may themselves be invalid in this tile and take
    // their colour from a chain of filled neighbours reaching `support` cells
    // further out again. One support would leave the second hop clipped at the
    // window border, which is exactly the difference the blocked-vs-whole-grid
    // blend showed.
    const int needed = 2 * ortho::pyramid_support_cells(levels) + kBlockMarginSlack;
    // Round UP to a multiple of kBlockMarginAlignment. That is not cosmetic: the
    // blend's pyramid decimates by two per level, so a window whose origin is not
    // a multiple of 2^levels has its coarse levels offset by a fraction of a
    // cell relative to the whole grid's, and a block's coarse-level samples would
    // then be interpolations BETWEEN global ones rather than equal to them.
    // Block origins are multiples of the block side (a power of two >= 512), so
    // an aligned margin makes every window origin a multiple of 64 as well.
    const int aligned = ((needed + kBlockMarginAlignment - 1) / kBlockMarginAlignment) *
                        kBlockMarginAlignment;
    return std::max(kBlockMarginAlignment, aligned);
}

/// The peak of the fill_holes stage that runs ONCE on the assembled raster after
/// the last block (fill_holes=true). It is a first-class part of a blocked run's
/// peak, not an afterthought: the assembled labels and RGBA stay alive while one
/// channel's float plane, validity mask, EDT and Sibson accumulators are built,
/// and on a big grid that is several hundred megabytes.
double estimate_fill_holes_peak_bytes(std::uint64_t output_cells) {
    const double cells = static_cast<double>(output_cells);
    return cells * (kBytesPerCellGridLabels + 4.0 /* result RGBA */ +
                    kBytesPerCellFillHoles) +
           kFillHolesFixedBytes;
}

double estimate_block_peak_bytes(std::uint64_t window_cells, double window_footprint_cells,
                                 bool has_node_map, double photo_cells,
                                 std::uint64_t output_cells,
                                 std::uint64_t photo_cache_bytes, bool fill_holes) {
    // The window's stages are the estimator the whole-grid path already uses,
    // with the window's (block + margin) cell count and its own footprint. The
    // assembled output raster (RGBA, 4 B) plus its seam labels (int32, 4 B) is
    // resident from the first block on and grows to the full output grid.
    const double window = estimate_peak_bytes(window_cells, window_footprint_cells, has_node_map,
                                              photo_cells);
    const double assembled = static_cast<double>(output_cells) *
                             (kBytesPerCellGridLabels + 4.0 /* result RGBA */);
    const double staged = window + assembled;
    // fill_holes runs after the last block on the assembled grid, so its own
    // peak is a separate candidate for the run's maximum, not a summand.
    const double filled =
        fill_holes ? estimate_fill_holes_peak_bytes(output_cells) : 0.0;
    // The photo cache is alive through all of it.
    return std::max(staged, filled) + static_cast<double>(photo_cache_bytes);
}

OrthoBlockPlan plan_ortho_blocks(const OrthoBlockPlanInput& input) {
    const std::uint64_t width = static_cast<std::uint64_t>(std::max(0, input.grid_width));
    const std::uint64_t height = static_cast<std::uint64_t>(std::max(0, input.grid_height));
    const std::uint64_t output_cells = width * height;

    OrthoBlockPlan plan;

    /// Everything about one candidate block side, so the override, the search
    /// and the "even the minimum does not fit" fallback all agree.
    const auto plan_for_side = [&](int side) {
        OrthoBlockPlan out;
        out.block_cells = side;
        // The margin must cover the mosaic pyramid's support at the depth every
        // block applies: that depth is the WHOLE grid's, so all blocks use the
        // same operator (a shallower pyramid leaves a visible step at its next
        // scale). The margin is therefore raised to the support plus a slack for
        // the edge-clamped blurs, never the depth lowered to a smaller margin.
        int depth = 0;
        if (width > 0 && height > 0) {
            depth = ortho::pyramid_level_count(static_cast<int>(width),
                                              static_cast<int>(height));
        }
        out.levels = depth;
        out.margin_cells = block_margin_cells(side, depth);
        const std::uint64_t step = static_cast<std::uint64_t>(std::max(1, side));
        out.columns = std::max(1, static_cast<int>((width + step - 1) / step));
        out.rows = std::max(1, static_cast<int>((height + step - 1) / step));
        const std::uint64_t window_w =
            std::min<std::uint64_t>(width, static_cast<std::uint64_t>(side + 2 * out.margin_cells));
        const std::uint64_t window_h =
            std::min<std::uint64_t>(height, static_cast<std::uint64_t>(side + 2 * out.margin_cells));
        const std::uint64_t window_cells = std::max<std::uint64_t>(1, window_w * window_h);
        // A window sees every camera whose crop box touches it. Clipping never
        // grows a box, so the window's footprint is at most the whole grid's AND
        // at most one window per camera; the smaller bound keeps a fully
        // overlapping oblique flight (every camera sees the whole site) honest
        // without overstating a sparse one.
        const double by_cameras =
            static_cast<double>(std::max(1, input.camera_count)) *
            static_cast<double>(window_cells);
        const double window_footprint =
            std::max(1.0, std::min(input.footprint_cells, by_cameras));

        // The fixed part: one window's stages plus the assembled raster, and the
        // assembled-grid fill_holes that runs after the last block. Whichever is
        // larger is the block's own peak; the cache comes on top of the larger.
        const double staged =
            estimate_peak_bytes(window_cells, window_footprint, input.has_node_map,
                                input.photo_cells) +
            static_cast<double>(output_cells) * (kBytesPerCellGridLabels + 4.0);
        const double filled =
            input.fill_holes ? estimate_fill_holes_peak_bytes(output_cells) : 0.0;
        const double fixed = std::max(staged, filled);
        out.estimated_window_bytes = staged;
        out.estimated_fill_bytes = filled;

        // What is left of the budget may go to decoded photographs, up to half of
        // it (the rest is headroom for the process and the document), never more
        // than the whole flight, and never above the absolute cap. At least one
        // frame: a cache smaller than the frame it is asked to hold would evict it
        // immediately and make the miss path pointless.
        //
        // An UNKNOWN budget (0) gets one frame, not the cap: the budget is
        // unknown precisely when the platform could not report physical memory,
        // which is the worst possible moment to reserve gigabytes for a cache.
        const double photo_bytes = 3.0 * std::max(1.0, input.photo_cells);
        const double flight = photo_bytes * static_cast<double>(std::max(1, input.camera_count));
        const double remaining =
            input.budget_bytes != 0
                ? std::max(0.0, static_cast<double>(input.budget_bytes) - fixed)
                : 0.0;
        const double share = kPhotoCacheBudgetFraction * remaining;
        std::uint64_t cache = static_cast<std::uint64_t>(
            std::min(share, std::min(flight, static_cast<double>(kPhotoCacheMaxBytes))));
        cache = std::max(cache, static_cast<std::uint64_t>(std::ceil(photo_bytes)));
        out.photo_cache_bytes = cache;
        out.estimated_peak_bytes = fixed + static_cast<double>(cache);
        return out;
    };

    // One piece: the whole grid fits the budget, so keep exactly the previous
    // behaviour (one block, no margin, no photo cache, no re-indexing) and the
    // output stays byte-identical. This is the path every small raster takes.
    if (input.forced_block_cells <= 0 && input.budget_bytes == 0) {
        plan.block_cells = 0;
        plan.levels = -1;
        // Exactly one decoded frame resident, as the single-piece code has
        // always held: the cache reuses it across the one block only.
        plan.photo_cache_bytes =
            static_cast<std::uint64_t>(std::ceil(3.0 * std::max(1.0, input.photo_cells)));
        plan.estimated_peak_bytes =
            estimate_peak_bytes(output_cells, input.footprint_cells, input.has_node_map,
                                input.photo_cells);
        plan.estimated_window_bytes = plan.estimated_peak_bytes;
        plan.estimated_fill_bytes =
            input.fill_holes ? estimate_fill_holes_peak_bytes(output_cells) : 0.0;
        return plan;
    }
    if (input.forced_block_cells <= 0) {
        const double whole = estimate_peak_bytes(output_cells, input.footprint_cells,
                                                 input.has_node_map, input.photo_cells);
        if (whole <= static_cast<double>(input.budget_bytes)) {
            plan.block_cells = 0;
            plan.levels = -1;
            plan.photo_cache_bytes =
                static_cast<std::uint64_t>(std::ceil(3.0 * std::max(1.0, input.photo_cells)));
            plan.estimated_peak_bytes = whole;
            plan.estimated_window_bytes = whole;
            plan.estimated_fill_bytes =
                input.fill_holes ? estimate_fill_holes_peak_bytes(output_cells) : 0.0;
            return plan;
        }
    }

    if (input.forced_block_cells > 0) {
        // An explicit override is honoured as given (a test or an expert knob);
        // the margin, the pyramid depth and the estimate still follow the same
        // rules so the result is well defined.
        return plan_for_side(input.forced_block_cells);
    }

    // Automatic: the LARGEST side whose estimate fits the budget. The budget
    // (platform::memory_budget_bytes) already reserves headroom for the rest of
    // the process, so no further fraction is taken here: a machine with the RAM
    // does fewer, bigger blocks, which is strictly better -- a block's margin
    // overhead grows with its perimeter while its interior grows with its area.
    for (int side = kMaxBlockCells; side >= kMinBlockCells; side /= 2) {
        OrthoBlockPlan candidate = plan_for_side(side);
        if (candidate.estimated_peak_bytes <= static_cast<double>(input.budget_bytes)) {
            return candidate;
        }
    }

    // Even the smallest block does not fit at this resolution: the caller falls
    // back to coarsening the cell (the estimator's last resort).
    plan = plan_for_side(kMinBlockCells);
    plan.coarsened = true;
    return plan;
}

namespace ortho {
namespace {

/// Where the ray `origin + s * direction` first meets `height`, or nullopt.
///
/// The march steps down the ray in strides of `stride_metres`, comparing against
/// the height sampled at each point: cheap, accurate to one stride, far below
/// the tolerance a cell size needs. A ray that starts inside the terrain (an
/// oblique camera on a slope) is accepted at once.
std::optional<Eigen::Vector3d> march_to_surface(const SurfaceHeight& height,
                                                const Eigen::Vector3d& origin,
                                                const Eigen::Vector3d& direction,
                                                double stride_metres,
                                                double lowest_surface_metres) {
    if (!(stride_metres > 0.0) || !std::isfinite(stride_metres)) {
        return std::nullopt;
    }
    double lowest = lowest_surface_metres;
    if (!std::isfinite(lowest)) {
        // Probe the surface around the camera's ground track. A mapping camera
        // points down, so the surface is below it: `drop` is how far the ray can
        // fall before it is certainly past the terrain.
        lowest = origin.z();
        bool probed = false;
        for (const double dx : {-1.0, 0.0, 1.0}) {
            for (const double dy : {-1.0, 0.0, 1.0}) {
                const double h = height(origin.x() + dx * stride_metres,
                                        origin.y() + dy * stride_metres);
                if (std::isfinite(h)) {
                    lowest = std::min(lowest, h);
                    probed = true;
                }
            }
        }
        if (!probed) {
            return std::nullopt;  // nothing sampled below the camera
        }
    }
    const double drop = std::max(0.0, origin.z() - lowest);
    // sec(incidence) is capped at 1 / 0.15 below, so the ray can travel at most
    // ~6.7 * drop; 8 + one stride is a safe bound and keeps the loop finite when
    // the ray leaves the surface entirely.
    const double limit = 8.0 * drop + stride_metres;

    for (double s = 0.0; s <= limit; s += stride_metres) {
        const Eigen::Vector3d p = origin + s * direction;
        const double ground = height(p.x(), p.y());
        if (!std::isfinite(ground)) {
            continue;  // off the DEM or a hole: keep marching
        }
        if (p.z() <= ground) {
            return p;  // reached the surface
        }
    }
    return std::nullopt;
}

}  // namespace

double median_ground_sample_distance(const std::vector<FootprintCamera>& cameras,
                                     const SurfaceHeight& height, double stride_metres,
                                     double lowest_surface_metres) {
    if (!height) {
        return 0.0;
    }
    std::vector<double> gsds;
    gsds.reserve(cameras.size());
    for (const FootprintCamera& camera : cameras) {
        if (!(camera.focal_px > 0.0)) {
            continue;
        }
        const Eigen::Vector3d direction = camera.direction.normalized();
        const std::optional<Eigen::Vector3d> hit =
            march_to_surface(height, camera.centre, direction, stride_metres,
                             lowest_surface_metres);
        if (!hit.has_value()) {
            continue;
        }
        // Slant range from the optical centre to the ground point.
        const double range = (*hit - camera.centre).norm();
        if (!(range > 0.0)) {
            continue;
        }
        // The image pixel's linear size perpendicular to the optical axis.
        const double per_pixel = range / camera.focal_px;
        // Incidence: the angle between the looking direction and the surface
        // normal (the plane frame's up). The gravity-aligned vertical is the
        // normal a terrain model is written against, and using it keeps the
        // measurement independent of how noisy a local DEM gradient would be.
        const double cos_incidence = std::abs(direction.z());
        if (!(cos_incidence > 0.15)) {
            continue;  // grazing: the footprint is unbounded
        }
        // The footprint grows by sec(incidence) along the line of sight; a cell
        // that has to hold one ground sample per pixel must cover that axis.
        const double gsd = per_pixel / cos_incidence;
        if (std::isfinite(gsd) && gsd > 0.0) {
            gsds.push_back(gsd);
        }
    }
    if (gsds.empty()) {
        return 0.0;
    }
    return round_sig3(median_of(std::move(gsds)));
}

}  // namespace ortho

void BuildOrthomosaicTask::declare_params(Params& params) const {
    // Names follow Metashape's public buildOrthomosaic() Python API;
    // the accepted values, defaults and the true-ortho implementation are ours.
    ParamSpec surface_data{"surface_data", ParamType::String, std::string("elevation"),
                           "Surface the ortho is projected on: \"elevation\" (the chunk DEM) or "
                           "\"model\" (a DSM rasterised from the mesh)."};
    surface_data.choices = {"elevation", "model"};
    params.declare(surface_data);

    ParamSpec blending_mode{"blending_mode", ParamType::String, std::string("mosaic"),
                            "How camera tiles combine: \"mosaic\" (multiresolution spline), "
                            "\"average\" (score-weighted mean) or \"disabled\" (seam labels "
                            "as-is)."};
    blending_mode.choices = {"mosaic", "average", "disabled"};
    params.declare(blending_mode);

    params.declare({"fill_holes", ParamType::Bool, true,
                    "Fill interior nodata holes of the colour image from the surroundings."});
    params.declare({"refine_seamlines", ParamType::Bool, true,
                    "Optimise the seam paths (alpha-expansion). False takes the per-cell "
                    "best-score camera, seams visible where the score steps."});
    ParamSpec color_correction{"color_correction", ParamType::String, std::string("none"),
                               "Per-camera colour compensation applied before seams are chosen: "
                               "\"none\" (default), \"gain\" (one brightness multiplier per camera "
                               "per channel, Brown & Lowe 2007 section 6) or \"affine\" (a "
                               "multiplier and an offset per camera per channel, solved jointly "
                               "over surface points several cameras see, robustly, so a contrast "
                               "difference is removed too). Off by default: it changes every "
                               "colour in the mosaic, so it is opt-in until the benchmark shows a "
                               "clear improvement. A Bool from an older project file is still "
                               "read (true selects \"gain\")."};
    color_correction.choices = {"none", "gain", "affine"};
    params.declare(color_correction);

    ParamSpec resolution{"resolution", ParamType::Float, 0.0,
                         "Output cell size in the output CRS's own units: metres for a "
                         "projected CRS, DEGREES for a geographic one. 0 selects the true "
                         "ground sample distance seen at each camera's image centre (median "
                         "over the enabled cameras, oblique flights included), converted to "
                         "those units and never finer than 1/8 of the surface cell."};
    resolution.minimum = 0.0;
    params.declare(resolution);

    ParamSpec projection{"projection", ParamType::Int, std::int64_t{0},
                         "Requested output EPSG code. 0 means the DEM's own; a differing "
                         "request REPROJECTS the surface once and BUILDS the mosaic in that "
                         "CRS (with surface_data=\"model\" the DSM is rasterised there "
                         "directly), so the stored orthomosaic is in the requested CRS. "
                         "Setting `projection` and `crs` to disagreeing values is an "
                         "error."};
    projection.minimum = std::int64_t{0};
    params.declare(projection);

    params.declare({"crs", ParamType::String, std::string{},
                    "Requested output coordinate system as \"EPSG:<code>\", a PROJ string "
                    "or WKT. Empty uses `projection`; when both are set they must agree."});

    params.declare({"label", ParamType::String, std::string{},
                    "Label for the new orthomosaic asset. Empty selects "
                    "\"Orthomosaic <n>\". Each run creates a new asset."});

    ParamSpec block_size{"block_size", ParamType::Int, std::int64_t{0},
                         "Side of one processing block in output cells. 0 chooses the largest "
                         "power-of-two side (512..4096) whose estimated peak fits half the "
                         "memory budget, and processes the whole grid in one piece when even "
                         "that fits. The result is the same product either way; this exists "
                         "to bound memory on very large sites."};
    block_size.minimum = std::int64_t{0};
    params.declare(block_size);
}

std::vector<Stage> BuildOrthomosaicTask::stages() const {
    // "color_correction" only runs when the parameter is set; it is its own
    // stage so the coarse whole-flight pass does not rewind the initialize bar.
    return {Stage{"initialize", 0}, Stage{"color_correction", 0}, Stage{"rectify", 0},
            Stage{"seamlines", 0}, Stage{"blend", 0}, Stage{"finalize", 0}};
}

void BuildOrthomosaicTask::apply(Document& document, Params& params,
                                 const Progress& progress) {
    // std::length_error is what a vector whose size exceeds max_size() throws
    // (and what an impossible resize throws before it ever asks for memory), so
    // it belongs to the same "this run cannot fit" story as bad_alloc: both mean
    // the request outgrew what this process can hold, not that the document is
    // malformed. Catch them together.
    try {
        run(document, params, progress);
    } catch (const std::bad_alloc&) {
        // The guard below rejects a run it can PREDICT will not fit; this catches
        // the cases it cannot (a footprint far larger than the bound, another
        // process eating the machine). Same advice, same style, and the previous
        // orthomosaic is untouched because nothing was committed.
        throw ValueError(
            "BuildOrthomosaic ran out of memory. Set a coarser resolution (the "
            "default follows the ground sample distance; try 2-4x that), or close "
            "other applications and retry.");
    } catch (const std::length_error&) {
        throw ValueError(
            "BuildOrthomosaic ran out of memory. Set a coarser resolution (the "
            "default follows the ground sample distance; try 2-4x that), or close "
            "other applications and retry.");
    }
}

void BuildOrthomosaicTask::run(Document& document, Params& params,
                               const Progress& progress) {
    progress.update("initialize", 0.0);
    const ResolvedOptions options = resolve_options(params);
    const std::string asset_label = params.get_string("label");

    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("BuildOrthomosaic needs an active chunk");
    }
    // Read through a CONST view: the non-const legacy accessor would lazily
    // create an empty elevation asset on a chunk that has none.
    const Chunk& source_chunk = *chunk;

    // -- the surface the rays are projected on -------------------------------
    // The DEM is the ortho's ground truth in every mode: its extent defines the
    // output grid, and its projection defines the plane frame. For
    // surface "model" a DSM is rasterised from the mesh into the same frame at
    // the ortho resolution and used in its place.
    const Elevation& dem = source_chunk.elevation();
    if (options.surface == "elevation" && dem.empty()) {
        throw ValueError("BuildOrthomosaic: the chunk has no elevation model; run BuildDem "
                         "first or use surface_data=\"model\"");
    }
    progress.update("initialize", 0.2);

    // Every enabled camera with a pose and an image path; the photographs are
    // decoded one at a time inside the rectify loop.
    std::vector<CameraEntry> cameras = collect_cameras(document, *chunk);
    if (cameras.empty()) {
        throw ValueError("BuildOrthomosaic: no enabled camera with a pose and an image");
    }
    progress.update("initialize", 0.5);

    // The plane frame (M12 §4.9, §4.10): the DEM's when it records a CRS,
    // unless the run requests a DIFFERENT one — then the DEM surface is
    // reprojected into that CRS once up front and the mosaic is built there
    // directly, so the photographs are sampled in the requested projection
    // exactly once (no build-then-reproject double resampling of the colours).
    // A DEM without a recorded CRS (an older project, or a DSM-only run) falls
    // back to the requested CRS, then to the auto-UTM / local rule.
    elevation::SurfaceFrame frame;
    if (!dem.empty() && dem.crs.is_set()) {
        frame = elevation::for_options(*chunk, dem.crs);
    } else {
        frame = elevation::for_options(*chunk, options.crs);
    }
    if (Log::enabled(LogLevel::Info) && options.crs.is_set() && options.crs.epsg != 0 &&
        options.crs.epsg != frame.epsg()) {
        log(LogLevel::Info) << "BuildOrthomosaic: the DEM is in EPSG " << frame.epsg()
                            << "; the requested EPSG " << options.crs.epsg
                            << " reprojects the surface and the mosaic is built there";
    }

    // -- the surface raster and the output grid ------------------------------
    GeoRaster<float> surface = dem.raster;
    float surface_nodata = dem.nodata;
    // The DTO naming the surface's CRS, so a model-surface DSM is rasterised
    // into the same plane the frame uses.
    CRS surface_crs = dem.empty() ? options.crs : dem.crs;

    // A "model" surface is a DSM rasterised from the mesh BELOW, into the frame
    // the request chose (`surface_crs`); the DEM copy made here is discarded
    // there. Reprojecting it first would cost a whole-grid resample for a
    // raster nothing reads, so in that mode the frame moves and the DSM carries
    // the requested plane, with no DEM reprojection at all. The DEM copy is
    // blanked too: it is in the OLD plane, so measuring the automatic cell on
    // it would read the wrong coordinates. The DSM supplies that measurement
    // below, exactly as it does for a run with no DEM.
    if (options.surface == "model" && !dem.empty() && dem.crs.is_set() &&
        options.crs.is_set() && frame.epsg() != options.crs.epsg) {
        surface = GeoRaster<float>{};
        surface_crs = options.crs;
        frame = elevation::for_options(*chunk, options.crs);
        progress.update("initialize", 0.6);
    } else if (!dem.empty() && dem.crs.is_set() && options.crs.is_set() &&
               frame.epsg() != options.crs.epsg) {
        // Build straight in the requested CRS: reproject the DEM surface once
        // (bilinear, nodata-aware) and continue on the result. The photographs
        // are then sampled in the requested projection, so the colours are
        // never resampled — only the surface is, exactly once.
        const crs::CoordinateSystem from = static_cast<crs::CoordinateSystem>(dem.crs);
        const crs::CoordinateSystem to = static_cast<crs::CoordinateSystem>(options.crs);
        const RasterGrid surface_grid =
            suggest_grid(dem.raster.grid, from, to, options.resolution);
        // The surface reprojection holds the DEM AND its result at once. Check
        // that pair against the budget before either exists, so an impossible
        // request fails with an explanation instead of a bare bad_alloc.
        const std::uint64_t budget = platform::memory_budget_bytes();
        const double needed = 2.0 * static_cast<double>(dem.raster.data.size()) *
                              static_cast<double>(sizeof(float));
        if (budget != 0 && needed > static_cast<double>(budget)) {
            throw ValueError(
                "BuildOrthomosaic: not enough memory to reproject the surface (needs ~" +
                platform::format_gb(needed) + " for the source DEM and its reprojected copy, "
                "budget " + platform::format_gb(static_cast<double>(budget)) +
                "); build the DEM directly in the requested CRS (BuildDem projection) or "
                "free memory.");
        }
        ReprojectOptions reproject_options;
        reproject_options.nodata_in = dem.nodata;
        reproject_options.nodata_out = dem.nodata;
        surface = reproject(dem.raster, from, to, surface_grid, reproject_options);
        surface_nodata = dem.nodata;
        surface_crs = options.crs;
        frame = elevation::for_options(*chunk, options.crs);
        progress.update("initialize", 0.6);
    }

    // Camera centres in the plane frame (cheap, and the rigorous source of the
    // occlusion march target). For a Local frame to_plane is the identity.
    std::vector<Eigen::Vector3d> plane_centres;
    plane_centres.reserve(cameras.size());
    for (const CameraEntry& camera : cameras) {
        plane_centres.push_back(frame.to_plane(camera.centre_local));
    }

    // -- the automatic output cell -------------------------------------------
    // The default is the true ground sample distance at each camera's image
    // centre on the surface: slant range / (focal_px * cos incidence), the
    // median over the enabled cameras, in metres. It is then converted into the
    // frame's own plane units (a Geographic grid is in degrees) and floored at
    // only ONE EIGHTH of the surface cell, as a sanity check on the measurement.
    // The floor is deliberately not a meaningful fraction of the DEM: the
    // measured GSD is what the user's photographs actually resolve, and a
    // quarter-cell floor overrode it whenever the DEM was coarser than the
    // imagery (a 1 m DEM forced a 0.25 m ortho from 0.4 m pixels).
    //
    // Measuring each camera's footprint on the surface (rather than using
    // "camera height / focal") is what makes an oblique flight come out right:
    // a 45-degree camera photographs a footprint about twice as long along the
    // line of sight as a nadir one at the same height.
    const auto measure_gsd_metres = [&](const GeoRaster<float>& raster, float nodata,
                                        double plane_y) {
        if (raster.empty()) {
            return 0.0;
        }
        const Eigen::Vector2d metres_per_unit = frame.metres_per_plane_unit(plane_y);
        if (!(metres_per_unit.x() > 0.0) || !(metres_per_unit.y() > 0.0)) {
            return 0.0;
        }
        // Work in metres relative to the first camera HORIZONTALLY, so a UTM
        // northing in the millions does not cost precision in the ray march.
        // The vertical stays absolute: it is a height above the surface, not a
        // large offset, and the surface sampler returns the same datum.
        const Eigen::Vector3d origin = plane_centres.front();
        std::vector<ortho::FootprintCamera> metric;
        metric.reserve(cameras.size());
        for (std::size_t i = 0; i < cameras.size(); ++i) {
            const Eigen::Vector3d offset = plane_centres[i] - origin;
            ortho::FootprintCamera camera;
            camera.centre = Eigen::Vector3d(offset.x() * metres_per_unit.x(),
                                            offset.y() * metres_per_unit.y(),
                                            plane_centres[i].z());
            // R^T (0,0,1) is the camera's viewing axis in the chunk-local frame,
            // which is metric: use it unchanged. (The plane's horizontal scale
            // applies to POSITIONS, not to this local-frame direction.)
            camera.direction = cameras[i].R.transpose() * Eigen::Vector3d(0, 0, 1);
            camera.focal_px = cameras[i].calibration.f;
            metric.push_back(camera);
        }
        const ortho::SurfaceHeight height =
            [&raster, nodata, origin, metres_per_unit](double x, double y) {
                const double h = sample_grid(raster, nodata,
                                             origin.x() + x / metres_per_unit.x(),
                                             origin.y() + y / metres_per_unit.y());
                // The raster's nodata sentinel is a finite number; the march
                // needs NaN so "unknown" is never mistaken for ground.
                return h == static_cast<double>(nodata) ? std::numeric_limits<double>::quiet_NaN()
                                                        : h;
            };
        const double stride =
            std::max(raster.grid.pixel_size_x * metres_per_unit.x(),
                     raster.grid.pixel_size_y * metres_per_unit.y());
        // The true surface minimum bounds the ray march, so a camera on a ridge
        // above a valley still finds the ground.
        double lowest = std::numeric_limits<double>::quiet_NaN();
        for (const float h : raster.data) {
            if (h != nodata && std::isfinite(h)) {
                lowest = std::isfinite(lowest) ? std::min(lowest, static_cast<double>(h))
                                               : static_cast<double>(h);
            }
        }
        return ortho::median_ground_sample_distance(metric, height, stride, lowest);
    };
    const auto plane_resolution_for = [&frame](double gsd_metres,
                                               const GeoRaster<float>& raster) {
        const double plane_y = raster.empty()
                                   ? 0.0
                                   : raster.grid.pixel_center(0, raster.grid.height / 2).y();
        const Eigen::Vector2d metres_per_unit = frame.metres_per_plane_unit(plane_y);
        const double unit_m = std::max(metres_per_unit.x(), metres_per_unit.y());
        double cell = unit_m > 0.0 ? gsd_metres / unit_m : gsd_metres;
        // Sanity floor only. The measured GSD is the ground truth the user asked
        // for; the DEM cell is just a reminder that a grid far finer than the
        // surface it resamples cannot add detail. A floor of a quarter cell
        // (the old rule) overrode the measurement: a 1 m DEM forced a 0.25 m
        // ortho even when the photographs resolved 0.4 m, quadrupling the run
        // for no visible gain. An eighth of the DEM cell only catches a
        // measurement that is wrong by an order of magnitude.
        if (!raster.empty()) {
            cell = std::max(cell, 0.125 * std::max(raster.grid.pixel_size_x,
                                                   raster.grid.pixel_size_y));
        }
        return cell;
    };

    // What the photographs themselves resolve, in metres. Kept for the memory
    // message: a run that has to be coarsened should say what it was aiming for.
    double measured_gsd_metres = 0.0;
    double resolution = options.resolution;  // plane units; 0 = automatic
    if (!(resolution > 0.0)) {
        if (surface.empty() && options.surface != "model") {
            throw ValueError("BuildOrthomosaic: the chunk has no elevation model");
        }
        if (!surface.empty()) {
            const double plane_y = surface.grid.pixel_center(0, surface.grid.height / 2).y();
            const double gsd = measure_gsd_metres(surface, surface_nodata, plane_y);
            if (gsd > 0.0) {
                measured_gsd_metres = gsd;
                resolution = plane_resolution_for(gsd, surface);
            } else {
                // No camera's principal axis reaches this DEM (a narrow
                // reconstruction strip, say). The DEM's own cell is the honest
                // fallback: it is in the right units and is no coarser than the
                // surface the ortho resamples.
                resolution = std::max(surface.grid.pixel_size_x, surface.grid.pixel_size_y);
            }
        }
        // A model-only run still has no surface here; its measured cell is
        // resolved below once the DSM exists.
    }

    if (options.surface == "model") {
        // A DSM from the mesh, in the same frame and at the ortho resolution.
        // The builder picks its own grid from the mesh extent, which is what
        // the ortho should cover when the DEM is missing or stale.
        if (source_chunk.model().empty()) {
            throw ValueError("BuildOrthomosaic: surface_data=\"model\" needs a mesh; run "
                             "BuildModel first");
        }
        elevation::DemOptions dsm_options;
        dsm_options.source = "model";
        dsm_options.interpolation = "enabled";
        // Without a DEM the mesh has to be rasterised before a footprint can be
        // measured on it; rasterise coarsely first, measure, then rebuild at the
        // measured cell. The DSM's own automatic cell (the median mesh edge) is
        // the natural provisional value.
        dsm_options.resolution = resolution;
        // The DSM must land on the same plane the frame chose: `surface_crs`,
        // which is the DEM's own CRS, or the requested one when a reprojection
        // above switched the plane (unset = the same auto rule).
        dsm_options.crs = surface_crs;
        dsm_options.tgv = false;  // a rasterised mesh needs no smoothing
        Elevation dsm =
            elevation::build_dem(*chunk, dsm_options, [&](double fraction) {
                progress.update("initialize", 0.5 + 0.2 * fraction);
            });
        if (!(resolution > 0.0)) {
            // Automatic cell on the mesh surface alone.
            const double gsd = measure_gsd_metres(dsm.raster, dsm.nodata,
                                                 dsm.raster.grid.pixel_center(
                                                     0, dsm.raster.grid.height / 2)
                                                     .y());
            if (gsd > 0.0) {
                const double cell = plane_resolution_for(gsd, dsm.raster);
                if (std::abs(cell / dsm.raster.grid.pixel_size_x - 1.0) > 0.05) {
                    dsm_options.resolution = cell;
                    dsm = elevation::build_dem(*chunk, dsm_options, [&](double fraction) {
                        progress.update("initialize", 0.7 + 0.1 * fraction);
                    });
                }
                resolution = cell;
            }
        }
        // With no DEM, the ortho grid follows the DSM the mesh produced: its cell
        // is the measured one when the footprint could be resolved, or the
        // builder's own automatic value (median mesh edge) when it could not.
        if (!(resolution > 0.0)) {
            resolution = dsm.raster.grid.pixel_size_x;
        }
        surface = dsm.raster;
        surface_nodata = dsm.nodata;
        // dsm_options.crs above is exactly what the frame was built from (the
        // DEM's CRS, or the request / auto rule when there is no DEM), so the
        // DSM and the ortho share a plane by construction.
    }

    if (!(resolution > 0.0)) {
        // Nothing measurable and no surface cell to fall back on. This is only
        // reachable from a Local frame with an empty DEM, which the checks above
        // reject; guard against a non-positive grid rather than divide by zero.
        throw ValueError("BuildOrthomosaic: cannot derive an output resolution; pass "
                         "resolution explicitly");
    }

    const std::uint64_t budget = platform::memory_budget_bytes();
    const double photo_cells = largest_photograph_cells(cameras);
    const HeightRange heights = surface_height_range(surface, surface_nodata);

    // -- memory guard, BEFORE any cell-sized allocation ----------------------
    // Refuse (or, for the automatic cell, coarsen) a run whose peak cannot fit
    // before allocating anything: a bare std::bad_alloc half-way through
    // rectification -- after the node map, the tiles and several grid-sized
    // buffers exist -- leaves the OS thrashing and tells the user nothing. The
    // estimate is the per-stage anatomy above plus the summed footprint boxes.
    //
    // The plane->local node map is itself cells * 20 B, so the estimate has to be
    // produced WITHOUT building it: `predicted_footprint_cells` evaluates the
    // exact frame transform at the two points it needs (the grid's mid column at
    // the surface's lowest and highest height) and allocates nothing.
    const auto peak_at = [&](const RasterGrid& candidate_grid) {
        const double footprint_cells =
            predicted_footprint_cells(cameras, candidate_grid, heights, frame);
        const std::uint64_t cells = static_cast<std::uint64_t>(candidate_grid.width) *
                                    static_cast<std::uint64_t>(candidate_grid.height);
        return estimate_peak_bytes(cells, footprint_cells, !frame.is_local(), photo_cells);
    };
    const auto grid_at = [&surface](double cell) {
        return ortho_grid_for(surface.grid, cell);
    };

    // -- block plan for a candidate grid -------------------------------------
    // The block layout is what actually bounds memory now, so the guard asks
    // the planner rather than comparing the whole-grid peak with the budget:
    // a grid that does not fit in one piece is fine as long as a minimum block
    // of it does. Only a grid where even that fails falls back to coarsening.
    const auto plan_at = [&](const RasterGrid& candidate_grid) {
        OrthoBlockPlanInput input;
        input.grid_width = candidate_grid.width;
        input.grid_height = candidate_grid.height;
        input.footprint_cells = predicted_footprint_cells(cameras, candidate_grid, heights, frame);
        input.photo_cells = photo_cells;
        input.camera_count = static_cast<int>(cameras.size());
        input.has_node_map = !frame.is_local();
        input.fill_holes = options.fill_holes;
        input.budget_bytes = budget;
        input.forced_block_cells = 0;
        return plan_ortho_blocks(input);
    };

    RasterGrid grid = ortho_grid_for(surface.grid, resolution);
    const OrthoBlockPlan default_plan = plan_at(grid);
    // For the human-readable diagnostics: what the whole grid in one piece
    // would have needed. The planner's own figure is the blocked peak.
    const double default_peak = peak_at(grid);
    if (budget != 0 && default_plan.coarsened) {
        if (options.resolution > 0.0) {
            throw ValueError(
                "BuildOrthomosaic needs about " + platform::format_gb(default_peak) +
                " at resolution " + format_number(resolution) +
                " m even in " + std::to_string(kMinBlockCells) + "-cell blocks; this machine has " +
                platform::format_gb(static_cast<double>(
                    platform::total_physical_memory_bytes())) +
                ". Set a coarser resolution (e.g. " +
                format_number(round_sig3(resolution * 4.0)) + " m).");
        }

        // Automatic cell: coarsen to the finest value that fits rather than
        // refusing a run the user did not ask to be at a particular scale.
        //
        // The planner's verdict is monotone non-increasing in the cell size (a
        // coarser cell means fewer output cells, and every grid-proportional and
        // footprint-proportional term shrinks with it), so the finest fitting
        // cell can be found by bisection:
        //
        //   1. bracket it -- double the cell until a minimum block fits (coarsening
        //      always ends at a 1x1 grid, whose estimate is tiny, so this
        //      terminates unless even one cell's photograph does not fit);
        //   2. bisect geometrically between the last too-fine and the last
        //      fitting cell;
        //   3. round the answer UP to three significant figures (rounding to
        //      nearest could land back on the far side of the budget) and
        //      re-check; nudge up 1% if the rounding lost the fit.
        const double default_resolution = resolution;
        double too_fine = resolution;  // known: does not fit
        double fits = 0.0;             // known: fits
        for (int step = 0; step < 48; ++step) {
            const double trial = too_fine * 2.0;
            if (!std::isfinite(trial)) {
                break;
            }
            if (!plan_at(grid_at(trial)).coarsened) {
                fits = trial;
                break;
            }
            too_fine = trial;
        }
        if (!(fits > 0.0)) {
            throw ValueError(
                "BuildOrthomosaic cannot fit even the smallest ortho in this machine's "
                "budget of " +
                platform::format_gb(static_cast<double>(budget)) +
                " (the decoded photographs alone need more). Close other applications "
                "first, or free memory and retry.");
        }
        for (int step = 0; step < 60; ++step) {
            const double mid = std::sqrt(too_fine * fits);  // geometric midpoint
            if (!(mid > too_fine) || !(mid < fits)) {
                break;
            }
            if (!plan_at(grid_at(mid)).coarsened) {
                fits = mid;
            } else {
                too_fine = mid;
            }
        }
        // Round up, then make sure the rounding did not put it back over the
        // budget. Each nudge is at least 1% coarser, so this always terminates
        // long before the cap; the cap is only a guard.
        double chosen = ceil_sig3(fits);
        bool fits_after_rounding = false;
        for (int step = 0; step < 64; ++step) {
            if (!plan_at(grid_at(chosen)).coarsened) {
                fits_after_rounding = true;
                break;
            }
            chosen = ceil_sig3(chosen * 1.01);
        }
        if (!fits_after_rounding) {
            throw ValueError(
                "BuildOrthomosaic could not settle on a resolution that fits this "
                "machine's budget of " +
                platform::format_gb(static_cast<double>(budget)) +
                ". Set a resolution explicitly, or close other applications and retry.");
        }
        grid = grid_at(chosen);
        resolution = chosen;

        log(LogLevel::Warn)
            << "BuildOrthomosaic: ground sample distance is "
            << format_number(measured_gsd_metres > 0.0 ? measured_gsd_metres
                                                       : default_resolution)
            << " m but a " << format_number(default_resolution)
            << " m ortho needs ~" << platform::format_gb(default_peak) << " (budget "
            << platform::format_gb(static_cast<double>(budget)) << "); building at "
            << format_number(resolution)
            << " m instead. Set resolution explicitly to override (needs more memory) "
               "or free memory.";
    }
    progress.update("initialize", 1.0);

    // -- block layout --------------------------------------------------------
    // A run whose whole grid fits keeps the previous single-piece behaviour. A
    // larger one is processed block by block: each block rectifies only the
    // cameras whose footprint reaches it, into block+margin buffers, and only
    // the block's interior is copied into the assembled output.
    //
    // The layout is re-planned on the FINAL grid (the guard above may have
    // coarsened it) so the estimate and the margin match what actually runs.
    OrthoBlockPlanInput plan_input;
    plan_input.grid_width = grid.width;
    plan_input.grid_height = grid.height;
    plan_input.footprint_cells = predicted_footprint_cells(cameras, grid, heights, frame);
    plan_input.photo_cells = photo_cells;
    plan_input.camera_count = static_cast<int>(cameras.size());
    plan_input.has_node_map = !frame.is_local();
    plan_input.fill_holes = options.fill_holes;
    plan_input.budget_bytes = budget;
    plan_input.forced_block_cells =
        static_cast<int>(std::max<std::int64_t>(0, params.get_int("block_size")));
    OrthoBlockPlan plan = plan_ortho_blocks(plan_input);
    if (plan.block_cells > 0) {
        log(LogLevel::Info) << "BuildOrthomosaic: processing in " << plan.columns << "x"
                            << plan.rows << " blocks of " << plan.block_cells << " (+"
                            << plan.margin_cells << " margin, mosaic levels " << plan.levels
                            << "), predicted peak ~"
                            << platform::format_gb(plan.estimated_peak_bytes);
    }

    // The plane->local fallback height is a whole-DEM median: scanned once here
    // and handed to each block's map, instead of once per block.
    const double fallback_height = frame.is_local() ? 0.0 : median_height(surface, surface_nodata);

    // The DEM's own valid height range, measured once and handed to every
    // rectification: without it orthorectify() rescans the whole DEM for the
    // footprint clip and the occlusion ceiling once per (block, camera).
    ortho::OrthoSurfaceRange surface_range;
    surface_range.lowest = heights.valid() ? heights.lowest : 0.0;
    surface_range.highest = heights.valid() ? heights.highest : 0.0;

    // The ortho grid and the DEM share a plane: in a Geographic frame both are
    // in degrees, so heights divide by degrees. The orthorectifier's occlusion
    // march compares a ray against metre heights and needs the ground metric,
    // which is why it is handed the frame's metres-per-unit explicitly.
    const double occlusion_tolerance =
        kOcclusionToleranceCells * surface.grid.pixel_size_x *
        std::max(frame.metres_per_plane_unit(grid.pixel_center(0, grid.height / 2).y()).x(),
                 frame.metres_per_plane_unit(grid.pixel_center(0, grid.height / 2).y()).y());

    // The assembled product, written block by block. RGBA + labels is the only
    // grid-sized object the blocked path ever holds, and it is allocated at the
    // FIRST block's copy: the single-piece path then has exactly the previous
    // allocation order (mosaic alive, then the result), so its peak and bytes
    // are unchanged.
    Orthomosaic result;
    result.raster.grid = grid;
    result.raster.channels = 4;
    const std::size_t output_cells =
        static_cast<std::size_t>(grid.width) * static_cast<std::size_t>(grid.height);

    std::unordered_map<std::int32_t, std::size_t> camera_slot;
    camera_slot.reserve(cameras.size());
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        camera_slot.emplace(cameras[i].camera_id, i);
    }

    PhotoCache photos(plan.photo_cache_bytes);

    // -- per-camera colour correction -----------------------------------------
    // Solved ONCE over the whole flight, before any block runs, so every block
    // applies the same model to the same camera. The model lives outside the
    // block loop for exactly that reason: a per-block solve would make
    // neighbouring blocks disagree at their shared border.
    //
    // The two modes share this one place because they are two members of one
    // family (a gain is the affine model with b = 0), must not be combined, and
    // must both be active BEFORE the seam labeller sees the tiles -- a seam cost
    // computed on uncorrected colour would pick the seam that looks cheapest
    // before the correction and then step after it.
    //
    // The estimators share the block loop's PhotoCache, so the flight is decoded
    // at most twice even when the cache is too small to keep it.
    std::vector<std::array<double, 3>> gains(
        cameras.size(), std::array<double, 3>{1.0, 1.0, 1.0});
    std::vector<ortho::CameraColour> colours(cameras.size());
    if (options.colour == ColourMode::Gain) {
        gains = estimate_camera_gains(cameras, frame, surface, surface_nodata,
                                      surface_range, occlusion_tolerance, grid,
                                      kGainSampleStride, fallback_height, photos, progress,
                                      "color_correction");
        if (Log::enabled(LogLevel::Info)) {
            double lo = 1.0;
            double hi = 1.0;
            for (const auto& gain : gains) {
                for (const double g : gain) {
                    lo = std::min(lo, g);
                    hi = std::max(hi, g);
                }
            }
            log(LogLevel::Info) << "BuildOrthomosaic: gain compensation " << lo << " .. " << hi
                                << " over " << cameras.size() << " cameras (sample stride "
                                << kGainSampleStride << ")";
        }
    } else if (options.colour == ColourMode::Affine) {
        colours = estimate_camera_colours(cameras, frame, surface, surface_nodata, surface_range,
                                          occlusion_tolerance, grid, kGainSampleStride, photos,
                                          progress, "color_correction");
    }

    // The global cell lattice: a window's cell (col, row) is global cell
    // (col + wx0, row + wy0), and its centre is evaluated on the GLOBAL origin so
    // a block reproduces the whole grid's samples bit for bit.
    ortho::OrthoCellLattice cell_lattice;
    cell_lattice.origin_x = grid.origin_x;
    cell_lattice.origin_y = grid.origin_y;

    // The whole-grid node lattice and anchor, computed once so every block's
    // plane->local map reproduces the whole-grid map bit for bit inside its
    // window. Without this a block would recompute node coordinates from its own
    // origin, which differs from the global lattice in the last bit.
    PlaneToLocalLattice lattice;
    std::optional<PlaneToLocalLattice> block_lattice;
    if (!frame.is_local() && plan.block_cells > 0) {
        lattice.origin_x = grid.origin_x;
        lattice.origin_y = grid.origin_y;
        lattice.anchor = plane_to_local_anchor(frame, grid, surface, surface_nodata,
                                              fallback_height);
        block_lattice = lattice;
    }

    const int columns = std::max(1, plan.columns);
    const int rows = std::max(1, plan.rows);
    const int total_blocks = columns * rows;
    int block_index = 0;

    // The block's window grid, built from its own cell range. Used for the block
    // being processed and (cheaply) for the next one, so the photograph cache
    // knows which decoded frames the following block will still need.
    struct BlockWindow {
        RasterGrid grid;
        int x0 = 0;  ///< first window column, in output-grid columns
        int y0 = 0;
    };
    const auto window_grid_for = [&grid](int bx0, int by0, int bx1, int by1, int margin) {
        BlockWindow block;
        block.x0 = std::max(0, bx0 - margin);
        block.y0 = std::max(0, by0 - margin);
        const int wx1 = std::min(grid.width, bx1 + margin);
        const int wy1 = std::min(grid.height, by1 + margin);
        RasterGrid& window = block.grid;
        window.pixel_size_x = grid.pixel_size_x;
        window.pixel_size_y = grid.pixel_size_y;
        window.origin_x = grid.origin_x + static_cast<double>(block.x0) * grid.pixel_size_x;
        window.origin_y = grid.origin_y - static_cast<double>(block.y0) * grid.pixel_size_y;
        window.width = wx1 - block.x0;
        window.height = wy1 - block.y0;
        return block;
    };

    // Cameras whose footprint reaches `window`. Skipping the others is what
    // makes a block cheap: their tiles would be empty anyway.
    const auto cameras_reaching =
        [&cameras, &heights, &frame](const RasterGrid& window,
                                     const std::function<Eigen::Vector3d(const Eigen::Vector3d&)>&
                                         to_local,
                                     const Eigen::Vector2d& metres_per_unit) {
            std::vector<std::size_t> reaching;
            reaching.reserve(cameras.size());
            for (std::size_t i = 0; i < cameras.size(); ++i) {
                const CameraEntry& entry = cameras[i];
                ortho::OrthoCamera probe;
                probe.calibration = entry.calibration;
                probe.R = entry.R;
                probe.t = entry.t;
                probe.plane_metres_per_unit = metres_per_unit;
                double local_low = heights.lowest;
                double local_high = heights.highest;
                if (to_local) {
                    probe.plane_to_local = to_local;
                    probe.local_to_plane = [&frame](const Eigen::Vector3d& p) {
                        return frame.to_plane(p);
                    };
                    const Eigen::Vector2d mid =
                        window.pixel_center(window.width / 2, window.height / 2);
                    const Eigen::Vector3d lo =
                        frame.to_local(Eigen::Vector3d(mid.x(), mid.y(), heights.lowest));
                    const Eigen::Vector3d hi =
                        frame.to_local(Eigen::Vector3d(mid.x(), mid.y(), heights.highest));
                    local_low = std::min(lo.z(), hi.z());
                    local_high = std::max(lo.z(), hi.z());
                }
                const ortho::TileBounds bounds = ortho::camera_footprint_bounds(
                    window, probe, local_low, local_high, 2);
                if (bounds.width <= 0 || bounds.height <= 0) {
                    continue;  // does not reach this window
                }
                reaching.push_back(i);
            }
            return reaching;
        };

    for (int br = 0; br < rows; ++br) {
        for (int bc = 0; bc < columns; ++bc, ++block_index) {
            if (progress.is_cancelled()) {
                throw Error("BuildOrthomosaic cancelled");
            }
            const int side = plan.block_cells > 0 ? plan.block_cells : std::max(grid.width, grid.height);
            const int bx0 = bc * side;
            const int by0 = br * side;
            const int bx1 = std::min(grid.width, bx0 + side);
            const int by1 = std::min(grid.height, by0 + side);
            if (bx0 >= bx1 || by0 >= by1) {
                continue;
            }
            const int margin = plan.block_cells > 0 ? plan.margin_cells : 0;
            const BlockWindow block_window = window_grid_for(bx0, by0, bx1, by1, margin);
            const RasterGrid& window = block_window.grid;
            const int wx0 = block_window.x0;
            const int wy0 = block_window.y0;

            // Per-block plane->local map: the node lattice follows the block, so
            // a large output never carries the whole-grid map (20 B/cell).
            if (block_lattice.has_value()) {
                block_lattice->col_offset = wx0;
                block_lattice->row_offset = wy0;
            }
            cell_lattice.col_offset = wx0;
            cell_lattice.row_offset = wy0;
            const std::function<Eigen::Vector3d(const Eigen::Vector3d&)> to_local =
                make_plane_to_local(frame, window, surface, surface_nodata, fallback_height,
                                    block_lattice.has_value() ? &*block_lattice : nullptr);
            const Eigen::Vector2d metres_per_unit = frame.metres_per_plane_unit(
                window.pixel_center(0, window.height / 2).y());

            // Cameras whose footprint reaches this window. Skipping the others
            // is what makes a block cheap: their tiles would be empty anyway.
            const std::vector<std::size_t> block_cameras =
                cameras_reaching(window, to_local, metres_per_unit);

            // Keep the frames the NEXT block will want (raster order: the block
            // to the right, or the first block of the next row) near the head of
            // the LRU, so this block's decodes evict frames that are already done
            // with rather than ones the run is about to need again.
            //
            // The footprint probe reuses THIS block's plane->local map rather
            // than building the next window's: a prefetch is only an ordering
            // hint, so a map clamped to the neighbouring grid is close enough,
            // and building a second block-sized node map (20 B/cell, the largest
            // single object in the plan) beside this one would cost exactly the
            // memory the blocking exists to save.
            if (photos.budget_bytes() != 0) {
                int next_bc = bc + 1;
                int next_br = br;
                if (next_bc >= columns) {
                    next_bc = 0;
                    ++next_br;
                }
                if (next_br < rows) {
                    const int nx0 = next_bc * side;
                    const int ny0 = next_br * side;
                    const BlockWindow next_window =
                        window_grid_for(nx0, ny0, std::min(grid.width, nx0 + side),
                                        std::min(grid.height, ny0 + side), margin);
                    for (const std::size_t slot :
                         cameras_reaching(next_window.grid, to_local, metres_per_unit)) {
                        photos.touch(slot);
                    }
                }
            }

            std::vector<ortho::OrthoTile> tiles;
            tiles.reserve(block_cameras.size());
            for (std::size_t k = 0; k < block_cameras.size(); ++k) {
                if (progress.is_cancelled()) {
                    throw Error("BuildOrthomosaic cancelled");
                }
                const CameraEntry& entry = cameras[block_cameras[k]];
                const Image& image = photos.get(block_cameras[k], entry.image_path);
                if (image.empty()) {
                    throw ValueError("BuildOrthomosaic: camera " +
                                     std::to_string(entry.camera_id) +
                                     " has an unreadable image");
                }
                ortho::OrthoCamera camera;
                camera.calibration = entry.calibration;
                camera.R = entry.R;
                camera.t = entry.t;
                camera.image = &image;
                camera.plane_metres_per_unit = metres_per_unit;
                if (to_local) {
                    camera.plane_to_local = to_local;
                    camera.local_to_plane = [&frame](const Eigen::Vector3d& p) {
                        return frame.to_plane(p);
                    };
                    camera.plane_centre = plane_centres[block_cameras[k]];
                }
                tiles.push_back(ortho::orthorectify(
                    surface, surface_nodata, window, camera, static_cast<int>(k),
                    occlusion_tolerance, true, &surface_range,
                    plan.block_cells > 0 ? &cell_lattice : nullptr));
                // Apply the camera's precomputed correction to the tile just
                // rectified, BEFORE the seam labeller reads it. The model is a
                // whole-flight constant (see estimate_camera_gains /
                // estimate_camera_colours), so a block and the whole grid apply
                // exactly the same coefficients and no block border disagrees.
                // Applying here rather than after the blend is what makes the
                // seam cost see corrected colours, which is the point of the
                // stage: a seam chosen on uncorrected colour is chosen for the
                // wrong reason.
                if (options.colour == ColourMode::Gain) {
                    ortho::apply_gain(tiles.back(), gains[block_cameras[k]]);
                } else if (options.colour == ColourMode::Affine) {
                    ortho::apply_colour(tiles.back(), colours[block_cameras[k]]);
                }
                progress.update("rectify",
                                (static_cast<double>(block_index) +
                                 static_cast<double>(k + 1) /
                                     static_cast<double>(block_cameras.size())) /
                                    static_cast<double>(total_blocks));
            }

            // -- seam continuity ---------------------------------------------
            // Blocks run in raster order, so every cell above this block or to
            // its left is already decided. Those labels are pinned: the margin
            // keeps the seam paths continuous across a block border instead of
            // letting the neighbouring block re-decide the overlap.
            //
            // Only an INNER BAND of the overlap is pinned, not the whole margin.
            // What continuity needs is the cell just outside the border: this
            // block's first output column is decided against its left neighbour,
            // so pinning one band of margin beside the block carries the earlier
            // block's decision across. The rest of the margin is at the WINDOW
            // edge, where a cell sees only half its neighbourhood, so anything
            // decided there is a boundary artefact; leaving it free keeps the
            // optimiser's context intact, and those cells are never written to
            // the output (only this block's own region is copied out).
            std::vector<std::int32_t> seed;
            std::vector<std::uint8_t> frozen;
            if (plan.block_cells > 0) {
                const std::size_t window_cells =
                    static_cast<std::size_t>(window.width) * static_cast<std::size_t>(window.height);
                seed.assign(window_cells, -1);
                frozen.assign(window_cells, 0);
                // The pinned band around a decided region: keep it to the part of
                // the margin the seam optimiser itself operates over, so its own
                // multi-scale pass still has free cells to work in.
                const int pin_band = std::max(1, ortho::pyramid_support_cells(plan.levels));
                for (int row = 0; row < window.height; ++row) {
                    for (int col = 0; col < window.width; ++col) {
                        const int gx = wx0 + col;
                        const int gy = wy0 + row;
                        // Decided already iff the cell's own block precedes this
                        // one in raster order. The right and bottom margin
                        // belongs to later blocks and stays free.
                        const int cell_block_col = gx / side;
                        const int cell_block_row = gy / side;
                        const int cell_index = cell_block_row * columns + cell_block_col;
                        if (cell_index >= block_index) {
                            continue;
                        }
                        // Distance to this block's own region, in cells. Only the
                        // band nearest it is pinned; beyond that the optimiser is
                        // free again so it can carry a seam across the border
                        // rather than inherit wherever the previous window's edge
                        // happened to put one.
                        const int dx = std::max(0, std::max(bx0 - gx, gx - (bx1 - 1)));
                        const int dy = std::max(0, std::max(by0 - gy, gy - (by1 - 1)));
                        if (std::max(dx, dy) > pin_band) {
                            continue;
                        }
                        const std::size_t wc = static_cast<std::size_t>(row) *
                                                   static_cast<std::size_t>(window.width) +
                                               static_cast<std::size_t>(col);
                        const std::int32_t camera_id =
                            result.labels[static_cast<std::size_t>(gy) *
                                              static_cast<std::size_t>(grid.width) +
                                          static_cast<std::size_t>(gx)];
                        std::int32_t tile_index = -1;
                        if (camera_id >= 0) {
                            const auto it = camera_slot.find(camera_id);
                            if (it == camera_slot.end()) {
                                continue;
                            }
                            const auto tile_it = std::find(block_cameras.begin(),
                                                           block_cameras.end(), it->second);
                            if (tile_it == block_cameras.end()) {
                                continue;
                            }
                            tile_index = static_cast<std::int32_t>(
                                std::distance(block_cameras.begin(), tile_it));
                            // Only pin a camera that this window's tile actually
                            // covers the cell with. A decision made on the whole
                            // window the earlier block had is not automatically
                            // renderable here (the window clip and the visibility
                            // test can differ at the very edge), and a pin the
                            // labeller would drop is better not applied at all.
                            const ortho::OrthoTile& tile =
                                tiles[static_cast<std::size_t>(tile_index)];
                            if (!tile.contains(col, row) ||
                                tile.valid[tile.index(col, row)] == 0) {
                                continue;
                            }
                        }
                        seed[wc] = tile_index;
                        frozen[wc] = 1;
                    }
                }
            }

            std::vector<std::int32_t> labels;
            ortho::SeamOptions seam_options;
            if (!options.refine_seamlines) {
                seam_options.iterations = 0;
            }
            labels = ortho::assign_seams(window.width, window.height, tiles, seam_options,
                                        frozen.empty() ? nullptr : &seed,
                                        frozen.empty() ? nullptr : &frozen);

            // Blocks run in raster order and the overlap is pinned, so every
            // inherited cell must still carry the label the earlier block
            // decided. A change here would be a seam restarting at a block
            // border, which is exactly what the pin exists to prevent.
            std::size_t pinned_cells = 0;
            std::size_t pinned_changed = 0;
            if (!frozen.empty()) {
                for (std::size_t wc = 0; wc < frozen.size(); ++wc) {
                    if (frozen[wc] == 0) {
                        continue;
                    }
                    ++pinned_cells;
                    if (labels[wc] != seed[wc]) {
                        // The pin is a hard constraint on a cell this window can
                        // render, so this is an optimiser bug rather than a
                        // legitimate drop. Restore the earlier block's decision:
                        // the seam must not restart, and the cell is only in this
                        // block's margin so its colour is not written either way.
                        ++pinned_changed;
                        labels[wc] = seed[wc];
                    }
                }
            }
            if (pinned_changed != 0) {
                log(LogLevel::Warn) << "BuildOrthomosaic: block " << block_index << " changed "
                                    << pinned_changed << " of " << pinned_cells
                                    << " inherited seam labels";
            }
            if (Log::enabled(LogLevel::Info)) {
                log(LogLevel::Info) << "BuildOrthomosaic: block " << block_index << " pinned="
                                    << pinned_cells << " conflicts=" << pinned_changed
                                    << " cameras=" << block_cameras.size();
            }
            progress.update("seamlines",
                            static_cast<double>(block_index + 1) /
                                static_cast<double>(total_blocks));

            const ortho::OrthoMosaic mosaic = ortho::blend(
                window.width, window.height, tiles, labels, options.blending, plan.levels);
            progress.update("blend", static_cast<double>(block_index + 1) /
                                         static_cast<double>(total_blocks));

            // Copy the interior only: the margin exists to make the seams and
            // the blend continuous, it is not part of this block's output.
            if (result.raster.data.empty()) {
                result.raster.data.assign(output_cells * 4U, 0);
                result.labels.assign(output_cells, -1);
            }
            for (int row = by0; row < by1; ++row) {
                const std::size_t window_row =
                    static_cast<std::size_t>(row - wy0) * static_cast<std::size_t>(window.width);
                const std::size_t output_row =
                    static_cast<std::size_t>(row) * static_cast<std::size_t>(grid.width);
                for (int col = bx0; col < bx1; ++col) {
                    const std::size_t wc = window_row + static_cast<std::size_t>(col - wx0);
                    const std::size_t oc = output_row + static_cast<std::size_t>(col);
                    const std::int32_t label = mosaic.labels[wc];
                    result.labels[oc] =
                        label >= 0 ? cameras[block_cameras[static_cast<std::size_t>(label)]]
                                         .camera_id
                                   : -1;
                    result.raster.data[oc * 4 + 0] = mosaic.rgb[wc * 3 + 0];
                    result.raster.data[oc * 4 + 1] = mosaic.rgb[wc * 3 + 1];
                    result.raster.data[oc * 4 + 2] = mosaic.rgb[wc * 3 + 2];
                    result.raster.data[oc * 4 + 3] = mosaic.valid[wc] != 0 ? 255 : 0;
                }
            }
        }
    }

    if (options.fill_holes) {
        // Fill interior nodata holes in each colour channel from the
        // surroundings, ONCE on the assembled raster: `interior_only` decides
        // reachability from the output's own border, which a block cannot know.
        // Natural-neighbour keeps the colour smooth across the seams the hole
        // borders. Filled cells become opaque, otherwise the option would be
        // invisible in any RGBA export.
        elevation::HoleFillOptions fill;
        fill.method = elevation::HoleFill::NaturalNeighbour;
        fill.interior_only = true;
        for (int channel = 0; channel < 3; ++channel) {
            std::vector<float> values(output_cells, 0.0F);
            std::vector<std::uint8_t> channel_valid(output_cells, 0);
            for (std::size_t i = 0; i < output_cells; ++i) {
                values[i] = static_cast<float>(result.raster.data[i * 4 +
                                                                  static_cast<std::size_t>(channel)]);
                channel_valid[i] = result.raster.data[i * 4 + 3] != 0 ? 1 : 0;
            }
            elevation::fill_holes(grid.width, grid.height, values, channel_valid, fill);
            for (std::size_t i = 0; i < output_cells; ++i) {
                if (channel_valid[i] != 0) {
                    const float c = std::clamp(values[i], 0.0F, 255.0F);
                    result.raster.data[i * 4 + static_cast<std::size_t>(channel)] =
                        static_cast<std::uint8_t>(std::lround(c));
                    result.raster.data[i * 4 + 3] = 255;
                }
            }
        }
    }
    result.set_epsg(frame.epsg());
#if defined(ALPHASHAPE_USE_PROJ)
    if (!frame.is_local()) {
        result.crs.wkt = frame.crs().to_wkt();
    }
#endif
    result.surface = options.surface;
    progress.update("finalize", 1.0);

    if (Log::enabled(LogLevel::Info)) {
        std::size_t covered = 0;
        for (std::size_t i = 0; i < output_cells; ++i) {
            covered += result.raster.data[i * 4 + 3] != 0 ? 1U : 0U;
        }
        log(LogLevel::Info) << "BuildOrthomosaic: " << grid.width << "x" << grid.height
                            << " cells at " << format_number(resolution) << " m, " << covered
                            << " covered, blending " << params.get_string("blending_mode")
                            << ", blocks " << columns << "x" << rows << " of "
                            << plan.block_cells << " (+" << plan.margin_cells
                            << " margin), epsg " << result.epsg();
        // Decode accounting: a cache too small for the flight shows up here as
        // decodes well above the camera count, which is the number to watch if a
        // future change to the plan or the block order starts thrashing.
        log(LogLevel::Info) << "BuildOrthomosaic: photographs decoded " << photos.misses()
                            << " of " << (photos.hits() + photos.misses())
                            << " (cache " << platform::format_gb(
                                   static_cast<double>(photos.budget_bytes()))
                            << ", " << cameras.size() << " cameras)";
    }
    // Each run APPENDS a new orthomosaic asset and makes it active, so an
    // earlier orthomosaic is never overwritten.
    OrthomosaicAsset& asset = chunk->add_orthomosaic(asset_label);
    *asset.data = std::move(result);
}

}  // namespace alphashape
