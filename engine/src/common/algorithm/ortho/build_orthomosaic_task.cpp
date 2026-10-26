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
#include "common/algorithm/ortho/blend.hpp"
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

std::function<Eigen::Vector3d(const Eigen::Vector3d&)> make_plane_to_local(
    const elevation::SurfaceFrame& frame, const RasterGrid& grid,
    const GeoRaster<float>& heights, float nodata, double fallback_height) {
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
    double anchor_height = sample_grid(heights, nodata, map->x0, map->y0);
    if (!std::isfinite(anchor_height) || anchor_height == static_cast<double>(nodata)) {
        anchor_height = fallback;
    }
    map->anchor = frame.to_local(Eigen::Vector3d(map->x0, map->y0, anchor_height));
    const Eigen::Vector3d anchor = map->anchor;

    for (int row = 0; row < map->rows; ++row) {
        for (int col = 0; col < map->cols; ++col) {
            const double x = map->x0 + static_cast<double>(col) * map->dx;
            const double y = map->y0 - static_cast<double>(row) * map->dy;
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

/// The validated parameter bundle.
struct ResolvedOptions {
    std::string surface = "elevation";
    ortho::Blending blending = ortho::Blending::Mosaic;
    bool fill_holes = true;
    bool refine_seamlines = true;
    double resolution = 0.0;  ///< 0 = automatic
    int projection = 0;
};

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
    options.resolution = params.get_float("resolution");
    if (options.resolution < 0.0) {
        throw ValueError(
            "BuildOrthomosaic: resolution must be >= 0 (0 selects an automatic value)");
    }
    options.projection = static_cast<int>(params.get_int("projection"));
    if (options.projection < 0) {
        throw ValueError("BuildOrthomosaic: projection must be an EPSG code >= 0");
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
/// A block's estimate must stay inside this fraction of the budget. The rest of
/// the budget holds the assembled output (8 B/cell of the OUTPUT grid), the
/// decoded-photograph cache and the rest of the process (the document, the DEM,
/// the plane->local map's own pool).
constexpr double kBlockBudgetFraction = 0.5;
/// At most this fraction of the budget goes to decoded photographs.
constexpr double kPhotoCacheBudgetFraction = 0.15;
/// The absolute ceiling on the decoded-photograph cache, whatever the machine's
/// budget: beyond a handful of frames the reuse stops paying for the bytes.
constexpr std::uint64_t kPhotoCacheMaxBytes = 512ULL * 1024ULL * 1024ULL;
/// The absolute per-block peak the automatic plan aims for. Half a budget could
/// be 12 GB on a big workstation, which is not a bound anyone asked for; the
/// owner's target (a 28 M-cell, 36-photo, 1.9 cm site) is under 2 GB, and a
/// 1024-cell block with a 192-cell margin meets it with room to spare.
constexpr double kBlockPeakTargetBytes = 1.5e9;

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

}  // namespace

// -- block plan --------------------------------------------------------------

int block_margin_cells(int block_cells) {
    // A quarter of the side, clamped to [16, 192]. The upper bound is the mosaic
    // blend's reach: its pyramid stops at 6 levels (the level_count() cap), and
    // two 5-tap blurs per reduce and expand touch about 2 * 2^levels cells, so
    // 128 cells of margin hold the whole pyramid support at full depth. The
    // extra headroom absorbs the block plan's rounding and the seam labeller's
    // own 4x coarse level.
    if (block_cells <= 0) {
        return 0;
    }
    return std::clamp(block_cells / 4, 16, 192);
}

double estimate_block_peak_bytes(std::uint64_t window_cells, double window_footprint_cells,
                                 bool has_node_map, double photo_cells,
                                 std::uint64_t output_cells,
                                 std::uint64_t photo_cache_bytes) {
    // The window's stages are the estimator the whole-grid path already uses,
    // with the window's (block + margin) cell count and its own footprint. The
    // assembled output raster (RGBA, 4 B) plus its seam labels (int32, 4 B) is
    // resident from the first block on and grows to the full output grid.
    const double window = estimate_peak_bytes(window_cells, window_footprint_cells, has_node_map,
                                              photo_cells);
    const double assembled = static_cast<double>(output_cells) *
                             (kBytesPerCellGridLabels + 4.0 /* result RGBA */);
    return window + assembled + static_cast<double>(photo_cache_bytes);
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
        out.margin_cells = block_margin_cells(side);
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
        // A decoded-photograph cache can only ever hold the whole flight, and
        // its own share of the budget is capped so a huge machine does not trade
        // the memory saved by blocking back for cached frames.
        const double photo_bytes = 3.0 * std::max(1.0, input.photo_cells);
        std::uint64_t cache = 0;
        if (input.budget_bytes != 0) {
            const double share = kPhotoCacheBudgetFraction *
                                 static_cast<double>(input.budget_bytes);
            const double flight = photo_bytes * static_cast<double>(input.camera_count);
            cache = static_cast<std::uint64_t>(
                std::min(share, std::min(flight, static_cast<double>(kPhotoCacheMaxBytes))));
        }
        out.photo_cache_bytes = cache;
        out.estimated_peak_bytes = estimate_block_peak_bytes(
            window_cells, window_footprint, input.has_node_map, input.photo_cells,
            output_cells, cache);
        // The mosaic pyramid depth every block applies: the whole grid's depth,
        // never deeper than the margin can carry.
        const int margin_levels = static_cast<int>(
            std::floor(std::log2(static_cast<double>(std::max(1, out.margin_cells))))) - 1;
        out.levels = std::max(0, margin_levels);
        if (width > 0 && height > 0) {
            const int whole_levels = ortho::pyramid_level_count(
                static_cast<int>(width), static_cast<int>(height));
            out.levels = std::min(out.levels, whole_levels);
        }
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
            return plan;
        }
    }

    if (input.forced_block_cells > 0) {
        // An explicit override is honoured as given (a test or an expert knob);
        // the margin, the pyramid depth and the estimate still follow the same
        // rules so the result is well defined.
        return plan_for_side(input.forced_block_cells);
    }

    // Automatic: the largest side whose estimate stays inside the smaller of
    // half the budget and the absolute per-block target. The target is what
    // keeps the owner's 28 M-cell / 36-photo case under 2 GB on a machine whose
    // budget would otherwise allow much bigger blocks.
    const double limit = std::min(kBlockBudgetFraction * static_cast<double>(input.budget_bytes),
                                  kBlockPeakTargetBytes);
    for (int side = kMaxBlockCells; side >= kMinBlockCells; side /= 2) {
        OrthoBlockPlan candidate = plan_for_side(side);
        if (candidate.estimated_peak_bytes <= limit) {
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

    ParamSpec resolution{"resolution", ParamType::Float, 0.0,
                         "Output cell size in projection units. 0 selects the true ground "
                         "sample distance seen at each camera's image centre (median over "
                         "the enabled cameras, oblique flights included), in the DEM's own "
                         "units and never finer than DEM cell / 4."};
    resolution.minimum = 0.0;
    params.declare(resolution);

    ParamSpec projection{"projection", ParamType::Int, std::int64_t{0},
                         "Output EPSG code. 0 uses the DEM's own projection; any other value "
                         "must equal it (reproject the DEM first)."};
    projection.minimum = std::int64_t{0};
    params.declare(projection);

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
    return {Stage{"initialize", 0}, Stage{"rectify", 0}, Stage{"seamlines", 0},
            Stage{"blend", 0}, Stage{"finalize", 0}};
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
    if (options.projection != 0 && !dem.empty() && options.projection != dem.epsg) {
        // The DEM is in another CRS: a projection change here would silently
        // misplace every ray.
        throw ValueError(
            "BuildOrthomosaic: projection (" + std::to_string(options.projection) +
            ") differs from the DEM's EPSG (" + std::to_string(dem.epsg) +
            "); reproject the DEM first (a projection change is not applied here)");
    }
    progress.update("initialize", 0.2);

    // Every enabled camera with a pose and an image path; the photographs are
    // decoded one at a time inside the rectify loop.
    std::vector<CameraEntry> cameras = collect_cameras(document, *chunk);
    if (cameras.empty()) {
        throw ValueError("BuildOrthomosaic: no enabled camera with a pose and an image");
    }
    progress.update("initialize", 0.5);

    // The plane frame: the DEM's (0 = auto for a georeferenced chunk, identity
    // for an unreferenced one). An unreferenced chunk's DEM epsg is 0 too.
    const int epsg = options.projection != 0 ? options.projection : dem.epsg;
    const elevation::SurfaceFrame frame = elevation::SurfaceFrame::for_chunk(*chunk, epsg);
    if (epsg != 0 && frame.epsg() != epsg) {
        throw ValueError("BuildOrthomosaic: the chunk cannot be expressed in EPSG " +
                         std::to_string(epsg));
    }

    // -- the surface raster and the output grid ------------------------------
    GeoRaster<float> surface = dem.raster;
    float surface_nodata = dem.nodata;

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
        dsm_options.epsg = epsg;
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
        const auto fits_at = [&](double cell) { return !plan_at(grid_at(cell)).coarsened; };
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

    const int columns = std::max(1, plan.columns);
    const int rows = std::max(1, plan.rows);
    const int total_blocks = columns * rows;
    int block_index = 0;

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
            const int wx0 = std::max(0, bx0 - margin);
            const int wy0 = std::max(0, by0 - margin);
            const int wx1 = std::min(grid.width, bx1 + margin);
            const int wy1 = std::min(grid.height, by1 + margin);

            RasterGrid window;
            window.pixel_size_x = grid.pixel_size_x;
            window.pixel_size_y = grid.pixel_size_y;
            window.origin_x = grid.origin_x + static_cast<double>(wx0) * grid.pixel_size_x;
            window.origin_y = grid.origin_y - static_cast<double>(wy0) * grid.pixel_size_y;
            window.width = wx1 - wx0;
            window.height = wy1 - wy0;

            // Per-block plane->local map: the node lattice follows the block, so
            // a large output never carries the whole-grid map (20 B/cell).
            const std::function<Eigen::Vector3d(const Eigen::Vector3d&)> to_local =
                make_plane_to_local(frame, window, surface, surface_nodata, fallback_height);
            const Eigen::Vector2d metres_per_unit = frame.metres_per_plane_unit(
                window.pixel_center(0, window.height / 2).y());

            // Cameras whose footprint reaches this window. Skipping the others
            // is what makes a block cheap: their tiles would be empty anyway.
            std::vector<std::size_t> block_cameras;
            block_cameras.reserve(cameras.size());
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
                block_cameras.push_back(i);
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
                tiles.push_back(ortho::orthorectify(surface, surface_nodata, window, camera,
                                                   static_cast<int>(k), occlusion_tolerance));
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
            std::vector<std::int32_t> seed;
            std::vector<std::uint8_t> frozen;
            if (plan.block_cells > 0) {
                const std::size_t window_cells =
                    static_cast<std::size_t>(window.width) * static_cast<std::size_t>(window.height);
                seed.assign(window_cells, -1);
                frozen.assign(window_cells, 0);
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
                        }
                        const std::size_t wc = static_cast<std::size_t>(row) *
                                                   static_cast<std::size_t>(window.width) +
                                               static_cast<std::size_t>(col);
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
                        ++pinned_changed;
                        // The pin is a hard constraint, so this can only mean the
                        // pinned tile does not cover the cell (dropped by
                        // assign_seams) or an optimiser bug. Keep the earlier
                        // block's decision either way: the committed output for
                        // the cell is already written.
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
    result.epsg = frame.epsg();
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
                            << " margin), epsg " << result.epsg;
    }
    // Each run APPENDS a new orthomosaic asset and makes it active, so an
    // earlier orthomosaic is never overwritten.
    OrthomosaicAsset& asset = chunk->add_orthomosaic(asset_label);
    *asset.data = std::move(result);
}

}  // namespace alphashape
