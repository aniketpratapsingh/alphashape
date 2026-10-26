#include "common/algorithm/ortho/build_orthomosaic_task.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
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

/// The north-up grid covering `extent` at `resolution`, its origin snapped
/// outward to a multiple of the resolution (row 0 northernmost).
RasterGrid ortho_grid_for(const RasterGrid& extent, double resolution) {
    RasterGrid grid;
    grid.pixel_size_x = resolution;
    grid.pixel_size_y = resolution;
    grid.origin_x = std::floor(extent.origin_x / resolution) * resolution;
    grid.origin_y = std::ceil(extent.origin_y / resolution) * resolution;
    const double east =
        extent.origin_x + static_cast<double>(extent.width) * extent.pixel_size_x;
    const double south =
        extent.origin_y - static_cast<double>(extent.height) * extent.pixel_size_y;
    grid.width =
        std::max(1, static_cast<int>(std::ceil((east - grid.origin_x) / resolution)));
    grid.height =
        std::max(1, static_cast<int>(std::ceil((grid.origin_y - south) / resolution)));
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
        const double node_height =
            w00 * height[i00] + w10 * height[i10] + w01 * height[i01] + w11 * height[i11];
        const double scale =
            w00 * vertical[i00] + w10 * vertical[i10] + w01 * vertical[i01] +
            w11 * vertical[i11];
        out.z() += (p.z() - node_height) * scale;
        return out;
    }
};

}  // namespace

std::function<Eigen::Vector3d(const Eigen::Vector3d&)> make_plane_to_local(
    const elevation::SurfaceFrame& frame, const RasterGrid& grid,
    const GeoRaster<float>& heights, float nodata) {
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
    double fallback = median_height(heights, nodata);
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

/// Bytes per output cell the fine pipeline holds at its PEAK, stage by stage
/// (the stages are sequential, so the peak is the largest one, not the sum):
///
///   * seamlines: labels int32 (4) + `best` double (8) + `any_valid` u8 (1) +
///     the two checkerboard parity lists of size_t (8)         = 21
///   * blend mosaic: labels (4) + out.rgb (3) + out.valid (1) + the three
///     accumulator pyramids at sum(1/4^k)*4B (16) + their collapse result (12)
///     + one tile's Gaussian/Laplacian/mask planes (~20)        = 56
///   * finalize: mosaic (4) + result RGBA (4) + fill_holes' per-channel
///     float+u8 pair, EDT and masks (~19)                       = 27
///
/// 56 B/cell is the blend stage, i.e. the peak. The tiles are counted
/// separately: they are cropped to the camera footprints, so their cost is a
/// footprint area, not C x the grid.
constexpr double kBytesPerCellFine = 56.0;

/// Bytes per output-grid NODE in the plane->local map (float3 + 2 floats) when
/// the chunk is georeferenced: about one node per output cell.
constexpr double kBytesPerNodeMap = 20.0;

/// Bytes per camera footprint cell in a rectified tile (rgb u8x3 + valid u8 +
/// score float).
constexpr double kBytesPerTileCell = 8.0;

/// A coarse seamline pyramid level holds its tiles at 1/factor^2 of the fine
/// area; the two configured levels (2x and 4x) therefore add 1/4 + 1/16 of the
/// footprint.
constexpr double kCoarseFootprintFraction = 0.25 + 0.0625;

/// Estimates the peak memory BuildOrthomosaic needs, in bytes.
///
/// `footprint_cells` is the summed area of every camera's tile crop box (the
/// rectified-tile cost); it is predicted from camera_footprint_bounds() before
/// any tile is allocated.
double estimate_peak_bytes(std::uint64_t cells, double footprint_cells, bool has_node_map) {
    // The tile set is resident throughout the seam and blend stages, so it adds
    // to whichever stage is the peak.
    const double tiles =
        footprint_cells * kBytesPerTileCell * (1.0 + kCoarseFootprintFraction);
    // The plane->local node map is built before rectification and released with
    // the task's locals at the end, so it also overlaps every stage.
    const double node_map =
        has_node_map ? static_cast<double>(cells) * kBytesPerNodeMap : 0.0;
    // The blend stage (everything full-grid) is the peak; the final RGBA raster
    // and the mosaic exist together during finalize, which the per-cell figure
    // already covers.
    const double stage = static_cast<double>(cells) * kBytesPerCellFine;
    return tiles + node_map + stage;
}

/// The footprint area every camera's tile crop box will cover, padded the way
/// the rectifier pads it (two cells, and the cell rounding on each edge). This
/// is the same bound orthorectify() allocates against, so the guard's estimate
/// cannot be smaller than the allocation that follows.
double predicted_footprint_cells(const std::vector<CameraEntry>& cameras,
                                 const RasterGrid& grid, const GeoRaster<float>& surface,
                                 float surface_nodata, const elevation::SurfaceFrame& frame,
                                 const std::function<Eigen::Vector3d(const Eigen::Vector3d&)>&
                                     to_local) {
    double lowest = std::numeric_limits<double>::infinity();
    double highest = -std::numeric_limits<double>::infinity();
    for (const float h : surface.data) {
        if (h == surface_nodata || std::isnan(h)) {
            continue;
        }
        lowest = std::min(lowest, static_cast<double>(h));
        highest = std::max(highest, static_cast<double>(h));
    }
    if (!std::isfinite(lowest) || !std::isfinite(highest)) {
        return static_cast<double>(grid.width) * static_cast<double>(grid.height);
    }

    const Eigen::Vector2d metres_per_unit =
        frame.metres_per_plane_unit(grid.pixel_center(0, grid.height / 2).y());
    double total = 0.0;
    for (const CameraEntry& entry : cameras) {
        ortho::OrthoCamera camera;
        camera.calibration = entry.calibration;
        camera.R = entry.R;
        camera.t = entry.t;
        camera.plane_metres_per_unit = metres_per_unit;
        double local_low = lowest;
        double local_high = highest;
        if (to_local) {
            camera.plane_to_local = to_local;
            camera.local_to_plane = [&frame](const Eigen::Vector3d& p) {
                return frame.to_plane(p);
            };
            const Eigen::Vector3d mid = grid.pixel_center(grid.width / 2, grid.height / 2);
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

}  // namespace

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
}

std::vector<Stage> BuildOrthomosaicTask::stages() const {
    return {Stage{"initialize", 0}, Stage{"rectify", 0}, Stage{"seamlines", 0},
            Stage{"blend", 0}, Stage{"finalize", 0}};
}

void BuildOrthomosaicTask::apply(Document& document, Params& params,
                                 const Progress& progress) {
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
    // frame's own plane units (a Geographic grid is in degrees) and floored at a
    // quarter of the surface cell, so the ortho is never finer than the surface
    // it is resampled from.
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
        if (!raster.empty()) {
            cell = std::max(cell, 0.25 * std::max(raster.grid.pixel_size_x,
                                                  raster.grid.pixel_size_y));
        }
        return cell;
    };

    double resolution = options.resolution;  // plane units; 0 = automatic
    if (!(resolution > 0.0)) {
        if (surface.empty() && options.surface != "model") {
            throw ValueError("BuildOrthomosaic: the chunk has no elevation model");
        }
        if (!surface.empty()) {
            const double plane_y = surface.grid.pixel_center(0, surface.grid.height / 2).y();
            const double gsd = measure_gsd_metres(surface, surface_nodata, plane_y);
            if (gsd > 0.0) {
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

    const RasterGrid grid = ortho_grid_for(surface.grid, resolution);
    progress.update("initialize", 1.0);

    // -- rectification -------------------------------------------------------
    // plane_to_local is built once and shared by every camera: the map is a
    // property of the grid, not of the view. For a Local frame it is empty and
    // the pose is already in the plane frame.
    const std::function<Eigen::Vector3d(const Eigen::Vector3d&)> to_local =
        make_plane_to_local(frame, grid, surface, surface_nodata);

    // -- memory guard --------------------------------------------------------
    // Refuse a run whose peak cannot fit BEFORE allocating anything: a bare
    // std::bad_alloc half-way through rectification (after the grid, the node
    // map and several tiles exist) leaves the OS thrashing and tells the user
    // nothing. The estimate is per-cell anatomy plus the summed footprint boxes;
    // the budget is 60% of physical RAM, overridable for tests.
    const std::uint64_t cells = static_cast<std::uint64_t>(grid.width) *
                                static_cast<std::uint64_t>(grid.height);
    const double footprint_cells =
        predicted_footprint_cells(cameras, grid, surface, surface_nodata, frame, to_local);
    const double peak_bytes = estimate_peak_bytes(cells, footprint_cells, bool{to_local});
    const std::uint64_t budget = platform::memory_budget_bytes();
    if (budget != 0 && peak_bytes > static_cast<double>(budget)) {
        throw ValueError(
            "BuildOrthomosaic needs about " + platform::format_gb(peak_bytes) +
            " at resolution " + std::to_string(resolution) + " m; this machine has " +
            platform::format_gb(static_cast<double>(
                platform::total_physical_memory_bytes())) +
            ". Set a coarser resolution (e.g. " +
            std::to_string(round_sig3(resolution * 4.0)) + " m).");
    }

    // The ortho grid and the DEM share a plane: in a Geographic frame both are
    // in degrees, so heights divide by degrees. The orthorectifier's occlusion
    // march compares a ray against metre heights and needs the ground metric,
    // which is why it is handed the frame's metres-per-unit explicitly.
    const Eigen::Vector2d metres_per_unit =
        frame.metres_per_plane_unit(grid.pixel_center(0, grid.height / 2).y());
    const double occlusion_tolerance =
        kOcclusionToleranceCells * surface.grid.pixel_size_x *
        std::max(metres_per_unit.x(), metres_per_unit.y());
    std::vector<ortho::OrthoTile> tiles;
    tiles.reserve(cameras.size());
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        if (progress.is_cancelled()) {
            throw Error("BuildOrthomosaic cancelled");
        }
        const CameraEntry& entry = cameras[i];
        // Decode this camera's photograph, rectify it, and drop it again: only
        // one full-resolution frame is ever resident.
        const Image image = read_image(entry.image_path);
        if (image.empty()) {
            throw ValueError("BuildOrthomosaic: camera " +
                             std::to_string(entry.camera_id) + " has an unreadable image");
        }
        ortho::OrthoCamera camera;
        camera.calibration = entry.calibration;
        camera.R = entry.R;
        camera.t = entry.t;
        camera.image = &image;
        camera.plane_metres_per_unit = metres_per_unit;
        if (to_local) {
            camera.plane_to_local = to_local;
            // The inverse the tile crop needs to place the footprint on the
            // plane grid. Full-precision to_plane (not the bilinear map's
            // inverse): the crop only has to be conservative, and this is the
            // exact map the orthorectifier's own plane_to_local approximates.
            camera.local_to_plane = [&frame](const Eigen::Vector3d& p) {
                return frame.to_plane(p);
            };
            camera.plane_centre = plane_centres[i];
        }
        tiles.push_back(ortho::orthorectify(surface, surface_nodata, grid, camera,
                                           static_cast<int>(tiles.size()),
                                           occlusion_tolerance));
        progress.update("rectify",
                        static_cast<double>(i + 1) / static_cast<double>(cameras.size()));
    }
    if (Log::enabled(LogLevel::Info)) {
        std::size_t valid_total = 0;
        for (const ortho::OrthoTile& tile : tiles) {
            for (const std::uint8_t v : tile.valid) {
                valid_total += v != 0 ? 1U : 0U;
            }
        }
        log(LogLevel::Info) << "BuildOrthomosaic: rectified " << tiles.size()
                            << " cameras, " << valid_total << " valid tile cells";
    }

    // -- seamlines -----------------------------------------------------------
    std::vector<std::int32_t> labels;
    if (options.refine_seamlines) {
        labels = ortho::assign_seams(grid.width, grid.height, tiles, ortho::SeamOptions{});
    } else {
        ortho::SeamOptions argmax_only;
        argmax_only.iterations = 0;
        labels = ortho::assign_seams(grid.width, grid.height, tiles, argmax_only);
    }
    progress.update("seamlines", 1.0);

    // -- blend ---------------------------------------------------------------
    const ortho::OrthoMosaic mosaic =
        ortho::blend(grid.width, grid.height, tiles, labels, options.blending);
    progress.update("blend", 1.0);

    // -- finalize ------------------------------------------------------------
    // The blend's labels index `tiles`, which is camera order. Map them to the
    // chunk camera ids so the container means the same thing as Chunk::cameras().
    Orthomosaic result;
    result.raster.grid = grid;
    result.raster.channels = 4;
    result.raster.data.resize(static_cast<std::size_t>(grid.width) *
                              static_cast<std::size_t>(grid.height) * 4U);
    for (std::size_t i = 0; i < mosaic.labels.size(); ++i) {
        const std::int32_t label = mosaic.labels[i];
        result.labels.push_back(label >= 0 ? cameras[static_cast<std::size_t>(label)].camera_id
                                           : -1);
        result.raster.data[i * 4 + 0] = mosaic.rgb[i * 3 + 0];
        result.raster.data[i * 4 + 1] = mosaic.rgb[i * 3 + 1];
        result.raster.data[i * 4 + 2] = mosaic.rgb[i * 3 + 2];
        result.raster.data[i * 4 + 3] = mosaic.valid[i] != 0 ? 255 : 0;
    }

    if (options.fill_holes) {
        // Fill interior nodata holes in each colour channel from the
        // surroundings. Natural-neighbour keeps the colour smooth across the
        // seams the hole borders. Filled cells become opaque, otherwise the
        // option would be invisible in any RGBA export.
        elevation::HoleFillOptions fill;
        fill.method = elevation::HoleFill::NaturalNeighbour;
        fill.interior_only = true;
        for (int channel = 0; channel < 3; ++channel) {
            std::vector<float> values(mosaic.labels.size(), 0.0F);
            std::vector<std::uint8_t> channel_valid(mosaic.labels.size(), 0);
            for (std::size_t i = 0; i < mosaic.labels.size(); ++i) {
                values[i] = static_cast<float>(mosaic.rgb[i * 3 + static_cast<std::size_t>(channel)]);
                channel_valid[i] = mosaic.valid[i];
            }
            elevation::fill_holes(grid.width, grid.height, values, channel_valid, fill);
            for (std::size_t i = 0; i < mosaic.labels.size(); ++i) {
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
        for (const std::uint8_t a : mosaic.valid) {
            covered += a != 0 ? 1U : 0U;
        }
        log(LogLevel::Info) << "BuildOrthomosaic: " << grid.width << "x" << grid.height
                            << " cells at " << resolution << " m, " << covered
                            << " covered, blending " << params.get_string("blending_mode")
                            << ", epsg " << result.epsg;
    }
    // Each run APPENDS a new orthomosaic asset and makes it active, so an
    // earlier orthomosaic is never overwritten.
    OrthomosaicAsset& asset = chunk->add_orthomosaic(asset_label);
    *asset.data = std::move(result);
}

}  // namespace alphashape
