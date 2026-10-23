#include "common/algorithm/ortho/build_orthomosaic_task.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
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
struct PlaneToLocalMap {
    int cols = 0;  ///< node columns (grid.width + 1)
    int rows = 0;  ///< node rows (grid.height + 1)
    double x0 = 0.0;
    double y0 = 0.0;
    double dx = 1.0;
    double dy = 1.0;
    std::vector<Eigen::Vector3d> local;  ///< cols * rows, row-major
    std::vector<double> height;          ///< plane height each node was mapped at
    std::vector<double> vertical;        ///< d(local z)/dh at each node

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
        Eigen::Vector3d out = w00 * local[i00] + w10 * local[i10] + w01 * local[i01] +
                              w11 * local[i11];
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
            map->height[i] = h;
            map->local[i] = frame.to_local(Eigen::Vector3d(x, y, h));
            const Eigen::Vector3d up = frame.to_local(Eigen::Vector3d(x, y, h + 1.0));
            map->vertical[i] = up.z() - map->local[i].z();
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

/// One enabled camera with its decoded photograph, in chunk camera order.
struct CameraEntry {
    std::int32_t camera_id = -1;
    Calibration calibration;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Vector3d centre_local = Eigen::Vector3d::Zero();
    std::size_t image_index = 0;
};

/// Decode every enabled camera that has a pose and a linked image. A camera
/// without a linked frame is skipped (nothing to sample); one that advertises a
/// pose but has no sensor is a hard error, as in BuildTexture.
std::vector<CameraEntry> collect_cameras(const Document& document, const Chunk& chunk,
                                         std::vector<Image>& images) {
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
        Image image =
            read_image(resolve_frame_path(document, frame_it->second->path()).string());
        if (image.empty()) {
            throw ValueError("BuildOrthomosaic: camera " + std::to_string(camera.id) +
                             " has an unreadable image");
        }
        CameraEntry entry;
        entry.camera_id = camera.id;
        entry.calibration =
            chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration;
        entry.R = camera.pose.block<3, 3>(0, 0);
        entry.t = camera.pose.block<3, 1>(0, 3);
        entry.centre_local = -entry.R.transpose() * entry.t;
        entry.image_index = images.size();
        images.push_back(std::move(image));
        entries.push_back(entry);
    }
    return entries;
}

/// Automatic ortho resolution: the median ground sample distance of the enabled
/// cameras, (camera height above the DEM median) / focal_px, rounded to three
/// significant figures and never finer than a quarter of the DEM cell.
///
/// Returns 0 when nothing can be measured; the caller then falls back to the
/// DEM cell itself.
double auto_resolution(const std::vector<CameraEntry>& cameras,
                       const std::vector<double>& plane_centres_z, const RasterGrid& grid,
                       double dem_median_height) {
    std::vector<double> gsds;
    gsds.reserve(cameras.size());
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        const CameraEntry& camera = cameras[i];
        if (!(camera.calibration.f > 0.0)) {
            continue;
        }
        // Camera height above the plane's median surface. Both are plane-frame
        // metres: camera centres converted by SurfaceFrame::to_plane, the
        // surface read from the DEM. (A geographic frame would need a
        // degrees-to-metres scale; BuildDem never writes one here, a documented
        // limitation.)
        const double height = plane_centres_z[i] - dem_median_height;
        if (!(height > 0.0)) {
            continue;
        }
        gsds.push_back(height / camera.calibration.f);
    }
    if (gsds.empty()) {
        return 0.0;
    }
    const double resolution = round_sig3(median_of(std::move(gsds)));
    const double floor_resolution = 0.25 * grid.pixel_size_x;
    return std::max(resolution, floor_resolution);
}

}  // namespace

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
                         "Output cell size in projection units. 0 selects the median camera "
                         "ground sample distance (3 significant figures), never finer than DEM "
                         "cell / 4."};
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

    // Every enabled camera with a pose and an image, decoded once.
    std::vector<Image> images;
    std::vector<CameraEntry> cameras = collect_cameras(document, *chunk, images);
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

    // Camera centres in the plane frame (needed for the automatic resolution;
    // cheap either way and also the rigorous source of the march target). For
    // a Local frame to_plane is the identity.
    std::vector<Eigen::Vector3d> plane_centres;
    std::vector<double> plane_centres_z;
    plane_centres.reserve(cameras.size());
    plane_centres_z.reserve(cameras.size());
    for (const CameraEntry& camera : cameras) {
        plane_centres.push_back(frame.to_plane(camera.centre_local));
        plane_centres_z.push_back(plane_centres.back().z());
    }

    double resolution = options.resolution;
    if (!(resolution > 0.0)) {
        // surface "model" may run without a DEM; its DSM cell takes over below,
        // after the model check it needs anyway.
        if (surface.empty() && options.surface != "model") {
            throw ValueError("BuildOrthomosaic: the chunk has no elevation model");
        }
        if (!surface.empty()) {
            const double median_surface = median_height(surface, surface_nodata);
            resolution =
                auto_resolution(cameras, plane_centres_z, surface.grid, median_surface);
        }
        if (!(resolution > 0.0)) {
            resolution = 0.25;  // replaced by the model DSM's cell below
        }
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
        dsm_options.resolution = resolution;
        dsm_options.epsg = epsg;
        dsm_options.tgv = false;  // a rasterised mesh needs no smoothing
        const Elevation dsm =
            elevation::build_dem(*chunk, dsm_options, [&](double fraction) {
                progress.update("initialize", 0.5 + 0.2 * fraction);
            });
        // With no DEM, the ortho grid follows the DSM the mesh produced; its
        // cell stands in for the DEM cell in the auto-resolution floor.
        if (dem.empty()) {
            resolution = dsm.raster.grid.pixel_size_x;
        }
        surface = dsm.raster;
        surface_nodata = dsm.nodata;
    }

    const RasterGrid grid = ortho_grid_for(surface.grid, resolution);
    progress.update("initialize", 1.0);

    // -- rectification -------------------------------------------------------
    // plane_to_local is built once and shared by every camera: the map is a
    // property of the grid, not of the view. For a Local frame it is empty and
    // the pose is already in the plane frame.
    const std::function<Eigen::Vector3d(const Eigen::Vector3d&)> to_local =
        make_plane_to_local(frame, grid, surface, surface_nodata);

    const double occlusion_tolerance = kOcclusionToleranceCells * surface.grid.pixel_size_x;
    std::vector<ortho::OrthoTile> tiles;
    tiles.reserve(cameras.size());
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        if (progress.is_cancelled()) {
            throw Error("BuildOrthomosaic cancelled");
        }
        const CameraEntry& entry = cameras[i];
        ortho::OrthoCamera camera;
        camera.calibration = entry.calibration;
        camera.R = entry.R;
        camera.t = entry.t;
        camera.image = &images[entry.image_index];
        if (to_local) {
            camera.plane_to_local = to_local;
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
