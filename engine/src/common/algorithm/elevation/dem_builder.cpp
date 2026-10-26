#include "common/algorithm/elevation/dem_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "common/algorithm/elevation/surface_frame.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::elevation {
namespace {

TgvOptions make_dem_smoothing_options() {
    TgvOptions options;
    options.alpha1 = 0.2;
    options.alpha0 = 0.4;  // keep the piecewise-affine ratio alpha0/alpha1 = 2
    options.lambda = 1.0;
    options.data_term = TgvDataTerm::L1;
    return options;
}

/// The validated choice of `source`.
enum class Source { DepthMaps, PointCloud, Model };

/// The validated choice of `interpolation`.
enum class Interpolation { Disabled, Enabled, Extrapolated };

/// The automatic-resolution rule: three significant figures, so the cell size is
/// a round-looking number derived from the sample spacing (e.g. 0.101 -> 0.101,
/// 1.2334 -> 1.23).
constexpr int kAutoResolutionDigits = 3;

Source parse_source(const std::string& value) {
    if (value == "depth_maps") {
        return Source::DepthMaps;
    }
    if (value == "point_cloud") {
        return Source::PointCloud;
    }
    if (value == "model") {
        return Source::Model;
    }
    throw ValueError("build_dem: source must be one of \"depth_maps\", \"point_cloud\", "
                     "\"model\" (got \"" +
                     value + "\")");
}

Interpolation parse_interpolation(const std::string& value) {    if (value == "disabled") {
        return Interpolation::Disabled;
    }
    if (value == "enabled") {
        return Interpolation::Enabled;
    }
    if (value == "extrapolated") {
        return Interpolation::Extrapolated;
    }
    throw ValueError("build_dem: interpolation must be one of \"disabled\", \"enabled\", "
                     "\"extrapolated\" (got \"" +
                     value + "\")");
}

/// One sample in the plane frame: plane (x, y) plus its height and weight.
struct PlaneSample {
    Eigen::Vector2d p = Eigen::Vector2d::Zero();
    double h = 0.0;
    float confidence = 1.0F;
};

/// The per-map automatic-resolution signal: the ground sample distance of one
/// depth map, depth * downscale / focal, summarised by its median over the valid
/// pixels. Empty when the map has no valid pixel or a degenerate focal length.
struct DepthSamples {
    std::vector<PlaneSample> samples;
    std::vector<double> per_map_gsd;  ///< one entry per map with valid pixels
};

double median_of(std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid),
                     values.end());
    const double hi = values[mid];
    if (values.size() % 2 == 1) {
        return hi;
    }
    const double lo =
        *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid));
    return 0.5 * (lo + hi);
}

/// Round to `digits` significant figures (the auto-resolution contract).
double round_sig(double v, int digits) {
    if (!(v > 0.0) || !std::isfinite(v)) {
        return v;
    }
    const double magnitude = std::floor(std::log10(v));
    const double scale = std::pow(10.0, static_cast<double>(digits - 1) - magnitude);
    return std::round(v * scale) / scale;
}

Eigen::Matrix3d pose_rotation(const Transform& pose) { return pose.block<3, 3>(0, 0); }
Eigen::Vector3d pose_translation(const Transform& pose) { return pose.block<3, 1>(0, 3); }

/// Map every dense-cloud point into the plane frame, carrying its confidence
/// (1 when the cloud has no confidence vector).
std::vector<PlaneSample> collect_point_cloud_samples(const Chunk& chunk,
                                                     const SurfaceFrame& frame) {
    const PointCloud& cloud = chunk.point_cloud();
    std::vector<PlaneSample> samples;
    samples.reserve(cloud.points.size());
    const bool has_confidence = cloud.confidence.size() == cloud.points.size();
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        const Eigen::Vector3d plane = frame.to_plane(cloud.points[i]);
        if (!std::isfinite(plane.x()) || !std::isfinite(plane.y()) || !std::isfinite(plane.z())) {
            continue;
        }
        PlaneSample s;
        s.p = Eigen::Vector2d(plane.x(), plane.y());
        s.h = plane.z();
        s.confidence = has_confidence ? cloud.confidence[i] : 1.0F;
        samples.push_back(s);
    }
    return samples;
}

/// Back-project every valid depth pixel into chunk-local space exactly as
/// stereo/fusion.cpp does (p_chunk = R^T (depth * ray - t)) and map it into the
/// plane frame. Only cameras that exist and are enabled contribute.
DepthSamples collect_depth_map_samples(const Chunk& chunk, const SurfaceFrame& frame) {
    DepthSamples result;
    std::unordered_map<std::int32_t, const Camera*> camera_by_id;
    for (const Camera& camera : chunk.cameras()) {
        camera_by_id[camera.id] = &camera;
    }
    for (const DepthMap& map : chunk.depth_maps()) {
        const auto it = camera_by_id.find(map.camera_id);
        if (it == camera_by_id.end() || !it->second->enabled) {
            continue;  // camera gone, or not aligned: its map is stale
        }
        const Camera& camera = *it->second;
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk.sensors().size())) {
            throw ValueError("build_dem: camera " + std::to_string(map.camera_id) +
                             " has no sensor");
        }
        if (map.width <= 0 || map.height <= 0 || map.downscale < 1) {
            throw ValueError("build_dem: depth map for camera " + std::to_string(map.camera_id) +
                             " has invalid dimensions");
        }
        const std::size_t pixels =
            static_cast<std::size_t>(map.width) * static_cast<std::size_t>(map.height);
        if (map.depths.size() != pixels) {
            throw ValueError("build_dem: depth map for camera " +
                             std::to_string(map.camera_id) + " has a mismatched depth buffer");
        }
        const Calibration calibration = scaled_calibration(
            chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration,
            map.downscale);

        const Eigen::Matrix3d rt = pose_rotation(camera.pose).transpose();
        const Eigen::Vector3d t = pose_translation(camera.pose);
        // The full-resolution focal length in pixels: the calibration below is
        // scaled to this map's resolution, so undo the downscale. depth *
        // downscale / focal_px (the full-res focal) is the world footprint of
        // one depth-map pixel, the natural automatic cell size here.
        const double full_focal = calibration.f * static_cast<double>(map.downscale);
        std::vector<double> footprints;
        footprints.reserve(pixels);
        for (int j = 0; j < map.height; ++j) {
            for (int i = 0; i < map.width; ++i) {
                const std::size_t idx = static_cast<std::size_t>(j) *
                                            static_cast<std::size_t>(map.width) +
                                        static_cast<std::size_t>(i);
                const float depth = map.depths[idx];
                if (!(depth > 0.0F)) {
                    continue;  // -1 sentinel and 0 both mean "no depth"
                }
                const Eigen::Vector2d centre(static_cast<double>(i) + 0.5,
                                             static_cast<double>(j) + 0.5);
                const Eigen::Vector3d ray = calibration.unproject(centre);
                const Eigen::Vector3d p_cam = static_cast<double>(depth) * ray;
                const Eigen::Vector3d plane = frame.to_plane(rt * (p_cam - t));
                if (!std::isfinite(plane.x()) || !std::isfinite(plane.y()) ||
                    !std::isfinite(plane.z())) {
                    continue;
                }
                PlaneSample s;
                s.p = Eigen::Vector2d(plane.x(), plane.y());
                s.h = plane.z();
                s.confidence = map.confidence.empty() ? 1.0F : map.confidence[idx];
                result.samples.push_back(s);
                if (full_focal > 1e-12) {
                    footprints.push_back(static_cast<double>(depth) *
                                         static_cast<double>(map.downscale) / full_focal);
                }
            }
        }
        if (!footprints.empty()) {
            result.per_map_gsd.push_back(median_of(footprints));
        }
    }
    return result;
}

/// The plane-frame triangle soup of the model source, plus its vertices.
struct ModelSamples {
    std::vector<Eigen::Vector3d> plane_vertices;  ///< parallel to chunk vertices
    std::vector<Eigen::Vector3i> faces;
};

ModelSamples collect_model_samples(const Chunk& chunk, const SurfaceFrame& frame) {
    const Mesh& mesh = chunk.model();
    ModelSamples out;
    out.plane_vertices.reserve(mesh.vertices.size());
    for (const Eigen::Vector3d& v : mesh.vertices) {
        out.plane_vertices.push_back(frame.to_plane(v));
    }
    out.faces = mesh.faces;
    return out;
}

/// The plane-frame bounding box of every sample.
struct PlaneBox {
    Eigen::Vector2d min = Eigen::Vector2d::Zero();
    Eigen::Vector2d max = Eigen::Vector2d::Zero();
};

/// The plane-frame bounding box of the samples, made robust against outliers.
///
/// A raw min/max is wrong here: depth-map back-projection produces a tail of
/// far samples (a mismatched far stereo point, a sky-keyed patch that fused at
/// the infinity clamp, an oblique camera's grazing depth), and ONE of them at
/// 10 km sets the grid extent. At the DEM's auto cell that is a grid of
/// billions of cells -- the "bad allocation" an 8 GB laptop hits on a small
/// block long before any ortho exists.
///
/// The test is a comparison, not a blind trim, so a genuine extent is never
/// narrowed:
///
///   1. the exact box (min/max) and the 0.5th-99.5th percentile box (the
///      "core", which a handful of far samples cannot move);
///   2. if the exact extent on an axis is at most kOutlierExtentRatio times the
///      core extent, there are no outliers worth handling and the EXACT bound
///      is kept -- an ordinary lattice therefore produces exactly the grid it
///      always did;
///   3. otherwise the axis is clipped to the core plus kOutlierMarginFraction
///      of the core extent on each side. On a real block the clipped box still
///      contains the whole surveyed area (the discarded samples lie beyond it
///      by construction), and the grid stays a modest multiple of the real
///      extent instead of growing to the outlier's distance.
///
/// Small sample sets skip the analysis entirely: a percentile is meaningless
/// below kMinSamplesForRobustBox and the synthetic test scenes must keep their
/// historical grid.
constexpr std::size_t kMinSamplesForRobustBox = 1000;
/// One-sided tail fraction used to define the inlier core.
constexpr double kExtentTailFraction = 0.005;
/// Exact extent / core extent above which an axis is treated as outlier-poisoned.
constexpr double kOutlierExtentRatio = 2.0;
/// Margin added outside the core when clipping, as a fraction of the core extent.
constexpr double kOutlierMarginFraction = 0.03;

/// The `fraction` quantile of `values` (0..1), by nth_element. `values` is
/// reordered.
double quantile(std::vector<double>& values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    const double clamped = std::clamp(fraction, 0.0, 1.0);
    const std::size_t k = std::min(
        values.size() - 1, static_cast<std::size_t>(clamped * static_cast<double>(values.size())));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(k),
                     values.end());
    return values[k];
}

/// Clamp one axis of `box` to its core when the exact extent is outlier-poisoned.
void clip_axis(double core_lo, double core_hi, double* lo, double* hi) {
    const double core = core_hi - core_lo;
    if (!(core > 0.0) || !std::isfinite(core)) {
        return;  // a degenerate core says nothing; leave the exact bound alone
    }
    const double margin = kOutlierMarginFraction * core;
    if (*lo < core_lo) {
        // Only engage when the low tail is more than "a bit" outside the core.
        if ((core_hi - *lo) > kOutlierExtentRatio * core) {
            *lo = core_lo - margin;
        }
    }
    if (*hi > core_hi) {
        if ((*hi - core_lo) > kOutlierExtentRatio * core) {
            *hi = core_hi + margin;
        }
    }
}

PlaneBox samples_box(const std::vector<PlaneSample>& samples) {
    PlaneBox box;
    box.min = samples.front().p;
    box.max = samples.front().p;
    for (const PlaneSample& s : samples) {
        box.min = box.min.cwiseMin(s.p);
        box.max = box.max.cwiseMax(s.p);
    }
    if (samples.size() < kMinSamplesForRobustBox) {
        return box;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(samples.size());
    ys.reserve(samples.size());
    for (const PlaneSample& s : samples) {
        xs.push_back(s.p.x());
        ys.push_back(s.p.y());
    }
    const double x_lo = quantile(xs, kExtentTailFraction);
    const double x_hi = quantile(xs, 1.0 - kExtentTailFraction);
    const double y_lo = quantile(ys, kExtentTailFraction);
    const double y_hi = quantile(ys, 1.0 - kExtentTailFraction);

    clip_axis(x_lo, x_hi, &box.min.x(), &box.max.x());
    clip_axis(y_lo, y_hi, &box.min.y(), &box.max.y());

    // A clipped box must still be non-empty; an all-collinear cloud can make the
    // core degenerate and that has already been rejected above, but guard the
    // final result anyway so the grid can never come out inverted.
    if (!(box.max.x() > box.min.x()) || !(box.max.y() > box.min.y())) {
        return box;
    }
    return box;
}

PlaneBox vertices_box(const std::vector<Eigen::Vector3d>& vertices) {
    PlaneBox box;
    box.min = vertices.front().head<2>();
    box.max = vertices.front().head<2>();
    for (const Eigen::Vector3d& v : vertices) {
        box.min = box.min.cwiseMin(v.head<2>());
        box.max = box.max.cwiseMax(v.head<2>());
    }
    return box;
}

/// Automatic resolution for the point-cloud / depth-map sources: the median
/// depth-map ground sample distance, or sqrt(footprint area / N) when only
/// points are available. Rounded to kAutoResolutionDigits significant figures.
double auto_resolution(const std::vector<PlaneSample>& samples,
                       const std::vector<double>& per_map_gsd) {
    if (!per_map_gsd.empty()) {
        std::vector<double> copy = per_map_gsd;
        return round_sig(median_of(copy), kAutoResolutionDigits);
    }
    const PlaneBox box = samples_box(samples);
    const double area = std::abs((box.max.x() - box.min.x()) * (box.max.y() - box.min.y()));
    const double spacing = std::sqrt(area / static_cast<double>(samples.size()));
    return round_sig(spacing, kAutoResolutionDigits);
}

/// Automatic resolution for the model source: the median edge length in the
/// plane frame. Non-finite / degenerate edges are skipped.
double auto_resolution_model(const ModelSamples& model) {
    std::vector<double> edges;
    edges.reserve(model.faces.size() * 3);
    for (const Eigen::Vector3i& face : model.faces) {
        for (int e = 0; e < 3; ++e) {
            const Eigen::Vector3d& a = model.plane_vertices[static_cast<std::size_t>(face[e])];
            const Eigen::Vector3d& b =
                model.plane_vertices[static_cast<std::size_t>(face[(e + 1) % 3])];
            const double length = (b.head<2>() - a.head<2>()).norm();
            if (length > 0.0 && std::isfinite(length)) {
                edges.push_back(length);
            }
        }
    }
    return round_sig(median_of(edges), kAutoResolutionDigits);
}

/// The north-up grid covering `box`, its origin snapped outward to multiples of
/// `resolution` (row 0 is the northernmost row).
RasterGrid frame_grid(const PlaneBox& box, double resolution) {
    RasterGrid grid;
    grid.pixel_size_x = resolution;
    grid.pixel_size_y = resolution;
    grid.origin_x = std::floor(box.min.x() / resolution) * resolution;
    grid.origin_y = std::ceil(box.max.y() / resolution) * resolution;
    grid.width = std::max(
        1, static_cast<int>(std::ceil((box.max.x() - grid.origin_x) / resolution)));
    grid.height = std::max(
        1, static_cast<int>(std::ceil((grid.origin_y - box.min.y()) / resolution)));
    return grid;
}

/// The cell a sample falls in, or (-1, -1) when it lies outside the grid.
Eigen::Vector2i cell_of(const RasterGrid& grid, const Eigen::Vector2d& p) {
    const double col = std::floor((p.x() - grid.origin_x) / grid.pixel_size_x);
    const double row = std::floor((grid.origin_y - p.y()) / grid.pixel_size_y);
    if (!(col >= 0.0) || !(row >= 0.0) || col >= static_cast<double>(grid.width) ||
        row >= static_cast<double>(grid.height)) {
        return Eigen::Vector2i(-1, -1);
    }
    return Eigen::Vector2i(static_cast<int>(col), static_cast<int>(row));
}

/// Per-cell accumulator for the point-cloud / depth-map sources: the heights of
/// the samples that landed in the cell and their summed confidence.
struct CellAccumulator {
    std::vector<double> heights;
    double confidence = 0.0;
};

/// Rasterise point / depth-map samples into per-cell medians and summed
/// confidence. Returns the per-cell accumulators (an unordered map keyed by
/// linear cell index) so the median rule sees every sample.
std::unordered_map<std::size_t, CellAccumulator> rasterise_samples(
    const RasterGrid& grid, const std::vector<PlaneSample>& samples) {
    std::unordered_map<std::size_t, CellAccumulator> cells;
    for (const PlaneSample& s : samples) {
        const Eigen::Vector2i cell = cell_of(grid, s.p);
        if (cell.x() < 0) {
            continue;
        }
        const std::size_t index =
            static_cast<std::size_t>(cell.y()) * static_cast<std::size_t>(grid.width) +
            static_cast<std::size_t>(cell.x());
        CellAccumulator& acc = cells[index];
        acc.heights.push_back(s.h);
        acc.confidence += static_cast<double>(s.confidence);
    }
    return cells;
}

/// Rasterise the model top-down: every cell whose centre is inside a triangle
/// takes the interpolated height, keeping the MAXIMUM across triangles (the
/// visible top surface). Returns the height field (nodata where uncovered).
std::vector<float> rasterise_model(const RasterGrid& grid, const ModelSamples& model) {
    const std::size_t count = static_cast<std::size_t>(grid.width) *
                              static_cast<std::size_t>(grid.height);
    std::vector<float> heights(count, std::numeric_limits<float>::quiet_NaN());
    for (const Eigen::Vector3i& face : model.faces) {
        const Eigen::Vector3d& a = model.plane_vertices[static_cast<std::size_t>(face[0])];
        const Eigen::Vector3d& b = model.plane_vertices[static_cast<std::size_t>(face[1])];
        const Eigen::Vector3d& c = model.plane_vertices[static_cast<std::size_t>(face[2])];
        const double area = (b.x() - a.x()) * (c.y() - a.y()) - (c.x() - a.x()) * (b.y() - a.y());
        if (std::abs(area) < 1e-18) {
            continue;  // degenerate in the plane: no cell centre is meaningfully inside
        }
        const double min_x = std::min({a.x(), b.x(), c.x()});
        const double max_x = std::max({a.x(), b.x(), c.x()});
        const double min_y = std::min({a.y(), b.y(), c.y()});
        const double max_y = std::max({a.y(), b.y(), c.y()});
        const Eigen::Vector2d lo = grid.to_pixel(min_x, max_y);
        const Eigen::Vector2d hi = grid.to_pixel(max_x, min_y);
        const int col0 = std::max(0, static_cast<int>(std::floor(lo.x() - 0.5)));
        const int col1 = std::min(grid.width - 1, static_cast<int>(std::ceil(hi.x() - 0.5)));
        const int row0 = std::max(0, static_cast<int>(std::floor(lo.y() - 0.5)));
        const int row1 = std::min(grid.height - 1, static_cast<int>(std::ceil(hi.y() - 0.5)));
        for (int row = row0; row <= row1; ++row) {
            for (int col = col0; col <= col1; ++col) {
                const Eigen::Vector2d centre = grid.pixel_center(col, row);
                const double w0 = ((b.x() - centre.x()) * (c.y() - centre.y()) -
                                   (c.x() - centre.x()) * (b.y() - centre.y())) /
                                  area;
                const double w1 = ((c.x() - centre.x()) * (a.y() - centre.y()) -
                                   (a.x() - centre.x()) * (c.y() - centre.y())) /
                                  area;
                const double w2 = 1.0 - w0 - w1;
                constexpr double kInsideEpsilon = -1e-9;
                if (w0 < kInsideEpsilon || w1 < kInsideEpsilon || w2 < kInsideEpsilon) {
                    continue;  // centre outside this triangle
                }
                const double h = w0 * a.z() + w1 * b.z() + w2 * c.z();
                const std::size_t index =
                    static_cast<std::size_t>(row) * static_cast<std::size_t>(grid.width) +
                    static_cast<std::size_t>(col);
                float& slot = heights[index];
                if (!std::isfinite(slot) || h > static_cast<double>(slot)) {
                    slot = static_cast<float>(h);
                }
            }
        }
    }
    return heights;
}

}  // namespace

// -- orchestration -----------------------------------------------------------

TgvOptions dem_smoothing_options() { return make_dem_smoothing_options(); }

namespace {

/// Everything the assemble step needs: the observations, their validity, the
/// per-cell confidence and the frame's nodata bookkeeping.
struct AssembledSurface {
    std::vector<float> values;      ///< observations / filled heights
    std::vector<std::uint8_t> valid;
    std::vector<float> confidence;  ///< per cell, 0 where unobserved
};

}  // namespace

Elevation build_dem(const Chunk& chunk, const DemOptions& options,
                    const std::function<void(double)>& progress) {
    const Source source = parse_source(options.source);
    const Interpolation interpolation = parse_interpolation(options.interpolation);
    const auto tick = [&](double fraction) {
        if (progress) {
            progress(fraction);
        }
    };

    std::string source_label = "depth_maps";
    if (source == Source::PointCloud) {
        source_label = "point_cloud";
    } else if (source == Source::Model) {
        source_label = "model";
    }

    tick(0.05);
    const SurfaceFrame frame = SurfaceFrame::for_chunk(chunk, options.epsg);

    // -- collect the plane-frame samples / geometry --------------------------
    std::vector<PlaneSample> samples;
    std::vector<double> per_map_gsd;
    ModelSamples model;
    if (source == Source::PointCloud) {
        if (chunk.point_cloud().points.empty()) {
            throw ValueError("No point cloud");
        }
        samples = collect_point_cloud_samples(chunk, frame);
    } else if (source == Source::DepthMaps) {
        if (chunk.depth_maps().empty()) {
            throw ValueError("No depth maps");
        }
        DepthSamples collected = collect_depth_map_samples(chunk, frame);
        samples = std::move(collected.samples);
        per_map_gsd = std::move(collected.per_map_gsd);
    } else {
        if (chunk.model().empty()) {
            throw ValueError("No model");
        }
        model = collect_model_samples(chunk, frame);
    }
    tick(0.25);

    if (source != Source::Model && samples.empty()) {
        throw ValueError("build_dem: no samples were produced from the " + source_label +
                         " source");
    }
    if (source == Source::Model && model.plane_vertices.empty()) {
        throw ValueError("build_dem: the model has no vertices");
    }

    // -- resolve the cell size and the grid ----------------------------------
    const PlaneBox box = source == Source::Model ? vertices_box(model.plane_vertices)
                                                 : samples_box(samples);
    double resolution = options.resolution;
    if (!(resolution > 0.0)) {
        if (source == Source::Model) {
            resolution = auto_resolution_model(model);  // already in plane units
        } else {
            resolution = auto_resolution(samples, per_map_gsd);
            // The depth-map GSD is a GROUND distance in metres, while the plane
            // it has to be expressed in is degrees for EPSG:4326. Convert with
            // the arc lengths at the surface's own latitude, otherwise the cell
            // would be ~100000x too small. The point-cloud fallback is already a
            // plane-unit spacing (sqrt(plane area / N)) and needs no conversion.
            if (!per_map_gsd.empty() && !frame.is_local()) {
                const double plane_y = 0.5 * (box.max.y() + box.min.y());
                const Eigen::Vector2d metres_per_unit = frame.metres_per_plane_unit(plane_y);
                const double unit_m =
                    std::max(metres_per_unit.x(), metres_per_unit.y());
                if (unit_m > 0.0) {
                    resolution /= unit_m;
                }
            }
        }
    }
    if (!(resolution > 0.0) || !std::isfinite(resolution)) {
        throw ValueError("build_dem: cannot derive a cell size from the " + source_label +
                         " source; pass an explicit resolution");
    }
    const RasterGrid grid = frame_grid(box, resolution);
    const std::size_t count =
        static_cast<std::size_t>(grid.width) * static_cast<std::size_t>(grid.height);

    // -- rasterise into per-cell observations ---------------------------------
    AssembledSurface surface;
    surface.values.assign(count, std::numeric_limits<float>::quiet_NaN());
    surface.valid.assign(count, 0U);
    surface.confidence.assign(count, 0.0F);

    if (source == Source::Model) {
        const std::vector<float> heights = rasterise_model(grid, model);
        for (std::size_t i = 0; i < count; ++i) {
            if (std::isfinite(heights[i])) {
                surface.values[i] = heights[i];
                surface.valid[i] = 1U;
                surface.confidence[i] = 1.0F;
            }
        }
    } else {
        std::unordered_map<std::size_t, CellAccumulator> cells =
            rasterise_samples(grid, samples);
        for (auto& [index, acc] : cells) {
            std::vector<double>& heights = acc.heights;
            const std::size_t mid = heights.size() / 2;
            std::nth_element(heights.begin(),
                             heights.begin() + static_cast<std::ptrdiff_t>(mid), heights.end());
            double median = heights[mid];
            if (heights.size() % 2 == 0) {
                const double lo = *std::max_element(
                    heights.begin(), heights.begin() + static_cast<std::ptrdiff_t>(mid));
                median = 0.5 * (lo + heights[mid]);
            }
            surface.values[index] = static_cast<float>(median);
            surface.valid[index] = 1U;
            surface.confidence[index] = static_cast<float>(acc.confidence);
        }
    }
    tick(0.55);

    const std::vector<std::uint8_t> observed = surface.valid;  // the interpolation mask

    // -- hole filling ---------------------------------------------------------
    std::vector<float> filled = surface.values;
    std::vector<std::uint8_t> filled_valid = surface.valid;
    if (interpolation != Interpolation::Disabled) {
        HoleFillOptions fill_options;
        fill_options.method = options.fill;
        fill_options.interior_only = interpolation == Interpolation::Enabled;
        fill_holes(grid.width, grid.height, filled, filled_valid, fill_options);
    }
    tick(0.7);

    // -- TGV^2 smoothing / inpainting ----------------------------------------
    std::vector<float> final_values = filled;
    std::vector<std::uint8_t> final_valid = filled_valid;
    if (options.tgv && source != Source::Model) {
        // Heights are scaled so one cell step is one unit, which makes the
        // regulariser's slopes dimensionless (see tgv_denoise.hpp). A
        // Geographic grid's pixel size is in degrees, so the cell has to be
        // converted to metres first; dividing metre heights by degrees would
        // make the data term meaningless and flatten the surface.
        const double plane_y = 0.5 * (grid.pixel_center(0, grid.height / 2).y() +
                                      grid.pixel_center(0, 0).y());
        const Eigen::Vector2d metres_per_unit =
            frame.is_local() ? Eigen::Vector2d(1.0, 1.0)
                             : frame.metres_per_plane_unit(plane_y);
        double cell_size = grid.pixel_size_x * metres_per_unit.x();
        if (!(cell_size > 0.0) || !std::isfinite(cell_size)) {
            cell_size = grid.pixel_size_x;  // degenerate frame: keep the old rule
        }
        std::vector<float> f(count, 0.0F);
        std::vector<float> u(count, 0.0F);
        // The data term is the hole-filled surface: observed cells carry their
        // measured height, interpolated cells the value fill_holes produced for
        // them. Without the interpolated cells in f the regulariser would be
        // free to move them arbitrarily (they have no direct observation), which
        // on a sparse depth-map union wrecks every filled cell.
        for (std::size_t i = 0; i < count; ++i) {
            if (filled_valid[i] == 0U) {
                continue;
            }
            f[i] = static_cast<float>(static_cast<double>(filled[i]) / cell_size);
            u[i] = f[i];  // warm start from the filled surface
        }
        // Normalise the per-cell confidence to median 1 and clamp to [0, 4] so
        // the data term's weight is comparable across sources and a single
        // over-supported cell cannot dominate the fit. Interpolated cells have
        // no measured confidence; they enter at the normalised weight of 1.
        std::vector<float> weights(count, 0.0F);
        std::vector<float> observed_confidences;
        observed_confidences.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            if (observed[i] != 0U) {
                observed_confidences.push_back(surface.confidence[i]);
                weights[i] = surface.confidence[i];
            } else if (filled_valid[i] != 0U) {
                weights[i] = -1.0F;  // placeholder: resolved below to the median
            }
        }
        if (!observed_confidences.empty()) {
            std::vector<double> confidence_copy(observed_confidences.begin(),
                                                observed_confidences.end());
            const double median_confidence = median_of(confidence_copy);
            if (median_confidence > 0.0) {
                for (float& w : weights) {
                    w = static_cast<float>(static_cast<double>(w) / median_confidence);
                    w = std::min(std::max(w, 0.0F), 4.0F);
                }
            }
        }
        for (std::size_t i = 0; i < count; ++i) {
            if (filled_valid[i] != 0U && observed[i] == 0U) {
                weights[i] = 1.0F;  // interpolated cell: ordinary unit weight
            }
        }
        // Unmeasured cells carry zero weight: the regulariser alone fills them.
        bool any_weight = false;
        for (const float w : weights) {
            any_weight = any_weight || w > 0.0F;
        }
        if (any_weight) {
            TgvOptions tgv_options = options.tgv_options;
            tgv2_denoise(grid.width, grid.height, f, weights, u, tgv_options);
            for (std::size_t i = 0; i < count; ++i) {
                if (final_valid[i] != 0U) {
                    final_values[i] =
                        static_cast<float>(static_cast<double>(u[i]) * cell_size);
                }
            }
        }
    }
    // Re-apply the interpolation mask: cells that were neither observed nor
    // filled stay nodata, even though the regulariser assigned them a value.
    for (std::size_t i = 0; i < count; ++i) {
        if (final_valid[i] == 0U) {
            final_values[i] = -32767.0F;
        }
    }
    tick(0.95);

    Elevation elevation;
    elevation.raster.grid = grid;
    elevation.raster.channels = 1;
    elevation.raster.data = std::move(final_values);
    elevation.epsg = frame.epsg();
    elevation.nodata = -32767.0F;
    elevation.source = source_label;
    tick(1.0);
    return elevation;
}

}  // namespace alphashape::elevation