// BuildModel: turn the chunk's reconstruction into a triangle mesh (milestone 7).
//
// The adapter between the chunk's data model and the meshing stages. Depending
// on `source_data` it builds a scalar volume either by TV-L1 range-image fusion
// of the depth maps (tv_fusion.hpp) or by screened Poisson reconstruction from
// the dense point cloud (poisson.hpp), extracts the iso-surface (marching
// tetrahedra), cleans it up (component removal, quadric decimation) and — when
// depth maps exist — colours its vertices from the source images.
//
// Everything is computed into a local Mesh, which is then installed in a NEW
// model asset (appended and made active), so an error anywhere leaves the chunk
// untouched and an earlier model keeps its own mesh and texture pages.

#include "common/algorithm/mesh/build_model_task.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/mesh/poisson.hpp"
#include "common/algorithm/mesh/tv_fusion_gpu.hpp"
#include "common/algorithm/mesh/volume.hpp"
#include "common/core/assets.hpp"
#include "common/core/frame.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"
#include "model/camera/calibration.hpp"
#include "model/mesh/decimate.hpp"
#include "model/mesh/marching_tetrahedra.hpp"
#include "model/mesh/mesh_ops.hpp"

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

namespace alphashape {
namespace {

// -- tuning constants (ours) ------

/// Per-axis bounding box taken from the 2nd..98th percentile of the input
/// points, so a handful of stragglers cannot blow the volume up.
constexpr double kBboxPercentileLow = 0.02;
constexpr double kBboxPercentileHigh = 0.98;
/// The percentile box is grown by this fraction of the extent on every side,
/// leaving a margin of unmeasured space around the surface.
constexpr double kBboxMarginFraction = 0.10;
/// Fewer input points than this cannot define a volume at all.
constexpr std::size_t kBboxMinPoints = 10;

/// "enabled" interpolation grows the observed mask by this many 6-neighbourhood
/// iterations before extraction, so the surface may close small unmeasured
/// gaps; "disabled" uses the raw mask and "extrapolated" none.
constexpr int kInterpolationDilateVoxels = 4;

/// Components smaller than this fraction of the largest one's face count are
/// discarded after extraction.
constexpr double kRemoveSmallFraction = 0.01;

/// A depth-map sample colours a vertex only when the vertex's camera-frame
/// distance agrees with the measured depth to within this relative tolerance
/// (occlusion / silhouette guard).
constexpr double kColourDepthTolerance = 0.02;
/// Guard against projecting a point on the image plane itself.
constexpr double kProjectZEpsilon = 1e-9;
/// Colour stored for a vertex no view could sample.
constexpr std::uint8_t kUnseenColour = 128;

/// volume_resolution bounds (the longest bounding-box axis is divided into this
/// many voxels).
constexpr std::int64_t kVolumeResolutionMin = 16;
constexpr std::int64_t kVolumeResolutionMax = 512;

/// face_count presets: the decimation target as a fraction of the extracted
/// face count (low / medium / high), plus the default for "custom".
constexpr double kFaceCountLow = 1.0 / 20.0;
constexpr double kFaceCountMedium = 1.0 / 5.0;
constexpr double kFaceCountHigh = 1.0 / 2.0;
constexpr std::int64_t kFaceCountCustomDefault = 200000;

/// Resolve a frame's recorded path against the document's directory (the same
/// rule AddPhotos / BuildPointCloud use).
std::filesystem::path resolve_frame_path(const Document& document, const std::string& path) {
    const std::filesystem::path recorded(path);
    if (recorded.is_absolute() || document.path().empty()) {
        return recorded;
    }
    return std::filesystem::path(document.path()).parent_path() / recorded;
}

Eigen::Matrix3d pose_R(const Transform& pose) { return pose.block<3, 3>(0, 0); }
Eigen::Vector3d pose_t(const Transform& pose) { return pose.block<3, 1>(0, 3); }

/// Linear-interpolated percentile of an ASCENDING-sorted vector.
double percentile(const std::vector<double>& sorted, double fraction) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double idx = fraction * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double weight = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - weight) + sorted[hi] * weight;
}

/// Axis-aligned box over `points`: the per-axis 2nd..98th percentile range
/// grown by kBboxMarginFraction of the extent on each side. Throws ValueError
/// when there are too few points to bound anything.
std::pair<Eigen::Vector3d, Eigen::Vector3d> compute_bbox(std::vector<Eigen::Vector3d> points) {
    if (points.size() < kBboxMinPoints) {
        throw ValueError("BuildModel: too few points (" + std::to_string(points.size()) +
                         ") to bound the reconstruction volume");
    }
    Eigen::Vector3d lo = Eigen::Vector3d::Zero();
    Eigen::Vector3d hi = Eigen::Vector3d::Zero();
    std::vector<double> axis_values;
    axis_values.reserve(points.size());
    for (int axis = 0; axis < 3; ++axis) {
        axis_values.clear();
        for (const Eigen::Vector3d& p : points) {
            axis_values.push_back(p[axis]);
        }
        std::sort(axis_values.begin(), axis_values.end());
        const double a = percentile(axis_values, kBboxPercentileLow);
        const double b = percentile(axis_values, kBboxPercentileHigh);
        const double margin = kBboxMarginFraction * (b - a);
        lo[axis] = a - margin;
        hi[axis] = b + margin;
    }
    return {lo, hi};
}

/// Grow a voxel mask by kInterpolationDilateVoxels 6-neighbourhood iterations.
std::vector<std::uint8_t> dilate_observed(const std::vector<std::uint8_t>& mask,
                                          const GridSpec& grid) {
    std::vector<std::uint8_t> current = mask;
    std::vector<std::uint8_t> next = mask;
    for (int iteration = 0; iteration < kInterpolationDilateVoxels; ++iteration) {
        bool changed = false;
        for (int z = 0; z < grid.nz; ++z) {
            for (int y = 0; y < grid.ny; ++y) {
                for (int x = 0; x < grid.nx; ++x) {
                    const std::size_t index = grid.index(x, y, z);
                    if (current[index] != 0U) {
                        next[index] = 1U;
                        continue;
                    }
                    const int neighbours[6][3] = {{x - 1, y, z}, {x + 1, y, z}, {x, y - 1, z},
                                                  {x, y + 1, z}, {x, y, z - 1}, {x, y, z + 1}};
                    bool touches = false;
                    for (const auto& n : neighbours) {
                        if (grid.contains(n[0], n[1], n[2]) &&
                            current[grid.index(n[0], n[1], n[2])] != 0U) {
                            touches = true;
                            break;
                        }
                    }
                    next[index] = touches ? 1U : 0U;
                    changed = changed || touches;
                }
            }
        }
        current.swap(next);
        if (!changed) {
            break;  // the mask stopped growing: further sweeps are no-ops
        }
    }
    return current;
}

}  // namespace

std::pair<Eigen::Vector3d, Eigen::Vector3d> build_model_bbox(const Chunk& chunk,
                                                             const std::string& source_data) {
    std::vector<Eigen::Vector3d> points;
    if (source_data == "point_cloud") {
        points = chunk.point_cloud().points;
    } else {
        for (const Track& track : chunk.tie_points().tracks) {
            if (track.point.has_value()) {
                points.push_back(*track.point);
            }
        }
    }
    return compute_bbox(std::move(points));
}

// -- validated parameter bundle ----------------------------------------------

namespace {

/// The validated choice of `interpolation`.
enum class Interpolation { Disabled, Enabled, Extrapolated };

/// The validated choice of `source_data`.
enum class SourceData { DepthMaps, PointCloud };

/// Everything apply() needs, resolved once so the stage helpers take plain
/// values rather than the raw Params.
struct ResolvedOptions {
    SourceData source = SourceData::DepthMaps;
    Interpolation interpolation = Interpolation::Enabled;
    std::int64_t volume_resolution = 128;
    std::string face_count = "medium";
    std::int64_t face_count_custom = kFaceCountCustomDefault;
    bool vertex_colors = true;
    bool vertex_confidence = true;
};

/// The depth maps whose camera exists and is enabled, together with the pose
/// and scaled calibration each contributes. Shared by the volume builder and
/// the vertex colourizer so both see exactly the same views.
struct FusionViewSet {
    std::vector<meshing::FusionView> views;      ///< for tv_l1_fusion
    std::vector<const DepthMap*> maps;           ///< parallel to views
    std::vector<Calibration> calibrations;       ///< parallel to views (scaled)
    std::vector<Eigen::Matrix3d> rotations;      ///< parallel to views
    std::vector<Eigen::Vector3d> translations;   ///< parallel to views
};

/// Collect every depth map whose camera exists and is enabled (the same filter
/// BuildDepthMaps / BuildPointCloud apply), resolving the mesh-volume pose and
/// the scaled calibration for each.
FusionViewSet collect_fusion_views(const Chunk& chunk) {
    std::unordered_map<std::int32_t, const Camera*> camera_by_id;
    for (const Camera& camera : chunk.cameras()) {
        camera_by_id[camera.id] = &camera;
    }
    FusionViewSet set;
    for (const DepthMap& map : chunk.depth_maps()) {
        const auto it = camera_by_id.find(map.camera_id);
        if (it == camera_by_id.end() || !it->second->enabled) {
            continue;  // camera gone, or not aligned: its map is stale
        }
        const Camera& camera = *it->second;
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk.sensors().size())) {
            throw ValueError("BuildModel: camera " + std::to_string(map.camera_id) +
                             " has no sensor");
        }
        const Calibration calibration = scaled_calibration(
            chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration, map.downscale);

        meshing::FusionView view;
        view.map = &map;
        view.calibration = calibration;
        view.R = pose_R(camera.pose);
        view.t = pose_t(camera.pose);

        set.maps.push_back(&map);
        set.calibrations.push_back(calibration);
        set.rotations.push_back(view.R);
        set.translations.push_back(view.t);
        set.views.push_back(std::move(view));
    }
    return set;
}

/// Build the volume from the depth maps by TV-L1 fusion.
meshing::VolumeResult build_volume_from_depth_maps(const Chunk& chunk,
                                                   const Eigen::Vector3d& bbox_min,
                                                   const Eigen::Vector3d& bbox_max,
                                                   const ResolvedOptions& resolved) {
    const FusionViewSet set = collect_fusion_views(chunk);
    if (set.views.empty()) {
        throw ValueError("No depth maps");
    }
    meshing::TvFusionOptions options;
    options.max_resolution = static_cast<int>(resolved.volume_resolution);
    // Device path with an automatic per-build CPU fallback (M10). The GPU run
    // differs from the CPU only in arithmetic precision (float32 vs double) and
    // in stopping at a settle threshold; see LIMITATIONS.md.
    return meshing::tv_l1_fusion_device(set.views, bbox_min, bbox_max, options);
}

/// Build the volume from the dense point cloud by screened Poisson
/// reconstruction; the cloud must carry outward normals.
meshing::VolumeResult build_volume_from_point_cloud(const Chunk& chunk,
                                                    const Eigen::Vector3d& bbox_min,
                                                    const Eigen::Vector3d& bbox_max,
                                                    const ResolvedOptions& resolved) {
    const PointCloud& cloud = chunk.point_cloud();
    if (cloud.points.empty()) {
        throw ValueError("No point cloud");
    }
    if (cloud.normals.size() != cloud.points.size()) {
        throw ValueError("BuildModel: the point cloud has no normals; screened Poisson "
                         "reconstruction needs them (run BuildPointCloud first)");
    }
    meshing::PoissonOptions options;
    options.max_resolution = static_cast<int>(resolved.volume_resolution);
    return meshing::screened_poisson(cloud, bbox_min, bbox_max, options);
}

/// Resolve the mask handed to marching tetrahedra from the interpolation mode.
/// Returns a pointer to storage that outlives the call (or nullptr for
/// "extrapolated", which keeps every extracted face).
const std::vector<std::uint8_t>* resolve_mask(const meshing::VolumeResult& volume,
                                              Interpolation interpolation,
                                              std::vector<std::uint8_t>& scratch) {
    switch (interpolation) {
        case Interpolation::Disabled:
            return &volume.observed;
        case Interpolation::Enabled:
            scratch = dilate_observed(volume.observed, volume.field.spec);
            return &scratch;
        case Interpolation::Extrapolated:
        default:
            return nullptr;
    }
}

/// Extract the surface, drop small components, decimate to the face target and
/// compact the vertex arrays. Returns the cleaned mesh.
Mesh extract_and_clean(const meshing::VolumeResult& volume, const ResolvedOptions& resolved) {
    std::vector<std::uint8_t> scratch;
    const std::vector<std::uint8_t>* mask = resolve_mask(volume, resolved.interpolation, scratch);

    Mesh mesh = extract_isosurface(volume.field, volume.iso, mask);
    if (mesh.empty()) {
        throw ValueError("BuildModel: the reconstruction volume produced no surface; "
                         "try a larger volume_resolution or a different source_data");
    }
    remove_small_components(mesh, kRemoveSmallFraction);

    const std::size_t extracted_faces = mesh.faces.size();
    std::size_t target = extracted_faces;
    if (resolved.face_count == "low") {
        target = static_cast<std::size_t>(static_cast<double>(extracted_faces) * kFaceCountLow);
    } else if (resolved.face_count == "medium") {
        target = static_cast<std::size_t>(static_cast<double>(extracted_faces) * kFaceCountMedium);
    } else if (resolved.face_count == "high") {
        target = static_cast<std::size_t>(static_cast<double>(extracted_faces) * kFaceCountHigh);
    } else {  // "custom"
        target = static_cast<std::size_t>(resolved.face_count_custom);
    }

    // Never decimate up: a target above the extracted count is already met.
    if (target < mesh.faces.size()) {
        DecimateOptions options;
        options.target_faces = target;
        options.preserve_boundary = true;
        decimate_quadric(mesh, options);
    }
    remove_unreferenced_vertices(mesh);
    return mesh;
}

/// Pack a calibration's 11 doubles in the order try_project_point expects.
void pack_calibration(const Calibration& calibration, double* params) {
    params[0] = calibration.f;
    params[1] = calibration.cx;
    params[2] = calibration.cy;
    params[3] = calibration.b1;
    params[4] = calibration.b2;
    params[5] = calibration.k1;
    params[6] = calibration.k2;
    params[7] = calibration.k3;
    params[8] = calibration.k4;
    params[9] = calibration.p1;
    params[10] = calibration.p2;
}

/// Colour every vertex from the source images and count its supporting views.
///
/// A view supports a vertex when the vertex projects inside the depth map, the
/// map has a valid depth there, and the vertex's camera-frame distance agrees
/// with that depth to within kColourDepthTolerance (the occlusion / silhouette
/// guard). The colour is sampled from the FULL-RESOLUTION image. Vertices no
/// view supports get kUnseenColour and zero confidence.
void colour_vertices(const Document& document, const Chunk& chunk, const FusionViewSet& set,
                     Mesh& mesh) {
    const std::size_t vertex_count = mesh.vertices.size();
    if (vertex_count == 0 || set.views.empty()) {
        return;
    }

    // Decode each view's full-resolution image once, in view order.
    std::unordered_map<std::int32_t, const Frame*> frame_of;
    for (const auto& frame : chunk.frames()) {
        if (frame->camera_id() >= 0) {
            frame_of.emplace(frame->camera_id(), frame.get());
        }
    }
    std::vector<Image> images(set.views.size());
    for (std::size_t v = 0; v < set.views.size(); ++v) {
        const auto it = frame_of.find(set.maps[v]->camera_id);
        if (it == frame_of.end()) {
            throw ValueError("BuildModel: camera " + std::to_string(set.maps[v]->camera_id) +
                             " has no linked frame to colour from");
        }
        images[v] = read_image(resolve_frame_path(document, it->second->path()).string());
    }

    // Pre-pack each view's calibration parameters for the inner loop.
    std::vector<std::array<double, kCalibrationParamCount>> params(set.views.size());
    for (std::size_t v = 0; v < set.views.size(); ++v) {
        pack_calibration(set.calibrations[v], params[v].data());
    }

    std::vector<Eigen::Vector3d> accum(vertex_count, Eigen::Vector3d::Zero());
    std::vector<std::uint32_t> support(vertex_count, 0U);

    const std::size_t view_count = set.views.size();
    const Eigen::Vector3d* vertices = mesh.vertices.data();

#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t vi = 0; vi < static_cast<std::int64_t>(vertex_count); ++vi) {
        const std::size_t index = static_cast<std::size_t>(vi);
        const Eigen::Vector3d& world = vertices[index];
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        std::uint32_t views_seen = 0U;
        for (std::size_t v = 0; v < view_count; ++v) {
            const DepthMap& map = *set.maps[v];
            const Eigen::Vector3d cam = set.rotations[v] * world + set.translations[v];
            if (!(cam.z() > kProjectZEpsilon)) {
                continue;  // at or behind the image plane for a Frame sensor
            }
            const double p_cam[3] = {cam.x(), cam.y(), cam.z()};
            double pixel[2] = {0.0, 0.0};
            if (!try_project_point<double>(set.calibrations[v].type, map.width, map.height,
                                           params[v].data(), p_cam, pixel)) {
                continue;
            }
            const double u = pixel[0];
            const double w = pixel[1];
            if (u < 0.0 || w < 0.0 || u >= static_cast<double>(map.width) ||
                w >= static_cast<double>(map.height)) {
                continue;  // projects outside this depth map
            }
            const int mu = static_cast<int>(std::floor(u));
            const int mv = static_cast<int>(std::floor(w));
            const double depth =
                static_cast<double>(map.depths[static_cast<std::size_t>(mv) *
                                                    static_cast<std::size_t>(map.width) +
                                                static_cast<std::size_t>(mu)]);
            if (!(depth > 0.0)) {
                continue;  // no depth here
            }
            const double distance = cam.norm();
            if (std::abs(distance - depth) / depth >= kColourDepthTolerance) {
                continue;  // occluded, or a silhouette mismatch
            }

            // Full-resolution pixel: the depth-map pixel (mu, mv) covers the
            // full-res block starting at (mu * ds, mv * ds) (calibration comment:
            // full-res u = depth-map u' * ds).
            const int ds = map.downscale;
            const Image& image = images[v];
            const int fx = mu * ds;
            const int fy = mv * ds;
            if (fx < 0 || fy < 0 || fx >= image.width() || fy >= image.height()) {
                continue;
            }
            sum.x() += static_cast<double>(image.at(fx, fy, 0));
            sum.y() += static_cast<double>(image.at(fx, fy, image.channels() > 1 ? 1 : 0));
            sum.z() += static_cast<double>(image.at(fx, fy, image.channels() > 2 ? 2 : 0));
            ++views_seen;
        }
        accum[index] = sum;
        support[index] = views_seen;
    }

    // The extraction / decimation stages leave the attribute vectors empty, so
    // they are sized here before the accumulate loop writes into them.
    mesh.colours.assign(vertex_count * 3, kUnseenColour);
    mesh.confidence.assign(vertex_count, 0.0F);
    for (std::size_t i = 0; i < vertex_count; ++i) {
        const std::uint32_t seen = support[i];
        if (seen == 0U) {
            continue;  // leave the unseen colour / zero confidence in place
        }
        const double inverse = 1.0 / static_cast<double>(seen);
        mesh.colours[i * 3 + 0] = static_cast<std::uint8_t>(
            std::lround(accum[i].x() * inverse));
        mesh.colours[i * 3 + 1] = static_cast<std::uint8_t>(
            std::lround(accum[i].y() * inverse));
        mesh.colours[i * 3 + 2] = static_cast<std::uint8_t>(
            std::lround(accum[i].z() * inverse));
        mesh.confidence[i] = static_cast<float>(seen);
    }
}

}  // namespace

void BuildModelTask::declare_params(Params& params) const {
    // Names follow Metashape's public Python API; the value sets and
    // presets are ours.
    ParamSpec source_data{"source_data", ParamType::String, std::string("depth_maps"),
                          "Volume source: \"depth_maps\" (TV-L1 fusion) or \"point_cloud\" "
                          "(screened Poisson, needs cloud normals)."};
    source_data.choices = {"depth_maps", "point_cloud"};
    params.declare(source_data);

    ParamSpec surface_type{"surface_type", ParamType::String, std::string("arbitrary"),
                           "Surface class. Only \"arbitrary\" (general) is implemented; "
                           "\"height_field\" is rejected."};
    surface_type.choices = {"arbitrary"};
    params.declare(surface_type);

    ParamSpec interpolation{"interpolation", ParamType::String, std::string("enabled"),
                            "How far the surface may bridge unmeasured space: \"disabled\", "
                            "\"enabled\" or \"extrapolated\"."};
    interpolation.choices = {"disabled", "enabled", "extrapolated"};
    params.declare(interpolation);

    ParamSpec face_count{"face_count", ParamType::String, std::string("medium"),
                         "Decimation target: \"low\", \"medium\", \"high\" or \"custom\"."};
    face_count.choices = {"low", "medium", "high", "custom"};
    params.declare(face_count);

    ParamSpec face_count_custom{"face_count_custom", ParamType::Int, kFaceCountCustomDefault,
                                "Face target used when face_count is \"custom\"."};
    face_count_custom.minimum = std::int64_t{1};
    params.declare(face_count_custom);

    params.declare({"vertex_colors", ParamType::Bool, true,
                    "Colour the mesh vertices from the source images."});
    params.declare({"vertex_confidence", ParamType::Bool, true,
                    "Store a per-vertex confidence (the number of supporting views)."});

    ParamSpec volume_resolution{"volume_resolution", ParamType::Int, std::int64_t{128},
                                "Voxels along the longest bounding-box axis."};
    volume_resolution.minimum = kVolumeResolutionMin;
    volume_resolution.maximum = kVolumeResolutionMax;
    params.declare(volume_resolution);

    params.declare({"label", ParamType::String, std::string{},
                    "Label for the new model asset. Empty selects \"Model <n>\". Each run "
                    "creates a new asset; the previous model keeps its own textures."});
}

std::vector<Stage> BuildModelTask::stages() const {
    return {Stage{"initialize", 0}, Stage{"build_volume", 0}, Stage{"extract_surface", 0},
            Stage{"finalize", 0}};
}

void BuildModelTask::apply(Document& document, Params& params, const Progress& progress) {
    // -- resolve and validate every parameter before touching the chunk ------
    progress.update("initialize", 0.0);
    const ResolvedOptions resolved = [&] {
        ResolvedOptions r;
        r.source = [&] {
            const std::string value = params.get_string("source_data");
            if (value == "depth_maps") {
                return SourceData::DepthMaps;
            }
            if (value == "point_cloud") {
                return SourceData::PointCloud;
            }
            throw ValueError("BuildModel: source_data must be one of \"depth_maps\", "
                             "\"point_cloud\" (got \"" +
                             value + "\")");
        }();
        const std::string surface = params.get_string("surface_type");
        if (surface == "height_field") {
            throw ValueError("BuildModel: height_field surfaces are not supported yet");
        }
        if (surface != "arbitrary") {
            throw ValueError("BuildModel: surface_type must be \"arbitrary\" or \"height_field\" "
                             "(got \"" +
                             surface + "\")");
        }
        const std::string interpolation = params.get_string("interpolation");
        if (interpolation == "disabled") {
            r.interpolation = Interpolation::Disabled;
        } else if (interpolation == "enabled") {
            r.interpolation = Interpolation::Enabled;
        } else if (interpolation == "extrapolated") {
            r.interpolation = Interpolation::Extrapolated;
        } else {
            throw ValueError("BuildModel: interpolation must be one of \"disabled\", "
                             "\"enabled\", \"extrapolated\" (got \"" +
                             interpolation + "\")");
        }
        const std::string face_count = params.get_string("face_count");
        if (face_count != "low" && face_count != "medium" && face_count != "high" &&
            face_count != "custom") {
            throw ValueError("BuildModel: face_count must be one of \"low\", \"medium\", "
                             "\"high\", \"custom\" (got \"" +
                             face_count + "\")");
        }
        r.face_count = face_count;
        r.face_count_custom = params.get_int("face_count_custom");
        if (r.face_count_custom < 1) {
            throw ValueError("BuildModel: face_count_custom must be at least 1");
        }
        r.volume_resolution = params.get_int("volume_resolution");
        if (r.volume_resolution < kVolumeResolutionMin ||
            r.volume_resolution > kVolumeResolutionMax) {
            throw ValueError("BuildModel: volume_resolution must be between " +
                             std::to_string(kVolumeResolutionMin) + " and " +
                             std::to_string(kVolumeResolutionMax));
        }
        r.vertex_colors = params.get_bool("vertex_colors");
        r.vertex_confidence = params.get_bool("vertex_confidence");
        return r;
    }();

    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("BuildModel needs an active chunk");
    }
    // Read through a CONST view throughout: the non-const legacy accessors lazily
    // create an empty asset, so a rejected run would otherwise leave a phantom
    // depth-map set or point cloud behind.
    const Chunk& source_chunk = *chunk;
    if (resolved.source == SourceData::DepthMaps) {
        if (source_chunk.depth_maps().empty()) {
            throw ValueError("No depth maps");
        }
        // The fusion needs at least two aligned cameras to bound the volume.
        std::size_t enabled = 0;
        for (const Camera& camera : chunk->cameras()) {
            if (camera.enabled) {
                ++enabled;
            }
        }
        if (enabled < 2) {
            throw ValueError("BuildModel: source_data \"depth_maps\" needs at least two "
                             "aligned cameras");
        }
    } else {
        if (source_chunk.point_cloud().points.empty()) {
            throw ValueError("No point cloud");
        }
        if (source_chunk.point_cloud().normals.size() !=
            source_chunk.point_cloud().points.size()) {
            throw ValueError("BuildModel: the point cloud has no normals; screened Poisson "
                             "reconstruction needs them (run BuildPointCloud first)");
        }
    }

    // -- bounding box: per-axis percentile of the source points --------------
    const std::pair<Eigen::Vector3d, Eigen::Vector3d> box =
        build_model_bbox(*chunk, resolved.source == SourceData::PointCloud ? "point_cloud"
                                                                           : "depth_maps");
    progress.update("initialize", 1.0);

    // -- build the volume ----------------------------------------------------
    progress.update("build_volume", 0.0);
    meshing::VolumeResult volume =
        resolved.source == SourceData::DepthMaps
            ? build_volume_from_depth_maps(*chunk, box.first, box.second, resolved)
            : build_volume_from_point_cloud(*chunk, box.first, box.second, resolved);
    progress.update("build_volume", 1.0);

    // -- extract, clean up, decimate -----------------------------------------
    progress.update("extract_surface", 0.0);
    Mesh mesh = extract_and_clean(volume, resolved);
    const std::size_t extracted_faces = mesh.faces.size();
    progress.update("extract_surface", 1.0);

    // -- colour + confidence -------------------------------------------------
    progress.update("finalize", 0.0);
    const bool has_depth_maps = !source_chunk.depth_maps().empty();
    const bool want_attributes = resolved.vertex_colors || resolved.vertex_confidence;
    if (want_attributes && has_depth_maps) {
        const FusionViewSet set = collect_fusion_views(*chunk);
        if (!set.views.empty()) {
            colour_vertices(document, *chunk, set, mesh);
        }
    }
    if (!resolved.vertex_colors) {
        mesh.colours.clear();
    }
    if (!resolved.vertex_confidence) {
        mesh.confidence.clear();
    }

    const std::size_t vertex_count = mesh.vertices.size();
    const std::size_t face_count = mesh.faces.size();
    // A new model asset per run: an earlier model (and the texture pages that
    // address ITS uv layout) is kept untouched.
    ModelAsset& asset = chunk->add_model(params.get_string("label"));
    *asset.data = std::move(mesh);
    progress.update("finalize", 1.0);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "BuildModel: " << face_count << " faces, " << vertex_count
                            << " vertices from "
                            << (resolved.source == SourceData::DepthMaps ? "depth_maps"
                                                                         : "point_cloud")
                            << " (extracted " << extracted_faces << " faces, face_count "
                            << resolved.face_count << ") -> asset " << asset.id << " '"
                            << asset.label << "'";
    }
}

}  // namespace alphashape

