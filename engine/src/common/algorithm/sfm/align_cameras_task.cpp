// AlignCameras: incremental structure-from-motion over the chunk's tie points.
//
// The adapter between the chunk's data model and build_incremental
// (sfm/structure_builder.hpp):
//
//   * chunk -> BuilderInput: one BuilderCamera per camera that has a keypoint
//     set, in camera-id order, carrying its sensor's Calibration, its keypoints
//     converted to the Calibration pixel convention (keypoint_to_pixel), and --
//     when continuing rather than resetting -- its current pose. The tracks and
//     their existing points are passed straight through.
//   * BuilderResult -> chunk: each registered camera's pose, each track's point,
//     and the chunk's bounding region.
//
// Conventions:
//   * Camera::pose is the 4x4 world -> camera transform and AbsolutePose is
//     (R, t) with X_cam = R X_world + t, so the 4x4 is exactly [R t; 0 1].
//   * A commit gives EVERY camera a value: registered ones are enabled with
//     their pose, unregistered ones are disabled with the identity pose, so an
//     impossible alignment never leaves a stale pose behind.
//   * After the builder, the staged result is georeferenced (georeference.hpp):
//     if the cameras carry GNSS references the whole reconstruction is placed
//     into a metric ENU frame and the chunk transform becomes enu_to_ecef(origin);
//     otherwise the chunk is left in the arbitrary similarity frame the builder
//     produced (identity transform, unset CRS).

#include "common/algorithm/sfm/align_cameras_task.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/feature/detect_features_task.hpp"  // keypoint_to_pixel
#include "common/algorithm/sfm/absolute_pose.hpp"
#include "common/algorithm/sfm/georeference.hpp"
#include "common/algorithm/sfm/structure_builder.hpp"

namespace alphashape {
namespace {

/// The 4x4 [R t; 0 1] form of an absolute pose (X_cam = R X_world + t).
Transform pose_to_transform(const sfm::AbsolutePose& pose) {
    Transform transform = Transform::Identity();
    transform.block<3, 3>(0, 0) = pose.R;
    transform.block<3, 1>(0, 3) = pose.t;
    return transform;
}

/// The (R, t) form of a 4x4 world->camera transform. Exact inverse of the above.
sfm::AbsolutePose transform_to_pose(const Transform& transform) {
    sfm::AbsolutePose pose;
    pose.R = transform.block<3, 3>(0, 0);
    pose.t = transform.block<3, 1>(0, 3);
    return pose;
}

/// Linear-interpolated percentile of an already-sorted vector (ours: a plain
/// order-statistic interpolation; q is in [0, 1]).
double percentile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double pos = q * (static_cast<double>(sorted.size()) - 1.0);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

}  // namespace

void AlignCamerasTask::declare_params(Params& params) const {
    params.declare({"reset_alignment", ParamType::Bool, false,
                    "Discard the existing alignment and realign every camera from scratch."});
}

std::vector<Stage> AlignCamerasTask::stages() const {
    return {Stage{"prepare", 0}, Stage{"align", 0}};
}

void AlignCamerasTask::apply(Document& document, Params& params, const Progress& progress) {
    const bool reset_alignment = params.get_bool("reset_alignment");

    // -- validate everything before touching the chunk -----------------------
    progress.update("prepare", 0.0);
    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("AlignCameras needs an active chunk");
    }
    const TiePoints& tie_points = chunk->tie_points();
    if (tie_points.matches.empty() || tie_points.tracks.empty()) {
        throw ValueError("AlignCameras: no tie points \u2014 run MatchPhotos first");
    }

    std::unordered_map<std::int32_t, const Camera*> camera_by_id;
    for (const Camera& camera : chunk->cameras()) {
        camera_by_id[camera.id] = &camera;
    }
    std::unordered_map<std::int32_t, const KeypointSet*> set_by_id;
    for (const KeypointSet& set : tie_points.keypoint_sets) {
        set_by_id[set.camera_id] = &set;
    }

    // Every camera a match or a track references must be ready to align.
    const auto require_ready = [&](std::int32_t id) {
        const auto camera = camera_by_id.find(id);
        if (camera == camera_by_id.end()) {
            throw ValueError("AlignCameras: tie points reference unknown camera " +
                             std::to_string(id));
        }
        if (camera->second->sensor_id < 0 ||
            camera->second->sensor_id >= static_cast<std::int32_t>(chunk->sensors().size())) {
            throw ValueError("AlignCameras: camera " + std::to_string(id) + " has no sensor");
        }
        if (set_by_id.find(id) == set_by_id.end()) {
            throw ValueError("AlignCameras: camera " + std::to_string(id) +
                             " has no keypoint set");
        }
    };
    for (const PairMatches& pair : tie_points.matches) {
        require_ready(pair.camera_a);
        require_ready(pair.camera_b);
    }
    for (const Track& track : tie_points.tracks) {
        for (const Observation& observation : track.observations) {
            require_ready(observation.camera_id);
        }
    }

    // -- build the builder input --------------------------------------------
    // One BuilderCamera per camera that has a keypoint set, in camera-id order.
    std::vector<std::int32_t> input_ids;
    for (const Camera& camera : chunk->cameras()) {
        if (set_by_id.find(camera.id) != set_by_id.end()) {
            input_ids.push_back(camera.id);
        }
    }
    std::sort(input_ids.begin(), input_ids.end());

    sfm::BuilderInput input;
    input.cameras.reserve(input_ids.size());
    for (const std::int32_t id : input_ids) {
        const Camera& camera = *camera_by_id.at(id);
        const KeypointSet& set = *set_by_id.at(id);
        require_ready(id);  // a camera with keypoints but no sensor cannot be aligned

        sfm::BuilderCamera builder_camera;
        builder_camera.id = id;
        builder_camera.calibration =
            chunk->sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration;
        builder_camera.pixels.reserve(set.keypoints.size());
        for (const Keypoint& keypoint : set.keypoints) {
            builder_camera.pixels.push_back(keypoint_to_pixel(keypoint));
        }
        // Continue from the existing alignment unless explicitly resetting.
        if (!reset_alignment && camera.enabled) {
            builder_camera.initial_pose = transform_to_pose(camera.pose);
        }
        input.cameras.push_back(std::move(builder_camera));
    }
    input.matches = tie_points.matches;
    input.tracks = tie_points.tracks;
    if (reset_alignment) {
        for (Track& track : input.tracks) {
            track.point = std::nullopt;  // a reset recomputes every point
        }
    }
    progress.update("prepare", 1.0);

    // -- run incremental SfM -------------------------------------------------
    // The cancelled callback is polled between registrations, outside any
    // OpenMP region, so it may safely call back into the host.
    progress.update("align", 0.0);
    const sfm::BuilderResult result =
        sfm::build_incremental(input, sfm::BuilderOptions{},
                               [&progress]() { return progress.is_cancelled(); });
    progress.update("align", 1.0);

    // -- compute the new chunk contents into locals --------------------------
    // Everything below is pure computation; the chunk is replaced only once all
    // of it has succeeded, so an exception or a cancellation anywhere above
    // leaves the chunk exactly as it was.
    std::vector<Camera> new_cameras = chunk->cameras();
    std::unordered_map<std::int32_t, std::size_t> camera_position;
    for (std::size_t i = 0; i < new_cameras.size(); ++i) {
        camera_position[new_cameras[i].id] = i;
    }
    for (std::size_t k = 0; k < input_ids.size(); ++k) {
        Camera& camera = new_cameras[camera_position.at(input_ids[k])];
        if (result.poses[k].has_value()) {
            camera.pose = pose_to_transform(*result.poses[k]);
            camera.enabled = true;
        } else {
            camera.pose = Transform::Identity();
            camera.enabled = false;
        }
    }

    std::vector<Track> new_tracks = tie_points.tracks;
    for (std::size_t t = 0; t < new_tracks.size(); ++t) {
        new_tracks[t].point = result.points[t];
    }

    // The chunk region: ours. The centre is the per-axis median of the
    // triangulated points and the size is the per-axis 5th-95th percentile span
    // scaled by 1.1 (a little margin so the region contains the bulk of the
    // cloud). Fewer than four points is not enough to define a region, so the
    // existing one is left alone.
    Region new_region = chunk->region();
    std::vector<Eigen::Vector3d> points;
    points.reserve(new_tracks.size());
    for (const Track& track : new_tracks) {
        if (track.point.has_value()) {
            points.push_back(*track.point);
        }
    }
    if (points.size() >= 4) {
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<double> values;
            values.reserve(points.size());
            for (const Eigen::Vector3d& p : points) {
                values.push_back(p[axis]);
            }
            std::sort(values.begin(), values.end());
            new_region.centre[axis] = percentile(values, 0.5);
            const double span = percentile(values, 0.95) - percentile(values, 0.05);
            new_region.size[axis] = span * 1.1;
        }
    }

    // -- stage the result on a copy and georeference it ----------------------
    // Everything below is pure computation on a copy of the chunk; the real
    // chunk is replaced only once georeferencing has succeeded, so an exception
    // or a cancellation anywhere above leaves the chunk exactly as it was.
    Chunk staged = *chunk;
    staged.cameras() = std::move(new_cameras);
    staged.tie_points().tracks = std::move(new_tracks);
    staged.region() = new_region;
    // Place the reconstruction into a metric ENU frame when the cameras carry
    // GNSS references; otherwise it stays in the builder's arbitrary frame.
    const sfm::GeoreferenceReport geo_report = sfm::georeference_chunk(staged);

    // -- commit atomically ---------------------------------------------------
    *chunk = std::move(staged);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "AlignCameras: registered " << result.registered << "/"
                            << input_ids.size() << " cameras, RMS " << result.final_rms_px << " px";
        if (geo_report.applied) {
            log(LogLevel::Info) << "AlignCameras: georeferenced from "
                                << geo_report.references_used << "/" << geo_report.references_total
                                << " references, RMS " << geo_report.rms_error_m << " m";
        } else {
            log(LogLevel::Info) << "AlignCameras: not georeferenced (" << geo_report.reason << ")";
        }
    }
}

}  // namespace alphashape
