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
#include "common/io/camera_image.hpp"

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

}  // namespace

void AlignCamerasTask::declare_params(Params& params) const {
    params.declare({"reset_alignment", ParamType::Bool, false,
                    "Discard the existing alignment and realign every camera from scratch."});
    params.declare({"fit_intrinsics", ParamType::Bool, false,
                    "Refine each sensor's focal length, principal point and lens distortion "
                    "during alignment (self-calibration). Use it when the input calibration "
                    "is known to be wrong, e.g. a strong-distortion lens seeded from EXIF "
                    "with zero distortion; otherwise keep it off and refine with Optimize "
                    "Cameras after alignment."});
}

std::vector<Stage> AlignCamerasTask::stages() const {
    return {Stage{"prepare", 0}, Stage{"align", 0}};
}

void AlignCamerasTask::apply(Document& document, Params& params, const Progress& progress) {
    const bool reset_alignment = params.get_bool("reset_alignment");
    const bool fit_intrinsics = params.get_bool("fit_intrinsics");

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
    // A slave sensor shares its master's calibration; an invalid link is a hard
    // error before any work is done (the same check OptimizeCameras makes).
    validate_sensor_masters(chunk->sensors());

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
    // One BuilderSensor per effective sensor, so the builder can honour each
    // sensor's held coefficients and group its cameras into one calibration.
    // `used_sensors` keeps every raw sensor id a camera names, so a slave can
    // receive its master's refined calibration on commit.
    std::vector<std::int32_t> sensor_ids;
    std::vector<std::int32_t> used_sensors;
    for (const std::int32_t id : input_ids) {
        const Camera& camera = *camera_by_id.at(id);
        const KeypointSet& set = *set_by_id.at(id);
        require_ready(id);  // a camera with keypoints but no sensor cannot be aligned

        const std::int32_t effective = effective_sensor_id(chunk->sensors(), camera.sensor_id);
        sfm::BuilderCamera builder_camera;
        builder_camera.id = id;
        builder_camera.sensor_id = effective;
        builder_camera.calibration =
            sensor_pipeline_calibration(chunk->sensors()[static_cast<std::size_t>(effective)]);
        builder_camera.pixels.reserve(set.keypoints.size());
        for (const Keypoint& keypoint : set.keypoints) {
            builder_camera.pixels.push_back(keypoint_to_pixel(keypoint));
        }
        // Continue from the existing alignment unless explicitly resetting.
        if (!reset_alignment && camera.enabled) {
            builder_camera.initial_pose = transform_to_pose(camera.pose);
        }
        input.cameras.push_back(std::move(builder_camera));
        if (std::find(sensor_ids.begin(), sensor_ids.end(), effective) == sensor_ids.end()) {
            sensor_ids.push_back(effective);
        }
        if (std::find(used_sensors.begin(), used_sensors.end(), camera.sensor_id) ==
            used_sensors.end()) {
            used_sensors.push_back(camera.sensor_id);
        }
    }
    for (const std::int32_t effective : sensor_ids) {
        sfm::BuilderSensor sensor;
        sensor.id = effective;
        sensor.fixed_params =
            chunk->sensors()[static_cast<std::size_t>(effective)].fixed_params;
        input.sensors.push_back(std::move(sensor));
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
    sfm::BuilderOptions builder_options;
    builder_options.fit_intrinsics = fit_intrinsics;
    const sfm::BuilderResult result =
        sfm::build_incremental(input, builder_options,
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
        // Alignment rewrites every input pose; any posterior covariance from an
        // earlier solve describes the old pose and is dropped (WP-13.13b).
        camera.location_covariance.setZero();
        camera.rotation_covariance.setZero();
    }

    std::vector<Track> new_tracks = tie_points.tracks;
    for (std::size_t t = 0; t < new_tracks.size(); ++t) {
        new_tracks[t].point = result.points[t];
        // A re-triangulated point invalidates the covariance of the old one.
        new_tracks[t].covariance.reset();
    }

    // The chunk region: ours. Fitted to the triangulated points; fewer than
    // four is not enough to define one, so the existing region is left alone.
    std::vector<Eigen::Vector3d> points;
    points.reserve(new_tracks.size());
    for (const Track& track : new_tracks) {
        if (track.point.has_value()) {
            points.push_back(*track.point);
        }
    }
    const Region new_region = region_from_points(points, chunk->region());

    // -- stage the result on a copy and georeference it ----------------------
    // Everything below is pure computation on a copy of the chunk; the real
    // chunk is replaced only once georeferencing has succeeded, so an exception
    // or a cancellation anywhere above leaves the chunk exactly as it was.
    Chunk staged = *chunk;
    staged.cameras() = std::move(new_cameras);
    staged.tie_points().tracks = std::move(new_tracks);
    staged.region() = new_region;
    // Write each self-calibrated sensor's refined lens back to its ADJUSTED
    // calibration (the field OptimizeCameras writes). A sensor the builder did
    // not refine keeps its input calibration untouched.
    for (const auto& [sensor_id, calibration] : result.refined_calibrations) {
        if (sensor_id < 0 ||
            sensor_id >= static_cast<std::int32_t>(staged.sensors().size())) {
            continue;
        }
        staged.sensors()[static_cast<std::size_t>(sensor_id)].calibration = calibration;
    }
    // A slave sensor shares its master's calibration, so it receives the same
    // refined value (the builder refines the effective/master sensor).
    for (const std::int32_t sensor_id : used_sensors) {
        const std::int32_t master =
            staged.sensors()[static_cast<std::size_t>(sensor_id)].master_sensor_id;
        if (master >= 0) {
            staged.sensors()[static_cast<std::size_t>(sensor_id)].calibration =
                staged.sensors()[static_cast<std::size_t>(master)].calibration;
        }
    }
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
