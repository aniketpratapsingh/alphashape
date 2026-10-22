// OptimizeCameras: self-calibration bundle adjustment over the aligned chunk.
//
// The adapter between the chunk's data model and sfm::bundle_adjust:
//
//   * chunk -> BAProblem: one pose per aligned (enabled) camera; one calibration
//     per sensor used by those cameras (shared by every camera on the sensor, so
//     a repeated self-calibration is measurable and sensor fixed_params are
//     honoured); one point per triangulated track; one observation per track
//     observation in an aligned camera, at keypoint_to_pixel.
//   * free mask: coefficient X of a sensor is free iff the task's fit_X is true
//     AND X is not in that sensor's fixed_params.
//   * GNSS priors: only when the chunk is georeferenced (transform != identity).
//     Each aligned camera with an enabled reference contributes a centre prior in
//     the local ENU frame -- inverse(transform) * geodetic_to_ecef(lon, lat, alt)
//     -- weighted by reference.accuracy. A NaN altitude leaves the up axis
//     unconstrained and is converted as alt = 0.
//   * Gauge: when >= 3 priors have both horizontal axes constrained and their
//     centres are not collinear, the priors pin all 7 similarity DoF, so every
//     pose is left free and no scale anchor is set. Otherwise the first aligned
//     camera is held fully constant and the scale is anchored on the aligned
//     camera with the largest baseline to it (choose_scale_anchor_axis).
//
// Everything is computed into a staged copy and committed with one move, so an
// exception anywhere leaves the chunk untouched.

#include "common/algorithm/sfm/optimize_cameras_task.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/feature/detect_features_task.hpp"  // keypoint_to_pixel
#include "common/algorithm/sfm/bundle_adjustment.hpp"
#include "model/camera/calibration.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape {
namespace {

/// Calibration coefficient names, in the packed order (calibration.hpp):
/// f, cx, cy, b1, b2, k1, k2, k3, k4, p1, p2. Matches Sensor::fixed_params.
constexpr std::array<const char*, kCalibrationParamCount> kParamNames = {
    "f", "cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2"};

/// The 4x4 [R t; 0 1] form of a bundle-adjustment pose (X_cam = R X_world + t).
Transform transform_from_pose(const sfm::BAPose& pose) {
    Transform transform = Transform::Identity();
    transform.block<3, 3>(0, 0) = pose.R;
    transform.block<3, 1>(0, 3) = pose.t;
    return transform;
}

sfm::BAPose pose_from_transform(const Transform& transform) {
    sfm::BAPose pose;
    pose.R = transform.block<3, 3>(0, 0);
    pose.t = transform.block<3, 1>(0, 3);
    return pose;
}

/// True when the centres are not collinear: the farthest pair defines a line and
/// the other centres must deviate from it by more than a relative tolerance.
bool centres_not_collinear(const std::vector<Eigen::Vector3d>& centres) {
    if (centres.size() < 3) {
        return false;
    }
    std::size_t a = 0, b = 1;
    double best = -1.0;
    for (std::size_t i = 0; i < centres.size(); ++i) {
        for (std::size_t j = i + 1; j < centres.size(); ++j) {
            const double d = (centres[i] - centres[j]).squaredNorm();
            if (d > best) {
                best = d;
                a = i;
                b = j;
            }
        }
    }
    const Eigen::Vector3d dir = centres[a] - centres[b];
    const double len = dir.norm();
    if (!(len > 0.0)) {
        return false;
    }
    const Eigen::Vector3d u = dir / len;
    double max_perp = 0.0;
    for (const Eigen::Vector3d& c : centres) {
        const Eigen::Vector3d v = c - centres[a];
        max_perp = std::max(max_perp, (v - v.dot(u) * u).norm());
    }
    return max_perp > 1e-3 * len;
}

}  // namespace

void OptimizeCamerasTask::declare_params(Params& params) const {
    // Defaults follow Metashape's public Python API: the principal-point and
    // low-order radial/tangential terms are fit, the affine pair and the highest
    // radial term are not.
    params.declare({"fit_f", ParamType::Bool, true, "Adjust the focal length."});
    params.declare({"fit_cx", ParamType::Bool, true, "Adjust the principal point X offset."});
    params.declare({"fit_cy", ParamType::Bool, true, "Adjust the principal point Y offset."});
    params.declare({"fit_b1", ParamType::Bool, false, "Adjust the aspect (affinity) term b1."});
    params.declare({"fit_b2", ParamType::Bool, false, "Adjust the skew term b2."});
    params.declare({"fit_k1", ParamType::Bool, true, "Adjust the radial distortion k1."});
    params.declare({"fit_k2", ParamType::Bool, true, "Adjust the radial distortion k2."});
    params.declare({"fit_k3", ParamType::Bool, true, "Adjust the radial distortion k3."});
    params.declare({"fit_k4", ParamType::Bool, false, "Adjust the radial distortion k4."});
    params.declare({"fit_p1", ParamType::Bool, true, "Adjust the tangential distortion p1."});
    params.declare({"fit_p2", ParamType::Bool, true, "Adjust the tangential distortion p2."});
}

std::vector<Stage> OptimizeCamerasTask::stages() const {
    return {Stage{"prepare", 0}, Stage{"optimize", 0}};
}

void OptimizeCamerasTask::apply(Document& document, Params& params, const Progress& progress) {
    // -- validate before touching the chunk ---------------------------------
    progress.update("prepare", 0.0);
    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("OptimizeCameras needs an active chunk");
    }

    // The fit flags, in the packed coefficient order.
    std::array<bool, kCalibrationParamCount> fit = {};
    for (int k = 0; k < kCalibrationParamCount; ++k) {
        fit[static_cast<std::size_t>(k)] =
            params.get_bool("fit_" + std::string(kParamNames[static_cast<std::size_t>(k)]));
    }

    std::vector<std::size_t> camera_index;  // chunk camera -> BA camera
    for (std::size_t i = 0; i < chunk->cameras().size(); ++i) {
        if (chunk->cameras()[i].enabled) {
            camera_index.push_back(i);
        }
    }
    if (camera_index.size() < 2) {
        throw ValueError("OptimizeCameras: align cameras first");
    }

    const TiePoints& tie_points = chunk->tie_points();
    const bool have_points =
        std::any_of(tie_points.tracks.begin(), tie_points.tracks.end(),
                    [](const Track& track) { return track.point.has_value(); });
    if (!have_points) {
        throw ValueError("OptimizeCameras: align cameras first");
    }

    // -- gather the keypoints of every aligned camera -----------------------
    std::unordered_map<std::int32_t, const KeypointSet*> set_by_id;
    for (const KeypointSet& set : tie_points.keypoint_sets) {
        set_by_id[set.camera_id] = &set;
    }

    // -- build the problem --------------------------------------------------
    sfm::BAProblem problem;
    problem.poses.reserve(camera_index.size());

    // One calibration per sensor used by an aligned camera.
    std::unordered_map<std::int32_t, std::size_t> sensor_slot;
    std::vector<std::int32_t> slot_sensor;  // BA calibration -> chunk sensor id
    for (const std::size_t ci : camera_index) {
        const Camera& camera = chunk->cameras()[ci];
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk->sensors().size())) {
            throw ValueError("OptimizeCameras: camera " + std::to_string(camera.id) +
                             " has no sensor");
        }
        if (sensor_slot.find(camera.sensor_id) == sensor_slot.end()) {
            const std::size_t slot = problem.calibrations.size();
            sensor_slot[camera.sensor_id] = slot;
            slot_sensor.push_back(camera.sensor_id);
            problem.calibrations.push_back(
                chunk->sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration);
        }
        problem.camera_sensor.push_back(sensor_slot.at(camera.sensor_id));
        problem.poses.push_back(pose_from_transform(camera.pose));
    }

    // Free mask per sensor: fit_X && X not in sensor.fixed_params.
    problem.fit_intrinsics.resize(problem.calibrations.size());
    for (std::size_t s = 0; s < problem.calibrations.size(); ++s) {
        const Sensor& sensor =
            chunk->sensors()[static_cast<std::size_t>(slot_sensor[s])];
        std::array<bool, kCalibrationParamCount> mask = {};
        for (int k = 0; k < kCalibrationParamCount; ++k) {
            const std::size_t kk = static_cast<std::size_t>(k);
            const bool fixed = std::find(sensor.fixed_params.begin(), sensor.fixed_params.end(),
                                         std::string(kParamNames[kk])) != sensor.fixed_params.end();
            mask[kk] = fit[kk] && !fixed;
        }
        problem.fit_intrinsics[s] = mask;
    }

    // Points: every triangulated track, with the track -> BA point map.
    std::unordered_map<std::size_t, std::size_t> point_slot;
    for (std::size_t t = 0; t < tie_points.tracks.size(); ++t) {
        if (tie_points.tracks[t].point.has_value()) {
            point_slot[t] = problem.points.size();
            problem.points.push_back(*tie_points.tracks[t].point);
        }
    }

    // Observations: track observations in an aligned camera, at keypoint_to_pixel.
    std::unordered_map<std::int32_t, std::size_t> camera_slot;
    for (std::size_t b = 0; b < camera_index.size(); ++b) {
        camera_slot[chunk->cameras()[camera_index[b]].id] = b;
    }
    for (std::size_t t = 0; t < tie_points.tracks.size(); ++t) {
        const auto point = point_slot.find(t);
        if (point == point_slot.end()) {
            continue;
        }
        for (const Observation& observation : tie_points.tracks[t].observations) {
            const auto camera = camera_slot.find(observation.camera_id);
            if (camera == camera_slot.end()) {
                continue;  // not aligned: not part of this refinement
            }
            const KeypointSet* set = set_by_id.at(observation.camera_id);
            if (observation.keypoint_index >= set->keypoints.size()) {
                throw ValueError("OptimizeCameras: keypoint index out of range");
            }
            problem.observations.push_back(
                sfm::BAObservation{camera->second, point->second,
                                   keypoint_to_pixel(set->keypoints[observation.keypoint_index])});
        }
    }
    progress.update("prepare", 1.0);

    // -- GNSS centre priors (georeferenced chunks only) ---------------------
    const Transform& transform = chunk->transform();
    const bool georeferenced =
        (transform - Transform::Identity()).cwiseAbs().maxCoeff() > 1e-9;
    if (georeferenced) {
        const Transform inverse = transform.inverse();
        for (std::size_t b = 0; b < camera_index.size(); ++b) {
            const Camera& camera = chunk->cameras()[camera_index[b]];
            if (!camera.reference.enabled || !camera.reference.location.has_value()) {
                continue;
            }
            const Eigen::Vector3d& loc = *camera.reference.location;
            const bool has_alt = std::isfinite(loc.z());
            const geo::Geodetic geodetic{loc.x(), loc.y(), has_alt ? loc.z() : 0.0};
            const Eigen::Vector3d ecef = geo::geodetic_to_ecef(geodetic);
            sfm::BACentrePrior prior;
            prior.camera = b;
            prior.centre = (inverse.block<3, 3>(0, 0) * ecef) + inverse.block<3, 1>(0, 3);
            prior.sigma = camera.reference.accuracy;
            if (!has_alt) {
                prior.sigma.z() = 0.0;  // no altitude: leave the up axis unconstrained
            }
            problem.centre_priors.push_back(prior);
        }
    }

    // -- gauge --------------------------------------------------------------
    // Count priors with both horizontal axes constrained and collect their
    // centres; three non-collinear ones pin the whole similarity, otherwise fall
    // back to fixing a camera and anchoring the scale.
    std::vector<Eigen::Vector3d> prior_centres;
    for (const sfm::BACentrePrior& prior : problem.centre_priors) {
        const bool east = std::isfinite(prior.sigma.x()) && prior.sigma.x() > 0.0;
        const bool north = std::isfinite(prior.sigma.y()) && prior.sigma.y() > 0.0;
        if (east && north) {
            prior_centres.push_back(prior.centre);
        }
    }

    problem.fixed_pose.assign(problem.poses.size(), 0);
    if (prior_centres.size() >= 3 && centres_not_collinear(prior_centres)) {
        // The priors pin rotation, translation and scale: leave every pose free.
        problem.scale_anchor_camera = -1;
    } else {
        problem.fixed_pose[0] = 1;
        // Anchor the scale on the aligned camera with the largest baseline to
        // camera 0 (its centre is farthest).
        const Eigen::Vector3d centre0 = -problem.poses[0].R.transpose() * problem.poses[0].t;
        std::size_t anchor = 0;
        double best = -1.0;
        for (std::size_t b = 1; b < problem.poses.size(); ++b) {
            const Eigen::Vector3d centre = -problem.poses[b].R.transpose() * problem.poses[b].t;
            const double d = (centre - centre0).norm();
            if (d > best) {
                best = d;
                anchor = b;
            }
        }
        if (anchor != 0) {
            problem.scale_anchor_camera = static_cast<int>(anchor);
            problem.scale_anchor_axis =
                sfm::choose_scale_anchor_axis(problem.poses[0], problem.poses[anchor]);
        } else {
            problem.scale_anchor_camera = -1;  // degenerate: no usable anchor
        }
    }

    // -- solve --------------------------------------------------------------
    progress.update("optimize", 0.0);
    const sfm::BAReport report = sfm::bundle_adjust(problem, sfm::BAOptions{});
    if (!report.usable) {
        throw Error("OptimizeCameras: bundle adjustment did not produce a usable solution");
    }
    progress.update("optimize", 1.0);

    // -- stage the result on a copy -----------------------------------------
    Chunk staged = *chunk;
    for (std::size_t s = 0; s < problem.calibrations.size(); ++s) {
        staged.sensors()[static_cast<std::size_t>(slot_sensor[s])].calibration =
            problem.calibrations[s];
    }
    for (std::size_t b = 0; b < camera_index.size(); ++b) {
        staged.cameras()[camera_index[b]].pose = transform_from_pose(problem.poses[b]);
    }
    for (std::size_t t = 0; t < tie_points.tracks.size(); ++t) {
        const auto point = point_slot.find(t);
        if (point != point_slot.end()) {
            staged.tie_points().tracks[t].point = problem.points[point->second];
        }
    }

    // -- commit atomically --------------------------------------------------
    *chunk = std::move(staged);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "OptimizeCameras: RMS " << report.initial_rms_px << " -> "
                            << report.final_rms_px << " px (" << report.observations
                            << " observations, " << report.iterations << " iterations, prior RMS "
                            << report.prior_rms_sigma << " sigma, " << report.priors
                            << " prior axes)";
    }
}

}  // namespace alphashape
