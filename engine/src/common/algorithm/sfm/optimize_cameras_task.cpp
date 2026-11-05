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
//   * Gauge: when >= 3 anchors have all three axes constrained (finite sigma
//     > 0 on x, y AND z) and their centres are not collinear, they pin all 7
//     similarity DoF, so every pose is left free and no scale anchor is set.
//     Horizontal-only anchors (for example a reference without altitude) do
//     not count: they leave vertical translation free. Otherwise the first
//     aligned camera is held fully constant and the scale is anchored on the
//     aligned camera with the largest baseline to it (choose_scale_anchor_axis).
//
//   * Markers: every regular marker with >= 2 enabled projections in distinct
//     aligned cameras becomes a BA point whose observations are its projections
//     (one per camera). Vertex and fiducial markers are image-space target
//     marks, not ground points, and are skipped. A referenced marker becomes a
//     control point (or a check point, held out of the solve) when the chunk is
//     georeferenced. Scale bars between marker points enter as BAScalebar
//     constraints and fix scale even in a non-georeferenced chunk.
//
// Everything is computed into a staged copy and committed with one move, so an
// exception anywhere leaves the chunk untouched.

#include "common/algorithm/sfm/optimize_cameras_task.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/feature/detect_features_task.hpp"  // keypoint_to_pixel
#include "common/algorithm/reference/convert_reference.hpp"
#include "common/algorithm/sfm/adaptive_fit.hpp"
#include "common/algorithm/sfm/bundle_adjustment.hpp"
#include "common/algorithm/sfm/bundle_adjustment_detail.hpp"
#include "common/algorithm/sfm/triangulation.hpp"
#include "common/io/camera_image.hpp"
#include "model/camera/antenna.hpp"
#include "model/camera/calibration.hpp"
#include "model/crs/euler.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape {
namespace {

/// Calibration coefficient names are shared with the covariance row labels;
/// the packed order is f, cx, cy, b1, b2, k1, k2, k3, k4, p1, p2, p3, p4.

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

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;

/// Rz(angle) in the ENU / body axes.
Eigen::Matrix3d rotation_about_z(double degrees) {
    const double angle = degrees * kDegToRad;
    Eigen::Matrix3d m;
    m << std::cos(angle), -std::sin(angle), 0.0,  //
        std::sin(angle), std::cos(angle), 0.0,     //
        0.0, 0.0, 1.0;
    return m;
}

/// Rotation carried by a 3x3 block, with a uniform similarity scale divided
/// out (the geometric counterpart of georeference's `pure_rotation`). The
/// chunk transform and an AlignChunks pose may both carry a scale.
Eigen::Matrix3d rotation_of_similarity(const Eigen::Matrix3d& m) {
    const double determinant = m.determinant();
    if (!std::isfinite(determinant) || !(determinant > 0.0)) {
        throw ValueError("OptimizeCameras: chunk transform carries no valid rotation");
    }
    return m / std::cbrt(determinant);
}

/// Body-to-ENU rotation for a stored orientation triple, in `order`:
/// `R_wb = Rz(90 deg) * to_matrix({x, -y, z})`. The Rz(90) turns the
/// north-pointing zero-heading body frame into the ENU basis; the second
/// component is negated because a positive stored pitch is nose up while
/// `to_matrix` applies a positive rotation about its second axis (see the
/// `convert_reference.hpp` contract).
Eigen::Matrix3d world_from_body(const Eigen::Vector3d& angles, crs::RotationOrder order) {
    const crs::EulerAngles internal{angles.x(), -angles.y(), angles.z(), order};
    return rotation_about_z(90.0) * crs::to_matrix(internal);
}

/// Full-precision text for a metadata value.
std::string meta_number(double value) {
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

/// Rotation matrix of a rotation vector (angle-axis), matching the
/// parametrisation bundle_adjust uses for the 3-double rotation block.
Eigen::Matrix3d rotation_from_rotation_vector(const Eigen::Vector3d& vector) {
    const double angle = vector.norm();
    if (!(angle > 0.0)) {
        return Eigen::Matrix3d::Identity();
    }
    return Eigen::AngleAxisd(angle, vector / angle).toRotationMatrix();
}

/// Location covariance of the camera centre C = -R^T t given the 6x6 covariance
/// of (rotation vector, translation): J Sigma J^T with dC/dt = -R^T and dC/dw
/// by central differences on the rotation vector. Chunk-local, like the pose.
Eigen::Matrix3d camera_location_covariance(const sfm::BAPose& pose,
                                           const Eigen::Matrix<double, 6, 6>& covariance) {
    Eigen::Matrix<double, 3, 6> jacobian = Eigen::Matrix<double, 3, 6>::Zero();
    jacobian.block<3, 3>(0, 3) = -pose.R.transpose();
    const Eigen::AngleAxisd angle_axis(pose.R);
    const Eigen::Vector3d omega = angle_axis.angle() * angle_axis.axis();
    constexpr double kStep = 1e-7;
    for (int k = 0; k < 3; ++k) {
        Eigen::Vector3d plus = omega;
        Eigen::Vector3d minus = omega;
        plus[k] += kStep;
        minus[k] -= kStep;
        const Eigen::Vector3d centre_plus =
            -rotation_from_rotation_vector(plus).transpose() * pose.t;
        const Eigen::Vector3d centre_minus =
            -rotation_from_rotation_vector(minus).transpose() * pose.t;
        jacobian.col(k) = (centre_plus - centre_minus) / (2.0 * kStep);
    }
    return jacobian * covariance * jacobian.transpose();
}

}  // namespace

void OptimizeCamerasTask::declare_params(Params& params) const {
    // Defaults fit the principal point and the low-order radial/tangential
    // terms; the affine pair and the highest radial term stay fixed.
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
    params.declare(
        {"fit_p3", ParamType::Bool, false, "Adjust the higher-order tangential distortion p3."});
    params.declare(
        {"fit_p4", ParamType::Bool, false, "Adjust the higher-order tangential distortion p4."});
    // Adaptive mode treats the fit_* flags above as permission rather than
    // instruction: it frees only the base coefficients plus the higher-order
    // groups whose likelihood-ratio gain pays for their degrees of freedom.
    params.declare({"adaptive_fitting", ParamType::Bool, false,
                    "Select the distortion terms automatically (ordered forward selection)."});
    // Posterior covariance of the cameras and calibration is on by default
    // (subject to the BA caps); tie-point covariance is opt-in because it is
    // O(points) dense output a caller rarely needs.
    params.declare({"camera_covariance", ParamType::Bool, true,
                    "Store the posterior covariance of the cameras and calibration."});
    params.declare({"tiepoint_covariance", ParamType::Bool, false,
                    "Store the posterior covariance of every tie point."});
    // Measured attitudes (a gimbal log or an INS) enter as soft rotation
    // priors when the chunk is georeferenced. The heading axis is weighted by
    // its own sigma, which is typically much looser than pitch and roll.
    params.declare({"use_orientation", ParamType::Bool, true,
                    "Use measured camera attitudes as rotation priors."});
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

    const bool use_orientation = params.get_bool("use_orientation");

    // The fit flags, in the packed coefficient order.
    std::array<bool, kCalibrationParamCount> fit = {};
    for (int k = 0; k < kCalibrationParamCount; ++k) {
        fit[static_cast<std::size_t>(k)] =
            params.get_bool("fit_" + std::string(kCalibrationParamNames[static_cast<std::size_t>(k)]));
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

    // Master links (WP-13.16). A slave resolves to its master, so both share
    // one BA calibration slot and the free mask comes from the master's
    // fixed_params. An invalid link is a hard error before any work is done.
    validate_sensor_masters(chunk->sensors());

    // One calibration per EFFECTIVE sensor used by an aligned camera.
    std::unordered_map<std::int32_t, std::size_t> sensor_slot;
    std::vector<std::int32_t> slot_sensor;  // BA calibration -> chunk sensor id
    std::vector<std::int32_t> used_sensor;  // sensors named by an aligned camera
    for (const std::size_t ci : camera_index) {
        const Camera& camera = chunk->cameras()[ci];
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk->sensors().size())) {
            throw ValueError("OptimizeCameras: camera " + std::to_string(camera.id) +
                             " has no sensor");
        }
        const Sensor& sensor =
            chunk->sensors()[static_cast<std::size_t>(camera.sensor_id)];
        const std::int32_t effective = sensor.master_sensor_id >= 0
                                           ? sensor.master_sensor_id
                                           : camera.sensor_id;
        used_sensor.push_back(camera.sensor_id);
        if (sensor_slot.find(effective) == sensor_slot.end()) {
            const std::size_t slot = problem.calibrations.size();
            sensor_slot[effective] = slot;
            slot_sensor.push_back(effective);
            problem.calibrations.push_back(sensor_pipeline_calibration(
                chunk->sensors()[static_cast<std::size_t>(effective)]));
        }
        problem.camera_sensor.push_back(sensor_slot.at(effective));
        problem.poses.push_back(pose_from_transform(camera.pose));
    }

    // Free mask per sensor: fit_X && X not in the effective sensor's
    // fixed_params (the master's, for a slave).
    problem.fit_intrinsics.resize(problem.calibrations.size());
    for (std::size_t s = 0; s < problem.calibrations.size(); ++s) {
        const Sensor& sensor =
            chunk->sensors()[static_cast<std::size_t>(slot_sensor[s])];
        std::array<bool, kCalibrationParamCount> mask = {};
        for (int k = 0; k < kCalibrationParamCount; ++k) {
            const std::size_t kk = static_cast<std::size_t>(k);
            const bool fixed = std::find(sensor.fixed_params.begin(), sensor.fixed_params.end(),
                                         std::string(kCalibrationParamNames[kk])) != sensor.fixed_params.end();
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
    const Transform& transform = chunk->transform();
    const bool georeferenced =
        (transform - Transform::Identity()).cwiseAbs().maxCoeff() > 1e-9;
    // Local->ECEF scale. Chunk-local units are metres only when this is 1; every
    // metre-valued sigma and length below is divided by it, and every reported
    // metre-valued RMS is multiplied by it.
    const double local_scale =
        std::cbrt(std::abs(transform.block<3, 3>(0, 0).determinant()));
    if (!std::isfinite(local_scale) || local_scale <= 0.0) {
        throw ValueError("OptimizeCameras: chunk transform has no usable scale");
    }
    // -- marker points and their image observations -------------------------
    // A marker with at least two enabled projections in aligned cameras becomes
    // one BA point. Its projections are stored in the Calibration pixel
    // convention (top-left pixel centre (0.5, 0.5)), the same convention as
    // BAObservation, so they are copied without the keypoint +0.5 shift.
    // BAObservation carries no per-observation sigma, so a
    // marker_projection_accuracy parameter is deferred until it does.
    constexpr std::size_t kNoPoint = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> marker_point(chunk->markers().size(), kNoPoint);
    std::size_t markers_skipped = 0;
    std::size_t markers_non_regular = 0;
    for (std::size_t m = 0; m < chunk->markers().size(); ++m) {
        const Marker& marker = chunk->markers()[m];
        // Vertex and fiducial markers are image-space target marks, not ground
        // points: they never become BA points.
        if (marker.type != MarkerType::Regular) {
            ++markers_non_regular;
            ++markers_skipped;
            continue;
        }
        std::vector<std::pair<std::size_t, Eigen::Vector2d>> projections;
        // One observation per camera: a duplicated projection must not
        // double-weight the residual. Keep the first enabled entry.
        std::unordered_map<std::int32_t, bool> seen_camera;
        for (const MarkerProjection& projection : marker.projections) {
            if (!projection.enabled || !projection.pixel.allFinite()) {
                continue;
            }
            const auto camera = camera_slot.find(projection.camera_id);
            if (camera == camera_slot.end()) {
                continue;  // not aligned: not part of this refinement
            }
            if (!seen_camera.emplace(projection.camera_id, true).second) {
                continue;  // duplicate observation of the same camera
            }
            projections.emplace_back(camera->second, projection.pixel);
        }
        if (projections.size() < 2) {
            ++markers_skipped;
            continue;
        }
        Eigen::Vector3d point;
        // An existing position is only a usable seed when it lies in front of
        // every camera that observes the marker; a stale position (for example
        // from a previous alignment) would start the solve behind a camera.
        bool usable_seed = false;
        if (marker.position.has_value() && marker.position->allFinite()) {
            usable_seed = true;
            for (const std::pair<std::size_t, Eigen::Vector2d>& projection : projections) {
                const Camera& camera = chunk->cameras()[camera_index[projection.first]];
                const Eigen::Vector3d camera_point =
                    camera.pose.block<3, 3>(0, 0) * *marker.position +
                    camera.pose.block<3, 1>(0, 3);
                if (!(camera_point.z() > 0.0)) {
                    usable_seed = false;
                    break;
                }
            }
        }
        if (usable_seed) {
            point = *marker.position;
        } else {
            // Linear multi-view triangulation from the current poses. The
            // 0.5 deg parallax floor matches the marker-refinement task.
            std::vector<sfm::AbsolutePose> poses;
            std::vector<Eigen::Vector3d> bearings;
            poses.reserve(projections.size());
            bearings.reserve(projections.size());
            bool usable = true;
            for (const auto& [slot, pixel] : projections) {
                const Camera& camera = chunk->cameras()[camera_index[slot]];
                sfm::AbsolutePose pose;
                pose.R = camera.pose.block<3, 3>(0, 0);
                pose.t = camera.pose.block<3, 1>(0, 3);
                poses.push_back(pose);
                const Eigen::Vector3d bearing =
                    chunk->sensors()[static_cast<std::size_t>(camera.sensor_id)]
                        .calibration.unproject(pixel);
                if (!bearing.allFinite() || bearing.norm() <= 0.0) {
                    usable = false;
                    break;
                }
                bearings.push_back(bearing.normalized());
            }
            sfm::TriangulationOptions options;
            options.min_parallax_rad = 0.5 * 3.14159265358979323846 / 180.0;
            const auto estimate =
                usable ? sfm::triangulate_multiview(poses, bearings, options)
                       : std::optional<Eigen::Vector3d>{};
            if (!estimate || !estimate->allFinite()) {
                ++markers_skipped;
                continue;
            }
            point = *estimate;
        }
        const std::size_t slot = problem.points.size();
        problem.points.push_back(point);
        marker_point[m] = slot;
        for (const auto& [camera, pixel] : projections) {
            problem.observations.push_back(sfm::BAObservation{camera, slot, pixel});
        }
    }
    // -- control and check points -------------------------------------------
    // Only a georeferenced chunk can tie a marker reference to the chunk-local
    // frame. A referenced marker is control unless its meta role is
    // "check_point"; a point without a reference stays a free point.
    std::size_t control_markers = 0;
    std::size_t checkpoint_markers = 0;
    if (georeferenced) {
        const Transform inverse = transform.inverse();
        for (std::size_t m = 0; m < chunk->markers().size(); ++m) {
            const Marker& marker = chunk->markers()[m];
            if (!marker.reference.has_value() || marker_point[m] == kNoPoint) {
                continue;
            }
            const Eigen::Vector3d& loc = *marker.reference;
            if (!std::isfinite(loc.x()) || !std::isfinite(loc.y())) {
                continue;  // no usable horizontal reference
            }
            const bool has_alt = std::isfinite(loc.z());
            const geo::Geodetic geodetic{loc.x(), loc.y(), has_alt ? loc.z() : 0.0};
            const Eigen::Vector3d ecef = geo::geodetic_to_ecef(geodetic);
            sfm::BAControlPoint control;
            control.point = marker_point[m];
            control.reference =
                (inverse.block<3, 3>(0, 0) * ecef) + inverse.block<3, 1>(0, 3);
            // Per-axis accuracy applied along the local axes, exactly as the
            // camera priors do; metres -> chunk-local units.
            control.sigma = marker.reference_accuracy / local_scale;
            for (int axis = 0; axis < 3; ++axis) {
                const double sigma = control.sigma[axis];
                if (!std::isfinite(sigma) || sigma <= 0.0) {
                    control.sigma[axis] = 0.0;  // unconstrained axis
                }
            }
            if (!has_alt) {
                control.sigma.z() = 0.0;  // no altitude: leave the up axis free
            }
            const std::string* role = marker.meta.find("type");
            control.checkpoint = role != nullptr && *role == "check_point";
            if (control.checkpoint) {
                ++checkpoint_markers;
            } else {
                ++control_markers;
            }
            problem.control_points.push_back(control);
        }
    } else {
        std::size_t ungeoreferenced = 0;
        for (std::size_t m = 0; m < chunk->markers().size(); ++m) {
            if (chunk->markers()[m].reference.has_value() && marker_point[m] != kNoPoint) {
                ++ungeoreferenced;
            }
        }
        if (ungeoreferenced > 0) {
            log(LogLevel::Info)
                << "OptimizeCameras: markers have references but the chunk is not "
                   "georeferenced; run georeferencing first";
        }
    }
    // -- scale bars ---------------------------------------------------------
    // Only marker-to-marker bars with two BA points and a positive measured
    // length participate. Camera endpoints are a later item. A bar with a
    // check-point endpoint validates only and contributes no constraint.
    struct ScalebarEntry {
        std::size_t index = 0;     ///< index into chunk->scalebars()
        std::size_t marker_a = 0;  ///< index into chunk->markers()
        std::size_t marker_b = 0;
    };
    std::unordered_map<std::int32_t, std::size_t> marker_by_id;
    for (std::size_t m = 0; m < chunk->markers().size(); ++m) {
        marker_by_id[chunk->markers()[m].id] = m;
    }
    std::vector<ScalebarEntry> scalebar_entry;
    std::size_t camera_scalebars = 0;
    std::size_t scalebars_skipped = 0;
    for (std::size_t s = 0; s < chunk->scalebars().size(); ++s) {
        const Scalebar& bar = chunk->scalebars()[s];
        if (bar.from.kind != ScalebarEndpoint::Kind::Marker ||
            bar.to.kind != ScalebarEndpoint::Kind::Marker) {
            ++camera_scalebars;  // camera endpoints: not wired yet
            continue;
        }
        const auto from = marker_by_id.find(bar.from.id);
        const auto to = marker_by_id.find(bar.to.id);
        if (from == marker_by_id.end() || to == marker_by_id.end() ||
            marker_point[from->second] == kNoPoint || marker_point[to->second] == kNoPoint) {
            ++scalebars_skipped;
            continue;
        }
        const std::size_t point_a = marker_point[from->second];
        const std::size_t point_b = marker_point[to->second];
        // A bar whose endpoints resolve to the same point carries no direction
        // and must not fix the scale. A bar without a usable measured length or
        // accuracy cannot constrain anything either. Each counts once.
        if (point_a == point_b) {
            ++scalebars_skipped;
            continue;
        }
        if (!std::isfinite(bar.reference_length_m) || !(bar.reference_length_m > 0.0)) {
            ++scalebars_skipped;
            continue;
        }
        if (!std::isfinite(bar.reference_accuracy_m) ||
            !(bar.reference_accuracy_m > 0.0)) {
            ++scalebars_skipped;
            continue;
        }
        const std::string* role_a = chunk->markers()[from->second].meta.find("type");
        const std::string* role_b = chunk->markers()[to->second].meta.find("type");
        sfm::BAScalebar out;
        out.point_a = point_a;
        out.point_b = point_b;
        out.length = bar.reference_length_m / local_scale;  // metres -> local units
        out.sigma = bar.reference_accuracy_m / local_scale;
        out.checkpoint = (role_a != nullptr && *role_a == "check_point") ||
                         (role_b != nullptr && *role_b == "check_point");
        problem.scalebars.push_back(out);
        scalebar_entry.push_back({s, from->second, to->second});
    }
    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "OptimizeCameras: "
                            << (chunk->markers().size() - markers_skipped) << " marker points, "
                            << markers_skipped
                            << " markers skipped (non-regular, fewer than 2 projections, or "
                               "no triangulation; "
                            << markers_non_regular << " non-regular), "
                            << control_markers << " control points, " << checkpoint_markers
                            << " check points, " << problem.scalebars.size() << " scale bars ("
                            << camera_scalebars << " with a camera endpoint skipped, "
                            << scalebars_skipped << " skipped)";
    }
    progress.update("prepare", 1.0);

    // -- GNSS centre and attitude priors (georeferenced chunks only) ---------
    std::size_t rotation_prior_count = 0;
    if (georeferenced) {
        const Transform inverse = transform.inverse();
        const Eigen::Matrix3d ecef_from_internal =
            rotation_of_similarity(transform.block<3, 3>(0, 0));
        const crs::RotationOrder order = reference::rotation_order_of(*chunk);
        const double default_rotation_accuracy =
            chunk->reference_settings().camera_rotation_accuracy_deg;
        std::unordered_set<std::int32_t> logged_antenna_sensors;
        for (std::size_t b = 0; b < camera_index.size(); ++b) {
            const Camera& camera = chunk->cameras()[camera_index[b]];
            if (!camera.reference.enabled) {
                continue;
            }
            // -- attitude prior ---------------------------------------------
            if (use_orientation && camera.reference.orientation_deg.has_value()) {
                // The ENU frame is taken at the reference position; without
                // one, at the camera's current estimated centre converted to
                // geodetic. The difference only changes the local meridian
                // convergence over the block.
                geo::Geodetic geodetic;
                if (camera.reference.location.has_value()) {
                    const Eigen::Vector3d& loc = *camera.reference.location;
                    geodetic = geo::Geodetic{loc.x(), loc.y(),
                                             std::isfinite(loc.z()) ? loc.z() : 0.0};
                } else {
                    const sfm::BAPose& pose = problem.poses[b];
                    const Eigen::Vector3d centre = -(pose.R.inverse() * pose.t);
                    geodetic = geo::ecef_to_geodetic(
                        transform.block<3, 3>(0, 0) * centre + transform.block<3, 1>(0, 3));
                }
                const Sensor& sensor =
                    chunk->sensors()[static_cast<std::size_t>(camera.sensor_id)];
                const std::int32_t effective =
                    sensor.master_sensor_id >= 0 ? sensor.master_sensor_id : camera.sensor_id;
                const Antenna& antenna =
                    chunk->sensors()[static_cast<std::size_t>(effective)].antenna;
                const Eigen::Matrix3d ecef_from_enu =
                    geo::enu_to_ecef(geodetic).block<3, 3>(0, 0);
                const Eigen::Matrix3d r_wb =
                    world_from_body(*camera.reference.orientation_deg, order);
                // The stored triple is a BODY attitude (an aircraft or a gimbal
                // log composed through the fixed mount). The camera is mounted
                // by the nadir body-to-camera rotation unless the effective
                // master sensor's antenna carries a boresight, which replaces
                // that mount (WP-13.19). Camera-to-world is R_wb * B^T, so the
                // world->camera measurement below is B * R_wb^T.
                const Eigen::Matrix3d r_bc =
                    antenna.boresight_deg.has_value() ? antenna.body_to_camera()
                                                      : reference::body_to_camera_nadir();
                bool degenerate_heading = false;
                const Eigen::Matrix3d frame_enu =
                    sfm::detail::heading_frame_from_body(r_wb, &degenerate_heading);
                Eigen::Vector3d sigma_deg =
                    camera.reference.rotation_accuracy_deg.value_or(
                        Eigen::Vector3d::Constant(default_rotation_accuracy));
                if (degenerate_heading) {
                    // Body forward is vertical, so yaw and roll about the
                    // heading frame are not separable: give both the looser
                    // of their two sigmas.
                    const double shared = std::max(sigma_deg.x(), sigma_deg.z());
                    sigma_deg.x() = shared;
                    sigma_deg.z() = shared;
                }
                sfm::BARotationPrior prior;
                prior.camera = b;
                prior.measured = r_bc * r_wb.transpose() * ecef_from_enu.transpose() *
                                 ecef_from_internal;
                prior.heading_frame =
                    ecef_from_internal.transpose() * ecef_from_enu * frame_enu;
                prior.sigma_rad = sigma_deg * kDegToRad;
                problem.rotation_priors.push_back(prior);
                ++rotation_prior_count;
            }
            if (!camera.reference.location.has_value()) {
                continue;
            }
            const Eigen::Vector3d& loc = *camera.reference.location;
            const bool has_alt = std::isfinite(loc.z());
            const geo::Geodetic geodetic{loc.x(), loc.y(), has_alt ? loc.z() : 0.0};
            const Eigen::Vector3d ecef = geo::geodetic_to_ecef(geodetic);
            sfm::BACentrePrior prior;
            prior.camera = b;
            prior.centre = (inverse.block<3, 3>(0, 0) * ecef) + inverse.block<3, 1>(0, 3);
            prior.sigma = camera.reference.accuracy / local_scale;
            if (!has_alt) {
                prior.sigma.z() = 0.0;  // no altitude: leave the up axis unconstrained
            }
            // The GNSS reference is the antenna phase centre, so the prior
            // constrains the mounted antenna rather than the projection centre.
            // Sensors resolve their master's calibration, and the mount is part
            // of the physical body, so use the effective master's antenna.
            const Sensor& sensor =
                chunk->sensors()[static_cast<std::size_t>(camera.sensor_id)];
            const std::int32_t effective =
                sensor.master_sensor_id >= 0 ? sensor.master_sensor_id : camera.sensor_id;
            const Antenna& antenna =
                chunk->sensors()[static_cast<std::size_t>(effective)].antenna;
            if (antenna.is_set() || antenna.boresight_deg.has_value()) {
                prior.lever_arm = antenna.camera_offset() / local_scale;
                if (logged_antenna_sensors.insert(effective).second &&
                    Log::enabled(LogLevel::Info)) {
                    log(LogLevel::Info) << "OptimizeCameras: sensor " << effective
                                        << " antenna offset [" << prior.lever_arm.x()
                                        << ", " << prior.lever_arm.y() << ", "
                                        << prior.lever_arm.z()
                                        << "] local units (camera frame) "
                                        << "= " << antenna.camera_offset().x() << ", "
                                        << antenna.camera_offset().y() << ", "
                                        << antenna.camera_offset().z() << " m";
                }
            }
            problem.centre_priors.push_back(prior);
        }
        if (rotation_prior_count > 0 && Log::enabled(LogLevel::Info)) {
            log(LogLevel::Info) << "OptimizeCameras: " << rotation_prior_count
                                << " rotation priors from measured attitudes";
        }
    }

    // -- gauge --------------------------------------------------------------
    // An anchor can replace the fixed-camera gauge only when it constrains all
    // three axes: three horizontal-only anchors leave vertical translation a
    // null direction. Collect the centres of fully constrained priors; three
    // non-collinear ones pin the whole similarity, otherwise fall back to
    // fixing a camera and anchoring the scale.
    std::vector<Eigen::Vector3d> prior_centres;
    for (const sfm::BACentrePrior& prior : problem.centre_priors) {
        const bool east = std::isfinite(prior.sigma.x()) && prior.sigma.x() > 0.0;
        const bool north = std::isfinite(prior.sigma.y()) && prior.sigma.y() > 0.0;
        const bool up = std::isfinite(prior.sigma.z()) && prior.sigma.z() > 0.0;
        if (east && north && up) {
            prior_centres.push_back(prior.centre);
        }
    }

    // Non-check control points with all three axes constrained count as gauge
    // anchors alongside the camera priors. Their references are already in
    // the same chunk-local frame.
    for (const sfm::BAControlPoint& control : problem.control_points) {
        if (control.checkpoint) {
            continue;
        }
        const bool east = std::isfinite(control.sigma.x()) && control.sigma.x() > 0.0;
        const bool north = std::isfinite(control.sigma.y()) && control.sigma.y() > 0.0;
        const bool up = std::isfinite(control.sigma.z()) && control.sigma.z() > 0.0;
        if (east && north && up) {
            prior_centres.push_back(control.reference);
        }
    }

    problem.fixed_pose.assign(problem.poses.size(), 0);
    if (prior_centres.size() >= 3 && centres_not_collinear(prior_centres)) {
        // The priors pin rotation, translation and scale: leave every pose free.
        problem.scale_anchor_camera = -1;
    } else {
        problem.fixed_pose[0] = 1;
        // A real (non-check) scale bar fixes the scale itself, so pinning a
        // translation component as well would over-constrain the gauge. Every
        // bar still in problem.scalebars has already passed the degenerate,
        // length and accuracy filters.
        bool scalebar_fixes_scale = false;
        for (const sfm::BAScalebar& bar : problem.scalebars) {
            if (!bar.checkpoint) {
                scalebar_fixes_scale = true;
                break;
            }
        }
        if (scalebar_fixes_scale) {
            problem.scale_anchor_camera = -1;
        } else {
            // Anchor the scale on the aligned camera with the largest baseline
            // to camera 0 (its centre is farthest).
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
    }

    // -- solve --------------------------------------------------------------
    progress.update("optimize", 0.0);
    const bool adaptive_fitting = params.get_bool("adaptive_fitting");
    const bool camera_covariance = params.get_bool("camera_covariance");
    sfm::BAOptions ba_options;
    ba_options.compute_camera_covariance = camera_covariance;
    ba_options.compute_calibration_covariance = camera_covariance;
    ba_options.compute_point_covariance = params.get_bool("tiepoint_covariance");
    sfm::BAReport report;
    std::string adaptive_selected;
    int adaptive_solves = 0;
    if (adaptive_fitting) {
        // The fit flags are the allowed set; adaptive_fit narrows them further.
        sfm::AdaptiveFitOptions adaptive_options;
        adaptive_options.solve_options = ba_options;
        const sfm::AdaptiveFitResult adaptive =
            sfm::adaptive_fit(problem, fit, adaptive_options);
        report = adaptive.report;
        adaptive_solves = adaptive.solves;
        for (int k = 0; k < kCalibrationParamCount; ++k) {
            const std::size_t kk = static_cast<std::size_t>(k);
            if (!adaptive.selected[kk]) {
                continue;
            }
            if (!adaptive_selected.empty()) {
                adaptive_selected += ",";
            }
            adaptive_selected += kCalibrationParamNames[kk];
        }
    } else {
        report = sfm::bundle_adjust(problem, ba_options);
    }
    if (!report.usable) {
        throw Error("OptimizeCameras: bundle adjustment did not produce a usable solution");
    }
    progress.update("optimize", 1.0);

    // -- stage the result on a copy -----------------------------------------
    Chunk staged = *chunk;
    for (std::size_t s = 0; s < problem.calibrations.size(); ++s) {
        Calibration& calibration = staged.sensors()[static_cast<std::size_t>(slot_sensor[s])].calibration;
        calibration = problem.calibrations[s];
        calibration.covariance.resize(0, 0);
        calibration.covariance_params.clear();
        if (s < report.calibration_covariance.size() &&
            report.calibration_covariance[s].rows() == kCalibrationParamCount) {
            calibration.covariance = report.calibration_covariance[s];
            for (const char* name : kCalibrationParamNames) {
                calibration.covariance_params.emplace_back(name);
            }
        }
    }
    // Every slave used by an aligned camera receives a copy of its master's
    // solved calibration (and covariance).
    for (const std::int32_t sensor_id : used_sensor) {
        const std::int32_t master =
            chunk->sensors()[static_cast<std::size_t>(sensor_id)].master_sensor_id;
        if (master >= 0) {
            staged.sensors()[static_cast<std::size_t>(sensor_id)].calibration =
                staged.sensors()[static_cast<std::size_t>(master)].calibration;
        }
    }
    for (std::size_t b = 0; b < camera_index.size(); ++b) {
        Camera& camera = staged.cameras()[camera_index[b]];
        camera.pose = transform_from_pose(problem.poses[b]);
        camera.location_covariance.setZero();
        camera.rotation_covariance.setZero();
        if (b < report.camera_covariance.size()) {
            camera.location_covariance =
                camera_location_covariance(problem.poses[b], report.camera_covariance[b]);
            camera.rotation_covariance = report.camera_covariance[b].block<3, 3>(0, 0);
        }
    }
    for (std::size_t t = 0; t < tie_points.tracks.size(); ++t) {
        const auto point = point_slot.find(t);
        if (point == point_slot.end()) {
            continue;
        }
        Track& track = staged.tie_points().tracks[t];
        track.point = problem.points[point->second];
        track.covariance.reset();
        if (point->second < report.point_covariance.size()) {
            track.covariance = report.point_covariance[point->second].cast<float>();
        }
    }

    // -- markers and scale bars ---------------------------------------------
    // A check point is re-triangulated after the solve; checkpoint_residuals is
    // X - reference, so its position is reference + residual. Every other
    // marker takes its solved point.
    std::unordered_map<std::size_t, Eigen::Vector3d> rechecked_position;
    std::size_t check_index = 0;
    for (const sfm::BAControlPoint& control : problem.control_points) {
        if (!control.checkpoint) {
            continue;
        }
        if (check_index < report.checkpoint_residuals.size()) {
            const Eigen::Vector3d& residual = report.checkpoint_residuals[check_index];
            if (residual.allFinite()) {
                rechecked_position[control.point] = control.reference + residual;
            }
        }
        ++check_index;
    }
    std::size_t marker_points = 0;
    for (std::size_t m = 0; m < marker_point.size(); ++m) {
        if (marker_point[m] == kNoPoint) {
            continue;
        }
        ++marker_points;
        const auto checked = rechecked_position.find(marker_point[m]);
        staged.markers()[m].position = checked != rechecked_position.end()
                                           ? checked->second
                                           : problem.points[marker_point[m]];
        // Only replace a stored covariance when this run actually produced a
        // valid block for the point. A dropped point (or a run with tie-point
        // covariance off) leaves the previously stored value untouched.
        if (report.covariance_valid &&
            marker_point[m] < report.point_covariance.size() &&
            !report.point_covariance[marker_point[m]].isZero(0.0)) {
            staged.markers()[m].position_covariance =
                report.point_covariance[marker_point[m]];
        }
    }
    for (const ScalebarEntry& entry : scalebar_entry) {
        const auto& a = staged.markers()[entry.marker_a].position;
        const auto& b = staged.markers()[entry.marker_b].position;
        if (a.has_value() && b.has_value()) {
            staged.scalebars()[entry.index].estimated_length_m =
                (*a - *b).norm() * local_scale;
        }
    }

    // -- attitude residuals -------------------------------------------------
    // Unweighted (yaw, pitch, roll) delta_h of every rotation prior, in
    // degrees, plus their aggregate RMS over all components. Any axis held
    // unconstrained still reports its measured error.
    double attitude_sq = 0.0;
    std::size_t attitude_n = 0;
    for (const sfm::BARotationPrior& prior : problem.rotation_priors) {
        const Eigen::Vector3d error =
            sfm::predicted_rotation_residual(problem.poses[prior.camera], prior) *
            kRadToDeg;
        const Camera& camera = staged.cameras()[camera_index[prior.camera]];
        const std::string label =
            camera.label.empty() ? std::to_string(camera.id) : camera.label;
        staged.meta().set("optimize_cameras." + label + ".attitude_error_deg",
                          meta_number(error.x()) + "," + meta_number(error.y()) + "," +
                              meta_number(error.z()));
        attitude_sq += error.squaredNorm();
        attitude_n += 3U;
    }
    const double attitude_rms_deg =
        attitude_n == 0 ? 0.0 : std::sqrt(attitude_sq / static_cast<double>(attitude_n));
    // Written only when there is a rotation prior, so a project without
    // measured attitudes keeps exactly the metadata it had before WP-13.21.
    if (!problem.rotation_priors.empty()) {
        staged.meta().set("optimize_cameras.attitude_rms_deg", meta_number(attitude_rms_deg));
    }

    // -- report statistics (metres) -----------------------------------------
    // The solver reports in chunk-local units; multiply by the local->ECEF
    // scale so the stored and logged values are metres.
    const double control_rms_m = report.control_rms_m * local_scale;
    const double checkpoint_rms_m = report.checkpoint_rms_m * local_scale;
    const double scalebar_rms_m = report.scalebar_rms_m * local_scale;
    const double checkbar_rms_m = report.checkbar_rms_m * local_scale;
    staged.meta().set("optimize_cameras.control_rms_m", meta_number(control_rms_m));
    staged.meta().set("optimize_cameras.checkpoint_rms_m", meta_number(checkpoint_rms_m));
    staged.meta().set("optimize_cameras.scalebar_rms_m", meta_number(scalebar_rms_m));
    staged.meta().set("optimize_cameras.checkbar_rms_m", meta_number(checkbar_rms_m));
    staged.meta().set("optimize_cameras.control_points",
                      std::to_string(report.control_points));
    staged.meta().set("optimize_cameras.checkpoints", std::to_string(report.checkpoints));
    staged.meta().set("optimize_cameras.scalebars", std::to_string(report.scalebars));
    staged.meta().set("optimize_cameras.checkbars", std::to_string(report.checkbars));
    staged.meta().set("optimize_cameras.markers", std::to_string(marker_points));
    staged.meta().set("optimize_cameras.markers_skipped",
                      std::to_string(chunk->markers().size() - marker_points));
    staged.meta().set("optimize_cameras.markers_non_regular",
                      std::to_string(markers_non_regular));
    staged.meta().set("optimize_cameras.scalebars_skipped",
                      std::to_string(scalebars_skipped));
    // Always written, even when adaptive fitting is off, so a previous run's
    // selection cannot survive on the chunk.
    staged.meta().set("optimize_cameras.adaptive_selected", adaptive_selected);
    staged.meta().set("optimize_cameras.adaptive_solves",
                      std::to_string(adaptive_solves));
    staged.meta().set("optimize_cameras.sigma0", meta_number(std::sqrt(report.sigma0_squared)));
    staged.meta().set("optimize_cameras.covariance",
                      report.covariance_reason.empty()
                          ? "covariance skipped: not requested"
                          : report.covariance_reason);

    // -- commit atomically --------------------------------------------------
    *chunk = std::move(staged);

    if (Log::enabled(LogLevel::Info)) {
        // The attitude clause appears only when there is a rotation prior, so
        // orientation-free projects log the same line as before WP-13.21.
        std::ostringstream message;
        message << "OptimizeCameras: RMS " << report.initial_rms_px << " -> "
                << report.final_rms_px << " px (" << report.observations
                << " observations, " << report.iterations << " iterations, prior RMS "
                << report.prior_rms_sigma << " sigma, " << report.priors
                << " prior axes; control RMS " << control_rms_m << " m over "
                << report.control_points << ", check-point RMS " << checkpoint_rms_m
                << " m over " << report.checkpoints << ", scale-bar RMS " << scalebar_rms_m
                << " m over " << report.scalebars << ", check-bar RMS " << checkbar_rms_m
                << " m over " << report.checkbars;
        if (!problem.rotation_priors.empty()) {
            message << ", attitude RMS " << attitude_rms_deg << " deg over "
                    << problem.rotation_priors.size() << " cameras";
        }
        message << ")";
        log(LogLevel::Info) << message.str();
    }
}

}  // namespace alphashape
