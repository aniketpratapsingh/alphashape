// Milestone 4: bundle adjustment (see bundle_adjustment.hpp for the contract).

#include "common/algorithm/sfm/bundle_adjustment.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

// Ceres (and its Eigen/glog/suite-sparse headers) is pulled in here only — never
// in the public header — and its headers are not written to the engine's strict
// warning set. Wrap them so imported-target system-ness is not relied upon.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wpedantic"
#elif defined(__GNUC__)
// g++ needs the same third-party suppressions; its installed Ceres headers are
// not system-included here, so the warnings would otherwise reach -Werror.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wcast-align"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "base/utils/error.hpp"

namespace alphashape::sfm {
namespace {

/// Error charged to an observation whose point falls behind a Frame camera.
/// Kept large so such residuals can never be mistaken for an inlier, and bold
/// enough that any solution leaving a point behind the camera is rejected.
constexpr double kBehindCameraErrorPx = 1.0e6;

/// Pack a Calibration into the kCalibrationParamCount doubles project_point<T>
/// expects: f, cx, cy, b1, b2, k1..k4, p1..p4.
void pack_calibration(const Calibration& c, double* params) {
    params[0] = c.f;
    params[1] = c.cx;
    params[2] = c.cy;
    params[3] = c.b1;
    params[4] = c.b2;
    params[5] = c.k1;
    params[6] = c.k2;
    params[7] = c.k3;
    params[8] = c.k4;
    params[9] = c.p1;
    params[10] = c.p2;
    params[11] = c.p3;
    params[12] = c.p4;
}

/// One residual block: reprojection error of a single point in a single camera.
/// Parameter order is (angle-axis rotation, translation, 3-D point, intrinsics);
/// the intrinsics arrive as an 11-double parameter block (f, cx, cy, b1, b2,
/// k1, k2, k3, k4, p1, p2) so self-calibration can free selected coefficients.
/// The block is shared by every camera on the same sensor.
struct ReprojectionCost {
    ReprojectionCost(CameraType type, int width, int height, const Eigen::Vector2d& pixel)
        : pixel_(pixel), type_(type), width_(width), height_(height) {}

    template <typename T>
    bool operator()(const T* const rotation, const T* const translation, const T* const point,
                    const T* const intr, T* residual) const {
        T p[3];
        ceres::AngleAxisRotatePoint(rotation, point, p);
        p[0] += translation[0];
        p[1] += translation[1];
        p[2] += translation[2];

        // project_point<T> throws for a Frame camera when z <= 0, and throwing
        // through autodiff is forbidden; returning false makes Ceres reject the
        // step instead. Fisheye is valid behind the image plane (no check).
        if (type_ == CameraType::Frame && !(p[2] > T(0))) {
            return false;
        }

        T uv[2];
        project_point<T>(type_, width_, height_, intr, p, uv);
        residual[0] = uv[0] - T(pixel_[0]);
        residual[1] = uv[1] - T(pixel_[1]);
        return true;
    }

private:
    Eigen::Vector2d pixel_;
    CameraType type_ = CameraType::Frame;
    int width_ = 0;
    int height_ = 0;
};

/// Residual for a GPS-like prior on a camera centre. The pose block is
/// (angle-axis rotation, translation); the camera centre is C = -R^T t, which
/// with angle-axis `aa` is C = -AngleAxisRotatePoint(-aa, t) (rotating t by the
/// inverse rotation). A fixed-size 3-residual is written; an axis whose sigma is
/// <= 0 or non-finite is left 0 and so contributes nothing. Measure is held as a
/// member (a parameter block cannot carry it).
struct CentrePriorCost {
    CentrePriorCost(const Eigen::Vector3d& centre, const Eigen::Vector3d& sigma)
        : centre_(centre), sigma_(sigma) {}

    template <typename T>
    bool operator()(const T* const rotation, const T* const translation, T* residual) const {
        // R^T t: rotate t by the inverse of `rotation`, i.e. by -rotation.
        T neg_aa[3] = {-rotation[0], -rotation[1], -rotation[2]};
        T r_t[3];
        ceres::AngleAxisRotatePoint(neg_aa, translation, r_t);
        const T centre[3] = {-r_t[0], -r_t[1], -r_t[2]};

        for (int k = 0; k < 3; ++k) {
            const double s = sigma_[k];
            if (std::isfinite(s) && s > 0.0) {
                residual[k] = (centre[k] - T(centre_[k])) / T(s);
            } else {
                residual[k] = T(0);
            }
        }
        return true;
    }

private:
    Eigen::Vector3d centre_;
    Eigen::Vector3d sigma_;
};

/// Huber loss scale for centre priors, in sigma units (ours): residuals beyond
/// 3 sigma are downweighted linearly, so a single bad GPS fix cannot drag the
/// solution.
constexpr double kPriorLossScaleSigma = 3.0;

/// Calibration index for a camera: the sensor map when present, else the camera
/// index itself (the pre-existing one-calibration-per-camera layout).
std::size_t sensor_index(const BAProblem& problem, std::size_t camera) {
    return problem.camera_sensor.empty() ? camera : problem.camera_sensor[camera];
}

/// Number of constrained prior axes (finite sigma > 0) over every prior.
std::size_t count_prior_axes(const BAProblem& problem) {
    std::size_t n = 0;
    for (const BACentrePrior& prior : problem.centre_priors) {
        for (int k = 0; k < 3; ++k) {
            const double s = prior.sigma[k];
            if (std::isfinite(s) && s > 0.0) {
                ++n;
            }
        }
    }
    return n;
}

/// RMS of the normalised prior residuals (C - centre)/sigma over all constrained
/// axes, from the current poses. 0 when there are none.
double prior_rms_sigma(const BAProblem& problem) {
    double sum_sq = 0.0;
    std::size_t n = 0;
    for (const BACentrePrior& prior : problem.centre_priors) {
        const BAPose& pose = problem.poses[prior.camera];
        const Eigen::Vector3d centre = -pose.R.transpose() * pose.t;
        for (int k = 0; k < 3; ++k) {
            const double s = prior.sigma[k];
            if (std::isfinite(s) && s > 0.0) {
                const double r = (centre[k] - prior.centre[k]) / s;
                sum_sq += r * r;
                ++n;
            }
        }
    }
    return n == 0 ? 0.0 : std::sqrt(sum_sq / static_cast<double>(n));
}

}  // namespace

double reprojection_rms_px(const BAProblem& problem) {
    if (problem.observations.empty()) {
        return 0.0;
    }
    double sum_sq = 0.0;
    double params[kCalibrationParamCount] = {};
    for (const BAObservation& obs : problem.observations) {
        const Calibration& cal = problem.calibrations[sensor_index(problem, obs.camera)];
        pack_calibration(cal, params);

        const BAPose& pose = problem.poses[obs.camera];
        const Eigen::Vector3d p_world = problem.points[obs.point];
        const Eigen::Vector3d p_cam = pose.R * p_world + pose.t;

        double err = 0.0;
        if (cal.type == CameraType::Frame && !(p_cam.z() > 0.0)) {
            // Behind the image plane: charge the fixed large error instead of
            // letting project() throw.
            const double d = kBehindCameraErrorPx;
            err = std::sqrt(d * d + d * d);
        } else {
            double uv[2] = {};
            const double p_in[3] = {p_cam.x(), p_cam.y(), p_cam.z()};
            project_point<double>(cal.type, cal.width, cal.height, params, p_in, uv);
            const double du = uv[0] - obs.pixel.x();
            const double dv = uv[1] - obs.pixel.y();
            err = std::hypot(du, dv);
        }
        sum_sq += err * err;
    }
    const double n = static_cast<double>(problem.observations.size());
    return std::sqrt(sum_sq / n);
}

int choose_scale_anchor_axis(const BAPose& pose0, const BAPose& pose1) {
    // The baseline in camera-1 coordinates: the transform mapping camera 0's
    // centre into camera 1's frame. t_rel is the ONLY part of t1 that responds
    // to a scale about pose0's centre (see the gauge note in the header), so its
    // largest component is the one to pin.
    const Eigen::Vector3d t_rel = pose1.t - pose1.R * pose0.R.transpose() * pose0.t;
    int axis = 0;
    if (std::abs(t_rel.y()) > std::abs(t_rel[axis])) {
        axis = 1;
    }
    if (std::abs(t_rel.z()) > std::abs(t_rel[axis])) {
        axis = 2;
    }
    return axis;
}

BAReport bundle_adjust(BAProblem& problem, const BAOptions& options) {
    // ---- Validation ------------------------------------------------------
    const std::size_t num_cameras = problem.poses.size();
    const std::size_t num_calibrations = problem.calibrations.size();
    if (problem.camera_sensor.empty()) {
        // Pre-existing layout: one calibration per camera.
        if (num_calibrations != num_cameras) {
            throw ValueError("bundle_adjust: calibrations.size() != poses.size()");
        }
    } else {
        if (problem.camera_sensor.size() != num_cameras) {
            throw ValueError("bundle_adjust: camera_sensor.size() != poses.size()");
        }
        for (const std::size_t sensor : problem.camera_sensor) {
            if (sensor >= num_calibrations) {
                throw ValueError("bundle_adjust: camera_sensor index out of range");
            }
        }
    }
    if (!problem.fit_intrinsics.empty() && problem.fit_intrinsics.size() != num_calibrations) {
        throw ValueError("bundle_adjust: fit_intrinsics.size() != calibrations.size()");
    }
    if (!problem.fixed_pose.empty() && problem.fixed_pose.size() != num_cameras) {
        throw ValueError("bundle_adjust: fixed_pose.size() != poses.size()");
    }
    for (const BAObservation& obs : problem.observations) {
        if (obs.camera >= num_cameras) {
            throw ValueError("bundle_adjust: observation camera index out of range");
        }
        if (obs.point >= problem.points.size()) {
            throw ValueError("bundle_adjust: observation point index out of range");
        }
    }
    for (const BACentrePrior& prior : problem.centre_priors) {
        if (prior.camera >= num_cameras) {
            throw ValueError("bundle_adjust: centre_prior camera index out of range");
        }
    }
    if (problem.scale_anchor_camera >= 0) {
        if (static_cast<std::size_t>(problem.scale_anchor_camera) >= num_cameras) {
            throw ValueError("bundle_adjust: scale_anchor_camera out of range");
        }
        if (problem.scale_anchor_axis < 0 || problem.scale_anchor_axis > 2) {
            throw ValueError("bundle_adjust: scale_anchor_axis out of range");
        }
    }

    BAReport report;
    report.observations = problem.observations.size();
    report.priors = count_prior_axes(problem);
    const bool has_priors = !problem.centre_priors.empty();
    if ((problem.observations.empty() && !has_priors) ||
        (problem.points.empty() && !has_priors) || num_cameras == 0) {
        report.usable = true;
        report.initial_rms_px = reprojection_rms_px(problem);
        report.final_rms_px = report.initial_rms_px;
        report.prior_rms_sigma = prior_rms_sigma(problem);
        return report;
    }

    report.initial_rms_px = reprojection_rms_px(problem);

    // ---- Parameter blocks ------------------------------------------------
    const std::vector<BAPose> original_poses = problem.poses;
    std::vector<std::array<double, 6>> cam_blocks(num_cameras);
    for (std::size_t i = 0; i < num_cameras; ++i) {
        double angle_axis[3] = {};
        ceres::RotationMatrixToAngleAxis(problem.poses[i].R.data(), angle_axis);
        cam_blocks[i] = {angle_axis[0],
                         angle_axis[1],
                         angle_axis[2],
                         problem.poses[i].t.x(),
                         problem.poses[i].t.y(),
                         problem.poses[i].t.z()};
    }
    std::vector<std::array<double, 3>> point_blocks(problem.points.size());
    for (std::size_t i = 0; i < problem.points.size(); ++i) {
        point_blocks[i] = {problem.points[i].x(), problem.points[i].y(), problem.points[i].z()};
    }
    // One kCalibrationParamCount-double intrinsics block per calibration, shared by every camera
    // that maps to it. Unchanged coefficients are held via SubsetManifold below.
    std::vector<std::array<double, kCalibrationParamCount>> intr_blocks(num_calibrations);
    for (std::size_t i = 0; i < num_calibrations; ++i) {
        pack_calibration(problem.calibrations[i], intr_blocks[i].data());
    }

    // The Problem takes ownership of the loss function (allocated once, shared
    // by every residual block, freed by the Problem), so no smart pointer here.
    ceres::LossFunction* loss = nullptr;
    if (options.loss_scale_px > 0.0) {
        loss = new ceres::CauchyLoss(options.loss_scale_px);
    }
    // Priors use their own robust loss (Huber, 3 sigma), owned by the Problem.
    ceres::LossFunction* prior_loss = nullptr;
    if (!problem.centre_priors.empty()) {
        prior_loss = new ceres::HuberLoss(kPriorLossScaleSigma);
    }
    ceres::Problem ceres_problem;

    for (const BAObservation& obs : problem.observations) {
        const std::size_t sensor = sensor_index(problem, obs.camera);
        const Calibration& cal = problem.calibrations[sensor];
        auto* cost = new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 3, 3, 3, kCalibrationParamCount>(
            new ReprojectionCost(cal.type, cal.width, cal.height, obs.pixel));
        ceres_problem.AddResidualBlock(cost, loss, cam_blocks[obs.camera].data(),
                                       cam_blocks[obs.camera].data() + 3, point_blocks[obs.point].data(),
                                       intr_blocks[sensor].data());
    }
    for (const BACentrePrior& prior : problem.centre_priors) {
        auto* cost = new ceres::AutoDiffCostFunction<CentrePriorCost, 3, 3, 3>(
            new CentrePriorCost(prior.centre, prior.sigma));
        ceres_problem.AddResidualBlock(cost, prior_loss, cam_blocks[prior.camera].data(),
                                       cam_blocks[prior.camera].data() + 3);
    }

    // ---- Gauge fixes -----------------------------------------------------
    for (std::size_t i = 0; i < num_cameras; ++i) {
        const bool fixed = !problem.fixed_pose.empty() && problem.fixed_pose[i] != 0;
        if (fixed) {
            ceres_problem.SetParameterBlockConstant(cam_blocks[i].data());
            ceres_problem.SetParameterBlockConstant(cam_blocks[i].data() + 3);
        }
    }
    if (problem.scale_anchor_camera >= 0) {
        const std::size_t cam = static_cast<std::size_t>(problem.scale_anchor_camera);
        const bool pose_fixed = !problem.fixed_pose.empty() && problem.fixed_pose[cam] != 0;
        if (!pose_fixed) {
            ceres_problem.SetManifold(cam_blocks[cam].data() + 3,
                                      new ceres::SubsetManifold(
                                          3, {problem.scale_anchor_axis}));
        }
    }
    // Intrinsics: a block with no free coefficients is constant; otherwise the
    // fixed indices are pinned with a SubsetManifold, leaving the free ones to
    // the solver. f (index 0), when free, gets a lower bound of 1e-3 * max(W, H).
    for (std::size_t s = 0; s < num_calibrations; ++s) {
        if (problem.fit_intrinsics.empty()) {
            ceres_problem.SetParameterBlockConstant(intr_blocks[s].data());
            continue;
        }
        std::vector<int> fixed;
        for (int k = 0; k < kCalibrationParamCount; ++k) {
            if (!problem.fit_intrinsics[s][static_cast<std::size_t>(k)]) {
                fixed.push_back(k);
            }
        }
        if (fixed.size() == static_cast<std::size_t>(kCalibrationParamCount)) {
            ceres_problem.SetParameterBlockConstant(intr_blocks[s].data());
            continue;
        }
        if (!fixed.empty()) {
            ceres_problem.SetManifold(intr_blocks[s].data(),
                                      new ceres::SubsetManifold(kCalibrationParamCount, fixed));
        }
        if (problem.fit_intrinsics[s][0]) {
            const Calibration& cal = problem.calibrations[s];
            const double bound =
                1e-3 * static_cast<double>(std::max(cal.width, cal.height));
            ceres_problem.SetParameterLowerBound(intr_blocks[s].data(), 0, bound);
        }
    }
    // Points with no observations are simply never added to the Problem, so
    // they cannot move and need no explicit constant marking.

    // ---- Solve -----------------------------------------------------------
    ceres::Solver::Options solver_options;
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    solver_options.minimizer_type = ceres::TRUST_REGION;
    solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    solver_options.max_num_iterations = options.max_iterations;
    solver_options.function_tolerance = options.function_tolerance;
    if (options.num_threads > 0) {
        solver_options.num_threads = options.num_threads;
    }
    solver_options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &ceres_problem, &summary);

    // ---- Write back ------------------------------------------------------
    // Fixed cameras are restored bitwise, not round-tripped through Ceres:
    // R -> angle-axis -> R is only accurate to ~1e-16, and a fixed pose must
    // come back exactly as given.
    for (std::size_t i = 0; i < num_cameras; ++i) {
        const bool fixed = !problem.fixed_pose.empty() && problem.fixed_pose[i] != 0;
        if (fixed) {
            problem.poses[i] = original_poses[i];
            continue;
        }
        double R[9] = {};
        ceres::AngleAxisToRotationMatrix(cam_blocks[i].data(), R);
        problem.poses[i].R << R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8];
        problem.poses[i].t = Eigen::Vector3d(cam_blocks[i][3], cam_blocks[i][4], cam_blocks[i][5]);
    }
    for (std::size_t i = 0; i < problem.points.size(); ++i) {
        problem.points[i] = Eigen::Vector3d(point_blocks[i][0], point_blocks[i][1], point_blocks[i][2]);
    }
    // Intrinsics: write back only the coefficients marked free, so entries held
    // fixed come back bitwise as given (the block holds them unchanged anyway,
    // but this makes that guarantee explicit).
    if (!problem.fit_intrinsics.empty()) {
        for (std::size_t s = 0; s < num_calibrations; ++s) {
            Calibration& c = problem.calibrations[s];
            const std::array<double, kCalibrationParamCount>& b = intr_blocks[s];
            const std::array<bool, kCalibrationParamCount>& fit = problem.fit_intrinsics[s];
            if (fit[0]) c.f = b[0];
            if (fit[1]) c.cx = b[1];
            if (fit[2]) c.cy = b[2];
            if (fit[3]) c.b1 = b[3];
            if (fit[4]) c.b2 = b[4];
            if (fit[5]) c.k1 = b[5];
            if (fit[6]) c.k2 = b[6];
            if (fit[7]) c.k3 = b[7];
            if (fit[8]) c.k4 = b[8];
            if (fit[9]) c.p1 = b[9];
            if (fit[10]) c.p2 = b[10];
            if (fit[11]) c.p3 = b[11];
            if (fit[12]) c.p4 = b[12];
        }
    }

    report.usable = summary.IsSolutionUsable();
    report.iterations = static_cast<int>(summary.iterations.size());
    report.final_rms_px = reprojection_rms_px(problem);
    report.prior_rms_sigma = prior_rms_sigma(problem);
    return report;
}

}  // namespace alphashape::sfm
