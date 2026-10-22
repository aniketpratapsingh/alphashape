#pragma once

// Milestone 4: bundle adjustment over a fixed set of calibrated cameras.
//
// The problem is the standard one — minimise the sum of squared reprojection
// errors of 3-D points seen by pinhole / fisheye cameras — solved with Ceres
// (sparse Schur, Levenberg-Marquardt, autodiff through the *same*
// project_point<T> the runtime uses; see model/camera/calibration.hpp).
//
// This header deliberately keeps Ceres out of the public surface: callers get
// plain Eigen types, so the solver dependency never leaks into the rest of the
// engine. Interiors (cost functors, Ceres options) live in the .cpp.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "model/camera/calibration.hpp"

namespace alphashape::sfm {

/// World->camera: X_cam = R X_world + t.  (Same layout as sfm::AbsolutePose,
/// which another module defines; keep these two field-compatible: R then t.)
struct BAPose {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

/// One 2-D measurement of a 3-D point in one camera.
struct BAObservation {
    std::size_t camera = 0;  ///< index into BAProblem::poses / calibrations
    std::size_t point = 0;   ///< index into BAProblem::points
    Eigen::Vector2d pixel;   ///< Calibration pixel convention (top-left pixel centre = (0.5,0.5))
};

/// A GPS-like prior on one camera centre, in the problem's coordinate frame.
/// For each axis k with a finite sigma[k] > 0 the solve adds
///     ((C[k] - centre[k]) / sigma[k])^2
/// with C = -R^T t the camera centre of that pose. An axis with sigma <= 0 or
/// non-finite contributes nothing (residual 0). Priors are robustified with a
/// Huber loss of scale 3 in sigma units (see bundle_adjust) so one bad fix
/// cannot drag the solution.
struct BACentrePrior {
    std::size_t camera = 0;   ///< index into BAProblem::poses
    Eigen::Vector3d centre;   ///< measured camera centre, in the problem's coordinate frame
    Eigen::Vector3d sigma;    ///< per-axis standard deviation; sigma <= 0 or non-finite => axis unconstrained
};

/// The bundle-adjustment problem, in place. Intrinsics are held fixed in this
/// milestone, so `calibrations` is read-only during the solve.
///
/// Gauge. The objective is invariant under a global similarity: an arbitrary
/// rotation, translation and uniform scale applied to (poses, points) leaves
/// every reprojection error unchanged. One gauge is removed by holding a camera
/// fully constant (`fixed_pose`, i.e. 6 DoF); the 7th DoF, the global scale, is
/// removed by pinning one translation component of another camera
/// (`scale_anchor_camera` / `scale_anchor_axis`).
///
/// Which component to pin matters. Fixing camera 0 removes its 6 DoF, so the
/// remaining freedom is a similarity about camera 0's centre C0. Under a scale
/// factor s applied about C0, camera 1's translation does NOT scale as s*t1:
/// writing the world->camera-1 transform through camera 0,
///     X_cam1 = R1 R0^T X_cam0 + t_rel,   t_rel = t1 - R1 R0^T t0,
/// the transform C0 -> camera 1 is exactly `t_rel`, the baseline expressed in
/// camera 1's frame. Scaling the world about C0 multiplies t_rel by s and leaves
/// the camera-1 orientation untouched, so the component of t that actually
/// carries the scale signal is t_rel, not t1. t1 itself is dominated by the
/// depth to the world origin, which barely moves under a scale about C0 and so
/// barely constrains s — that is why `choose_scale_anchor_axis` picks
/// argmax_k |t_rel[k]|. A camera whose baseline to the fixed camera is near zero
/// (t_rel ~ 0) is a poor anchor: no component of its translation responds to s,
/// so the scale stays unconstrained.
///
/// Gauge with priors. When the caller supplies at least three centre priors
/// whose measured centres are NOT collinear, those priors pin all 7 similarity
/// degrees of freedom (3 rotation, 3 translation, 1 scale), so callers may leave
/// every pose free and set no scale anchor. Otherwise the caller must keep
/// fixing a camera plus the scale anchor, as above. `bundle_adjust` itself only
/// obeys the flags: it never auto-detects whether the priors suffice.
struct BAProblem {
    std::vector<Calibration> calibrations;  ///< one per camera; fixed unless fit_intrinsics frees terms
    std::vector<BAPose> poses;
    std::vector<Eigen::Vector3d> points;
    std::vector<BAObservation> observations;
    std::vector<std::uint8_t> fixed_pose;  ///< per camera, 1 = constant (gauge); size == poses.size()

    /// Camera -> index into `calibrations`. Empty => calibrations is one per
    /// camera (index = camera), the pre-existing behaviour.
    std::vector<std::size_t> camera_sensor;
    /// Per entry of `calibrations`: which of the 11 coefficients are free (order
    /// as in calibration.hpp). Empty => every calibration fully fixed
    /// (pre-existing behaviour). Otherwise size == calibrations.size().
    std::vector<std::array<bool, kCalibrationParamCount>> fit_intrinsics;

    /// GPS-like camera-centre priors. Empty => no priors (pre-existing
    /// behaviour). See BACentrePrior and the gauge note above.
    std::vector<BACentrePrior> centre_priors;

    /// Scale gauge: if scale_anchor_camera >= 0, that camera's world->camera
    /// translation component scale_anchor_axis (0..2) is held constant
    /// (ceres::SubsetManifold on the t block). Choose the axis with
    /// `choose_scale_anchor_axis` (see above) so the anchor binds the scale.
    int scale_anchor_camera = -1;
    int scale_anchor_axis = 0;
};

struct BAOptions {
    double loss_scale_px = 1.0;  ///< ours: Cauchy loss scale in pixels; <= 0 => plain least squares
    int max_iterations = 100;    ///< ours
    int num_threads = 0;         ///< 0 => hardware concurrency
    double function_tolerance = 1e-8;
};

struct BAReport {
    bool usable = false;  ///< Ceres returned a usable solution
    int iterations = 0;
    double initial_rms_px = 0, final_rms_px = 0;  ///< plain (non-robust) RMS over all observations
    std::size_t observations = 0;

    /// Number of constrained prior axes (sigma finite and > 0) over all priors.
    std::size_t priors = 0;
    /// RMS of the normalised prior residuals (C[k] - centre[k]) / sigma[k] after
    /// the solve, over every constrained axis. 0 when there are none.
    double prior_rms_sigma = 0;
};

/// Validates indices/sizes (ValueError from base/utils/error.hpp on any
/// mismatch), then solves in place with Ceres (SPARSE_SCHUR, LM, AutoDiff),
/// writing poses and points back.
BAReport bundle_adjust(BAProblem& problem, const BAOptions& options);

/// RMS reprojection error in pixels (no Ceres; usable anywhere). Observations
/// whose point falls behind a Frame camera count as a fixed large error (see
/// kBehindCameraErrorPx in the .cpp).
double reprojection_rms_px(const BAProblem& problem);

/// The translation axis of `pose1` best suited as the scale anchor when `pose0`
/// is held fixed: argmax_k |t_rel[k]|, t_rel = pose1.t - pose1.R * pose0.R^T *
/// pose0.t (the baseline in camera-1 coordinates). Fixing that component pins
/// the global scale. Returns 0 if pose0 and pose1 coincide (t_rel ~ 0), which is
/// the degenerate-anchor case.
int choose_scale_anchor_axis(const BAPose& pose0, const BAPose& pose1);

}  // namespace alphashape::sfm
