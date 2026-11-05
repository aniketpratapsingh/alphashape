#pragma once

/// @file calibrate.hpp
/// @brief Camera calibration from chessboard views (WP-13.15a).
///
/// Closed-form initialisation (normalised DLT homographies plus Zhang's
/// constraints with the principal point fixed at the image centre), followed by
/// a nonlinear least-squares refinement of the interior orientation and one
/// pose per view, an intrinsic covariance from the normal equations, and
/// pruning of insignificant distortion coefficients.
///
/// Board points are given in metres on the board plane (z = 0); pixels use the
/// calibration convention of `Calibration` (top-left pixel centre at
/// (0.5, 0.5)). A partial board view carries an arbitrary in-plane lattice
/// origin, but the per-view pose absorbs it, so partial views need no special
/// handling: every view is fitted with its own 6-DoF pose.

#include <array>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "model/camera/calibration.hpp"

namespace alphashape {

/// One view's observations, in corresponding order.
struct CalibrationView {
    int camera_id = 0;
    std::vector<Eigen::Vector2d> board_points;  ///< metres, z = 0
    std::vector<Eigen::Vector2d> pixels;
};

struct CalibrateOptions {
    int max_iterations = 100;
    /// Zero a free distortion coefficient whose |value| < ratio * sigma.
    ///
    /// The default 1.0 is our lenient starting value: a term is dropped only
    /// when it sits within one standard error of zero, which is the weakest
    /// conventional signal-to-noise cut-off. It is deliberately conservative
    /// because a wrongly pruned term cannot be recovered afterwards; raise it
    /// towards 2 (or 3) once real captures give us a false-positive rate.
    /// Purely ours; to be tuned on measured data, not a fixed rule.
    double prune_t_ratio = 1.0;
    int min_views = 3;
};

struct ViewError {
    int camera_id = 0;
    int corners = 0;
    double rms_px = 0.0;
    double max_px = 0.0;
};

struct CalibrationResult {
    Calibration calibration;
    double rms_px = 0.0;
    double max_px = 0.0;
    std::vector<ViewError> views;
    std::array<double, kCalibrationParamCount> sigma{};
    Eigen::MatrixXd correlation;
    std::vector<std::string> pruned;
    int views_used = 0;
};

using CalibrationFitFlags = std::array<bool, kCalibrationParamCount>;

/// Normalised DLT homography from board-plane XY to pixels (Hartley &
/// Zisserman 2004, 4.1). Accepts any correspondences from four upwards, i.e.
/// the minimal case is supported. Returns false for fewer than four
/// correspondences, for non-finite points or a degenerate point set, and for
/// a design matrix whose rank is below eight (three collinear points,
/// repeated points). The returned matrix has unit Frobenius norm.
bool homography_dlt(const std::vector<Eigen::Vector2d>& board,
                    const std::vector<Eigen::Vector2d>& pixels,
                    Eigen::Matrix3d& homography);

/// Fit flags matching the task defaults: f, cx, cy, k1, k2, k3, p1, p2 free;
/// b1, b2, k4, p3, p4 held.
CalibrationFitFlags default_calibration_fit_flags();

/// Solve. `initial` supplies the sensor type and image geometry; every
/// intrinsic is re-estimated subject to `fit`. Throws ValueError for invalid
/// arguments and Error when fewer than `options.min_views` views are usable or
/// the closed-form initialisation fails. Two of those failures name their
/// cause: views too close to fronto-parallel to separate focal length from
/// distance, and a closed-form focal length outside [0.1, 100] * max(width,
/// height) of the image.
CalibrationResult calibrate_camera(int image_width, int image_height,
                                   const std::vector<CalibrationView>& views,
                                   const Calibration& initial,
                                   const CalibrationFitFlags& fit,
                                   const CalibrateOptions& options = {});

}  // namespace alphashape
