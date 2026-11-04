#pragma once

// Control points, check points and scale bars for the bundle adjustment.
//
// A control point ties the 3-D position of one BAProblem::points entry to a
// known reference position, axis by axis. A scale bar ties the distance between
// two points to a measured length. Both residuals are normalised by their
// sigma and robustified with the same Huber loss (3 sigma) the camera-centre
// priors use, so one blundered point cannot drag the solution.
//
// Check points only validate. A check point contributes no control or scale-bar
// residual, and every image observation of it is excluded from the solve. A
// point is a check point when it is flagged in BAProblem::checkpoint_point OR
// carries a BAControlPoint with `checkpoint == true`; the two spellings are
// combined so they can never disagree. After the solve the point is
// re-triangulated from the final poses (inverting distortion and fisheye) and
// compared with its reference; the error is reported in BAReport. Observations
// of control and scale-bar points use a non-collapsing Huber loss, so a control
// residual that moves only the point block cannot detach it from its rays.

#include <cstddef>

#include <Eigen/Core>

namespace alphashape::sfm {

/// One known 3-D position, in the problem's coordinate frame.
/// `point` indexes BAProblem::points. Each axis carries its own sigma, so
/// horizontal and vertical accuracy may differ; an axis with sigma <= 0 or
/// non-finite is unconstrained (residual 0). `checkpoint == true` makes this a
/// validation-only control point: it contributes no control residual and also
/// marks `point` as a check point (equivalent to setting
/// BAProblem::checkpoint_point[point]).
struct BAControlPoint {
    std::size_t point = 0;
    Eigen::Vector3d reference = Eigen::Vector3d::Zero();
    Eigen::Vector3d sigma = Eigen::Vector3d::Zero();
    bool checkpoint = false;
};

/// A measured distance between two points. The residual is
/// (|Xa - Xb| - length) / sigma. Skipped as a solve constraint when `checkpoint`
/// is true or when either endpoint is a check point; such a bar is scored after
/// re-triangulation and reported separately (BAReport::checkbars). A bar with
/// invalid length/sigma is ignored. `point_a` and `point_b` index
/// BAProblem::points. When the endpoints coincide the residual is the honest
/// -length/sigma with a zero Jacobian.
struct BAScalebar {
    std::size_t point_a = 0;
    std::size_t point_b = 0;
    double length = 0.0;
    double sigma = 0.001;
    bool checkpoint = false;
};

namespace detail {

/// Evaluate the scale-bar residual and its analytic Jacobian. `a` and `b` are
/// the two points (3 doubles each); `residual` receives 1 value, `jac_a` and
/// `jac_b` 3 values each (may be null). An invalid length/sigma or a non-finite
/// distance yields a zero residual and zero Jacobians. Coincident endpoints keep
/// the honest residual -length/sigma with a zero Jacobian. Declared here so
/// tests can compare the analytic derivative with finite differences without
/// pulling Ceres into the public header.
void scalebar_residual_and_jacobian(const double* a, const double* b, double length,
                                    double sigma, double* residual, double* jac_a,
                                    double* jac_b);

}  // namespace detail

}  // namespace alphashape::sfm
