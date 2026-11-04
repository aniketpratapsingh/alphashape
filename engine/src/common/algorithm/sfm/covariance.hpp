#pragma once

// Posterior covariance for a solved bundle-adjustment problem.
//
// Triggs et al. 2000 ("Bundle Adjustment -- A Modern Synthesis", sec. 4) and
// Forstner & Wrobel 2016: the covariance of the estimate is
// sigma0^2 * (J^T J)^-1 at the solution. The full inverse is never formed;
// the point columns are eliminated with the Schur complement so the cost is
// linear in the number of points (the sparse normal equations of the cameras
// and intrinsics are factorised with Eigen::SimplicialLDLT).
//
// Constant parameter blocks (fixed cameras, fixed calibrations) get zero
// covariance: the estimate is relative to the datum those blocks define (a
// fixed camera, control points or priors). A rank-deficient problem or a
// singular reduced normal matrix is reported, never thrown.
//
// This header is internal to the sfm implementation and may expose Ceres; the
// public bundle-adjustment surface (bundle_adjustment.hpp) does not.

#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <ceres/problem.h>

namespace alphashape::sfm {

/// Which blocks to compute and the caps above which a block is skipped.
struct CovarianceRequest {
    bool cameras = false;
    bool points = false;
    bool calibrations = false;
    std::size_t max_cameras = 2000;
    std::size_t max_points = 50000;
    /// Free camera and intrinsic columns above which the reduced normal matrix
    /// is too large to factorise: the whole request is skipped with a reason.
    std::size_t max_columns = 12000;
};

/// Non-owning views of the solved parameter blocks. A camera is two consecutive
/// three-double Ceres parameter blocks (angle-axis rotation, then translation),
/// and `cameras[i]` points at the rotation block; a point block is three
/// doubles; a calibration block is kCalibrationParamCount doubles in the packed
/// order.
struct CovarianceBlocks {
    std::vector<double*> cameras;
    std::vector<double*> points;
    std::vector<double*> calibrations;
};

/// Result of compute_covariance(). Blocks are scaled by sigma0_squared already.
struct CovarianceOutput {
    bool valid = false;
    std::string reason;
    /// Points whose 3x3 normal block was rank-deficient (for example observed
    /// by fewer than two cameras) and therefore dropped from the elimination.
    std::size_t points_dropped = 0;
    std::vector<Eigen::Matrix<double, 6, 6>> camera_covariance;
    std::vector<Eigen::Matrix3d> point_covariance;
    std::vector<Eigen::MatrixXd> calibration_covariance;
};

/// Computes sigma0_squared * (J^T J)^-1 for the requested block pairs on the
/// already-solved `problem` by Schur-complement elimination of the point
/// columns. Blocks that are not registered with the problem (no observation
/// reached them) come back zero. On any failure the output's vectors are empty
/// (or partially filled when only some requests were capped) and `reason` is
/// set.
CovarianceOutput compute_covariance(ceres::Problem& problem,
                                    const CovarianceBlocks& blocks,
                                    const CovarianceRequest& request,
                                    double sigma0_squared);

}  // namespace alphashape::sfm
