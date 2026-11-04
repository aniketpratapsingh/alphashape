#pragma once

// Posterior covariance for a solved bundle-adjustment problem.
//
// Triggs et al. 2000 ("Bundle Adjustment -- A Modern Synthesis", sec. 4):
// the covariance of the estimate is sigma0^2 * (J^T J)^-1 at the solution.
// The full inverse is never formed; ceres::Covariance computes only the block
// pairs the caller asks for (per camera, per point, per calibration).
//
// Constant parameter blocks (fixed cameras, fixed calibrations) are handled by
// Ceres as zero covariance: the estimate is relative to the datum those blocks
// define (a fixed camera, control points or priors). A rank-deficient problem
// makes Compute() fail; that is reported, never thrown.
//
// This header is internal to the sfm implementation and may expose Ceres; the
// public bundle-adjustment surface (bundle_adjustment.hpp) does not.

#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <ceres/covariance.h>
#include <ceres/problem.h>

namespace alphashape::sfm {

/// Which blocks to compute and the caps above which a block is skipped.
struct CovarianceRequest {
    bool cameras = false;
    bool points = false;
    bool calibrations = false;
    std::size_t max_cameras = 2000;
    std::size_t max_points = 50000;
};

/// Non-owning views of the solved parameter blocks. A camera block is six
/// doubles (angle-axis rotation, then translation); a point block is three; a
/// calibration block is kCalibrationParamCount doubles in the packed order.
struct CovarianceBlocks {
    std::vector<double*> cameras;
    std::vector<double*> points;
    std::vector<double*> calibrations;
};

/// Result of compute_covariance(). Blocks are scaled by sigma0_squared already.
struct CovarianceOutput {
    bool valid = false;
    std::string reason;
    std::vector<Eigen::Matrix<double, 6, 6>> camera_covariance;
    std::vector<Eigen::Matrix3d> point_covariance;
    std::vector<Eigen::MatrixXd> calibration_covariance;
};

/// Computes sigma0_squared * (J^T J)^-1 for the requested block pairs on the
/// already-solved `problem`, using a sparse QR (SPARSE_QR). Blocks that are not
/// registered with the problem (no observation reached them) come back zero.
/// On any failure the output's vectors are empty and `reason` is set.
CovarianceOutput compute_covariance(ceres::Problem& problem,
                                    const CovarianceBlocks& blocks,
                                    const CovarianceRequest& request,
                                    double sigma0_squared);

}  // namespace alphashape::sfm
