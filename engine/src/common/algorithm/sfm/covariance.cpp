// Posterior covariance for bundle adjustment: sigma0^2 * (J^T J)^-1 for the
// requested blocks. See covariance.hpp for the contract.

#include "common/algorithm/sfm/covariance.hpp"

#include <cstddef>
#include <cmath>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "model/camera/calibration.hpp"

namespace alphashape::sfm {
namespace {

/// Copies a Ceres covariance block (row-major dense) into `target`.
template <typename Target>
void copy_block(const double* data, Eigen::Index rows, Eigen::Index cols, Target* target) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        source(data, rows, cols);
    *target = source;
}

}  // namespace

CovarianceOutput compute_covariance(ceres::Problem& problem,
                                    const CovarianceBlocks& blocks,
                                    const CovarianceRequest& request,
                                    double sigma0_squared) {
    CovarianceOutput out;
    auto clear_blocks = [&out]() {
        out.camera_covariance.clear();
        out.point_covariance.clear();
        out.calibration_covariance.clear();
    };
    const std::size_t camera_count = blocks.cameras.size();
    const std::size_t point_count = blocks.points.size();
    const std::size_t calibration_count = blocks.calibrations.size();

    bool want_cameras = request.cameras;
    bool want_points = request.points;
    bool want_calibrations = request.calibrations;
    std::string reason;
    if (want_cameras && camera_count > request.max_cameras) {
        want_cameras = false;
        reason = "camera covariance skipped: " + std::to_string(camera_count) +
                 " cameras exceed the cap of " + std::to_string(request.max_cameras);
    }
    if (want_points && point_count > request.max_points) {
        want_points = false;
        if (!reason.empty()) {
            reason += "; ";
        }
        reason += "point covariance skipped: " + std::to_string(point_count) +
                  " points exceed the cap of " + std::to_string(request.max_points);
    }
    if (!want_cameras && !want_points && !want_calibrations) {
        out.reason = reason.empty() ? "no covariance requested" : reason;
        return out;
    }
    if (!(sigma0_squared >= 0.0) || !std::isfinite(sigma0_squared)) {
        out.reason = "covariance skipped: sigma0 squared is not finite";
        return out;
    }

    // Ask only for the block pairs we need. A block that never received a
    // residual is not registered with the Problem; requesting it would make
    // Compute() fail, so those stay zero in the result.
    std::vector<std::pair<const double*, const double*>> pairs;
    if (want_cameras) {
        for (double* camera : blocks.cameras) {
            if (!problem.HasParameterBlock(camera) ||
                problem.IsParameterBlockConstant(camera)) {
                continue;
            }
            pairs.emplace_back(camera, camera);
            pairs.emplace_back(camera, camera + 3);
            pairs.emplace_back(camera + 3, camera + 3);
        }
    }
    if (want_points) {
        for (double* point : blocks.points) {
            if (problem.HasParameterBlock(point)) {
                pairs.emplace_back(point, point);
            }
        }
    }
    if (want_calibrations) {
        for (double* calibration : blocks.calibrations) {
            if (problem.HasParameterBlock(calibration) &&
                !problem.IsParameterBlockConstant(calibration)) {
                pairs.emplace_back(calibration, calibration);
            }
        }
    }
    if (pairs.empty()) {
        out.reason = "covariance skipped: no parameter block was part of the solve";
        return out;
    }

    try {
        ceres::Covariance::Options options;
        options.algorithm_type = ceres::SPARSE_QR;
        options.apply_loss_function = true;
        options.null_space_rank = 0;
        ceres::Covariance covariance(options);
        if (!covariance.Compute(pairs, &problem)) {
            out.reason =
                "covariance computation failed (rank deficient or no free parameters)";
            return out;
        }

        bool ok = true;
        if (want_cameras) {
            out.camera_covariance.assign(camera_count,
                                         Eigen::Matrix<double, 6, 6>::Zero());
            for (std::size_t i = 0; i < camera_count && ok; ++i) {
                double* camera = blocks.cameras[i];
                if (!problem.HasParameterBlock(camera) ||
                    problem.IsParameterBlockConstant(camera)) {
                    continue;  // constant camera: zero covariance
                }
                double rr[9] = {};
                double rt[9] = {};
                double tt[9] = {};
                ok = covariance.GetCovarianceBlock(camera, camera, rr) &&
                     covariance.GetCovarianceBlock(camera, camera + 3, rt) &&
                     covariance.GetCovarianceBlock(camera + 3, camera + 3, tt);
                if (!ok) {
                    break;
                }
                Eigen::Matrix3d rotation = Eigen::Matrix3d::Zero();
                Eigen::Matrix3d cross = Eigen::Matrix3d::Zero();
                Eigen::Matrix3d translation = Eigen::Matrix3d::Zero();
                copy_block(rr, 3, 3, &rotation);
                copy_block(rt, 3, 3, &cross);
                copy_block(tt, 3, 3, &translation);
                Eigen::Matrix<double, 6, 6> block = Eigen::Matrix<double, 6, 6>::Zero();
                block.block<3, 3>(0, 0) = rotation;
                block.block<3, 3>(0, 3) = cross;
                block.block<3, 3>(3, 0) = cross.transpose();
                block.block<3, 3>(3, 3) = translation;
                out.camera_covariance[i] = block * sigma0_squared;
            }
        }
        if (want_points && ok) {
            out.point_covariance.assign(point_count, Eigen::Matrix3d::Zero());
            for (std::size_t i = 0; i < point_count && ok; ++i) {
                double* point = blocks.points[i];
                if (!problem.HasParameterBlock(point)) {
                    continue;
                }
                double pp[9] = {};
                ok = covariance.GetCovarianceBlock(point, point, pp);
                if (!ok) {
                    break;
                }
                Eigen::Matrix3d block = Eigen::Matrix3d::Zero();
                copy_block(pp, 3, 3, &block);
                out.point_covariance[i] = block * sigma0_squared;
            }
        }
        if (want_calibrations && ok) {
            const Eigen::Index n = static_cast<Eigen::Index>(kCalibrationParamCount);
            out.calibration_covariance.assign(
                calibration_count, Eigen::MatrixXd::Zero(n, n));
            std::vector<double> data(static_cast<std::size_t>(n) *
                                     static_cast<std::size_t>(n));
            for (std::size_t i = 0; i < calibration_count && ok; ++i) {
                double* calibration = blocks.calibrations[i];
                if (!problem.HasParameterBlock(calibration) ||
                    problem.IsParameterBlockConstant(calibration)) {
                    continue;
                }
                ok = covariance.GetCovarianceBlock(calibration, calibration,
                                                   data.data());
                if (!ok) {
                    break;
                }
                Eigen::MatrixXd block(n, n);
                copy_block(data.data(), n, n, &block);
                out.calibration_covariance[i] = block * sigma0_squared;
            }
        }
        if (!ok) {
            clear_blocks();
            out.reason = "covariance block query failed";
            return out;
        }
        out.valid = reason.empty();
        out.reason = reason.empty() ? std::string("ok") : reason;
        return out;
    } catch (const std::exception& error) {
        clear_blocks();
        out.reason = std::string("covariance computation failed: ") + error.what();
        return out;
    } catch (...) {
        clear_blocks();
        out.reason = "covariance computation failed";
        return out;
    }
}

}  // namespace alphashape::sfm
