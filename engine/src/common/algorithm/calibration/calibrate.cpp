#include "common/algorithm/calibration/calibrate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SVD>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace alphashape {
namespace {

constexpr int kMinimumCorners = 6;

using Packed = std::array<double, kCalibrationParamCount>;

Packed pack_calibration(const Calibration& calibration) {
    return {calibration.f,  calibration.cx, calibration.cy, calibration.b1,
            calibration.b2, calibration.k1, calibration.k2, calibration.k3,
            calibration.k4, calibration.p1, calibration.p2, calibration.p3,
            calibration.p4};
}

void unpack_calibration(const Packed& packed, Calibration& calibration) {
    calibration.f = packed[0];
    calibration.cx = packed[1];
    calibration.cy = packed[2];
    calibration.b1 = packed[3];
    calibration.b2 = packed[4];
    calibration.k1 = packed[5];
    calibration.k2 = packed[6];
    calibration.k3 = packed[7];
    calibration.k4 = packed[8];
    calibration.p1 = packed[9];
    calibration.p2 = packed[10];
    calibration.p3 = packed[11];
    calibration.p4 = packed[12];
}

/// Similarity normalisation: centroid to the origin, mean radius sqrt(2).
bool similarity_normalisation(const std::vector<Eigen::Vector2d>& points,
                              Eigen::Matrix3d& transform) {
    if (points.size() < std::size_t{4}) {
        return false;
    }
    Eigen::Vector2d centre = Eigen::Vector2d::Zero();
    for (const Eigen::Vector2d& point : points) {
        if (!point.allFinite()) {
            return false;
        }
        centre += point;
    }
    centre /= static_cast<double>(points.size());
    double mean_distance = 0.0;
    for (const Eigen::Vector2d& point : points) {
        mean_distance += (point - centre).norm();
    }
    mean_distance /= static_cast<double>(points.size());
    if (!(mean_distance > 1e-12) || !std::isfinite(mean_distance)) {
        return false;
    }
    const double scale = std::sqrt(2.0) / mean_distance;
    transform = Eigen::Matrix3d::Identity();
    transform(0, 0) = scale;
    transform(1, 1) = scale;
    transform(0, 2) = -scale * centre.x();
    transform(1, 2) = -scale * centre.y();
    return true;
}

}  // namespace

/// Normalised DLT homography from board-plane XY to pixels (Hartley &
/// Zisserman 2004, 4.1). Works for the four-correspondence minimum as well as
/// longer lists; see the declaration for the failure conditions.
bool homography_dlt(const std::vector<Eigen::Vector2d>& board,
                    const std::vector<Eigen::Vector2d>& pixels,
                    Eigen::Matrix3d& homography) {
    const std::size_t count = board.size();
    if (count < std::size_t{4} || pixels.size() != count) {
        return false;
    }
    Eigen::Matrix3d board_transform;
    Eigen::Matrix3d pixel_transform;
    if (!similarity_normalisation(board, board_transform) ||
        !similarity_normalisation(pixels, pixel_transform)) {
        return false;
    }
    // The four-point case yields only eight rows, so the thin SVD would return
    // eight right singular vectors and there would be no second-smallest
    // value to test. Pad with zero rows up to nine: a zero row carries no
    // information and leaves the null space (and therefore the homography)
    // unchanged, while giving the rank test the same quantity it has for
    // longer systems.
    const Eigen::Index rows =
        std::max<Eigen::Index>(static_cast<Eigen::Index>(2 * count), 9);
    Eigen::MatrixXd design = Eigen::MatrixXd::Zero(rows, 9);
    for (std::size_t i = 0; i < count; ++i) {
        const Eigen::Vector3d p =
            board_transform * Eigen::Vector3d(board[i].x(), board[i].y(), 1.0);
        const Eigen::Vector3d q =
            pixel_transform * Eigen::Vector3d(pixels[i].x(), pixels[i].y(), 1.0);
        const Eigen::Index row = static_cast<Eigen::Index>(2 * i);
        design(row, 0) = p.x();
        design(row, 1) = p.y();
        design(row, 2) = 1.0;
        design(row, 6) = -q.x() * p.x();
        design(row, 7) = -q.x() * p.y();
        design(row, 8) = -q.x();
        design(row + 1, 3) = p.x();
        design(row + 1, 4) = p.y();
        design(row + 1, 5) = 1.0;
        design(row + 1, 6) = -q.y() * p.x();
        design(row + 1, 7) = -q.y() * p.y();
        design(row + 1, 8) = -q.y();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(design, Eigen::ComputeThinV);
    const Eigen::VectorXd& singular = svd.singularValues();
    // The homography is the smallest right singular vector, so the smallest
    // value is expected to be (near) zero; rank deficiency shows up in the
    // second-smallest value. 1e-8 of the largest value is far above the
    // normalised DLT's rounding noise (typically 1e-15) yet far below the
    // 1e-3-or-worse conditioning of a genuinely usable board, so it rejects
    // only configurations that really are rank deficient (three collinear
    // points, repeated points).
    if (singular.size() < 9 || !(singular(0) > 0.0) ||
        !(singular(7) > 1e-8 * singular(0))) {
        return false;
    }
    const Eigen::VectorXd h = svd.matrixV().col(8);
    Eigen::Matrix3d normalised;
    normalised << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);
    homography = pixel_transform.inverse() * normalised * board_transform;
    const double norm = homography.norm();
    if (!homography.allFinite() || !(norm > 0.0)) {
        return false;
    }
    homography /= norm;
    return true;
}

namespace {

/// Zhang's two orthogonality constraints with the principal point fixed at the
/// image centre, zero skew and unit aspect reduce to one scalar per view in
/// g = 1/f^2:
///     g (h1x h2x + h1y h2y) + h1z h2z = 0
///     g (h1x^2 + h1y^2 - h2x^2 - h2y^2) + (h1z^2 - h2z^2) = 0
/// with h1, h2 the first two columns of the homography in centred pixels.
bool closed_form_focal(const std::vector<Eigen::Matrix3d>& homographies,
                       int width, int height, double& focal) {
    if (homographies.size() < std::size_t{3}) {
        return false;
    }
    double normal = 0.0;
    double rhs = 0.0;
    // Data scale for the conditioning test: (|h1|^2 + |h2|^2)^2, which is
    // (2 f^2)^2 for a well-conditioned view. The constraint coefficients grow
    // as f^2 sin^2(tilt) while this scale grows as f^4, so their ratio
    // behaves like sin^4(tilt) / 4.
    double scale = 0.0;
    for (const Eigen::Matrix3d& h : homographies) {
        const double a1 = h(0, 0) * h(0, 1) + h(1, 0) * h(1, 1);
        const double b1 = h(2, 0) * h(2, 1);
        const double a2 = h(0, 0) * h(0, 0) + h(1, 0) * h(1, 0) -
                          h(0, 1) * h(0, 1) - h(1, 1) * h(1, 1);
        const double b2 = h(2, 0) * h(2, 0) - h(2, 1) * h(2, 1);
        normal += a1 * a1 + a2 * a2;
        rhs += a1 * b1 + a2 * b2;
        const double level = h.col(0).squaredNorm() + h.col(1).squaredNorm();
        scale += level * level;
    }
    if (!std::isfinite(normal) || !std::isfinite(rhs) ||
        !std::isfinite(scale)) {
        return false;
    }
    // The relative floor 1e-8 sits at sin^4(tilt)/4 = 1e-8, i.e. a combined
    // board tilt of about 0.8 degrees: below that the two Zhang constraints
    // are indistinguishable from rounding noise and the focal length is not
    // recoverable, while any usable calibration set tilts by tens of degrees
    // and clears the floor by more than six orders of magnitude.
    if (!(normal > 1e-8 * scale)) {
        throw Error(
            "calibrate_camera: views are too close to fronto-parallel to "
            "initialise the focal length");
    }
    const double g = -rhs / normal;
    if (!(g > 0.0) || !std::isfinite(g)) {
        throw Error(
            "calibrate_camera: closed-form focal length is out of range");
    }
    focal = 1.0 / std::sqrt(g);
    // A focal length shorter than a tenth of the image or longer than a
    // hundred times it cannot belong to a real lens on this sensor; the ill
    // conditioning that produces such values is a failure, not a result.
    const double extent =
        0.1 * static_cast<double>(std::max(width, height));
    if (!std::isfinite(focal) || focal < extent || focal > 1000.0 * extent) {
        throw Error(
            "calibrate_camera: closed-form focal length is out of range");
    }
    return true;
}

/// Pose from K^-1 H (Zhang 2000, 3.3), with R projected onto SO(3) by SVD and
/// the sign chosen so the board centre lies in front of the camera.
bool pose_from_homography(const Eigen::Matrix3d& homography, double focal,
                          const Eigen::Vector2d& board_centre,
                          Eigen::Matrix3d& rotation, Eigen::Vector3d& translation) {
    Eigen::Matrix3d inverse_k = Eigen::Matrix3d::Identity();
    inverse_k(0, 0) = 1.0 / focal;
    inverse_k(1, 1) = 1.0 / focal;
    const Eigen::Matrix3d a = inverse_k * homography;
    const double scale = 0.5 * (a.col(0).norm() + a.col(1).norm());
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        return false;
    }
    const Eigen::Vector3d r1 = a.col(0) / scale;
    const Eigen::Vector3d r2 = a.col(1) / scale;
    Eigen::Matrix3d raw;
    raw.col(0) = r1;
    raw.col(1) = r2;
    raw.col(2) = r1.cross(r2);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(raw, Eigen::ComputeFullU |
                                                   Eigen::ComputeFullV);
    Eigen::Matrix3d corrected = Eigen::Matrix3d::Identity();
    corrected(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant() > 0.0
                          ? 1.0
                          : -1.0;
    rotation = svd.matrixU() * corrected * svd.matrixV().transpose();
    translation = a.col(2) / scale;
    const Eigen::Vector3d centre =
        rotation * Eigen::Vector3d(board_centre.x(), board_centre.y(), 0.0) +
        translation;
    if (!(centre.z() > 0.0)) {
        // Negating two rotation columns keeps det = +1 while flipping the
        // homography's sign; the translation must follow.
        rotation.col(0) = -rotation.col(0);
        rotation.col(1) = -rotation.col(1);
        translation = -translation;
    }
    return rotation.allFinite() && translation.allFinite() &&
           (rotation * Eigen::Vector3d(board_centre.x(), board_centre.y(), 0.0) +
                translation)
                   .z() > 0.0;
}

/// Reprojection residual for one board corner. The pose block is
/// (angle-axis rotation, translation) mapping board coordinates to the camera
/// frame (X_cam = R X_board + t); the intrinsics block is the packed 13-double
/// calibration. The projection is the single implementation in
/// calibration.hpp, shared with the bundle adjuster, so autodiff sees exactly
/// the runtime model.
struct ObservationCost {
    ObservationCost(CameraType type, int width, int height,
                    const Eigen::Vector3d& board_point,
                    const Eigen::Vector2d& pixel)
        : board_point_(board_point), pixel_(pixel), type_(type), width_(width),
          height_(height) {}

    template <typename T>
    bool operator()(const T* const pose, const T* const intrinsics,
                    T* residual) const {
        const T point[3] = {T(board_point_.x()), T(board_point_.y()),
                            T(board_point_.z())};
        T camera_point[3];
        ceres::AngleAxisRotatePoint(pose, point, camera_point);
        camera_point[0] += pose[3];
        camera_point[1] += pose[4];
        camera_point[2] += pose[5];
        T uv[2];
        if (!try_project_point<T>(type_, width_, height_, intrinsics,
                                  camera_point, uv)) {
            return false;  // behind the image plane: let Ceres reject the step
        }
        residual[0] = uv[0] - T(pixel_.x());
        residual[1] = uv[1] - T(pixel_.y());
        return true;
    }

private:
    Eigen::Vector3d board_point_;
    Eigen::Vector2d pixel_;
    CameraType type_ = CameraType::Frame;
    int width_ = 0;
    int height_ = 0;
};

struct Refinement {
    Packed intrinsics{};
    std::vector<std::array<double, 6>> poses;
    double cost = 0.0;
    Eigen::MatrixXd covariance;  ///< 13x13, unscaled inverse normal matrix
    bool usable = false;
    bool covariance_ok = false;
};

/// One nonlinear solve of the intrinsic block plus one 6-DoF pose per view,
/// followed by the intrinsic covariance. Held coefficients are removed from
/// the manifold with SubsetManifold, so the covariance is automatically
/// marginalised over the poses.
bool solve_refinement(const std::vector<CalibrationView>& views,
                      CameraType type, int width, int height, const Packed& start,
                      const CalibrationFitFlags& fit,
                      const std::vector<std::array<double, 6>>& start_poses,
                      int max_iterations, Refinement& out) {
    // The pruning re-solve passes this same Refinement's blocks back in as the
    // start, so copy them before resetting `out` or the inputs alias the
    // storage that the reset clears.
    const Packed start_copy = start;
    const std::vector<std::array<double, 6>> poses_copy = start_poses;
    out = Refinement{};
    out.intrinsics = start_copy;
    out.poses = poses_copy;
    ceres::Problem problem;
    std::vector<int> held;
    for (int k = 0; k < kCalibrationParamCount; ++k) {
        if (!fit[static_cast<std::size_t>(k)]) {
            held.push_back(k);
        }
    }
    if (!held.empty()) {
        // SetManifold requires the parameter block to be registered already;
        // AddResidualBlock would otherwise register it only afterwards.
        problem.AddParameterBlock(out.intrinsics.data(), kCalibrationParamCount);
        problem.SetManifold(
            out.intrinsics.data(),
            new ceres::SubsetManifold(kCalibrationParamCount, held));
    }
    for (std::size_t v = 0; v < views.size(); ++v) {
        const CalibrationView& view = views[v];
        for (std::size_t i = 0; i < view.board_points.size(); ++i) {
            const Eigen::Vector3d point(view.board_points[i].x(),
                                        view.board_points[i].y(), 0.0);
            auto* cost = new ceres::AutoDiffCostFunction<ObservationCost, 2, 6,
                                                         kCalibrationParamCount>(
                new ObservationCost(type, width, height, point, view.pixels[i]));
            problem.AddResidualBlock(cost, nullptr, out.poses[v].data(),
                                     out.intrinsics.data());
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = max_iterations;
    options.num_threads = 1;
    options.logging_type = ceres::SILENT;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable()) {
        return false;
    }
    out.cost = summary.final_cost;
    ceres::Covariance::Options covariance_options;
    covariance_options.algorithm_type = ceres::DENSE_SVD;
    ceres::Covariance covariance(covariance_options);
    const std::vector<std::pair<const double*, const double*>> blocks = {
        {out.intrinsics.data(), out.intrinsics.data()}};
    if (covariance.Compute(blocks, &problem)) {
        Eigen::Matrix<double, kCalibrationParamCount, kCalibrationParamCount> block;
        if (covariance.GetCovarianceBlock(out.intrinsics.data(),
                                          out.intrinsics.data(), block.data())) {
            out.covariance = block;
            out.covariance_ok = true;
        }
    }
    out.usable = true;
    return true;
}

struct InitialFit {
    std::vector<CalibrationView> views;
    std::vector<std::array<double, 6>> poses;
    double focal = 0.0;
};

/// Homographies, closed-form focal and per-view poses. Views with fewer than
/// six corners, a degenerate homography or a failed pose are skipped.
bool initialise_views(const std::vector<CalibrationView>& views, int width,
                      int height, InitialFit& out) {
    out = InitialFit{};
    Eigen::Matrix3d centre_shift = Eigen::Matrix3d::Identity();
    centre_shift(0, 2) = -0.5 * static_cast<double>(width);
    centre_shift(1, 2) = -0.5 * static_cast<double>(height);
    std::vector<std::pair<CalibrationView, Eigen::Matrix3d>> pairs;
    std::vector<Eigen::Matrix3d> homographies;
    for (const CalibrationView& view : views) {
        Eigen::Matrix3d homography;
        if (!homography_dlt(view.board_points, view.pixels, homography)) {
            continue;
        }
        pairs.emplace_back(view, centre_shift * homography);
        homographies.push_back(centre_shift * homography);
    }
    if (!closed_form_focal(homographies, width, height, out.focal)) {
        return false;
    }
    for (const auto& [view, homography] : pairs) {
        Eigen::Vector2d board_centre = Eigen::Vector2d::Zero();
        for (const Eigen::Vector2d& point : view.board_points) {
            board_centre += point;
        }
        board_centre /= static_cast<double>(view.board_points.size());
        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;
        if (!pose_from_homography(homography, out.focal, board_centre, rotation,
                                  translation)) {
            continue;
        }
        std::array<double, 6> pose{};
        ceres::RotationMatrixToAngleAxis(rotation.data(), pose.data());
        pose[3] = translation.x();
        pose[4] = translation.y();
        pose[5] = translation.z();
        out.views.push_back(view);
        out.poses.push_back(pose);
    }
    return out.views.size() >= std::size_t{3};
}

}  // namespace

CalibrationFitFlags default_calibration_fit_flags() {
    CalibrationFitFlags fit{};
    fit[0] = true;   // f
    fit[1] = true;   // cx
    fit[2] = true;   // cy
    fit[5] = true;   // k1
    fit[6] = true;   // k2
    fit[7] = true;   // k3
    fit[9] = true;   // p1
    fit[10] = true;  // p2
    return fit;
}

namespace {

/// Closed-form starting point: held coefficients keep the caller's values,
/// free ones take the focal estimate (f) or zero (everything else).
Packed initial_packed(const Calibration& initial, const CalibrationFitFlags& fit,
                      double focal) {
    Packed packed = pack_calibration(initial);
    for (int k = 0; k < kCalibrationParamCount; ++k) {
        const std::size_t kk = static_cast<std::size_t>(k);
        if (fit[kk]) {
            packed[kk] = k == 0 ? focal : 0.0;
        }
    }
    return packed;
}

/// Scale the inverse normal matrix by the residual variance and split it into
/// per-parameter sigma and the correlation matrix. Held coefficients report
/// sigma 0 and correlation 0; the diagonal of the correlation is always 1.
bool covariance_outputs(const Refinement& refinement, const CalibrationFitFlags& fit,
                        int residual_count, int free_count,
                        std::array<double, kCalibrationParamCount>& sigma,
                        Eigen::MatrixXd& correlation) {
    if (!refinement.covariance_ok) {
        return false;
    }
    const int degrees = residual_count - free_count;
    if (degrees <= 0) {
        return false;
    }
    const double variance = 2.0 * refinement.cost / static_cast<double>(degrees);
    if (!(variance >= 0.0) || !std::isfinite(variance)) {
        return false;
    }
    const Eigen::MatrixXd covariance =
        0.5 * variance * (refinement.covariance + refinement.covariance.transpose());
    sigma.fill(0.0);
    for (int k = 0; k < kCalibrationParamCount; ++k) {
        if (fit[static_cast<std::size_t>(k)]) {
            sigma[static_cast<std::size_t>(k)] =
                std::sqrt(std::max(0.0, covariance(k, k)));
        }
    }
    correlation = Eigen::MatrixXd::Identity(kCalibrationParamCount,
                                            kCalibrationParamCount);
    for (int i = 0; i < kCalibrationParamCount; ++i) {
        for (int j = 0; j < kCalibrationParamCount; ++j) {
            if (i == j || !fit[static_cast<std::size_t>(i)] ||
                !fit[static_cast<std::size_t>(j)]) {
                continue;
            }
            const double denominator = sigma[static_cast<std::size_t>(i)] *
                                       sigma[static_cast<std::size_t>(j)];
            if (!(denominator > 0.0)) {
                continue;
            }
            correlation(i, j) =
                std::clamp(covariance(i, j) / denominator, -1.0, 1.0);
        }
    }
    return true;
}

struct ErrorSummary {
    double rms_px = 0.0;
    double max_px = 0.0;
    std::vector<ViewError> views;
};

/// Per-coordinate RMS and per-corner maximum, evaluated with the same forward
/// model as the solve.
ErrorSummary measure_errors(const std::vector<CalibrationView>& views,
                            const std::vector<std::array<double, 6>>& poses,
                            CameraType type, int width, int height,
                            const Packed& intrinsics) {
    ErrorSummary out;
    out.views.reserve(views.size());
    double total_squared = 0.0;
    double global_max_squared = 0.0;
    std::size_t total_corners = 0;
    for (std::size_t v = 0; v < views.size(); ++v) {
        const CalibrationView& view = views[v];
        const std::array<double, 6>& pose = poses[v];
        // Rebuild the rotation the same way bundle_adjustment.cpp does, so the
        // angle-axis <-> matrix convention is identical to the solver's.
        double raw_rotation[9] = {};
        ceres::AngleAxisToRotationMatrix(pose.data(), raw_rotation);
        Eigen::Matrix3d rotation;
        rotation << raw_rotation[0], raw_rotation[3], raw_rotation[6],
            raw_rotation[1], raw_rotation[4], raw_rotation[7],
            raw_rotation[2], raw_rotation[5], raw_rotation[8];
        const Eigen::Vector3d translation(pose[3], pose[4], pose[5]);
        ViewError error;
        error.camera_id = view.camera_id;
        double view_squared = 0.0;
        double view_max_squared = 0.0;
        int counted = 0;
        for (std::size_t i = 0; i < view.board_points.size(); ++i) {
            const Eigen::Vector3d camera_point =
                rotation * Eigen::Vector3d(view.board_points[i].x(),
                                           view.board_points[i].y(), 0.0) +
                translation;
            double uv[2];
            if (!try_project_point<double>(type, width, height, intrinsics.data(),
                                           camera_point.data(), uv)) {
                continue;
            }
            const double dx = uv[0] - view.pixels[i].x();
            const double dy = uv[1] - view.pixels[i].y();
            const double squared = dx * dx + dy * dy;
            view_squared += squared;
            view_max_squared = std::max(view_max_squared, squared);
            ++counted;
        }
        error.corners = counted;
        error.rms_px =
            counted > 0 ? std::sqrt(view_squared / (2.0 * counted)) : 0.0;
        error.max_px = std::sqrt(view_max_squared);
        total_squared += view_squared;
        global_max_squared = std::max(global_max_squared, view_max_squared);
        total_corners += static_cast<std::size_t>(counted);
        out.views.push_back(error);
    }
    out.rms_px = total_corners > 0
                     ? std::sqrt(total_squared /
                                 (2.0 * static_cast<double>(total_corners)))
                     : 0.0;
    out.max_px = std::sqrt(global_max_squared);
    return out;
}

}  // namespace

CalibrationResult calibrate_camera(int image_width, int image_height,
                                   const std::vector<CalibrationView>& views,
                                   const Calibration& initial,
                                   const CalibrationFitFlags& fit,
                                   const CalibrateOptions& options) {
    if (image_width <= 0 || image_height <= 0) {
        throw ValueError("calibrate_camera: image size must be positive");
    }
    if (initial.type != CameraType::Frame) {
        throw ValueError(
            "calibrate_camera: closed-form initialisation needs a frame camera");
    }
    if (options.max_iterations < 0) {
        throw ValueError("calibrate_camera: max_iterations must be non-negative");
    }
    if (!(options.prune_t_ratio >= 0.0) || !std::isfinite(options.prune_t_ratio)) {
        throw ValueError("calibrate_camera: prune_t_ratio must be finite and >= 0");
    }
    std::vector<CalibrationView> usable;
    for (const CalibrationView& view : views) {
        if (view.board_points.size() < static_cast<std::size_t>(kMinimumCorners) ||
            view.board_points.size() != view.pixels.size()) {
            continue;
        }
        bool finite = true;
        for (std::size_t i = 0; i < view.board_points.size(); ++i) {
            if (!view.board_points[i].allFinite() || !view.pixels[i].allFinite()) {
                finite = false;
                break;
            }
        }
        if (finite) {
            usable.push_back(view);
        }
    }
    if (usable.size() < static_cast<std::size_t>(std::max(3, options.min_views))) {
        throw Error("calibrate_camera: need at least 3 views with a detected board");
    }
    InitialFit initial_fit;
    if (!initialise_views(usable, image_width, image_height, initial_fit)) {
        throw Error("calibrate_camera: closed-form initialisation failed");
    }
    const Packed start = initial_packed(initial, fit, initial_fit.focal);
    Refinement refinement;
    if (!solve_refinement(initial_fit.views, initial.type, image_width,
                          image_height, start, fit, initial_fit.poses,
                          options.max_iterations, refinement) ||
        !refinement.usable) {
        throw Error("calibrate_camera: refinement failed");
    }
    int residual_count = 0;
    for (const CalibrationView& view : initial_fit.views) {
        residual_count += static_cast<int>(2 * view.board_points.size());
    }
    int free_count = static_cast<int>(6 * initial_fit.views.size());
    for (int k = 0; k < kCalibrationParamCount; ++k) {
        if (fit[static_cast<std::size_t>(k)]) {
            ++free_count;
        }
    }
    std::array<double, kCalibrationParamCount> sigma{};
    Eigen::MatrixXd correlation;
    if (!covariance_outputs(refinement, fit, residual_count, free_count, sigma,
                            correlation)) {
        throw Error("calibrate_camera: covariance estimation failed");
    }
    CalibrationFitFlags final_fit = fit;
    std::vector<std::string> pruned;
    for (int k = 3; k < kCalibrationParamCount; ++k) {
        const std::size_t kk = static_cast<std::size_t>(k);
        if (!fit[kk]) {
            continue;
        }
        if (std::abs(refinement.intrinsics[kk]) < options.prune_t_ratio * sigma[kk]) {
            pruned.emplace_back(kCalibrationParamNames[kk]);
            refinement.intrinsics[kk] = 0.0;
            final_fit[kk] = false;
        }
    }
    if (!pruned.empty()) {
        free_count = static_cast<int>(6 * initial_fit.views.size());
        for (int k = 0; k < kCalibrationParamCount; ++k) {
            if (final_fit[static_cast<std::size_t>(k)]) {
                ++free_count;
            }
        }
        if (!solve_refinement(initial_fit.views, initial.type, image_width,
                              image_height, refinement.intrinsics, final_fit,
                              refinement.poses, options.max_iterations,
                              refinement) ||
            !refinement.usable ||
            !covariance_outputs(refinement, final_fit, residual_count, free_count,
                                sigma, correlation)) {
            throw Error("calibrate_camera: covariance estimation failed");
        }
    }
    CalibrationResult result;
    result.calibration = initial;
    result.calibration.width = image_width;
    result.calibration.height = image_height;
    unpack_calibration(refinement.intrinsics, result.calibration);
    const ErrorSummary errors =
        measure_errors(initial_fit.views, refinement.poses, initial.type,
                       image_width, image_height, refinement.intrinsics);
    result.rms_px = errors.rms_px;
    result.max_px = errors.max_px;
    result.views = errors.views;
    result.sigma = sigma;
    result.correlation = correlation;
    result.pruned = std::move(pruned);
    result.views_used = static_cast<int>(initial_fit.views.size());
    return result;
}

}  // namespace alphashape
