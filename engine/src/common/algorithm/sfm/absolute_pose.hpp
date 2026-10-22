#pragma once

// absolute_pose.hpp -- absolute camera pose (PnP): the minimal 3-point solver
// P3P, a pose-only refinement, and an absolute-pose RANSAC.
//
// Clean-room implementation from the published literature only:
//   * M. Persson & K. Nordberg, "Lambda Twist: An Accurate Fast Robust
//     Perspective Three Point (P3P) Solver", ECCV 2018 -- the P3P solver
//     (pose-invariant quadratic constraints, the cubic in gamma that makes a
//     rank-2 linear combination, the lambda/ tau parameterisation, and the
//     Y = R X recovery).
//   * Fischler & Bolles, "Random Sample Consensus", CACM 1981 -- the RANSAC
//     loop, inlier counting and the adaptive stopping rule
//     k = ceil(log(1-p) / log(1-w^3)).
//
// Engineering constants (our own choices, not vendor values): the pose
// refinement runs up to 10 Levenberg-Marquardt iterations and stops when the
// relative cost change drops below 1e-10, and RANSAC requires at least 4
// inliers for success. These are conservative values appropriate to
// structure-from-motion initialisation.
//
// Conventions:
//   * Bearings are unit vectors in the camera frame (+X right, +Y down,
//     +Z forward). They may be more than 90 degrees off-axis because fisheye
//     cameras are supported.
//   * The world->camera pose is X_cam = R X_world + t.
//   * A point is in front of a camera iff (R X + t).dot(bearing) > 0.

#include "common/algorithm/sfm/essential.hpp"  // RansacOptions

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace alphashape::sfm {

struct AbsolutePose {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

/// Minimal solver: 3 world points and their unit bearings -> up to 4 poses.
std::vector<AbsolutePose> p3p(const std::array<Eigen::Vector3d, 3>& world,
                              const std::array<Eigen::Vector3d, 3>& bearings);

/// Angle (radians, in [0, pi]) between `bearing` and normalize(R*world + t);
/// pi if R*world + t == 0.
double reprojection_angle_error(const AbsolutePose& pose, const Eigen::Vector3d& world,
                                const Eigen::Vector3d& bearing);

/// Pose-only refinement: Gauss-Newton / LM over the 6-dof pose minimising the
/// sum of squared tangent-plane residuals of the given correspondences (ours:
/// up to 10 iterations, stop on relative cost change < 1e-10). Returns the
/// refined pose; never worse than the input in cost.
AbsolutePose refine_absolute_pose(const AbsolutePose& initial,
                                  const std::vector<Eigen::Vector3d>& world,
                                  const std::vector<Eigen::Vector3d>& bearings);

struct AbsolutePoseResult {
    bool success = false;
    AbsolutePose pose;
    std::vector<std::uint8_t> inliers;  // size == input, 1 = inlier
    std::size_t inlier_count = 0;
    int iterations = 0;
};

/// RANSAC over p3p with adaptive stopping k = log(1-p)/log(1-w^3), clamped to
/// [min_iterations, max_iterations]. Deterministic std::mt19937_64(options.seed).
/// Inlier: reprojection_angle_error < threshold AND in front. After the loop:
/// refine_absolute_pose on the inliers, recompute inliers, keep the refined pose
/// if not worse. Sizes differ or threshold <= 0 => ValueError; < 3 points =>
/// success = false; success requires inlier_count >= 4.
AbsolutePoseResult estimate_absolute_pose_ransac(const std::vector<Eigen::Vector3d>& world,
                                                 const std::vector<Eigen::Vector3d>& bearings,
                                                 const RansacOptions& options);

}  // namespace alphashape::sfm
