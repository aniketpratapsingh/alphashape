#pragma once

// essential.hpp -- relative pose from an essential matrix.
//
// Clean-room implementation from the literature only:
//   * Hartley & Zisserman, "Multiple View Geometry in Computer Vision",
//     2nd ed. -- Ch. 9 (epipolar geometry, E decomposition into four
//     candidate (R,t) pairs via SVD with the det-fixing trick) and Ch. 12
//     (mid-point triangulation).
//   * Fischler & Bolles, "Random Sample Consensus" (CACM 1981) -- the
//     RANSAC loop, inlier counting and the adaptive iteration-count
//     stopping rule  k = ceil(log(1-p) / log(1-w^s)).
//
// Engineering constants: the RANSAC defaults confidence = 0.999,
// max_iterations = 10000 and min_iterations = 50 are our own conservative
// choices for bundle-adjustment-grade two-view initialisation, not values
// taken from any vendor. kMinParallaxRad is likewise our own threshold,
// chosen small enough to accept well-conditioned near-degenerate pairs
// while rejecting true ray parallelism (see essential.cpp).

#include "common/algorithm/sfm/five_point.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Core>

namespace alphashape::sfm {

/// X2 = R*X1 + t (camera-1 coords to camera-2 coords). ||t|| = 1.
struct RelativePose {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::UnitX();
};

Eigen::Matrix3d essential_from_pose(const RelativePose& pose);  // [t]_x R

/// Radians: max of angle(b2, plane with normal E*b1) and angle(b1, plane with
/// normal E^T*b2) = max(asin(|b2.n1|), asin(|b1.n2|)) with n1 = normalize(E b1),
/// n2 = normalize(E^T b2). Clamp the asin argument to [0,1]; if a normal is ~0
/// return pi/2.
double epipolar_angle_error(const Eigen::Matrix3d& E, const Eigen::Vector3d& b1,
                            const Eigen::Vector3d& b2);

/// Mid-point triangulation in camera-1 coordinates. Rays: camera1 centre 0,
/// dir b1; camera2 centre C2 = -R^T t, dir R^T b2. Returns nullopt if the angle
/// between the two rays is below kMinParallaxRad (ours, ~1e-4 rad, documented).
std::optional<Eigen::Vector3d> triangulate(const RelativePose& pose, const Eigen::Vector3d& b1,
                                           const Eigen::Vector3d& b2);

/// Cheirality: X in front of camera 1 iff X.dot(b1) > 0; in front of camera 2
/// iff (R X + t).dot(b2) > 0. Four candidates from SVD E = U diag(1,1,0) V^T
/// (force det(U)=det(V)=+1 by negating), W = [[0,-1,0],[1,0,0],[0,0,1]]:
/// R in {U W V^T, U W^T V^T}, t in {+u3, -u3}. Pick the candidate with most
/// points in front of both (only where mask[i]!=0 if mask given); nullopt if
/// the best count is 0.
std::optional<RelativePose> decompose_essential(const Eigen::Matrix3d& E,
                                                const std::vector<Eigen::Vector3d>& b1,
                                                const std::vector<Eigen::Vector3d>& b2,
                                                const std::vector<std::uint8_t>* mask = nullptr);

struct RansacOptions {
    double threshold_rad = 0.0;
    double confidence = 0.999;
    int max_iterations = 10000;
    int min_iterations = 50;
    std::uint64_t seed = 0x5eed;
};

struct EssentialResult {
    bool success = false;
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
    RelativePose pose;
    std::vector<std::uint8_t> inliers;  // size == input, 1 = inlier
    std::size_t inlier_count = 0;
    int iterations = 0;
};

EssentialResult estimate_essential_ransac(const std::vector<Eigen::Vector3d>& b1,
                                          const std::vector<Eigen::Vector3d>& b2,
                                          const RansacOptions& options);

}  // namespace alphashape::sfm
