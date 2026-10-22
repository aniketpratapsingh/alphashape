#pragma once

// triangulation.hpp -- multi-view linear triangulation from bearing vectors.
//
// Clean-room implementation from the published literature only:
//   * Hartley & Zisserman, "Multiple View Geometry in Computer Vision", 2nd ed.
//     -- the direct linear transform (DLT): each observation contributes the
//     rows of [b]_x (R X + t) = 0, the homogeneous system is solved by SVD and
//     dehomogenised.
//
// Engineering constants (our own choices, not vendor values): the default
// minimum parallax of 1.5 degrees and the optional per-observation reprojection
// angle limit are our own thresholds for accepting a triangulated point. They
// are not taken from any vendor or reference implementation.
//
// Conventions (shared with absolute_pose.hpp): bearings are unit vectors in the
// camera frame, the pose is X_cam = R X_world + t, and a point is in front iff
// (R X + t).dot(bearing) > 0.

#include "common/algorithm/sfm/absolute_pose.hpp"  // AbsolutePose

#include <optional>
#include <vector>

#include <Eigen/Core>

namespace alphashape::sfm {

struct TriangulationOptions {
    double min_parallax_rad = 1.5 * 3.14159265358979 / 180.0;  ///< ours
    double max_angle_error_rad = 0.0;  ///< 0 = no check; else every observation
                                       ///< must have reprojection angle < this
};

/// N-view linear triangulation (DLT on bearings): each observation contributes
/// the rows of [b]_x (R X + t) = 0 (we use all three rows per observation; the
/// third is a linear combination and only improves the SVD conditioning). Solve
/// the homogeneous system by SVD, dehomogenise (reject w ~ 0). Then require:
/// positive depth in every view, max pairwise parallax >= min_parallax_rad, and
/// the optional angle check. nullopt on failure.
std::optional<Eigen::Vector3d> triangulate_multiview(const std::vector<AbsolutePose>& poses,
                                                     const std::vector<Eigen::Vector3d>& bearings,
                                                     const TriangulationOptions& options);

/// Max angle between the viewing rays of X from the given camera centres
/// (C = -R^T t), radians.
double max_parallax(const std::vector<AbsolutePose>& poses, const Eigen::Vector3d& X);

}  // namespace alphashape::sfm
