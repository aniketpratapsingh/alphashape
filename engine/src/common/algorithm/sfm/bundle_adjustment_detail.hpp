#pragma once

/// @file bundle_adjustment_detail.hpp
/// @brief Test-visible helpers of the bundle-adjustment internals.
///
/// These are not part of the public solve contract. They are exposed so unit
/// tests can pin the near-pi log map and the heading frame directly, without
/// reconstructing them through a full solve.

#include <Eigen/Core>

namespace alphashape::sfm::detail {

/// Log map of a rotation matrix, `theta * axis`, for a proper rotation.
///
/// This is the same routine `RotationPriorCost` uses (the double-precision
/// instantiation), in the same 9-element column-major layout. Near pi the
/// skew part vanishes and the axis is rebuilt from the symmetric part; the
/// unit test pins that branch, which a generic residual at truth cannot see.
Eigen::Vector3d rotation_log_matrix(const Eigen::Matrix3d& rotation);

/// The heading frame of a measured body attitude, in the frame `R_wb` maps
/// into (the slot CRS's local east/north/up basis).
///
/// The columns are (forward-level, left-level, up): `up` is ENU +z, forward
/// is the body forward axis `R_wb * e_x` projected onto the horizontal plane
/// and normalised, and left is `up x forward`. When the body forward axis is
/// within 1e-3 of vertical it carries no heading, so the most horizontal body
/// axis is used instead (body down, `-R_wb * e_z`, is tried first). In that
/// degenerate case yaw and roll are not separable around this frame; the
/// caller must give them the same sigma. `degenerate` reports which branch
/// ran (it may be null).
Eigen::Matrix3d heading_frame_from_body(const Eigen::Matrix3d& world_from_body,
                                        bool* degenerate = nullptr);

}  // namespace alphashape::sfm::detail
