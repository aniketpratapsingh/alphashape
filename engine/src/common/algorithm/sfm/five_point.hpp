#pragma once
#include <array>
#include <vector>
#include <Eigen/Core>
namespace alphashape::sfm {
/// b1[i], b2[i]: unit bearing vectors of the same scene point in cameras 1 and 2.
/// Returns every real essential matrix (0..10) with b2[i]^T E b1[i] = 0, each scaled to ||E||_F = 1.
std::vector<Eigen::Matrix3d> five_point(const std::array<Eigen::Vector3d, 5>& b1,
                                        const std::array<Eigen::Vector3d, 5>& b2);
}
