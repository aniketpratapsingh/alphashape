#pragma once

// similarity.hpp -- 3D similarity (scale + rotation + translation) estimation.
//
// Clean-room implementation from the published literature only:
//   * Umeyama (1991), "Least-squares estimation of transformation parameters
//     between two point patterns", IEEE TPAMI 13(4):376-380. The SVD solution
//     with the determinant-correction step that forces a proper rotation
//     (det = +1) and so never returns a reflection.
//   * The weighted least-squares variant uses the same construction with
//     per-point weights folded into the centred cross-covariance and the scale.
//   * RANSAC (Fischler & Bolles 1981) over minimal 3-point samples, with a
//     deterministic mt19937_64 and a final least-squares refit on the inliers.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace alphashape::sfm {

/// A 3D similarity transform: dst = scale * R * src + t, with R a proper
/// rotation (det = +1).
struct Similarity {
    double scale = 1.0;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();

    Eigen::Vector3d apply(const Eigen::Vector3d& p) const;
    Eigen::Matrix4d matrix() const;  ///< homogeneous 4x4
    Similarity inverse() const;
};

/// Umeyama least squares. Returns nullopt if fewer than 3 points, if the two
/// inputs differ in size, or if the (centred) source points are (near-)collinear
/// -- tested as: second singular value of the centred source / first singular
/// value < 1e-6. The rotation is a proper rotation (det = +1 enforced).
std::optional<Similarity> estimate_similarity(const std::vector<Eigen::Vector3d>& src,
                                              const std::vector<Eigen::Vector3d>& dst);

/// Weighted variant: per-point weights >= 0 (Umeyama with weights). A weight of
/// zero removes that point from the estimate entirely.
std::optional<Similarity> estimate_similarity_weighted(const std::vector<Eigen::Vector3d>& src,
        const std::vector<Eigen::Vector3d>& dst, const std::vector<double>& weights);

struct SimilarityRansacResult {
    std::optional<Similarity> similarity;
    std::vector<std::uint8_t> inliers;
    std::size_t inlier_count = 0;
};

/// RANSAC over minimal 3-point samples, inlier if |apply(src)-dst| < threshold;
/// deterministic std::mt19937_64(seed); final least-squares refit on inliers.
/// If n <= 3, plain estimate.
SimilarityRansacResult estimate_similarity_ransac(const std::vector<Eigen::Vector3d>& src,
        const std::vector<Eigen::Vector3d>& dst, double threshold, std::uint64_t seed = 0x5eed,
        int max_iterations = 2000);

}  // namespace alphashape::sfm
