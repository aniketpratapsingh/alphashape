// similarity.cpp -- 3D similarity estimation. Clean-room from Umeyama (1991),
// "Least-squares estimation of transformation parameters between two point
// patterns", IEEE TPAMI 13(4):376-380; weighted variant with per-point weights
// folded into the means, covariance and scale; RANSAC (Fischler & Bolles 1981).

#include "common/algorithm/sfm/similarity.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "base/utils/portable_rng.hpp"

namespace alphashape::sfm {
namespace {

// Second singular value over the first, of the centred source columns. Below
// kCollinearRatio the points are treated as (near-)collinear and the estimate
// is refused. Centred point clouds are what R and scale are measured against,
// so the test is on the centred matrix, not the raw points.
constexpr double kCollinearRatio = 1e-6;

// SVD-based proper-rotation solve shared by both estimators. `cross` is the
// (weighted) centred cross-covariance sum y x^T, `sigma_x2` the (weighted) mean
// squared distance of the source to its mean. Writes R; returns the scale.
double solve_rotation_scale(const Eigen::Matrix3d& cross, double sigma_x2,
                            Eigen::Matrix3d& r_out) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cross, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d& u = svd.matrixU();
    const Eigen::Matrix3d& v = svd.matrixV();
    const Eigen::Vector3d d = svd.singularValues();

    Eigen::Matrix3d s = Eigen::Matrix3d::Identity();
    if ((u * v.transpose()).determinant() < 0.0) {
        s(2, 2) = -1.0;
    }
    r_out = u * s * v.transpose();

    // trace(D * S): D diag, S flips only the (2,2) entry when needed.
    double trace_ds = d(0) + d(1) + s(2, 2) * d(2);
    if (sigma_x2 <= 0.0) {
        trace_ds = 0.0;
    }
    return trace_ds / sigma_x2;
}

}  // namespace

Eigen::Vector3d Similarity::apply(const Eigen::Vector3d& p) const {
    return scale * (R * p) + t;
}

Eigen::Matrix4d Similarity::matrix() const {
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    m.topLeftCorner<3, 3>() = scale * R;
    m.topRightCorner<3, 1>() = t;
    return m;
}

Similarity Similarity::inverse() const {
    Similarity inv;
    inv.scale = 1.0 / scale;
    inv.R = R.transpose();
    inv.t = -inv.scale * (R.transpose() * t);
    return inv;
}

std::optional<Similarity> estimate_similarity(const std::vector<Eigen::Vector3d>& src,
                                              const std::vector<Eigen::Vector3d>& dst) {
    const Eigen::Index n = static_cast<Eigen::Index>(src.size());
    if (n < 3 || dst.size() != src.size()) {
        return std::nullopt;
    }

    Eigen::Vector3d mu_x = Eigen::Vector3d::Zero();
    Eigen::Vector3d mu_y = Eigen::Vector3d::Zero();
    for (Eigen::Index i = 0; i < n; ++i) {
        mu_x += src[static_cast<std::size_t>(i)];
        mu_y += dst[static_cast<std::size_t>(i)];
    }
    mu_x /= static_cast<double>(n);
    mu_y /= static_cast<double>(n);

    Eigen::Matrix3d cross = Eigen::Matrix3d::Zero();
    double sigma_x2 = 0.0;
    Eigen::Matrix<double, 3, Eigen::Dynamic> xs(3, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const Eigen::Vector3d xc = src[static_cast<std::size_t>(i)] - mu_x;
        const Eigen::Vector3d yc = dst[static_cast<std::size_t>(i)] - mu_y;
        xs.col(i) = xc;
        cross += yc * xc.transpose();
        sigma_x2 += xc.squaredNorm();
    }
    cross /= static_cast<double>(n);
    sigma_x2 /= static_cast<double>(n);

    // Collinearity test: second singular value / first singular value of the
    // centred source < 1e-6 -> refuse.
    const Eigen::JacobiSVD<Eigen::Matrix<double, 3, Eigen::Dynamic>> xsvd(
        xs, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::Vector3d xd = xsvd.singularValues();
    if (xd(0) <= 0.0 || xd(1) / xd(0) < kCollinearRatio) {
        return std::nullopt;
    }

    Similarity s;
    s.scale = solve_rotation_scale(cross, sigma_x2, s.R);
    s.t = mu_y - s.scale * (s.R * mu_x);
    return s;
}

std::optional<Similarity> estimate_similarity_weighted(const std::vector<Eigen::Vector3d>& src,
        const std::vector<Eigen::Vector3d>& dst, const std::vector<double>& weights) {
    const Eigen::Index n = static_cast<Eigen::Index>(src.size());
    if (n < 3 || dst.size() != src.size() || weights.size() != src.size()) {
        return std::nullopt;
    }

    double wsum = 0.0;
    for (double w : weights) {
        wsum += w;
    }
    if (wsum <= 0.0) {
        return std::nullopt;
    }

    Eigen::Vector3d mu_x = Eigen::Vector3d::Zero();
    Eigen::Vector3d mu_y = Eigen::Vector3d::Zero();
    for (Eigen::Index i = 0; i < n; ++i) {
        const double w = weights[static_cast<std::size_t>(i)];
        mu_x += w * src[static_cast<std::size_t>(i)];
        mu_y += w * dst[static_cast<std::size_t>(i)];
    }
    mu_x /= wsum;
    mu_y /= wsum;

    Eigen::Matrix3d cross = Eigen::Matrix3d::Zero();
    double sigma_x2 = 0.0;
    Eigen::Matrix<double, 3, Eigen::Dynamic> xs(3, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const double w = weights[static_cast<std::size_t>(i)];
        const Eigen::Vector3d xc = src[static_cast<std::size_t>(i)] - mu_x;
        const Eigen::Vector3d yc = dst[static_cast<std::size_t>(i)] - mu_y;
        xs.col(i) = std::sqrt(w) * xc;
        cross += w * (yc * xc.transpose());
        sigma_x2 += w * xc.squaredNorm();
    }
    cross /= wsum;
    sigma_x2 /= wsum;

    const Eigen::JacobiSVD<Eigen::Matrix<double, 3, Eigen::Dynamic>> xsvd(
        xs, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::Vector3d xd = xsvd.singularValues();
    if (xd(0) <= 0.0 || xd(1) / xd(0) < kCollinearRatio) {
        return std::nullopt;
    }

    Similarity s;
    s.scale = solve_rotation_scale(cross, sigma_x2, s.R);
    s.t = mu_y - s.scale * (s.R * mu_x);
    return s;
}

namespace {

// Inlier mask for a candidate similarity: |apply(src_i) - dst_i| < threshold.
std::vector<std::uint8_t> mark_inliers(const Similarity& sim,
                                       const std::vector<Eigen::Vector3d>& src,
                                       const std::vector<Eigen::Vector3d>& dst,
                                       double threshold) {
    std::vector<std::uint8_t> mask(src.size(), 0);
    for (std::size_t i = 0; i < src.size(); ++i) {
        if ((sim.apply(src[i]) - dst[i]).norm() < threshold) {
            mask[i] = 1;
        }
    }
    return mask;
}

std::size_t count_true(const std::vector<std::uint8_t>& v) {
    std::size_t c = 0;
    for (std::uint8_t b : v) {
        c += b;
    }
    return c;
}

}  // namespace

SimilarityRansacResult estimate_similarity_ransac(const std::vector<Eigen::Vector3d>& src,
        const std::vector<Eigen::Vector3d>& dst, double threshold, std::uint64_t seed,
        int max_iterations) {
    SimilarityRansacResult result;
    const std::size_t n = src.size();
    if (n < 3 || dst.size() != n) {
        return result;
    }

    if (n == 3) {
        result.similarity = estimate_similarity(src, dst);
        result.inliers.assign(n, 0);
        if (result.similarity) {
            result.inliers = mark_inliers(*result.similarity, src, dst, threshold);
            result.inlier_count = count_true(result.inliers);
        }
        return result;
    }

    std::mt19937_64 rng(seed);
    // Portable draws: the standard distributions are implementation-defined, so
    // a fixed seed chose different triples under libstdc++ and libc++
    // (see base/utils/portable_rng.hpp).
    std::vector<std::size_t> order(n, 0);

    std::optional<Similarity> best;
    std::size_t best_count = 0;
    std::vector<std::uint8_t> best_mask(n, 0);

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Draw 3 distinct indices.
        std::size_t idx[3];
        rng::distinct_indices(rng, n, 3, idx, order.data());
        const std::vector<Eigen::Vector3d> s3{src[idx[0]], src[idx[1]], src[idx[2]]};
        const std::vector<Eigen::Vector3d> d3{dst[idx[0]], dst[idx[1]], dst[idx[2]]};
        const std::optional<Similarity> cand = estimate_similarity(s3, d3);
        if (!cand) {
            continue;  // degenerate sample
        }
        const std::vector<std::uint8_t> mask = mark_inliers(*cand, src, dst, threshold);
        const std::size_t c = count_true(mask);
        if (c > best_count) {
            best_count = c;
            best = cand;
            best_mask = mask;
        }
    }

    // Final least-squares refit on the inliers of the best model.
    if (best) {
        std::vector<Eigen::Vector3d> isrc;
        std::vector<Eigen::Vector3d> idst;
        isrc.reserve(best_count);
        idst.reserve(best_count);
        for (std::size_t i = 0; i < n; ++i) {
            if (best_mask[i] != 0) {
                isrc.push_back(src[i]);
                idst.push_back(dst[i]);
            }
        }
        const std::optional<Similarity> refit = estimate_similarity(isrc, idst);
        if (refit) {
            best = refit;
        }
        result.similarity = best;
        result.inliers = mark_inliers(*best, src, dst, threshold);
        result.inlier_count = count_true(result.inliers);
    } else {
        result.inliers.assign(n, 0);
    }
    return result;
}

}  // namespace alphashape::sfm
