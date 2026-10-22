// essential.cpp -- relative-pose recovery from an essential matrix.
//
// Clean-room implementation from Hartley & Zisserman (Multiple View Geometry,
// 2nd ed., Ch. 9 and 12) and Fischler & Bolles (RANSAC, CACM 1981).  No vendor
// code or vendor constants are used; see essential.hpp for the statement that
// the RANSAC defaults and the parallax threshold are our own choices.

#include "common/algorithm/sfm/essential.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"

namespace alphashape::sfm {
namespace {

// kMinParallaxRad: our own threshold (documented in essential.hpp).  Two rays
// whose directions differ by less than this are treated as parallel and the
// triangulation is reported as unrecoverable (nullopt).  ~1e-4 rad keeps the
// mid-point estimate well conditioned while accepting ordinary near-degenerate
// pairs; it is not a value copied from any vendor implementation.
constexpr double kMinParallaxRad = 1e-4;

constexpr double kPi = 3.14159265358979323846;

}  // namespace

Eigen::Matrix3d essential_from_pose(const RelativePose& pose) {
    Eigen::Matrix3d tx;
    const Eigen::Vector3d& t = pose.t;
    tx << 0.0, -t.z(), t.y(), t.z(), 0.0, -t.x(), -t.y(), t.x(), 0.0;
    return tx * pose.R;
}

double epipolar_angle_error(const Eigen::Matrix3d& E, const Eigen::Vector3d& b1,
                            const Eigen::Vector3d& b2) {
    const Eigen::Vector3d n1_raw = E * b1;
    const Eigen::Vector3d n2_raw = E.transpose() * b2;

    double a1 = kPi / 2.0;
    if (n1_raw.norm() > 1e-300) {
        const Eigen::Vector3d n1 = n1_raw.normalized();
        const double s = std::min(1.0, std::abs(b2.dot(n1)));
        a1 = std::asin(s);
    }
    double a2 = kPi / 2.0;
    if (n2_raw.norm() > 1e-300) {
        const Eigen::Vector3d n2 = n2_raw.normalized();
        const double s = std::min(1.0, std::abs(b1.dot(n2)));
        a2 = std::asin(s);
    }
    return std::max(a1, a2);
}

std::optional<Eigen::Vector3d> triangulate(const RelativePose& pose, const Eigen::Vector3d& b1,
                                           const Eigen::Vector3d& b2) {
    const Eigen::Matrix3d Rt = pose.R.transpose();
    const Eigen::Vector3d c2 = -Rt * pose.t;  // camera-2 centre in camera-1 coords
    const Eigen::Vector3d d1 = b1;
    const Eigen::Vector3d d2 = Rt * b2;

    // Mid-point: P(s) = s*d1 (camera-1 ray), Q(u) = c2 + u*d2 (camera-2 ray).
    // Least squares on |P(s) - Q(u)|^2 gives
    //   [ a  -b ] [s]   [w1]        a = d1.d1, b = d1.d2, c = d2.d2,
    //   [ b  -c ] [u] = [w2],       w1 = d1.c2, w2 = d2.c2.
    const double a = d1.dot(d1);
    const double b = d1.dot(d2);
    const double c = d2.dot(d2);
    const double w1 = d1.dot(c2);
    const double w2 = d2.dot(c2);

    // Parallax: the angle between the ray directions (accounting for the sign of
    // the d2 term, since the rays diverge).
    const double cos_ang = std::abs(b) / std::sqrt(a * c);
    if (std::acos(std::min(1.0, cos_ang)) < kMinParallaxRad) {
        return std::nullopt;
    }

    const double det = b * b - a * c;  // = -(a c - b^2) <= 0, zero iff parallel
    if (det == 0.0) {
        return std::nullopt;
    }
    const double s = (b * w2 - c * w1) / det;
    const double u = (a * w2 - b * w1) / det;
    return (s * d1 + c2 + u * d2) * 0.5;
}

std::optional<RelativePose> decompose_essential(const Eigen::Matrix3d& E,
                                                const std::vector<Eigen::Vector3d>& b1,
                                                const std::vector<Eigen::Vector3d>& b2,
                                                const std::vector<std::uint8_t>* mask) {
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // Force det(U) = det(V) = +1 by flipping the sign of the third column
    // (which is scaled by the zero singular value, so E is unchanged).
    if (U.determinant() < 0.0) {
        U.col(2) = -U.col(2);
    }
    if (V.determinant() < 0.0) {
        V.col(2) = -V.col(2);
    }

    Eigen::Matrix3d W;
    W << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;

    const Eigen::Matrix3d R1 = U * W * V.transpose();
    const Eigen::Matrix3d R2 = U * W.transpose() * V.transpose();
    const Eigen::Vector3d u3 = U.col(2);

    const std::array<RelativePose, 4> candidates = {{
        RelativePose{R1, u3},
        RelativePose{R1, -u3},
        RelativePose{R2, u3},
        RelativePose{R2, -u3},
    }};

    const auto in_front_count = [&](const RelativePose& pose) -> std::size_t {
        std::size_t count = 0;
        for (std::size_t i = 0; i < b1.size(); ++i) {
            if (mask != nullptr && (*mask)[i] == 0) {
                continue;
            }
            const std::optional<Eigen::Vector3d> x = triangulate(pose, b1[i], b2[i]);
            if (!x.has_value()) {
                continue;
            }
            const Eigen::Vector3d x2 = pose.R * (*x) + pose.t;
            if (x->dot(b1[i]) > 0.0 && x2.dot(b2[i]) > 0.0) {
                ++count;
            }
        }
        return count;
    };

    std::size_t best = 0;
    std::size_t best_idx = 0;
    for (std::size_t c = 0; c < candidates.size(); ++c) {
        const std::size_t n = in_front_count(candidates[c]);
        if (n > best) {
            best = n;
            best_idx = c;
        }
    }
    if (best == 0) {
        return std::nullopt;
    }
    return candidates[best_idx];
}

namespace {

// Project an arbitrary 3x3 to the nearest essential matrix: force the two
// (equal) nonzero singular values and a zero third, per H&Z Ch. 9.
Eigen::Matrix3d project_to_essential(const Eigen::Matrix3d& m) {
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    const Eigen::Vector3d sv = svd.singularValues();
    const double mean = (sv(0) + sv(1)) * 0.5;
    Eigen::Matrix3d d = Eigen::Matrix3d::Zero();
    d(0, 0) = mean;
    d(1, 1) = mean;
    if (U.determinant() * V.determinant() < 0.0) {
        d(2, 2) = 0.0;
        d(0, 0) = -mean;  // keep E = U diag V^T a valid essential matrix (det fix)
    }
    return U * d * V.transpose();
}

// Linear least-squares essential matrix from >= 8 correspondences: rows are
// kron(b2, b1) == row-major vec(b2 * b1^T), matching five_point.cpp's ordering.
Eigen::Matrix3d fit_essential_linear(const std::vector<Eigen::Vector3d>& b1,
                                     const std::vector<Eigen::Vector3d>& b2,
                                     const std::vector<std::uint8_t>& inliers) {
    const Eigen::Index n = static_cast<Eigen::Index>(b1.size());
    // Size A to the actual number of inlier rows. Allocating it at the input
    // size and filling only the inlier rows would leave the trailing rows
    // uninitialised, and JacobiSVD below would read that garbage: the smallest
    // right singular vector is then process-dependent (heap/ASLR), which broke
    // run-to-run determinism.
    Eigen::Index rows = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        rows += inliers[static_cast<std::size_t>(i)] != 0 ? 1 : 0;
    }
    if (rows < 8) {
        return Eigen::Matrix3d::Zero();
    }
    Eigen::MatrixXd A(rows, 9);
    Eigen::Index row = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        const auto s = static_cast<std::size_t>(i);
        if (inliers[s] == 0) {
            continue;
        }
        for (int u = 0; u < 3; ++u) {
            for (int v = 0; v < 3; ++v) {
                A(row, static_cast<Eigen::Index>(u * 3 + v)) =
                    b2[s](static_cast<Eigen::Index>(u)) * b1[s](static_cast<Eigen::Index>(v));
            }
        }
        ++row;
    }
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::VectorXd vec = svd.matrixV().col(8);
    Eigen::Matrix3d E;
    for (int u = 0; u < 3; ++u) {
        for (int v = 0; v < 3; ++v) {
            E(static_cast<Eigen::Index>(u), static_cast<Eigen::Index>(v)) =
                vec(static_cast<Eigen::Index>(u * 3 + v));
        }
    }
    return project_to_essential(E);
}

}  // namespace

EssentialResult estimate_essential_ransac(const std::vector<Eigen::Vector3d>& b1,
                                          const std::vector<Eigen::Vector3d>& b2,
                                          const RansacOptions& options) {
    EssentialResult result;
    if (b1.size() != b2.size()) {
        throw ValueError("estimate_essential_ransac: b1 and b2 size mismatch");
    }
    if (options.threshold_rad <= 0.0) {
        throw ValueError("estimate_essential_ransac: threshold_rad must be positive");
    }
    const std::size_t m = b1.size();
    result.inliers.assign(m, 0);
    if (m < 5) {
        return result;  // success stays false
    }

    const double thresh = options.threshold_rad;
    const int max_iter = options.max_iterations;
    const int min_iter = options.min_iterations;
    const double conf = options.confidence;

    std::mt19937_64 rng(options.seed);

    const auto run_model = [&](const Eigen::Matrix3d& E, std::size_t& count,
                               double& sum_err) -> std::vector<std::uint8_t> {
        std::vector<std::uint8_t> inl(m, 0);
        count = 0;
        sum_err = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            const double e = epipolar_angle_error(E, b1[i], b2[i]);
            if (e < thresh) {
                inl[i] = 1;
                ++count;
                sum_err += e;
            }
        }
        return inl;
    };

    Eigen::Matrix3d best_E = Eigen::Matrix3d::Zero();
    std::vector<std::uint8_t> best_inliers(m, 0);
    std::size_t best_count = 0;
    double best_sum = std::numeric_limits<double>::infinity();
    int iterations = 0;
    int k = max_iter;  // required iterations, adaptively updated

    // Portable sampling: std::uniform_int_distribution's algorithm is
    // implementation-defined, so the same seed drew different samples under
    // libstdc++ and libc++ (see base/utils/portable_rng.hpp).
    std::vector<std::size_t> order(m, 0);

    while (iterations < k && iterations < max_iter) {
        ++iterations;
        std::array<std::size_t, 5> idx{};
        rng::distinct_indices(rng, m, 5, idx.data(), order.data());
        std::array<Eigen::Vector3d, 5> s1{};
        std::array<Eigen::Vector3d, 5> s2{};
        for (int j = 0; j < 5; ++j) {
            const std::size_t s = idx[static_cast<std::size_t>(j)];
            s1[static_cast<std::size_t>(j)] = b1[s];
            s2[static_cast<std::size_t>(j)] = b2[s];
        }

        const std::vector<Eigen::Matrix3d> models = five_point(s1, s2);
        for (const Eigen::Matrix3d& E : models) {
            std::size_t count = 0;
            double sum_err = 0.0;
            std::vector<std::uint8_t> inl = run_model(E, count, sum_err);
            const bool better =
                count > best_count || (count == best_count && sum_err < best_sum);
            if (count > 0 && better) {
                best_count = count;
                best_sum = sum_err;
                best_E = E;
                best_inliers = inl;

                const double w = static_cast<double>(count) / static_cast<double>(m);
                const double w5 = w * w * w * w * w;
                if (w5 >= 1.0 - 1e-12) {
                    k = min_iter;
                } else if (w5 <= 0.0) {
                    k = max_iter;
                } else {
                    const double denom = std::log(1.0 - w5);
                    const double num = std::log(1.0 - conf);
                    const double kd = std::ceil(num / denom);
                    if (kd < static_cast<double>(min_iter)) {
                        k = min_iter;
                    } else if (kd > static_cast<double>(max_iter)) {
                        k = max_iter;
                    } else {
                        k = static_cast<int>(kd);
                    }
                }
            }
        }
    }
    result.iterations = iterations;

    if (best_count == 0) {
        return result;  // success stays false
    }
    result.E = best_E;
    result.inliers = best_inliers;

    // Local refinement: up to 3 rounds of linear refit on the current inliers.
    for (int round = 0; round < 3; ++round) {
        std::size_t prev = 0;
        for (std::uint8_t v : result.inliers) {
            prev += v;
        }
        if (prev < 8) {
            break;
        }
        const Eigen::Matrix3d refit = fit_essential_linear(b1, b2, result.inliers);
        if (refit.isZero(0.0)) {
            break;
        }
        std::size_t count = 0;
        double sum_err = 0.0;
        std::vector<std::uint8_t> inl = run_model(refit, count, sum_err);
        if (count >= prev) {
            result.E = refit;
            result.inliers = inl;
        }
    }

    // Pose from the final essential matrix, then cheirality-consistent cleanup.
    const std::optional<RelativePose> pose =
        decompose_essential(result.E, b1, b2, &result.inliers);
    if (!pose.has_value()) {
        return result;  // success stays false
    }
    result.pose = *pose;

    std::size_t final_count = 0;
    for (std::size_t i = 0; i < m; ++i) {
        if (result.inliers[i] == 0) {
            continue;
        }
        const std::optional<Eigen::Vector3d> x =
            triangulate(result.pose, b1[i], b2[i]);
        if (!x.has_value()) {
            ++final_count;  // low parallax: keep as an inlier
            continue;
        }
        const Eigen::Vector3d x2 = result.pose.R * (*x) + result.pose.t;
        if (x->dot(b1[i]) > 0.0 && x2.dot(b2[i]) > 0.0) {
            ++final_count;
        } else {
            result.inliers[i] = 0;
        }
    }
    result.inlier_count = final_count;
    result.success = final_count >= 5;
    return result;
}

}  // namespace alphashape::sfm
