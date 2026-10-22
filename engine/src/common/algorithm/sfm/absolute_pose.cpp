// absolute_pose.cpp -- Lambda Twist P3P (Persson & Nordberg, ECCV 2018),
// pose-only LM refinement, and RANSAC (Fischler & Bolles, CACM 1981).
// Clean-room implementation from the published equations only.

#include "common/algorithm/sfm/absolute_pose.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"

namespace alphashape::sfm {
namespace {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

Mat3 skew(const Vec3& v) {
    Mat3 s;
    s << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
    return s;
}

// Exp map of a rotation vector, accurate near zero (Rodrigues, Taylor near 0).
Mat3 exp_so3(const Vec3& w) {
    const double theta2 = w.squaredNorm();
    if (theta2 < 1e-16) {
        return Mat3::Identity() + skew(w);
    }
    const double theta = std::sqrt(theta2);
    const Mat3 W = skew(w);
    return Mat3::Identity() + (std::sin(theta) / theta) * W +
           ((1.0 - std::cos(theta)) / theta2) * (W * W);
}

constexpr double kPi = 3.14159265358979323846;

// Smallest-magnitude real root of c3 g^3 + c2 g^2 + c1 g + c0 = 0 (Sec. 3.4).
double real_cubic_root(double c3, double c2, double c1, double c0) {
    if (std::abs(c3) < 1e-14) {
        if (std::abs(c2) < 1e-14) {
            return std::abs(c1) < 1e-14 ? 0.0 : -c0 / c1;
        }
        const double disc = c1 * c1 - 4.0 * c2 * c0;
        if (disc < 0.0) {
            return -c1 / (2.0 * c2);  // complex conjugate: return the real part
        }
        const double sq = std::sqrt(disc);
        const double r1 = (-c1 + sq) / (2.0 * c2);
        const double r2 = (-c1 - sq) / (2.0 * c2);
        return std::abs(r1) < std::abs(r2) ? r1 : r2;
    }
    const double b = c2 / c3;
    const double c = c1 / c3;
    const double d = c0 / c3;
    const double p = c - b * b / 3.0;
    const double q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d;
    const double half_q = 0.5 * q;
    const double disc = half_q * half_q + p * p * p / 27.0;
    double y = 0.0;
    if (disc >= 0.0) {
        const double sq = std::sqrt(disc);
        const double u = std::cbrt(-half_q + sq);
        const double v = std::cbrt(-half_q - sq);
        y = u + v;
    } else {
        const double r = std::sqrt(-p * p * p / 27.0);
        const double phi = std::acos(std::clamp(-half_q / r, -1.0, 1.0));
        const double m = 2.0 * std::sqrt(-p / 3.0);
        y = m * std::cos(phi / 3.0);
        const double y2 = m * std::cos((phi + 2.0 * kPi) / 3.0);
        const double y3 = m * std::cos((phi + 4.0 * kPi) / 3.0);
        if (std::abs(y2) < std::abs(y)) {
            y = y2;
        }
        if (std::abs(y3) < std::abs(y)) {
            y = y3;
        }
    }
    return y - b / 3.0;
}

// Eigendecomposition of a symmetric 3x3 known to have a zero eigenvalue
// (paper Sec. 3.5). The zero eigenvalue is identified as the one of smallest
// magnitude; the two non-zero eigenvalues are ordered so |sigma1| <= |sigma2|,
// matching the reordering in Algorithm 2.
struct Eig3 {
    Mat3 E = Mat3::Identity();
    double sigma1 = 0.0;
    double sigma2 = 0.0;
};

Eig3 eig3x3_known0(const Mat3& M) {
    Eig3 out;
    Eigen::SelfAdjointEigenSolver<Mat3> es(M);
    if (es.info() != Eigen::Success) {
        return out;  // sigma1 == sigma2 == 0 signals failure to the caller
    }
    const Eigen::Vector3d vals = es.eigenvalues();  // increasing order
    const Mat3 vecs = es.eigenvectors();
    int zero_i = 0;
    for (int i = 1; i < 3; ++i) {
        if (std::abs(vals[i]) < std::abs(vals[zero_i])) {
            zero_i = i;
        }
    }
    int i1 = -1;
    int i2 = -1;
    for (int i = 0; i < 3; ++i) {
        if (i == zero_i) {
            continue;
        }
        if (i1 < 0) {
            i1 = i;
        } else {
            i2 = i;
        }
    }
    if (std::abs(vals[i1]) > std::abs(vals[i2])) {
        std::swap(i1, i2);
    }
    out.E.col(0) = vecs.col(i1);
    out.E.col(1) = vecs.col(i2);
    out.E.col(2) = vecs.col(zero_i);
    out.sigma1 = vals[i1];
    out.sigma2 = vals[i2];
    return out;
}

// Pairwise geometry for the three point pairs {12, 13, 23} (Eqn 3): the squared
// 3D distance a_ij and the bearing cosine b_ij.
struct PairData {
    double a = 0.0;
    double b = 0.0;
};

// Evaluates the Lambda Twist constraint residuals (4) in the lambda-space.
double lambda_residual(const Vec3& L, double a12, double a13, double a23,
                       double b12, double b13, double b23) {
    const double r0 = L.x() * L.x() + L.y() * L.y() - 2.0 * b12 * L.x() * L.y() - a12;
    const double r1 = L.x() * L.x() + L.z() * L.z() - 2.0 * b13 * L.x() * L.z() - a13;
    const double r2 = L.y() * L.y() + L.z() * L.z() - 2.0 * b23 * L.y() * L.z() - a23;
    return r0 * r0 + r1 * r1 + r2 * r2;
}

// Gauss-Newton refinement of the signed distances Lambda (Sec. 3.8): minimise
// the sum of squares of the three constraint residuals. Two verified steps.
Vec3 refine_lambda(const Vec3& L0, double a12, double a13, double a23,
                   double b12, double b13, double b23) {
    Vec3 L = L0;
    double cost = lambda_residual(L, a12, a13, a23, b12, b13, b23);
    for (int it = 0; it < 2; ++it) {
        const double l1 = L.x(), l2 = L.y(), l3 = L.z();
        const double r0 = l1 * l1 + l2 * l2 - 2.0 * b12 * l1 * l2 - a12;
        const double r1 = l1 * l1 + l3 * l3 - 2.0 * b13 * l1 * l3 - a13;
        const double r2 = l2 * l2 + l3 * l3 - 2.0 * b23 * l2 * l3 - a23;
        Eigen::Matrix<double, 3, 3> J;
        J << 2.0 * l1 - 2.0 * b12 * l2, 2.0 * l2 - 2.0 * b12 * l1, 0.0,
             2.0 * l1 - 2.0 * b13 * l3, 0.0, 2.0 * l3 - 2.0 * b13 * l1,
             0.0, 2.0 * l2 - 2.0 * b23 * l3, 2.0 * l3 - 2.0 * b23 * l2;
        const Eigen::Vector3d r(r0, r1, r2);
        Eigen::Matrix<double, 3, 3> JtJ = J.transpose() * J;
        JtJ.diagonal().array() += 1e-12;  // Levenberg damping
        const Vec3 delta = JtJ.ldlt().solve(-J.transpose() * r);
        const Vec3 candidate = L + delta;
        const double cand_cost = lambda_residual(candidate, a12, a13, a23, b12, b13, b23);
        if (cand_cost <= cost) {
            L = candidate;
            cost = cand_cost;
        } else {
            break;
        }
    }
    return L;
}

// Lambda Twist Sec. 3.3-3.6: up to four geometrically feasible Lambda vectors.
std::vector<Vec3> p3p_lambda_solutions(const std::array<Vec3, 3>& x,
                                       const std::array<Vec3, 3>& y) {
    std::vector<Vec3> lambdas;
    const PairData p12{(x[0] - x[1]).squaredNorm(), y[0].dot(y[1])};
    const PairData p13{(x[0] - x[2]).squaredNorm(), y[0].dot(y[2])};
    const PairData p23{(x[1] - x[2]).squaredNorm(), y[1].dot(y[2])};
    if (p12.a <= 0.0 || p13.a <= 0.0 || p23.a <= 0.0) {
        return lambdas;  // coincident world points: degenerate
    }
    // The M-matrix for pair ij (Eqn 4): unit on the i,j diagonal, -b on (i,j).
    const auto make_m = [](int i, int j, double b) {
        Mat3 m = Mat3::Zero();
        m(i, i) = 1.0;
        m(j, j) = 1.0;
        m(i, j) = -b;
        m(j, i) = -b;
        return m;
    };
    const Mat3 M12 = make_m(0, 1, p12.b);
    const Mat3 M13 = make_m(0, 2, p13.b);
    const Mat3 M23 = make_m(1, 2, p23.b);
    // D1, D2 (Eqn 5-6); D1 eliminates lambda_3, D2 eliminates lambda_1.
    const Mat3 D1 = M12 * p23.a - M23 * p12.a;
    const Mat3 D2 = M13 * p23.a - M23 * p13.a;
    // Cubic coefficients (Eqn 8-10).
    const double c3 = D2.determinant();
    const double c2 = D2.col(1).cross(D2.col(2)).dot(D1.col(0)) +
                      D2.col(2).cross(D2.col(0)).dot(D1.col(1)) +
                      D2.col(0).cross(D2.col(1)).dot(D1.col(2));
    const double c1 = D1.col(1).cross(D1.col(2)).dot(D2.col(0)) +
                      D1.col(2).cross(D1.col(0)).dot(D2.col(1)) +
                      D1.col(0).cross(D1.col(1)).dot(D2.col(2));
    const double c0 = D1.determinant();
    const double gamma = real_cubic_root(c3, c2, c1, c0);
    const Mat3 D0 = D1 + gamma * D2;
    const Eig3 eig = eig3x3_known0(D0);
    if (std::abs(eig.sigma1) < 1e-300) {
        return lambdas;
    }
    const double ratio = -eig.sigma2 / eig.sigma1;  // s^2 = -sigma2/sigma1
    if (ratio < 0.0) {
        return lambdas;  // no real s: no feasible solution
    }
    const double s_abs = std::sqrt(ratio);
    const double s_vals[2] = {s_abs, -s_abs};
    for (int si = 0; si < 2; ++si) {
        const double s = s_vals[si];
        // Columns of E: e1 -> E(:,0), e2 -> E(:,1) (Eqn 11-13).
        const double den = s * eig.E(0, 1) - eig.E(0, 0);
        if (std::abs(den) < 1e-14) {
            continue;
        }
        const double w0 = (eig.E(1, 0) - s * eig.E(1, 1)) / den;
        const double w1 = (eig.E(2, 0) - s * eig.E(2, 1)) / den;
        // With t = tau = lambda_3/lambda_2 and lambda_1 = w0 l2 + w1 l3, the
        // constraint lambda^T D1 lambda = 0 (equivalently n^T M12 n - a12 = 0 for
        // the pair {12}) is a quadratic A t^2 + B t + C = 0.
        const double A = p23.a * w1 * w1 - p12.a;
        const double B = 2.0 * p23.a * w0 * w1 - 2.0 * p23.a * p12.b * w1 + 2.0 * p12.a * p23.b;
        const double C = p23.a * w0 * w0 + p23.a - p12.a - 2.0 * p23.a * p12.b * w0;
        double taus[2];
        int n_tau = 0;
        if (std::abs(A) < 1e-14) {
            if (std::abs(B) > 1e-14) {
                const double t = -C / B;
                if (t > 0.0) {
                    taus[n_tau++] = t;
                }
            }
        } else {
            const double disc = B * B - 4.0 * A * C;
            if (disc >= 0.0) {
                const double sq = std::sqrt(disc);
                const double t1 = (-B + sq) / (2.0 * A);
                const double t2 = (-B - sq) / (2.0 * A);
                if (t1 > 0.0) {
                    taus[n_tau++] = t1;
                }
                if (t2 > 0.0) {
                    taus[n_tau++] = t2;
                }
            }
        }
        for (int ti = 0; ti < n_tau; ++ti) {
            const double tau = taus[ti];
            // lambda_2 from the {23} constraint (Eqn 16): a23/|(1,tau)|_M23^2.
            const double denom = tau * tau - 2.0 * p23.b * tau + 1.0;
            if (denom <= 0.0) {
                continue;
            }
            const double l2 = std::sqrt(p23.a / denom);
            const double l3 = tau * l2;
            const double l1 = w0 * l2 + w1 * l3;
            if (l1 <= 0.0) {
                continue;
            }
            const Vec3 L = refine_lambda(Vec3(l1, l2, l3), p12.a, p13.a, p23.a, p12.b,
                                         p13.b, p23.b);
            if (L.x() > 0.0 && L.y() > 0.0 && L.z() > 0.0) {
                lambdas.push_back(L);
            }
        }
    }
    return lambdas;
}

// Sec. 3.7: recover R and t from a feasible Lambda via Y = R X.
AbsolutePose pose_from_lambda(const Vec3& L, const std::array<Vec3, 3>& x,
                              const std::array<Vec3, 3>& y) {
    AbsolutePose pose;
    const Vec3 z1 = L.x() * y[0] - L.y() * y[1];
    const Vec3 z2 = L.x() * y[0] - L.z() * y[2];
    Mat3 Ymat;
    Ymat.col(0) = z1;
    Ymat.col(1) = z2;
    Ymat.col(2) = z1.cross(z2);
    Mat3 Xmat;
    Xmat.col(0) = x[0] - x[1];
    Xmat.col(1) = x[0] - x[2];
    Xmat.col(2) = (x[0] - x[1]).cross(x[0] - x[2]);
    pose.R = Ymat * Xmat.inverse();
    // Orthonormalise via SVD to remove any rounding drift (exact inputs give the
    // identity correction; the paper notes this step is exact for true data).
    Eigen::JacobiSVD<Mat3> svd(pose.R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    pose.R = svd.matrixU() * svd.matrixV().transpose();
    if (pose.R.determinant() < 0.0) {
        Mat3 U = svd.matrixU();
        U.col(2) *= -1.0;
        pose.R = U * svd.matrixV().transpose();
    }
    pose.t = L.x() * y[0] - pose.R * x[0];
    return pose;
}

}  // namespace

std::vector<AbsolutePose> p3p(const std::array<Eigen::Vector3d, 3>& world,
                              const std::array<Eigen::Vector3d, 3>& bearings) {
    std::array<Vec3, 3> x;
    std::array<Vec3, 3> y;
    for (int i = 0; i < 3; ++i) {
        x[static_cast<std::size_t>(i)] = world[static_cast<std::size_t>(i)];
        const double n = bearings[static_cast<std::size_t>(i)].norm();
        if (n <= 0.0) {
            return {};
        }
        y[static_cast<std::size_t>(i)] = bearings[static_cast<std::size_t>(i)] / n;
    }
    std::vector<AbsolutePose> poses;
    for (const Vec3& L : p3p_lambda_solutions(x, y)) {
        poses.push_back(pose_from_lambda(L, x, y));
    }
    return poses;
}

double reprojection_angle_error(const AbsolutePose& pose, const Eigen::Vector3d& world,
                                const Eigen::Vector3d& bearing) {
    const Vec3 d = pose.R * world + pose.t;
    const double dn = d.norm();
    const double bn = bearing.norm();
    if (dn <= 0.0 || bn <= 0.0) {
        return kPi;
    }
    // atan2(|a x b|, a . b) is accurate near 0, unlike acos (which loses ~1e-8
    // rad to rounding whenever the vectors are nearly parallel).
    const Vec3 a = d / dn;
    const Vec3 b = bearing / bn;
    return std::atan2(a.cross(b).norm(), std::clamp(a.dot(b), -1.0, 1.0));
}

AbsolutePose refine_absolute_pose(const AbsolutePose& initial,
                                  const std::vector<Eigen::Vector3d>& world,
                                  const std::vector<Eigen::Vector3d>& bearings) {
    AbsolutePose pose = initial;
    if (world.size() != bearings.size() || world.empty()) {
        return pose;
    }
    // 2 residuals per observation; skip any point behind the camera (b.d <= 0).
    const auto cost_of = [&](const AbsolutePose& p) {
        double sum = 0.0;
        for (std::size_t i = 0; i < world.size(); ++i) {
            const Vec3 d = p.R * world[i] + p.t;
            const double bd = bearings[i].dot(d);
            if (bd <= 0.0) {
                continue;  // outlier for refinement
            }
            const Vec3 b = bearings[i];
            Vec3 e1 = b.cross(Vec3::UnitZ());
            if (e1.squaredNorm() < 1e-12) {
                e1 = b.cross(Vec3::UnitX());
            }
            e1.normalize();
            const Vec3 e2 = b.cross(e1);
            const double r0 = e1.dot(d) / bd;
            const double r1 = e2.dot(d) / bd;
            sum += r0 * r0 + r1 * r1;
        }
        return sum;
    };
    // Numeric Jacobian by central differences over (omega, t) at 6 params.
    const auto residuals = [&](const AbsolutePose& p, std::vector<double>& out) {
        out.clear();
        for (std::size_t i = 0; i < world.size(); ++i) {
            const Vec3 d = p.R * world[i] + p.t;
            const double bd = bearings[i].dot(d);
            if (bd <= 0.0) {
                continue;
            }
            const Vec3 b = bearings[i];
            Vec3 e1 = b.cross(Vec3::UnitZ());
            if (e1.squaredNorm() < 1e-12) {
                e1 = b.cross(Vec3::UnitX());
            }
            e1.normalize();
            const Vec3 e2 = b.cross(e1);
            out.push_back(e1.dot(d) / bd);
            out.push_back(e2.dot(d) / bd);
        }
    };
    std::vector<double> base;
    residuals(pose, base);
    if (base.empty()) {
        return pose;
    }
    double cost = 0.0;
    for (double v : base) {
        cost += v * v;
    }
    double lambda_damp = 1e-3;
    const double h = 1e-6;
    for (int it = 0; it < 10; ++it) {
        std::vector<double> res0;
        residuals(pose, res0);
        const Eigen::Index m = static_cast<Eigen::Index>(res0.size());
        Eigen::MatrixXd J(m, 6);
        for (int k = 0; k < 6; ++k) {
            AbsolutePose pp = pose;
            AbsolutePose pm = pose;
            if (k < 3) {
                Vec3 dp = Vec3::Zero();
                dp[k] = h;
                pp.R = exp_so3(dp) * pose.R;
                pm.R = exp_so3(-dp) * pose.R;
            } else {
                pp.t[k - 3] += h;
                pm.t[k - 3] -= h;
            }
            std::vector<double> rp;
            std::vector<double> rm;
            residuals(pp, rp);
            residuals(pm, rm);
            if (rp.size() != res0.size() || rm.size() != res0.size()) {
                J.col(k).setZero();
                continue;
            }
            for (Eigen::Index r = 0; r < m; ++r) {
                J(r, k) = (rp[static_cast<std::size_t>(r)] -
                           rm[static_cast<std::size_t>(r)]) / (2.0 * h);
            }
        }
        Eigen::VectorXd rvec(m);
        for (Eigen::Index r = 0; r < m; ++r) {
            rvec[r] = res0[static_cast<std::size_t>(r)];
        }
        const Eigen::MatrixXd JtJ = J.transpose() * J;
        const Eigen::VectorXd g = J.transpose() * rvec;
        const double prev_cost = cost;
        for (int lm = 0; lm < 6; ++lm) {
            Eigen::MatrixXd A = JtJ;
            A.diagonal().array() += lambda_damp;
            const Eigen::VectorXd step = A.ldlt().solve(-g);
            AbsolutePose cand = pose;
            cand.R = exp_so3(step.head<3>()) * pose.R;
            cand.t = pose.t + step.tail<3>();
            const double cand_cost = cost_of(cand);
            if (cand_cost <= cost) {
                pose = cand;
                cost = cand_cost;
                lambda_damp = std::max(lambda_damp * 0.5, 1e-12);
                break;
            }
            lambda_damp *= 10.0;
        }
        if (cost >= prev_cost) {
            break;  // no accepted step this iteration
        }
        const double rel = (prev_cost > 0.0) ? (prev_cost - cost) / prev_cost : 0.0;
        if (rel < 1e-10) {
            break;
        }
    }
    // Never worse than the input in cost.
    return cost_of(pose) <= cost_of(initial) ? pose : initial;
}

AbsolutePoseResult estimate_absolute_pose_ransac(const std::vector<Eigen::Vector3d>& world,
                                                 const std::vector<Eigen::Vector3d>& bearings,
                                                 const RansacOptions& options) {
    AbsolutePoseResult result;
    if (world.size() != bearings.size()) {
        throw ValueError("estimate_absolute_pose_ransac: world and bearings size mismatch");
    }
    if (options.threshold_rad <= 0.0) {
        throw ValueError("estimate_absolute_pose_ransac: threshold_rad must be positive");
    }
    const std::size_t m = world.size();
    result.inliers.assign(m, 0);
    if (m < 3) {
        return result;  // success stays false
    }

    const double thresh = options.threshold_rad;
    const int max_iter = options.max_iterations;
    const int min_iter = options.min_iterations;
    const double conf = options.confidence;

    std::mt19937_64 rng(options.seed);
    // Sample indices with the portable draw, not std::uniform_int_distribution:
    // a distribution's algorithm is implementation-defined, so the same seed
    // produced different RANSAC samples under libstdc++ and libc++.
    std::vector<std::size_t> order(m, 0);

    // Inlier iff the reprojection angle is below threshold AND the point is in
    // front of the camera.
    const auto score = [&](const AbsolutePose& pose, std::size_t& count,
                           double& sum_err, std::vector<std::uint8_t>& inl) {
        inl.assign(m, 0);
        count = 0;
        sum_err = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            const Vec3 d = pose.R * world[i] + pose.t;
            if (d.dot(bearings[i]) <= 0.0) {
                continue;
            }
            const double e = reprojection_angle_error(pose, world[i], bearings[i]);
            if (e < thresh) {
                inl[i] = 1;
                ++count;
                sum_err += e;
            }
        }
    };

    AbsolutePose best_pose;
    std::vector<std::uint8_t> best_inliers(m, 0);
    std::size_t best_count = 0;
    double best_sum = std::numeric_limits<double>::infinity();
    int iterations = 0;
    int k = max_iter;

    while (iterations < k && iterations < max_iter) {
        ++iterations;
        std::array<std::size_t, 3> idx{};
        rng::distinct_indices(rng, m, 3, idx.data(), order.data());
        std::array<Vec3, 3> sx{};
        std::array<Vec3, 3> sy{};
        for (int j = 0; j < 3; ++j) {
            const std::size_t s = idx[static_cast<std::size_t>(j)];
            sx[static_cast<std::size_t>(j)] = world[s];
            const double n = bearings[s].norm();
            sy[static_cast<std::size_t>(j)] =
                n > 0.0 ? bearings[s] / n : bearings[s];
        }
        for (const Vec3& L : p3p_lambda_solutions(sx, sy)) {
            const AbsolutePose cand = pose_from_lambda(L, sx, sy);
            std::size_t count = 0;
            double sum_err = 0.0;
            std::vector<std::uint8_t> inl;
            score(cand, count, sum_err, inl);
            const bool better =
                count > best_count || (count == best_count && sum_err < best_sum);
            if (count > 0 && better) {
                best_count = count;
                best_sum = sum_err;
                best_pose = cand;
                best_inliers = inl;
                const double w = static_cast<double>(count) / static_cast<double>(m);
                const double w3 = w * w * w;
                if (w3 >= 1.0 - 1e-12) {
                    k = min_iter;
                } else if (w3 <= 0.0) {
                    k = max_iter;
                } else {
                    const double kd = std::ceil(std::log(1.0 - conf) / std::log(1.0 - w3));
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

    // Refine on the inliers, recompute inliers, keep the refined pose if not
    // worse (more inliers, or equal inliers with lower summed error).
    std::vector<Vec3> iw;
    std::vector<Vec3> ib;
    iw.reserve(best_count);
    ib.reserve(best_count);
    for (std::size_t i = 0; i < m; ++i) {
        if (best_inliers[i] != 0) {
            iw.push_back(world[i]);
            ib.push_back(bearings[i]);
        }
    }
    const AbsolutePose refined = refine_absolute_pose(best_pose, iw, ib);
    std::size_t rcount = 0;
    double rsum = 0.0;
    std::vector<std::uint8_t> rinl;
    score(refined, rcount, rsum, rinl);
    if (rcount > best_count || (rcount == best_count && rsum <= best_sum)) {
        best_pose = refined;
        best_inliers = rinl;
        best_count = rcount;
        best_sum = rsum;
    }

    result.pose = best_pose;
    result.inliers = best_inliers;
    result.inlier_count = best_count;
    result.success = best_count >= 4;
    return result;
}

}  // namespace alphashape::sfm
