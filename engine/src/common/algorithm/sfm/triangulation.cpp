// triangulation.cpp -- N-view linear triangulation (DLT on bearings) and the
// max-parallax helper. Clean-room implementation from Hartley & Zisserman,
// "Multiple View Geometry in Computer Vision", Ch. 12 (direct linear transform).

#include "common/algorithm/sfm/triangulation.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

namespace alphashape::sfm {
namespace {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

Mat3 skew(const Vec3& v) {
    Mat3 s;
    s << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
    return s;
}

}  // namespace

double max_parallax(const std::vector<AbsolutePose>& poses, const Eigen::Vector3d& X) {
    const std::size_t n = poses.size();
    if (n < 2) {
        return 0.0;
    }
    std::vector<Vec3> dirs(n);
    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 centre = -poses[i].R.transpose() * poses[i].t;  // C = -R^T t
        Vec3 d = X - centre;
        const double dn = d.norm();
        dirs[i] = dn > 0.0 ? d / dn : d;
    }
    double best = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double c = std::clamp(dirs[i].dot(dirs[j]), -1.0, 1.0);
            const double ang = std::acos(c);
            if (ang > best) {
                best = ang;
            }
        }
    }
    return best;
}

std::optional<Eigen::Vector3d> triangulate_multiview(const std::vector<AbsolutePose>& poses,
                                                     const std::vector<Eigen::Vector3d>& bearings,
                                                     const TriangulationOptions& options) {
    const std::size_t n = poses.size();
    if (n < 2 || bearings.size() != n) {
        return std::nullopt;
    }
    // Each observation contributes the rows of [b]_x (R X + t) = 0, i.e. the
    // 3x4 block  [ [b]_x R | [b]_x t ]  acting on the homogeneous X = (X,1).
    // We keep all three rows: the third is a linear combination of the first two
    // in exact arithmetic, but including it only improves the SVD conditioning
    // and adds no spurious constraints.
    Eigen::MatrixXd A(3 * n, 4);
    A.setZero();
    for (std::size_t i = 0; i < n; ++i) {
        const double bn = bearings[i].norm();
        if (bn <= 0.0) {
            return std::nullopt;
        }
        const Vec3 b = bearings[i] / bn;
        const Mat3 bx = skew(b);
        A.block<3, 3>(static_cast<Eigen::Index>(3 * i), 0) = bx * poses[i].R;
        A.block<3, 1>(static_cast<Eigen::Index>(3 * i), 3) = bx * poses[i].t;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::VectorXd y = svd.matrixV().col(3);
    if (std::abs(y[3]) < 1e-12) {
        return std::nullopt;  // point at infinity
    }
    const Vec3 X = y.head<3>() / y[3];

    // Positive depth in every view.
    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 b = bearings[i] / bearings[i].norm();
        if ((poses[i].R * X + poses[i].t).dot(b) <= 0.0) {
            return std::nullopt;
        }
    }
    // Parallax gate.
    if (max_parallax(poses, X) < options.min_parallax_rad) {
        return std::nullopt;
    }
    // Optional per-observation reprojection angle gate.
    if (options.max_angle_error_rad > 0.0) {
        for (std::size_t i = 0; i < n; ++i) {
            const Vec3 d = poses[i].R * X + poses[i].t;
            const double dn = d.norm();
            if (dn <= 0.0) {
                return std::nullopt;
            }
            const Vec3 b = bearings[i] / bearings[i].norm();
            const double c = std::clamp(d.dot(b) / dn, -1.0, 1.0);
            if (std::acos(c) >= options.max_angle_error_rad) {
                return std::nullopt;
            }
        }
    }
    return X;
}

}  // namespace alphashape::sfm
