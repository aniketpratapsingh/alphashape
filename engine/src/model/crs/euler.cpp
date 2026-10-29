#include "model/crs/euler.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <string>

#include <Eigen/Geometry>

#include "base/utils/error.hpp"

namespace alphashape::crs {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

/// Axis index (0 = x, 1 = y, 2 = z) of each field, in yaw/pitch/roll order.
std::array<int, 3> axes_of(RotationOrder order) {
    switch (order) {
        case RotationOrder::XYZ:
            return {0, 1, 2};
        case RotationOrder::XZY:
            return {0, 2, 1};
        case RotationOrder::YXZ:
            return {1, 0, 2};
        case RotationOrder::YZX:
            return {1, 2, 0};
        case RotationOrder::ZXY:
            return {2, 0, 1};
        case RotationOrder::ZYX:
            break;
    }
    return {2, 1, 0};
}

Eigen::Vector3d unit_axis(int axis) {
    return Eigen::Vector3d(axis == 0 ? 1.0 : 0.0, axis == 1 ? 1.0 : 0.0,
                           axis == 2 ? 1.0 : 0.0);
}

Eigen::Matrix3d axis_rotation(int axis, double degrees) {
    return Eigen::AngleAxisd(degrees * kDeg, unit_axis(axis)).toRotationMatrix();
}

/// The signed permutation that sends axis `axes[k]` to e_k. Its determinant is
/// the sign that enters every recovered angle: for an orthogonal Q with
/// det(Q) = s, `Q R_v(t) Q^T = R_{Qv}(s t)`.
Eigen::Matrix3d permutation_of(const std::array<int, 3>& axes, double* det) {
    Eigen::Matrix3d p = Eigen::Matrix3d::Zero();
    for (int row = 0; row < 3; ++row) {
        p(row, axes[static_cast<std::size_t>(row)]) = 1.0;
    }
    *det = p.determinant();
    return p;
}

double clamp_unit(double value) { return std::max(-1.0, std::min(1.0, value)); }

}  // namespace

const char* rotation_order_name(RotationOrder order) {
    switch (order) {
        case RotationOrder::XYZ:
            return "XYZ";
        case RotationOrder::XZY:
            return "XZY";
        case RotationOrder::YXZ:
            return "YXZ";
        case RotationOrder::YZX:
            return "YZX";
        case RotationOrder::ZXY:
            return "ZXY";
        case RotationOrder::ZYX:
            return "ZYX";
    }
    return "";
}

bool parse_rotation_order(const std::string& text, RotationOrder* order) {
    std::string upper;
    upper.reserve(text.size());
    for (const char c : text) {
        upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }
    for (const RotationOrder candidate : {
             RotationOrder::XYZ, RotationOrder::XZY, RotationOrder::YXZ,
             RotationOrder::YZX, RotationOrder::ZXY, RotationOrder::ZYX}) {
        if (upper == rotation_order_name(candidate)) {
            if (order != nullptr) {
                *order = candidate;
            }
            return true;
        }
    }
    return false;
}

const RotationOrder* all_rotation_orders() {
    static const RotationOrder kOrders[] = {
        RotationOrder::XYZ, RotationOrder::XZY, RotationOrder::YXZ,
        RotationOrder::YZX, RotationOrder::ZXY, RotationOrder::ZYX};
    return kOrders;
}

EulerAngles EulerAngles::radians() const {
    EulerAngles out;
    out.yaw = yaw * kDeg;
    out.pitch = pitch * kDeg;
    out.roll = roll * kDeg;
    out.order = order;
    return out;
}

Eigen::Matrix3d to_matrix(const EulerAngles& angles) {
    const std::array<int, 3> axes = axes_of(angles.order);
    // Intrinsic = the product in yaw/pitch/roll order, each about the axis its
    // field names. See the header for the convention.
    return axis_rotation(axes[0], angles.yaw) * axis_rotation(axes[1], angles.pitch) *
           axis_rotation(axes[2], angles.roll);
}

// --- chunk 2
EulerAngles from_matrix(const Eigen::Matrix3d& matrix, RotationOrder order) {
    constexpr double kOrthonormalTolerance = 1e-6;
    if (!matrix.allFinite() ||
        (matrix.transpose() * matrix - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() >
            kOrthonormalTolerance ||
        matrix.determinant() <= 0.0) {
        throw ValueError("from_matrix: the input is not a rotation matrix");
    }
    double det = 1.0;
    const Eigen::Matrix3d p = permutation_of(axes_of(order), &det);
    // Q R_n(t) Q^T = R_{Qn}(det(Q) t), so conjugating by the axis permutation
    // turns any Tait-Bryan sequence into the canonical Rx(A) Ry(B) Rz(C).
    const Eigen::Matrix3d r = p * matrix * p.transpose();
    const double b = std::asin(clamp_unit(r(0, 2)));
    double a = 0.0;
    double c = 0.0;
    if (std::abs(std::cos(b)) > 1e-9) {
        a = std::atan2(-r(1, 2), r(2, 2));
        c = std::atan2(-r(0, 1), r(0, 0));
    } else {
        // Gimbal lock: the outer axes coincide. Keep the middle angle at its
        // singular value, set the third angle to zero, and absorb everything
        // observable into the first (documented alternative solution).
        c = 0.0;
        a = std::atan2(r(2, 1), r(1, 1));
    }
    EulerAngles angles;
    angles.yaw = det * a / kDeg;
    angles.pitch = det * b / kDeg;
    angles.roll = det * c / kDeg;
    angles.order = order;
    return angles;
}

}  // namespace alphashape::crs
