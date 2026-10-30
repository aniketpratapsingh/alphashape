// WP-12.7: Tait-Bryan rotation orders (CRS-10).
//
// The convention under test is the one the header states: intrinsic, active,
// yaw/pitch/roll are the first/second/third axis of the order, and degrees at
// the API. Two invariants carry it:
//
//   * every order round-trips a random rotation to <= 1e-12 rad, which fails if
//     the sequence, the axis permutation or the permutation's determinant sign
//     is wrong for any of the six;
//   * ZYX agrees with Eigen's own RollPitchYaw composition, so "ZYX" here means
//     the same thing as everywhere else in the ecosystem.
//
// The gimbal-lock cases assert the documented alternative solution: the middle
// angle stays at +/-90, roll becomes 0, and the whole remaining rotation is in
// yaw -- and that this still reproduces the input matrix exactly.

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"
#include "model/crs/euler.hpp"

namespace alphashape::crs {
namespace {

constexpr double kDeg = 3.14159265358979323846 / 180.0;

std::array<RotationOrder, 6> orders() {
    return {RotationOrder::XYZ, RotationOrder::XZY, RotationOrder::YXZ,
            RotationOrder::YZX, RotationOrder::ZXY, RotationOrder::ZYX};
}

/// Angle of the relative rotation A^T B, in radians. Computed from the
/// antisymmetric part plus atan2 so it stays accurate near 0, where acos of the
/// trace loses most of its digits.
double rotation_angle(const Eigen::Matrix3d& a, const Eigen::Matrix3d& b) {
    const Eigen::Matrix3d r = a.transpose() * b;
    const Eigen::Vector3d v(r(2, 1) - r(1, 2), r(0, 2) - r(2, 0), r(1, 0) - r(0, 1));
    return std::atan2(v.norm() / 2.0, (r.trace() - 1.0) / 2.0);
}

TEST(Crs, EulerRoundTripAllOrders) {
    std::mt19937 rng(0x5eedu);
    std::size_t checked = 0;
    for (const RotationOrder order : orders()) {
        double worst = 0.0;
        for (int i = 0; i < 1000; ++i) {
            EulerAngles input;
            input.order = order;
            input.yaw = rng::uniform_real(rng, -180.0, 180.0);
            input.pitch = rng::uniform_real(rng, -180.0, 180.0);
            input.roll = rng::uniform_real(rng, -180.0, 180.0);
            const EulerAngles recovered = from_matrix(to_matrix(input), order);
            const double error =
                rotation_angle(to_matrix(input), to_matrix(recovered));
            worst = std::max(worst, error);
            EXPECT_LE(error, 1e-12) << rotation_order_name(order) << " yaw " << input.yaw
                                    << " pitch " << input.pitch << " roll " << input.roll;
            ++checked;
        }
        // The two gimbal rows, explicitly, for every order: the middle angle is
        // kept at ±90 degrees, roll is forced to 0 and yaw carries the rest,
        // and the recovered angles still reproduce the matrix. Sampling never
        // hits these exactly, so they are asserted here rather than left to
        // luck.
        for (const double middle : {90.0, -90.0}) {
            EulerAngles input;
            input.order = order;
            input.yaw = 37.5;
            input.pitch = middle;
            input.roll = -112.25;
            const Eigen::Matrix3d expected = to_matrix(input);
            const EulerAngles recovered = from_matrix(expected, order);
            EXPECT_DOUBLE_EQ(recovered.pitch, middle) << rotation_order_name(order);
            EXPECT_EQ(recovered.roll, 0.0) << rotation_order_name(order);
            EXPECT_LE(rotation_angle(expected, to_matrix(recovered)), 1e-12)
                << rotation_order_name(order) << " pitch " << middle;
            EXPECT_EQ(recovered.order, order);
        }
        RecordProperty((std::string("worst_") + rotation_order_name(order)).c_str(), worst);
    }
    EXPECT_EQ(checked, 6u * 1000u);
}

// The singular case: any roll is unobservable, the recovered angles put the
// whole rotation into yaw and report roll == 0.
TEST(Crs, EulerGimbalLockPutsEverythingInYaw) {
    std::mt19937 rng(0xbeefu);
    for (const RotationOrder order : orders()) {
        for (const double middle : {90.0, -90.0}) {
            for (int i = 0; i < 20; ++i) {
                EulerAngles input;
                input.order = order;
                input.yaw = rng::uniform_real(rng, -180.0, 180.0);
                input.pitch = middle;
                input.roll = rng::uniform_real(rng, -180.0, 180.0);
                const Eigen::Matrix3d expected = to_matrix(input);
                const EulerAngles recovered = from_matrix(expected, order);
                // The documented alternative solution.
                EXPECT_EQ(recovered.roll, 0.0) << rotation_order_name(order);
                EXPECT_DOUBLE_EQ(recovered.pitch, middle) << rotation_order_name(order);
                // Still the same rotation.
                EXPECT_LE(rotation_angle(expected, to_matrix(recovered)), 1e-12)
                    << rotation_order_name(order) << " yaw " << input.yaw << " roll "
                    << input.roll;
                // And the yaw absorbs yaw and roll: rotating by the recovered
                // angles is the same as rotating by the input's yaw then a
                // pure middle rotation.
                EulerAngles reference;
                reference.order = order;
                reference.yaw = recovered.yaw;
                reference.pitch = middle;
                EXPECT_LE(rotation_angle(expected, to_matrix(reference)), 1e-12)
                    << rotation_order_name(order);
            }
        }
    }
}

// ZYX is Eigen's RollPitchYaw: R = Rz(yaw) Ry(pitch) Rx(roll).
TEST(Crs, EulerMatchesEigenForZyx) {
    std::mt19937 rng(0x1234u);
    for (int i = 0; i < 500; ++i) {
        EulerAngles input;
        input.yaw = rng::uniform_real(rng, -180.0, 180.0);
        input.pitch = rng::uniform_real(rng, -90.0, 90.0);
        input.roll = rng::uniform_real(rng, -180.0, 180.0);
        const Eigen::Matrix3d expected =
            (Eigen::AngleAxisd(input.yaw * kDeg, Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(input.pitch * kDeg, Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(input.roll * kDeg, Eigen::Vector3d::UnitX()))
                .toRotationMatrix();
        EXPECT_LE((to_matrix(input) - expected).cwiseAbs().maxCoeff(), 1e-15);
        const EulerAngles back = from_matrix(expected, RotationOrder::ZYX);
        EXPECT_NEAR(back.yaw, input.yaw, 1e-9);
        EXPECT_NEAR(back.pitch, input.pitch, 1e-9);
        EXPECT_NEAR(back.roll, input.roll, 1e-9);
    }
}

TEST(Crs, EulerOrderNamesParseBothWays) {
    for (const RotationOrder order : orders()) {
        const std::string name = rotation_order_name(order);
        EXPECT_EQ(name.size(), 3u);
        RotationOrder parsed = RotationOrder::ZYX;
        EXPECT_TRUE(parse_rotation_order(name, &parsed));
        EXPECT_EQ(parsed, order);
        // Case-insensitive, because the meta key is free text.
        std::string lower = name;
        for (char& c : lower) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        EXPECT_TRUE(parse_rotation_order(lower, &parsed));
        EXPECT_EQ(parsed, order);
    }
    RotationOrder untouched = RotationOrder::ZYX;
    EXPECT_FALSE(parse_rotation_order("ZYXW", &untouched));
    EXPECT_FALSE(parse_rotation_order("XYX", &untouched));
    EXPECT_FALSE(parse_rotation_order("", &untouched));
    EXPECT_EQ(untouched, RotationOrder::ZYX);
}

TEST(Crs, EulerRejectsNonRotations) {
    Eigen::Matrix3d scaled = Eigen::Matrix3d::Identity() * 2.0;
    EXPECT_THROW(from_matrix(scaled, RotationOrder::ZYX), ValueError);
    Eigen::Matrix3d mirror = Eigen::Matrix3d::Identity();
    mirror(0, 0) = -1.0;
    EXPECT_THROW(from_matrix(mirror, RotationOrder::ZYX), ValueError);
    Eigen::Matrix3d nan = Eigen::Matrix3d::Identity();
    nan(1, 1) = std::nan("");
    EXPECT_THROW(from_matrix(nan, RotationOrder::ZYX), ValueError);
}

TEST(Crs, EulerAnglesToRadiansPreserveOrder) {
    EulerAngles degrees;
    degrees.yaw = 180.0;
    degrees.pitch = -90.0;
    degrees.roll = 45.0;
    degrees.order = RotationOrder::XYZ;
    const EulerAngles radians = degrees.radians();
    EXPECT_DOUBLE_EQ(radians.yaw, 3.14159265358979323846);
    EXPECT_DOUBLE_EQ(radians.pitch, -3.14159265358979323846 / 2.0);
    EXPECT_DOUBLE_EQ(radians.roll, 3.14159265358979323846 / 4.0);
    EXPECT_EQ(radians.order, RotationOrder::XYZ);
}

}  // namespace
}  // namespace alphashape::crs
