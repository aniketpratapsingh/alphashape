// Tests for multi-view linear triangulation (DLT on bearings).
#include "common/algorithm/sfm/triangulation.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "base/utils/portable_rng.hpp"
#include "test_rng_helpers.hpp"

namespace alphashape::sfm {
namespace {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

// A viewing rig: cameras spread around the origin, all looking at the origin,
// with the point of interest near the origin.
struct Rig {
    std::vector<AbsolutePose> poses;
    Vec3 X;
    std::vector<Vec3> bearings;
};

Rig make_rig(std::mt19937_64& rng, int n_views, const Vec3& X) {
    Rig rig;
    rig.X = X;
    // Portable draws, not std::distributions: those map engine bits to values
    // differently under libc++ and libstdc++ (TESTING.md, "Numeric consistency").
    const auto ang = [](std::mt19937_64& g) { return alphashape::rng::uniform_real(g, 0.0, 2.0 * 3.14159265358979); };
    const auto rad = [](std::mt19937_64& g) { return alphashape::rng::uniform_real(g, 8.0, 12.0); };
    for (int i = 0; i < n_views; ++i) {
        const double a = ang(rng);
        const double r = rad(rng);
        const Vec3 centre(r * std::cos(a), r * std::sin(a), 0.3 * r);
        // Look at the origin: camera +Z points from centre toward origin.
        const Vec3 fwd = (-centre).normalized();
        Vec3 up = Vec3::UnitZ();
        if (std::abs(fwd.dot(up)) > 0.9) {
            up = Vec3::UnitY();
        }
        const Vec3 right = up.cross(fwd).normalized();
        const Vec3 down = fwd.cross(right).normalized();
        AbsolutePose p;
        p.R.row(0) = right.transpose();
        p.R.row(1) = down.transpose();
        p.R.row(2) = fwd.transpose();
        p.t = -p.R * centre;
        rig.poses.push_back(p);
        rig.bearings.push_back((p.R * X + p.t).normalized());
    }
    return rig;
}

}  // namespace

TEST(Triangulation, ExactRecoveryMultiView) {
    for (int n_views : {2, 3, 8}) {
        std::mt19937_64 rng(static_cast<std::uint64_t>(1000 + n_views));
        const Vec3 X(0.5, -0.3, 0.2);
        const Rig rig = make_rig(rng, n_views, X);
        TriangulationOptions opt;
        const std::optional<Vec3> est =
            triangulate_multiview(rig.poses, rig.bearings, opt);
        ASSERT_TRUE(est.has_value());
        const double rel = (est.value() - X).norm() / X.norm();
        std::cout << "[ExactRecoveryMultiView] n=" << n_views << " rel=" << rel
                  << std::endl;
        EXPECT_LT(rel, 1e-9);
    }
}

TEST(Triangulation, NoisyRecovery) {
    std::mt19937_64 rng(2025);
    const double sigma = 0.5 / 1000.0;
    const Vec3 X(0.5, -0.3, 0.2);
    const Rig rig_base = make_rig(rng, 8, X);
    const auto noise = [sigma](std::mt19937_64& g) { return alphashape::rng::normal(g, 0.0, sigma); };
    std::vector<Vec3> b = rig_base.bearings;
    for (Vec3& bv : b) {
        Vec3 t1 = bv.cross(Vec3::UnitZ());
        if (t1.norm() < 1e-6) {
            t1 = bv.cross(Vec3::UnitX());
        }
        t1.normalize();
        const Vec3 t2 = bv.cross(t1);
        // The outer `+` takes its right operand first under g++, so the t2
        // noise was drawn before the t1 noise.
        const double n_t2 = noise(rng);
        const double n_t1 = noise(rng);
        bv = (bv + n_t1 * t1 + n_t2 * t2).normalized();
    }
    TriangulationOptions opt;
    const std::optional<Vec3> est =
        triangulate_multiview(rig_base.poses, b, opt);
    ASSERT_TRUE(est.has_value());
    const double err = (est.value() - X).norm();
    std::cout << "[NoisyRecovery] err=" << err << std::endl;
    EXPECT_LT(err, 0.05);
}

TEST(Triangulation, LowParallaxReturnsNullopt) {
    std::mt19937_64 rng(3030);
    const Vec3 X(0.0, 0.0, 0.0);
    const Rig rig = make_rig(rng, 2, X);
    // Almost coincident cameras: move the second camera centre onto the first.
    std::vector<AbsolutePose> poses = rig.poses;
    poses[1] = poses[0];
    poses[1].t += Vec3(1e-4, 0.0, 0.0);
    TriangulationOptions opt;
    const std::optional<Vec3> est = triangulate_multiview(poses, rig.bearings, opt);
    EXPECT_FALSE(est.has_value());
}

TEST(Triangulation, PointBehindCameraReturnsNullopt) {
    std::mt19937_64 rng(4040);
    const Vec3 X(0.5, -0.3, 0.2);
    const Rig rig = make_rig(rng, 3, X);
    // Reverse the third bearing: the observation line is unchanged, so the DLT
    // still recovers X, but the point is now behind that camera.
    std::vector<Vec3> b = rig.bearings;
    b[2] = -b[2];
    TriangulationOptions opt;
    const std::optional<Vec3> est = triangulate_multiview(rig.poses, b, opt);
    EXPECT_FALSE(est.has_value());
}

TEST(Triangulation, MaxAngleErrorRejectsCorruptedObservation) {
    std::mt19937_64 rng(5050);
    const Vec3 X(0.5, -0.3, 0.2);
    const Rig rig = make_rig(rng, 4, X);
    std::vector<Vec3> b = rig.bearings;
    // Corrupt the second observation enough to fail a tight angle gate.
    Vec3 t1 = b[1].cross(Vec3::UnitZ());
    if (t1.norm() < 1e-6) {
        t1 = b[1].cross(Vec3::UnitX());
    }
    t1.normalize();
    b[1] = (b[1] + 0.02 * t1).normalized();

    TriangulationOptions permissive;
    permissive.max_angle_error_rad = 0.0;  // no check
    EXPECT_TRUE(triangulate_multiview(rig.poses, b, permissive).has_value());

    TriangulationOptions strict;
    strict.max_angle_error_rad = 1e-3;  // 0.02 rad corruption must be rejected
    EXPECT_FALSE(triangulate_multiview(rig.poses, b, strict).has_value());
}

TEST(Triangulation, MaxParallaxKnownConfiguration) {
    // Two cameras 10 units apart on the x-axis, both looking at the origin from
    // (0,0,0)-ish target. Rays to X=(0,0,-5) form a known angle.
    AbsolutePose a;  // identity at origin, +Z forward
    AbsolutePose c;  // at (10,0,0) looking back toward -X
    c.R << 0.0, 0.0, -1.0,
           0.0, 1.0, 0.0,
           1.0, 0.0, 0.0;
    c.t = -c.R * Vec3(10.0, 0.0, 0.0);
    const std::vector<AbsolutePose> poses{a, c};
    const Vec3 X(5.0, 0.0, -5.0);
    const double ang = max_parallax(poses, X);
    // Ray from origin: (5,0,-5) -> 45 deg below +Z toward -Z. Ray from
    // (10,0,0): (-5,0,-5) -> 45 deg. Angle between the two rays = 90 deg.
    std::cout << "[MaxParallaxKnownConfiguration] ang_deg="
              << ang * 180.0 / 3.14159265358979 << std::endl;
    EXPECT_NEAR(ang, 3.14159265358979 / 2.0, 1e-9);
    // A single camera has no parallax.
    EXPECT_DOUBLE_EQ(max_parallax({a}, X), 0.0);
}

}  // namespace alphashape::sfm
