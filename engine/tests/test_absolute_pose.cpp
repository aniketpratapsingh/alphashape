// Tests for the Lambda Twist P3P solver, pose refinement and absolute-pose RANSAC.
#include "common/algorithm/sfm/absolute_pose.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"
#include "test_rng_helpers.hpp"

namespace alphashape::sfm {
namespace {

using alphashape::ValueError;
using test_rng::draw3;

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

// Portable draws, not std::distributions: those map engine bits to values
// differently under libc++ and libstdc++, so a seed selected a different scene
// on macOS and Linux (TESTING.md, "Numeric consistency").
const auto unit_draw = [](std::mt19937_64& g) { return rng::uniform_real(g, -1.0, 1.0); };

// Rotation of up to `max_deg` about a random axis.
Mat3 random_rotation(std::mt19937_64& rng, double max_deg) {
    // draw3 sequences the three draws in g++'s right-to-left order, which is
    // the order this scene was calibrated with (see test_rng_helpers.hpp).
    Vec3 axis = draw3(rng, unit_draw, unit_draw, unit_draw);
    if (axis.norm() < 1e-6) {
        axis = Vec3::UnitX();
    }
    axis.normalize();
    const double ang = rng::uniform_real(rng, 0.0, max_deg * 3.14159265358979 / 180.0);
    return Eigen::AngleAxisd(ang, axis).toRotationMatrix();
}

// Ground-truth camera pose looking roughly along +Z.
AbsolutePose make_pose(std::mt19937_64& rng) {
    AbsolutePose p;
    p.R = random_rotation(rng, 20.0);
    const auto t = [](std::mt19937_64& g) { return rng::uniform_real(g, -2.0, 2.0); };
    p.t = draw3(rng, t, t, t);
    return p;
}

std::array<Vec3, 3> random_world(std::mt19937_64& rng) {
    const auto x = [](std::mt19937_64& g) { return rng::uniform_real(g, -5.0, 5.0); };
    const auto z = [](std::mt19937_64& g) { return rng::uniform_real(g, 10.0, 20.0); };
    std::array<Vec3, 3> w;
    for (auto& p : w) {
        p = draw3(rng, x, x, z);
    }
    return w;
}

Vec3 bearing(const AbsolutePose& pose, const Vec3& x) {
    return (pose.R * x + pose.t).normalized();
}

}  // namespace

TEST(AbsolutePose, P3PRecoversPoseOver200Scenes) {
    std::mt19937_64 rng(12345);
    int failures = 0;
    for (int trial = 0; trial < 200; ++trial) {
        const std::array<Vec3, 3> w = random_world(rng);
        const AbsolutePose gt = make_pose(rng);
        std::array<Vec3, 3> b;
        for (int i = 0; i < 3; ++i) {
            b[static_cast<std::size_t>(i)] = bearing(gt, w[static_cast<std::size_t>(i)]);
        }
        const std::vector<AbsolutePose> sols = p3p(w, b);
        bool found = false;
        for (const AbsolutePose& s : sols) {
            if ((s.R - gt.R).norm() < 1e-8 && (s.t - gt.t).norm() < 1e-8) {
                found = true;
                break;
            }
        }
        if (!found) {
            ++failures;
        }
    }
    std::cout << "[P3PRecoversPoseOver200Scenes] failures=" << failures << std::endl;
    EXPECT_LE(failures, 1);
}

TEST(AbsolutePose, P3PSolutionsReprojectExactly) {
    std::mt19937_64 rng(777);
    for (int trial = 0; trial < 50; ++trial) {
        const std::array<Vec3, 3> w = random_world(rng);
        const AbsolutePose gt = make_pose(rng);
        std::array<Vec3, 3> b;
        for (int i = 0; i < 3; ++i) {
            b[static_cast<std::size_t>(i)] = bearing(gt, w[static_cast<std::size_t>(i)]);
        }
        for (const AbsolutePose& s : p3p(w, b)) {
            for (int i = 0; i < 3; ++i) {
                const std::size_t k = static_cast<std::size_t>(i);
                EXPECT_LT(reprojection_angle_error(s, w[k], b[k]), 1e-9);
            }
        }
    }
}

TEST(AbsolutePose, WideAngle) {
    std::mt19937_64 rng(2024);
    int wide_trials = 0;
    int found_count = 0;
    for (int trial = 0; trial < 100; ++trial) {
        // World points on the far (negative-Z) side of the camera so that their
        // camera-frame z is negative: the bearing is then more than 90 degrees
        // off the +Z axis, yet the point is "in front" because it lies along
        // that same bearing.
        const auto u = [](std::mt19937_64& g) { return rng::uniform_real(g, -5.0, 5.0); };
        const auto nz = [](std::mt19937_64& g) { return rng::uniform_real(g, -20.0, -10.0); };
        std::array<Vec3, 3> w;
        for (auto& p : w) {
            p = draw3(rng, u, u, nz);
        }
        AbsolutePose gt;
        gt.R = random_rotation(rng, 20.0);
        gt.t = draw3(rng, u, u, u) * 0.5;
        std::array<Vec3, 3> b;
        bool any_wide = false;
        for (int i = 0; i < 3; ++i) {
            const Vec3 cam = gt.R * w[static_cast<std::size_t>(i)] + gt.t;
            b[static_cast<std::size_t>(i)] = cam.normalized();
            if (cam.z() < 0.0) {
                any_wide = true;
            }
        }
        if (!any_wide) {
            continue;
        }
        ++wide_trials;
        for (const AbsolutePose& s : p3p(w, b)) {
            if ((s.R - gt.R).norm() < 1e-6 && (s.t - gt.t).norm() < 1e-6) {
                ++found_count;
                break;
            }
        }
    }
    std::cout << "[WideAngle] wide_trials=" << wide_trials << std::endl;
    EXPECT_GT(wide_trials, 0);
    EXPECT_EQ(found_count, wide_trials);
}

TEST(AbsolutePose, RansacWithOutliers) {
    std::mt19937_64 rng(4242);
    const AbsolutePose gt = make_pose(rng);
    const int n = 200;
    const double sigma = 0.5 / 1000.0;
    const auto xy = [](std::mt19937_64& g) { return rng::uniform_real(g, -5.0, 5.0); };
    const auto z = [](std::mt19937_64& g) { return rng::uniform_real(g, 10.0, 20.0); };
    const auto u = [](std::mt19937_64& g) { return rng::uniform_real(g, -1.0, 1.0); };
    const auto noise = [sigma](std::mt19937_64& g) { return rng::normal(g, 0.0, sigma); };
    const auto prob = [](std::mt19937_64& g) { return rng::uniform_real(g, 0.0, 1.0); };

    std::vector<Vec3> world;
    std::vector<Vec3> bearing_vec;
    std::vector<std::uint8_t> is_outlier(n, 0);
    for (int i = 0; i < n; ++i) {
        Vec3 x = draw3(rng, xy, xy, z);
        Vec3 b = bearing(gt, x);
        if (prob(rng) < 0.4) {
            is_outlier[static_cast<std::size_t>(i)] = 1;
            // Random bearing far from the true projection.
            Vec3 r = draw3(rng, u, u, u);
            if (r.norm() < 1e-6) {
                r = Vec3::UnitZ();
            }
            b = r.normalized();
            // Random world point so the "world" is also inconsistent.
            x = draw3(rng, u, u, u) * 50.0;
        } else {
            // Add angular noise by perturbing the bearing on the tangent plane.
            Vec3 t1 = b.cross(Vec3::UnitZ());
            if (t1.norm() < 1e-6) {
                t1 = b.cross(Vec3::UnitX());
            }
            t1.normalize();
            const Vec3 t2 = b.cross(t1);
            // b + noise*t1 + noise*t2: g++ takes the right operand of the outer
            // + first, so the t2 noise was drawn before the t1 noise.
            const double n_t2 = noise(rng);
            const double n_t1 = noise(rng);
            b = (b + n_t1 * t1 + n_t2 * t2).normalized();
        }
        world.push_back(x);
        bearing_vec.push_back(b);
    }

    RansacOptions opt;
    opt.threshold_rad = 3.0 * sigma;
    opt.seed = 99;
    auto t0 = std::chrono::steady_clock::now();
    const AbsolutePoseResult res = estimate_absolute_pose_ransac(world, bearing_vec, opt);
    auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[RansacWithOutliers] ms=" << ms << " inliers=" << res.inlier_count
              << " iters=" << res.iterations << std::endl;
    EXPECT_TRUE(res.success);
    // Timing budget, not a correctness bound. This configuration is Debug with
    // ASan+UBSan and is several times slower than Release, so the sanitized
    // build gets the same order of magnitude more headroom.
#ifdef ALPHASHAPE_SANITIZED_BUILD
    EXPECT_LT(ms, 2000.0);
#else
    EXPECT_LT(ms, 100.0);
#endif

    const double rot_err = [&]() {
        const Mat3 dR = res.pose.R * gt.R.transpose();
        const double c = std::clamp((dR.trace() - 1.0) * 0.5, -1.0, 1.0);
        return std::acos(c) * 180.0 / 3.14159265358979;
    }();
    const Vec3 c_gt = -gt.R.transpose() * gt.t;
    const Vec3 c_est = -res.pose.R.transpose() * res.pose.t;
    const double centre_err = (c_est - c_gt).norm();
    EXPECT_LT(rot_err, 0.2);
    EXPECT_LT(centre_err, 0.01 * 15.0);  // 1% of the ~15-unit scene depth

    std::size_t tp = 0;
    std::size_t fp = 0;
    std::size_t fn = 0;
    for (int i = 0; i < n; ++i) {
        const bool inl = res.inliers[static_cast<std::size_t>(i)] != 0;
        const bool out = is_outlier[static_cast<std::size_t>(i)] != 0;
        if (inl && !out) {
            ++tp;
        } else if (inl && out) {
            ++fp;
        } else if (!inl && !out) {
            ++fn;
        }
    }
    const double recall = static_cast<double>(tp) / static_cast<double>(tp + fn);
    const double precision = static_cast<double>(tp) / static_cast<double>(tp + fp);
    std::cout << "[RansacWithOutliers] recall=" << recall << " precision=" << precision
              << std::endl;
    EXPECT_GT(recall, 0.90);
    EXPECT_GT(precision, 0.95);
}

TEST(AbsolutePose, RefinementImprovesNoisyPose) {
    std::mt19937_64 rng(31337);
    const AbsolutePose gt = make_pose(rng);
    const auto u = [](std::mt19937_64& g) { return rng::uniform_real(g, -1.0, 1.0); };
    std::vector<Vec3> world;
    std::vector<Vec3> bvec;
    for (int i = 0; i < 50; ++i) {
        const Vec3 uv = draw3(rng, u, u, u) * 5.0;
        const Vec3 x(uv.x(), uv.y(), 15.0 + uv.z());
        world.push_back(x);
        bvec.push_back(bearing(gt, x));
    }
    // Perturb by 1 degree and 0.5 units.
    AbsolutePose noisy = gt;
    noisy.R = Eigen::AngleAxisd(1.0 * 3.14159265358979 / 180.0, Vec3::UnitY()) *
              gt.R;
    noisy.t = gt.t + Vec3(0.5, 0.0, 0.0);

    const auto pose_error = [&](const AbsolutePose& p) {
        const double dR = std::acos(std::clamp(
            (p.R * gt.R.transpose()).trace() * 0.5 - 0.5, -1.0, 1.0));
        return dR + (p.t - gt.t).norm();
    };
    const double err_before = pose_error(noisy);
    const AbsolutePose refined = refine_absolute_pose(noisy, world, bvec);
    const double err_after = pose_error(refined);
    std::cout << "[RefinementImprovesNoisyPose] before=" << err_before
              << " after=" << err_after << std::endl;
    EXPECT_LT(err_after * 10.0, err_before);
}

TEST(AbsolutePose, Deterministic) {
    std::mt19937_64 rng(555);
    const AbsolutePose gt = make_pose(rng);
    const auto u = [](std::mt19937_64& g) { return rng::uniform_real(g, -1.0, 1.0); };
    std::vector<Vec3> world;
    std::vector<Vec3> bvec;
    for (int i = 0; i < 100; ++i) {
        const Vec3 uv = draw3(rng, u, u, u) * 5.0;
        const Vec3 x(uv.x(), uv.y(), 15.0 + uv.z());
        world.push_back(x);
        bvec.push_back(bearing(gt, x));
    }
    RansacOptions opt;
    opt.threshold_rad = 2.0 / 1000.0;
    opt.seed = 7;
    const AbsolutePoseResult a = estimate_absolute_pose_ransac(world, bvec, opt);
    const AbsolutePoseResult b = estimate_absolute_pose_ransac(world, bvec, opt);
    EXPECT_EQ(a.success, b.success);
    EXPECT_EQ(a.inlier_count, b.inlier_count);
    EXPECT_EQ(a.iterations, b.iterations);
    EXPECT_EQ(a.pose.R, b.pose.R);
    EXPECT_EQ(a.pose.t, b.pose.t);
}

TEST(AbsolutePose, DegenerateInputs) {
    std::mt19937_64 rng(888);
    const AbsolutePose gt = make_pose(rng);
    const auto u = [](std::mt19937_64& g) { return rng::uniform_real(g, -1.0, 1.0); };

    // Two points -> success = false.
    {
        std::vector<Vec3> world;
        std::vector<Vec3> bvec;
        for (int i = 0; i < 2; ++i) {
            const double x0 = u(rng);
            const double x1 = u(rng);
            const Vec3 x(x0, x1, 15.0);
            world.push_back(x);
            bvec.push_back(bearing(gt, x));
        }
        RansacOptions opt;
        opt.threshold_rad = 2.0 / 1000.0;
        const AbsolutePoseResult res = estimate_absolute_pose_ransac(world, bvec, opt);
        EXPECT_FALSE(res.success);
    }
    // Size mismatch -> ValueError.
    {
        std::vector<Vec3> world(3, Vec3(0.0, 0.0, 15.0));
        std::vector<Vec3> bvec(2, Vec3(0.0, 0.0, 1.0));
        RansacOptions opt;
        opt.threshold_rad = 2.0 / 1000.0;
        EXPECT_THROW(estimate_absolute_pose_ransac(world, bvec, opt), ValueError);
    }
    // Threshold <= 0 -> ValueError.
    {
        std::vector<Vec3> world;
        std::vector<Vec3> bvec;
        for (int i = 0; i < 10; ++i) {
            const double x0 = u(rng);
            const double x1 = u(rng);
            const Vec3 x(x0, x1, 15.0);
            world.push_back(x);
            bvec.push_back(bearing(gt, x));
        }
        RansacOptions opt;
        opt.threshold_rad = 0.0;
        EXPECT_THROW(estimate_absolute_pose_ransac(world, bvec, opt), ValueError);
    }
    // All-random input: no crash, and no meaningful inlier set.
    {
        std::vector<Vec3> world;
        std::vector<Vec3> bvec;
        for (int i = 0; i < 100; ++i) {
            world.push_back(draw3(rng, u, u, u) * 50.0);
            Vec3 b = draw3(rng, u, u, u);
            bvec.push_back(b.normalized());
        }
        RansacOptions opt;
        opt.threshold_rad = 1e-4;
        opt.max_iterations = 200;
        const AbsolutePoseResult res = estimate_absolute_pose_ransac(world, bvec, opt);
        EXPECT_TRUE(!res.success || res.inlier_count < 10);
    }
}

}  // namespace alphashape::sfm
