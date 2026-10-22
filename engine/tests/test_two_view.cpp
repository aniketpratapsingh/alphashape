// test_two_view.cpp -- tests for essential-matrix relative pose, triangulation
// and RANSAC.  Clean-room; see essential.hpp.

#include "common/algorithm/sfm/essential.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"
#include "test_rng_helpers.hpp"

namespace {

using alphashape::sfm::essential_from_pose;
using alphashape::sfm::RelativePose;
using test_rng::draw2;
using test_rng::draw3;

constexpr double kPi = 3.14159265358979323846;

// Portable draws, not std::distributions: those map engine bits to values
// differently under libc++ and libstdc++, so a seed selected a different scene
// on macOS and Linux (TESTING.md, "Numeric consistency").
const auto gauss = [](std::mt19937_64& g) { return alphashape::rng::normal(g, 0.0, 1.0); };

Eigen::Vector3d random_unit(std::mt19937_64& rng) {
    // draw3 sequences the three draws in g++'s right-to-left order, the order
    // these scenes were calibrated with (see test_rng_helpers.hpp).
    return draw3(rng, gauss, gauss, gauss).normalized();
}

Eigen::Matrix3d random_small_rotation(std::mt19937_64& rng) {
    const Eigen::Vector3d axis = random_unit(rng);
    const double a = alphashape::rng::uniform_real(rng, 0.0, 20.0 * kPi / 180.0);
    return Eigen::AngleAxisd(a, axis).toRotationMatrix();
}

struct Scene {
    RelativePose pose;
    std::vector<Eigen::Vector3d> X;
    std::vector<Eigen::Vector3d> b1;
    std::vector<Eigen::Vector3d> b2;
};

// Deterministic scene: small rotation, random unit translation, points with
// depth 3..15 in front of BOTH cameras.
Scene make_scene(std::uint64_t seed, int n, bool allow_negative_z = false) {
    std::mt19937_64 rng(seed);
    Scene s;
    s.pose.R = random_small_rotation(rng);
    s.pose.t = random_unit(rng);
    const auto depth = [](std::mt19937_64& g) { return alphashape::rng::uniform_real(g, 3.0, 15.0); };
    const auto lateral = [](std::mt19937_64& g) { return alphashape::rng::uniform_real(g, -10.0, 10.0); };
    const auto negdepth = [](std::mt19937_64& g) { return alphashape::rng::uniform_real(g, -15.0, 15.0); };
    while (static_cast<int>(s.X.size()) < n) {
        Eigen::Vector3d X;
        X.x() = lateral(rng);
        X.y() = lateral(rng);
        // Only one arm of the conditional runs, so the statement draws once;
        // the explicit branch keeps the engine call out of the condition.
        double z = 0.0;
        if (allow_negative_z) {
            z = negdepth(rng);
        } else {
            z = depth(rng);
        }
        X.z() = z;
        if (X.norm() < 1e-6) {
            continue;
        }
        const Eigen::Vector3d X2 = s.pose.R * X + s.pose.t;
        if (X2.z() <= 0.0) {
            continue;  // require in front of camera 2
        }
        s.X.push_back(X);
        s.b1.push_back(X.normalized());
        s.b2.push_back(X2.normalized());
    }
    return s;
}

double rotation_angle(const Eigen::Matrix3d& R) {
    const double tr = (R.trace() - 1.0) * 0.5;
    return std::acos(std::max(-1.0, std::min(1.0, tr)));
}

}  // namespace

TEST(TwoView, EssentialFromPoseSatisfiesEpipolar) {
    const Scene s = make_scene(1, 20);
    const Eigen::Matrix3d E = essential_from_pose(s.pose);
    for (std::size_t i = 0; i < s.X.size(); ++i) {
        const double res = std::abs(s.b2[i].dot(E * s.b1[i]));
        EXPECT_LT(res, 1e-12);
        EXPECT_LT(alphashape::sfm::epipolar_angle_error(E, s.b1[i], s.b2[i]), 1e-12);
    }
}

TEST(TwoView, DecomposeRecoversPose) {
    for (std::uint64_t seed = 0; seed < 50; ++seed) {
        const Scene s = make_scene(100 + seed, 12);
        const Eigen::Matrix3d E = essential_from_pose(s.pose);
        const auto pose = alphashape::sfm::decompose_essential(E, s.b1, s.b2);
        ASSERT_TRUE(pose.has_value());
        EXPECT_LT((pose->R - s.pose.R).norm(), 1e-9);
        const double t_err = (pose->t - s.pose.t).norm();
        const double t_err_flip = (pose->t + s.pose.t).norm();
        EXPECT_LT(std::min(t_err, t_err_flip), 1e-9);
    }
}

TEST(TwoView, TriangulateRecoversPoint) {
    const Scene s = make_scene(7, 15);
    for (std::size_t i = 0; i < s.X.size(); ++i) {
        const auto x = alphashape::sfm::triangulate(s.pose, s.b1[i], s.b2[i]);
        ASSERT_TRUE(x.has_value());
        const double rel = (*x - s.X[i]).norm() / s.X[i].norm();
        EXPECT_LT(rel, 1e-9);
    }
    // Parallel rays: pure rotation, point at infinity, identical bearings.
    RelativePose par;
    par.R = Eigen::Matrix3d::Identity();
    par.t = s.b1[0];
    const auto x = alphashape::sfm::triangulate(par, s.b1[0], s.b1[0]);
    EXPECT_FALSE(x.has_value());
}

TEST(TwoView, WideAngleCheirality) {
    const Scene s = make_scene(3, 20, /*allow_negative_z=*/true);
    const Eigen::Matrix3d E = essential_from_pose(s.pose);
    const auto pose = alphashape::sfm::decompose_essential(E, s.b1, s.b2);
    ASSERT_TRUE(pose.has_value());
    EXPECT_LT((pose->R - s.pose.R).norm(), 1e-9);
    std::size_t in_front = 0;
    for (std::size_t i = 0; i < s.X.size(); ++i) {
        const auto x = alphashape::sfm::triangulate(*pose, s.b1[i], s.b2[i]);
        if (!x.has_value()) {
            continue;
        }
        const Eigen::Vector3d x2 = pose->R * (*x) + pose->t;
        if (x->dot(s.b1[i]) > 0.0 && x2.dot(s.b2[i]) > 0.0) {
            ++in_front;
        }
    }
    EXPECT_EQ(in_front, s.X.size());
}

TEST(TwoView, RansacRecoversPoseWithOutliers) {
    const std::uint64_t seed = 42;
    std::mt19937_64 rng(seed);
    const Scene gt = make_scene(seed, 300);

    const auto ang_noise = [](std::mt19937_64& g) { return alphashape::rng::normal(g, 0.0, 0.5 / 1000.0); };
    // A Bernoulli(0.4) is one uniform draw compared against the probability.
    const auto is_outlier = [](std::mt19937_64& g) { return alphashape::rng::unit_interval(g) < 0.4; };

    std::vector<Eigen::Vector3d> b1 = gt.b1;
    std::vector<Eigen::Vector3d> b2(gt.b2.size());
    std::vector<bool> gt_inlier(gt.b2.size(), true);
    for (std::size_t i = 0; i < gt.b2.size(); ++i) {
        if (is_outlier(rng)) {
            gt_inlier[i] = false;
            b2[i] = draw3(rng, gauss, gauss, gauss).normalized();
        } else {
            // Perturb the inlier bearing in a random perpendicular direction.
            const Eigen::Vector3d r = draw3(rng, gauss, gauss, gauss);
            const Eigen::Vector3d perp = r.cross(gt.b2[i]);
            const Eigen::Vector3d perturbed =
                gt.b2[i] + ang_noise(rng) * perp.normalized();
            b2[i] = perturbed.normalized();
        }
    }

    alphashape::sfm::RansacOptions opt;
    opt.threshold_rad = 3.0 * 0.5 / 1000.0;
    opt.seed = 0x5eed;

    const auto t0 = std::chrono::steady_clock::now();
    const alphashape::sfm::EssentialResult res =
        alphashape::sfm::estimate_essential_ransac(b1, b2, opt);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[RansacRecoversPoseWithOutliers] " << ms
              << " ms, iterations=" << res.iterations
              << ", inliers=" << res.inlier_count << std::endl;

    ASSERT_TRUE(res.success);
    const double rot_err = rotation_angle(res.pose.R.transpose() * gt.pose.R);
    EXPECT_LT(rot_err, 0.5 * kPi / 180.0);
    const double t_dir =
        std::acos(std::min(1.0, std::abs(res.pose.t.dot(gt.pose.t))));
    EXPECT_LT(t_dir, 2.0 * kPi / 180.0);

    std::size_t tp = 0, fp = 0, fn = 0;
    for (std::size_t i = 0; i < b1.size(); ++i) {
        const bool reported = res.inliers[i] != 0;
        if (reported && gt_inlier[i]) {
            ++tp;
        } else if (reported && !gt_inlier[i]) {
            ++fp;
        } else if (!reported && gt_inlier[i]) {
            ++fn;
        }
    }
    const double recall = static_cast<double>(tp) / static_cast<double>(tp + fn);
    const double precision = static_cast<double>(tp) / static_cast<double>(tp + fp);
    EXPECT_GT(recall, 0.90);
    EXPECT_GT(precision, 0.95);
}

TEST(TwoView, RansacIsDeterministic) {
    const Scene gt = make_scene(11, 60);
    std::mt19937_64 rng(99);
    const auto is_outlier = [](std::mt19937_64& g) { return alphashape::rng::unit_interval(g) < 0.3; };
    std::vector<Eigen::Vector3d> b2(gt.b2.size());
    for (std::size_t i = 0; i < gt.b2.size(); ++i) {
        if (is_outlier(rng)) {
            b2[i] = draw3(rng, gauss, gauss, gauss).normalized();
        } else {
            b2[i] = gt.b2[i];
        }
    }
    alphashape::sfm::RansacOptions opt;
    opt.threshold_rad = 1e-3;
    opt.seed = 0x5eed;
    const auto a = alphashape::sfm::estimate_essential_ransac(gt.b1, b2, opt);
    const auto b = alphashape::sfm::estimate_essential_ransac(gt.b1, b2, opt);
    EXPECT_EQ((a.E - b.E).norm(), 0.0);
    EXPECT_EQ(a.inliers, b.inliers);
    EXPECT_EQ(a.iterations, b.iterations);
}

TEST(TwoView, RansacDegenerateInputs) {
    const Scene few = make_scene(5, 4);
    alphashape::sfm::RansacOptions opt;
    opt.threshold_rad = 1e-3;
    const auto r_few = alphashape::sfm::estimate_essential_ransac(few.b1, few.b2, opt);
    EXPECT_FALSE(r_few.success);

    const Scene big = make_scene(6, 40);
    std::vector<Eigen::Vector3d> short_b1(big.b1.begin(), big.b1.begin() + 10);
    EXPECT_THROW(alphashape::sfm::estimate_essential_ransac(short_b1, big.b2, opt),
                 alphashape::ValueError);

    std::mt19937_64 rng(123);
    std::vector<Eigen::Vector3d> r1(100), r2(100);
    for (std::size_t i = 0; i < 100; ++i) {
        r1[i] = draw3(rng, gauss, gauss, gauss).normalized();
        r2[i] = draw3(rng, gauss, gauss, gauss).normalized();
    }
    const auto r_rand = alphashape::sfm::estimate_essential_ransac(r1, r2, opt);
    EXPECT_TRUE(!r_rand.success || r_rand.inlier_count < 20);
}

TEST(TwoView, RansacPlanarScene) {
    const std::uint64_t seed = 77;
    std::mt19937_64 rng(seed);
    // The braced form is sequenced left to right by the standard, so keep the
    // rotation draw first; the two locals also keep the guard quiet.
    const Eigen::Matrix3d R = random_small_rotation(rng);
    const Eigen::Vector3d t = random_unit(rng);
    const RelativePose pose{R, t};
    const auto lateral = [](std::mt19937_64& g) { return alphashape::rng::uniform_real(g, -8.0, 8.0); };
    std::vector<Eigen::Vector3d> b1, b2;
    for (int i = 0; i < 60; ++i) {
        // Plane z = 5; the two lateral draws are sequenced as g++ did.
        const Eigen::Vector2d xy = draw2(rng, lateral, lateral);
        const Eigen::Vector3d X(xy.x(), xy.y(), 5.0);
        const Eigen::Vector3d X2 = pose.R * X + pose.t;
        if (X2.z() <= 0.0) {
            continue;
        }
        b1.push_back(X.normalized());
        b2.push_back(X2.normalized());
    }
    alphashape::sfm::RansacOptions opt;
    opt.threshold_rad = 1e-4;
    opt.seed = 0x5eed;
    const auto res = alphashape::sfm::estimate_essential_ransac(b1, b2, opt);
    EXPECT_TRUE(res.success);
    const Eigen::Matrix3d E = res.E;
    for (std::size_t i = 0; i < b1.size(); ++i) {
        if (res.inliers[i] != 0) {
            EXPECT_LT(alphashape::sfm::epipolar_angle_error(E, b1[i], b2[i]),
                      opt.threshold_rad);
        }
    }
}
