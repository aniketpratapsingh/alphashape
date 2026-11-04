// WP-13.11a: control points, check points and scale bars in the bundle
// adjustment (suite Optimize). Synthetic network with a known truth pose.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"
#include "common/algorithm/sfm/bundle_adjustment.hpp"
#include "model/camera/calibration.hpp"
#include "test_rng_helpers.hpp"

namespace alphashape::sfm {
namespace {

using test_rng::draw3;

constexpr int kCameras = 8;
constexpr int kPoints = 200;

Eigen::Matrix3d rotation_from_angle_axis(const Eigen::Vector3d& aa) {
    const Eigen::Vector3d axis =
        aa.norm() > 1e-12 ? aa.normalized() : Eigen::Vector3d::UnitX();
    return Eigen::AngleAxisd(aa.norm(), axis).toRotationMatrix();
}

Calibration frame_calibration() {
    Calibration c;
    c.type = CameraType::Frame;
    c.width = 640;
    c.height = 480;
    c.f = 600.0;
    // cx/cy are offsets from the image centre; 0 is a centred principal point.
    return c;
}

struct SyntheticScene {
    BAProblem problem;
    std::vector<Eigen::Vector3d> true_centres;
    std::vector<Eigen::Vector3d> true_points;
    std::vector<BAPose> true_poses;
};

/// Apply X -> scale * (q X) + shift to every point and pose. Pose update:
/// R' = R q^T, t' = scale t - R' shift, which keeps every camera coordinate
/// scaled by the same factor and therefore every projection unchanged.
void apply_similarity(BAProblem& problem, double scale, const Eigen::Matrix3d& q,
                      const Eigen::Vector3d& shift) {
    for (Eigen::Vector3d& p : problem.points) {
        p = scale * (q * p) + shift;
    }
    for (BAPose& pose : problem.poses) {
        pose.R = pose.R * q.transpose();
        pose.t = scale * pose.t - pose.R * shift;
    }
}

/// Noise-free projections for every point/camera pair that lands in frame.
void add_observations(SyntheticScene& scene, const Calibration& cal) {
    for (std::size_t pt = 0; pt < scene.true_points.size(); ++pt) {
        for (std::size_t cam = 0; cam < static_cast<std::size_t>(kCameras); ++cam) {
            const BAPose& pose = scene.true_poses[cam];
            const Eigen::Vector3d p_cam = pose.R * scene.true_points[pt] + pose.t;
            Eigen::Vector2d px;
            try {
                px = cal.project(p_cam);
            } catch (const ValueError&) {
                continue;
            }
            if (px.x() < 0.0 || px.x() > static_cast<double>(cal.width) || px.y() < 0.0 ||
                px.y() > static_cast<double>(cal.height)) {
                continue;
            }
            scene.problem.observations.push_back(BAObservation{cam, pt, px});
        }
    }
}

/// Ring of 8 cameras looking at the origin, 200 points in [-2,2]^3.
SyntheticScene make_scene(std::mt19937& rng) {
    SyntheticScene scene;
    const Calibration cal = frame_calibration();
    scene.problem.calibrations.assign(static_cast<std::size_t>(kCameras), cal);

    const auto coord = [](std::mt19937& g) { return rng::uniform_real(g, -2.0, 2.0); };
    for (int i = 0; i < kPoints; ++i) {
        const Eigen::Vector3d p = draw3(rng, coord, coord, coord);
        scene.true_points.push_back(p);
        scene.problem.points.push_back(p);
    }

    constexpr double kPi = 3.14159265358979323846;
    for (int i = 0; i < kCameras; ++i) {
        const double angle =
            2.0 * kPi * static_cast<double>(i) / static_cast<double>(kCameras);
        const Eigen::Vector3d centre =
            6.0 * Eigen::Vector3d(std::sin(angle), 0.0, std::cos(angle));
        scene.true_centres.push_back(centre);
        const Eigen::Vector3d forward = (-centre).normalized();
        const Eigen::Vector3d right = forward.cross(Eigen::Vector3d::UnitY()).normalized();
        const Eigen::Vector3d up = right.cross(forward);
        Eigen::Matrix3d r_wc;  // columns: right, down, forward (camera +Y is down)
        r_wc.col(0) = right;
        r_wc.col(1) = -up;
        r_wc.col(2) = forward;
        BAPose pose;
        pose.R = r_wc.transpose();
        pose.t = -pose.R * centre;
        scene.true_poses.push_back(pose);
        scene.problem.poses.push_back(pose);
    }

    add_observations(scene, cal);
    scene.problem.fixed_pose.assign(scene.problem.poses.size(), 0);
    return scene;
}

TEST(Optimize, GcpNetworkRecoversTransform) {
    std::mt19937 rng(20261104);
    const SyntheticScene truth = make_scene(rng);

    BAProblem problem = truth.problem;
    const Eigen::Matrix3d q = rotation_from_angle_axis(Eigen::Vector3d(0.2, -0.3, 0.15));
    apply_similarity(problem, 1.3, q, Eigen::Vector3d(2.0, -1.0, 3.0));

    const std::array<std::size_t, 5> control = {0, 17, 42, 88, 150};
    ASSERT_FALSE(truth.problem.observations.empty());
    for (const std::size_t index : control) {
        std::size_t seen = 0;
        for (const BAObservation& obs : truth.problem.observations) {
            if (obs.point == index) {
                ++seen;
            }
        }
        ASSERT_GE(seen, 2u);
    }
    for (const std::size_t index : control) {
        BAControlPoint cp;
        cp.point = index;
        cp.reference = truth.true_points[index];
        cp.sigma = Eigen::Vector3d::Constant(0.01);
        problem.control_points.push_back(cp);
    }

    BAOptions options;
    options.max_iterations = 1000;
    options.function_tolerance = 1e-12;
    // The start is an exact similarity, so every reprojection is zero. A robust
    // reprojection loss can then prefer a spurious local minimum where only the
    // control points move; this test is noise-free, so plain least squares.
    options.loss_scale_px = 0.0;
    const BAReport report = bundle_adjust(problem, options);
    ASSERT_TRUE(report.usable);
    EXPECT_EQ(report.control_points, control.size());

    Eigen::Vector3d lo = truth.true_points[0];
    Eigen::Vector3d hi = lo;
    for (const Eigen::Vector3d& p : truth.true_points) {
        lo = lo.cwiseMin(p);
        hi = hi.cwiseMax(p);
    }
    const double scene_size = (hi - lo).norm();
    for (std::size_t i = 0; i < problem.poses.size(); ++i) {
        const Eigen::Vector3d centre = -problem.poses[i].R.transpose() * problem.poses[i].t;
        EXPECT_LT((centre - truth.true_centres[i]).norm(), 1e-3 * scene_size);
    }
}

TEST(Optimize, GcpNetworkRecoversTransformWithCauchy) {
    std::mt19937 rng(20261104);
    const SyntheticScene truth = make_scene(rng);

    BAProblem problem = truth.problem;
    const Eigen::Matrix3d q = rotation_from_angle_axis(Eigen::Vector3d(0.2, -0.3, 0.15));
    apply_similarity(problem, 1.3, q, Eigen::Vector3d(2.0, -1.0, 3.0));

    const std::array<std::size_t, 5> control = {0, 17, 42, 88, 150};
    for (const std::size_t index : control) {
        BAControlPoint cp;
        cp.point = index;
        cp.reference = truth.true_points[index];
        cp.sigma = Eigen::Vector3d::Constant(0.01);
        problem.control_points.push_back(cp);
    }

    // Default options: the shared Cauchy loss is active. Observations of the
    // control points switch to Huber, so the rays keep pulling the poses and the
    // exact-similarity start converges to the truth instead of stalling with the
    // markers detached from their rays.
    BAOptions options;
    options.max_iterations = 1000;
    options.function_tolerance = 1e-12;
    const BAReport report = bundle_adjust(problem, options);
    ASSERT_TRUE(report.usable);
    EXPECT_EQ(report.control_points, control.size());
    EXPECT_LT(report.control_rms_m, 0.02);

    Eigen::Vector3d lo = truth.true_points[0];
    Eigen::Vector3d hi = lo;
    for (const Eigen::Vector3d& p : truth.true_points) {
        lo = lo.cwiseMin(p);
        hi = hi.cwiseMax(p);
    }
    const double scene_size = (hi - lo).norm();
    for (std::size_t i = 0; i < problem.poses.size(); ++i) {
        const Eigen::Vector3d centre = -problem.poses[i].R.transpose() * problem.poses[i].t;
        EXPECT_LT((centre - truth.true_centres[i]).norm(), 1e-3 * scene_size);
    }
}

TEST(Optimize, ScalebarFixesScale) {
    std::mt19937 rng(20261105);
    const SyntheticScene truth = make_scene(rng);
    BAProblem problem = truth.problem;

    // Similarity about camera 0's centre: that pose is unchanged bitwise, every
    // projection stays exact, and the only thing that breaks the scale freedom
    // is the scale bar.
    const double start_scale = 0.7;
    const Eigen::Vector3d& c0 = truth.true_centres[0];
    apply_similarity(problem, start_scale, Eigen::Matrix3d::Identity(),
                     (1.0 - start_scale) * c0);
    problem.fixed_pose[0] = 1;

    BAScalebar bar;
    bar.point_a = 0;
    bar.point_b = 100;
    bar.length = (truth.true_points[0] - truth.true_points[100]).norm();
    bar.sigma = 0.001;
    ASSERT_GT(bar.length, 0.1);
    problem.scalebars.push_back(bar);

    BAOptions options;
    options.max_iterations = 1000;
    options.function_tolerance = 1e-12;
    const BAReport report = bundle_adjust(problem, options);
    ASSERT_TRUE(report.usable);
    EXPECT_EQ(report.scalebars, 1u);

    const double distance = (problem.points[0] - problem.points[100]).norm();
    EXPECT_LT(std::abs(distance / bar.length - 1.0), 1e-4);
}

TEST(Optimize, RobustLossRejectsBlunder) {
    std::mt19937 rng(20261108);
    const SyntheticScene truth = make_scene(rng);
    BAProblem problem = truth.problem;
    const Eigen::Matrix3d q = rotation_from_angle_axis(Eigen::Vector3d(-0.15, 0.25, -0.1));
    apply_similarity(problem, 1.2, q, Eigen::Vector3d(-1.5, 2.0, -2.5));

    const std::array<std::size_t, 7> control = {0, 17, 42, 88, 150, 3, 120};
    constexpr std::size_t kBlunderRank = 6;
    for (std::size_t i = 0; i < control.size(); ++i) {
        BAControlPoint cp;
        cp.point = control[i];
        cp.reference = truth.true_points[control[i]];
        if (i == kBlunderRank) {
            cp.reference.x() += 5.0;
        }
        cp.sigma = Eigen::Vector3d::Constant(0.01);
        problem.control_points.push_back(cp);
    }

    BAOptions options;
    options.max_iterations = 1000;
    options.function_tolerance = 1e-12;
    // The blunder is in a control reference, not in an observation. Plain least
    // squares keeps the bad point pinned by its (noise-free) observations so the
    // test isolates the Huber loss on the control residual.
    options.loss_scale_px = 0.0;
    const BAReport report = bundle_adjust(problem, options);
    ASSERT_TRUE(report.usable);

    for (std::size_t i = 0; i < control.size(); ++i) {
        if (i == kBlunderRank) {
            continue;
        }
        const double residual =
            (problem.points[control[i]] - truth.true_points[control[i]]).norm();
        EXPECT_LT(residual, 0.02);
    }
}

TEST(Optimize, CheckPointsReportedButNotUsed) {
    std::mt19937 rng(20261106);
    const SyntheticScene truth = make_scene(rng);
    const std::array<std::size_t, 5> control = {0, 17, 42, 88, 150};
    constexpr std::size_t kCheck = 199;

    BAProblem with_check = truth.problem;
    const Eigen::Matrix3d q = rotation_from_angle_axis(Eigen::Vector3d(0.1, 0.2, -0.3));
    apply_similarity(with_check, 1.3, q, Eigen::Vector3d(1.0, -2.0, 0.5));

    for (const std::size_t index : control) {
        BAControlPoint cp;
        cp.point = index;
        cp.reference = truth.true_points[index];
        cp.sigma = Eigen::Vector3d::Constant(0.01);
        with_check.control_points.push_back(cp);
    }
    BAControlPoint check;
    check.point = kCheck;
    check.reference = truth.true_points[kCheck];
    check.sigma = Eigen::Vector3d::Constant(0.01);
    check.checkpoint = true;
    with_check.control_points.push_back(check);
    with_check.checkpoint_point.assign(with_check.points.size(), 0);
    with_check.checkpoint_point[kCheck] = 1;

    // The same initial problem with the check point and its observations cut.
    BAProblem without = with_check;
    without.control_points.pop_back();
    without.checkpoint_point.clear();
    std::vector<BAObservation> kept;
    for (const BAObservation& obs : without.observations) {
        if (obs.point != kCheck) {
            kept.push_back(obs);
        }
    }
    without.observations = kept;

    // Same inputs, but the check point is expressed ONLY through
    // BAControlPoint::checkpoint: the mask is left empty. The derived effective
    // mask must make this solve identical to the masked run.
    BAProblem flag_only = with_check;
    flag_only.checkpoint_point.clear();

    BAOptions options;
    options.max_iterations = 1000;
    options.function_tolerance = 1e-12;
    options.loss_scale_px = 0.0;  // noise-free, exact-similarity start (see above)
    const BAReport report = bundle_adjust(with_check, options);
    ASSERT_TRUE(report.usable);
    ASSERT_EQ(report.checkpoints, 1u);
    ASSERT_EQ(report.checkpoint_residuals.size(), 1u);
    EXPECT_TRUE(std::isfinite(report.checkpoint_rms_m));
    EXPECT_LT(report.checkpoint_rms_m, 1e-3);

    const BAReport flag_report = bundle_adjust(flag_only, options);
    ASSERT_TRUE(flag_report.usable);
    ASSERT_EQ(flag_report.checkpoints, 1u);
    for (std::size_t i = 0; i < with_check.poses.size(); ++i) {
        EXPECT_LT((with_check.poses[i].R - flag_only.poses[i].R).norm(), 1e-9);
        EXPECT_LT((with_check.poses[i].t - flag_only.poses[i].t).norm(), 1e-9);
    }

    const BAReport plain = bundle_adjust(without, options);
    ASSERT_TRUE(plain.usable);
    for (std::size_t i = 0; i < with_check.poses.size(); ++i) {
        EXPECT_LT((with_check.poses[i].R - without.poses[i].R).norm(), 1e-9);
        EXPECT_LT((with_check.poses[i].t - without.poses[i].t).norm(), 1e-9);
    }
}

TEST(Optimize, ScalebarJacobianMatchesAutodiff) {
    // Ceres is a private dependency of the algorithms layer, so autodiff is not
    // reachable from this test; compare the analytic Jacobian with central
    // differences of the same residual instead.
    std::mt19937 rng(20261107);
    const auto coord = [](std::mt19937& g) { return rng::uniform_real(g, -5.0, 5.0); };
    constexpr double kLength = 3.0;
    constexpr double kSigma = 0.01;
    constexpr double kStep = 1e-6;

    int compared = 0;
    for (int trial = 0; trial < 40; ++trial) {
        const Eigen::Vector3d a = draw3(rng, coord, coord, coord);
        const Eigen::Vector3d b = draw3(rng, coord, coord, coord);
        if ((a - b).norm() < 0.5) {
            continue;
        }
        double residual = 0.0;
        double jac_a[3] = {};
        double jac_b[3] = {};
        detail::scalebar_residual_and_jacobian(a.data(), b.data(), kLength, kSigma,
                                               &residual, jac_a, jac_b);
        for (int k = 0; k < 3; ++k) {
            double rp = 0.0;
            double rm = 0.0;
            Eigen::Vector3d ap = a;
            Eigen::Vector3d am = a;
            ap[k] += kStep;
            am[k] -= kStep;
            detail::scalebar_residual_and_jacobian(ap.data(), b.data(), kLength, kSigma,
                                                   &rp, nullptr, nullptr);
            detail::scalebar_residual_and_jacobian(am.data(), b.data(), kLength, kSigma,
                                                   &rm, nullptr, nullptr);
            EXPECT_LT(std::abs((rp - rm) / (2.0 * kStep) - jac_a[k]), 1e-6);

            Eigen::Vector3d bp = b;
            Eigen::Vector3d bm = b;
            bp[k] += kStep;
            bm[k] -= kStep;
            detail::scalebar_residual_and_jacobian(a.data(), bp.data(), kLength, kSigma,
                                                   &rp, nullptr, nullptr);
            detail::scalebar_residual_and_jacobian(a.data(), bm.data(), kLength, kSigma,
                                                   &rm, nullptr, nullptr);
            EXPECT_LT(std::abs((rp - rm) / (2.0 * kStep) - jac_b[k]), 1e-6);
        }
        ++compared;
    }
    EXPECT_GT(compared, 0);
}

}  // namespace
}  // namespace alphashape::sfm
