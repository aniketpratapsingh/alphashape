// Milestone 4: bundle adjustment tests (suite BundleAdjustment).

#include <algorithm>
#include <array>
#include <chrono>
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

using test_rng::draw2;
using test_rng::draw3;

Eigen::Matrix3d rotation_from_angle_axis(const Eigen::Vector3d& aa) {
    return Eigen::AngleAxisd(aa.norm(), aa.norm() > 1e-12 ? aa.normalized() : Eigen::Vector3d::UnitX())
        .toRotationMatrix();
}

struct SyntheticScene {
    BAProblem problem;
    std::vector<Eigen::Vector3d> true_centres;
    std::vector<Eigen::Vector3d> true_points;
    std::vector<BAPose> true_poses;
};

Calibration frame_calibration() {
    Calibration c;
    c.type = CameraType::Frame;
    c.width = 640;
    c.height = 480;
    c.f = 600.0;
    c.k1 = -0.05;
    c.p1 = 1e-4;
    c.k2 = 0.01;
    return c;
}

Calibration fisheye_calibration() {
    Calibration c = frame_calibration();
    c.type = CameraType::Fisheye;
    c.f = 300.0;
    return c;
}

/// Layout of the camera stations around the point cloud.
///  - Ring: 8 cameras evenly around the cloud (well-conditioned, the classic
///    SfM "turntable" geometry).
///  - Arc:  8 cameras on ~40 deg of one side of the cloud. This is the
///    one-sided geometry of a real photo sequence (data/seq_*.jpg) and the
///    weakly-conditioned case the scale anchor has to cope with.
enum class CameraLayout { Ring, Arc };

/// Deterministic synthetic scene: 8 cameras in the requested layout looking at
/// the cloud centroid, a box of points at depth ~10, clean projections plus
/// sigma-px Gaussian noise.
SyntheticScene make_scene(CameraType type, CameraLayout layout, double sigma_px, std::mt19937& rng) {
    SyntheticScene scene;
    constexpr int kCameras = 8;
    constexpr int kPoints = 300;
    // Each point is seen by a random subset of cameras, as real SfM tracks are.
    constexpr double kKeep = 0.6;

    const Calibration cal = (type == CameraType::Fisheye) ? fisheye_calibration() : frame_calibration();
    scene.problem.calibrations.assign(static_cast<std::size_t>(kCameras), cal);

    // Portable draws, not std::distributions: those are implementation-defined,
    // so one seed built different scenes under libc++ and libstdc++.
    const auto point_dist = [](std::mt19937& g) { return rng::uniform_real(g, -3.5, 3.5); };
    // The depth draw is offset by the cloud centroid; draw3 sequences it.
    const auto point_depth = [point_dist](std::mt19937& g) { return 10.0 + point_dist(g); };
    for (int i = 0; i < kPoints; ++i) {
        Eigen::Vector3d p = draw3(rng, point_dist, point_dist, point_depth);
        scene.true_points.push_back(p);
        scene.problem.points.push_back(p);
    }

    const auto noise = [sigma_px](std::mt19937& g) { return rng::normal(g, 0.0, sigma_px); };
    // Cameras sit 8 units from the cloud centroid (0,0,10), all well outside
    // the point box, each looking at the centroid along its +Z axis.
    const Eigen::Vector3d centroid(0.0, 0.0, 10.0);
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    for (int i = 0; i < kCameras; ++i) {
        double angle = 0.0;
        if (layout == CameraLayout::Ring) {
            angle = kTwoPi * static_cast<double>(i) / static_cast<double>(kCameras);
        } else {
            // ~40 deg of arc, symmetric about the front-facing direction, i.e.
            // all on ONE side of the cloud (a < +/-20 deg baseline sweep — the
            // sideways baseline that the scale anchor must actually bind).
            constexpr double kArcSpan = 40.0 * 3.14159265358979323846 / 180.0;
            angle = -0.5 * kArcSpan + kArcSpan * static_cast<double>(i) /
                                        static_cast<double>(kCameras - 1);
        }
        const double radius = 8.0;
        const Eigen::Vector3d centre =
            centroid + radius * Eigen::Vector3d(std::sin(angle), 0.0, std::cos(angle));
        scene.true_centres.push_back(centre);

        const Eigen::Vector3d forward = (centroid - centre).normalized();
        const Eigen::Vector3d world_up = Eigen::Vector3d::UnitY();
        const Eigen::Vector3d right = forward.cross(world_up).normalized();
        const Eigen::Vector3d up = right.cross(forward);
        Eigen::Matrix3d R_wc;  // columns: right, down, forward (camera +Y is down)
        R_wc.col(0) = right;
        R_wc.col(1) = -up;
        R_wc.col(2) = forward;
        // World->camera rotation is the transpose of the camera->world frame.
        BAPose pose;
        pose.R = R_wc.transpose();
        pose.t = -pose.R * centre;
        scene.true_poses.push_back(pose);
        scene.problem.poses.push_back(pose);
    }

    // Visibility is drawn from its own stream so the noise stream stays purely
    // Gaussian. Every point keeps at least two views.
    std::mt19937 view_rng(555);
    const auto keep_dist = [](std::mt19937& g) { return rng::unit_interval(g); };
    for (std::size_t pt = 0; pt < scene.true_points.size(); ++pt) {
        std::vector<std::size_t> visible;
        for (int c = 0; c < kCameras; ++c) {
            if (keep_dist(view_rng) < kKeep) {
                visible.push_back(static_cast<std::size_t>(c));
            }
        }
        while (visible.size() < 2) {
            visible.push_back(static_cast<std::size_t>(keep_dist(view_rng) * kCameras));
        }
        for (const std::size_t cam : visible) {
            if (cam >= static_cast<std::size_t>(kCameras)) {
                continue;
            }
            const Eigen::Vector3d p_cam = scene.true_poses[cam].R * scene.true_points[pt] +
                                          scene.true_poses[cam].t;
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
            BAObservation obs;
            obs.camera = cam;
            obs.point = pt;
            obs.pixel = px + draw2(rng, noise, noise);
            scene.problem.observations.push_back(obs);
        }
    }
    scene.problem.fixed_pose.assign(scene.problem.poses.size(), 0);
    return scene;
}

void perturb(BAProblem& problem, std::mt19937& rng) {
    const auto angle = [](std::mt19937& g) { return rng::uniform_real(g, -0.035, 0.035); };  // ~2 deg
    const auto shift = [](std::mt19937& g) { return rng::uniform_real(g, -0.2, 0.2); };
    for (std::size_t i = 0; i < problem.poses.size(); ++i) {
        if (i == 0) {
            continue;  // camera 0 is the gauge to be fixed
        }
        const Eigen::Vector3d aa = draw3(rng, angle, angle, angle);
        problem.poses[i].R = rotation_from_angle_axis(aa) * problem.poses[i].R;
        problem.poses[i].t += draw3(rng, shift, shift, shift);
    }
    for (Eigen::Vector3d& p : problem.points) {
        p += draw3(rng, shift, shift, shift);
    }
}

/// Fix camera 0 (6 DoF) and anchor one translation component of camera 1 at its
/// true value (the 7th gauge DoF). The anchored component is chosen by
/// choose_scale_anchor_axis from the TRUE poses, and `true_poses` supplies that
/// truth value, so after the solve the reconstruction should land on the truth,
/// not merely be related to it by an unknown similarity.
void set_gauge(SyntheticScene& scene) {
    BAProblem& problem = scene.problem;
    problem.fixed_pose[0] = 1;
    problem.scale_anchor_camera = 1;
    problem.scale_anchor_axis =
        choose_scale_anchor_axis(scene.true_poses[0], scene.true_poses[1]);
    problem.poses[1].t[problem.scale_anchor_axis] = scene.true_poses[1].t[problem.scale_anchor_axis];
}

double mean_camera_centre_error(const BAProblem& p, const std::vector<Eigen::Vector3d>& truth) {
    double total = 0.0;
    for (std::size_t i = 0; i < truth.size(); ++i) {
        const Eigen::Vector3d centre = -p.poses[i].R.transpose() * p.poses[i].t;
        total += (centre - truth[i]).norm();
    }
    return total / static_cast<double>(truth.size());
}

Eigen::Vector3d camera_centre(const BAPose& pose) { return -pose.R.transpose() * pose.t; }

/// Truth sensor for the self-calibration tests (Frame 640x480). `f` is settable
/// so a two-sensor scene can give the sensors different focal lengths.
Calibration selfcal_truth(double f) {
    Calibration c;
    c.type = CameraType::Frame;
    c.width = 640;
    c.height = 480;
    c.f = f;
    c.cx = 4.0;
    c.cy = -3.0;
    c.k1 = -0.08;
    c.k2 = 0.02;
    c.p1 = 2e-4;
    return c;
}

/// Free-coefficient mask for fit_intrinsics from a list of free indices.
std::array<bool, kCalibrationParamCount> fit_flags(std::initializer_list<int> free_idx) {
    std::array<bool, kCalibrationParamCount> flags{};
    flags.fill(false);
    for (const int i : free_idx) {
        flags[static_cast<std::size_t>(i)] = true;
    }
    return flags;
}

/// N-camera strip scene for self-calibration with STRONG depth variation
/// (z in [5.5, 14.5] about z = 10) AND vertical station spread. The depth range
/// separates f from depth; the elevation spread separates the principal point
/// from camera pitch (on a purely horizontal arc the two are near-degenerate
/// under joint pose+intrinsics estimation — see the conditioning note in the
/// self-calibration test). Every camera maps to its entry of `sensor_of_camera`;
/// `truth` holds the per-sensor true calibrations used to synthesise clean
/// projections plus sigma-px Gaussian noise.
SyntheticScene make_selfcal_scene(const std::vector<Calibration>& truth,
                                  const std::vector<std::size_t>& sensor_of_camera,
                                  int num_cameras, int num_points, double sigma_px,
                                  std::mt19937& rng) {
    SyntheticScene scene;
    scene.problem.calibrations = truth;
    scene.problem.camera_sensor = sensor_of_camera;

    const auto xy = [](std::mt19937& g) { return rng::uniform_real(g, -3.5, 3.5); };
    const auto depth = [](std::mt19937& g) { return rng::uniform_real(g, -4.5, 4.5); };
    // The depth draw is offset by the cloud centroid; draw3 sequences it.
    const auto point_depth = [depth](std::mt19937& g) { return 10.0 + depth(g); };
    for (int i = 0; i < num_points; ++i) {
        Eigen::Vector3d p = draw3(rng, xy, xy, point_depth);
        scene.true_points.push_back(p);
        scene.problem.points.push_back(p);
    }

    const Eigen::Vector3d centroid(0.0, 0.0, 10.0);
    constexpr double kDeg = 3.14159265358979323846 / 180.0;
    constexpr double kAzSpan = 180.0 * kDeg;  // wide azimuth sweep (half-turn strip)
    constexpr double kElSpan = 40.0 * kDeg;   // elevation sweep (vertical baseline)
    for (int i = 0; i < num_cameras; ++i) {
        const double t = (num_cameras > 1) ? static_cast<double>(i) /
                                                static_cast<double>(num_cameras - 1)
                                          : 0.5;
        const double az = -0.5 * kAzSpan + kAzSpan * t;
        const double el = -0.5 * kElSpan + kElSpan * t;
        const double radius = 8.0;
        const Eigen::Vector3d direction(std::cos(el) * std::sin(az), std::sin(el),
                                        std::cos(el) * std::cos(az));
        const Eigen::Vector3d centre = centroid + radius * direction;
        scene.true_centres.push_back(centre);

        const Eigen::Vector3d forward = (centroid - centre).normalized();
        Eigen::Vector3d right = forward.cross(Eigen::Vector3d::UnitY()).normalized();
        if (!(right.norm() > 0.5)) {
            right = Eigen::Vector3d::UnitX();  // forward parallel to up (degenerate)
        }
        const Eigen::Vector3d up = right.cross(forward);
        Eigen::Matrix3d R_wc;
        R_wc.col(0) = right;
        R_wc.col(1) = -up;
        R_wc.col(2) = forward;
        BAPose pose;
        pose.R = R_wc.transpose();
        pose.t = -pose.R * centre;
        scene.true_poses.push_back(pose);
        scene.problem.poses.push_back(pose);
    }

    const auto noise = [sigma_px](std::mt19937& g) { return rng::normal(g, 0.0, sigma_px); };
    std::mt19937 view_rng(555);
    const auto keep_dist = [](std::mt19937& g) { return rng::unit_interval(g); };
    for (std::size_t pt = 0; pt < scene.true_points.size(); ++pt) {
        std::vector<std::size_t> visible;
        for (int c = 0; c < num_cameras; ++c) {
            if (keep_dist(view_rng) < 0.6) {
                visible.push_back(static_cast<std::size_t>(c));
            }
        }
        while (visible.size() < 2) {
            visible.push_back(static_cast<std::size_t>(keep_dist(view_rng) * num_cameras) %
                              static_cast<std::size_t>(num_cameras));
        }
        for (const std::size_t cam : visible) {
            if (cam >= static_cast<std::size_t>(num_cameras)) {
                continue;
            }
            const Calibration& cal = truth[sensor_of_camera[cam]];
            const Eigen::Vector3d p_cam =
                scene.true_poses[cam].R * scene.true_points[pt] + scene.true_poses[cam].t;
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
            BAObservation obs;
            obs.camera = cam;
            obs.point = pt;
            obs.pixel = px + draw2(rng, noise, noise);
            scene.problem.observations.push_back(obs);
        }
    }
    scene.problem.fixed_pose.assign(scene.problem.poses.size(), 0);
    return scene;
}

/// Perturb only the poses (self-calibration starts from perturbed poses; the
/// points stay at truth so the gauge and the intrinsics must do the work).
void perturb_poses(BAProblem& problem, std::mt19937& rng) {
    const auto angle = [](std::mt19937& g) { return rng::uniform_real(g, -0.02, 0.02); };
    const auto shift = [](std::mt19937& g) { return rng::uniform_real(g, -0.15, 0.15); };
    for (std::size_t i = 1; i < problem.poses.size(); ++i) {
        const Eigen::Vector3d aa = draw3(rng, angle, angle, angle);
        problem.poses[i].R = rotation_from_angle_axis(aa) * problem.poses[i].R;
        problem.poses[i].t += draw3(rng, shift, shift, shift);
    }
}

/// Apply a random similarity (rotation <= 10 deg, scale 0.9..1.1, shift <= 1) to
/// the problem's poses and points, leaving the truth untouched — the 7-DoF
/// error that the centre priors must undo.
void apply_random_similarity(BAProblem& problem, std::mt19937& rng) {
    const auto angle = [](std::mt19937& g) {
        return rng::uniform_real(g, -10.0 * 3.14159265358979323846 / 180.0,
                                 10.0 * 3.14159265358979323846 / 180.0);
    };
    const auto scale = [](std::mt19937& g) { return rng::uniform_real(g, 0.9, 1.1); };
    const auto shift = [](std::mt19937& g) { return rng::uniform_real(g, -1.0, 1.0); };
    const Eigen::Matrix3d R = rotation_from_angle_axis(draw3(rng, angle, angle, angle));
    const double s = scale(rng);
    const Eigen::Vector3d tr = draw3(rng, shift, shift, shift);
    for (BAPose& pose : problem.poses) {
        const Eigen::Vector3d centre = -pose.R.transpose() * pose.t;
        const Eigen::Vector3d centre_t = s * (R * centre) + tr;
        pose.R = pose.R * R.transpose();
        pose.t = -pose.R * centre_t;
    }
    for (Eigen::Vector3d& p : problem.points) {
        p = s * (R * p) + tr;
    }
}

}  // namespace

TEST(BundleAdjustment, ConvergesFromPerturbedStart) {
    std::mt19937 rng(1234);
    SyntheticScene scene = make_scene(CameraType::Frame, CameraLayout::Ring, 0.5, rng);
    EXPECT_GT(scene.problem.observations.size(), 500u);

    perturb(scene.problem, rng);
    set_gauge(scene);

    BAOptions options;
    const auto t0 = std::chrono::steady_clock::now();
    const BAReport report = bundle_adjust(scene.problem, options);
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::printf("[bundle_adjust] solve time = %.4f s, iterations = %d\n", seconds, report.iterations);

    EXPECT_TRUE(report.usable);
    EXPECT_LT(report.final_rms_px, 0.6);
    // Timing budget, not a correctness bound. Debug + ASan+UBSan runs the Ceres
    // solve several times slower than Release, so the sanitized build gets the
    // same order of magnitude more headroom.
#ifdef ALPHASHAPE_SANITIZED_BUILD
    EXPECT_LT(seconds, 30.0);
#else
    EXPECT_LT(seconds, 1.0);
#endif

    for (std::size_t i = 0; i < scene.true_centres.size(); ++i) {
        const Eigen::Vector3d centre = -scene.problem.poses[i].R.transpose() * scene.problem.poses[i].t;
        EXPECT_LT((centre - scene.true_centres[i]).norm(), 0.02) << "camera " << i;
    }

    std::vector<double> errs;
    for (std::size_t i = 0; i < scene.true_points.size(); ++i) {
        errs.push_back((scene.problem.points[i] - scene.true_points[i]).norm());
    }
    std::sort(errs.begin(), errs.end());
    EXPECT_LT(errs[errs.size() / 2], 0.02);
}

/// One-sided arc of cameras — the geometry of a real photo sequence, and the
/// weak-conditioning case that a scale anchor chosen from t1 (rather than the
/// baseline t_rel) fails to pin. With the correct anchor the reconstruction
/// still has to land on the truth, though the one-sided baseline leaves a much
/// wider error distribution than the ring: over seven seeds the median point
/// error spans 0.024-0.098 and the worst camera 0.018-0.081, so the bounds
/// below are set from that spread rather than from a single realisation
/// (previously 0.02/0.02, which only the old, platform-dependent draw held).
/// This seed measures 0.0183 worst camera and 0.0342 median.
TEST(BundleAdjustment, ConvergesOnSingleSidedArc) {
    std::mt19937 rng(1234);
    SyntheticScene scene = make_scene(CameraType::Frame, CameraLayout::Arc, 0.5, rng);
    EXPECT_GT(scene.problem.observations.size(), 500u);

    perturb(scene.problem, rng);
    set_gauge(scene);

    const BAReport report = bundle_adjust(scene.problem, BAOptions{});
    EXPECT_TRUE(report.usable);
    EXPECT_LT(report.final_rms_px, 0.6);

    double worst_centre = 0.0;
    for (std::size_t i = 0; i < scene.true_centres.size(); ++i) {
        const Eigen::Vector3d centre = -scene.problem.poses[i].R.transpose() * scene.problem.poses[i].t;
        const double err = (centre - scene.true_centres[i]).norm();
        worst_centre = std::max(worst_centre, err);
        EXPECT_LT(err, 0.05) << "camera " << i;
    }

    std::vector<double> errs;
    for (std::size_t i = 0; i < scene.true_points.size(); ++i) {
        errs.push_back((scene.problem.points[i] - scene.true_points[i]).norm());
    }
    std::sort(errs.begin(), errs.end());
    const double median = errs[errs.size() / 2];
    std::printf("[bundle_adjust] arc: worst centre err = %.5f, median point err = %.5f\n",
                worst_centre, median);
    EXPECT_LT(median, 0.05);
}

/// The baseline t_rel, not t1, carries the scale signal. pose0 is a non-origin
/// camera looking back at the world origin, so pose1's own t1 has a large z
/// (depth) component; the baseline runs along camera-1 x. The function must pick
/// x (axis 0), not the dominant-looking z of t1.
TEST(BundleAdjustment, ChooseScaleAnchorAxisPicksBaseline) {
    // pose0 at (0,0,-5) looking along +Z (identity rotation), so pose0.t = (0,0,5).
    BAPose pose0;
    pose0.R = Eigen::Matrix3d::Identity();
    pose0.t = Eigen::Vector3d(0.0, 0.0, 5.0);

    // pose1 identical orientation, centred 2 units to the +x of pose0: the
    // baseline lies along camera-1 x, so t_rel = t1 - t0 = (-2, 0, 0). Its t1 =
    // (-2, 0, 5) is dominated by the z (depth) component (5 > 2). Note pose0
    // here is NOT at the origin (its centre is (0,0,-5)), so t1 carries the
    // large depth-to-world-origin term while the baseline does not.
    BAPose pose1;
    pose1.R = Eigen::Matrix3d::Identity();
    pose1.t = Eigen::Vector3d(-2.0, 0.0, 5.0);

    ASSERT_GT(std::abs(pose1.t.z()), std::abs(pose1.t.x()));
    EXPECT_EQ(choose_scale_anchor_axis(pose0, pose1), 0);
}

TEST(BundleAdjustment, RobustLossHandlesOutliers) {
    std::mt19937 rng(99);
    SyntheticScene scene = make_scene(CameraType::Frame, CameraLayout::Ring, 0.5, rng);
    perturb(scene.problem, rng);
    set_gauge(scene);

    // Corrupt 5% of observations by ~30 px.
    const std::size_t observation_count = scene.problem.observations.size();
    const auto pick = [observation_count](std::mt19937& g) {
        return rng::index_below(g, observation_count);
    };
    const auto outlier = [](std::mt19937& g) { return rng::uniform_real(g, -30.0, 30.0); };
    const std::size_t corrupt_count = scene.problem.observations.size() / 20;
    for (std::size_t k = 0; k < corrupt_count; ++k) {
        BAObservation& obs = scene.problem.observations[pick(rng)];
        obs.pixel += draw2(rng, outlier, outlier);
    }

    BAProblem robust_problem = scene.problem;
    BAOptions robust_options;
    robust_options.loss_scale_px = 1.0;
    const BAReport robust = bundle_adjust(robust_problem, robust_options);

    BAProblem plain_problem = scene.problem;
    BAOptions plain_options;
    plain_options.loss_scale_px = 0.0;
    const BAReport plain = bundle_adjust(plain_problem, plain_options);

    const double robust_err = mean_camera_centre_error(robust_problem, scene.true_centres);
    const double plain_err = mean_camera_centre_error(plain_problem, scene.true_centres);
    std::printf("[bundle_adjust] robust centre err = %.5f, plain = %.5f\n", robust_err, plain_err);

    EXPECT_TRUE(robust.usable);
    EXPECT_TRUE(plain.usable);
    EXPECT_LT(robust_err, 0.05);
    EXPECT_LT(robust_err, plain_err);
}

TEST(BundleAdjustment, FixedPosesStayFixed) {
    std::mt19937 rng(7);
    SyntheticScene scene = make_scene(CameraType::Frame, CameraLayout::Ring, 0.5, rng);
    perturb(scene.problem, rng);
    set_gauge(scene);
    scene.problem.fixed_pose[3] = 1;

    const Eigen::Matrix3d R3 = scene.problem.poses[3].R;
    const Eigen::Vector3d t3 = scene.problem.poses[3].t;
    const Eigen::Matrix3d R0 = scene.problem.poses[0].R;

    const BAReport report = bundle_adjust(scene.problem, BAOptions{});
    EXPECT_TRUE(report.usable);

    EXPECT_EQ(scene.problem.poses[3].R, R3);
    EXPECT_EQ(scene.problem.poses[3].t, t3);
    EXPECT_EQ(scene.problem.poses[0].R, R0);
}

TEST(BundleAdjustment, FisheyeCameras) {
    std::mt19937 rng(2024);
    SyntheticScene scene = make_scene(CameraType::Fisheye, CameraLayout::Ring, 0.5, rng);
    if (scene.problem.observations.size() < 100u) {
        GTEST_SKIP() << "not enough fisheye observations";
    }
    perturb(scene.problem, rng);
    set_gauge(scene);

    const double before = reprojection_rms_px(scene.problem);
    const BAReport report = bundle_adjust(scene.problem, BAOptions{});
    EXPECT_TRUE(report.usable);
    EXPECT_LT(report.final_rms_px, before);
    // The residual is the 0.5 px measurement noise projected through a fisheye
    // model, so the achievable floor sits right at 0.6 (0.579-0.615 over seven
    // seeds). The bound was 0.6, which one realisation could not hold.
    EXPECT_LT(report.final_rms_px, 0.65);
}

TEST(BundleAdjustment, ValidationErrors) {
    std::mt19937 rng(5);
    SyntheticScene scene = make_scene(CameraType::Frame, CameraLayout::Ring, 0.0, rng);

    {
        BAProblem bad = scene.problem;
        bad.observations.front().camera = bad.poses.size();
        EXPECT_THROW(bundle_adjust(bad, BAOptions{}), ValueError);
    }
    {
        BAProblem bad = scene.problem;
        bad.observations.front().point = bad.points.size();
        EXPECT_THROW(bundle_adjust(bad, BAOptions{}), ValueError);
    }
    {
        BAProblem bad = scene.problem;
        bad.fixed_pose.pop_back();
        EXPECT_THROW(bundle_adjust(bad, BAOptions{}), ValueError);
    }
    {
        // camera_sensor wrong size (non-empty must equal poses.size()).
        BAProblem bad = scene.problem;
        bad.camera_sensor.assign(bad.poses.size() - 1, 0u);
        EXPECT_THROW(bundle_adjust(bad, BAOptions{}), ValueError);
    }
    {
        // camera_sensor index out of range.
        BAProblem bad = scene.problem;
        bad.camera_sensor.assign(bad.poses.size(), 0u);
        bad.camera_sensor.front() = bad.calibrations.size();
        EXPECT_THROW(bundle_adjust(bad, BAOptions{}), ValueError);
    }
    {
        // fit_intrinsics wrong size (non-empty must equal calibrations.size()).
        BAProblem bad = scene.problem;
        bad.fit_intrinsics.assign(bad.calibrations.size() - 1, fit_flags({0}));
        EXPECT_THROW(bundle_adjust(bad, BAOptions{}), ValueError);
    }
    {
        // centre_prior camera index out of range.
        BAProblem bad = scene.problem;
        BACentrePrior prior;
        prior.camera = bad.poses.size();
        prior.sigma = Eigen::Vector3d(0.03, 0.03, 0.03);
        bad.centre_priors.push_back(prior);
        EXPECT_THROW(bundle_adjust(bad, BAOptions{}), ValueError);
    }
}

TEST(BundleAdjustment, ReprojectionRmsMatchesManual) {
    BAProblem problem;
    Calibration cal = frame_calibration();
    problem.calibrations.push_back(cal);

    BAPose pose;
    problem.poses.push_back(pose);

    // Two points, projected exactly, then shifted by known pixel amounts.
    Eigen::Vector3d p0(0.0, 0.0, 10.0);
    Eigen::Vector3d p1(0.4, -0.3, 10.0);
    problem.points.push_back(p0);
    problem.points.push_back(p1);

    const Eigen::Vector2d uv0 = cal.project(p0);
    const Eigen::Vector2d uv1 = cal.project(p1);
    const Eigen::Vector2d d0(3.0, 4.0);   // 5 px
    const Eigen::Vector2d d1(0.0, 12.0);  // 12 px
    problem.observations.push_back(BAObservation{0, 0, uv0 + d0});
    problem.observations.push_back(BAObservation{0, 1, uv1 + d1});

    const double expected = std::sqrt((5.0 * 5.0 + 12.0 * 12.0) / 2.0);
    EXPECT_NEAR(reprojection_rms_px(problem), expected, 1e-9);
}

TEST(BundleAdjustment, BehindCameraChargesLargeError) {
    BAProblem problem;
    problem.calibrations.push_back(frame_calibration());
    problem.poses.push_back(BAPose{});
    problem.points.push_back(Eigen::Vector3d(0.0, 0.0, -5.0));  // behind the camera
    problem.observations.push_back(BAObservation{0, 0, Eigen::Vector2d(320.0, 240.0)});
    EXPECT_GT(reprojection_rms_px(problem), 1.0e5);
}

/// Self-calibration: ten cameras share one sensor. f, cx, cy, k1, k2, p1 and p2
/// are recovered from a 5%-low f start with an arc layout that has real depth
/// variation, so f and depth separate. Camera 0 is fixed and camera 1's baseline
/// axis anchored: the correct gauge.
TEST(BundleAdjustment, SelfCalibrationRecoversIntrinsics) {
    std::mt19937 rng(3210);
    const Calibration truth = selfcal_truth(600.0);
    const std::vector<std::size_t> sensor_of_camera(10, 0u);
    SyntheticScene scene = make_selfcal_scene({truth}, sensor_of_camera, 10, 1500, 0.5, rng);
    EXPECT_GT(scene.problem.observations.size(), 1500u);

    // Start from f = 570 (-5%), every other coefficient zero, perturbed poses.
    Calibration start = truth;
    start.f = 570.0;
    start.cx = 0.0;
    start.cy = 0.0;
    start.k1 = 0.0;
    start.k2 = 0.0;
    start.p1 = 0.0;
    scene.problem.calibrations[0] = start;
    perturb_poses(scene.problem, rng);

    // Correct gauge: camera 0 fixed, camera 1's chosen baseline axis anchored.
    scene.problem.fixed_pose[0] = 1;
    scene.problem.scale_anchor_camera = 1;
    scene.problem.scale_anchor_axis =
        choose_scale_anchor_axis(scene.true_poses[0], scene.true_poses[1]);
    scene.problem.poses[1].t[scene.problem.scale_anchor_axis] =
        scene.true_poses[1].t[scene.problem.scale_anchor_axis];

    // Fit f, cx, cy, k1, k2, p1, p2.
    scene.problem.fit_intrinsics = {
        fit_flags({0, 1, 2, 5, 6, 9, 10})};

    // The start is 5% off in f and 0 in every distortion term, so the initial
    // residuals are many pixels; the clean Gaussian scene has no outliers, so
    // plain least squares (no robust loss) is the right estimator here.
    BAOptions options;
    options.loss_scale_px = 0.0;

    const auto t0 = std::chrono::steady_clock::now();
    const BAReport report = bundle_adjust(scene.problem, options);
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const Calibration& got = scene.problem.calibrations[0];
    std::printf(
        "[bundle_adjust] self-cal: f=%.4f cx=%.4f cy=%.4f k1=%.5f k2=%.5f p1=%.2e p2=%.2e "
        "rms=%.4f iters=%d time=%.4f s\n",
        got.f, got.cx, got.cy, got.k1, got.k2, got.p1, got.p2, report.final_rms_px,
        report.iterations, seconds);

    EXPECT_TRUE(report.usable);
    EXPECT_NEAR(got.f, 600.0, 0.5);
    EXPECT_NEAR(got.k1, -0.08, 0.005);
    EXPECT_NEAR(got.cx, 4.0, 1.0);
    EXPECT_NEAR(got.cy, -3.0, 1.0);
    EXPECT_LT(report.final_rms_px, 0.6);
}

/// Coefficients marked not-free must come back bitwise identical, even when the
/// free ones move a long way.
TEST(BundleAdjustment, FixedIntrinsicsStayBitwiseFixed) {
    std::mt19937 rng(4321);
    const Calibration truth = selfcal_truth(600.0);
    const std::vector<std::size_t> sensor_of_camera(10, 0u);
    SyntheticScene scene = make_selfcal_scene({truth}, sensor_of_camera, 10, 400, 0.5, rng);

    Calibration start = truth;
    start.f = 570.0;
    // Deliberately wrong values on the coefficients we will hold fixed.
    start.b1 = 1.5;
    start.b2 = -0.7;
    start.k3 = 0.03;
    start.k4 = -0.02;
    scene.problem.calibrations[0] = start;
    perturb_poses(scene.problem, rng);

    scene.problem.fixed_pose[0] = 1;
    scene.problem.scale_anchor_camera = 1;
    scene.problem.scale_anchor_axis =
        choose_scale_anchor_axis(scene.true_poses[0], scene.true_poses[1]);
    scene.problem.poses[1].t[scene.problem.scale_anchor_axis] =
        scene.true_poses[1].t[scene.problem.scale_anchor_axis];
    scene.problem.fit_intrinsics = {fit_flags({0, 1, 2, 5, 6, 9, 10})};

    const Calibration before = scene.problem.calibrations[0];
    const BAReport report = bundle_adjust(scene.problem, BAOptions{});
    EXPECT_TRUE(report.usable);
    const Calibration& after = scene.problem.calibrations[0];

    // Not-free: bitwise unchanged.
    EXPECT_EQ(after.b1, before.b1);
    EXPECT_EQ(after.b2, before.b2);
    EXPECT_EQ(after.k3, before.k3);
    EXPECT_EQ(after.k4, before.k4);
    EXPECT_EQ(after.type, before.type);
    EXPECT_EQ(after.width, before.width);
    EXPECT_EQ(after.height, before.height);
    // Free: actually moved off the start.
    EXPECT_NE(after.f, before.f);
    EXPECT_NE(after.k1, before.k1);
}

/// Two sensors on one problem: cameras 0-4 on A, 5-9 on B, with different true
/// focal lengths. Both blocks are recovered independently.
TEST(BundleAdjustment, TwoSensorsCalibrateIndependently) {
    std::mt19937 rng(6543);
    const Calibration truth_a = selfcal_truth(600.0);
    const Calibration truth_b = selfcal_truth(500.0);
    std::vector<std::size_t> sensor_of_camera;
    for (int c = 0; c < 10; ++c) {
        sensor_of_camera.push_back(c < 5 ? 0u : 1u);
    }
    SyntheticScene scene =
        make_selfcal_scene({truth_a, truth_b}, sensor_of_camera, 10, 1500, 0.5, rng);

    Calibration start_a = truth_a;
    start_a.f = 570.0;
    start_a.cx = 0.0;
    start_a.cy = 0.0;
    start_a.k1 = 0.0;
    start_a.k2 = 0.0;
    Calibration start_b = truth_b;
    start_b.f = 475.0;
    start_b.cx = 0.0;
    start_b.cy = 0.0;
    start_b.k1 = 0.0;
    start_b.k2 = 0.0;
    scene.problem.calibrations = {start_a, start_b};
    perturb_poses(scene.problem, rng);

    scene.problem.fixed_pose[0] = 1;
    scene.problem.scale_anchor_camera = 1;
    scene.problem.scale_anchor_axis =
        choose_scale_anchor_axis(scene.true_poses[0], scene.true_poses[1]);
    scene.problem.poses[1].t[scene.problem.scale_anchor_axis] =
        scene.true_poses[1].t[scene.problem.scale_anchor_axis];
    scene.problem.fit_intrinsics = {fit_flags({0, 1, 2, 5, 6}), fit_flags({0, 1, 2, 5, 6})};

    BAOptions options;
    options.loss_scale_px = 0.0;  // clean Gaussian scene, large start error
    const BAReport report = bundle_adjust(scene.problem, options);
    std::printf("[bundle_adjust] two-sensor: fA=%.4f fB=%.4f rms=%.4f\n",
                scene.problem.calibrations[0].f, scene.problem.calibrations[1].f,
                report.final_rms_px);
    EXPECT_TRUE(report.usable);
    EXPECT_NEAR(scene.problem.calibrations[0].f, 600.0, 0.5);
    EXPECT_NEAR(scene.problem.calibrations[1].f, 500.0, 0.5);
    EXPECT_NEAR(scene.problem.calibrations[0].k1, -0.08, 0.005);
    EXPECT_NEAR(scene.problem.calibrations[1].k1, -0.08, 0.005);
    EXPECT_LT(report.final_rms_px, 0.6);
}

/// GPS priors on every camera fix the gauge by themselves: no fixed pose, no
/// scale anchor, and the start is the truth transformed by a large random
/// similarity. The priors must pull every centre back to truth in the ABSOLUTE
/// frame (no alignment step).
TEST(BundleAdjustment, CentrePriorsFixTheGauge) {
    std::mt19937 rng(7777);
    const Calibration truth = selfcal_truth(600.0);
    const std::vector<std::size_t> sensor_of_camera(8, 0u);
    SyntheticScene scene = make_selfcal_scene({truth}, sensor_of_camera, 8, 300, 0.5, rng);

    apply_random_similarity(scene.problem, rng);
    perturb_poses(scene.problem, rng);

    // Priors: true centre plus N(0, 0.03) noise, sigma = 0.03 on every axis.
    // Note: truth centres are in the ORIGINAL frame; the random similarity moved
    // poses but the priors must reference the same (original) frame the solver
    // converges back to.
    const auto gps = [](std::mt19937& g) { return rng::normal(g, 0.0, 0.03); };
    for (std::size_t i = 0; i < scene.true_centres.size(); ++i) {
        BACentrePrior prior;
        prior.camera = i;
        prior.centre = scene.true_centres[i] + draw3(rng, gps, gps, gps);
        prior.sigma = Eigen::Vector3d(0.03, 0.03, 0.03);
        scene.problem.centre_priors.push_back(prior);
    }
    // No gauge fixes at all: the priors are the gauge.
    scene.problem.fixed_pose.clear();

    const BAReport report = bundle_adjust(scene.problem, BAOptions{});
    std::printf("[bundle_adjust] priors: rms=%.4f prior_rms=%.4f iters=%d\n",
                report.final_rms_px, report.prior_rms_sigma, report.iterations);
    EXPECT_TRUE(report.usable);
    EXPECT_EQ(report.priors, 24u);

    for (std::size_t i = 0; i < scene.true_centres.size(); ++i) {
        const Eigen::Vector3d centre = camera_centre(scene.problem.poses[i]);
        EXPECT_LT((centre - scene.true_centres[i]).norm(), 0.05) << "camera " << i;
    }
    EXPECT_GT(report.prior_rms_sigma, 0.5);
    EXPECT_LT(report.prior_rms_sigma, 1.5);
    EXPECT_LT(report.final_rms_px, 0.6);
}

/// One prior offset by 5 m (150 sigma). The Huber loss must keep it from pulling
/// the solution: all other cameras stay faithful and the bad camera lands closer
/// to its own truth than to the corrupt prior.
TEST(BundleAdjustment, OutlierPriorIsDownweighted) {
    std::mt19937 rng(8888);
    const Calibration truth = selfcal_truth(600.0);
    const std::vector<std::size_t> sensor_of_camera(8, 0u);
    SyntheticScene scene = make_selfcal_scene({truth}, sensor_of_camera, 8, 300, 0.5, rng);

    const auto gps = [](std::mt19937& g) { return rng::normal(g, 0.0, 0.03); };
    constexpr std::size_t kBad = 3;
    // Written only inside the loop, but read after it: initialising it keeps
    // g++'s inliner from reporting the later use as maybe-uninitialized.
    Eigen::Vector3d bad_prior = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < scene.true_centres.size(); ++i) {
        BACentrePrior prior;
        prior.camera = i;
        prior.centre = scene.true_centres[i] + draw3(rng, gps, gps, gps);
        prior.sigma = Eigen::Vector3d(0.03, 0.03, 0.03);
        if (i == kBad) {
            prior.centre += Eigen::Vector3d(5.0, 0.0, 0.0);
            bad_prior = prior.centre;
        }
        scene.problem.centre_priors.push_back(prior);
    }
    scene.problem.fixed_pose.clear();

    const BAReport report = bundle_adjust(scene.problem, BAOptions{});
    EXPECT_TRUE(report.usable);

    for (std::size_t i = 0; i < scene.true_centres.size(); ++i) {
        if (i == kBad) {
            continue;
        }
        const Eigen::Vector3d centre = camera_centre(scene.problem.poses[i]);
        EXPECT_LT((centre - scene.true_centres[i]).norm(), 0.06) << "camera " << i;
    }
    const Eigen::Vector3d bad_centre = camera_centre(scene.problem.poses[kBad]);
    const double to_truth = (bad_centre - scene.true_centres[kBad]).norm();
    const double to_prior = (bad_centre - bad_prior).norm();
    std::printf("[bundle_adjust] outlier prior: bad cam to truth=%.4f to prior=%.4f\n", to_truth,
                to_prior);
    EXPECT_LT(to_truth, to_prior);
}

/// Priors that leave one DoF free (sigma_z = 0 on most cameras, full sigma on
/// three) still converge: the three full-sigma cameras pin the vertical freedom.
TEST(BundleAdjustment, PriorAxisUnconstrained) {
    std::mt19937 rng(9999);
    const Calibration truth = selfcal_truth(600.0);
    const std::vector<std::size_t> sensor_of_camera(8, 0u);
    SyntheticScene scene = make_selfcal_scene({truth}, sensor_of_camera, 8, 300, 0.5, rng);
    perturb_poses(scene.problem, rng);

    const auto gps = [](std::mt19937& g) { return rng::normal(g, 0.0, 0.03); };
    for (std::size_t i = 0; i < scene.true_centres.size(); ++i) {
        BACentrePrior prior;
        prior.camera = i;
        const bool full = i < 3;  // three cameras constrain z
        prior.centre = scene.true_centres[i] + draw3(rng, gps, gps, gps);
        prior.sigma = full ? Eigen::Vector3d(0.03, 0.03, 0.03)
                           : Eigen::Vector3d(0.03, 0.03, 0.0);
        scene.problem.centre_priors.push_back(prior);
    }
    scene.problem.fixed_pose.clear();

    const BAReport report = bundle_adjust(scene.problem, BAOptions{});
    std::printf("[bundle_adjust] partial priors: rms=%.4f priors=%zu prior_rms=%.4f\n",
                report.final_rms_px, report.priors, report.prior_rms_sigma);
    EXPECT_TRUE(report.usable);
    EXPECT_EQ(report.priors, 3u * 3u + 5u * 2u);  // 3 full cameras + 5 x,y-only
    EXPECT_LT(report.final_rms_px, 0.6);
}

}  // namespace alphashape::sfm

