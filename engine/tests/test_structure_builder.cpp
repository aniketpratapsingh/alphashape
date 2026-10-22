// Milestone 4: incremental structure-from-motion tests (suite StructureBuilder).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"
#include "common/algorithm/sfm/structure_builder.hpp"
#include "model/camera/calibration.hpp"
#include "test_rng_helpers.hpp"

namespace alphashape::sfm {
namespace {

using test_rng::draw2;
using test_rng::draw3;

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr double kFocal = 600.0;
constexpr double kArcRadius = 15.0;
constexpr double kBoxHalf = 5.0;   // points span 10 x 10 in x,y
constexpr double kBoxHalfZ = 2.0;  // and 4 in z

Calibration frame_calibration() {
    Calibration c;
    c.type = CameraType::Frame;
    c.width = kWidth;
    c.height = kHeight;
    c.f = kFocal;
    return c;
}

struct GtScene {
    Calibration cal;
    std::vector<Eigen::Vector3d> true_points;    ///< per true point index
    std::vector<AbsolutePose> true_poses;        ///< per camera index
    std::vector<Eigen::Vector3d> true_centres;   ///< per camera index
    std::vector<std::size_t> track_to_point;     ///< track index -> true point index
    BuilderInput input;
};

Eigen::Vector3d camera_centre(const AbsolutePose& pose) {
    return -pose.R.transpose() * pose.t;
}

/// 12 cameras on a one-sided arc of `arc_deg` at `radius`, looking at the
/// origin, and `n_points` points in the given box. Observations are the clean
/// projections plus sigma-px noise, kept when inside the image and in front.
/// Tracks are built directly from the ground truth (one per point), with the
/// PairMatches implied by those tracks.
GtScene make_arc_scene(std::uint32_t seed, double sigma_px, int n_cameras = 12,
                       double arc_deg = 60.0, int n_points = 600) {
    GtScene scene;
    scene.cal = frame_calibration();
    std::mt19937 rng(seed);

    // Portable draws, not std::distributions: those map engine bits to values
    // differently under libc++ and libstdc++, so one seed built different
    // scenes on macOS and Linux (worst rotation 0.0498 vs 0.0624, 277 vs 279
    // corrupted tracks). rng::uniform_real/normal are portable.
    const auto dx = [](std::mt19937& g) { return rng::uniform_real(g, -kBoxHalf, kBoxHalf); };
    const auto dz = [](std::mt19937& g) { return rng::uniform_real(g, -kBoxHalfZ, kBoxHalfZ); };
    for (int i = 0; i < n_points; ++i) {
        scene.true_points.push_back(draw3(rng, dx, dx, dz));
    }

    const double pi = 3.14159265358979323846;
    const double arc = arc_deg * pi / 180.0;
    for (int c = 0; c < n_cameras; ++c) {
        const double a = -0.5 * arc + arc * static_cast<double>(c) / static_cast<double>(n_cameras - 1);
        const Eigen::Vector3d centre = kArcRadius * Eigen::Vector3d(std::sin(a), 0.0, std::cos(a));
        scene.true_centres.push_back(centre);

        const Eigen::Vector3d forward = (-centre).normalized();
        const Eigen::Vector3d world_up = Eigen::Vector3d::UnitY();
        const Eigen::Vector3d right = forward.cross(world_up).normalized();
        const Eigen::Vector3d up = right.cross(forward);
        Eigen::Matrix3d R_wc;  // columns: right, down, forward (camera +Y is down)
        R_wc.col(0) = right;
        R_wc.col(1) = -up;
        R_wc.col(2) = forward;
        AbsolutePose pose;
        pose.R = R_wc.transpose();
        pose.t = -pose.R * centre;
        scene.true_poses.push_back(pose);
    }

    // pixels[cam][kp]: keypoint kp of camera cam is true point kp. Invisible or
    // unprojectable keypoints keep a placeholder no track ever references.
    const auto noise = [sigma_px](std::mt19937& g) { return rng::normal(g, 0.0, sigma_px); };
    std::vector<std::vector<std::uint8_t>> visible(static_cast<std::size_t>(n_cameras));
    BuilderCamera bc;
    for (int c = 0; c < n_cameras; ++c) {
        bc.id = c;
        bc.calibration = scene.cal;
        bc.pixels.assign(scene.true_points.size(), Eigen::Vector2d(0.0, 0.0));
        visible[static_cast<std::size_t>(c)].assign(scene.true_points.size(), 0);
        for (std::size_t p = 0; p < scene.true_points.size(); ++p) {
            const Eigen::Vector3d p_cam = scene.true_poses[static_cast<std::size_t>(c)].R *
                                              scene.true_points[p] +
                                          scene.true_poses[static_cast<std::size_t>(c)].t;
            if (!(p_cam.z() > 0.0)) {
                continue;
            }
            const Eigen::Vector2d px = scene.cal.project(p_cam);
            if (px.x() < 0.0 || px.x() > kWidth || px.y() < 0.0 || px.y() > kHeight) {
                continue;
            }
            visible[static_cast<std::size_t>(c)][p] = 1;
            bc.pixels[p] = px + draw2(rng, noise, noise);
        }
        scene.input.cameras.push_back(bc);
    }

    // One track per point that at least two cameras see.
    for (std::size_t p = 0; p < scene.true_points.size(); ++p) {
        Track track;
        for (int c = 0; c < n_cameras; ++c) {
            if (visible[static_cast<std::size_t>(c)][p] != 0) {
                track.observations.push_back(Observation{c, static_cast<std::uint32_t>(p)});
            }
        }
        if (track.observations.size() < 2) {
            continue;
        }
        scene.track_to_point.push_back(p);
        scene.input.tracks.push_back(track);
    }

    // PairMatches implied by the tracks.
    for (int i = 0; i < n_cameras; ++i) {
        for (int j = i + 1; j < n_cameras; ++j) {
            PairMatches pair;
            pair.camera_a = i;
            pair.camera_b = j;
            for (std::size_t t = 0; t < scene.input.tracks.size(); ++t) {
                bool in_i = false, in_j = false;
                for (const Observation& o : scene.input.tracks[t].observations) {
                    in_i = in_i || o.camera_id == i;
                    in_j = in_j || o.camera_id == j;
                }
                if (in_i && in_j) {
                    const std::uint32_t p = static_cast<std::uint32_t>(scene.track_to_point[t]);
                    pair.matches.push_back(KeypointMatch{p, p});
                }
            }
            if (!pair.matches.empty()) {
                scene.input.matches.push_back(pair);
            }
        }
    }
    return scene;
}

struct Similarity {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    double scale = 1.0;
};

/// Umeyama similarity (rotation + translation + uniform scale) that best maps
/// `src` onto `dst`.
Similarity umeyama(const std::vector<Eigen::Vector3d>& src,
                   const std::vector<Eigen::Vector3d>& dst) {
    const double n = static_cast<double>(src.size());
    Eigen::Vector3d mu_src = Eigen::Vector3d::Zero(), mu_dst = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < src.size(); ++i) {
        mu_src += src[i];
        mu_dst += dst[i];
    }
    mu_src /= n;
    mu_dst /= n;

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    double var_src = 0.0;
    for (std::size_t i = 0; i < src.size(); ++i) {
        const Eigen::Vector3d xs = src[i] - mu_src;
        cov += (dst[i] - mu_dst) * xs.transpose();
        var_src += xs.squaredNorm();
    }
    cov /= n;
    var_src /= n;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d d = Eigen::Matrix3d::Identity();
    if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0.0) {
        d(2, 2) = -1.0;
    }
    Similarity sim;
    sim.R = svd.matrixU() * d * svd.matrixV().transpose();
    sim.scale = (svd.singularValues().asDiagonal() * d).trace() / var_src;
    sim.t = mu_dst - sim.scale * sim.R * mu_src;
    return sim;
}

double rotation_error_deg(const Eigen::Matrix3d& a, const Eigen::Matrix3d& b) {
    const Eigen::Matrix3d r = a.transpose() * b;
    const double cos_angle = std::min(1.0, std::max(-1.0, (r.trace() - 1.0) * 0.5));
    return std::acos(cos_angle) * 180.0 / 3.14159265358979323846;
}

/// Align the registered reconstructed camera centres to the truth, then report
/// the worst rotation and centre errors and how many tracks got a point.
struct AlignmentReport {
    Similarity sim;
    double worst_rotation_deg = 0.0;
    double worst_centre = 0.0;
    std::size_t aligned_points = 0;
};

AlignmentReport evaluate(const GtScene& scene, const BuilderResult& result) {
    AlignmentReport report;

    std::vector<Eigen::Vector3d> recon, truth;
    std::vector<std::size_t> cam_indices;
    for (std::size_t c = 0; c < result.poses.size(); ++c) {
        if (result.poses[c].has_value()) {
            recon.push_back(camera_centre(*result.poses[c]));
            truth.push_back(scene.true_centres[c]);
            cam_indices.push_back(c);
        }
    }

    // Align on the reconstructed points when there are enough of them: 600
    // points spanning the whole box give a far better-conditioned similarity
    // than 12 cameras on an arc. Fall back to the camera centres otherwise.
    std::vector<Eigen::Vector3d> recon_points, truth_points;
    for (std::size_t t = 0; t < result.points.size(); ++t) {
        if (!result.points[t].has_value()) {
            continue;
        }
        report.aligned_points += 1;
        recon_points.push_back(*result.points[t]);
        truth_points.push_back(scene.true_points[scene.track_to_point[t]]);
    }
    report.sim = recon_points.size() >= 20 ? umeyama(recon_points, truth_points)
                                           : umeyama(recon, truth);

    for (std::size_t k = 0; k < cam_indices.size(); ++k) {
        const std::size_t c = cam_indices[k];
        // The similarity maps reconstruction -> truth (x_true = s R x_recon +
        // t), so a camera's rotation in the truth frame is R_cam * R^T.
        const Eigen::Matrix3d R_aligned = result.poses[c]->R * report.sim.R.transpose();
        report.worst_rotation_deg =
            std::max(report.worst_rotation_deg, rotation_error_deg(R_aligned, scene.true_poses[c].R));
        const Eigen::Vector3d centre = report.sim.scale * report.sim.R * recon[k] + report.sim.t;
        report.worst_centre = std::max(report.worst_centre, (centre - scene.true_centres[c]).norm());
    }
    return report;
}

// TESTS

TEST(StructureBuilder, RecoversArcScene) {
    const GtScene scene = make_arc_scene(1, 0.5);
    BuilderOptions options;

    const auto t0 = std::chrono::steady_clock::now();
    const BuilderResult result = build_incremental(scene.input, options);
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::printf("[structure_builder] build time = %.2f s\n", seconds);

    EXPECT_EQ(result.registered, scene.input.cameras.size());
    EXPECT_LT(result.final_rms_px, 0.8);

    const AlignmentReport report = evaluate(scene, result);
    std::printf("[structure_builder] worst rot = %.4f deg, worst centre = %.4f (radius %.1f) rms=%.4f\n",
                report.worst_rotation_deg, report.worst_centre, kArcRadius, result.final_rms_px);
    EXPECT_LT(report.worst_rotation_deg, 0.1);
    EXPECT_LT(report.worst_centre, 0.005 * kArcRadius);

    const std::size_t total = scene.input.tracks.size();
    EXPECT_GE(report.aligned_points, total * 9 / 10) << report.aligned_points << " / " << total;
}

TEST(StructureBuilder, ToleratesOutlierObservations) {
    GtScene scene = make_arc_scene(2, 0.5);

    // Corrupt 5% of the observations with a random pixel in the image.
    std::mt19937 rng(4242);
    const auto px = [](std::mt19937& g) { return rng::uniform_real(g, 0.0, kWidth); };
    const auto py = [](std::mt19937& g) { return rng::uniform_real(g, 0.0, kHeight); };
    std::vector<std::size_t> corrupted_tracks;
    for (std::size_t t = 0; t < scene.input.tracks.size(); ++t) {
        bool corrupted = false;
        for (Observation& o : scene.input.tracks[t].observations) {
            if ((rng() % 20) == 0) {
                BuilderCamera& cam = scene.input.cameras[static_cast<std::size_t>(o.camera_id)];
                cam.pixels[o.keypoint_index] = draw2(rng, px, py);
                corrupted = true;
            }
        }
        if (corrupted) {
            corrupted_tracks.push_back(t);
        }
    }

    BuilderOptions options;
    const BuilderResult result = build_incremental(scene.input, options);

    EXPECT_EQ(result.registered, scene.input.cameras.size());
    const AlignmentReport report = evaluate(scene, result);
    std::printf("[structure_builder] outlier run: worst centre = %.4f, points = %zu\n",
                report.worst_centre, report.aligned_points);
    EXPECT_LT(report.worst_centre, 0.01 * kArcRadius);

    // The corrupted tracks were mostly filtered out (their points cleared).
    std::size_t filtered = 0;
    for (std::size_t t : corrupted_tracks) {
        // size_t on both arms: g++ rejects the int 0/1 mix (-Wsign-conversion).
        filtered += result.points[t].has_value() ? std::size_t{0} : std::size_t{1};
    }
    std::printf("[structure_builder] corrupted tracks = %zu, filtered out = %zu\n",
                corrupted_tracks.size(), filtered);
    EXPECT_GE(filtered * 2, corrupted_tracks.size());
}

/// A private two-camera cluster (cameras `a` and `b` looking at their own cloud)
/// that no arc camera shares tracks with. Both stay unregistered; the arc is
/// still recovered.
TEST(StructureBuilder, DisconnectedCameraIsLeftUnregistered) {
    GtScene scene = make_arc_scene(3, 0.5);
    const std::size_t extra_a = scene.input.cameras.size();
    const std::size_t extra_b = extra_a + 1;

    // Two cameras facing -Z, flanking a cloud centred at (0,0,-kArcRadius).
    const Eigen::Vector3d centre_a(2.0, 0.0, -kArcRadius);
    const Eigen::Vector3d centre_b(-2.0, 0.0, -kArcRadius);
    auto make_private = [&](std::size_t id, const Eigen::Vector3d& centre) {
        BuilderCamera bc;
        bc.id = static_cast<std::int32_t>(id);
        bc.calibration = scene.cal;
        const Eigen::Vector3d forward = (-centre).normalized();
        const Eigen::Vector3d right = forward.cross(Eigen::Vector3d::UnitY()).normalized();
        const Eigen::Vector3d up = right.cross(forward);
        Eigen::Matrix3d R_wc;
        R_wc.col(0) = right;
        R_wc.col(1) = -up;
        R_wc.col(2) = forward;
        AbsolutePose pose;
        pose.R = R_wc.transpose();
        pose.t = -pose.R * centre;
        scene.input.cameras.push_back(bc);
        return pose;
    };
    const AbsolutePose pose_a = make_private(extra_a, centre_a);
    const AbsolutePose pose_b = make_private(extra_b, centre_b);

    // A private cloud, one track per point, both private cameras seeing it.
    std::mt19937 rng(11);
    const auto noise = [](std::mt19937& g) { return rng::normal(g, 0.0, 0.5); };
    const auto d = [](std::mt19937& g) { return rng::uniform_real(g, -2.0, 2.0); };
    std::vector<Eigen::Vector2d> pix_a, pix_b;
    std::vector<std::pair<int, int>> obs;
    for (int k = 0; k < 120; ++k) {
        const Eigen::Vector3d ddd = draw3(rng, d, d, d);
        const Eigen::Vector3d p(ddd.x(), ddd.y(), -kArcRadius + ddd.z());
        Eigen::Vector2d ua, ub;
        try {
            ua = scene.cal.project(pose_a.R * p + pose_a.t);
            ub = scene.cal.project(pose_b.R * p + pose_b.t);
        } catch (const ValueError&) {
            continue;
        }
        if (ua.x() < 0 || ua.x() > kWidth || ua.y() < 0 || ua.y() > kHeight ||
            ub.x() < 0 || ub.x() > kWidth || ub.y() < 0 || ub.y() > kHeight) {
            continue;
        }
        obs.emplace_back(static_cast<int>(pix_a.size()), static_cast<int>(pix_b.size()));
        pix_a.push_back(ua + draw2(rng, noise, noise));
        pix_b.push_back(ub + draw2(rng, noise, noise));
    }
    scene.input.cameras[extra_a].pixels = pix_a;
    scene.input.cameras[extra_b].pixels = pix_b;

    PairMatches private_pair;
    private_pair.camera_a = static_cast<std::int32_t>(extra_a);
    private_pair.camera_b = static_cast<std::int32_t>(extra_b);
    for (const auto& [ia, ib] : obs) {
        private_pair.matches.push_back(
            KeypointMatch{static_cast<std::uint32_t>(ia), static_cast<std::uint32_t>(ib)});
        Track track;
        track.observations.push_back(
            Observation{static_cast<std::int32_t>(extra_a), static_cast<std::uint32_t>(ia)});
        track.observations.push_back(
            Observation{static_cast<std::int32_t>(extra_b), static_cast<std::uint32_t>(ib)});
        scene.track_to_point.push_back(0);  // unused: this cloud is never checked
        scene.input.tracks.push_back(track);
    }
    scene.input.matches.push_back(private_pair);

    const BuilderResult result = build_incremental(scene.input, BuilderOptions{});

    EXPECT_FALSE(result.poses[extra_a].has_value()) << "disconnected camera must stay unregistered";
    EXPECT_FALSE(result.poses[extra_b].has_value()) << "disconnected camera must stay unregistered";
    // The 12 arc cameras still register.
    for (std::size_t c = 0; c < extra_a; ++c) {
        EXPECT_TRUE(result.poses[c].has_value()) << "arc camera " << c;
    }
    EXPECT_EQ(result.registered, extra_a);
}

TEST(StructureBuilder, ContinuesFromInitialPoses) {
    const GtScene scene = make_arc_scene(4, 0.5);

    const BuilderResult full = build_incremental(scene.input, BuilderOptions{});
    ASSERT_EQ(full.registered, scene.input.cameras.size());

    // Rerun with the first 6 cameras' poses and the points of the tracks those
    // cameras observe as the initial state.
    BuilderInput continued = scene.input;
    for (std::size_t c = 0; c < 6; ++c) {
        ASSERT_TRUE(full.poses[c].has_value());
        continued.cameras[c].initial_pose = full.poses[c];
    }
    for (std::size_t t = 0; t < continued.tracks.size(); ++t) {
        continued.tracks[t].point = full.points[t];
    }

    const BuilderResult restart = build_incremental(continued, BuilderOptions{});
    EXPECT_EQ(restart.registered, scene.input.cameras.size());
    for (std::size_t c = 0; c < 6; ++c) {
        EXPECT_TRUE(restart.poses[c].has_value());
    }

    // The 6 seeded cameras must barely move relative to the scene size.
    const double scale = kArcRadius;
    for (std::size_t c = 0; c < 6; ++c) {
        const Eigen::Vector3d before = camera_centre(*full.poses[c]);
        const Eigen::Vector3d after = camera_centre(*restart.poses[c]);
        EXPECT_LT((after - before).norm(), 1e-3 * scale) << "camera " << c;
    }
}

TEST(StructureBuilder, Deterministic) {
    const GtScene scene = make_arc_scene(5, 0.5);

    const BuilderResult first = build_incremental(scene.input, BuilderOptions{});
    const BuilderResult second = build_incremental(scene.input, BuilderOptions{});

    ASSERT_EQ(first.poses.size(), second.poses.size());
    ASSERT_EQ(first.points.size(), second.points.size());
    EXPECT_EQ(first.registered, second.registered);
    for (std::size_t c = 0; c < first.poses.size(); ++c) {
        ASSERT_EQ(first.poses[c].has_value(), second.poses[c].has_value());
        if (first.poses[c].has_value()) {
            EXPECT_TRUE(first.poses[c]->R == second.poses[c]->R) << "camera " << c;
            EXPECT_TRUE(first.poses[c]->t == second.poses[c]->t) << "camera " << c;
        }
    }
    for (std::size_t t = 0; t < first.points.size(); ++t) {
        ASSERT_EQ(first.points[t].has_value(), second.points[t].has_value());
        if (first.points[t].has_value()) {
            EXPECT_TRUE(*first.points[t] == *second.points[t]) << "track " << t;
        }
    }
}

TEST(StructureBuilder, Cancellation) {
    const GtScene scene = make_arc_scene(6, 0.5);

    std::size_t calls = 0;
    const auto cancelled = [&calls]() {
        ++calls;
        return calls > 3;  // true after 3 registrations
    };
    EXPECT_THROW(build_incremental(scene.input, BuilderOptions{}, cancelled), Error);
}

TEST(StructureBuilder, MalformedInputThrows) {
    const GtScene scene = make_arc_scene(7, 0.5);

    {
        BuilderInput bad = scene.input;
        bad.tracks.front().observations.front().camera_id = 9999;
        EXPECT_THROW(build_incremental(bad, BuilderOptions{}), ValueError);
    }
    {
        BuilderInput bad = scene.input;
        bad.tracks.front().observations.front().keypoint_index = 100000;
        EXPECT_THROW(build_incremental(bad, BuilderOptions{}), ValueError);
    }
}

// TESTS

}  // namespace
}  // namespace alphashape::sfm
