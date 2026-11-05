// test_attitude_prior.cpp -- measured camera attitude priors (WP-13.21).
//
// A CameraReference may carry an orientation (yaw/pitch/roll) and its 1-sigma
// accuracy. OptimizeCameras turns it into a soft SO(3) prior in bundle
// adjustment, weighted per axis so heading (yaw) can be much looser than the
// stabilised pitch/roll. These tests pin the residual at truth, the per-axis
// split, the stabilisation of a weakly observed camera, the no-op path, the
// boresight mount, persistence and the Python-visible property.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/io/zip.hpp"
#include "common/algorithm/reference/convert_reference.hpp"
#include "common/algorithm/reference/xmp_reference.hpp"
#include "common/algorithm/sfm/bundle_adjustment.hpp"
#include "common/algorithm/sfm/bundle_adjustment_detail.hpp"
#include "common/algorithm/sfm/georeference.hpp"
#include "common/core/document.hpp"
#include "common/io/doc_xml.hpp"
#include "common/io/project_io.hpp"
#include "common/tasks/registry.hpp"
#include "model/crs/euler.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

Transform pose_with_centre(const Eigen::Matrix3d& r, const Eigen::Vector3d& centre) {
    Transform pose = Transform::Identity();
    pose.block<3, 3>(0, 0) = r;
    pose.block<3, 1>(0, 3) = -r * centre;
    return pose;
}

/// Rz(90 deg), the fixed rotation between the ENU basis and the zero-heading
/// body frame in the stored-orientation contract.
Eigen::Matrix3d rz90() {
    Eigen::Matrix3d m;
    m << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    return m;
}

/// The inverse of the `convert_reference` contract for `order`:
/// R_wb = Rz(90) * to_matrix({x, -y, z}), so the stored triple is
/// (angles.yaw, -angles.pitch, angles.roll) of `Rz(90)^T * R_wb`.
Eigen::Vector3d stored_from_world_body(const Eigen::Matrix3d& r_wb,
                                       crs::RotationOrder order) {
    const crs::EulerAngles a = crs::from_matrix(rz90().transpose() * r_wb, order);
    return Eigen::Vector3d(a.yaw, -a.pitch, a.roll);
}

/// A ring of nadir cameras over a ground point grid, in a metric ENU local
/// frame (the chunk transform is ENU->ECEF at `origin`). The airframe is level
/// (body forward horizontal, pitch and roll zero), so the stored body attitude
/// is well conditioned in every rotation order. `mount` is the body->camera
/// mount used to synthesise the truth: the pose's world->camera rotation is
/// R = mount * R_wb^T, so the stored body attitude is R_wb = R^T * mount.
struct Scene {
    Document document;
    geo::Geodetic origin;
    Eigen::Matrix3d mount = reference::body_to_camera_nadir();
    /// True world->camera rotation per camera.
    std::vector<Eigen::Matrix3d> rotations;
    /// True body->world rotation per camera.
    std::vector<Eigen::Matrix3d> bodies;
    /// True stored orientation triple per camera.
    std::vector<Eigen::Vector3d> orientations;
};

/// Rz(angle) about ENU up, i.e. a body yaw.
Eigen::Matrix3d yaw_about_up(double radians) {
    Eigen::Matrix3d m;
    m << std::cos(radians), -std::sin(radians), 0.0,  //
        std::sin(radians), std::cos(radians), 0.0,     //
        0.0, 0.0, 1.0;
    return m;
}

Scene make_scene(crs::RotationOrder order, const Eigen::Matrix3d& mount,
                 bool with_orientation) {
    Scene scene;
    scene.mount = mount;
    Chunk& chunk = scene.document.workspace().add_chunk("attitude");
    scene.document.set_active_chunk_id(chunk.id());
    scene.origin = geo::Geodetic{7.0, 45.0, 100.0};
    chunk.transform() = geo::enu_to_ecef(scene.origin);
    chunk.meta().set(reference::kRotationOrderKey, crs::rotation_order_name(order));

    const Calibration truth = Calibration::initial(CameraType::Frame, 1000, 800, 900.0);
    Sensor sensor;
    sensor.label = "synth";
    sensor.type = CameraType::Frame;
    sensor.width = truth.width;
    sensor.height = truth.height;
    sensor.calibration = truth;
    chunk.sensors().push_back(sensor);

    // The stored attitude is relative to the CRS local ENU at the camera's own
    // position, while the poses live in the chunk-local ENU at the origin.
    const Eigen::Matrix3d enu_origin = geo::enu_to_ecef(scene.origin).block<3, 3>(0, 0);
    constexpr int kCameras = 8;
    std::vector<KeypointSet> sets(static_cast<std::size_t>(kCameras));
    for (int i = 0; i < kCameras; ++i) {
        const double azimuth =
            2.0 * kPi * static_cast<double>(i) / static_cast<double>(kCameras);
        // Level airframe at a compass heading, positioned above the grid. The
        // camera looks straight down because the mount is the nadir one. The
        // pose rotation is world->camera: R = mount * R_wb^T.
        const Eigen::Vector3d centre(2.0 * std::cos(azimuth), 2.0 * std::sin(azimuth),
                                     10.0);
        const Eigen::Matrix3d body = yaw_about_up(azimuth);
        const Eigen::Matrix3d rotation = mount * body.transpose();

        Camera camera;
        camera.id = i;
        camera.sensor_id = 0;
        camera.enabled = true;
        camera.pose = pose_with_centre(rotation, centre);
        const geo::Geodetic g = geo::enu_to_geodetic(centre, scene.origin);
        camera.reference.location =
            Eigen::Vector3d(g.longitude_deg, g.latitude_deg, g.height_m);
        camera.reference.accuracy = Eigen::Vector3d::Constant(0.02);
        camera.reference.enabled = true;
        // R_wb in the camera's own ENU: E_c^T E_o * R_wb, where E is
        // ECEF-from-ENU. The chunk-local body rotation is in E_o's ENU.
        const Eigen::Matrix3d enu_cam = geo::enu_to_ecef(g).block<3, 3>(0, 0);
        const Eigen::Matrix3d r_wb = enu_cam.transpose() * enu_origin * body;
        const Eigen::Vector3d stored = stored_from_world_body(r_wb, order);
        if (with_orientation) {
            camera.reference.orientation_deg = stored;
        }
        chunk.cameras().push_back(camera);
        scene.rotations.push_back(rotation);
        scene.bodies.push_back(body);
        scene.orientations.push_back(stored);
        sets[static_cast<std::size_t>(i)].camera_id = i;
    }

    for (int gx = -2; gx <= 2; ++gx) {
        for (int gy = -2; gy <= 2; ++gy) {
            for (int gz = -1; gz <= 1; ++gz) {
                const Eigen::Vector3d point(
                    0.6 * static_cast<double>(gx), 0.6 * static_cast<double>(gy),
                    0.6 * static_cast<double>(gz));
                Track track;
                for (int i = 0; i < kCameras; ++i) {
                    const Transform& pose =
                        chunk.cameras()[static_cast<std::size_t>(i)].pose;
                    const Eigen::Vector3d p_cam =
                        pose.block<3, 3>(0, 0) * point + pose.block<3, 1>(0, 3);
                    if (!(p_cam.z() > 1.0)) {
                        continue;
                    }
                    KeypointSet& set = sets[static_cast<std::size_t>(i)];
                    const Eigen::Vector2d pixel = truth.project(p_cam);
                    Keypoint keypoint;
                    keypoint.x = static_cast<float>(pixel.x() - 0.5);
                    keypoint.y = static_cast<float>(pixel.y() - 0.5);
                    set.keypoints.push_back(keypoint);
                    Observation observation;
                    observation.camera_id = i;
                    observation.keypoint_index =
                        static_cast<std::uint32_t>(set.keypoints.size() - 1);
                    track.observations.push_back(observation);
                }
                if (track.observations.size() >= 2U) {
                    track.point = point;
                    chunk.tie_points().tracks.push_back(track);
                }
            }
        }
    }
    for (KeypointSet& set : sets) {
        chunk.tie_points().keypoint_sets.push_back(std::move(set));
    }
    return scene;
}

void run_optimize(Scene& scene, bool use_orientation = true) {
    Params params = make_params(builtin_registry().require("OptimizeCameras"));
    for (const char* name :
         {"f", "cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4"}) {
        params.set(std::string("fit_") + name, false);
    }
    params.set("camera_covariance", false);
    params.set("use_orientation", use_orientation);
    run_task(builtin_registry(), scene.document, "OptimizeCameras", params);
}

double meta_double(const Chunk& chunk, const std::string& key) {
    const std::string* text = chunk.meta().find(key);
    return text == nullptr ? std::nan("") : std::stod(*text);
}

/// The (yaw, pitch, roll) rotation-vector error of `solved` against `truth`,
/// in degrees, in the heading frame supplied (columns forward/left/up).
Eigen::Vector3d heading_error_deg(const Eigen::Matrix3d& solved,
                                  const Eigen::Matrix3d& truth,
                                  const Eigen::Matrix3d& frame) {
    const Eigen::AngleAxisd aa(solved.transpose() * truth);
    const Eigen::Vector3d dw = aa.angle() * aa.axis();
    const Eigen::Vector3d dh = frame.transpose() * dw;
    return Eigen::Vector3d(dh.z(), dh.y(), dh.x()) / kDeg;
}

/// ENU(camera) from chunk-local for camera `i`, i.e. E_c^T E_o.
Eigen::Matrix3d enu_cam_from_internal(const Scene& scene, std::size_t i) {
    const Camera& camera = scene.document.active_chunk()->cameras()[i];
    const Eigen::Vector3d& loc = *camera.reference.location;
    const Eigen::Matrix3d e_o = geo::enu_to_ecef(scene.origin).block<3, 3>(0, 0);
    const Eigen::Matrix3d e_c =
        geo::enu_to_ecef(geo::Geodetic{loc.x(), loc.y(), loc.z()}).block<3, 3>(0, 0);
    return e_c.transpose() * e_o;
}

/// The heading frame of camera `i`'s true body attitude, in the chunk-local
/// frame. Built from R_wb (not the stored angles), so it is valid in every
/// rotation order.
Eigen::Matrix3d heading_frame(const Scene& scene, std::size_t i) {
    // scene.bodies holds the true body attitude in chunk-local axes, which is
    // the frame the heading frame is expressed in.
    return sfm::detail::heading_frame_from_body(scene.bodies[i]);
}

/// The forward map of the stored-orientation contract.
Eigen::Matrix3d r_wb_from_stored(const Eigen::Vector3d& stored,
                                 crs::RotationOrder order) {
    return rz90() * crs::to_matrix(
                        crs::EulerAngles{stored.x(), -stored.y(), stored.z(), order});
}

TEST(AttitudePrior, ZeroResidualAtTruth) {
    for (crs::RotationOrder order : {crs::RotationOrder::ZYX, crs::RotationOrder::XYZ}) {
        Scene scene = make_scene(order, reference::body_to_camera_nadir(), true);
        // The BA residual itself at truth: Log(R^T R) = 0.
        sfm::BARotationPrior prior;
        prior.measured = scene.rotations[0];
        prior.heading_frame = heading_frame(scene, 0);
        prior.sigma_rad = Eigen::Vector3d::Ones();
        sfm::BAPose pose;
        pose.R = scene.rotations[0];
        const Eigen::Vector3d residual = sfm::predicted_rotation_residual(pose, prior);

        run_optimize(scene);
        const double rms = meta_double(*scene.document.active_chunk(),
                                       "optimize_cameras.attitude_rms_deg");
        std::printf("[attitude] %s truth: residual %.3e rad, task RMS %.3e deg\n",
                    crs::rotation_order_name(order), residual.norm(), rms);
        EXPECT_LT(residual.norm(), 1e-9);
        // End to end: keypoints are stored as float, so a sub-1e-4 px rounding
        // error is expected; the attitude residual stays far below that.
        EXPECT_LT(rms, 1e-4);
    }
}

TEST(AttitudePrior, YawSigmaSplit) {
    const Eigen::Matrix3d r_bc = reference::body_to_camera_nadir();
    // Yaw: a 2 degree measurement error with a huge yaw sigma must be ignored
    // by the image-driven solution.
    {
        Scene scene = make_scene(crs::RotationOrder::ZYX, r_bc, true);
        Chunk& chunk = *scene.document.active_chunk();
        Eigen::Vector3d stored = scene.orientations[0];
        stored.x() += 2.0;
        chunk.cameras()[0].reference.orientation_deg = stored;
        chunk.cameras()[0].reference.rotation_accuracy_deg =
            Eigen::Vector3d(1e6, 0.01, 0.01);
        const Eigen::Matrix3d hf = heading_frame(scene, 0);
        // World->camera of the measurement: R_bc * R_wb^T.
        const Eigen::Matrix3d measured =
            r_bc * r_wb_from_stored(stored, crs::RotationOrder::ZYX).transpose() *
            enu_cam_from_internal(scene, std::size_t{0});
        const Eigen::Vector3d injected =
            heading_error_deg(scene.rotations[0], measured, hf);
        run_optimize(scene);
        const Eigen::Vector3d solved = heading_error_deg(
            scene.document.active_chunk()->cameras()[0].pose.block<3, 3>(0, 0),
            scene.rotations[0], hf);
        std::printf("[attitude] yaw split: injected yaw %.4f deg, solved yaw %.4f deg\n",
                    injected.x(), solved.x());
        EXPECT_LT(std::abs(solved.x()), 0.1 * std::abs(injected.x()));
    }
    // Pitch: the same 2 degree error with a tight pitch sigma must pull the
    // solution onto the measurement about the pitch axis, and only that axis.
    {
        Scene scene = make_scene(crs::RotationOrder::ZYX, r_bc, true);
        Chunk& chunk = *scene.document.active_chunk();
        Eigen::Vector3d stored = scene.orientations[0];
        stored.y() += 2.0;
        chunk.cameras()[0].reference.orientation_deg = stored;
        chunk.cameras()[0].reference.rotation_accuracy_deg =
            Eigen::Vector3d(1e6, 0.005, 0.005);
        const Eigen::Matrix3d hf = heading_frame(scene, 0);
        const Eigen::Matrix3d measured =
            r_bc * r_wb_from_stored(stored, crs::RotationOrder::ZYX).transpose() *
            enu_cam_from_internal(scene, std::size_t{0});
        const Eigen::Vector3d injected =
            heading_error_deg(scene.rotations[0], measured, hf);
        run_optimize(scene);
        const Eigen::Vector3d solved = heading_error_deg(
            scene.document.active_chunk()->cameras()[0].pose.block<3, 3>(0, 0),
            scene.rotations[0], hf);
        std::printf("[attitude] pitch split: injected pitch %.4f deg, solved pitch "
                    "%.4f deg, solved yaw %.4f deg\n",
                    injected.y(), solved.y(), solved.x());
        // The estimate moves towards the measurement, i.e. opposite the
        // truth->measurement error, by most of the injected amount.
        EXPECT_LT(solved.y() * injected.y(), 0.0);
        EXPECT_GT(std::abs(solved.y()), 0.5 * std::abs(injected.y()));
        // The response is on the pitch axis, not the heading one.
        EXPECT_LT(std::abs(solved.x()), 0.5 * std::abs(solved.y()));
    }
}

/// A two-camera problem whose second camera observes points very close to its
/// optical axis with 1 px of deterministic noise, so its roll is only weakly
/// determined. Camera 0 is the gauge; a scale anchor frees the other five DoF.
struct WeakScene {
    sfm::BAProblem problem;
    sfm::BARotationPrior prior;
};

WeakScene make_weak_scene() {
    WeakScene out;
    const Calibration cal = Calibration::initial(CameraType::Frame, 640, 480, 500.0);
    out.problem.calibrations = {cal};
    out.problem.camera_sensor = {0, 0};
    const Eigen::Vector3d c1(2.0, 0.0, 10.0);
    sfm::BAPose p0;
    sfm::BAPose p1;
    // The estimate starts 2 degrees of roll away from the truth (R = identity).
    p1.R = Eigen::AngleAxisd(2.0 * kDeg, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    p1.t = -p1.R * c1;
    out.problem.poses = {p0, p1};
    out.problem.fixed_pose = {1, 0};
    out.problem.scale_anchor_camera = 1;
    out.problem.scale_anchor_axis = sfm::choose_scale_anchor_axis(p0, p1);
    for (int i = 0; i < 8; ++i) {
        const double angle = 2.0 * kPi * static_cast<double>(i) / 8.0;
        const Eigen::Vector3d offset(0.04 * std::cos(angle), 0.04 * std::sin(angle), 0.0);
        const Eigen::Vector3d point =
            c1 + offset + Eigen::Vector3d(0.0, 0.0, 1.5 + 0.2 * static_cast<double>(i));
        const std::size_t index = out.problem.points.size();
        out.problem.points.push_back(point);
        const Eigen::Vector2d uv0 = cal.project(point);
        out.problem.observations.push_back({std::size_t{0}, index, uv0});
        // A tangential 1 px shift is what a small roll looks like: it makes
        // the roll genuinely (if weakly) observable from noisy data.
        const Eigen::Vector2d tangential(-std::sin(angle), std::cos(angle));
        const Eigen::Vector2d uv1 = cal.project(point - c1) + tangential;
        out.problem.observations.push_back({std::size_t{1}, index, uv1});
    }
    out.prior.camera = 1;
    out.prior.measured = Eigen::Matrix3d::Identity();  // R_wc truth
    // The camera is level, so its body frame is the heading frame and its
    // forward axis (camera +Z) is the roll axis.
    out.prior.heading_frame = reference::body_to_camera();
    out.prior.sigma_rad = Eigen::Vector3d(1.0, 1.0, 0.001) * kDeg;
    return out;
}

TEST(AttitudePrior, WeakCameraStabilised) {
    sfm::BAOptions options;
    WeakScene loose = make_weak_scene();
    sfm::bundle_adjust(loose.problem, options);
    const double loose_roll =
        std::abs(sfm::predicted_rotation_residual(loose.problem.poses[1], loose.prior).z()) /
        kDeg;
    WeakScene stabilised = make_weak_scene();
    stabilised.problem.rotation_priors.push_back(stabilised.prior);
    sfm::bundle_adjust(stabilised.problem, options);
    const double tight_roll =
        std::abs(sfm::predicted_rotation_residual(stabilised.problem.poses[1],
                                                  stabilised.prior)
                     .z()) /
        kDeg;
    std::printf("[attitude] weak camera roll error: no prior %.4f deg, prior %.4f deg\n",
                loose_roll, tight_roll);
    EXPECT_GT(loose_roll, 0.01);
    EXPECT_LT(tight_roll, loose_roll / 5.0);
}

TEST(AttitudePrior, NoPriorWithoutOrientation) {
    Scene on = make_scene(crs::RotationOrder::ZYX, reference::body_to_camera_nadir(), false);
    Scene off = make_scene(crs::RotationOrder::ZYX, reference::body_to_camera_nadir(), false);
    run_optimize(on, true);
    run_optimize(off, false);
    const Chunk& a = *on.document.active_chunk();
    const Chunk& b = *off.document.active_chunk();
    // No rotation priors exist either way, so the geometry is bit-identical.
    // (The whole Chunk differs only in the recorded use_orientation flag.)
    EXPECT_TRUE(a.cameras() == b.cameras());
    EXPECT_TRUE(a.tie_points() == b.tie_points());
    EXPECT_TRUE(a.sensors() == b.sensors());
    EXPECT_EQ((a.transform() - b.transform()).cwiseAbs().maxCoeff(), 0.0);
    EXPECT_EQ(a.meta().find("optimize_cameras.0.attitude_error_deg"), nullptr);
    // Orientation-free projects keep their pre-WP-13.21 metadata exactly.
    EXPECT_EQ(a.meta().find("optimize_cameras.attitude_rms_deg"), nullptr);
}

TEST(AttitudePrior, OrientationIgnoredWhenDisabled) {
    // A chunk that carries orientations but solves with use_orientation=false
    // must produce the same answer as one with no orientations at all, and
    // must not record any attitude output.
    Scene carried = make_scene(crs::RotationOrder::ZYX, reference::body_to_camera_nadir(),
                               true);
    Scene absent = make_scene(crs::RotationOrder::ZYX, reference::body_to_camera_nadir(),
                              false);
    run_optimize(carried, false);
    run_optimize(absent, true);
    const Chunk& a = *carried.document.active_chunk();
    const Chunk& b = *absent.document.active_chunk();
    // The references differ (one carries orientations and the other does not),
    // so compare the solved poses rather than the whole Camera records.
    ASSERT_EQ(a.cameras().size(), b.cameras().size());
    for (std::size_t i = 0; i < a.cameras().size(); ++i) {
        EXPECT_EQ((a.cameras()[i].pose - b.cameras()[i].pose).cwiseAbs().maxCoeff(), 0.0);
    }
    EXPECT_TRUE(a.tie_points() == b.tie_points());
    EXPECT_TRUE(a.sensors() == b.sensors());
    EXPECT_EQ((a.transform() - b.transform()).cwiseAbs().maxCoeff(), 0.0);
    EXPECT_EQ(a.meta().find("optimize_cameras.0.attitude_error_deg"), nullptr);
    EXPECT_EQ(a.meta().find("optimize_cameras.attitude_rms_deg"), nullptr);
}

TEST(AttitudePrior, BoresightMount) {
    // A boresight replaces the default nadir mount, so start from the nadir
    // mount's Euler angles and perturb them. The scene's ground grid then stays
    // in front of the camera while the mount differs from the default by a few
    // degrees.
    const crs::EulerAngles nadir =
        crs::from_matrix(reference::body_to_camera_nadir(), crs::RotationOrder::ZYX);
    Antenna antenna;
    antenna.boresight_deg =
        Eigen::Vector3d(nadir.yaw + 2.0, nadir.pitch - 3.0, nadir.roll + 2.0);
    Scene scene = make_scene(crs::RotationOrder::ZYX, antenna.body_to_camera(), true);
    scene.document.active_chunk()->sensors()[0].antenna = antenna;
    run_optimize(scene);
    const double rms =
        meta_double(*scene.document.active_chunk(), "optimize_cameras.attitude_rms_deg");
    std::printf("[attitude] boresight mount RMS %.3e deg\n", rms);
    // Synthesised with the boresight mount, so the prior is at truth. The
    // default nadir mount here would leave a large non-zero residual.
    EXPECT_LT(rms, 1e-4);
}

TEST(AttitudePrior, XmpNadirPriorIsZeroAtTruth) {
    // A gimbal log describing a straight-down camera with heading 30 degrees.
    // AddPhotos stores orientation_zyx() unchanged in a ZYX chunk.
    XmpReference xmp;
    xmp.gimbal_yaw_deg = 30.0;
    xmp.gimbal_pitch_deg = -90.0;
    const std::optional<XmpOrientation> angles = xmp.orientation_zyx();
    ASSERT_TRUE(angles.has_value());
    const Eigen::Vector3d stored(angles->yaw_deg, angles->pitch_deg, angles->roll_deg);

    const Eigen::Matrix3d r_wb = r_wb_from_stored(stored, crs::RotationOrder::ZYX);
    const Eigen::Matrix3d mount = reference::body_to_camera_nadir();
    // The documented rule: the stored triple is a BODY attitude, and the
    // default mount is nadir, so the world->camera rotation is
    // R_wc = B * R_wb^T.
    const Eigen::Matrix3d r_wc = mount * r_wb.transpose();
    // A nadir camera has its optical axis straight down.
    EXPECT_NEAR(r_wc(2, 2), -1.0, 1e-9);

    sfm::BARotationPrior prior;
    prior.measured = mount * r_wb.transpose();
    prior.heading_frame = sfm::detail::heading_frame_from_body(r_wb);
    prior.sigma_rad = Eigen::Vector3d::Ones();
    sfm::BAPose pose;
    pose.R = r_wc;
    std::printf("[attitude] XMP nadir truth: residual %.3e rad\n",
                sfm::predicted_rotation_residual(pose, prior).norm());
    EXPECT_LT(sfm::predicted_rotation_residual(pose, prior).norm(), 1e-9);
}

TEST(AttitudePrior, RotationLogMatchesAngleAxis) {
    const Eigen::Vector3d axis = Eigen::Vector3d(1.0, 2.0, 3.0).normalized();
    // The near-pi branch is entered at cos(theta) < -0.999999, i.e. theta >
    // pi - 1.4e-3; pi - 1e-4 pins it, while pi - 0.01 exercises the generic
    // skew branch a step away from the limit.
    struct Case {
        double theta;
        double tolerance;
    };
    for (const Case& c : {Case{kPi - 0.01, 1e-6}, Case{kPi - 1e-4, 1e-6},
                          Case{1e-8, 1e-12}, Case{1.0, 1e-9}}) {
        const Eigen::Matrix3d rotation =
            Eigen::AngleAxisd(c.theta, axis).toRotationMatrix();
        const Eigen::Vector3d log = sfm::detail::rotation_log_matrix(rotation);
        EXPECT_LT((log - c.theta * axis).norm(), c.tolerance)
            << "theta " << c.theta << " log " << log.transpose();
    }
}

TEST(AttitudePrior, HeadingFrameFromBodyNotStoredAngles) {
    // The same body attitude written as a ZYX and an XYZ triple. The frame is
    // built from R_wb, so the order the angles were stored in cannot matter
    // (the old first-angle heading was only a yaw in ZYX).
    const Eigen::Matrix3d r_wb = rz90() * crs::to_matrix(
        crs::EulerAngles{25.0, -10.0, 5.0, crs::RotationOrder::ZYX});
    const Eigen::Matrix3d r_wb_xyz =
        crs::to_matrix(crs::from_matrix(r_wb, crs::RotationOrder::XYZ));
    EXPECT_TRUE(r_wb.isApprox(r_wb_xyz, 1e-12));
    bool degenerate_zyx = true;
    bool degenerate_xyz = true;
    const Eigen::Matrix3d frame_zyx =
        sfm::detail::heading_frame_from_body(r_wb, &degenerate_zyx);
    const Eigen::Matrix3d frame_xyz =
        sfm::detail::heading_frame_from_body(r_wb_xyz, &degenerate_xyz);
    EXPECT_FALSE(degenerate_zyx);
    EXPECT_FALSE(degenerate_xyz);
    EXPECT_TRUE(frame_zyx.isApprox(frame_xyz, 1e-12));
    // Columns are (forward-level, left-level, up).
    EXPECT_NEAR(frame_zyx(2, 0), 0.0, 1e-12);
    EXPECT_NEAR(frame_zyx.col(0).norm(), 1.0, 1e-12);
    EXPECT_TRUE(frame_zyx.col(2).isApprox(Eigen::Vector3d(0.0, 0.0, 1.0), 1e-15));
    EXPECT_TRUE(frame_zyx.col(1).isApprox(Eigen::Vector3d(0.0, 0.0, 1.0).cross(
                                              frame_zyx.col(0)),
                                          1e-12));
    // Forward is body x projected onto the horizon.
    Eigen::Vector3d flat = r_wb.col(0);
    flat.z() = 0.0;
    EXPECT_TRUE(frame_zyx.col(0).isApprox(flat.normalized(), 1e-12));
}

TEST(AttitudePrior, HeadingFrameFiniteAtVerticalBodyForward) {
    // Nose straight up (body pitch 90 degrees): the forward axis has no
    // horizontal component, so the frame must fall back to another body axis.
    const Eigen::Matrix3d r_wb = rz90() * crs::to_matrix(
        crs::EulerAngles{30.0, -90.0, 10.0, crs::RotationOrder::ZYX});
    bool degenerate = false;
    const Eigen::Matrix3d frame =
        sfm::detail::heading_frame_from_body(r_wb, &degenerate);
    EXPECT_TRUE(degenerate);
    EXPECT_TRUE(frame.allFinite());
    EXPECT_NEAR(frame.determinant(), 1.0, 1e-12);
    EXPECT_NEAR(frame.col(0).norm(), 1.0, 1e-12);
    EXPECT_NEAR(frame(2, 0), 0.0, 1e-12);
    sfm::BARotationPrior prior;
    prior.measured = Eigen::AngleAxisd(0.01, Eigen::Vector3d::UnitX()).toRotationMatrix();
    prior.heading_frame = frame;
    prior.sigma_rad = Eigen::Vector3d::Ones();
    sfm::BAPose pose;
    EXPECT_TRUE(sfm::predicted_rotation_residual(pose, prior).allFinite());
}

/// One chunk, one camera with a reference position and orientation. The
/// optional per-axis accuracy is what the persistence test round-trips.
Document persistence_document(bool with_rotation_accuracy) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("persist");
    document.set_active_chunk_id(chunk.id());
    Sensor sensor;
    sensor.label = "s";
    sensor.width = 640;
    sensor.height = 480;
    sensor.calibration = Calibration::initial(CameraType::Frame, 640, 480, 500.0);
    chunk.sensors().push_back(sensor);
    Camera camera;
    camera.id = 0;
    camera.sensor_id = 0;
    camera.label = "cam0";
    camera.reference.location = Eigen::Vector3d(7.0, 45.0, 100.0);
    camera.reference.orientation_deg = Eigen::Vector3d(10.0, 20.0, 30.0);
    if (with_rotation_accuracy) {
        camera.reference.rotation_accuracy_deg = Eigen::Vector3d(5.0, 0.5, 0.25);
    }
    chunk.cameras().push_back(camera);
    return document;
}

TEST(AttitudePrior, PersistRoundTrip) {
    Document document = persistence_document(true);
    ZipArchive archive;
    write_document(document, archive);
    EXPECT_NE(archive.get_text("doc.xml").find("rotation_accuracy=\""),
              std::string::npos);
    const std::shared_ptr<Document> loaded =
        read_document(ZipArchive::parse(archive.serialise()));
    ASSERT_NE(loaded, nullptr);
    const Chunk& back = *loaded->workspace().chunks().front();
    ASSERT_EQ(back.cameras().size(), 1U);
    ASSERT_TRUE(back.cameras()[0].reference.rotation_accuracy_deg.has_value());
    EXPECT_TRUE(back.cameras()[0].reference ==
                document.active_chunk()->cameras()[0].reference);

    // Omitting the field keeps the old layout: no attribute is written and
    // the loaded reference has no per-axis accuracy.
    Document plain = persistence_document(false);
    ZipArchive plain_archive;
    write_document(plain, plain_archive);
    EXPECT_EQ(plain_archive.get_text("doc.xml").find("rotation_accuracy"),
              std::string::npos);
    const std::shared_ptr<Document> plain_loaded =
        read_document(ZipArchive::parse(plain_archive.serialise()));
    ASSERT_NE(plain_loaded, nullptr);
    EXPECT_FALSE(plain_loaded->workspace()
                     .chunks()
                     .front()
                     ->cameras()[0]
                     .reference.rotation_accuracy_deg.has_value());
}

}  // namespace
}  // namespace alphashape
