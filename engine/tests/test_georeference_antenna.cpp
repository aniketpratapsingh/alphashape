// test_georeference_antenna.cpp -- UpdateTransform consumes the antenna mount.
//
// A camera's GNSS reference records the antenna phase centre, not the
// projection centre. georeference_chunk always corrected for the mount;
// update_transform (the weighted datum refit OptimizeCameras pairs with) did
// not, so with a mount set the datum and the BA centre prior disagreed. These
// tests build a synthetic block whose references are the exact antenna
// positions for a known lever arm, then check that update_transform recovers
// the true camera centres to numerical precision with the mount and misses by
// about the lever-arm length without it.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "common/algorithm/sfm/georeference.hpp"
#include "common/algorithm/sfm/similarity.hpp"
#include "common/core/chunk.hpp"
#include "model/camera/antenna.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape {
namespace {

namespace geo = alphashape::geo;

constexpr double kPi = 3.14159265358979323846;

/// World->camera 4x4 with rotation `r` whose projection centre is `centre`.
Transform pose_with_centre(const Eigen::Matrix3d& r, const Eigen::Vector3d& centre) {
    Transform pose = Transform::Identity();
    pose.block<3, 3>(0, 0) = r;
    pose.block<3, 1>(0, 3) = -r * centre;
    return pose;
}

/// The camera centre C = -R^T t of a world->camera pose.
Eigen::Vector3d centre_of(const Transform& pose) {
    return -pose.block<3, 3>(0, 0).transpose() * pose.block<3, 1>(0, 3);
}

/// The reference-origin ENU frame the synthetic block is built in.
const geo::Geodetic& reference_origin() {
    static const geo::Geodetic origin{7.0, 45.0, 100.0};
    return origin;
}

/// A known local <- ENU similarity the chunk starts in: the fit must undo it.
sfm::Similarity bad_similarity() {
    sfm::Similarity s;
    s.scale = 0.37;
    s.R = Eigen::AngleAxisd(35.0 * kPi / 180.0,
                            Eigen::Vector3d(0.2, 0.4, 0.9).normalized())
              .toRotationMatrix();
    s.t = Eigen::Vector3d(11.0, -6.0, 4.0);
    return s;
}

/// A ring of cameras looking at a central point, with alternating height and
/// per-camera roll so the lever arm is genuinely tangential and its world
/// direction varies from camera to camera (a constant offset could be absorbed
/// by the similarity translation and would not prove the rotation is applied).
struct TruthCamera {
    Eigen::Vector3d centre;
    Eigen::Matrix3d world_to_camera;
};

std::vector<TruthCamera> truth_cameras() {
    constexpr int kCameras = 8;
    std::vector<TruthCamera> cameras;
    cameras.reserve(kCameras);
    for (int i = 0; i < kCameras; ++i) {
        const double azimuth = 2.0 * kPi * static_cast<double>(i) / kCameras;
        const Eigen::Vector3d centre(6.0 * std::cos(azimuth),
                                     6.0 * std::sin(azimuth),
                                     (i % 2 == 0) ? 1.5 : -1.5);
        const Eigen::Vector3d z = -centre.normalized();
        const Eigen::Vector3d down(0.0, 0.0, -1.0);
        const Eigen::Vector3d y = (down - down.dot(z) * z).normalized();
        const Eigen::Vector3d x = y.cross(z).normalized();
        Eigen::Matrix3d rotation;
        rotation.row(0) = x.transpose();
        rotation.row(1) = y.transpose();
        rotation.row(2) = z.transpose();
        // A small roll about the optical axis varies the camera y direction.
        const Eigen::Matrix3d roll =
            Eigen::AngleAxisd(0.35 * static_cast<double>(i) / kCameras,
                              Eigen::Vector3d::UnitZ())
                .toRotationMatrix();
        cameras.push_back({centre, roll * rotation});
    }
    return cameras;
}

/// Build a chunk whose references are the ANTENNA positions for `lever`.
///
/// `mount_sensor` is the index carrying the mount; `camera_sensor` is the
/// sensor the cameras name. They differ when the mount lives on a master and
/// the cameras are slaves of it. When `with_antenna` is false the same
/// references are used but no mount is recorded: the uncorrected fit then
/// places the centres on the antenna positions themselves, missing the truth
/// by the lever-arm length.
Chunk make_chunk(const Eigen::Vector3d& lever, bool with_antenna,
                 bool mount_on_master, double accuracy_m = 0.02) {
    Chunk chunk("antenna-datum");
    const sfm::Similarity bad = bad_similarity();
    const std::vector<TruthCamera> truth = truth_cameras();

    const Calibration calibration =
        Calibration::initial(CameraType::Frame, 1000, 800, 900.0);
    Sensor master;
    master.label = "master";
    master.type = CameraType::Frame;
    master.width = calibration.width;
    master.height = calibration.height;
    master.calibration = calibration;
    Sensor slave;
    slave.label = "slave";
    slave.type = CameraType::Frame;
    slave.width = calibration.width;
    slave.height = calibration.height;
    slave.calibration = calibration;
    if (with_antenna) {
        Antenna antenna;
        antenna.offset_m = lever;
        if (mount_on_master) {
            master.antenna = antenna;
            // A contradictory mount on the slave must never be consumed: the
            // effective master's mount is the physical one.
            Antenna wrong;
            wrong.offset_m = -2.0 * lever;
            slave.antenna = wrong;
            slave.master_sensor_id = 0;
        } else {
            slave.antenna = antenna;
        }
    }
    chunk.sensors().push_back(master);
    chunk.sensors().push_back(slave);
    // The cameras always name the slave, in both layouts.
    const std::int32_t sensor_id = 1;

    for (std::size_t i = 0; i < truth.size(); ++i) {
        const TruthCamera& camera_truth = truth[i];
        // The camera-frame lever rotates with the camera; the reference is the
        // antenna phase centre at the true pose.
        const Eigen::Vector3d antenna_enu =
            camera_truth.centre + camera_truth.world_to_camera.transpose() * lever;
        const geo::Geodetic geodetic =
            geo::enu_to_geodetic(antenna_enu, reference_origin());

        Camera camera;
        camera.id = static_cast<std::int32_t>(i);
        camera.sensor_id = sensor_id;
        camera.enabled = true;
        // Local pose: the local->ENU rotation is bad.R^T, so the camera
        // rotation in the local frame is R_enu * bad.R^T; the centre is the
        // similarity applied to the true ENU centre.
        camera.pose = pose_with_centre(
            camera_truth.world_to_camera * bad.R.transpose(),
            bad.apply(camera_truth.centre));
        camera.reference.location = Eigen::Vector3d(
            geodetic.longitude_deg, geodetic.latitude_deg, geodetic.height_m);
        camera.reference.accuracy = Eigen::Vector3d::Constant(accuracy_m);
        camera.reference.enabled = true;
        chunk.cameras().push_back(camera);
    }
    return chunk;
}

/// The mean-ECEF origin of the chunk's references (the new ENU frame).
geo::Geodetic mean_ecef_origin(const Chunk& chunk) {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const Camera& camera : chunk.cameras()) {
        const Eigen::Vector3d& loc = *camera.reference.location;
        mean += geo::geodetic_to_ecef(geo::Geodetic{loc.x(), loc.y(), loc.z()});
    }
    mean /= static_cast<double>(chunk.cameras().size());
    return geo::ecef_to_geodetic(mean);
}

/// Worst distance between the transformed camera centres and the truth centres
/// expressed in the chunk's new ENU frame.
double centre_error(const Chunk& chunk) {
    const Transform to_new =
        geo::ecef_to_enu(mean_ecef_origin(chunk)) * geo::enu_to_ecef(reference_origin());
    const std::vector<TruthCamera> truth = truth_cameras();
    double worst = 0.0;
    for (std::size_t i = 0; i < truth.size(); ++i) {
        const Eigen::Vector3d expected =
            (to_new * truth[i].centre.homogeneous()).head<3>();
        worst = std::max(worst, (centre_of(chunk.cameras()[i].pose) - expected).norm());
    }
    return worst;
}

constexpr double kLeverMagnitude = 0.55;  // |(0.45, -0.2, -0.3)|

TEST(GeoreferenceAntenna, UpdateTransformRecoversDatumWithMount) {
    const Eigen::Vector3d lever(0.45, -0.2, -0.3);
    // A moderately loose accuracy lets the uncorrected seed include every
    // camera (the lever offsets would otherwise exceed a 0.1 m gate); the
    // correction then converges to the exact fit.
    Chunk chunk = make_chunk(lever, /*with_antenna=*/true, /*mount_on_master=*/false,
                             /*accuracy_m=*/0.2);
    const sfm::GeoreferenceReport report = sfm::update_transform(chunk);
    ASSERT_TRUE(report.applied) << report.reason;
    const double error = centre_error(chunk);
    std::printf("[georeference-antenna] mounted centre error: %.3e m\n", error);
    EXPECT_LT(error, 1e-5);
}

TEST(GeoreferenceAntenna, UpdateTransformWithoutMountMissesByLeverArm) {
    // Same references, same poses, only the mount missing: the fit has no way
    // to know the reference is not the centre, so it lands on the antenna and
    // misses the truth by about the lever-arm length.
    const Eigen::Vector3d lever(0.45, -0.2, -0.3);
    // A loose declared accuracy keeps the robust gate from discarding cameras
    // over the very lever arm this control is meant to expose.
    Chunk chunk = make_chunk(lever, /*with_antenna=*/false, /*mount_on_master=*/false,
                             /*accuracy_m=*/1.0);
    const sfm::GeoreferenceReport report = sfm::update_transform(chunk);
    ASSERT_TRUE(report.applied) << report.reason;
    const double error = centre_error(chunk);
    std::printf("[georeference-antenna] unmounted centre error: %.3f m\n", error);
    EXPECT_GT(error, 0.1);
    EXPECT_LT(error, 2.0 * kLeverMagnitude);
}

TEST(GeoreferenceAntenna, UpdateTransformResolvesMasterMount) {
    // The cameras are slaves; the mount is on the master, and the slave carries
    // a contradictory offset that must be ignored. If the direct sensor's
    // antenna were used, the recovered centres would be off by about twice the
    // lever arm and both assertions below fail.
    const Eigen::Vector3d lever(0.45, -0.2, -0.3);
    Chunk chunk = make_chunk(lever, /*with_antenna=*/true, /*mount_on_master=*/true,
                             /*accuracy_m=*/0.2);
    ASSERT_EQ(chunk.sensors()[1].master_sensor_id, 0);
    const sfm::GeoreferenceReport report = sfm::update_transform(chunk);
    ASSERT_TRUE(report.applied) << report.reason;
    const double error = centre_error(chunk);
    std::printf("[georeference-antenna] master-mount centre error: %.3e m\n", error);
    EXPECT_LT(error, 1e-5);
}

}  // namespace
}  // namespace alphashape
