// test_antenna.cpp -- GNSS antenna lever arm and boresight (WP-13.19).
//
// A camera reference records the antenna phase centre, not the projection
// centre. These tests pin the frame conventions of Antenna, the centre-prior
// evaluation that consumes the lever arm, persistence, georeferencing, the
// OptimizeCameras wiring and the undoable command.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/io/zip.hpp"
#include "common/algorithm/sfm/bundle_adjustment.hpp"
#include "common/algorithm/sfm/georeference.hpp"
#include "common/commands/command.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/document.hpp"
#include "common/io/doc_xml.hpp"
#include "common/tasks/registry.hpp"
#include "model/crs/euler.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape {
namespace {

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

TEST(Antenna, LeverArmShiftsPrediction) {
    // A camera rotated 90 degrees about world z: its x axis points along
    // world -y, so a 1 m lever arm along camera x moves the antenna 1 m along
    // world -y from the projection centre.
    Eigen::Matrix3d r;
    r << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    const Eigen::Vector3d centre(2.0, 3.0, 4.0);
    sfm::BAPose pose;
    pose.R = r;
    pose.t = -r * centre;

    const Eigen::Vector3d lever(1.0, 0.0, 0.0);  // camera x, metres
    const Eigen::Vector3d predicted = sfm::predicted_antenna_position(pose, lever);
    const Eigen::Vector3d shift = predicted - centre;
    const Eigen::Vector3d expected(0.0, -1.0, 0.0);

    EXPECT_LT((shift - expected).norm(), 1e-9);
    EXPECT_LT((predicted - (centre + expected)).norm(), 1e-9);
    // A zero lever still means "the reference measures the centre".
    EXPECT_LT((sfm::predicted_antenna_position(pose, Eigen::Vector3d::Zero()) - centre)
                  .norm(),
              1e-9);
}

TEST(Antenna, BoresightRotatesOffset) {
    // Positional Euler fields in the default ZYX order: (0, 0, 90) is a 90
    // degree roll about camera x, so a body +y offset becomes camera +z.
    Antenna antenna;
    antenna.offset_m = Eigen::Vector3d(0.0, 1.0, 0.0);
    antenna.boresight_deg = Eigen::Vector3d(0.0, 0.0, 90.0);
    const Eigen::Matrix3d expected =
        crs::to_matrix(crs::EulerAngles{0.0, 0.0, 90.0});
    EXPECT_LT((antenna.body_to_camera() - expected).norm(), 1e-12);
    EXPECT_LT((antenna.camera_offset() - Eigen::Vector3d(0.0, 0.0, 1.0)).norm(), 1e-12);

    sfm::BAPose pose;
    pose.R = Eigen::Matrix3d::Identity();
    pose.t = -pose.R * Eigen::Vector3d::Zero();
    EXPECT_LT((sfm::predicted_antenna_position(pose, antenna.camera_offset()) -
               Eigen::Vector3d(0.0, 0.0, 1.0))
                  .norm(),
              1e-12);
}

TEST(Antenna, UprightAndTiltedSigns) {
    // Nadir: camera x = world x, camera y = world -y (down), camera z = world
    // -z (forward/down). R^T o with o = (0.5, 0, 0.2) is (0.5, 0, -0.2).
    Eigen::Matrix3d nadir;
    nadir << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0;
    const Eigen::Vector3d centre(10.0, 20.0, 30.0);
    const Eigen::Vector3d offset(0.5, 0.0, 0.2);

    sfm::BAPose upright_pose;
    upright_pose.R = nadir;
    upright_pose.t = -nadir * centre;
    const Eigen::Vector3d upright = sfm::predicted_antenna_position(upright_pose, offset);
    EXPECT_LT((upright - Eigen::Vector3d(10.5, 20.0, 29.8)).norm(), 1e-12);

    // Tilt 30 degrees about world x: R' = Qx(30) * R_nadir, so
    // R'^T o = (0.5, -0.2 sin30, -0.2 cos30).
    const double c = std::sqrt(3.0) / 2.0;
    Eigen::Matrix3d tilted;
    tilted << 1.0, 0.0, 0.0, 0.0, -c, 0.5, 0.0, -0.5, -c;
    sfm::BAPose tilted_pose;
    tilted_pose.R = tilted;
    tilted_pose.t = -tilted * centre;
    const Eigen::Vector3d expected(10.5, 20.0 - 0.1, 30.0 - 0.2 * c);
    const Eigen::Vector3d predicted =
        sfm::predicted_antenna_position(tilted_pose, offset);
    EXPECT_LT((predicted - expected).norm(), 1e-12);
    EXPECT_LT((sfm::predicted_antenna_position(tilted_pose, Eigen::Vector3d::Zero()) -
               centre)
                  .norm(),
              1e-12);
}

TEST(Antenna, PersistRoundTrip) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("antenna");
    Sensor plain;
    plain.label = "plain";
    plain.width = 640;
    plain.height = 480;
    plain.calibration = Calibration::initial(CameraType::Frame, 640, 480, 500.0);
    plain.antenna.offset_m = Eigen::Vector3d(0.1, -0.2, 0.3);
    chunk.sensors().push_back(plain);

    Sensor bodied = plain;
    bodied.label = "bodied";
    bodied.antenna.boresight_deg = Eigen::Vector3d(1.0, 2.0, 3.0);
    chunk.sensors().push_back(bodied);

    ZipArchive archive;
    write_document(document, archive);
    const std::shared_ptr<Document> loaded =
        read_document(ZipArchive::parse(archive.serialise()));
    ASSERT_NE(loaded, nullptr);
    const Chunk& back = *loaded->workspace().chunks().front();
    ASSERT_EQ(back.sensors().size(), 2U);
    EXPECT_TRUE(back.sensors()[0] == plain);
    EXPECT_TRUE(back.sensors()[1] == bodied);
    EXPECT_FALSE(back.sensors()[0].antenna.boresight_deg.has_value());
    EXPECT_TRUE(back.sensors()[1].antenna.boresight_deg.has_value());
}

TEST(Antenna, NoAntennaKeepsLayout) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("plain");
    Sensor sensor;
    sensor.label = "plain";
    sensor.width = 640;
    sensor.height = 480;
    sensor.calibration = Calibration::initial(CameraType::Frame, 640, 480, 500.0);
    chunk.sensors().push_back(sensor);

    ZipArchive archive;
    write_document(document, archive);
    const std::string xml = ZipArchive::parse(archive.serialise()).get_text("doc.xml");
    EXPECT_EQ(xml.find("<antenna"), std::string::npos);
}

/// A ring of nadir-ish cameras observing a point grid, with a metric ENU local
/// frame (the chunk transform is ENU->ECEF at `origin`). Every GNSS reference
/// is the antenna position for the given camera-frame lever arm, so the truth
/// camera centres are known exactly.
struct AntennaScene {
    Document document;
    std::vector<Eigen::Vector3d> centres;
    geo::Geodetic origin;
};

AntennaScene make_antenna_scene(bool with_antenna, const Eigen::Vector3d& lever) {
    AntennaScene scene;
    Chunk& chunk = scene.document.workspace().add_chunk("antenna");
    scene.document.set_active_chunk_id(chunk.id());
    scene.origin = geo::Geodetic{7.0, 45.0, 100.0};

    const Calibration truth = Calibration::initial(CameraType::Frame, 1000, 800, 900.0);
    Sensor sensor;
    sensor.label = "synth";
    sensor.type = CameraType::Frame;
    sensor.width = truth.width;
    sensor.height = truth.height;
    sensor.calibration = truth;
    if (with_antenna) {
        sensor.antenna.offset_m = lever;
    }
    chunk.sensors().push_back(sensor);
    chunk.transform() = geo::enu_to_ecef(scene.origin);

    constexpr int kCameras = 8;
    std::vector<KeypointSet> sets(static_cast<std::size_t>(kCameras));
    for (int i = 0; i < kCameras; ++i) {
        const double azimuth = 2.0 * kPi * static_cast<double>(i) / kCameras;
        const Eigen::Vector3d centre(6.0 * std::cos(azimuth), 6.0 * std::sin(azimuth),
                                     (i % 2 == 0) ? 1.5 : -1.5);
        const Eigen::Vector3d z = -centre.normalized();
        const Eigen::Vector3d down(0.0, 0.0, -1.0);
        const Eigen::Vector3d y = (down - down.dot(z) * z).normalized();
        const Eigen::Vector3d x = y.cross(z).normalized();
        Eigen::Matrix3d rotation;
        rotation.row(0) = x.transpose();
        rotation.row(1) = y.transpose();
        rotation.row(2) = z.transpose();

        Camera camera;
        camera.id = i;
        camera.sensor_id = 0;
        camera.enabled = true;
        camera.pose = pose_with_centre(rotation, centre);
        // The reference measures the mounted antenna, not the centre.
        const Eigen::Vector3d antenna = centre + rotation.transpose() * lever;
        const geo::Geodetic g = geo::enu_to_geodetic(antenna, scene.origin);
        camera.reference.location =
            Eigen::Vector3d(g.longitude_deg, g.latitude_deg, g.height_m);
        camera.reference.accuracy = Eigen::Vector3d::Constant(0.02);
        camera.reference.enabled = true;
        chunk.cameras().push_back(camera);
        scene.centres.push_back(centre);
        sets[static_cast<std::size_t>(i)].camera_id = i;
    }
    for (int gx = -2; gx <= 2; ++gx) {
        for (int gy = -2; gy <= 2; ++gy) {
            for (int gz = -1; gz <= 1; ++gz) {
                const Eigen::Vector3d point(1.1 * gx, 1.1 * gy, 1.2 * gz);
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

/// Re-express a metric-ENU scene in a local frame whose unit is `1 / scale`
/// metres and whose axes are `r_s`-rotated. The chunk transform then carries a
/// similarity (scale `scale`), while the world geometry -- camera poses, tie
/// points and GNSS references -- is unchanged.
void reframe_in_scaled_local_frame(AntennaScene& scene, double scale,
                                   const Eigen::Matrix3d& r_s) {
    Chunk& chunk = *scene.document.active_chunk();
    Transform local_to_enu = Transform::Identity();
    local_to_enu.block<3, 3>(0, 0) = scale * r_s;
    chunk.transform() = geo::enu_to_ecef(scene.origin) * local_to_enu;

    for (Camera& camera : chunk.cameras()) {
        const Eigen::Vector3d centre = centre_of(camera.pose);
        const Eigen::Matrix3d rotation = camera.pose.block<3, 3>(0, 0) * r_s;
        camera.pose = pose_with_centre(rotation, r_s.transpose() * centre / scale);
    }
    for (Track& track : chunk.tie_points().tracks) {
        if (track.point.has_value()) {
            *track.point = r_s.transpose() * *track.point / scale;
        }
    }
}

double optimize_and_measure(AntennaScene scene) {
    Params params = make_params(builtin_registry().require("OptimizeCameras"));
    for (const char* name :
         {"f", "cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4"}) {
        params.set(std::string("fit_") + name, false);
    }
    params.set("camera_covariance", false);
    run_task(builtin_registry(), scene.document, "OptimizeCameras", params);
    const Chunk& chunk = *scene.document.active_chunk();
    double sum = 0.0;
    for (std::size_t i = 0; i < scene.centres.size(); ++i) {
        sum += (centre_of(chunk.cameras()[i].pose) - scene.centres[i]).norm();
    }
    return sum / static_cast<double>(scene.centres.size());
}

TEST(Antenna, OptimizeUsesAntennaPosition) {
    const Eigen::Vector3d lever(0.5, 0.0, 0.0);
    const double with = optimize_and_measure(make_antenna_scene(true, lever));
    const double control = optimize_and_measure(make_antenna_scene(false, lever));
    std::printf("[antenna] centre error: with antenna = %.4f m, control = %.4f m\n",
                with, control);
    EXPECT_LT(with, 0.01);
    EXPECT_GT(control, 0.10);
}

TEST(Antenna, GeoreferenceWithScaledChunkTransform) {
    // A 0.5 m lever along camera y: in this nadir scene it is an almost
    // constant world offset, so the first similarity fit is unbiased and the
    // truth comparison measures the lever-arm shift itself rather than the
    // rotation that a tangential lever would mimic.
    const Eigen::Vector3d lever(0.0, 0.5, 0.0);
    AntennaScene scene = make_antenna_scene(true, lever);
    Chunk& chunk = *scene.document.active_chunk();

    // True ENU projection centres, captured before the frame is rescaled.
    std::vector<Eigen::Vector3d> truth;
    for (const Camera& camera : chunk.cameras()) {
        truth.push_back(centre_of(camera.pose));
    }

    // Local units are 2.5 m and the local axes are rotated, so the chunk
    // transform is a similarity rather than a rigid transform.
    const Eigen::Matrix3d r_s =
        Eigen::AngleAxisd(0.7, Eigen::Vector3d(0.2, -0.3, 0.9).normalized())
            .toRotationMatrix();
    reframe_in_scaled_local_frame(scene, 2.5, r_s);

    // The georeferenced origin is the mean ECEF of the antenna references.
    Eigen::Vector3d mean_ecef = Eigen::Vector3d::Zero();
    for (const Camera& camera : chunk.cameras()) {
        const Eigen::Vector3d& loc = *camera.reference.location;
        mean_ecef += geo::geodetic_to_ecef(geo::Geodetic{loc.x(), loc.y(), loc.z()});
    }
    mean_ecef /= static_cast<double>(chunk.cameras().size());
    const Transform to_new =
        geo::ecef_to_enu(geo::ecef_to_geodetic(mean_ecef)) * geo::enu_to_ecef(scene.origin);

    const sfm::GeoreferenceReport report = sfm::georeference_chunk(chunk);
    ASSERT_TRUE(report.applied) << report.reason;

    double worst = 0.0;
    for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
        const Eigen::Vector3d expected = (to_new * truth[i].homogeneous()).head<3>();
        const Eigen::Vector3d actual = centre_of(chunk.cameras()[i].pose);
        worst = std::max(worst, (actual - expected).norm());
    }
    std::printf("[antenna] scaled-chunk centre error: %.6f m\n", worst);
    EXPECT_LT(worst, 0.01);
}

TEST(Antenna, GeoreferenceSidewaysLeverArm) {
    // A 0.5 m lever along camera x is tangential to this ring of nadir cameras:
    // the first similarity fit's rotation is biased by the uncorrected lever
    // arm, so a single correction leaves centimetres of error. The iterated fit
    // must converge to the truth well inside a millimetre.
    const Eigen::Vector3d lever(0.5, 0.0, 0.0);
    const auto measure = [&lever](bool scaled) {
        AntennaScene scene = make_antenna_scene(true, lever);
        Chunk& chunk = *scene.document.active_chunk();
        std::vector<Eigen::Vector3d> truth;
        for (const Camera& camera : chunk.cameras()) {
            truth.push_back(centre_of(camera.pose));
        }
        if (scaled) {
            const Eigen::Matrix3d r_s =
                Eigen::AngleAxisd(0.7, Eigen::Vector3d(0.2, -0.3, 0.9).normalized())
                    .toRotationMatrix();
            reframe_in_scaled_local_frame(scene, 2.5, r_s);
        }
        Eigen::Vector3d mean_ecef = Eigen::Vector3d::Zero();
        for (const Camera& camera : chunk.cameras()) {
            const Eigen::Vector3d& loc = *camera.reference.location;
            mean_ecef +=
                geo::geodetic_to_ecef(geo::Geodetic{loc.x(), loc.y(), loc.z()});
        }
        mean_ecef /= static_cast<double>(chunk.cameras().size());
        const Transform to_new = geo::ecef_to_enu(geo::ecef_to_geodetic(mean_ecef)) *
                                 geo::enu_to_ecef(scene.origin);
        const sfm::GeoreferenceReport report = sfm::georeference_chunk(chunk);
        EXPECT_TRUE(report.applied) << report.reason;
        double worst = 0.0;
        for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
            const Eigen::Vector3d expected =
                (to_new * truth[i].homogeneous()).head<3>();
            worst = std::max(worst,
                             (centre_of(chunk.cameras()[i].pose) - expected).norm());
        }
        return worst;
    };
    const double metric = measure(false);
    std::printf("[antenna] sideways-lever centre error: %.6f m\n", metric);
    EXPECT_LT(metric, 0.001);
    const double scaled = measure(true);
    std::printf("[antenna] sideways-lever scaled-chunk centre error: %.6f m\n", scaled);
    EXPECT_LT(scaled, 0.001);
}

TEST(Commands, SetSensorAntennaUndoes) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("chunk");
    document.set_active_chunk_id(chunk.id());
    const std::int32_t chunk_id = chunk.id();
    Sensor sensor;
    sensor.label = "s0";
    sensor.width = 640;
    sensor.height = 480;
    sensor.calibration = Calibration::initial(CameraType::Frame, 640, 480, 500.0);
    chunk.sensors().push_back(sensor);
    UndoStack stack(&document);

    Antenna antenna;
    antenna.offset_m = Eigen::Vector3d(0.4, -0.1, 0.2);
    antenna.boresight_deg = Eigen::Vector3d(0.0, 1.0, 2.0);
    stack.execute(make_set_sensor_antenna_command(chunk_id, 0, antenna));
    EXPECT_TRUE(chunk.sensors()[0].antenna == antenna);
    stack.undo();
    EXPECT_TRUE(chunk.sensors()[0].antenna == Antenna{});

    EXPECT_THROW(
        make_set_sensor_antenna_command(chunk_id, 3, antenna)->apply(document),
        ValueError);
    Antenna bad;
    bad.offset_m = Eigen::Vector3d(std::nan(""), 0.0, 0.0);
    EXPECT_THROW(make_set_sensor_antenna_command(chunk_id, 0, bad)->apply(document),
                 ValueError);
}

}  // namespace
}  // namespace alphashape
