// Tests for the film-normalised sensor invariant (WP-13.18b) and the stages
// that must consume it: the set-sensor-film command, CalibrateFiducials,
// MatchPhotos' epipolar geometry, OptimizeCameras' marker seeding, the
// depth-map fingerprint and doc.xml load normalisation.
//
// The scene is synthetic: 3-D points seen by four known poses, projected with
// the NORMALISED calibration. Keypoints are injected in normalised pixels, so
// nothing depends on image content.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "common/algorithm/calibration/calibrate_fiducials_task.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/core/sensor_film.hpp"
#include "common/tasks/registry.hpp"
#include "image/image.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

constexpr int kWidth = 2000;
constexpr int kHeight = 2000;
constexpr double kFocalNormalised = 1000.0;
constexpr double kFocalRaw = 2000.0;  // the same lens in raw scan pixels (2x)
constexpr double kResolution = 0.02;   // mm per normalised pixel
constexpr double kScanPxPerMm = 200.0; // scan pixels per millimetre (2x frame)
constexpr double kHalfExtentMm = 20.0; // 40 mm frame -> 2000 normalised pixels

/// Camera pose X_cam = R X_world + t, looking at `target` with world +Z up.
Transform look_at(const Eigen::Vector3d& centre, const Eigen::Vector3d& target) {
    const Eigen::Vector3d z = (target - centre).normalized();
    const Eigen::Vector3d up(0.0, 0.0, 1.0);
    const Eigen::Vector3d x = up.cross(z).normalized();
    const Eigen::Vector3d y = z.cross(x);
    // The Transform's rotation is world -> camera, so its ROWS are the camera
    // axes expressed in world coordinates (X_cam = R (X_world - C)).
    Transform pose = Transform::Identity();
    pose.block<1, 3>(0, 0) = x.transpose();
    pose.block<1, 3>(1, 0) = y.transpose();
    pose.block<1, 3>(2, 0) = z.transpose();
    pose.block<3, 1>(0, 3) = -pose.block<3, 3>(0, 0) * centre;
    return pose;
}

/// A genuine 3-D cloud (not a height field) so the essential-matrix solve is
/// never fed a coplanar sample.
std::vector<Eigen::Vector3d> scene_points() {
    std::vector<Eigen::Vector3d> points;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            for (int k = -1; k <= 1; ++k) {
                points.emplace_back(0.9 * static_cast<double>(i),
                                    0.9 * static_cast<double>(j),
                                    0.9 * static_cast<double>(k));
            }
        }
    }
    return points;
}

std::vector<Transform> scene_poses() {
    std::vector<Transform> poses;
    for (int k = 0; k < 4; ++k) {
        const double angle =
            (45.0 + 90.0 * static_cast<double>(k)) * std::numbers::pi_v<double> / 180.0;
        const Eigen::Vector3d centre(7.0 * std::cos(angle), 7.0 * std::sin(angle), 6.0);
        poses.push_back(look_at(centre, Eigen::Vector3d::Zero()));
    }
    return poses;
}

Calibration normalised_calibration() {
    Calibration calibration =
        Calibration::initial(CameraType::Frame, kWidth, kHeight, kFocalNormalised);
    return calibration;
}

FilmSettings complete_film() {
    FilmSettings film;
    film.enabled = true;
    film.resolution_mm_per_px = kResolution;
    film.film_size_mm = Eigen::Vector2d(2.0 * kHalfExtentMm, 2.0 * kHalfExtentMm);
    film.calibration_normalised = true;
    return film;
}

/// A film -> scan transform at kScanPxPerMm, centred on a scan of `size`.
Eigen::Matrix3d scan_transform(int size) {
    Eigen::Matrix3d transform = Eigen::Matrix3d::Identity();
    transform(0, 0) = kScanPxPerMm;
    transform(1, 1) = -kScanPxPerMm;  // film y is up, scan y is down
    transform(0, 2) = 0.5 * static_cast<double>(size);
    transform(1, 2) = 0.5 * static_cast<double>(size);
    return transform;
}

/// One film camera whose keypoints are the scene projected with the
/// NORMALISED calibration. The camera is attached to the SLAVE sensor, whose
/// own calibration is deliberately the raw one: only an effective-sensor
/// lookup yields the pixel space the keypoints are in.
void add_scene_camera(Chunk& chunk, std::int32_t camera_id,
                      const Transform& pose, const std::vector<Eigen::Vector3d>& points,
                      const Calibration& projection) {
    Camera camera;
    camera.id = camera_id;
    camera.label = "scan " + std::to_string(camera_id);
    camera.sensor_id = 1;
    camera.enabled = true;
    camera.pose = pose;
    chunk.cameras().push_back(camera);

    KeypointSet set;
    set.camera_id = camera_id;
    for (std::size_t p = 0; p < points.size(); ++p) {
        const Eigen::Vector3d camera_point =
            pose.block<3, 3>(0, 0) * points[p] + pose.block<3, 1>(0, 3);
        const Eigen::Vector2d pixel = projection.project(camera_point);
        Keypoint keypoint;
        keypoint.x = static_cast<float>(pixel.x() - 0.5);
        keypoint.y = static_cast<float>(pixel.y() - 0.5);
        set.keypoints.push_back(keypoint);
        Descriptor descriptor;
        for (std::size_t b = 0; b < Descriptor::kBytes; ++b) {
            descriptor.bytes[b] = static_cast<std::uint8_t>(
                (static_cast<std::size_t>(31) * p + static_cast<std::size_t>(17) * b) %
                251U);
        }
        set.descriptors.push_back(descriptor);
    }
    chunk.tie_points().keypoint_sets.push_back(std::move(set));

    auto frame = std::make_shared<Frame>("/nonexistent/film-scan.png",
                                         projection.width, projection.height);
    frame->set_id(chunk.allocate_id());
    frame->set_camera_id(camera_id);
    chunk.frames().push_back(frame);
}

/// Master sensor 0 carries the normalised calibration; slave sensor 1 carries
/// the RAW one. Cameras sit on the slave, so every stage must resolve the
/// master to work in the frame the keypoints are in.
Chunk& build_film_scene(Document& document) {
    Chunk& chunk = document.workspace().add_chunk("film");
    document.set_active_chunk_id(chunk.id());
    Sensor master;
    master.label = "master";
    master.width = kWidth;
    master.height = kHeight;
    master.calibration = normalised_calibration();
    master.film = complete_film();
    chunk.sensors().push_back(master);
    Sensor slave;
    slave.label = "slave";
    slave.width = kWidth;
    slave.height = kHeight;
    slave.calibration =
        Calibration::initial(CameraType::Frame, kWidth, kHeight, kFocalRaw);
    slave.calibration.cx = 25.0;
    slave.master_sensor_id = 0;
    chunk.sensors().push_back(slave);

    const std::vector<Eigen::Vector3d> points = scene_points();
    const std::vector<Transform> poses = scene_poses();
    const Calibration projection = normalised_calibration();
    for (std::size_t k = 0; k < poses.size(); ++k) {
        add_scene_camera(chunk, static_cast<std::int32_t>(k), poses[k], points,
                         projection);
    }
    return chunk;
}

TEST(FilmSensorInvariant, SetSensorFilmNormalisesAndUndoes) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("film");
    document.set_active_chunk_id(chunk.id());
    Sensor sensor;
    sensor.label = "film-camera";
    sensor.width = 4000;   // raw scan pixels
    sensor.height = 3000;
    sensor.calibration =
        Calibration::initial(CameraType::Frame, 4000, 3000, 4000.0);
    sensor.calibration.cx = 40.0;
    sensor.calibration.cy = -20.0;
    sensor.calibration.b1 = 3.0;
    sensor.calibration.b2 = -2.0;
    sensor.calibration.k1 = 0.01;  // dimensionless: must NOT change
    chunk.sensors().push_back(sensor);
    Camera camera;
    camera.id = chunk.allocate_id();
    camera.sensor_id = 0;
    camera.film_transform = scan_transform(4000);
    chunk.cameras().push_back(camera);
    const Sensor before = chunk.sensors()[0];

    FilmSettings film;
    film.enabled = true;
    film.resolution_mm_per_px = kResolution;
    film.film_size_mm = Eigen::Vector2d(40.0, 40.0);
    auto command = make_set_sensor_film_command(chunk.id(), 0, film);
    command->apply(document);

    const Sensor& after = chunk.sensors()[0];
    // A normalised pixel is 0.02 mm; a scan pixel is 1/200 mm, so s = 1/4.
    EXPECT_EQ(after.width, 2000);
    EXPECT_EQ(after.height, 2000);
    EXPECT_EQ(after.calibration.width, 2000);
    EXPECT_EQ(after.calibration.height, 2000);
    EXPECT_NEAR(after.calibration.f, 1000.0, 1e-9);
    EXPECT_NEAR(after.calibration.cx, 10.0, 1e-12);
    EXPECT_NEAR(after.calibration.cy, -5.0, 1e-12);
    EXPECT_NEAR(after.calibration.b1, 0.75, 1e-12);
    EXPECT_NEAR(after.calibration.b2, -0.5, 1e-12);
    EXPECT_DOUBLE_EQ(after.calibration.k1, 0.01);
    EXPECT_TRUE(after.film.calibration_normalised);

    // A later layout change keeps the already-normalised coefficients.
    film.film_size_mm = Eigen::Vector2d(50.0, 50.0);
    make_set_sensor_film_command(chunk.id(), 0, film)->apply(document);
    EXPECT_NEAR(chunk.sensors()[0].calibration.f, 1000.0, 1e-9);
    EXPECT_EQ(chunk.sensors()[0].width, 2500);
    command->revert(document);
    EXPECT_TRUE(chunk.sensors()[0] == before);
}

TEST(FilmSensorInvariant, CalibrateFiducialsStoresResolution) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("film");
    document.set_active_chunk_id(chunk.id());
    Sensor sensor;
    sensor.label = "film-camera";
    sensor.width = 8000;   // raw scan pixels
    sensor.height = 6000;
    sensor.calibration =
        Calibration::initial(CameraType::Frame, sensor.width, sensor.height, 4000.0);
    sensor.film.enabled = true;
    for (int i = 0; i < 4; ++i) {
        Fiducial fiducial;
        fiducial.label = "m" + std::to_string(i);
        fiducial.film_mm = Eigen::Vector2d((i == 0 || i == 3) ? -10.0 : 10.0,
                                           (i < 2) ? 10.0 : -10.0);
        sensor.film.fiducials.push_back(fiducial);
    }
    chunk.sensors().push_back(sensor);
    Camera camera;
    camera.id = chunk.allocate_id();
    camera.sensor_id = 0;
    for (const Fiducial& fiducial : chunk.sensors()[0].film.fiducials) {
        // 200 scan pixels per millimetre around the scan centre.
        camera.fiducial_pixels.emplace_back(
            Eigen::Vector2d(2000.0 + kScanPxPerMm * fiducial.film_mm.x(),
                            1500.0 - kScanPxPerMm * fiducial.film_mm.y()));
    }
    chunk.cameras().push_back(camera);

    Params params = make_params(builtin_registry().require("CalibrateFiducials"));
    params.set("resolution", 0.01);
    run_task(builtin_registry(), document, "CalibrateFiducials", params);

    const Sensor& after = chunk.sensors()[0];
    EXPECT_DOUBLE_EQ(after.film.resolution_mm_per_px, 0.01);
    // The pixels are 200 per millimetre and the marks are measured 20 mm
    // either side of centre, so the derived extent is 40 mm:
    // 40 mm / 0.01 = 4000 normalised pixels and s = 1 / (0.01 * 200) = 0.5.
    EXPECT_EQ(after.width, 4000);
    EXPECT_EQ(after.height, 4000);
    EXPECT_EQ(after.calibration.width, 4000);
    EXPECT_EQ(after.calibration.height, 4000);
    EXPECT_NEAR(after.calibration.f, 2000.0, 1e-9);
    EXPECT_TRUE(after.film.calibration_normalised);
}

TEST(FilmSensorInvariant, DepthFingerprintTracksFilmGeometry) {
    Camera camera;
    camera.id = 3;
    camera.sensor_id = 0;
    Sensor sensor;
    sensor.width = kWidth;
    sensor.height = kHeight;
    sensor.calibration = normalised_calibration();
    const std::uint64_t digital = depth_map_fingerprint(camera, sensor, 2);
    camera.film_transform = scan_transform(4000);
    const std::uint64_t with_transform = depth_map_fingerprint(camera, sensor, 2);
    EXPECT_NE(digital, with_transform);
    sensor.film = complete_film();
    const std::uint64_t with_settings = depth_map_fingerprint(camera, sensor, 2);
    EXPECT_NE(with_transform, with_settings);
    // Re-fitting the same geometry is stable.
    EXPECT_EQ(with_settings, depth_map_fingerprint(camera, sensor, 2));
}

TEST(FilmPipeline, MatchPhotosUsesNormalisedCalibration) {
    Document document;
    Chunk& chunk = build_film_scene(document);
    const std::size_t points = scene_points().size();

    Params params = make_params(builtin_registry().require("MatchPhotos"));
    params.set("reference_preselection", false);
    params.set("generic_preselection", false);
    run_task(builtin_registry(), document, "MatchPhotos", params);

    // Four cameras, exhaustive pairs. Every injected correspondence is a
    // perfect inlier under the NORMALISED bearings; under the slave's raw
    // (2x focal) calibration the epipolar geometry would reject most of them.
    const std::vector<PairMatches>& matches = chunk.tie_points().matches;
    ASSERT_EQ(matches.size(), 6U);
    for (const PairMatches& pair : matches) {
        EXPECT_EQ(pair.matches.size(), points) << "pair " << pair.camera_a << "-"
                                               << pair.camera_b;
    }
}

TEST(FilmPipeline, OptimizeMarkerSeedUsesNormalisedCalibration) {
    Document document;
    Chunk& chunk = build_film_scene(document);
    const std::vector<Eigen::Vector3d> points = scene_points();
    const std::vector<Transform> poses = scene_poses();
    const Calibration projection = normalised_calibration();

    // A full track set so the adjustment has a well-posed camera geometry.
    for (std::size_t p = 0; p < points.size(); ++p) {
        Track track;
        track.point = points[p];
        for (std::size_t k = 0; k < poses.size(); ++k) {
            track.observations.push_back(
                Observation{static_cast<std::int32_t>(k), static_cast<std::uint32_t>(p)});
        }
        chunk.tie_points().tracks.push_back(track);
    }
    // One regular marker observed by two cameras: its position comes straight
    // from the marker-seeding triangulation, not from a stored seed.
    const Eigen::Vector3d truth = points[12];
    Marker marker;
    marker.id = chunk.allocate_id();
    marker.label = "seed-check";
    marker.type = MarkerType::Regular;
    for (std::int32_t camera_id = 1; camera_id <= 2; ++camera_id) {
        const Transform& pose = poses[static_cast<std::size_t>(camera_id)];
        const Eigen::Vector3d camera_point =
            pose.block<3, 3>(0, 0) * truth + pose.block<3, 1>(0, 3);
        marker.projections.push_back(
            MarkerProjection{camera_id, projection.project(camera_point), true});
    }
    chunk.markers().push_back(marker);

    Params params = make_params(builtin_registry().require("OptimizeCameras"));
    for (const char* name : {"fit_f", "fit_cx", "fit_cy", "fit_k1", "fit_k2", "fit_k3",
                             "fit_p1", "fit_p2"}) {
        params.set(name, false);
    }
    params.set("camera_covariance", false);
    params.set("use_orientation", false);
    run_task(builtin_registry(), document, "OptimizeCameras", params);

    ASSERT_TRUE(chunk.markers()[0].position.has_value());
    EXPECT_LT((*chunk.markers()[0].position - truth).norm(), 0.05);
}

}  // namespace
}  // namespace alphashape
