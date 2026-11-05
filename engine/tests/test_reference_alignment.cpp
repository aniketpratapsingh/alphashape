// test_reference_alignment.cpp -- exterior orientation from navigation
// metadata and tie-point triangulation with known poses (WP-13.21b).
//
// The reference fixture is built in a metric ENU frame: every camera has a true
// world->camera rotation and centre, from which the antenna-phase-centre ECEF
// position and the stored body attitude are synthesised. The task must invert
// that construction, so the truth is independent of the code under test.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "common/algorithm/feature/detect_features_task.hpp"  // keypoint_to_pixel
#include "common/algorithm/reference/convert_reference.hpp"
#include "common/algorithm/reference/reference_orientation.hpp"
#include "common/algorithm/sfm/absolute_pose.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/document.hpp"
#include "common/tasks/registry.hpp"
#include "model/crs/euler.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr crs::RotationOrder kOrder = crs::RotationOrder::ZYX;

Transform pose_with_centre(const Eigen::Matrix3d& r, const Eigen::Vector3d& centre) {
    Transform pose = Transform::Identity();
    pose.block<3, 3>(0, 0) = r;
    pose.block<3, 1>(0, 3) = -r * centre;
    return pose;
}

Eigen::Matrix3d rz90() {
    Eigen::Matrix3d m;
    m << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    return m;
}

Eigen::Matrix3d yaw_about_up(double radians) {
    Eigen::Matrix3d m;
    m << std::cos(radians), -std::sin(radians), 0.0,  //
        std::sin(radians), std::cos(radians), 0.0,     //
        0.0, 0.0, 1.0;
    return m;
}

Eigen::Vector3d stored_from_world_body(const Eigen::Matrix3d& r_wb) {
    const crs::EulerAngles a = crs::from_matrix(rz90().transpose() * r_wb, kOrder);
    return Eigen::Vector3d(a.yaw, -a.pitch, a.roll);
}

double rotation_angle(const Eigen::Matrix3d& a, const Eigen::Matrix3d& b) {
    return Eigen::AngleAxisd(a * b.transpose()).angle();
}

struct Scene {
    Document document;
    Chunk* chunk = nullptr;
    geo::Geodetic origin{7.0, 45.0, 100.0};
    Eigen::Matrix4d frame = Eigen::Matrix4d::Identity();  ///< internal->ECEF
    Eigen::Matrix4d reference_frame = Eigen::Matrix4d::Identity();  ///< ENU->ECEF
    Eigen::Matrix3d mount = reference::body_to_camera_nadir();
    std::vector<Eigen::Vector3d> centres;   ///< truth, chunk-internal
    std::vector<Eigen::Vector3d> antennas;  ///< truth antenna position, ECEF
    std::vector<Eigen::Matrix3d> rotations;  ///< truth world(ECEF)->camera
};

/// `lever` is the antenna offset in the camera frame (metres). With
/// `georeferenced` the chunk already carries ENU->ECEF and poses are
/// feel-invariant; without it the truth poses live in local ENU metres and the
/// task must build the frame itself.
Scene make_scene(std::size_t count, const Eigen::Vector3d& lever, bool georeferenced,
                 const std::vector<bool>& has_orientation) {
    Scene scene;
    Chunk& chunk = scene.document.workspace().add_chunk("reference");
    scene.document.set_active_chunk_id(chunk.id());
    scene.chunk = &chunk;
    scene.reference_frame = geo::enu_to_ecef(scene.origin);
    scene.frame = georeferenced ? scene.reference_frame : Eigen::Matrix4d::Identity();
    chunk.transform() = scene.frame;
    chunk.meta().set(reference::kRotationOrderKey, crs::rotation_order_name(kOrder));

    Sensor sensor;
    sensor.label = "synth";
    sensor.type = CameraType::Frame;
    sensor.width = 1200;
    sensor.height = 900;
    sensor.calibration = Calibration::initial(CameraType::Frame, 1200, 900, 1000.0);
    sensor.antenna.offset_m = lever;
    chunk.sensors().push_back(sensor);

    const Eigen::Matrix3d e_reference = scene.reference_frame.block<3, 3>(0, 0);
    for (std::size_t i = 0; i < count; ++i) {
        const double azimuth = 2.0 * kPi * static_cast<double>(i) /
                               static_cast<double>(count);
        const Eigen::Vector3d centre(2.0 * std::cos(azimuth), 2.0 * std::sin(azimuth),
                                     10.0 + 0.5 * static_cast<double>(i % 3));
        const Eigen::Matrix3d body = yaw_about_up(azimuth);
        // The pose rotation is chunk-internal; the body attitude is the same
        // rotation expressed through the physical ENU frame at the antenna.
        const Eigen::Matrix3d rotation = scene.mount * body.transpose();
        const Eigen::Vector3d centre_ecef =
            (scene.reference_frame * centre.homogeneous()).head<3>();
        const Eigen::Vector3d antenna =
            centre_ecef + e_reference * rotation.transpose() * lever;
        const geo::Geodetic g = geo::ecef_to_geodetic(antenna);
        const Eigen::Matrix3d enu_antenna =
            geo::enu_to_ecef(g).block<3, 3>(0, 0);
        const Eigen::Matrix3d r_wb = enu_antenna.transpose() * e_reference * body;

        Camera camera;
        camera.id = static_cast<std::int32_t>(i);
        camera.sensor_id = 0;
        camera.enabled = true;
        camera.pose = pose_with_centre(rotation, centre);
        camera.reference.location =
            Eigen::Vector3d(g.longitude_deg, g.latitude_deg, g.height_m);
        camera.reference.accuracy = Eigen::Vector3d::Constant(0.02);
        camera.reference.enabled = true;
        if (i >= has_orientation.size() || has_orientation[i]) {
            camera.reference.orientation_deg = stored_from_world_body(r_wb);
        }
        chunk.cameras().push_back(camera);
        scene.centres.push_back(centre);
        scene.antennas.push_back(antenna);
        scene.rotations.push_back(rotation);
    }
    return scene;
}

void run_task_named(Document& document, const std::string& name,
                    const std::function<void(Params&)>& bind = {}) {
    Params params = make_params(builtin_registry().require(name));
    if (bind) {
        bind(params);
    }
    run_task(builtin_registry(), document, name, params);
}

TEST(ReferenceAlignment, PlacesCamerasAtTruth) {
    const Eigen::Vector3d lever(0.30, -0.12, 0.20);
    Scene scene = make_scene(6, lever, true, {});
    run_task_named(scene.document, "AlignFromReference");

    Chunk& chunk = *scene.chunk;
    ASSERT_EQ(chunk.cameras().size(), scene.centres.size());
    double max_centre_error = 0.0;
    double max_rotation_error = 0.0;
    for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
        const Transform& pose = chunk.cameras()[i].pose;
        const Eigen::Matrix3d r = pose.block<3, 3>(0, 0);
        const Eigen::Vector3d centre = -(r.transpose() * pose.block<3, 1>(0, 3));
        max_centre_error = std::max(max_centre_error,
                                    (centre - scene.centres[i]).norm());
        max_rotation_error =
            std::max(max_rotation_error, rotation_angle(r, scene.rotations[i]));
    }
    EXPECT_LE(max_centre_error, 1e-6) << "centre error (m)";
    EXPECT_LE(max_rotation_error, 1e-9) << "rotation error (rad)";
    const std::string* placed = chunk.meta().find("align_from_reference.placed");
    ASSERT_NE(placed, nullptr);
    EXPECT_EQ(*placed, "6");
    std::printf("[RefAlign] truth placement max centre %.3e m, max rotation %.3e rad\n",
                max_centre_error, max_rotation_error);
}

TEST(ReferenceAlignment, BuildsLocalFrameWhenUngeoreferenced) {
    const Eigen::Vector3d lever(0.10, 0.05, -0.08);
    Scene scene = make_scene(5, lever, false, {});

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d& antenna : scene.antennas) {
        mean += antenna;
    }
    mean /= static_cast<double>(scene.antennas.size());
    const Eigen::Matrix4d expected = geo::enu_to_ecef(geo::ecef_to_geodetic(mean));

    run_task_named(scene.document, "AlignFromReference");
    Chunk& chunk = *scene.chunk;
    EXPECT_LE((chunk.transform() - expected).cwiseAbs().maxCoeff(), 1e-6);
    ASSERT_TRUE(chunk.crs().is_set());
    EXPECT_EQ(chunk.crs().epsg, 4326);

    const Eigen::Matrix4d inverse = expected.inverse();
    double max_error = 0.0;
    for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
        const Transform& pose = chunk.cameras()[i].pose;
        const Eigen::Matrix3d r = pose.block<3, 3>(0, 0);
        const Eigen::Vector3d centre = -(r.transpose() * pose.block<3, 1>(0, 3));
        const Eigen::Vector3d truth =
            (inverse * (scene.reference_frame * scene.centres[i].homogeneous()))
                .head<3>();
        max_error = std::max(max_error, (centre - truth).norm());
    }
    EXPECT_LE(max_error, 1e-6);
    std::printf("[RefAlign] local frame centre error %.3e m\n", max_error);
}

TEST(ReferenceAlignment, SkipsCamerasWithoutOrientation) {
    Scene scene = make_scene(4, Eigen::Vector3d(0.1, 0.0, 0.2), true,
                             {true, false, true, false});
    const std::vector<Transform> before = {scene.chunk->cameras()[0].pose,
                                           scene.chunk->cameras()[1].pose,
                                           scene.chunk->cameras()[2].pose,
                                           scene.chunk->cameras()[3].pose};
    run_task_named(scene.document, "AlignFromReference");
    Chunk& chunk = *scene.chunk;
    ASSERT_NE(chunk.meta().find("align_from_reference.placed"), nullptr);
    EXPECT_EQ(*chunk.meta().find("align_from_reference.placed"), "2");
    ASSERT_NE(chunk.meta().find("align_from_reference.skipped"), nullptr);
    EXPECT_EQ(*chunk.meta().find("align_from_reference.skipped"), "2");
    // A skipped camera keeps its pose unless reset_unreferenced is set.
    EXPECT_EQ(chunk.cameras()[1].pose, before[1]);
    EXPECT_EQ(chunk.cameras()[3].pose, before[3]);
    EXPECT_TRUE(chunk.cameras()[1].enabled);

    run_task_named(scene.document, "AlignFromReference",
                   [](Params& p) { p.set("reset_unreferenced", true); });
    EXPECT_FALSE(chunk.cameras()[1].enabled);
    EXPECT_EQ(chunk.cameras()[1].pose, Transform::Identity());
}

TEST(ReferenceAlignment, FailsWhenNothingPlaced) {
    Scene scene = make_scene(3, Eigen::Vector3d(0.1, 0.0, 0.1), true,
                             {false, false, false});
    const ChunkPoseState before = capture_chunk_pose_state(*scene.chunk);
    EXPECT_THROW(run_task_named(scene.document, "AlignFromReference"), Error);
    const ChunkPoseState after = capture_chunk_pose_state(*scene.chunk);
    EXPECT_EQ(after.transform, before.transform);
    EXPECT_EQ(after.meta, before.meta);
    ASSERT_EQ(after.cameras.size(), before.cameras.size());
    for (std::size_t i = 0; i < before.cameras.size(); ++i) {
        EXPECT_EQ(after.cameras[i].pose, before.cameras[i].pose);
    }
}

/// A nadir block looking at a 4x4 ground grid, with matches synthesised from
/// the projections. Every 20th observation is moved 30 px: those are the
/// outliers the task must drop.
struct TriScene {
    Document document;
    Chunk* chunk = nullptr;
    std::vector<Eigen::Vector3d> points;
    std::size_t outliers = 0;
};

TriScene make_tri_scene() {
    TriScene scene;
    Chunk& chunk = scene.document.workspace().add_chunk("tri");
    scene.document.set_active_chunk_id(chunk.id());
    scene.chunk = &chunk;
    Sensor sensor;
    sensor.type = CameraType::Frame;
    sensor.width = 1200;
    sensor.height = 900;
    sensor.calibration = Calibration::initial(CameraType::Frame, 1200, 900, 1000.0);
    chunk.sensors().push_back(sensor);
    const Eigen::Vector3d lever(0.25, -0.10, 0.15);
    chunk.sensors()[0].antenna.offset_m = lever;
    const Calibration& truth = chunk.sensors()[0].calibration;
    const Eigen::Matrix3d mount = reference::body_to_camera_nadir();
    const geo::Geodetic origin{7.0, 45.0, 100.0};
    const Eigen::Matrix4d reference_frame = geo::enu_to_ecef(origin);
    const Eigen::Matrix3d e_reference = reference_frame.block<3, 3>(0, 0);
    chunk.transform() = reference_frame;
    chunk.meta().set(reference::kRotationOrderKey, crs::rotation_order_name(kOrder));

    constexpr int kCameras = 6;
    for (int i = 0; i < kCameras; ++i) {
        const double azimuth =
            2.0 * kPi * static_cast<double>(i) / static_cast<double>(kCameras);
        const Eigen::Vector3d centre(3.0 * std::cos(azimuth),
                                     3.0 * std::sin(azimuth), 8.0);
        const Eigen::Matrix3d body = yaw_about_up(azimuth);
        const Eigen::Matrix3d rotation = mount * body.transpose();
        const Eigen::Vector3d centre_ecef =
            (reference_frame * centre.homogeneous()).head<3>();
        const Eigen::Vector3d antenna =
            centre_ecef + e_reference * rotation.transpose() * lever;
        const geo::Geodetic g = geo::ecef_to_geodetic(antenna);
        const Eigen::Matrix3d enu_antenna = geo::enu_to_ecef(g).block<3, 3>(0, 0);
        const Eigen::Matrix3d r_wb = enu_antenna.transpose() * e_reference * body;

        Camera camera;
        camera.id = i;
        camera.sensor_id = 0;
        camera.enabled = true;
        camera.pose = pose_with_centre(rotation, centre);
        camera.reference.location =
            Eigen::Vector3d(g.longitude_deg, g.latitude_deg, g.height_m);
        camera.reference.accuracy = Eigen::Vector3d::Constant(0.02);
        camera.reference.enabled = true;
        camera.reference.orientation_deg = stored_from_world_body(r_wb);
        chunk.cameras().push_back(camera);
    }

    std::vector<std::vector<int>> index;
    for (int gx = 0; gx < 4; ++gx) {
        for (int gy = 0; gy < 4; ++gy) {
            scene.points.emplace_back(0.4 * static_cast<double>(gx) - 0.6,
                                      0.4 * static_cast<double>(gy) - 0.6, 0.0);
            index.emplace_back(static_cast<std::size_t>(kCameras), -1);
        }
    }

    std::mt19937 rng(1234567u);
    std::size_t observations = 0;
    std::vector<KeypointSet> sets(static_cast<std::size_t>(kCameras));
    for (int c = 0; c < kCameras; ++c) {
        KeypointSet& set = sets[static_cast<std::size_t>(c)];
        set.camera_id = c;
        const Transform& pose = chunk.cameras()[static_cast<std::size_t>(c)].pose;
        for (std::size_t p = 0; p < scene.points.size(); ++p) {
            const Eigen::Vector3d p_cam =
                pose.block<3, 3>(0, 0) * scene.points[p] + pose.block<3, 1>(0, 3);
            if (!(p_cam.z() > 1.0)) {
                continue;
            }
            Eigen::Vector2d pixel = truth.project(p_cam);
            if (pixel.x() < 4.0 || pixel.x() > 1196.0 || pixel.y() < 4.0 ||
                pixel.y() > 896.0) {
                continue;
            }
            const double noise_x = static_cast<double>(rng() % 3U) - 1.0;
            const double noise_y = static_cast<double>(rng() % 3U) - 1.0;
            pixel += Eigen::Vector2d(noise_x, noise_y);
            if (observations % 20U == 19U) {
                pixel.x() += 250.0;  // gross, so any reasonable gate drops it
                ++scene.outliers;
            }
            ++observations;
            Keypoint keypoint;
            keypoint.x = static_cast<float>(pixel.x() - 0.5);
            keypoint.y = static_cast<float>(pixel.y() - 0.5);
            index[p][static_cast<std::size_t>(c)] =
                static_cast<int>(set.keypoints.size());
            set.keypoints.push_back(keypoint);
        }
        chunk.tie_points().keypoint_sets.push_back(std::move(set));
    }

    // Keep only points seen in four or more cameras: with that redundancy a
    // single bad observation can be identified and dropped without losing the
    // point, which is what the outlier test measures.
    std::vector<Eigen::Vector3d> kept_points;
    std::vector<std::vector<int>> kept_index;
    for (std::size_t p = 0; p < scene.points.size(); ++p) {
        const std::size_t visible =
            static_cast<std::size_t>(std::count_if(
                index[p].begin(), index[p].end(), [](int value) { return value >= 0; }));
        if (visible >= 4U) {
            kept_points.push_back(scene.points[p]);
            kept_index.push_back(index[p]);
        }
    }
    scene.points = std::move(kept_points);
    index = std::move(kept_index);

    for (int a = 0; a < kCameras; ++a) {
        for (int b = a + 1; b < kCameras; ++b) {
            PairMatches pair;
            pair.camera_a = a;
            pair.camera_b = b;
            for (std::size_t p = 0; p < scene.points.size(); ++p) {
                const int ia = index[p][static_cast<std::size_t>(a)];
                const int ib = index[p][static_cast<std::size_t>(b)];
                if (ia < 0 || ib < 0) {
                    continue;
                }
                pair.matches.push_back(KeypointMatch{static_cast<std::uint32_t>(ia),
                                                     static_cast<std::uint32_t>(ib)});
            }
            if (pair.matches.size() >= 2U) {
                chunk.tie_points().matches.push_back(std::move(pair));
            }
        }
    }
    return scene;
}

struct RealisticScene {
    Document document;
    Chunk* chunk = nullptr;
    std::vector<Eigen::Vector3d> truth_centres;  ///< chunk-internal (ENU)
};

/// A 6x6 nadir block at 80 m. Each camera's reference carries a deterministic
/// ~2 m GNSS offset and ~1 degree attitude error; matches are synthesised from
/// the truth poses with sub-pixel noise and no outliers.
RealisticScene make_realistic_scene() {
    RealisticScene scene;
    Chunk& chunk = scene.document.workspace().add_chunk("realistic");
    scene.document.set_active_chunk_id(chunk.id());
    scene.chunk = &chunk;
    Sensor sensor;
    sensor.type = CameraType::Frame;
    sensor.width = 1200;
    sensor.height = 900;
    sensor.calibration = Calibration::initial(CameraType::Frame, 1200, 900, 1000.0);
    chunk.sensors().push_back(sensor);
    const Eigen::Vector3d lever(0.20, -0.10, 0.15);
    chunk.sensors()[0].antenna.offset_m = lever;
    const Calibration& truth = chunk.sensors()[0].calibration;
    const Eigen::Matrix3d mount = reference::body_to_camera_nadir();
    const Eigen::Matrix3d body = Eigen::Matrix3d::Identity();
    const geo::Geodetic origin{7.0, 45.0, 80.0};
    const Eigen::Matrix4d reference_frame = geo::enu_to_ecef(origin);
    const Eigen::Matrix3d e_reference = reference_frame.block<3, 3>(0, 0);
    chunk.transform() = reference_frame;
    chunk.meta().set(reference::kRotationOrderKey, crs::rotation_order_name(kOrder));

    constexpr int kGrid = 6;  // 6x6 cameras
    constexpr double kSpacing = 12.0;
    constexpr double kAltitude = 80.0;
    for (int gy = 0; gy < kGrid; ++gy) {
        for (int gx = 0; gx < kGrid; ++gx) {
            const int i = gy * kGrid + gx;
            const Eigen::Vector3d centre(kSpacing * (static_cast<double>(gx) - 2.5),
                                         kSpacing * (static_cast<double>(gy) - 2.5),
                                         kAltitude);
            const Eigen::Matrix3d rotation = mount * body.transpose();
            const Eigen::Vector3d centre_ecef =
                (reference_frame * centre.homogeneous()).head<3>();
            const Eigen::Vector3d antenna =
                centre_ecef + e_reference * rotation.transpose() * lever;
            const double phase = static_cast<double>(i);
            Eigen::Vector3d gnss(1.7 * std::cos(1.3 * phase),
                                 1.7 * std::sin(2.1 * phase),
                                 0.6 * std::cos(0.9 * phase + 1.0));
            gnss *= 2.0 / gnss.norm();  // 2 m 3D per camera
            const geo::Geodetic g = geo::ecef_to_geodetic(antenna + e_reference * gnss);
            const Eigen::Matrix3d enu_antenna = geo::enu_to_ecef(g).block<3, 3>(0, 0);
            const Eigen::Matrix3d r_wb =
                enu_antenna.transpose() * e_reference * body;
            Eigen::Vector3d stored = stored_from_world_body(r_wb);
            stored += Eigen::Vector3d(std::cos(0.7 * phase), std::sin(1.1 * phase),
                                      0.8 * std::cos(0.5 * phase + 0.4));

            Camera camera;
            camera.id = i;
            camera.sensor_id = 0;
            camera.enabled = true;
            camera.pose = pose_with_centre(rotation, centre);
            camera.reference.location =
                Eigen::Vector3d(g.longitude_deg, g.latitude_deg, g.height_m);
            camera.reference.accuracy = Eigen::Vector3d::Constant(2.0);
            camera.reference.enabled = true;
            camera.reference.orientation_deg = stored;
            camera.reference.rotation_accuracy_deg = Eigen::Vector3d::Constant(1.0);
            chunk.cameras().push_back(camera);
            scene.truth_centres.push_back(centre);
        }
    }

    constexpr int kPointGrid = 9;
    constexpr double kPointSpacing = 7.5;
    std::vector<Eigen::Vector3d> points;
    for (int gy = 0; gy < kPointGrid; ++gy) {
        for (int gx = 0; gx < kPointGrid; ++gx) {
            points.emplace_back(kPointSpacing * (static_cast<double>(gx) - 4.0),
                                kPointSpacing * (static_cast<double>(gy) - 4.0), 0.0);
        }
    }
    const std::size_t camera_count = chunk.cameras().size();
    std::vector<std::vector<int>> index(points.size(),
                                        std::vector<int>(camera_count, -1));
    std::vector<KeypointSet> sets(camera_count);
    std::mt19937 rng(20261105u);
    for (std::size_t c = 0; c < camera_count; ++c) {
        KeypointSet& set = sets[c];
        set.camera_id = static_cast<std::int32_t>(c);
        const Transform& pose = chunk.cameras()[c].pose;
        for (std::size_t p = 0; p < points.size(); ++p) {
            const Eigen::Vector3d p_cam =
                pose.block<3, 3>(0, 0) * points[p] + pose.block<3, 1>(0, 3);
            if (!(p_cam.z() > 1.0)) {
                continue;
            }
            Eigen::Vector2d pixel = truth.project(p_cam);
            if (pixel.x() < 4.0 || pixel.x() > 1196.0 || pixel.y() < 4.0 ||
                pixel.y() > 896.0) {
                continue;
            }
            const double noise_x =
                0.8 * (static_cast<double>(rng() % 1000U) / 999.0 - 0.5);
            const double noise_y =
                0.8 * (static_cast<double>(rng() % 1000U) / 999.0 - 0.5);
            pixel += Eigen::Vector2d(noise_x, noise_y);
            Keypoint keypoint;
            keypoint.x = static_cast<float>(pixel.x() - 0.5);
            keypoint.y = static_cast<float>(pixel.y() - 0.5);
            index[p][c] = static_cast<int>(set.keypoints.size());
            set.keypoints.push_back(keypoint);
        }
        chunk.tie_points().keypoint_sets.push_back(std::move(set));
    }
    for (std::size_t a = 0; a < camera_count; ++a) {
        for (std::size_t b = a + 1; b < camera_count; ++b) {
            PairMatches pair;
            pair.camera_a = static_cast<std::int32_t>(a);
            pair.camera_b = static_cast<std::int32_t>(b);
            for (std::size_t p = 0; p < points.size(); ++p) {
                const int ia = index[p][a];
                const int ib = index[p][b];
                if (ia < 0 || ib < 0) {
                    continue;
                }
                pair.matches.push_back(KeypointMatch{static_cast<std::uint32_t>(ia),
                                                     static_cast<std::uint32_t>(ib)});
            }
            if (pair.matches.size() >= 2U) {
                chunk.tie_points().matches.push_back(std::move(pair));
            }
        }
    }
    return scene;
}

TEST(ReferenceAlignment, TriangulatesWithKnownPoses) {
    TriScene scene = make_tri_scene();
    run_task_named(scene.document, "TriangulateTiePoints");
    Chunk& chunk = *scene.chunk;

    std::vector<bool> found(scene.points.size(), false);
    double max_error = 0.0;
    for (const Track& track : chunk.tie_points().tracks) {
        if (!track.point.has_value()) {
            continue;
        }
        std::size_t best = 0;
        double best_distance = std::numeric_limits<double>::infinity();
        for (std::size_t p = 0; p < scene.points.size(); ++p) {
            const double distance = (*track.point - scene.points[p]).norm();
            if (distance < best_distance) {
                best_distance = distance;
                best = p;
            }
        }
        max_error = std::max(max_error, best_distance);
        if (best_distance <= 0.10) {
            found[best] = true;
        }
    }
    std::size_t recovered = 0;
    for (const bool value : found) {
        recovered += value ? 1U : 0U;
    }
    const double fraction =
        static_cast<double>(recovered) / static_cast<double>(scene.points.size());
    EXPECT_GE(fraction, 0.95);
    EXPECT_LE(max_error, 0.10);

    const std::string* dropped =
        chunk.meta().find("triangulate_tie_points.observations_dropped");
    ASSERT_NE(dropped, nullptr);
    EXPECT_GE(std::stoul(*dropped) * 2U, scene.outliers);
    const std::string* triangulated =
        chunk.meta().find("triangulate_tie_points.triangulated");
    ASSERT_NE(triangulated, nullptr);
    EXPECT_LE(std::stoul(*triangulated), scene.points.size());
    std::printf("[Triangulate] recovered %.1f%% of %zu points, max error %.4f m, "
                "%s observations dropped of %zu outliers\n",
                100.0 * fraction, scene.points.size(), max_error, dropped->c_str(),
                scene.outliers);
}

/// Camera-centre RMS of `chunk` against the truth centres, in internal units.
double centre_rms(const Chunk& chunk, const std::vector<Eigen::Vector3d>& truth) {
    double sum = 0.0;
    for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
        const Transform& pose = chunk.cameras()[i].pose;
        const Eigen::Matrix3d r = pose.block<3, 3>(0, 0);
        const Eigen::Vector3d centre = -(r.transpose() * pose.block<3, 1>(0, 3));
        sum += (centre - truth[i]).squaredNorm();
    }
    return std::sqrt(sum / static_cast<double>(chunk.cameras().size()));
}

TEST(ReferenceAlignment, ReferenceThenOptimize) {
    TriScene scene = make_tri_scene();
    Chunk& chunk = *scene.chunk;
    std::vector<Eigen::Vector3d> truth;
    for (const Camera& camera : chunk.cameras()) {
        const Eigen::Matrix3d r = camera.pose.block<3, 3>(0, 0);
        truth.push_back(-(r.transpose() * camera.pose.block<3, 1>(0, 3)));
    }
    const Eigen::Matrix4d inverse = chunk.transform().inverse();

    // Perturb every reference: about 0.25 m in ECEF and 2 degrees of yaw.
    // At this 8 m range that is roughly 30 px, so it defeats the default 4 px
    // gate without putting the rays beyond what a wide-gate initial
    // triangulation can still meet.
    std::size_t perturbed = 0;
    for (std::size_t index = 0; index < chunk.cameras().size(); ++index) {
        Camera& camera = chunk.cameras()[index];
        const double sign = index % 2U == 0U ? 1.0 : -1.0;
        const Eigen::Vector3d& loc = *camera.reference.location;
        Eigen::Vector3d ecef = geo::geodetic_to_ecef(
            geo::Geodetic{loc.x(), loc.y(), loc.z()});
        // A per-camera error, not a common translation: reprojection alone
        // cannot see a translation shared by every camera and point.
        ecef += Eigen::Vector3d(0.20 * sign, 0.15 * (index % 3U == 0U ? 1.0 : -1.0),
                                0.05);
        const geo::Geodetic g = geo::ecef_to_geodetic(ecef);
        camera.reference.location =
            Eigen::Vector3d(g.longitude_deg, g.latitude_deg, g.height_m);
        camera.reference.orientation_deg->x() += 2.0 * sign;
        // Realistic sigma: the priors must not pin the cameras at the error.
        camera.reference.accuracy = Eigen::Vector3d::Constant(1.0);
        camera.reference.rotation_accuracy_deg = Eigen::Vector3d::Constant(3.0);
        camera.pose = Transform::Identity();
        ++perturbed;
    }
    ASSERT_EQ(perturbed, chunk.cameras().size());

    // Raw reference error: the centre the unadjusted metadata alone indicates.
    double raw_sum = 0.0;
    for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
        const Camera& camera = chunk.cameras()[i];
        const std::optional<reference::ReferenceRotation> measured =
            reference::measured_rotation(chunk, camera, chunk.transform());
        ASSERT_TRUE(measured.has_value());
        const Eigen::Vector3d& loc = *camera.reference.location;
        const Eigen::Vector3d antenna =
            geo::geodetic_to_ecef(geo::Geodetic{loc.x(), loc.y(), loc.z()});
        const Eigen::Vector3d lever =
            reference::effective_antenna(chunk, camera).camera_offset();
        const Eigen::Vector3d antenna_internal =
            (inverse * antenna.homogeneous()).head<3>();
        const Eigen::Vector3d centre =
            antenna_internal - measured->world_to_camera.transpose() * lever;
        raw_sum += (centre - truth[i]).squaredNorm();
    }
    const double raw_rms =
        std::sqrt(raw_sum / static_cast<double>(chunk.cameras().size()));

    run_task_named(scene.document, "AlignFromReference");
    // The deliberately coarse references leave poses tens of pixels off the
    // keypoints, so the initial triangulation needs a wide gate; the refinement
    // is what tightens it.
    run_task_named(scene.document, "TriangulateTiePoints", [](Params& p) {
        p.set("max_reprojection_error_px", 60.0);
    });
    Params params = make_params(builtin_registry().require("OptimizeCameras"));
    for (const char* name :
         {"f", "cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4"}) {
        params.set(std::string("fit_") + name, false);
    }
    params.set("camera_covariance", false);
    params.set("use_orientation", true);
    run_task(builtin_registry(), scene.document, "OptimizeCameras", params);

    const double refined_rms = centre_rms(chunk, truth);
    EXPECT_LT(refined_rms, raw_rms);
    std::printf("[RefAlign] camera centre RMS: raw reference %.4f m, "
                "after align+triangulate+optimize %.4f m\n",
                raw_rms, refined_rms);
}

void expect_pose_state_equal(const ChunkPoseState& a, const ChunkPoseState& b) {
    EXPECT_EQ(a.transform, b.transform);
    EXPECT_EQ(a.region, b.region);
    EXPECT_EQ(a.meta, b.meta);
    EXPECT_EQ(a.crs, b.crs);
    ASSERT_EQ(a.cameras.size(), b.cameras.size());
    for (std::size_t i = 0; i < a.cameras.size(); ++i) {
        EXPECT_EQ(a.cameras[i].pose, b.cameras[i].pose);
        EXPECT_EQ(a.cameras[i].enabled, b.cameras[i].enabled);
    }
    ASSERT_EQ(a.tracks.size(), b.tracks.size());
    for (std::size_t i = 0; i < a.tracks.size(); ++i) {
        EXPECT_EQ(a.tracks[i], b.tracks[i]);
    }
}

TEST(ReferenceAlignment, AlignFromReferenceIsUndoable) {
    Scene scene = make_scene(4, Eigen::Vector3d(0.2, -0.1, 0.15), true, {});
    const ChunkPoseState before = capture_chunk_pose_state(*scene.chunk);
    run_task_named(scene.document, "AlignFromReference");
    EXPECT_NE(scene.chunk->meta().find("align_from_reference.placed"), nullptr);
    auto command = make_set_chunk_pose_state_command(scene.chunk->id(), before,
                                                     "Align From Reference");
    command->apply(scene.document);
    expect_pose_state_equal(capture_chunk_pose_state(*scene.chunk), before);
}

TEST(ReferenceAlignment, TriangulateTiePointsIsUndoable) {
    TriScene scene = make_tri_scene();
    const ChunkPoseState before = capture_chunk_pose_state(*scene.chunk);
    run_task_named(scene.document, "TriangulateTiePoints");
    EXPECT_FALSE(scene.chunk->tie_points().tracks.empty());
    auto command = make_set_chunk_pose_state_command(scene.chunk->id(), before,
                                                     "Triangulate Tie Points");
    command->apply(scene.document);
    expect_pose_state_equal(capture_chunk_pose_state(*scene.chunk), before);
}

TEST(ReferenceAlignment, ReusedFrameAppliesSimilarityScale) {
    const Eigen::Vector3d lever(0.30, -0.12, 0.20);
    Scene scene = make_scene(6, lever, true, {});
    Chunk& chunk = *scene.chunk;
    constexpr double kScale = 2.5;
    Transform scaled = scene.frame;
    scaled.block<3, 3>(0, 0) *= kScale;
    chunk.transform() = scaled;
    // The truth poses live in internal units, so their centres are the ENU
    // centres divided by the transform's scale.
    for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
        Camera& camera = chunk.cameras()[i];
        const Eigen::Matrix3d r = camera.pose.block<3, 3>(0, 0);
        camera.pose.block<3, 1>(0, 3) = -r * (scene.centres[i] / kScale);
    }
    const CRS crs_before = chunk.crs();
    run_task_named(scene.document, "AlignFromReference");
    EXPECT_EQ(chunk.transform(), scaled);
    EXPECT_EQ(chunk.crs(), crs_before);
    double max_error = 0.0;
    for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
        const Transform& pose = chunk.cameras()[i].pose;
        const Eigen::Matrix3d r = pose.block<3, 3>(0, 0);
        const Eigen::Vector3d centre = -(r.transpose() * pose.block<3, 1>(0, 3));
        max_error = std::max(max_error, (centre - scene.centres[i] / kScale).norm());
    }
    // The measured rotation goes through cbrt(det), so ~1 nm of rounding is
    // expected; anything above the placement tolerance would mean the lever
    // arm kept its metric length.
    EXPECT_LE(max_error, 1e-6) << "lever arm must be divided by the similarity scale";
}

TEST(ReferenceAlignment, RefusesProjectedCameraCrs) {
    Scene scene = make_scene(4, Eigen::Vector3d(0.2, 0.0, 0.1), true, {});
    CRS projected;
    projected.epsg = 32632;  // a projected UTM slot
    scene.chunk->set_camera_crs(projected);
    const ChunkPoseState before = capture_chunk_pose_state(*scene.chunk);
    EXPECT_THROW(run_task_named(scene.document, "AlignFromReference"), ValueError);
    const ChunkPoseState after = capture_chunk_pose_state(*scene.chunk);
    EXPECT_EQ(after.transform, before.transform);
    EXPECT_EQ(after.meta, before.meta);
}

TEST(ReferenceAlignment, BuiltFrameDisablesUnplacedCameras) {
    Scene scene = make_scene(6, Eigen::Vector3d(0.2, -0.1, 0.1), false,
                             {true, false, true, true, true, true});
    Chunk& chunk = *scene.chunk;
    Marker marker;
    marker.id = 100;
    marker.position = Eigen::Vector3d(1.0, 2.0, 3.0);
    marker.position_covariance = Eigen::Matrix3d::Identity();
    chunk.markers().push_back(marker);
    const Region initial_region = chunk.region();

    // The built frame's origin is the centroid of every measured antenna.
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d& antenna : scene.antennas) {
        mean += antenna;
    }
    mean /= static_cast<double>(scene.antennas.size());
    const Eigen::Matrix4d expected = geo::enu_to_ecef(geo::ecef_to_geodetic(mean));
    std::vector<Eigen::Vector3d> placed_centres;
    for (std::size_t i = 0; i < scene.centres.size(); ++i) {
        if (i == 1U) {
            continue;
        }
        placed_centres.push_back(
            (expected.inverse() *
             (scene.reference_frame * scene.centres[i].homogeneous()))
                .head<3>());
    }
    const Region expected_region = region_from_points(placed_centres, initial_region);
    run_task_named(scene.document, "AlignFromReference");

    EXPECT_FALSE(chunk.cameras()[1].enabled);
    EXPECT_EQ(chunk.cameras()[1].pose, Transform::Identity());
    EXPECT_LT((chunk.transform() - expected).cwiseAbs().maxCoeff(), 1e-6);
    // The placed centres are exact to placement precision, so the refitted
    // region matches the same helper applied to the truth centres.
    EXPECT_LE((chunk.region().centre - expected_region.centre).norm(), 1e-5);
    EXPECT_LE((chunk.region().size - expected_region.size).norm(), 1e-5);
    ASSERT_FALSE(chunk.markers()[0].position.has_value());
    EXPECT_TRUE(chunk.markers()[0].position_covariance.isZero());
    ASSERT_NE(chunk.meta().find("align_from_reference.disabled"), nullptr);
    EXPECT_EQ(*chunk.meta().find("align_from_reference.disabled"), "1");
}

TEST(ReferenceAlignment, CountsMissingAltitudeSeparately) {
    Scene scene = make_scene(4, Eigen::Vector3d(0.1, 0.0, 0.1), true, {});
    scene.chunk->cameras()[2].reference.location->z() =
        std::numeric_limits<double>::quiet_NaN();
    run_task_named(scene.document, "AlignFromReference");
    EXPECT_EQ(*scene.chunk->meta().find("align_from_reference.placed"), "3");
    EXPECT_EQ(*scene.chunk->meta().find("align_from_reference.skipped"), "0");
    EXPECT_EQ(*scene.chunk->meta().find("align_from_reference.skipped_no_altitude"), "1");
}

TEST(ReferenceAlignment, ExplicitListNeedsEligibleReference) {
    Scene scene = make_scene(3, Eigen::Vector3d(0.1, 0.0, 0.1), true, {});
    for (Camera& camera : scene.chunk->cameras()) {
        camera.reference.enabled = false;
    }
    EXPECT_THROW(run_task_named(scene.document, "AlignFromReference",
                                [](Params& p) {
                                    p.set("cameras",
                                          std::vector<std::string>{"0"});
                                }),
                 ValueError);
    // An empty list with nothing referenced stays the parity no-op.
    EXPECT_NO_THROW(run_task_named(scene.document, "AlignFromReference"));
}

TEST(ReferenceAlignment, ResetsRequestedCameraWithDisabledReference) {
    Scene scene = make_scene(3, Eigen::Vector3d(0.1, 0.0, 0.1), true, {});
    Chunk& chunk = *scene.chunk;
    chunk.cameras()[1].reference.enabled = false;
    run_task_named(scene.document, "AlignFromReference", [](Params& p) {
        p.set("cameras", std::vector<std::string>{"0", "1"});
        p.set("reset_unreferenced", true);
    });
    EXPECT_EQ(*chunk.meta().find("align_from_reference.placed"), "1");
    EXPECT_FALSE(chunk.cameras()[1].enabled);
    EXPECT_EQ(chunk.cameras()[1].pose, Transform::Identity());
    EXPECT_TRUE(chunk.cameras()[2].enabled);  // not in the requested list
}

TEST(ReferenceAlignment, TriangulatesWithMasterCalibration) {
    TriScene scene = make_tri_scene();
    Chunk& chunk = *scene.chunk;
    Sensor slave = chunk.sensors()[0];
    slave.label = "slave";
    slave.master_sensor_id = 0;
    // Deliberately different: using the slave's own values would bias every
    // ray and fail to recover the points.
    slave.calibration = Calibration::initial(CameraType::Frame, 1200, 900, 800.0);
    chunk.sensors().push_back(slave);
    for (std::size_t i = 1; i < chunk.cameras().size(); ++i) {
        chunk.cameras()[i].sensor_id = 1;
    }
    run_task_named(scene.document, "TriangulateTiePoints");
    double max_error = 0.0;
    std::size_t matched = 0;
    for (const Track& track : chunk.tie_points().tracks) {
        if (!track.point.has_value()) {
            continue;
        }
        double best = std::numeric_limits<double>::infinity();
        for (const Eigen::Vector3d& point : scene.points) {
            best = std::min(best, (*track.point - point).norm());
        }
        max_error = std::max(max_error, best);
        matched += best <= 0.10 ? 1U : 0U;
    }
    const double fraction =
        static_cast<double>(matched) / static_cast<double>(scene.points.size());
    EXPECT_GE(fraction, 0.95);
    EXPECT_LE(max_error, 0.10);
}

TEST(ReferenceAlignment, RejectsInvalidSensorMaster) {
    TriScene tri = make_tri_scene();
    Sensor slave = tri.chunk->sensors()[0];
    slave.master_sensor_id = 42;
    tri.chunk->sensors().push_back(slave);
    EXPECT_THROW(run_task_named(tri.document, "TriangulateTiePoints"), ValueError);

    Scene scene = make_scene(4, Eigen::Vector3d(0.1, 0.0, 0.1), true, {});
    Sensor bad = scene.chunk->sensors()[0];
    bad.master_sensor_id = 42;
    scene.chunk->sensors().push_back(bad);
    EXPECT_THROW(run_task_named(scene.document, "AlignFromReference"), ValueError);
}

TEST(ReferenceAlignment, RealisticErrorsConverge) {
    RealisticScene scene = make_realistic_scene();
    Chunk& chunk = *scene.chunk;
    run_task_named(scene.document, "AlignFromReference");
    // Reprojection gate: a 2 m position error at 80 m with f = 1000 px is
    // 2/80 * 1000 = 25 px; a 1 degree attitude error is 0.01745 * 1000 = 17.5
    // px. Their root-sum-square is 30.5 px, so 60 px (2x) admits every good
    // observation while still rejecting gross mismatches.
    run_task_named(scene.document, "TriangulateTiePoints",
                   [](Params& p) { p.set("max_reprojection_error_px", 60.0); });
    Params params = make_params(builtin_registry().require("OptimizeCameras"));
    for (const char* name :
         {"f", "cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4"}) {
        params.set(std::string("fit_") + name, false);
    }
    params.set("camera_covariance", false);
    params.set("use_orientation", true);
    run_task(builtin_registry(), scene.document, "OptimizeCameras", params);

    const Calibration& calibration = chunk.sensors()[0].calibration;
    double error_sum = 0.0;
    std::size_t observations = 0;
    for (const Track& track : chunk.tie_points().tracks) {
        if (!track.point.has_value()) {
            continue;
        }
        for (const Observation& observation : track.observations) {
            const Camera& camera =
                chunk.cameras()[static_cast<std::size_t>(observation.camera_id)];
            const KeypointSet* set = chunk.tie_points().find_by_camera(camera.id);
            ASSERT_NE(set, nullptr);
            const Eigen::Vector2d pixel =
                keypoint_to_pixel(set->keypoints[static_cast<std::size_t>(
                    observation.keypoint_index)]);
            const Eigen::Vector3d x_cam =
                camera.pose.block<3, 3>(0, 0) * (*track.point) +
                camera.pose.block<3, 1>(0, 3);
            const double error = (calibration.project(x_cam) - pixel).norm();
            error_sum += error * error;
            ++observations;
        }
    }
    const double rms = std::sqrt(error_sum / static_cast<double>(observations));
    EXPECT_LT(rms, 1.0) << "final reprojection RMS (px)";
    double max_centre_error = 0.0;
    for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
        const Transform& pose = chunk.cameras()[i].pose;
        const Eigen::Matrix3d r = pose.block<3, 3>(0, 0);
        const Eigen::Vector3d centre = -(r.transpose() * pose.block<3, 1>(0, 3));
        max_centre_error =
            std::max(max_centre_error, (centre - scene.truth_centres[i]).norm());
    }
    EXPECT_LT(max_centre_error, 0.5) << "max camera centre error (m)";
    std::printf("[RefAlign] realistic block: %zu observations, RMS %.3f px, "
                "max centre error %.3f m\n",
                observations, rms, max_centre_error);
}

}  // namespace

}  // namespace alphashape
