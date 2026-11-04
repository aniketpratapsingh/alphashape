// test_georeference_markers.cpp -- unit tests for update_transform (WP-13.12).
//
// A synthetic chunk is built in an arbitrary local frame: every true ENU point
// is mapped through one known similarity (scale 0.37, a rotation and a shift).
// Control markers and camera references then let update_transform recover the
// datum, while check points, blunders, scale-bar-only input and a repeated call
// exercise the reporting, rejection, refusal and idempotence paths.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "common/algorithm/sfm/georeference.hpp"
#include "common/algorithm/sfm/similarity.hpp"
#include "common/core/chunk.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape {
namespace {

namespace geo = alphashape::geo;

constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kNoIndex = std::numeric_limits<std::size_t>::max();

Eigen::Matrix3d rotation(double degrees, const Eigen::Vector3d& axis) {
    return Eigen::AngleAxisd(degrees * kPi / 180.0, axis.normalized()).toRotationMatrix();
}

Eigen::Vector3d camera_centre(const Transform& pose) {
    const Eigen::Matrix3d r = pose.block<3, 3>(0, 0);
    const Eigen::Vector3d t = pose.block<3, 1>(0, 3);
    return -r.transpose() * t;
}

Transform pose_with_centre(const Eigen::Matrix3d& r, const Eigen::Vector3d& centre) {
    Transform pose = Transform::Identity();
    pose.block<3, 3>(0, 0) = r;
    pose.block<3, 1>(0, 3) = -r * centre;
    return pose;
}

/// Five non-collinear true ENU points used by the markers.
const std::vector<Eigen::Vector3d>& truth_enu() {
    static const std::vector<Eigen::Vector3d> points = {
        {-12.0, -9.0, 1.0}, {8.0, -11.0, 2.5}, {10.0, 6.0, 0.5},
        {-7.0, 8.0, 2.0},   {0.0, 0.0, 4.0},   {2.0, -2.0, -1.0}};
    return points;
}

/// Six non-collinear true ENU camera centres (the first four cover the older
/// scenes; the last two let a coarse-camera case use distinct positions).
const std::vector<Eigen::Vector3d>& camera_truth_enu() {
    static const std::vector<Eigen::Vector3d> points = {
        {-15.0, -15.0, 3.0}, {15.0, -14.0, 2.0}, {14.0, 15.0, 3.5},
        {-14.0, 16.0, 1.0},  {0.0, -17.0, 2.5},  {0.0, 17.0, 1.5}};
    return points;
}

/// Reference origin for turning ENU into lon/lat/alt.
const geo::Geodetic& reference_origin() {
    static const geo::Geodetic origin{7.0, 45.0, 100.0};
    return origin;
}

/// The arbitrary local frame: local = truth_similarity.apply(true ENU).
sfm::Similarity truth_similarity() {
    sfm::Similarity s;
    s.scale = 0.37;
    s.R = rotation(37.0, Eigen::Vector3d(0.2, 0.7, 0.4));
    s.t = Eigen::Vector3d(9.0, -4.0, 2.5);
    return s;
}

/// Geodetic (lon, lat, alt) of a true ENU point.
Eigen::Vector3d geodetic_of(const Eigen::Vector3d& enu) {
    const geo::Geodetic g = geo::enu_to_geodetic(enu, reference_origin());
    return Eigen::Vector3d(g.longitude_deg, g.latitude_deg, g.height_m);
}

/// Deterministic per-axis noise in [-amplitude/2, amplitude/2], recentred so the
/// sum is exactly zero: the reference origin then matches the noiseless one.
std::vector<Eigen::Vector3d> zero_mean_noise(std::size_t count, std::uint64_t seed,
                                             double amplitude) {
    std::mt19937_64 rng(seed);
    std::vector<Eigen::Vector3d> noise(count, Eigen::Vector3d::Zero());
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < count; ++i) {
        const double a = static_cast<double>(rng() % 1000001) / 1000000.0;
        const double b = static_cast<double>(rng() % 1000001) / 1000000.0;
        const double c = static_cast<double>(rng() % 1000001) / 1000000.0;
        noise[i] = amplitude * Eigen::Vector3d(a - 0.5, b - 0.5, c - 0.5);
        mean += noise[i];
    }
    if (count > 0) {
        mean /= static_cast<double>(count);
    }
    for (Eigen::Vector3d& n : noise) {
        n -= mean;
    }
    return noise;
}

/// What to build. Marker positions are always truth_similarity(true ENU);
/// `marker_offsets` move only the marker's REFERENCE (a blunder).
struct Scene {
    std::size_t marker_count = 5;
    std::size_t camera_count = 4;
    bool camera_references = false;
    double camera_accuracy = 5.0;
    double marker_accuracy = 0.01;
    std::vector<Eigen::Vector3d> camera_noise;  ///< per camera, ENU metres
    std::size_t check_index = kNoIndex;
    std::vector<std::pair<std::size_t, Eigen::Vector3d>> marker_offsets;
};

Chunk make_scene(const Scene& scene) {
    Chunk chunk("georef");
    const sfm::Similarity truth = truth_similarity();
    for (std::size_t i = 0; i < scene.camera_count; ++i) {
        Camera camera;
        camera.id = static_cast<std::int32_t>(i);
        camera.enabled = true;
        const Eigen::Vector3d enu =
            camera_truth_enu()[i % camera_truth_enu().size()];
        camera.pose = pose_with_centre(
            rotation(12.0 + 3.0 * static_cast<double>(i), Eigen::Vector3d(0.0, 1.0, 0.15)),
            truth.apply(enu));
        if (scene.camera_references) {
            const Eigen::Vector3d noise =
                i < scene.camera_noise.size() ? scene.camera_noise[i] : Eigen::Vector3d::Zero();
            camera.reference.location = geodetic_of(enu + noise);
            camera.reference.accuracy = Eigen::Vector3d::Constant(scene.camera_accuracy);
            camera.reference.enabled = true;
        }
        chunk.cameras().push_back(camera);
    }
    for (std::size_t i = 0; i < scene.marker_count; ++i) {
        const Eigen::Vector3d exact = truth_enu()[i % truth_enu().size()];
        Marker marker;
        marker.id = static_cast<std::int32_t>(i);
        marker.type = MarkerType::Regular;
        marker.position = truth.apply(exact);
        Eigen::Vector3d reference = exact;
        for (const auto& [index, offset] : scene.marker_offsets) {
            if (index == i) {
                reference += offset;
            }
        }
        marker.reference = geodetic_of(reference);
        marker.reference_accuracy = Eigen::Vector3d::Constant(scene.marker_accuracy);
        marker.meta.set("type", i == scene.check_index ? "check_point" : "gcp");
        chunk.markers().push_back(marker);
    }
    return chunk;
}

/// Reference ENU points the fit will see, in the same order as update_transform
/// reads them: cameras first, then markers (check points excluded).
std::vector<Eigen::Vector3d> reference_points(const Scene& scene) {
    std::vector<Eigen::Vector3d> points;
    if (scene.camera_references) {
        for (std::size_t i = 0; i < scene.camera_count; ++i) {
            const Eigen::Vector3d enu = camera_truth_enu()[i % camera_truth_enu().size()];
            const Eigen::Vector3d noise =
                i < scene.camera_noise.size() ? scene.camera_noise[i] : Eigen::Vector3d::Zero();
            points.push_back(enu + noise);
        }
    }
    for (std::size_t i = 0; i < scene.marker_count; ++i) {
        if (i == scene.check_index) {
            continue;
        }
        Eigen::Vector3d reference = truth_enu()[i % truth_enu().size()];
        for (const auto& [index, offset] : scene.marker_offsets) {
            if (index == i) {
                reference += offset;
            }
        }
        points.push_back(reference);
    }
    return points;
}

/// A true ENU point expressed in the NEW ENU frame built from `references`.
Eigen::Vector3d expected_new_frame(const Eigen::Vector3d& true_enu,
                                   const std::vector<Eigen::Vector3d>& references) {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d& p : references) {
        mean += geo::geodetic_to_ecef(geo::enu_to_geodetic(p, reference_origin()));
    }
    mean /= static_cast<double>(references.size());
    const Transform to_local = geo::ecef_to_enu(geo::ecef_to_geodetic(mean));
    const Eigen::Vector3d ecef =
        geo::geodetic_to_ecef(geo::enu_to_geodetic(true_enu, reference_origin()));
    return (to_local * ecef.homogeneous()).head<3>();
}

// ---------------------------------------------------------------------------
// 1. Markers alone recover the datum, cameras follow
// ---------------------------------------------------------------------------

TEST(Georeference, MarkersOnlyRecoversDatum) {
    Scene scene;
    scene.marker_count = 5;
    scene.camera_count = 4;
    Chunk chunk = make_scene(scene);

    const sfm::GeoreferenceReport report = sfm::update_transform(chunk);
    ASSERT_TRUE(report.applied) << report.reason;
    EXPECT_EQ(report.control_points_used, 5U);
    EXPECT_EQ(report.cameras_used, 0U);
    EXPECT_EQ(report.check_points_used, 0U);
    EXPECT_NEAR(report.scale, 1.0 / 0.37, 1e-6);

    const std::vector<Eigen::Vector3d> references = reference_points(scene);
    for (std::size_t i = 0; i < chunk.markers().size(); ++i) {
        ASSERT_TRUE(chunk.markers()[i].position.has_value());
        const Eigen::Vector3d expected = expected_new_frame(truth_enu()[i], references);
        EXPECT_LT((*chunk.markers()[i].position - expected).norm(), 1e-6)
            << "marker " << i;
    }
    for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
        const Eigen::Vector3d expected =
            expected_new_frame(camera_truth_enu()[i], references);
        EXPECT_LT((camera_centre(chunk.cameras()[i].pose) - expected).norm(), 1e-6)
            << "camera " << i;
    }
}

// ---------------------------------------------------------------------------
// 2. A check point is reported and never influences the control fit
// ---------------------------------------------------------------------------

TEST(Georeference, CheckPointOutlierReported) {
    Scene base;
    base.marker_count = 5;
    base.camera_count = 3;
    Chunk clean = make_scene(base);

    Scene checked_scene = base;
    checked_scene.marker_count = 6;
    checked_scene.check_index = 5;
    checked_scene.marker_offsets = {{5, Eigen::Vector3d(0.10, 0.0, 0.0)}};
    Chunk checked = make_scene(checked_scene);

    const sfm::GeoreferenceReport a = sfm::update_transform(clean);
    const sfm::GeoreferenceReport b = sfm::update_transform(checked);
    ASSERT_TRUE(a.applied) << a.reason;
    ASSERT_TRUE(b.applied) << b.reason;

    for (std::size_t i = 0; i < 5; ++i) {
        ASSERT_TRUE(clean.markers()[i].position.has_value());
        ASSERT_TRUE(checked.markers()[i].position.has_value());
        EXPECT_LT((*clean.markers()[i].position - *checked.markers()[i].position).norm(),
                  1e-9)
            << "marker " << i;
    }
    for (std::size_t i = 0; i < clean.cameras().size(); ++i) {
        EXPECT_LT((clean.cameras()[i].pose - checked.cameras()[i].pose).cwiseAbs().maxCoeff(),
                  1e-9)
            << "camera " << i;
    }

    ASSERT_EQ(b.check_errors.size(), 1U);
    EXPECT_EQ(b.check_errors[0].first, 5);
    EXPECT_NEAR(b.check_errors[0].second, 0.10, 1e-3);
    EXPECT_NEAR(b.max_check_m, 0.10, 1e-3);
    EXPECT_EQ(b.check_points_used, 1U);
    EXPECT_EQ(b.control_points_used, 5U);
}

// ---------------------------------------------------------------------------
// 3. Two markers and scale bars cannot georeference
// ---------------------------------------------------------------------------

TEST(Georeference, ScalebarsOnlyNotApplied) {
    Scene scene;
    scene.marker_count = 2;
    scene.camera_count = 2;
    Chunk chunk = make_scene(scene);
    Scalebar bar;
    bar.id = 0;
    bar.from = {ScalebarEndpoint::Kind::Marker, 0};
    bar.to = {ScalebarEndpoint::Kind::Marker, 1};
    bar.reference_length_m = 10.0;
    chunk.scalebars().push_back(bar);

    const Transform transform_before = chunk.transform();
    const Eigen::Vector3d position_before = *chunk.markers()[0].position;
    const sfm::GeoreferenceReport report = sfm::update_transform(chunk);

    EXPECT_FALSE(report.applied);
    EXPECT_NE(report.reason.find("scale bars"), std::string::npos) << report.reason;
    EXPECT_LT((chunk.transform() - transform_before).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT((*chunk.markers()[0].position - position_before).norm(), 1e-12);
    EXPECT_EQ(chunk.crs().epsg, 0);
}

// ---------------------------------------------------------------------------
// 4. Tight markers outweigh noisy cameras
// ---------------------------------------------------------------------------

TEST(Georeference, MarkersOutweighNoisyCameras) {
    Scene scene;
    scene.marker_count = 4;
    scene.camera_count = 3;
    scene.camera_references = true;
    scene.camera_accuracy = 5.0;
    scene.camera_noise = zero_mean_noise(3, 0x20261104ULL, 10.0);
    Chunk chunk = make_scene(scene);

    const sfm::GeoreferenceReport report = sfm::update_transform(chunk);
    ASSERT_TRUE(report.applied) << report.reason;
    EXPECT_EQ(report.control_points_used, 4U);
    EXPECT_EQ(report.cameras_used, 3U);

    const std::vector<Eigen::Vector3d> references = reference_points(scene);
    for (std::size_t i = 0; i < chunk.markers().size(); ++i) {
        ASSERT_TRUE(chunk.markers()[i].position.has_value());
        const Eigen::Vector3d expected = expected_new_frame(truth_enu()[i], references);
        EXPECT_LT((*chunk.markers()[i].position - expected).norm(), 0.05)
            << "marker " << i;
    }
}

// ---------------------------------------------------------------------------
// 5. One blunder is rejected without disturbing the other control points
// ---------------------------------------------------------------------------

TEST(Georeference, BlunderMarkerRejected) {
    Scene scene;
    scene.marker_count = 5;
    scene.camera_count = 2;
    scene.marker_offsets = {{2, Eigen::Vector3d(20.0, 0.0, 0.0)}};
    Chunk chunk = make_scene(scene);

    const sfm::GeoreferenceReport report = sfm::update_transform(chunk);
    ASSERT_TRUE(report.applied) << report.reason;
    ASSERT_EQ(report.rejected_markers.size(), 1U);
    EXPECT_EQ(report.rejected_markers[0], 2);
    EXPECT_EQ(report.control_points_used, 4U);

    const std::vector<Eigen::Vector3d> references = reference_points(scene);
    for (std::size_t i = 0; i < chunk.markers().size(); ++i) {
        if (i == 2) {
            continue;
        }
        ASSERT_TRUE(chunk.markers()[i].position.has_value());
        const Eigen::Vector3d expected = expected_new_frame(truth_enu()[i], references);
        EXPECT_LT((*chunk.markers()[i].position - expected).norm(), 1e-3)
            << "marker " << i;
    }
}

// ---------------------------------------------------------------------------
// 6. A second call changes nothing
// ---------------------------------------------------------------------------

TEST(Georeference, RepeatsIdempotently) {
    Scene scene;
    Chunk chunk = make_scene(scene);
    ASSERT_TRUE(sfm::update_transform(chunk).applied);

    std::vector<Eigen::Vector3d> positions;
    for (const Marker& marker : chunk.markers()) {
        positions.push_back(*marker.position);
    }
    std::vector<Transform> poses;
    for (const Camera& camera : chunk.cameras()) {
        poses.push_back(camera.pose);
    }
    const Transform transform = chunk.transform();

    ASSERT_TRUE(sfm::update_transform(chunk).applied);

    for (std::size_t i = 0; i < chunk.markers().size(); ++i) {
        EXPECT_LT((*chunk.markers()[i].position - positions[i]).norm(), 1e-9)
            << "marker " << i;
    }
    for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
        EXPECT_LT((chunk.cameras()[i].pose - poses[i]).cwiseAbs().maxCoeff(), 1e-9)
            << "camera " << i;
    }
    EXPECT_LT((chunk.transform() - transform).cwiseAbs().maxCoeff(), 1e-9);
}

// ---------------------------------------------------------------------------
// 7. Marker positions move into the new frame
// ---------------------------------------------------------------------------

TEST(Georeference, MarkerPositionsTransformed) {
    Scene scene;
    Chunk chunk = make_scene(scene);
    std::vector<Eigen::Vector3d> before;
    for (const Marker& marker : chunk.markers()) {
        before.push_back(*marker.position);
    }

    ASSERT_TRUE(sfm::update_transform(chunk).applied);

    const std::vector<Eigen::Vector3d> references = reference_points(scene);
    for (std::size_t i = 0; i < chunk.markers().size(); ++i) {
        ASSERT_TRUE(chunk.markers()[i].position.has_value());
        const Eigen::Vector3d expected = expected_new_frame(truth_enu()[i], references);
        EXPECT_LT((*chunk.markers()[i].position - expected).norm(), 1e-6)
            << "marker " << i;
        EXPECT_GT((*chunk.markers()[i].position - before[i]).norm(), 1.0)
            << "marker " << i << " did not move into the new frame";
    }
}

// ---------------------------------------------------------------------------
// 8. A projected marker CRS is refused before any mutation
// ---------------------------------------------------------------------------

TEST(Georeference, ProjectedMarkerCrsRefused) {
    Scene scene;
    scene.marker_count = 5;
    Chunk chunk = make_scene(scene);
    CRS projected;
    projected.epsg = 32633;  // UTM zone 33N: projected, not lon/lat
    chunk.set_marker_crs(projected);

    const Transform transform_before = chunk.transform();
    std::vector<Eigen::Vector3d> positions_before;
    for (const Marker& marker : chunk.markers()) {
        positions_before.push_back(*marker.position);
    }
    std::vector<Transform> poses_before;
    for (const Camera& camera : chunk.cameras()) {
        poses_before.push_back(camera.pose);
    }

    const sfm::GeoreferenceReport report = sfm::update_transform(chunk);

    EXPECT_FALSE(report.applied);
    EXPECT_NE(report.reason.find("not geographic"), std::string::npos) << report.reason;
    EXPECT_LT((chunk.transform() - transform_before).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_EQ(chunk.crs().epsg, 0);
    for (std::size_t i = 0; i < chunk.markers().size(); ++i) {
        ASSERT_TRUE(chunk.markers()[i].position.has_value());
        EXPECT_LT((*chunk.markers()[i].position - positions_before[i]).norm(), 1e-12)
            << "marker " << i;
    }
    for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
        EXPECT_LT((chunk.cameras()[i].pose - poses_before[i]).cwiseAbs().maxCoeff(), 1e-12)
            << "camera " << i;
    }
}

// ---------------------------------------------------------------------------
// 9. Coincident references are degenerate and leave the chunk untouched
// ---------------------------------------------------------------------------

TEST(Georeference, IdenticalReferencesRefused) {
    Scene scene;
    scene.marker_count = 5;
    Chunk chunk = make_scene(scene);
    const Eigen::Vector3d location = *chunk.markers()[0].reference;
    for (Marker& marker : chunk.markers()) {
        marker.reference = location;
    }

    const Transform transform_before = chunk.transform();
    std::vector<Eigen::Vector3d> positions_before;
    for (const Marker& marker : chunk.markers()) {
        positions_before.push_back(*marker.position);
    }
    std::vector<Transform> poses_before;
    for (const Camera& camera : chunk.cameras()) {
        poses_before.push_back(camera.pose);
    }

    const sfm::GeoreferenceReport report = sfm::update_transform(chunk);

    EXPECT_FALSE(report.applied);
    EXPECT_NE(report.reason.find("degenerate"), std::string::npos) << report.reason;
    EXPECT_LT((chunk.transform() - transform_before).cwiseAbs().maxCoeff(), 1e-12);
    for (std::size_t i = 0; i < chunk.markers().size(); ++i) {
        ASSERT_TRUE(chunk.markers()[i].position.has_value());
        EXPECT_LT((*chunk.markers()[i].position - positions_before[i]).norm(), 1e-12)
            << "marker " << i;
    }
    for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
        EXPECT_LT((chunk.cameras()[i].pose - poses_before[i]).cwiseAbs().maxCoeff(), 1e-12)
            << "camera " << i;
    }
}

// ---------------------------------------------------------------------------
// 10. Per-point gate: coarse cameras stay in, a marker blunder goes out
// ---------------------------------------------------------------------------

TEST(Georeference, PerPointGateRejectsBlunderWithCoarseCameras) {
    Scene scene;
    scene.marker_count = 5;
    scene.camera_count = 6;
    scene.camera_references = true;
    scene.camera_accuracy = 10.0;   // coarse, but their centres are noise-free
    scene.marker_accuracy = 0.02;
    scene.marker_offsets = {{4, Eigen::Vector3d(20.0, 0.0, 0.0)}};
    Chunk chunk = make_scene(scene);

    const sfm::GeoreferenceReport report = sfm::update_transform(chunk);
    ASSERT_TRUE(report.applied) << report.reason;

    // A global max() threshold would have dropped every 10 m camera or admitted
    // the 20 m blunder; the per-point gate keeps the cameras and drops marker 4.
    ASSERT_EQ(report.rejected_markers.size(), 1U);
    EXPECT_EQ(report.rejected_markers[0], 4);
    EXPECT_EQ(report.control_points_used, 4U);
    EXPECT_EQ(report.cameras_used, 6U);

    const std::vector<Eigen::Vector3d> references = reference_points(scene);
    for (std::size_t i = 0; i < 4; ++i) {
        ASSERT_TRUE(chunk.markers()[i].position.has_value());
        const Eigen::Vector3d expected = expected_new_frame(truth_enu()[i], references);
        EXPECT_LT((*chunk.markers()[i].position - expected).norm(), 0.05) << "marker " << i;
    }
}

// ---------------------------------------------------------------------------
// 11. Derived geometry follows the new similarity
// ---------------------------------------------------------------------------

TEST(Georeference, DerivedGeometryFollowsTheTransform) {
    Scene scene;
    Chunk chunk = make_scene(scene);
    const sfm::Similarity truth = truth_similarity();

    auto cloud = std::make_shared<PointCloud>();
    cloud->points = {truth.apply(truth_enu()[0]), truth.apply(truth_enu()[1])};
    PointCloudAsset cloud_slot;
    cloud_slot.data = cloud;
    chunk.point_clouds().push_back(cloud_slot);

    auto mesh = std::make_shared<Mesh>();
    mesh->vertices = {truth.apply(truth_enu()[2]), truth.apply(truth_enu()[3])};
    ModelAsset model_slot;
    model_slot.data = mesh;
    chunk.models().push_back(model_slot);

    Shape shape;
    shape.type = ShapeType::Polyline;
    shape.vertices = {truth.apply(truth_enu()[4]), truth.apply(truth_enu()[5])};
    chunk.shapes().features.push_back(shape);

    Trajectory trajectory;
    TrajectoryPoint point;
    point.position = truth.apply(truth_enu()[0]);
    trajectory.points.push_back(point);
    chunk.trajectories().push_back(trajectory);

    const sfm::GeoreferenceReport report = sfm::update_transform(chunk);
    ASSERT_TRUE(report.applied) << report.reason;
    const std::vector<Eigen::Vector3d> references = reference_points(scene);

    ASSERT_EQ(chunk.point_clouds().size(), 1U);
    ASSERT_NE(chunk.point_clouds()[0].data, nullptr);
    EXPECT_LT((chunk.point_clouds()[0].data->points[0] -
               expected_new_frame(truth_enu()[0], references)).norm(), 1e-6);
    EXPECT_LT((chunk.point_clouds()[0].data->points[1] -
               expected_new_frame(truth_enu()[1], references)).norm(), 1e-6);

    ASSERT_EQ(chunk.models().size(), 1U);
    ASSERT_NE(chunk.models()[0].data, nullptr);
    EXPECT_LT((chunk.models()[0].data->vertices[0] -
               expected_new_frame(truth_enu()[2], references)).norm(), 1e-6);
    EXPECT_LT((chunk.models()[0].data->vertices[1] -
               expected_new_frame(truth_enu()[3], references)).norm(), 1e-6);

    ASSERT_EQ(chunk.shapes().features.size(), 1U);
    ASSERT_EQ(chunk.shapes().features[0].vertices.size(), 2U);
    EXPECT_LT((chunk.shapes().features[0].vertices[0] -
               expected_new_frame(truth_enu()[4], references)).norm(), 1e-6);
    EXPECT_LT((chunk.shapes().features[0].vertices[1] -
               expected_new_frame(truth_enu()[5], references)).norm(), 1e-6);

    ASSERT_EQ(chunk.trajectories().size(), 1U);
    ASSERT_EQ(chunk.trajectories()[0].points.size(), 1U);
    EXPECT_LT((chunk.trajectories()[0].points[0].position -
               expected_new_frame(truth_enu()[0], references)).norm(), 1e-6);
}

// ---------------------------------------------------------------------------
// 12. Region becomes the mapped-corner AABB and marker covariance scales
// ---------------------------------------------------------------------------

TEST(Georeference, RegionAndMarkerCovarianceFollowTheTransform) {
    Scene scene;
    Chunk chunk = make_scene(scene);
    const sfm::Similarity truth = truth_similarity();
    const Eigen::Vector3d corner_centre = truth.apply(Eigen::Vector3d(1.0, 2.0, 3.0));
    const Eigen::Vector3d corner_size(30.0, 20.0, 10.0);
    chunk.region().centre = corner_centre;
    chunk.region().size = corner_size;
    chunk.markers()[0].position_covariance = 0.04 * Eigen::Matrix3d::Identity();

    const sfm::GeoreferenceReport report = sfm::update_transform(chunk);
    ASSERT_TRUE(report.applied) << report.reason;

    const std::vector<Eigen::Vector3d> references = reference_points(scene);
    const auto local_to_true = [&](const Eigen::Vector3d& local) {
        return truth.R.transpose() * (local - truth.t) / truth.scale;
    };
    const Eigen::Vector3d half = 0.5 * corner_size;
    Eigen::Vector3d low = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
    Eigen::Vector3d high = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
    for (int corner = 0; corner < 8; ++corner) {
        Eigen::Vector3d offset;
        offset.x() = (corner & 1) != 0 ? half.x() : -half.x();
        offset.y() = (corner & 2) != 0 ? half.y() : -half.y();
        offset.z() = (corner & 4) != 0 ? half.z() : -half.z();
        const Eigen::Vector3d enu = local_to_true(corner_centre + offset);
        const Eigen::Vector3d mapped = expected_new_frame(enu, references);
        low = low.cwiseMin(mapped);
        high = high.cwiseMax(mapped);
    }
    EXPECT_LT((chunk.region().centre - 0.5 * (low + high)).norm(), 1e-6);
    EXPECT_LT((chunk.region().size - (high - low)).norm(), 1e-6);

    // Isotropic covariance: C' = s^2 R C R^T collapses to s^2 C.
    const double expected_var = report.scale * report.scale * 0.04;
    EXPECT_NEAR(chunk.markers()[0].position_covariance(0, 0), expected_var, 1e-12);
    EXPECT_NEAR(chunk.markers()[0].position_covariance(1, 1), expected_var, 1e-12);
    EXPECT_NEAR(chunk.markers()[0].position_covariance(2, 2), expected_var, 1e-12);
}

// ---------------------------------------------------------------------------
// 13. Exactly three points are enough for the fit
// ---------------------------------------------------------------------------

TEST(Georeference, ThreePointsSuffice) {
    Scene scene;
    scene.marker_count = 3;
    scene.camera_count = 0;
    Chunk chunk = make_scene(scene);

    const sfm::GeoreferenceReport report = sfm::update_transform(chunk);
    ASSERT_TRUE(report.applied) << report.reason;
    EXPECT_EQ(report.control_points_used, 3U);
    EXPECT_EQ(report.cameras_used, 0U);
    EXPECT_NEAR(report.scale, 1.0 / 0.37, 1e-6);
}

// ---------------------------------------------------------------------------
// 14. Zero, negative and NaN accuracies fall back to the default sigma
// ---------------------------------------------------------------------------

TEST(Georeference, UnusableAccuracyFallsBack) {
    Scene scene;
    scene.marker_count = 3;
    scene.camera_count = 0;
    Chunk chunk = make_scene(scene);
    chunk.markers()[0].reference_accuracy = Eigen::Vector3d::Zero();
    chunk.markers()[1].reference_accuracy = Eigen::Vector3d::Constant(-1.0);
    chunk.markers()[2].reference_accuracy =
        Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());

    const sfm::GeoreferenceReport report = sfm::update_transform(chunk);
    ASSERT_TRUE(report.applied) << report.reason;
    EXPECT_TRUE(std::isfinite(report.scale));
    EXPECT_TRUE(std::isfinite(report.rms_control_m));
    EXPECT_EQ(report.control_points_used, 3U);
    EXPECT_NEAR(report.scale, 1.0 / 0.37, 1e-6);
    for (const Marker& marker : chunk.markers()) {
        ASSERT_TRUE(marker.position.has_value());
        EXPECT_TRUE(marker.position->allFinite());
    }
}

// ---------------------------------------------------------------------------
// 15. A refusal leaves the chunk bit-identical
// ---------------------------------------------------------------------------

TEST(Georeference, RefusalLeavesChunkUntouched) {
    Scene scene;
    scene.marker_count = 5;
    Chunk chunk = make_scene(scene);
    const Eigen::Vector3d location = *chunk.markers()[2].reference;
    for (Marker& marker : chunk.markers()) {
        marker.reference = location;  // coincident references: degenerate
    }

    const Transform transform = chunk.transform();
    std::vector<Transform> poses;
    for (const Camera& camera : chunk.cameras()) {
        poses.push_back(camera.pose);
    }
    std::vector<std::optional<Eigen::Vector3d>> positions;
    for (const Marker& marker : chunk.markers()) {
        positions.push_back(marker.position);
    }
    const Eigen::Vector3d region_centre = chunk.region().centre;
    const Eigen::Vector3d region_size = chunk.region().size;

    const sfm::GeoreferenceReport report = sfm::update_transform(chunk);
    EXPECT_FALSE(report.applied);

    EXPECT_EQ((chunk.transform() - transform).cwiseAbs().maxCoeff(), 0.0);
    EXPECT_EQ(chunk.crs().epsg, 0);
    EXPECT_EQ(chunk.camera_crs().epsg, 0);
    EXPECT_EQ(chunk.marker_crs().epsg, 0);
    EXPECT_EQ((chunk.region().centre - region_centre).cwiseAbs().maxCoeff(), 0.0);
    EXPECT_EQ((chunk.region().size - region_size).cwiseAbs().maxCoeff(), 0.0);
    for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
        EXPECT_EQ((chunk.cameras()[i].pose - poses[i]).cwiseAbs().maxCoeff(), 0.0)
            << "camera " << i;
    }
    for (std::size_t i = 0; i < chunk.markers().size(); ++i) {
        ASSERT_EQ(chunk.markers()[i].position.has_value(), positions[i].has_value());
        EXPECT_EQ((*chunk.markers()[i].position - *positions[i]).norm(), 0.0)
            << "marker " << i;
    }
}

}  // namespace
}  // namespace alphashape
