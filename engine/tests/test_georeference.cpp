// test_georeference.cpp -- unit tests for georeference_chunk on synthetic chunks.
//
// A chunk is built in memory: a handful of aligned cameras whose local centres
// are a known similarity away from a set of true ENU positions, and GNSS
// references computed exactly from those ENU positions through our geodesy. The
// tests check that the recovered similarity puts every centre back on its truth,
// that the chunk's ENU->ECEF transform is right, that camera projections are
// invariant, and that degenerate inputs are refused without touching the chunk.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
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

Eigen::Matrix3d rotation(double degrees, const Eigen::Vector3d& axis) {
    return Eigen::AngleAxisd(degrees * kPi / 180.0, axis.normalized()).toRotationMatrix();
}

Eigen::Vector3d camera_centre(const Transform& pose) {
    const Eigen::Matrix3d r = pose.block<3, 3>(0, 0);
    const Eigen::Vector3d t = pose.block<3, 1>(0, 3);
    return -r.transpose() * t;
}

/// A world->camera 4x4 with the given rotation whose centre is exactly `centre`.
Transform pose_with_centre(const Eigen::Matrix3d& r, const Eigen::Vector3d& centre) {
    Transform pose = Transform::Identity();
    pose.block<3, 3>(0, 0) = r;
    pose.block<3, 1>(0, 3) = -r * centre;  // centre = -R^T t  =>  t = -R centre
    return pose;
}

/// The four non-collinear truth points used throughout (a rough tetrahedron).
const std::vector<Eigen::Vector3d>& truth_enu() {
    static const std::vector<Eigen::Vector3d> points = {
        {-5.0, -12.0, 2.0}, {4.0, -10.0, 1.5}, {3.0, -3.0, 3.0}, {-6.0, -2.0, 0.5}};
    return points;
}

/// The reference origin: an arbitrary point far enough from the truth that the
/// implementation's own (mean-ECEF) origin differs from it.
const geo::Geodetic& reference_origin() {
    static const geo::Geodetic origin{7.0, 45.0, 100.0};
    return origin;
}

/// The similarity the local frame is displaced by (s = 0.3, 40 deg, shift).
sfm::Similarity bad_similarity() {
    sfm::Similarity s;
    s.scale = 0.3;
    s.R = rotation(40.0, Eigen::Vector3d(0.3, 0.5, 0.8));
    s.t = Eigen::Vector3d(12.0, -7.0, 3.0);
    return s;
}

/// Build a chunk whose local camera centres are bad_similarity(true ENU) and
/// whose references are the exact geodetic positions of the true ENU points.
Chunk make_synthetic_chunk(std::size_t count = 4) {
    Chunk chunk("georef");
    const geo::Geodetic& origin = reference_origin();
    const sfm::Similarity bad = bad_similarity();
    const std::vector<Eigen::Vector3d>& truth = truth_enu();

    for (std::size_t i = 0; i < count; ++i) {
        const Eigen::Vector3d enu = truth[i % truth.size()];
        const Eigen::Vector3d local = bad.apply(enu);

        Camera camera;
        camera.id = static_cast<std::int32_t>(i);
        camera.enabled = true;
        camera.pose = pose_with_centre(rotation(20.0 + 5.0 * static_cast<double>(i),
                                                Eigen::Vector3d(0.0, 1.0, 0.2)),
                                       local);

        const geo::Geodetic g = geo::enu_to_geodetic(enu, origin);
        camera.reference.location = Eigen::Vector3d(g.longitude_deg, g.latitude_deg, g.height_m);
        camera.reference.accuracy = Eigen::Vector3d::Constant(0.02);
        camera.reference.enabled = true;
        chunk.cameras().push_back(camera);
    }
    return chunk;
}

/// The implementation's origin (geodetic of the mean ECEF of the references),
/// recomputed independently in the test from a list of reference ENU positions.
geo::Geodetic mean_ecef_origin(const std::vector<Eigen::Vector3d>& reference_enu) {
    const geo::Geodetic& origin = reference_origin();
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d& enu : reference_enu) {
        mean += geo::geodetic_to_ecef(geo::enu_to_geodetic(enu, origin));
    }
    mean /= static_cast<double>(reference_enu.size());
    return geo::ecef_to_geodetic(mean);
}

/// An ENU point in `reference_origin`'s frame, expressed in the NEW (mean-ECEF)
/// ENU frame built from `reference_enu`.
Eigen::Vector3d new_frame(const Eigen::Vector3d& enu,
                          const std::vector<Eigen::Vector3d>& reference_enu) {
    const Eigen::Vector3d ecef =
        (geo::enu_to_ecef(reference_origin()) * enu.homogeneous()).head<3>();
    return (geo::ecef_to_enu(mean_ecef_origin(reference_enu)) * ecef.homogeneous()).head<3>();
}

/// The true ENU centre of index `i` (of truth_enu) expressed in the NEW frame.
Eigen::Vector3d truth_in_new_frame(std::size_t i) {
    return new_frame(truth_enu()[i % truth_enu().size()], truth_enu());
}

// ---------------------------------------------------------------------------
// 1. Recovers the known similarity
// ---------------------------------------------------------------------------

TEST(Georeference, RecoversKnownSimilarity) {
    Chunk chunk = make_synthetic_chunk();

    // A track point placed exactly at camera 0's local centre must land on that
    // camera's true ENU centre after the call.
    const Eigen::Vector3d point_local = camera_centre(chunk.cameras()[0].pose);
    Track track;
    track.observations.push_back(Observation{0, 0});
    track.observations.push_back(Observation{1, 0});
    track.point = point_local;
    chunk.tie_points().tracks.push_back(track);

    const sfm::GeoreferenceReport report = sfm::georeference_chunk(chunk);

    ASSERT_TRUE(report.applied) << report.reason;
    ASSERT_EQ(report.references_used, 4U);

    for (std::size_t i = 0; i < 4; ++i) {
        const Eigen::Vector3d centre = camera_centre(chunk.cameras()[i].pose);
        const Eigen::Vector3d truth = truth_in_new_frame(i);
        EXPECT_LT((centre - truth).norm(), 1e-6) << "camera " << i;
    }

    const geo::Geodetic origin_new = mean_ecef_origin(truth_enu());
    const Eigen::Vector3d mapped = (chunk.transform() * Eigen::Vector4d(0, 0, 0, 1)).head<3>();
    EXPECT_LT((mapped - geo::geodetic_to_ecef(origin_new)).norm(), 1e-6);
    EXPECT_EQ(chunk.crs().epsg, 4326);
    // WP-12.6: the camera references and the (now metric ENU) world are in the
    // chunk CRS, so both slots are set explicitly to it; the marker slot keeps
    // following the chunk CRS because nothing here touched markers.
    ASSERT_TRUE(chunk.explicit_camera_crs().has_value());
    ASSERT_TRUE(chunk.explicit_world_crs().has_value());
    EXPECT_EQ(chunk.camera_crs().epsg, 4326);
    EXPECT_EQ(chunk.world_crs().epsg, 4326);
    EXPECT_FALSE(chunk.explicit_marker_crs().has_value());

    ASSERT_TRUE(chunk.tie_points().tracks[0].point.has_value());
    EXPECT_LT((*chunk.tie_points().tracks[0].point - truth_in_new_frame(0)).norm(), 1e-6);
}

// ---------------------------------------------------------------------------
// 2. The pose update keeps every projection invariant
// ---------------------------------------------------------------------------

Eigen::Vector2d project(const Transform& pose, const Eigen::Vector3d& world, double f) {
    const Eigen::Vector3d xc =
        pose.block<3, 3>(0, 0) * world + pose.block<3, 1>(0, 3);
    return Eigen::Vector2d(f * xc.x() / xc.z(), f * xc.y() / xc.z());
}

TEST(Georeference, PoseFormulaKeepsProjectionsInvariant) {
    Chunk chunk = make_synthetic_chunk();

    // A world point 5 m in front of camera 0 (in its own frame it is (0, 0, 5)).
    const Transform pose0 = chunk.cameras()[0].pose;
    const Eigen::Matrix3d r0 = pose0.block<3, 3>(0, 0);
    const Eigen::Vector3d world = camera_centre(pose0) + r0.transpose() * Eigen::Vector3d(0, 0, 5);
    const Eigen::Vector2d before = project(pose0, world, 500.0);
    ASSERT_LT((before - Eigen::Vector2d(0, 0)).norm(), 1e-9);  // at the principal point

    Track track;
    track.observations.push_back(Observation{0, 0});
    track.observations.push_back(Observation{1, 0});
    track.point = world;
    chunk.tie_points().tracks.push_back(track);

    ASSERT_TRUE(sfm::georeference_chunk(chunk).applied);

    ASSERT_TRUE(chunk.tie_points().tracks[0].point.has_value());
    const Eigen::Vector2d after =
        project(chunk.cameras()[0].pose, *chunk.tie_points().tracks[0].point, 500.0);
    EXPECT_LT((after - before).norm(), 1e-9);
}

// ---------------------------------------------------------------------------
// 3. Fewer than three references, or collinear ones, leave the chunk untouched
// ---------------------------------------------------------------------------

Chunk make_collinear_chunk() {
    Chunk chunk("collinear");
    const geo::Geodetic& origin = reference_origin();
    const sfm::Similarity bad = bad_similarity();
    for (int i = 0; i < 3; ++i) {
        const Eigen::Vector3d enu(static_cast<double>(i), 0.0, 0.0);  // collinear
        Camera camera;
        camera.id = i;
        camera.enabled = true;
        camera.pose = pose_with_centre(rotation(20.0, Eigen::Vector3d(0, 1, 0.2)), bad.apply(enu));
        const geo::Geodetic g = geo::enu_to_geodetic(enu, origin);
        camera.reference.location = Eigen::Vector3d(g.longitude_deg, g.latitude_deg, g.height_m);
        chunk.cameras().push_back(camera);
    }
    return chunk;
}

TEST(Georeference, SkipsWithFewerThanThreeOrCollinear) {
    {
        Chunk chunk = make_synthetic_chunk(2);  // only two references
        const Chunk before = chunk;
        const sfm::GeoreferenceReport report = sfm::georeference_chunk(chunk);
        EXPECT_FALSE(report.applied);
        EXPECT_FALSE(report.reason.empty());
        EXPECT_TRUE(chunk == before);
    }
    {
        Chunk chunk = make_collinear_chunk();
        const Chunk before = chunk;
        const sfm::GeoreferenceReport report = sfm::georeference_chunk(chunk);
        EXPECT_FALSE(report.applied);
        EXPECT_FALSE(report.reason.empty());
        EXPECT_TRUE(chunk == before);
    }
}

// ---------------------------------------------------------------------------
// 4. A single outlier reference is rejected
// ---------------------------------------------------------------------------

void set_reference(Camera& camera, const Eigen::Vector3d& enu) {
    const geo::Geodetic g = geo::enu_to_geodetic(enu, reference_origin());
    camera.reference.location = Eigen::Vector3d(g.longitude_deg, g.latitude_deg, g.height_m);
}

TEST(Georeference, RejectsOutlierReference) {
    Chunk chunk = make_synthetic_chunk();

    // Push camera 3's reference 50 m east of the truth its centre was built from.
    const std::vector<Eigen::Vector3d> outlier_references = {
        truth_enu()[0], truth_enu()[1], truth_enu()[2],
        truth_enu()[3] + Eigen::Vector3d(50.0, 0.0, 0.0)};
    set_reference(chunk.cameras()[3], outlier_references[3]);

    const sfm::GeoreferenceReport report = sfm::georeference_chunk(chunk);
    ASSERT_TRUE(report.applied) << report.reason;
    EXPECT_EQ(report.references_used, 3U);
    EXPECT_EQ(report.references_total, 4U);

    // The implementation's origin is the mean ECEF of ALL four references.
    for (std::size_t i = 0; i < 3; ++i) {
        const Eigen::Vector3d centre = camera_centre(chunk.cameras()[i].pose);
        const Eigen::Vector3d truth = new_frame(truth_enu()[i], outlier_references);
        EXPECT_LT((centre - truth).norm(), 1e-3) << "camera " << i;
    }
}

// ---------------------------------------------------------------------------
// 5. A second call is stable
// ---------------------------------------------------------------------------

TEST(Georeference, RepeatedCallIsStable) {
    Chunk chunk = make_synthetic_chunk();
    ASSERT_TRUE(sfm::georeference_chunk(chunk).applied);

    std::vector<Eigen::Vector3d> before;
    for (const Camera& camera : chunk.cameras()) {
        before.push_back(camera_centre(camera.pose));
    }
    const Transform transform_before = chunk.transform();

    ASSERT_TRUE(sfm::georeference_chunk(chunk).applied);

    double worst = 0.0;
    for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
        worst = std::max(worst, (camera_centre(chunk.cameras()[i].pose) - before[i]).norm());
    }
    EXPECT_LT(worst, 1e-6);
    EXPECT_LT((chunk.transform() - transform_before).cwiseAbs().maxCoeff(), 1e-6);
    EXPECT_EQ(chunk.crs().epsg, 4326);
}

}  // namespace
}  // namespace alphashape
