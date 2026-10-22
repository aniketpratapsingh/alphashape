// Tests for stereo view selection (Milestone 6): per-camera neighbour
// selection plus the observed depth band, built on synthetic chunks.
#include "common/algorithm/stereo/view_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"
#include "test_rng_helpers.hpp"

namespace alphashape::stereo {
namespace {

constexpr double kPi = 3.14159265358979323846;

using Vec3 = Eigen::Vector3d;

/// A world->camera pose whose camera frame looks at `target` from `centre`.
/// Row 2 of R is the forward axis, so a point X is "in front" iff its camera
/// Z coordinate is positive.
Eigen::Matrix4d look_at_pose(const Vec3& centre, const Vec3& target) {
    const Vec3 forward = (target - centre).normalized();
    Vec3 up = Vec3::UnitZ();
    if (std::abs(forward.dot(up)) > 0.99) {
        up = Vec3::UnitY();
    }
    const Vec3 right = forward.cross(up).normalized();
    const Vec3 down = forward.cross(right);

    Eigen::Matrix3d r;
    r.row(0) = right.transpose();
    r.row(1) = down.transpose();
    r.row(2) = forward.transpose();

    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.topLeftCorner<3, 3>() = r;
    pose.topRightCorner<3, 1>() = -r * centre;
    return pose;
}

Vec3 camera_centre_of(const Camera& camera) {
    const Eigen::Matrix3d r = camera.pose.topLeftCorner<3, 3>();
    const Vec3 t = camera.pose.topRightCorner<3, 1>();
    return -r.transpose() * t;
}

/// A synthetic rig: cameras on an arc in the XY plane at `radius`, all looking
/// at the origin, plus a deterministic cloud of points near the origin that
/// every camera which sees them in front (camera Z > 0) observes.
struct ArcScene {
    Chunk chunk;
    std::vector<std::int32_t> ids;   // chunk camera order
    std::vector<double> angles_deg;
};

ArcScene make_arc_scene(const std::vector<double>& angles_deg, double radius,
                        int n_points, std::uint64_t seed) {
    ArcScene scene;
    scene.angles_deg = angles_deg;
    const Vec3 target = Vec3::Zero();

    for (std::size_t k = 0; k < angles_deg.size(); ++k) {
        const double a = angles_deg[k] * kPi / 180.0;
        const Vec3 centre(radius * std::cos(a), radius * std::sin(a), 0.0);
        Camera camera;
        camera.id = static_cast<std::int32_t>(k) + 1;
        camera.enabled = true;
        camera.pose = look_at_pose(centre, target);
        scene.chunk.cameras().push_back(camera);
        scene.ids.push_back(camera.id);
    }

    std::mt19937_64 rng(seed);
    // Portable draws, not std::distributions: those map engine bits to values
    // differently under libc++ and libstdc++ (TESTING.md, "Numeric consistency").
    const auto unit = [](std::mt19937_64& g) { return alphashape::rng::uniform_real(g, -1.0, 1.0); };
    std::vector<Camera>& cameras = scene.chunk.cameras();
    TiePoints& tie = scene.chunk.tie_points();
    for (int p = 0; p < n_points; ++p) {
        const Vec3 point = test_rng::draw3(rng, unit, unit, unit);
        Track track;
        track.point = point;
        for (std::size_t k = 0; k < cameras.size(); ++k) {
            const Eigen::Matrix3d r = cameras[k].pose.topLeftCorner<3, 3>();
            const Vec3 t = cameras[k].pose.topRightCorner<3, 1>();
            const Vec3 x_cam = r * point + t;
            if (x_cam.z() > 0.0) {
                track.observations.push_back(
                    Observation{cameras[k].id, static_cast<std::uint32_t>(p)});
            }
        }
        tie.tracks.push_back(std::move(track));
    }
    return scene;
}

const ViewNeighbourhood* find_neighbourhood(const std::vector<ViewNeighbourhood>& result,
                                            std::int32_t camera_id) {
    for (const ViewNeighbourhood& n : result) {
        if (n.camera_id == camera_id) {
            return &n;
        }
    }
    return nullptr;
}

// -- test cases -------------------------------------------------------------

TEST(ViewSelection, FindsSortedTruncatedNeighbours) {
    const ArcScene scene = make_arc_scene({0.0, 12.0, 24.0, 36.0, 48.0, 60.0}, 20.0, 300, 7);
    ViewSelectionOptions options;
    options.max_neighbors = 2;
    options.min_common_points = 10;

    const std::vector<ViewNeighbourhood> result = select_views(scene.chunk, options);
    ASSERT_FALSE(result.empty());

    // Output is in chunk camera order, one row per enabled camera with a
    // neighbour and a valid range.
    std::vector<std::int32_t> output_ids;
    for (const ViewNeighbourhood& n : result) {
        output_ids.push_back(n.camera_id);
    }
    EXPECT_TRUE(std::is_sorted(output_ids.begin(), output_ids.end()));

    const ViewNeighbourhood* first = find_neighbourhood(result, scene.ids[0]);
    ASSERT_NE(first, nullptr);
    ASSERT_EQ(first->neighbours.size(), 2U);          // truncated to max_neighbors
    ASSERT_EQ(first->scores.size(), first->neighbours.size());
    // Cameras 12/24/36 deg away are in-window; 48/60 are out. Top two by score,
    // ties broken by ascending id -> the two nearest.
    EXPECT_EQ(first->neighbours[0], scene.ids[1]);
    EXPECT_EQ(first->neighbours[1], scene.ids[2]);

    for (const ViewNeighbourhood& n : result) {
        for (std::size_t i = 0; i < n.neighbours.size(); ++i) {
            EXPECT_NE(n.neighbours[i], n.camera_id);            // no self
            EXPECT_NE(std::find(scene.ids.begin(), scene.ids.end(), n.neighbours[i]),
                      scene.ids.end());                          // valid id
            if (i > 0) {
                EXPECT_GE(n.scores[i - 1], n.scores[i]);         // non-increasing
            }
        }
        EXPECT_LT(n.depth_min, n.depth_max);
    }
}

TEST(ViewSelection, NearDuplicateCameraIsNotANeighbour) {
    // Camera 2 sits 0.5 deg from camera 1: baseline ~0.17 on a radius-20 arc, so
    // the triangulation angle at the cloud (~0.5 deg) is below the 2 deg floor.
    const ArcScene scene = make_arc_scene({0.0, 0.5, 12.0, 24.0}, 20.0, 50, 11);
    ViewSelectionOptions options;

    const std::vector<ViewNeighbourhood> result = select_views(scene.chunk, options);
    const std::int32_t twin_a = scene.ids[0];
    const std::int32_t twin_b = scene.ids[1];

    const ViewNeighbourhood* a = find_neighbourhood(result, twin_a);
    const ViewNeighbourhood* b = find_neighbourhood(result, twin_b);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(std::find(a->neighbours.begin(), a->neighbours.end(), twin_b), a->neighbours.end());
    EXPECT_EQ(std::find(b->neighbours.begin(), b->neighbours.end(), twin_a), b->neighbours.end());
}

TEST(ViewSelection, MinCommonPointsThresholdExcludesSparsePair) {
    const ArcScene scene = make_arc_scene({0.0, 20.0}, 20.0, 5, 13);

    ViewSelectionOptions strict;
    strict.min_common_points = 10;  // only 5 shared points -> pair excluded
    EXPECT_TRUE(select_views(scene.chunk, strict).empty());

    ViewSelectionOptions permissive;
    permissive.min_common_points = 5;  // 5 shared points meets the bar
    const std::vector<ViewNeighbourhood> result = select_views(scene.chunk, permissive);
    ASSERT_EQ(result.size(), 2U);
    EXPECT_EQ(result[0].neighbours.size(), 1U);
    EXPECT_EQ(result[0].neighbours[0], scene.ids[1]);
}

TEST(ViewSelection, DisabledCamerasAreNeitherOutputNorNeighbours) {
    ArcScene scene = make_arc_scene({0.0, 12.0, 24.0}, 20.0, 50, 17);
    scene.chunk.cameras()[1].enabled = false;  // middle camera, id scene.ids[1]
    const std::int32_t disabled = scene.ids[1];

    ViewSelectionOptions options;
    const std::vector<ViewNeighbourhood> result = select_views(scene.chunk, options);
    ASSERT_EQ(result.size(), 2U);
    for (const ViewNeighbourhood& n : result) {
        EXPECT_NE(n.camera_id, disabled);
        EXPECT_EQ(std::find(n.neighbours.begin(), n.neighbours.end(), disabled),
                  n.neighbours.end());
    }
}

TEST(ViewSelection, DepthBandContainsTheCloudAndIsPositive) {
    const ArcScene scene = make_arc_scene({0.0, 12.0, 24.0, 36.0}, 20.0, 300, 19);
    ViewSelectionOptions options;
    const std::vector<ViewNeighbourhood> result = select_views(scene.chunk, options);
    ASSERT_FALSE(result.empty());

    for (const ViewNeighbourhood& n : result) {
        EXPECT_GT(n.depth_min, 0.0);

        // Recompute this camera's observed depths exactly as the implementation
        // does, then confirm every in-band (2..98 percentile) depth sits strictly
        // inside the widened [depth_min, depth_max].
        const Camera* camera = nullptr;
        for (const Camera& c : scene.chunk.cameras()) {
            if (c.id == n.camera_id) {
                camera = &c;
            }
        }
        ASSERT_NE(camera, nullptr);
        const Vec3 centre = camera_centre_of(*camera);

        std::vector<double> depths;
        for (const Track& track : scene.chunk.tie_points().tracks) {
            if (!track.point.has_value()) {
                continue;
            }
            for (const Observation& observation : track.observations) {
                if (observation.camera_id == n.camera_id) {
                    depths.push_back((*track.point - centre).norm());
                }
            }
        }
        ASSERT_GE(depths.size(), 3U);
        std::sort(depths.begin(), depths.end());
        const std::size_t lo = static_cast<std::size_t>(
            std::llround(0.02 * static_cast<double>(depths.size() - 1)));
        const std::size_t hi = static_cast<std::size_t>(
            std::llround(0.98 * static_cast<double>(depths.size() - 1)));
        for (std::size_t i = lo; i <= hi; ++i) {
            EXPECT_LT(n.depth_min, depths[i]);
            EXPECT_LT(depths[i], n.depth_max);
        }
    }
}

TEST(ViewSelection, IsDeterministic) {
    const ArcScene scene = make_arc_scene({0.0, 12.0, 24.0, 36.0, 48.0}, 20.0, 300, 23);
    ViewSelectionOptions options;

    const std::vector<ViewNeighbourhood> a = select_views(scene.chunk, options);
    const std::vector<ViewNeighbourhood> b = select_views(scene.chunk, options);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].camera_id, b[i].camera_id);
        EXPECT_EQ(a[i].neighbours, b[i].neighbours);
        EXPECT_EQ(a[i].scores, b[i].scores);
        EXPECT_EQ(a[i].depth_min, b[i].depth_min);
        EXPECT_EQ(a[i].depth_max, b[i].depth_max);
    }
}

TEST(ViewSelection, InvalidOptionsThrow) {
    const ArcScene scene = make_arc_scene({0.0, 12.0}, 20.0, 20, 29);

    ViewSelectionOptions options;
    options.max_neighbors = 0;
    EXPECT_THROW(select_views(scene.chunk, options), ValueError);

    options = ViewSelectionOptions{};
    options.max_neighbors = 256;
    EXPECT_THROW(select_views(scene.chunk, options), ValueError);

    options = ViewSelectionOptions{};
    options.min_common_points = 0;
    EXPECT_THROW(select_views(scene.chunk, options), ValueError);

    options = ViewSelectionOptions{};
    options.min_angle_deg = 45.0;
    options.max_angle_deg = 45.0;
    EXPECT_THROW(select_views(scene.chunk, options), ValueError);

    options = ViewSelectionOptions{};
    options.min_angle_deg = -1.0;
    EXPECT_THROW(select_views(scene.chunk, options), ValueError);

    options = ViewSelectionOptions{};
    options.max_angle_deg = 200.0;
    EXPECT_THROW(select_views(scene.chunk, options), ValueError);
}

}  // namespace
}  // namespace alphashape::stereo
