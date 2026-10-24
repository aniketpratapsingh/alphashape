// ViewerCamera.*: the orbit camera of the 3-D view. Pure math, no widget and no graphics, so
// these run on every platform (and in every CI lane) with nothing to skip.

#include <gtest/gtest.h>

#include <cmath>
#include <optional>

#include <Eigen/Core>

#include "apps/gui/viewer/camera_controller.hpp"

using namespace alphashape::gui;

namespace {

constexpr double kTolerance = 1e-9;

/// The scene the tests frame: a 4 x 3 x 2 box centred at (10, 20, 5).
struct Box {
    Eigen::Vector3d min = Eigen::Vector3d(8.0, 18.5, 4.0);
    Eigen::Vector3d max = Eigen::Vector3d(12.0, 21.5, 6.0);
};

CameraController make_camera(const Box& box) {
    CameraController camera;
    camera.set_scene_bounds(box.min, box.max);
    return camera;
}

/// True when `world` maps inside the normalised device cube after the perspective divide.
bool inside_frustum(const CameraController& camera, const Eigen::Vector3d& world,
                    double aspect) {
    const std::optional<Eigen::Vector3d> ndc = camera.project(world, aspect);
    if (!ndc) {
        return false;
    }
    return std::abs(ndc->x()) <= 1.0 && std::abs(ndc->y()) <= 1.0 &&
           ndc->z() >= -1.0 && ndc->z() <= 1.0;
}

}  // namespace

TEST(ViewerCamera, DefaultsAreFiniteAndLookAtTheOrigin) {
    CameraController camera;
    EXPECT_FALSE(camera.has_scene());
    EXPECT_GT(camera.distance(), 0.0);
    EXPECT_TRUE(camera.eye().allFinite());
    EXPECT_NEAR(camera.forward().norm(), 1.0, kTolerance);
    EXPECT_NEAR(camera.right().dot(camera.forward()), 0.0, kTolerance);
    EXPECT_NEAR(camera.up().norm(), 1.0, kTolerance);
    EXPECT_TRUE(camera.view_matrix().allFinite());
    EXPECT_TRUE(camera.projection_matrix(1.5).allFinite());
}

TEST(ViewerCamera, FitFramesEveryCornerOfTheBox) {
    const Box box;
    CameraController camera = make_camera(box);
    for (const double aspect : {1.0, 1.6, 0.6}) {
        camera.fit(aspect);
        EXPECT_EQ(camera.pivot(), (box.min + box.max) * 0.5);
        const double diagonal = (box.max - box.min).norm();
        EXPECT_GE(camera.distance(), CameraController::kMinDistanceScale * diagonal);
        EXPECT_LE(camera.distance(), CameraController::kMaxDistanceScale * diagonal);

        // Every corner of the box (not only the bounding sphere) has to be on screen.
        for (int corner = 0; corner < 8; ++corner) {
            const Eigen::Vector3d point((corner & 1) != 0 ? box.max.x() : box.min.x(),
                                        (corner & 2) != 0 ? box.max.y() : box.min.y(),
                                        (corner & 4) != 0 ? box.max.z() : box.min.z());
            EXPECT_TRUE(inside_frustum(camera, point, aspect))
                << "aspect " << aspect << " corner " << point.transpose();
        }
    }
}

TEST(ViewerCamera, FitIsMonotoneInTheSceneSize) {
    CameraController small;
    small.set_scene_bounds(Eigen::Vector3d::Zero(), Eigen::Vector3d::Constant(1.0));
    small.fit(1.6);
    CameraController large;
    large.set_scene_bounds(Eigen::Vector3d::Zero(), Eigen::Vector3d::Constant(1000.0));
    large.fit(1.6);
    EXPECT_LT(small.distance(), large.distance());
    // Both are the same multiple of their own diagonal: fit() scales with the scene.
    EXPECT_NEAR(small.distance() / small.scene_diagonal(),
                large.distance() / large.scene_diagonal(), 1e-9);
}

TEST(ViewerCamera, FitWithoutSceneIsANoOp) {
    CameraController camera;
    const double before = camera.distance();
    camera.fit(1.6);
    EXPECT_EQ(camera.distance(), before);
    EXPECT_EQ(camera.pivot(), Eigen::Vector3d::Zero());
}

TEST(ViewerCamera, OrbitKeepsTheDistanceToThePivotAndTurns) {
    CameraController camera = make_camera(Box{});
    camera.fit(1.6);
    const double distance = camera.distance();
    const Eigen::Vector3d pivot = camera.pivot();
    const Eigen::Vector3d before = camera.eye();
    const double yaw_before = camera.yaw();
    const double pitch_before = camera.pitch();

    camera.orbit(120.0, 40.0);
    EXPECT_NEAR((camera.eye() - pivot).norm(), distance, 1e-9);
    EXPECT_NEAR(camera.distance(), distance, kTolerance);
    EXPECT_NEAR(camera.yaw(), yaw_before - 120.0 * CameraController::kRadiansPerPixel, 1e-12);
    EXPECT_NEAR(camera.pitch(), pitch_before + 40.0 * CameraController::kRadiansPerPixel, 1e-12);
    EXPECT_GT((camera.eye() - before).norm(), 0.0);
}

TEST(ViewerCamera, OrbitClampsPitchShortOfThePoles) {
    CameraController camera = make_camera(Box{});
    camera.fit(1.6);
    camera.orbit(0.0, 100000.0);  // far past the pole
    EXPECT_NEAR(camera.pitch(), CameraController::kMaxPitch, 1e-12);
    camera.orbit(0.0, -1000000.0);
    EXPECT_NEAR(camera.pitch(), -CameraController::kMaxPitch, 1e-12);
    // Up and right stay defined at the clamp (no NaN from a cross product of parallel vectors).
    EXPECT_TRUE(camera.up().allFinite());
    EXPECT_TRUE(camera.right().allFinite());
    EXPECT_NEAR(camera.right().norm(), 1.0, 1e-9);
}

TEST(ViewerCamera, OrbitIsReversible) {
    CameraController camera = make_camera(Box{});
    camera.fit(1.6);
    const Eigen::Vector3d before = camera.eye();
    camera.orbit(37.0, -12.0);
    camera.orbit(-37.0, 12.0);
    EXPECT_NEAR((camera.eye() - before).norm(), 0.0, 1e-9);
}

TEST(ViewerCamera, DollyZoomsInAndOut) {
    CameraController camera = make_camera(Box{});
    camera.fit(1.6);
    const double start = camera.distance();
    camera.dolly(1.0);
    EXPECT_LT(camera.distance(), start);
    camera.dolly(-1.0);
    EXPECT_NEAR(camera.distance(), start, 1e-9);
}

TEST(ViewerCamera, DollyIsClampedToTheSceneScaledRange) {
    const Box box;
    CameraController camera = make_camera(box);
    camera.fit(1.6);
    const double diagonal = camera.scene_diagonal();

    camera.dolly(10000.0);
    EXPECT_NEAR(camera.distance(), CameraController::kMinDistanceScale * diagonal, 1e-12);
    // The eye is still a finite point just off the pivot.
    EXPECT_TRUE(camera.eye().allFinite());

    camera.dolly(-100000.0);
    EXPECT_NEAR(camera.distance(), CameraController::kMaxDistanceScale * diagonal, 1e-12);
    EXPECT_TRUE(camera.view_matrix().allFinite());
}

TEST(ViewerCamera, SetDistanceClampsTheSameWay) {
    CameraController camera = make_camera(Box{});
    camera.fit(1.6);
    const double diagonal = camera.scene_diagonal();
    camera.set_distance(-5.0);
    EXPECT_NEAR(camera.distance(), CameraController::kMinDistanceScale * diagonal, 1e-12);
    camera.set_distance(1e30);
    EXPECT_NEAR(camera.distance(), CameraController::kMaxDistanceScale * diagonal, 1e-12);
}

TEST(ViewerCamera, PanMovesThePivotInTheViewPlaneOnly) {
    CameraController camera = make_camera(Box{});
    camera.fit(1.6);
    const Eigen::Vector3d pivot = camera.pivot();
    const double distance = camera.distance();
    const Eigen::Vector3d forward = camera.forward();
    const Eigen::Vector3d right = camera.right();
    const Eigen::Vector3d up = camera.up();

    // A rightward + upward drag of the mouse (dx > 0, dy < 0 in screen coordinates).
    camera.pan(50.0, -20.0, 800);
    const Eigen::Vector3d moved = camera.pivot() - pivot;
    // Screen-space pan: the pivot moves in the view plane (perpendicular to the view direction).
    EXPECT_NEAR(moved.dot(forward), 0.0, 1e-9);
    EXPECT_GT(moved.norm(), 0.0);
    // "The scene follows the cursor": dragging right moves the scene right, so the camera's pivot
    // travels left (-right); dragging up moves it up, so the pivot travels down (-up).
    EXPECT_LT(moved.dot(right), 0.0);
    EXPECT_LT(moved.dot(up), 0.0);
    // Panning never changes the distance or the viewing direction.
    EXPECT_NEAR(camera.distance(), distance, 1e-12);
    EXPECT_NEAR(camera.forward().dot(forward), 1.0, 1e-12);
}

TEST(ViewerCamera, PanScaleIsProportionalToTheDistance) {
    CameraController near_camera = make_camera(Box{});
    near_camera.fit(1.6);
    CameraController far_camera = near_camera;
    far_camera.set_distance(near_camera.distance() * 4.0);
    const Eigen::Vector3d near_pivot = near_camera.pivot();
    const Eigen::Vector3d far_pivot = far_camera.pivot();

    near_camera.pan(10.0, 0.0, 600);
    far_camera.pan(10.0, 0.0, 600);
    const double near_travel = (near_camera.pivot() - near_pivot).norm();
    const double far_travel = (far_camera.pivot() - far_pivot).norm();
    EXPECT_GT(near_travel, 0.0);
    // Four times as far away: the same drag covers four times the world distance.
    EXPECT_NEAR(far_travel / near_travel, 4.0, 1e-6);
}

TEST(ViewerCamera, PerspectiveProjectionMapsThePivotToTheScreenCentre) {
    CameraController camera = make_camera(Box{});
    camera.fit(1.6);
    const std::optional<Eigen::Vector3d> ndc = camera.project(camera.pivot(), 1.6);
    ASSERT_TRUE(ndc.has_value());
    EXPECT_NEAR(ndc->x(), 0.0, 1e-9);
    EXPECT_NEAR(ndc->y(), 0.0, 1e-9);
    // Between the near and far planes, in the [-1, 1] depth convention.
    EXPECT_GT(ndc->z(), -1.0);
    EXPECT_LT(ndc->z(), 1.0);
}

TEST(ViewerCamera, PerspectiveProjectionBracketsNearAndFar) {
    CameraController camera = make_camera(Box{});
    camera.fit(1.6);
    const double near_z = camera.near_plane();
    const double far_z = camera.far_plane();
    EXPECT_GT(near_z, 0.0);
    EXPECT_LT(near_z, camera.distance());
    EXPECT_GT(far_z, camera.distance());
    EXPECT_GT(far_z, near_z);

    // A point on the near plane maps to z = -1, one on the far plane to z = +1.
    const Eigen::Matrix4d projection = camera.projection_matrix(1.6);
    const auto depth_ndc = [&](double distance) {
        const Eigen::Vector4d camera_point(0.0, 0.0, -distance, 1.0);
        const Eigen::Vector4d clip = projection * camera_point;
        return clip.z() / clip.w();
    };
    EXPECT_NEAR(depth_ndc(near_z), -1.0, 1e-9);
    EXPECT_NEAR(depth_ndc(far_z), 1.0, 1e-9);
}

TEST(ViewerCamera, PerspectiveBehindTheEyeHasNoProjection) {
    CameraController camera = make_camera(Box{});
    camera.fit(1.6);
    const Eigen::Vector3d behind = camera.eye() - camera.forward() * 10.0;
    EXPECT_FALSE(camera.project(behind, 1.6).has_value());
}

TEST(ViewerCamera, OrthographicProjectionKeepsTheScaleAtThePivot) {
    CameraController camera = make_camera(Box{});
    camera.fit(1.6);
    const std::optional<Eigen::Vector3d> pivot_persp = camera.project(camera.pivot(), 1.6);
    camera.set_orthographic(true);
    EXPECT_TRUE(camera.orthographic());
    const std::optional<Eigen::Vector3d> pivot_ortho = camera.project(camera.pivot(), 1.6);
    ASSERT_TRUE(pivot_persp.has_value());
    ASSERT_TRUE(pivot_ortho.has_value());
    EXPECT_NEAR(pivot_ortho->x(), pivot_persp->x(), 1e-9);
    EXPECT_NEAR(pivot_ortho->y(), pivot_persp->y(), 1e-9);

    // No perspective divide: a point at twice the depth has the same screen size.
    const Eigen::Vector3d pivot = camera.pivot();
    const Eigen::Vector3d offset = camera.right() * camera.distance() * 0.1;
    const std::optional<Eigen::Vector3d> a = camera.project(pivot + offset, 1.6);
    const std::optional<Eigen::Vector3d> b =
        camera.project(pivot + offset + camera.forward() * camera.distance(), 1.6);
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    EXPECT_NEAR(a->x(), b->x(), 1e-9);
}

TEST(ViewerCamera, OrthographicStillFramesTheScene) {
    const Box box;
    CameraController camera = make_camera(box);
    camera.set_orthographic(true);
    camera.fit(1.6);
    for (int corner = 0; corner < 8; ++corner) {
        const Eigen::Vector3d point((corner & 1) != 0 ? box.max.x() : box.min.x(),
                                    (corner & 2) != 0 ? box.max.y() : box.min.y(),
                                    (corner & 4) != 0 ? box.max.z() : box.min.z());
        EXPECT_TRUE(inside_frustum(camera, point, 1.6)) << "corner " << point.transpose();
    }
}

TEST(ViewerCamera, DegenerateSceneStillProducesAUsableCamera) {
    // A single point (one camera before alignment) has a zero-size box.
    CameraController camera;
    camera.set_scene_bounds(Eigen::Vector3d(3.0, 4.0, 5.0), Eigen::Vector3d(3.0, 4.0, 5.0));
    EXPECT_TRUE(camera.has_scene());
    EXPECT_GT(camera.scene_diagonal(), 0.0);
    camera.fit(1.6);
    EXPECT_GT(camera.distance(), 0.0);
    EXPECT_TRUE(camera.view_matrix().allFinite());
    EXPECT_TRUE(inside_frustum(camera, Eigen::Vector3d(3.0, 4.0, 5.0), 1.6));
}

TEST(ViewerCamera, SetAnglesClampsPitch) {
    CameraController camera = make_camera(Box{});
    camera.set_angles(0.5, 10.0);
    EXPECT_NEAR(camera.pitch(), CameraController::kMaxPitch, 1e-12);
    EXPECT_NEAR(camera.yaw(), 0.5, 1e-12);
}

TEST(ViewerCamera, ToggleProjectionFlipsTheFlag) {
    CameraController camera;
    EXPECT_FALSE(camera.orthographic());
    camera.toggle_projection();
    EXPECT_TRUE(camera.orthographic());
    camera.toggle_projection();
    EXPECT_FALSE(camera.orthographic());
}
