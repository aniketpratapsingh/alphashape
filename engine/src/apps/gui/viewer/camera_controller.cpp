#include "apps/gui/viewer/camera_controller.hpp"

#include <algorithm>
#include <cmath>

// Geometry, not just Core: .cross() and .homogeneous() live in it (a Core-only translation unit
// links against an undefined cross_impl).
#include <Eigen/Geometry>

namespace alphashape::gui {

namespace {

/// The distance rules are expressed as multiples of the scene diagonal; with no scene yet the
/// controller behaves as if the diagonal were one unit (a decoding default, never a crash).
double distance_base(double scene_diagonal) {
    return scene_diagonal > 0.0 ? scene_diagonal : 1.0;
}

}  // namespace

void CameraController::set_scene_bounds(const Eigen::Vector3d& min, const Eigen::Vector3d& max) {
    scene_centre_ = (min + max) * 0.5;
    const Eigen::Vector3d extent = (max - min).cwiseAbs();
    const double diagonal = extent.norm();
    // A single point, or a camera-only chunk before alignment, has a zero box. Widen it so the
    // distance clamps and the near/far derivation stay in a sane range.
    scene_diagonal_ = diagonal > 0.0 ? diagonal : 1.0;
}

void CameraController::fit(double aspect) {
    if (!has_scene()) {
        return;
    }
    pivot_ = scene_centre_;
    const double radius = scene_diagonal_ * 0.5;
    const double safe_aspect = aspect > 0.0 ? aspect : 1.0;
    const double half_vertical = kFieldOfViewY * 0.5;
    const double half_horizontal = std::atan(std::tan(half_vertical) * safe_aspect);
    const double half_angle = std::min(half_vertical, half_horizontal);
    // The bounding sphere fits the frustum when its silhouette subtends at most the half-angle.
    // A little margin keeps the corners of the box strictly inside rather than on the edge.
    constexpr double kFitMargin = 1.08;
    distance_ = clamp_distance(kFitMargin * radius / std::sin(half_angle));
}

void CameraController::orbit(double dx_pixels, double dy_pixels) {
    yaw_ -= dx_pixels * kRadiansPerPixel;
    pitch_ += dy_pixels * kRadiansPerPixel;
    pitch_ = std::clamp(pitch_, -kMaxPitch, kMaxPitch);
}

void CameraController::pan(double dx_pixels, double dy_pixels, int viewport_height) {
    const int height = std::max(viewport_height, 1);
    // Metres per pixel at the pivot plane: the vertical extent the frustum covers there.
    const double world_per_pixel = 2.0 * distance_ * std::tan(kFieldOfViewY * 0.5) /
                                   static_cast<double>(height);
    // The point under the cursor has to follow it, so the pivot moves against the drag.
    pivot_ -= right() * (dx_pixels * world_per_pixel);
    pivot_ += up() * (dy_pixels * world_per_pixel);
}

void CameraController::dolly(double steps) {
    const double factor = std::pow(1.15, -steps);
    distance_ = clamp_distance(distance_ * factor);
}

void CameraController::set_distance(double distance) { distance_ = clamp_distance(distance); }

void CameraController::set_angles(double yaw, double pitch) {
    yaw_ = yaw;
    pitch_ = std::clamp(pitch, -kMaxPitch, kMaxPitch);
}

double CameraController::clamp_distance(double distance) const {
    const double base = distance_base(scene_diagonal_);
    return std::clamp(distance, kMinDistanceScale * base, kMaxDistanceScale * base);
}

Eigen::Vector3d CameraController::eye() const {
    const double cos_pitch = std::cos(pitch_);
    return pivot_ + distance_ * Eigen::Vector3d(cos_pitch * std::cos(yaw_),
                                                cos_pitch * std::sin(yaw_),
                                                std::sin(pitch_));
}

Eigen::Vector3d CameraController::forward() const { return (pivot_ - eye()).normalized(); }

Eigen::Vector3d CameraController::right() const {
    // forward x world-up; degenerate only at the pitch clamp, which is kept short of the poles.
    Eigen::Vector3d side = forward().cross(Eigen::Vector3d::UnitZ());
    const double length = side.norm();
    if (length < 1e-12) {
        return Eigen::Vector3d::UnitX();
    }
    return side / length;
}

Eigen::Vector3d CameraController::up() const { return right().cross(forward()).normalized(); }

double CameraController::near_plane() const {
    // The near plane follows the distance, so zooming into a small object keeps depth precision
    // and zooming out to a large scene does not clip the foreground. It never crosses the pivot:
    // pushed that far in, the scene would be clipped away before anything of it could be seen.
    const double near_relative = distance_ * 1e-3;
    const double near_scene = distance_ - scene_diagonal_;
    const double near_z = std::max(near_relative, near_scene);
    return std::clamp(near_z, distance_ * 1e-6, distance_ * 0.5);
}

double CameraController::far_plane() const {
    // Far enough past the scene that orbiting never clips it, and always beyond the near plane.
    return std::max(distance_ + scene_diagonal_ * 4.0, near_plane() * 2.0);
}

Eigen::Matrix4d CameraController::view_matrix() const {
    const Eigen::Vector3d eye_pos = eye();
    const Eigen::Vector3d f = forward();
    const Eigen::Vector3d s = right();
    const Eigen::Vector3d u = up();

    Eigen::Matrix4d view = Eigen::Matrix4d::Identity();
    view(0, 0) = s.x();
    view(0, 1) = s.y();
    view(0, 2) = s.z();
    view(0, 3) = -s.dot(eye_pos);
    view(1, 0) = u.x();
    view(1, 1) = u.y();
    view(1, 2) = u.z();
    view(1, 3) = -u.dot(eye_pos);
    view(2, 0) = -f.x();
    view(2, 1) = -f.y();
    view(2, 2) = -f.z();
    view(2, 3) = f.dot(eye_pos);
    return view;
}

Eigen::Matrix4d CameraController::projection_matrix(double aspect) const {
    const double safe_aspect = aspect > 0.0 ? aspect : 1.0;
    const double near_z = near_plane();
    const double far_z = std::max(far_plane(), near_z * 1.0001);
    const double top = near_z * std::tan(kFieldOfViewY * 0.5);
    const double right_extent = top * safe_aspect;

    Eigen::Matrix4d projection = Eigen::Matrix4d::Zero();
    if (ortho_) {
        // The same frustum as the perspective projection at the pivot plane, so toggling keeps
        // the scene size (an orthographic half-height of distance * tan(fov/2)).
        const double half_height = distance_ * std::tan(kFieldOfViewY * 0.5);
        const double half_width = half_height * safe_aspect;
        projection(0, 0) = 1.0 / half_width;
        projection(1, 1) = 1.0 / half_height;
        projection(2, 2) = -2.0 / (far_z - near_z);
        projection(2, 3) = -(far_z + near_z) / (far_z - near_z);
        projection(3, 3) = 1.0;
    } else {
        projection(0, 0) = near_z / right_extent;
        projection(1, 1) = near_z / top;
        projection(2, 2) = -(far_z + near_z) / (far_z - near_z);
        projection(2, 3) = -2.0 * far_z * near_z / (far_z - near_z);
        projection(3, 2) = -1.0;
    }
    return projection;
}

std::optional<Eigen::Vector3d> CameraController::project(const Eigen::Vector3d& world,
                                                          double aspect) const {
    const Eigen::Matrix4d matrix = view_projection(aspect);
    const Eigen::Vector4d point = world.homogeneous();
    const Eigen::Vector4d clip = matrix * point;
    if (!ortho_) {
        if (clip.w() <= 1e-12) {
            return std::nullopt;
        }
        return Eigen::Vector3d(clip.x() / clip.w(), clip.y() / clip.w(), clip.z() / clip.w());
    }
    return Eigen::Vector3d(clip.x(), clip.y(), clip.z());
}

}  // namespace alphashape::gui
