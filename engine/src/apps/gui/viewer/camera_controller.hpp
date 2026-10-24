#pragma once

// CameraController: the orbit camera of the 3-D viewport.
//
// Pure math (Eigen + <cmath>): no Qt, no rendering, so the interaction rules (fit, orbit, pan,
// dolly, projection switch) are unit-tested without a graphics context.
//
// CONVENTIONS
//   * The world is Z-up, in chunk-local units.
//   * The camera orbits the pivot at `distance`. The eye sits at
//     pivot + distance * (cos(pitch)·cos(yaw), cos(pitch)·sin(yaw), sin(pitch)), so a positive
//     pitch looks down at the scene and yaw turns around the vertical axis.
//   * Both matrices follow the OpenGL convention (-Z forward, NDC z in [-1, 1]). The renderer
//     multiplies QRhi::clipSpaceCorrMatrix() on the left, which is what maps that to Vulkan /
//     Metal / D3D depth ranges; every unit below the renderer therefore works in one convention.
//   * The perspective and orthographic projections use the same vertical field of view and the
//     same distance, so switching them keeps the scale of the scene where the pivot is.

#include <optional>

#include <Eigen/Core>

namespace alphashape::gui {

class CameraController {
public:
    /// Dolly limits, as a multiple of the scene diagonal.
    static constexpr double kMinDistanceScale = 1e-4;
    static constexpr double kMaxDistanceScale = 1e4;
    /// Vertical field of view, radians (45 degrees).
    static constexpr double kFieldOfViewY = 0.78539816339744831;
    /// Orbit speed, radians per pixel of drag.
    static constexpr double kRadiansPerPixel = 0.006;
    /// Pitch is clamped this far short of the poles, radians (89 degrees).
    static constexpr double kMaxPitch = 1.5533430342749532;

    /// The scene box the camera frames. A degenerate box is widened so the bracket is never
    /// zero: a single point (or a brand-new project) still yields a usable distance.
    void set_scene_bounds(const Eigen::Vector3d& min, const Eigen::Vector3d& max);
    bool has_scene() const { return scene_diagonal_ > 0.0; }
    const Eigen::Vector3d& scene_centre() const { return scene_centre_; }
    double scene_diagonal() const { return scene_diagonal_; }

    /// Frame the scene: the pivot moves to the box centre and the distance is set so the
    /// bounding sphere of the box fits the smaller of the two field-of-view half-angles at
    /// `aspect` (width / height). The viewing direction is kept, so F after orbiting does not
    /// spin the view back.
    void fit(double aspect = 1.0);

    /// Orbit around the pivot by a mouse drag in pixels. Positive dx turns the scene to the
    /// right; positive dy (downwards on screen) raises the eye, so the scene tips forwards.
    void orbit(double dx_pixels, double dy_pixels);

    /// Pan: move the pivot in the view plane so the scene under the cursor follows it.
    /// `viewport_height` is the widget height in pixels (the pan scale is per-pixel at the
    /// pivot plane).
    void pan(double dx_pixels, double dy_pixels, int viewport_height);

    /// Dolly: `steps` > 0 moves the eye toward the pivot (zoom in). The distance is clamped to
    /// [kMinDistanceScale, kMaxDistanceScale] * scene diagonal.
    void dolly(double steps);

    /// Same as dolly(), to an absolute distance, clamped the same way.
    void set_distance(double distance);
    double distance() const { return distance_; }

    void set_orthographic(bool on) { ortho_ = on; }
    bool orthographic() const { return ortho_; }
    void toggle_projection() { ortho_ = !ortho_; }

    const Eigen::Vector3d& pivot() const { return pivot_; }
    void set_pivot(const Eigen::Vector3d& pivot) { pivot_ = pivot; }
    double yaw() const { return yaw_; }
    double pitch() const { return pitch_; }
    /// Replace the viewing direction (pitch is clamped).
    void set_angles(double yaw, double pitch);

    Eigen::Vector3d eye() const;
    /// Unit vector from the eye toward the pivot.
    Eigen::Vector3d forward() const;
    Eigen::Vector3d right() const;
    Eigen::Vector3d up() const;

    /// Perspective near plane, derived from the distance.
    double near_plane() const;
    /// Far plane: beyond the scene, so nothing of it is clipped while orbiting.
    double far_plane() const;

    /// World -> camera (camera looks along -Z, +X right, +Y up).
    Eigen::Matrix4d view_matrix() const;
    /// Camera -> clip, NDC z in [-1, 1]. `aspect` is width / height.
    Eigen::Matrix4d projection_matrix(double aspect) const;
    /// projection * view, before the renderer's clip-space correction.
    Eigen::Matrix4d view_projection(double aspect) const {
        return projection_matrix(aspect) * view_matrix();
    }

    /// Normalised device coordinates of `world` (z in [-1, 1]), or nullopt when it is at or
    /// behind the eye plane in the perspective projection.
    std::optional<Eigen::Vector3d> project(const Eigen::Vector3d& world, double aspect) const;

private:
    double clamp_distance(double distance) const;

    Eigen::Vector3d pivot_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d scene_centre_ = Eigen::Vector3d::Zero();
    double scene_diagonal_ = 0.0;
    double distance_ = 2.5;
    double yaw_ = -2.3561944901923448;  // -135 degrees: looking from the south-east
    double pitch_ = 0.5235987755982988;  // 30 degrees above the horizon
    bool ortho_ = false;
};

}  // namespace alphashape::gui
