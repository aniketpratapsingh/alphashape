#pragma once

// RegionGizmo: the pure interaction core for the region box (WP-14.15, GUI-19).
//
// Qt-free and renderer-free: a ray in chunk-local space drives every mode, so the hit tests
// and the drag math are unit-tested without a graphics context. ViewportWidget holds the thin
// adapter that turns mouse pixels into rays with the camera projection.
//
// MODES
//   * Move: drag a face; the whole box translates along that face's normal.
//   * Resize: drag a face, edge or corner; the opposite face stays fixed.
//   * Rotate: drag the ring about the region's local Z axis; the chunk transform absorbs the
//     rotation (Region has no orientation field of its own).
//
// The region is axis-aligned in chunk-local coordinates, the same frame the scene is drawn in
// (scene.hpp). The chunk Transform maps that local frame to world; Rotate bakes a local-space
// yaw about the region centre into it.

#include <optional>
#include <vector>

#include <Eigen/Core>

#include "apps/gui/viewer/scene.hpp"
#include "common/core/types.hpp"

namespace alphashape::gui {

enum class RegionGizmoMode { None, Move, Resize, Rotate };

/// A ray in chunk-local units. `direction` need not be normalised; every helper does that.
struct GizmoRay {
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    Eigen::Vector3d direction = -Eigen::Vector3d::UnitZ();
};

/// The editable state: the box and the chunk transform that rotates it about its centre.
struct GizmoState {
    Region region;
    Transform transform = Transform::Identity();
};

enum class GizmoHandleKind { None, Face, Corner, Edge, Ring };

/// One pickable part of the gizmo. Face uses `axis`/`sign`; Corner and Edge use `axes`/`signs`;
/// Ring is always about Z.
struct GizmoHandle {
    RegionGizmoMode mode = RegionGizmoMode::None;
    GizmoHandleKind kind = GizmoHandleKind::None;
    int axis = -1;
    int sign = 0;
    std::vector<int> axes;
    std::vector<int> signs;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();

    bool valid() const { return mode != RegionGizmoMode::None && kind != GizmoHandleKind::None; }
};

/// An in-progress drag: the start state, the picked handle and the fixed line/plane it uses.
struct GizmoDrag {
    GizmoState start;
    GizmoHandle handle;
    double start_parameter = 0.0;
    double start_angle = 0.0;
    Eigen::Vector3d line_origin = Eigen::Vector3d::Zero();
    Eigen::Vector3d line_direction = Eigen::Vector3d::UnitX();
};

/// Unproject a pixel into a chunk-local ray. Pixels are widget pixels, origin top-left.
GizmoRay ray_from_screen(const Eigen::Matrix4d& view_projection, int width, int height,
                         const Eigen::Vector2d& pixel);

/// Pick the part a click at `pixel` manipulates, or nullopt when nothing is hit.
std::optional<GizmoHandle> pick_handle(const GizmoState& state, RegionGizmoMode mode,
                                       const Eigen::Matrix4d& view_projection, int width,
                                       int height, const Eigen::Vector2d& pixel);

/// Pick and start a drag in one step. Nullopt when the click misses the gizmo.
std::optional<GizmoDrag> begin_drag(const GizmoState& state, RegionGizmoMode mode,
                                    const Eigen::Matrix4d& view_projection, int width,
                                    int height, const Eigen::Vector2d& pixel);

/// Start a drag from a handle chosen by the caller (the pure tests drive this directly).
std::optional<GizmoDrag> begin_drag_with_handle(const GizmoState& state,
                                                const GizmoHandle& handle,
                                                const GizmoRay& ray);

/// The state after a mouse-move ray. The camera projection is frozen by the caller, so the same
/// ray construction is used from press to release.
GizmoState update_drag(const GizmoDrag& drag, const GizmoRay& ray);

/// True when `state` differs from the drag's start state and a command should be pushed.
bool drag_changed(const GizmoDrag& drag, const GizmoState& state);

/// Line geometry for the in-progress box (and the ring / handle markers of `mode`), ready to
/// append to a Scene. Colours come from `style.region`.
std::vector<SceneLineVertex> preview_lines(const GizmoState& state, RegionGizmoMode mode,
                                           const SceneStyle& style);

}  // namespace alphashape::gui
