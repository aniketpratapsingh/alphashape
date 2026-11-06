#include "apps/gui/viewer/region_gizmo.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <utility>

#include <Eigen/LU>

namespace alphashape::gui {

namespace {

constexpr double kMinRegionSize = 1e-9;
constexpr double kPickRadiusPixels = 12.0;
constexpr int kRingSegments = 64;

Eigen::Vector3d normalised(const Eigen::Vector3d& value) {
    // Scalar arithmetic rather than Eigen's packet paths: GCC 13 otherwise reports a
    // false-positive -Wmaybe-uninitialized for the tail of the padded vector.
    const double length =
        std::sqrt(value.x() * value.x() + value.y() * value.y() + value.z() * value.z());
    if (!(length > 0.0)) {
        return Eigen::Vector3d::UnitZ();
    }
    Eigen::Vector3d result;
    result.x() = value.x() / length;
    result.y() = value.y() / length;
    result.z() = value.z() / length;
    return result;
}

std::array<Eigen::Vector3d, 8> box_corners(const Region& region) {
    std::array<Eigen::Vector3d, 8> corners{};
    const Eigen::Vector3d half = region.size * 0.5;
    for (int i = 0; i < 8; ++i) {
        const Eigen::Vector3d offset((i & 1) != 0 ? half.x() : -half.x(),
                                     (i & 2) != 0 ? half.y() : -half.y(),
                                     (i & 4) != 0 ? half.z() : -half.z());
        corners[static_cast<std::size_t>(i)] = region.centre + region.rotation * offset;
    }
    return corners;
}

/// Express a chunk-local ray in the box's own frame: a point p maps to
/// R^T (p - centre), so the oriented box becomes an axis-aligned box centred
/// on the origin and the existing hit test applies unchanged.
GizmoRay to_region_local(const Region& region, const GizmoRay& ray) {
    GizmoRay local;
    local.origin = region.rotation.transpose() * (ray.origin - region.centre);
    local.direction = region.rotation.transpose() * ray.direction;
    return local;
}

Eigen::Vector3d from_region_local(const Region& region,
                                  const Eigen::Vector3d& local) {
    return region.centre + region.rotation * local;
}

/// Parameter `t` on P + t*d of the point closest to the ray O + s*r. Falls back to the
/// projection of O onto the line when the two are parallel (the ray's origin is then the best
/// available reference).
double line_parameter(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction,
                      const GizmoRay& ray) {
    const Eigen::Vector3d d = normalised(direction);
    const Eigen::Vector3d r = normalised(ray.direction);
    const Eigen::Vector3d w0 = origin - ray.origin;
    const double b = d.dot(r);
    const double denom = 1.0 - b * b;
    if (std::abs(denom) < 1e-12) {
        return -d.dot(w0);
    }
    return (b * r.dot(w0) - d.dot(w0)) / denom;
}

/// Where a ray crosses the plane z = `height`, in the box's local frame. Nullopt when parallel.
std::optional<Eigen::Vector3d> plane_hit(double height, const GizmoRay& ray) {
    const Eigen::Vector3d r = normalised(ray.direction);
    if (std::abs(r.z()) < 1e-9) {
        return std::nullopt;
    }
    const double t = (height - ray.origin.z()) / r.z();
    if (t < 0.0) {
        return std::nullopt;
    }
    return ray.origin + r * t;
}

/// Ray vs axis-aligned box. Returns the entry face (nearest hit) as axis/sign, or the exit face
/// when the ray starts inside.
struct BoxHit {
    double entry = 0.0;
    double exit = 0.0;
    int entry_axis = -1;
    int entry_sign = 0;
    int exit_axis = -1;
    int exit_sign = 0;
};

std::optional<BoxHit> ray_box(const Region& region, const GizmoRay& ray) {
    const Eigen::Vector3d r = normalised(ray.direction);
    const Eigen::Vector3d half = region.size.cwiseMax(Eigen::Vector3d::Constant(kMinRegionSize)) *
                                 0.5;
    const Eigen::Vector3d lo = region.centre - half;
    const Eigen::Vector3d hi = region.centre + half;
    double t_min = -std::numeric_limits<double>::infinity();
    double t_max = std::numeric_limits<double>::infinity();
    int near_axis = -1;
    int near_sign = 0;
    int far_axis = -1;
    int far_sign = 0;
    for (int axis = 0; axis < 3; ++axis) {
        const double origin = ray.origin[axis];
        const double direction = r[axis];
        if (std::abs(direction) < 1e-12) {
            if (origin < lo[axis] || origin > hi[axis]) {
                return std::nullopt;
            }
            continue;
        }
        const double inv = 1.0 / direction;
        double t0 = (lo[axis] - origin) * inv;
        double t1 = (hi[axis] - origin) * inv;
        int sign0 = -1;
        int sign1 = 1;
        if (t0 > t1) {
            std::swap(t0, t1);
            sign0 = 1;
            sign1 = -1;
        }
        if (t0 > t_min) {
            t_min = t0;
            near_axis = axis;
            near_sign = sign0;
        }
        if (t1 < t_max) {
            t_max = t1;
            far_axis = axis;
            far_sign = sign1;
        }
    }
    if (t_max < 0.0 || t_min > t_max) {
        return std::nullopt;
    }
    BoxHit hit;
    hit.entry = t_min;
    hit.exit = t_max;
    hit.entry_axis = near_axis;
    hit.entry_sign = near_sign;
    hit.exit_axis = far_axis;
    hit.exit_sign = far_sign;
    return hit;
}

std::optional<Eigen::Vector2d> project_to_pixels(const Eigen::Matrix4d& view_projection,
                                                 int width, int height,
                                                 const Eigen::Vector3d& world) {
    if (width <= 0 || height <= 0) {
        return std::nullopt;
    }
    const Eigen::Matrix4d& m = view_projection;
    const double clip_x = m(0, 0) * world.x() + m(0, 1) * world.y() +
                          m(0, 2) * world.z() + m(0, 3);
    const double clip_y = m(1, 0) * world.x() + m(1, 1) * world.y() +
                          m(1, 2) * world.z() + m(1, 3);
    const double w = m(3, 0) * world.x() + m(3, 1) * world.y() + m(3, 2) * world.z() + m(3, 3);
    if (!(w > 1e-12)) {
        return std::nullopt;
    }
    const double ndc_x = clip_x / w;
    const double ndc_y = clip_y / w;
    return Eigen::Vector2d((ndc_x * 0.5 + 0.5) * static_cast<double>(width),
                           (0.5 - ndc_y * 0.5) * static_cast<double>(height));
}

}  // namespace

GizmoRay ray_from_screen(const Eigen::Matrix4d& view_projection, int width, int height,
                         const Eigen::Vector2d& pixel) {
    GizmoRay ray;
    if (width <= 0 || height <= 0) {
        return ray;
    }
    const Eigen::Matrix4d inverse = view_projection.inverse();
    const double ndc_x = pixel.x() / static_cast<double>(width) * 2.0 - 1.0;
    const double ndc_y = 1.0 - pixel.y() / static_cast<double>(height) * 2.0;
    const auto unproject = [&](double z) {
        const double x = inverse(0, 0) * ndc_x + inverse(0, 1) * ndc_y +
                         inverse(0, 2) * z + inverse(0, 3);
        const double y = inverse(1, 0) * ndc_x + inverse(1, 1) * ndc_y +
                         inverse(1, 2) * z + inverse(1, 3);
        const double wz = inverse(2, 0) * ndc_x + inverse(2, 1) * ndc_y +
                          inverse(2, 2) * z + inverse(2, 3);
        const double w = inverse(3, 0) * ndc_x + inverse(3, 1) * ndc_y +
                         inverse(3, 2) * z + inverse(3, 3);
        Eigen::Vector3d point;
        point.x() = x / w;
        point.y() = y / w;
        point.z() = wz / w;
        return point;
    };
    const Eigen::Vector3d near_point = unproject(-1.0);
    const Eigen::Vector3d far_point = unproject(1.0);
    ray.origin = near_point;
    ray.direction = normalised(far_point - near_point);
    return ray;
}

std::optional<GizmoHandle> pick_handle(const GizmoState& state, RegionGizmoMode mode,
                                       const Eigen::Matrix4d& view_projection, int width,
                                       int height, const Eigen::Vector2d& pixel) {
    if (mode == RegionGizmoMode::None) {
        return std::nullopt;
    }
    const auto projected = [&](const Eigen::Vector3d& point) {
        return project_to_pixels(view_projection, width, height, point);
    };
    if (mode == RegionGizmoMode::Rotate) {
        const Region& region = state.region;
        const double radius = std::max(region.size.x(), region.size.y()) * 0.7;
        double best = kPickRadiusPixels;
        std::optional<GizmoHandle> found;
        for (int i = 0; i < kRingSegments; ++i) {
            const double angle = 2.0 * std::numbers::pi * static_cast<double>(i) /
                                 static_cast<double>(kRingSegments);
            const Eigen::Vector3d point = from_region_local(
                region, Eigen::Vector3d(std::cos(angle) * radius,
                                        std::sin(angle) * radius, 0.0));
            const std::optional<Eigen::Vector2d> screen = projected(point);
            if (!screen) {
                continue;
            }
            const double distance = (*screen - pixel).norm();
            if (distance < best) {
                best = distance;
                GizmoHandle handle;
                handle.mode = RegionGizmoMode::Rotate;
                handle.kind = GizmoHandleKind::Ring;
                handle.axis = 2;
                handle.point = point;
                found = handle;
            }
        }
        return found;
    }
    if (mode == RegionGizmoMode::Move) {
        const GizmoRay ray = ray_from_screen(view_projection, width, height, pixel);
        const GizmoRay local_ray = to_region_local(state.region, ray);
        // The local test box is axis-aligned on the origin with the region's
        // extents; `axis`/`sign` therefore already name region-local axes.
        Region local_box;
        local_box.size = state.region.size;
        const std::optional<BoxHit> hit = ray_box(local_box, local_ray);
        if (!hit) {
            return std::nullopt;
        }
        const bool inside = hit->entry < 0.0;
        GizmoHandle handle;
        handle.mode = RegionGizmoMode::Move;
        handle.kind = GizmoHandleKind::Face;
        handle.axis = inside ? hit->exit_axis : hit->entry_axis;
        handle.sign = inside ? hit->exit_sign : hit->entry_sign;
        const Eigen::Vector3d local_point =
            local_ray.origin + normalised(local_ray.direction) *
                                   (inside ? hit->exit : hit->entry);
        handle.point = from_region_local(state.region, local_point);
        return handle;
    }
    // Resize: the nearest corner or edge midpoint in screen space.
    const std::array<Eigen::Vector3d, 8> corners = box_corners(state.region);
    double best = kPickRadiusPixels;
    std::optional<GizmoHandle> found;
    for (int i = 0; i < 8; ++i) {
        const std::optional<Eigen::Vector2d> screen =
            projected(corners[static_cast<std::size_t>(i)]);
        if (!screen) {
            continue;
        }
        const double distance = (*screen - pixel).norm();
        if (distance < best) {
            best = distance;
            GizmoHandle handle;
            handle.mode = RegionGizmoMode::Resize;
            handle.kind = GizmoHandleKind::Corner;
            handle.point = corners[static_cast<std::size_t>(i)];
            for (int axis = 0; axis < 3; ++axis) {
                const bool positive = (i & (1 << axis)) != 0;
                handle.axes.push_back(axis);
                handle.signs.push_back(positive ? 1 : -1);
            }
            found = handle;
        }
    }
    for (int fixed_axis = 0; fixed_axis < 3; ++fixed_axis) {
        for (int pair = 0; pair < 4; ++pair) {
            const int low = pair & 3;
            // Enumerate the two corner choices for the remaining bits by stepping corner ids.
            for (int edge_variant = 0; edge_variant < 2; ++edge_variant) {
                int base = 0;
                for (int axis = 0, bit = 0; axis < 3; ++axis) {
                    if (axis == fixed_axis) {
                        continue;
                    }
                    if (((low >> bit) & 1) != 0) {
                        base |= 1 << axis;
                    }
                    ++bit;
                }
                int a = base;
                int b = base;
                if (edge_variant == 0) {
                    a |= 1 << fixed_axis;
                } else {
                    b |= 1 << fixed_axis;
                }
                const Eigen::Vector3d midpoint =
                    (corners[static_cast<std::size_t>(a)] +
                     corners[static_cast<std::size_t>(b)]) * 0.5;
                const std::optional<Eigen::Vector2d> screen = projected(midpoint);
                if (!screen) {
                    continue;
                }
                const double distance = (*screen - pixel).norm();
                if (distance < best) {
                    best = distance;
                    GizmoHandle handle;
                    handle.mode = RegionGizmoMode::Resize;
                    handle.kind = GizmoHandleKind::Edge;
                    handle.point = midpoint;
                    for (int axis = 0; axis < 3; ++axis) {
                        if (axis == fixed_axis) {
                            continue;
                        }
                        const bool positive = (base & (1 << axis)) != 0;
                        handle.axes.push_back(axis);
                        handle.signs.push_back(positive ? 1 : -1);
                    }
                    found = handle;
                }
            }
        }
    }
    return found;
}

std::optional<GizmoDrag> begin_drag_with_handle(const GizmoState& state,
                                                const GizmoHandle& handle,
                                                const GizmoRay& ray) {
    if (!handle.valid()) {
        return std::nullopt;
    }
    GizmoDrag drag;
    drag.start = state;
    drag.handle = handle;
    if (handle.kind == GizmoHandleKind::Ring) {
        // The ring lies in the box's local XY plane, so the hit test runs in
        // the box frame and the angle is measured about its local Z.
        const GizmoRay local_ray = to_region_local(state.region, ray);
        const std::optional<Eigen::Vector3d> hit = plane_hit(0.0, local_ray);
        if (!hit) {
            return std::nullopt;
        }
        drag.start_angle = std::atan2(hit->y(), hit->x());
        drag.line_origin = Eigen::Vector3d::Zero();
        drag.line_direction = Eigen::Vector3d::UnitZ();
        return drag;
    }
    if (handle.kind == GizmoHandleKind::Face) {
        Eigen::Vector3d axis = Eigen::Vector3d::Zero();
        if (handle.axis >= 0 && handle.axis < 3) {
            // Move always follows +axis (the face normal), so the centre
            // tracks the cursor on a negative face too. Resize measures along
            // the signed outward normal, so its delta is the growth itself.
            axis[handle.axis] =
                handle.mode == RegionGizmoMode::Move
                    ? 1.0
                    : (handle.sign < 0 ? -1.0 : 1.0);
        }
        drag.line_origin = state.region.centre;
        drag.line_direction = state.region.rotation * axis;
        drag.start_parameter = line_parameter(drag.line_origin, drag.line_direction, ray);
        return drag;
    }
    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
    const std::size_t count = std::min(handle.axes.size(), handle.signs.size());
    for (std::size_t i = 0; i < count; ++i) {
        const int axis = handle.axes[i];
        if (axis >= 0 && axis < 3) {
            // The signed outward diagonal: dragging that corner away from the
            // box advances this line, and its delta is already the per-axis
            // growth amount (divided by sqrt(count) below).
            direction[axis] += handle.signs[i] < 0 ? -1.0 : 1.0;
        }
    }
    drag.line_origin = handle.point;
    drag.line_direction = state.region.rotation * normalised(direction);
    drag.start_parameter = line_parameter(drag.line_origin, drag.line_direction, ray);
    return drag;
}

std::optional<GizmoDrag> begin_drag(const GizmoState& state, RegionGizmoMode mode,
                                    const Eigen::Matrix4d& view_projection, int width,
                                    int height, const Eigen::Vector2d& pixel) {
    const std::optional<GizmoHandle> handle =
        pick_handle(state, mode, view_projection, width, height, pixel);
    if (!handle) {
        return std::nullopt;
    }
    return begin_drag_with_handle(
        state, *handle, ray_from_screen(view_projection, width, height, pixel));
}

namespace {

void resize_axis(GizmoState& state, int axis, int sign, double movement) {
    if (axis < 0 || axis > 2) {
        return;
    }
    const double old_size = state.region.size[axis];
    // `movement` is measured along the signed outward normal, so it is the
    // growth itself: applying the sign again would flip a negative face's
    // resize and move it the wrong way.
    const double new_size = std::max(kMinRegionSize, old_size + movement);
    // Keep the opposite face fixed. The box centre moves half the size change
    // along the box's own axis, which is `region.rotation` column `axis`, not
    // a chunk axis: on an oriented box those differ.
    Eigen::Vector3d local_shift = Eigen::Vector3d::Zero();
    local_shift[axis] = static_cast<double>(sign) * (new_size - old_size) * 0.5;
    state.region.centre += state.region.rotation * local_shift;
    state.region.size[axis] = new_size;
}

}  // namespace

GizmoState update_drag(const GizmoDrag& drag, const GizmoRay& ray) {
    GizmoState state = drag.start;
    if (drag.handle.kind == GizmoHandleKind::Ring) {
        const GizmoRay local_ray = to_region_local(state.region, ray);
        const std::optional<Eigen::Vector3d> hit = plane_hit(0.0, local_ray);
        if (!hit) {
            return state;
        }
        const double angle = std::atan2(hit->y(), hit->x());
        double delta = angle - drag.start_angle;
        while (delta > std::numbers::pi) {
            delta -= 2.0 * std::numbers::pi;
        }
        while (delta < -std::numbers::pi) {
            delta += 2.0 * std::numbers::pi;
        }
        const double c = std::cos(delta);
        const double s = std::sin(delta);
        // Turn the box about its own local Z: the axes rotate with it
        // (R' = R * Rz). The chunk transform is deliberately untouched.
        Eigen::Matrix3d turn = Eigen::Matrix3d::Identity();
        turn(0, 0) = c;
        turn(0, 1) = -s;
        turn(1, 0) = s;
        turn(1, 1) = c;
        state.region.rotation = drag.start.region.rotation * turn;
        return state;
    }
    const double parameter =
        line_parameter(drag.line_origin, drag.line_direction, ray);
    const double delta = parameter - drag.start_parameter;
    if (drag.handle.kind == GizmoHandleKind::Face) {
        if (drag.handle.mode == RegionGizmoMode::Move) {
            if (drag.handle.axis >= 0 && drag.handle.axis < 3) {
                // The line direction is +axis for both faces, so the centre
                // follows the cursor on a negative face too; re-signing here
                // would move it the wrong way. The +axis is the box's own
                // axis, so a rotated box moves along its face normal.
                state.region.centre +=
                    state.region.rotation.col(drag.handle.axis) * delta;
            }
        } else {
            resize_axis(state, drag.handle.axis, drag.handle.sign, delta);
        }
        return state;
    }
    const std::size_t count = std::min(drag.handle.axes.size(), drag.handle.signs.size());
    if (count == 0U) {
        return state;
    }
    const double per_axis = delta / std::sqrt(static_cast<double>(count));
    for (std::size_t i = 0; i < count; ++i) {
        resize_axis(state, drag.handle.axes[i], drag.handle.signs[i], per_axis);
    }
    return state;
}

bool drag_changed(const GizmoDrag& drag, const GizmoState& state) {
    if (!(state.region == drag.start.region)) {
        return true;
    }
    return (state.transform.array() != drag.start.transform.array()).any();
}

std::vector<SceneLineVertex> preview_lines(const GizmoState& state, RegionGizmoMode mode,
                                           const SceneStyle& style) {
    std::vector<SceneLineVertex> lines;
    const auto add = [&lines, &style](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
        lines.push_back({Eigen::Vector3f(static_cast<float>(a.x()), static_cast<float>(a.y()),
                                        static_cast<float>(a.z())),
                         style.region});
        lines.push_back({Eigen::Vector3f(static_cast<float>(b.x()), static_cast<float>(b.y()),
                                        static_cast<float>(b.z())),
                         style.region});
    };
    const std::array<Eigen::Vector3d, 8> corners = box_corners(state.region);
    // The twelve box edges: corners differ by one bit.
    for (int i = 0; i < 8; ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            if ((i & (1 << axis)) == 0) {
                add(corners[static_cast<std::size_t>(i)],
                    corners[static_cast<std::size_t>(i | (1 << axis))]);
            }
        }
    }
    const double span = std::max({state.region.size.x(), state.region.size.y(),
                                  state.region.size.z(), 1.0});
    const double tick = span * 0.03;
    if (mode == RegionGizmoMode::Resize) {
        for (const Eigen::Vector3d& corner : corners) {
            for (int axis = 0; axis < 3; ++axis) {
                Eigen::Vector3d axis_direction = Eigen::Vector3d::Zero();
                axis_direction[axis] = tick;
                const Eigen::Vector3d direction = state.region.rotation * axis_direction;
                add(corner - direction, corner + direction);
            }
        }
    } else if (mode == RegionGizmoMode::Rotate) {
        const double radius = std::max(state.region.size.x(), state.region.size.y()) * 0.7;
        constexpr int kSegments = 32;
        for (int i = 0; i < kSegments; ++i) {
            const double a = 2.0 * std::numbers::pi * static_cast<double>(i) /
                             static_cast<double>(kSegments);
            const double b = 2.0 * std::numbers::pi * static_cast<double>(i + 1) /
                             static_cast<double>(kSegments);
            add(from_region_local(state.region,
                                  Eigen::Vector3d(std::cos(a) * radius,
                                                  std::sin(a) * radius, 0.0)),
                from_region_local(state.region,
                                  Eigen::Vector3d(std::cos(b) * radius,
                                                  std::sin(b) * radius, 0.0)));
        }
    } else if (mode == RegionGizmoMode::Move) {
        const double reach = span * 0.18;
        for (int axis = 0; axis < 3; ++axis) {
            Eigen::Vector3d axis_direction = Eigen::Vector3d::Zero();
            axis_direction[axis] = reach;
            const Eigen::Vector3d direction = state.region.rotation * axis_direction;
            add(state.region.centre + direction * 0.5,
                state.region.centre + direction * 0.5 + direction);
            add(state.region.centre - direction * 0.5,
                state.region.centre - direction * 0.5 - direction);
        }
    }
    return lines;
}


}  // namespace alphashape::gui
