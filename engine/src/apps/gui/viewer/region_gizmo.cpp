#include "apps/gui/viewer/region_gizmo.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include <Eigen/LU>

namespace alphashape::gui {

namespace {

constexpr double kMinRegionSize = 1e-9;
constexpr double kPickRadiusPixels = 12.0;
constexpr int kRingSegments = 64;

Eigen::Vector3d normalised(const Eigen::Vector3d& value) {
    const double length = value.norm();
    if (!(length > 0.0)) {
        return Eigen::Vector3d::UnitZ();
    }
    return value / length;
}

std::array<Eigen::Vector3d, 8> box_corners(const Region& region) {
    std::array<Eigen::Vector3d, 8> corners{};
    const Eigen::Vector3d half = region.size * 0.5;
    for (int i = 0; i < 8; ++i) {
        corners[static_cast<std::size_t>(i)] =
            region.centre +
            Eigen::Vector3d((i & 1) != 0 ? half.x() : -half.x(),
                            (i & 2) != 0 ? half.y() : -half.y(),
                            (i & 4) != 0 ? half.z() : -half.z());
    }
    return corners;
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
    const Eigen::Vector4d clip = view_projection * Eigen::Vector4d(world.x(), world.y(), world.z(), 1.0);
    if (!(clip.w() > 1e-12)) {
        return std::nullopt;
    }
    const Eigen::Vector3d ndc = clip.head<3>() / clip.w();
    return Eigen::Vector2d((ndc.x() * 0.5 + 0.5) * static_cast<double>(width),
                           (0.5 - ndc.y() * 0.5) * static_cast<double>(height));
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
        const Eigen::Vector4d p = inverse * Eigen::Vector4d(ndc_x, ndc_y, z, 1.0);
        return p.head<3>() / p.w();
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
            const double angle = 2.0 * M_PI * static_cast<double>(i) /
                                 static_cast<double>(kRingSegments);
            const Eigen::Vector3d point =
                region.centre + Eigen::Vector3d(std::cos(angle) * radius,
                                                std::sin(angle) * radius, 0.0);
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
        const std::optional<BoxHit> hit = ray_box(state.region, ray);
        if (!hit) {
            return std::nullopt;
        }
        const bool inside = hit->entry < 0.0;
        GizmoHandle handle;
        handle.mode = RegionGizmoMode::Move;
        handle.kind = GizmoHandleKind::Face;
        handle.axis = inside ? hit->exit_axis : hit->entry_axis;
        handle.sign = inside ? hit->exit_sign : hit->entry_sign;
        handle.point = ray.origin + normalised(ray.direction) * (inside ? hit->exit : hit->entry);
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
        const std::optional<Eigen::Vector3d> hit = plane_hit(state.region.centre.z(), ray);
        if (!hit) {
            return std::nullopt;
        }
        drag.start_angle =
            std::atan2(hit->y() - state.region.centre.y(), hit->x() - state.region.centre.x());
        drag.line_origin = state.region.centre;
        drag.line_direction = Eigen::Vector3d::UnitZ();
        return drag;
    }
    if (handle.kind == GizmoHandleKind::Face) {
        Eigen::Vector3d axis = Eigen::Vector3d::Zero();
        if (handle.axis >= 0 && handle.axis < 3) {
            axis[handle.axis] = 1.0;
        }
        drag.line_origin = state.region.centre;
        drag.line_direction = axis;
        drag.start_parameter = line_parameter(drag.line_origin, axis, ray);
        return drag;
    }
    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
    const std::size_t count = std::min(handle.axes.size(), handle.signs.size());
    for (std::size_t i = 0; i < count; ++i) {
        const int axis = handle.axes[i];
        if (axis >= 0 && axis < 3) {
            direction[axis] += static_cast<double>(handle.signs[i]);
        }
    }
    drag.line_origin = handle.point;
    drag.line_direction = normalised(direction);
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
    const double opposite_face = state.region.centre[axis] - static_cast<double>(sign) *
                                                              old_size * 0.5;
    const double new_size = std::max(
        kMinRegionSize, old_size + static_cast<double>(sign) * movement);
    state.region.size[axis] = new_size;
    state.region.centre[axis] =
        opposite_face + static_cast<double>(sign) * new_size * 0.5;
}

}  // namespace

GizmoState update_drag(const GizmoDrag& drag, const GizmoRay& ray) {
    GizmoState state = drag.start;
    if (drag.handle.kind == GizmoHandleKind::Ring) {
        const std::optional<Eigen::Vector3d> hit = plane_hit(state.region.centre.z(), ray);
        if (!hit) {
            return state;
        }
        const double angle =
            std::atan2(hit->y() - state.region.centre.y(), hit->x() - state.region.centre.x());
        double delta = angle - drag.start_angle;
        constexpr double kPi = 3.14159265358979323846;
        while (delta > kPi) {
            delta -= 2.0 * kPi;
        }
        while (delta < -kPi) {
            delta += 2.0 * kPi;
        }
        const double c = std::cos(delta);
        const double s = std::sin(delta);
        const Eigen::Vector3d& centre = state.region.centre;
        Eigen::Matrix4d about = Eigen::Matrix4d::Identity();
        about(0, 0) = c;
        about(0, 1) = -s;
        about(1, 0) = s;
        about(1, 1) = c;
        about(0, 3) = centre.x() - c * centre.x() + s * centre.y();
        about(1, 3) = centre.y() - s * centre.x() - c * centre.y();
        state.transform = drag.start.transform * about;
        return state;
    }
    const double parameter =
        line_parameter(drag.line_origin, drag.line_direction, ray);
    const double delta = parameter - drag.start_parameter;
    if (drag.handle.kind == GizmoHandleKind::Face) {
        if (drag.handle.mode == RegionGizmoMode::Move) {
            if (drag.handle.axis >= 0 && drag.handle.axis < 3) {
                state.region.centre[drag.handle.axis] +=
                    static_cast<double>(drag.handle.sign) * delta;
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
                Eigen::Vector3d direction = Eigen::Vector3d::Zero();
                direction[axis] = tick;
                add(corner - direction, corner + direction);
            }
        }
    } else if (mode == RegionGizmoMode::Rotate) {
        const double radius = std::max(state.region.size.x(), state.region.size.y()) * 0.7;
        constexpr int kSegments = 32;
        for (int i = 0; i < kSegments; ++i) {
            const double a = 2.0 * 3.14159265358979323846 * static_cast<double>(i) /
                             static_cast<double>(kSegments);
            const double b = 2.0 * 3.14159265358979323846 * static_cast<double>(i + 1) /
                             static_cast<double>(kSegments);
            add(state.region.centre + Eigen::Vector3d(std::cos(a) * radius, std::sin(a) * radius,
                                                      0.0),
                state.region.centre + Eigen::Vector3d(std::cos(b) * radius, std::sin(b) * radius,
                                                      0.0));
        }
    } else if (mode == RegionGizmoMode::Move) {
        const double reach = span * 0.18;
        for (int axis = 0; axis < 3; ++axis) {
            Eigen::Vector3d direction = Eigen::Vector3d::Zero();
            direction[axis] = reach;
            add(state.region.centre + direction * 0.5,
                state.region.centre + direction * 0.5 + direction);
            add(state.region.centre - direction * 0.5,
                state.region.centre - direction * 0.5 - direction);
        }
    }
    return lines;
}


}  // namespace alphashape::gui
