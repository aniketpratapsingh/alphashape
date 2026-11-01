#pragma once

/// @file project_markers.hpp
/// @brief Manual marker projection (WP-13.4).
///
/// A marker is a chunk-local 3-D point measured in one or more images. This
/// module answers the one question the GUI and the Python front end need when a
/// user places or inspects a marker: where does it land in a given camera?
///
/// The answer is deliberately "no pixel" rather than an exception or a
/// wraparound pixel when the point is not observable:
///   * the camera is not aligned (`Camera::enabled` false), or
///   * the point is at or behind the image plane (camera-frame z <= 0), or
///   * the lens model refuses the point, or
///   * the pixel is outside [0, w] x [0, h] and `clip_to_image` is set.
///
/// The pose convention matches the rest of the engine: `Camera::pose` maps a
/// chunk-local point into the camera frame (+X right, +Y down, +Z forward), and
/// the calibration then maps that camera-frame point to pixels.

#include <optional>

#include <Eigen/Core>

#include "common/core/marker.hpp"
#include "common/core/types.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {

/// Project a chunk-local point through one camera. Returns nullopt when the
/// point is not observable in that image (see the file comment).
///
/// `calibration` is the calibration of the camera's sensor, which the caller
/// resolves through `Camera::sensor_id`; it is passed explicitly so this
/// function stays free of the chunk container.
///
/// With `clip_to_image` false an off-image pixel is still returned, which the
/// UI uses to draw a marker just outside the frame; a non-finite pixel is never
/// returned.
std::optional<Eigen::Vector2d> project_marker(
    const Eigen::Vector3d& point_local, const Camera& camera,
    const Calibration& calibration, bool clip_to_image = true);

/// Convenience overload: project `marker.position`. A marker without an
/// estimated position has no projection (nullopt).
std::optional<Eigen::Vector2d> project_marker(
    const Marker& marker, const Camera& camera, const Calibration& calibration,
    bool clip_to_image = true);

}  // namespace alphashape
