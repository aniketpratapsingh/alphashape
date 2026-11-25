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
///   * a Frame point is at or behind the image plane (camera-frame z <= 0;
///     a fisheye sees the whole sphere, so it has no such cut), or
///   * the lens model refuses the point, or
///   * the pixel is outside [0, w] x [0, h] and `clip_to_image` is set.
///
/// The pose convention matches the rest of the engine: `Camera::pose` maps a
/// chunk-local point into the camera frame (+X right, +Y down, +Z forward), and
/// the calibration then maps that camera-frame point to pixels.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "common/core/marker.hpp"
#include "common/core/types.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {

class Chunk;

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

/// The chunk-local point a marker's surveyed reference implies, for a
/// georeferenced chunk. This is what lets an imported control point -- which
/// has a reference but no estimated `position` until it is adjusted -- be
/// predicted on a photo.
///
/// The reference is read through `reference::geodetic_marker_references`, so a
/// marker slot in any CRS (a projected grid such as an MGA zone, a geographic
/// one) is converted to lon/lat/ellipsoidal height first. That geodetic point
/// becomes ECEF, and the inverse of `chunk.transform()` (which maps local ->
/// ECEF) carries it into the chunk's local frame.
///
/// Returns nullopt when the chunk is not georeferenced (an identity transform,
/// or a CRS that is unset/local -- the transform alone yields ECEF metres, not
/// a chunk frame), when the marker has no usable reference, or when the
/// reference cannot be converted. Never throws: a conversion failure is "no
/// prediction", not an error the UI has to handle.
///
/// @note This is the fallback for a marker without `position`; a marker that
///       has one should be projected from it, because the adjusted position is
///       the better estimate.
std::optional<Eigen::Vector3d> marker_local_from_reference(const Chunk& chunk,
                                                           std::size_t marker_index);

/// The same for every marker, computed in one pass (the geodetic conversion
/// reads the whole list at once). Entries follow `chunk.markers()`; nullopt
/// where `marker_local_from_reference` would return nullopt.
std::vector<std::optional<Eigen::Vector3d>> marker_locals_from_references(
    const Chunk& chunk);

/// One camera that sees a marker, with the pixel it lands on.
struct MarkerCameraHit {
    std::int32_t camera_id = -1;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
    /// Distance of `pixel` from the image centre, in pixels. The list is sorted
    /// by this, nearest first, so the photo where the marker is most central
    /// (and easiest to find) comes first.
    double distance_from_centre = 0.0;
};

/// The cameras of `chunk` in which `marker_index` lands inside the image, for
/// the "show photos with this marker" action. A camera counts when the marker
/// has a placed projection there, or when it projects from the marker's
/// estimated position, or -- for a georeferenced chunk -- from its surveyed
/// reference (see `marker_local_from_reference`). Only in-image pixels count
/// (`clip_to_image` true). Sorted by `distance_from_centre`, nearest first.
///
/// Returns an empty list for an out-of-range index or a marker with no
/// position and no usable reference. Never throws.
std::vector<MarkerCameraHit> cameras_seeing_marker(const Chunk& chunk,
                                                   std::size_t marker_index);

}  // namespace alphashape
