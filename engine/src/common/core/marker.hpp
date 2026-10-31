#pragma once

// Markers: user-placed or imported points that are measurable in images and
// (optionally) carry a geodetic reference. They are the input to georeferencing
// optimisation and the anchors for scale bars (WP-11.1, M11 §4.1).
//
// Every marker belongs to a chunk and carries a chunk-local id allocated by
// `Chunk::allocate_id()`. Ids are never reused.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "common/core/types.hpp"

namespace alphashape {

/// One image measurement of a marker: the pixel it was clicked at in a camera.
/// `camera_id` indexes `Chunk::cameras()`.
struct MarkerProjection {
    std::int32_t camera_id = -1;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
    bool enabled = true;

    bool operator==(const MarkerProjection&) const = default;
};

/// Marker role. Regular markers are ordinary control points; vertex markers are
/// body/tie points of a fiducial target; fiducial markers are target centres.
enum class MarkerType { Regular, Vertex, Fiducial };

/// A marker with its per-camera projections and its (optional) reference.
struct Marker {
    std::int32_t id = -1;
    std::string label;
    std::int32_t group_id = -1;
    MarkerType type = MarkerType::Regular;

    /// Estimates in image space, sorted by camera_id and unique per camera.
    std::vector<MarkerProjection> projections;

    /// Reference position as (lon_deg, lat_deg, alt_m), if the marker is a
    /// control point. `reference_accuracy` is the 1-sigma per-axis accuracy in
    /// the same units (metres for alt, metres for the projected lon/lat).
    std::optional<Eigen::Vector3d> reference;
    Eigen::Vector3d reference_accuracy = Eigen::Vector3d::Constant(0.5);

    /// Estimated position in chunk-local coordinates, once adjusted.
    std::optional<Eigen::Vector3d> position;
    Eigen::Matrix3d position_covariance = Eigen::Matrix3d::Zero();

    bool pinned = false;    ///< excluded from adjustment (stationary control)
    bool valid = false;     ///< has enough projections / passed filtering
    bool internal = false;  ///< created by the engine, not the user

    /// Free-form marker properties. WP-13.1 stores the reference role under the
    /// key "type": "gcp" (control point) or "check_point" (held out of the
    /// adjustment). Absent means "gcp" when a reference exists.
    Meta meta;

    bool operator==(const Marker& other) const {
        if (id != other.id || label != other.label || group_id != other.group_id ||
            type != other.type || projections != other.projections ||
            !(meta == other.meta)) {
            return false;
        }
        if (reference.has_value() != other.reference.has_value() ||
            position.has_value() != other.position.has_value()) {
            return false;
        }
        if (reference.has_value() && !(*reference == *other.reference)) {
            return false;
        }
        if (position.has_value() && !(*position == *other.position)) {
            return false;
        }
        if (!(reference_accuracy == other.reference_accuracy)) {
            return false;
        }
        if (!(position_covariance == other.position_covariance)) {
            return false;
        }
        return pinned == other.pinned && valid == other.valid &&
               internal == other.internal;
    }
};

}  // namespace alphashape
