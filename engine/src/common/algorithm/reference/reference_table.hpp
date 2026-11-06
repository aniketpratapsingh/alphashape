#pragma once

/// @file reference_table.hpp
/// @brief The chunk's reference table (WP-13.1, M13).
///
/// One row per camera, per marker and per check point, carrying the measured
/// (source) position, the adjusted (estimated) position, their difference, the
/// per-row sigma and the capture distance. Both positions are expressed in a
/// display CRS; an unreferenced chunk stays in chunk-local units, and errors
/// are reported only when both ends exist.
///
/// Capture distance (metres of chunk-local geometry):
///   * camera row -- distance from the camera centre to the mean estimated
///     position of the markers it observes; NaN when it observes none.
///   * marker row -- mean distance from the cameras that observe it to the
///     marker's estimated position; NaN when it has no estimate or observer.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "common/core/chunk.hpp"
#include "common/core/reference_settings.hpp"
#include "model/crs/coordinate_system.hpp"

namespace alphashape::reference {

enum class ReferenceRowKind { Camera, Marker, CheckPoint };

struct ReferenceRow {
    std::int32_t id = -1;
    std::string label;
    ReferenceRowKind kind = ReferenceRowKind::Camera;
    bool enabled = true;
    std::optional<Eigen::Vector3d> source;
    std::optional<Eigen::Vector3d> estimated;
    /// estimated - source. When the display CRS is geographic this is the
    /// difference in local east/north/up metres rather than degrees.
    std::optional<Eigen::Vector3d> error;
    /// |error| in metres, or nullopt when either end is missing.
    std::optional<double> error_m;
    double sigma = 0.0;
    /// Per-axis accuracies (east, north, up) for `source`. Falls back to
    /// `sigma` on an axis whose stored accuracy is not usable.
    Eigen::Vector3d sigma_xyz = Eigen::Vector3d::Constant(0.0);
    double capture_distance_m = std::numeric_limits<double>::quiet_NaN();
    /// Camera projections of a marker row (0 for a camera).
    std::size_t projection_count = 0;
    /// Marker placed interactively by the user (markers only).
    bool pinned = false;
};

/// RMSE over the enabled rows that carry both ends of an error.
struct ReferenceSummary {
    std::size_t camera_count = 0;
    std::size_t marker_count = 0;
    double camera_rmse_m = std::numeric_limits<double>::quiet_NaN();
    double marker_rmse_m = std::numeric_limits<double>::quiet_NaN();
};

struct ReferenceTable {
    std::vector<ReferenceRow> rows;
    bool georeferenced = false;
    /// Markers that are control points (role "gcp" with a reference source);
    /// check points and markers without a source are excluded.
    std::size_t control_count() const;
    ReferenceSummary summary() const;
};

/// Marker reference role, stored under `Marker::meta` key "type".
enum class MarkerReferenceRole { Gcp, CheckPoint };
MarkerReferenceRole marker_role(const Marker& marker);
void set_marker_role(Marker& marker, MarkerReferenceRole role);
const char* marker_reference_role_name(MarkerReferenceRole role);

/// Build the table. An unset/local `display_crs`, or an unreferenced chunk,
/// keeps coordinates in chunk-local units; otherwise sources are converted from
/// the camera/marker slot CRS and estimates from chunk-local through the chunk
/// transform and the CRS engine.
ReferenceTable build_reference_table(const Chunk& chunk,
                                     const crs::CoordinateSystem& display_crs);

}  // namespace alphashape::reference
