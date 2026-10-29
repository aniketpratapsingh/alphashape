#pragma once

/// @file convert_reference.hpp
/// @brief Convert camera/marker reference data into a target CRS (WP-12.7, MK-17).
///
/// A chunk stores reference values (a camera's GNSS position and accuracy, a
/// marker's surveyed position and accuracy) in whichever CRS the slot named by
/// `camera_crs()`/`marker_crs()` describes. This module rewrites those values
/// into a different CRS -- the explicit step WP-12.6 left to a separate call --
/// and relabels the slot. It converts only reference data: camera poses, tie
/// points and assets stay in `world_crs` and are untouched.
///
/// ## What each component does
///
/// * **Horizontal position** goes through `CrsTransform`: every point in this
///   API is in visualisation order, x = lon/east and y = lat/north, so a
///   lon/lat or easting/northing pair needs no per-call axis handling.
/// * **Height** goes through the vertical CRS when either side carries one
///   (`VerticalTransform`): the source slot's `VerticalCrs` is the datum the
///   stored heights are in, the target's is the one they must end in. With no
///   datum on either side heights are ellipsoidal and travel with the
///   horizontal transform, exactly as `CrsTransform` handles them.
/// * **Horizontal accuracy** is a 1-sigma per-axis figure stored as a ground
///   distance in metres (`CameraReference::accuracy`,
///   `Marker::reference_accuracy`), even when the slot is geographic and the
///   *coordinates* are degrees. It is a weight, not a coordinate, and a datum
///   shift does not change how well a point was measured, so it is carried
///   unchanged. No unit conversion is ever applied to it.
/// * **Orientation angles** are a camera-to-world rotation in the slot CRS's
///   local east/north/up axes (`TrajectoryPoint::orientation` documents the
///   same convention). `convert_orientation` re-expresses one in the target's
///   axes; see its own comment for the exact rule. The current data model has
///   no per-camera orientation field (M13 adds `orientation_deg` to
///   `CameraReference`), so `convert_reference` converts positions only until
///   that field exists; the orientation entry point is already public and is
///   what the reference importer will call.
///
/// ## Failures
///
/// The conversion is all-or-nothing: a position the target CRS cannot express
/// (outside a projection's domain, a height outside a geoid's coverage) throws
/// `Error` naming the object, and the chunk is left untouched. A reference
/// with a non-finite latitude or longitude is a data error and throws
/// `ValueError`; a NaN altitude is a real state (GNSS reported no height) and
/// is preserved -- the horizontal conversion runs with a placeholder height and
/// the NaN is put back.
///
/// The chunk edit is undoable through `make_convert_reference_command`.

#include <cstdint>
#include <memory>
#include <optional>

#include <Eigen/Core>

#include "common/core/chunk.hpp"
#include "model/crs/coordinate_system.hpp"
#include "model/crs/euler.hpp"
#include "model/crs/geoid.hpp"
#include "model/crs/transform.hpp"

namespace alphashape {

class AbstractCommand;

namespace reference {

/// Which references a conversion touches.
enum class ReferenceKind {
    Cameras,
    Markers,
    Both,
};

/// Chunk meta key naming the rotation order of orientation angles. The value
/// is one of the six Tait-Bryan names; absent means `ZYX`.
inline constexpr const char* kRotationOrderKey = "reference/rotation_order";

/// The chunk's configured rotation order, `ZYX` when the meta key is absent.
/// @throws ValueError when the key holds a name that is not a rotation order.
crs::RotationOrder rotation_order_of(const Chunk& chunk);

/// The rotation that re-expresses a world-frame direction from `source`'s local
/// axes in `target`'s local axes, at `point` (given in `source`).
///
/// Both frames are the CRS's own x/y/z axes (east/north/up for a projected or
/// geographic CRS), so for two UTM zones the result is the convergence plus the
/// grid rotation, and for geographic -> UTM it is exactly the meridian
/// convergence `gamma` about up. The matrix is derived by finite-differencing
/// each CRS's mapping to ECEF (central differences), so no per-projection
/// closed form is needed and it is correct for a geocentric CRS too.
///
/// @throws ValueError when a CRS is unset or the local engineering frame.
/// @throws Error when the point cannot be mapped into ECEF by either CRS.
Eigen::Matrix3d frame_rotation(const crs::CoordinateSystem& source,
                               const crs::CoordinateSystem& target,
                               const crs::CrsPoint& point);

/// Re-express a camera-to-world orientation in `target`'s local axes.
///
/// The physical orientation is unchanged; only the axes it is written in are.
/// With `R` the camera-to-world rotation in `source`'s axes and `G` the
/// `frame_rotation` above, the result is `G * R`: the camera's body frame is
/// untouched and the world frame is relabelled, which adds the meridian
/// convergence to the heading rather than rotating the camera relative to the
/// ground. (Conjugating instead would leave a yaw-about-up unchanged and put
/// the convergence into roll, which is not what a reference importer means.)
///
/// The angles keep their order; only what the matrix means changes.
crs::EulerAngles convert_orientation(const crs::EulerAngles& angles,
                                     const crs::CoordinateSystem& source,
                                     const crs::CoordinateSystem& target,
                                     const crs::CrsPoint& point);

/// Convert camera and/or marker references of `chunk` into `target`.
///
/// `geoids` supplies the grid for a `VerticalCrs` that is not an EPSG vertical
/// CRS (a project geoid). An EPSG vertical datum is resolved by PROJ through
/// `VerticalTransform::compound` and needs no store; a geoid datum with no
/// store, or an id the store does not carry, throws naming the datum. The store
/// id is the datum's `grid` when set, otherwise its `name`.
///
/// On success the camera/marker slot is set to `target`, so the stored numbers
/// and their label agree. On failure the chunk is unchanged.
///
/// @throws ValueError when a source slot is unset/local or the chunk's
///         rotation-order meta is malformed.
/// @throws Error when a transform cannot be built or a point cannot be
///         converted.
void convert_reference(Chunk& chunk, const crs::CoordinateSystem& target,
                       ReferenceKind kind, const crs::GeoidStore* geoids = nullptr);

/// The undoable form of `convert_reference`: one undo entry that restores every
/// camera reference, marker reference and both CRS slots.
///
/// `geoids` (if any) must outlive the command, i.e. for as long as it can be
/// re-applied from the undo stack.
std::unique_ptr<AbstractCommand> make_convert_reference_command(
    std::int32_t chunk_id, crs::CoordinateSystem target, ReferenceKind kind,
    const crs::GeoidStore* geoids = nullptr);

/// Human-readable label for a kind, used in the command name ("Convert Camera
/// References").
const char* reference_kind_name(ReferenceKind kind);

}  // namespace reference
}  // namespace alphashape
