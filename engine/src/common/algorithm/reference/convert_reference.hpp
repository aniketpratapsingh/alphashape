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
/// * **Orientation angles** use an explicit contract, not an "ENU-aligned"
///   hand-wave. The camera's body frame is x forward, y left, z up (FLU). Yaw,
///   pitch and roll describe that frame relative to the slot CRS's local
///   east-north-up basis: yaw is the rotation about up, counter-clockwise from
///   north (towards east); pitch is about the left axis, positive nose up; roll
///   is about the forward axis, positive left wing up. They compose in the
///   chunk's rotation order (`rotation_order_of`). Because yaw is
///   counter-clockwise from north, a compass heading -- clockwise from true
///   north, as in a flight log's yaw-degree field -- maps to `yaw = -heading`
///   (heading 90 degrees east is yaw -90 degrees).
///
///   Because ENU's x axis is east while a zero yaw points north, the zero-yaw
///   body frame is x = north, y = west, z = up, not the ENU basis itself. The
///   body-to-world rotation is therefore
///
///       R_wb = Rz(90 deg) * Rz(yaw) * Ry(-pitch) * Rx(roll)
///            = Rz(yaw + 90 deg) * Ry(-pitch) * Rx(roll)
///
///   for the default `ZYX` order; heading east (yaw = -90 deg) is the
///   identity: body x = east, y = north, z = up. In the `crs::EulerAngles`
///   representation the same matrix is `Rz(90 deg) *
///   crs::to_matrix({yaw, -pitch, roll})`, since `to_matrix` composes
///   `Rz(yaw) * Ry(pitch) * Rx(roll)` and takes the angles in the ENU basis.
///
///   `body_to_camera()` and `body_to_camera_nadir()` convert that FLU body
///   frame into the engine camera frame (x right, y down, z forward; see
///   `model/camera/calibration.hpp`) for a level and a straight-down camera.
///   A camera-to-world rotation is `R_wb * body_to_camera().transpose()`.
///
///   `convert_orientation` re-expresses an orientation in the target's axes;
///   see its own comment for the exact rule. The current data model has
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

/// The fixed rotation from the FLU body frame to the engine camera frame
/// (+X right, +Y down, +Z forward; `model/camera/calibration.hpp:8`) for a
/// camera whose optical axis is body +x, i.e. a level, forward-looking camera.
///
/// Derived from the two frames (`v_camera = M * v_body`, rows = camera axes in
/// body coordinates): the boresight camera +Z is body x, so row 2 is (1, 0, 0);
/// image down camera +Y is body -z, so row 1 is (0, 0, -1); image right camera
/// +X is then `row1 x row2` = (0, -1, 0), i.e. body -y. Hence
///
///     body_to_camera() = | 0 -1  0 |
///                        | 0  0 -1 |
///                        | 1  0  0 |
///
/// @see body_to_camera_nadir for the straight-down camera.
Eigen::Matrix3d body_to_camera();

/// `body_to_camera()` for a nadir (pitch -90 degrees) camera:
/// `body_to_camera() * Ry(-90 deg)`. The boresight is body -z (down), image
/// down camera +Y is body -x (aft), image right camera +X is body -y.
Eigen::Matrix3d body_to_camera_nadir();

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
/// CRS (a project geoid). The route follows the datum's description, never what
/// the store happens to hold: an EPSG vertical CRS always goes through PROJ's
/// `VerticalTransform::compound` pipeline (its exact `cs2cs` answer, and a
/// store grid never shadows it); a datum with no EPSG code uses the store's
/// grid through `VerticalTransform::custom`. A geoid datum with no store, or an
/// id the store does not carry, throws naming the datum. The store id is the
/// datum's `grid` when set, otherwise its `name`.
///
/// A custom grid's undulation N is evaluated in the geoid's own horizontal
/// reference, taken from the datum's `horizontal_epsg` (persisted in doc.xml);
/// when that is unset it defaults to EPSG:4979, the frame the global
/// EGM96/EGM2008 grids are referred to.
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
