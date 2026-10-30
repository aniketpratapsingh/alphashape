#pragma once

/// @file euler.hpp
/// @brief Tait-Bryan rotation angles and their matrix form (M12 §4.11, CRS-10).
///
/// Three angles describe a rotation in a fixed sequence of axes. The engine
/// implements the six **Tait-Bryan** sequences -- three rotations about three
/// *distinct* axes -- which is the set a yaw/pitch/roll reference uses. The
/// other six proper-Euler sequences (XZX, XYX, ... with a repeated outer axis)
/// are not part of this API: nothing in the data model spells an orientation
/// that way, and the repeated-axis gimbal geometry differs (the singular angle
/// is 0/180 rather than +/-90).
///
/// ## Convention (normative)
///
/// * **Intrinsic** and **active**: the sequence `ZYX` means "rotate about the
///   body z, then about the *new* y, then about the *new* x". The matrix is
///   `R = Rz(yaw) * Ry(pitch) * Rx(roll)` -- the same product read right to
///   left is the extrinsic sequence `XYZ` about fixed axes. Every order here
///   follows that rule: `R = R_axis0(angle0) * R_axis1(angle1) *
///   R_axis2(angle2)`, with `angle0/1/2` the struct's `yaw/pitch/roll` fields.
/// * **Fields are positional**: `yaw` is the angle about the FIRST axis of
///   `RotationOrder`, `pitch` the second, `roll` the third. For the default
///   `ZYX` that is the familiar yaw about up (z), pitch about y, roll about x.
///   For `XYZ`, `yaw` is the angle about x. Changing `order` reinterprets the
///   three numbers; it never permutes them behind the caller's back.
/// * **Right-handed, active** rotations about the axes x = 0, y = 1, z = 2, in
///   the local east/north/up frame the rest of the CRS engine uses (see
///   `CrsPoint`): a positive yaw about +z turns +y (north) toward -x (west), so
///   a positive `ZYX` yaw is west of north. A compass bearing (clockwise from
///   north) is therefore `yaw = -bearing`.
/// * **Degrees at the API.** `to_matrix`/`from_matrix` take and return degrees;
///   the header's one radian entry point is `EulerAngles::radians(order)`.
///
/// ## Gimbal lock (documented alternative solution)
///
/// For a Tait-Bryan sequence the second angle at +/-90 degrees aligns the first
/// and third axes: their rotations no longer act independently and only their
/// sum (or difference) is observable. `from_matrix` resolves this the same way
/// every order: it keeps the recovered middle angle at +/-90, sets `roll = 0`,
/// and puts the entire remaining rotation into `yaw`. The returned angles
/// therefore reproduce the input matrix exactly, but are not the only ones that
/// do. `to_matrix(from_matrix(R)) == R` still holds (to floating-point
/// precision), which is the invariant the round-trip test asserts.
///
/// `to_matrix`/`from_matrix` use only Eigen's rotation primitives, so they are
/// independent of PROJ; they live in `alphashape_crs` because the only consumer
/// (reference conversion, M13 import/export) is part of that layer.

#include <string>

#include <Eigen/Core>

namespace alphashape::crs {

/// The six Tait-Bryan sequences, named first-axis-second-axis-third-axis.
enum class RotationOrder {
    XYZ,
    XZY,
    YXZ,
    YZX,
    ZXY,
    ZYX,  ///< default: yaw about z, pitch about y, roll about x
};

/// "ZYX" for `RotationOrder::ZYX`; "" for an out-of-range value.
const char* rotation_order_name(RotationOrder order);

/// Parse "XYZ" (case-insensitive). Returns false and leaves `order` untouched
/// when `text` is not one of the six names; used for the chunk meta key, where
/// a corrupt value must be reported rather than silently defaulted.
bool parse_rotation_order(const std::string& text, RotationOrder* order);

/// All six orders in declaration order, for tests and UI lists.
const RotationOrder* all_rotation_orders();

/// One rotation in a Tait-Bryan sequence. See the file comment for the
/// convention; the default is a zero rotation in `ZYX`.
struct EulerAngles {
    double yaw = 0.0;    ///< degrees about the first axis of `order`
    double pitch = 0.0;  ///< degrees about the second axis
    double roll = 0.0;   ///< degrees about the third axis
    RotationOrder order = RotationOrder::ZYX;

    bool operator==(const EulerAngles& other) const {
        return yaw == other.yaw && pitch == other.pitch && roll == other.roll &&
               order == other.order;
    }

    /// The same angles converted to radians, order preserved.
    EulerAngles radians() const;
};

/// The rotation matrix of `angles`: the intrinsic product in `angles.order`.
/// Orthonormal to rounding; exact for multiples of 90 degrees.
Eigen::Matrix3d to_matrix(const EulerAngles& angles);

/// The Tait-Bryan angles of the rotation `matrix` in `order`. The gimbal-lock
/// branch sets `roll = 0` (see the file comment). Throws `ValueError` when
/// `matrix` is not a rotation (not finite, or not orthonormal to a 1e-6
/// relative tolerance), because a silent square-root of a negative value would
/// return NaN angles.
EulerAngles from_matrix(const Eigen::Matrix3d& matrix, RotationOrder order);

}  // namespace alphashape::crs
