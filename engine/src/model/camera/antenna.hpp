#pragma once

/// @file antenna.hpp
/// @brief GNSS antenna mounting relative to a camera (WP-13.19).
///
/// A camera's GNSS reference records the position of the *antenna phase
/// centre*, not the camera's projection centre. The rigid offset between the
/// two -- the lever arm -- must be applied before a measured reference can
/// constrain a camera centre in bundle adjustment or georeferencing.
///
/// ## Frames (normative)
///
/// `offset_m` is the antenna phase centre relative to the projection centre,
/// in metres, in the CAMERA frame: x right, y down, z forward -- the same
/// frame as a world-to-camera point `X_c = R X_w + t`.
///
/// When `boresight_deg` is set the offset is instead expressed in a body
/// (IMU) frame. The rotation that maps body -> camera is
/// `to_matrix(EulerAngles{boresight_deg->x(), boresight_deg->y(),
/// boresight_deg->z()})`, which uses the EulerAngles default rotation order
/// `ZYX` (intrinsic, active: `R = Rz(yaw) * Ry(pitch) * Rx(roll)`). The three
/// components are positional: x = first axis, y = second, z = third. The
/// camera-frame offset is therefore `R_b * offset_m`.
///
/// With the world->camera rotation `R` of the camera, the predicted antenna
/// position in the world/local frame is
///
///     A = C + R^T * (R_b * offset_m),      C = -R^T t
///
/// where `R^T` maps camera -> world. The boresight also matters for attitude
/// (orientation) references, which the engine does not consume yet; this type
/// records it so a future adjustment can use it without a format change.
///
/// An antenna with a zero offset and no boresight is "unset": it contributes
/// nothing and a project containing it serialises exactly like one written
/// before the field existed.

#include <optional>

#include <Eigen/Core>

#include "model/crs/euler.hpp"

namespace alphashape {

struct Antenna {
    /// Antenna phase centre relative to the projection centre, in the camera
    /// frame -- or, when `boresight_deg` is set, in the body frame. Metres.
    Eigen::Vector3d offset_m = Eigen::Vector3d::Zero();

    /// Body -> camera rotation as Tait-Bryan angles in degrees (default order
    /// `ZYX`). nullopt means `offset_m` is already in the camera frame.
    std::optional<Eigen::Vector3d> boresight_deg;

    /// True when a non-zero offset is recorded. A boresight alone does not
    /// move the antenna, but it is still persisted (see operator==).
    bool is_set() const { return offset_m != Eigen::Vector3d::Zero; }

    /// Body -> camera rotation: `R_b` from `boresight_deg`, or identity when
    /// the offset is already expressed in the camera frame.
    Eigen::Matrix3d body_to_camera() const {
        if (!boresight_deg.has_value()) {
            return Eigen::Matrix3d::Identity();
        }
        const Eigen::Vector3d& a = *boresight_deg;
        const crs::EulerAngles angles{a.x(), a.y(), a.z()};
        return crs::to_matrix(angles);
    }

    /// The offset in the camera frame: `R_b * offset_m`.
    Eigen::Vector3d camera_offset() const { return body_to_camera() * offset_m; }

    bool operator==(const Antenna& other) const {
        return offset_m == other.offset_m && boresight_deg == other.boresight_deg;
    }
    bool operator!=(const Antenna& other) const { return !(*this == other); }
};

}  // namespace alphashape
