#pragma once

// reference_orientation.hpp -- measured world->camera rotation of a referenced
// camera in the chunk's internal frame (WP-13.21).
//
// A reference record holds the antenna (phase-centre) ECEF position and, when
// the source carries an attitude log, a body orientation triple in degrees.
// Turning that into a chunk-local world->camera rotation takes four frames:
//
//   internal --> ECEF      chunk transform (uniform similarity; the scale is
//                          divided out of the 3x3 block)
//   ECEF     --> ENU       local frame at the reference position
//   ENU      --> body      R_wb from the stored triple and the chunk's
//                          rotation order, with the north-zero convention
//   body     --> camera    fixed mount: nadir, or the effective master
//                          sensor's antenna boresight when one is set
//
// The result is  R_meas = R_bc * R_wb^T * R_enu_from_ecef * R_ecef_from_internal.
//
// This is shared by OptimizeCameras (as the SO(3) prior) and AlignFromReference
// (which places the camera pose directly). The construction is kept in one
// place so the two can never disagree.

#include <optional>

#include <Eigen/Core>

#include "common/core/chunk.hpp"
#include "common/core/types.hpp"

namespace alphashape::reference {

struct ReferenceRotation {
    /// Measured world(camera) -> camera rotation in the chunk's internal frame.
    Eigen::Matrix3d world_to_camera = Eigen::Matrix3d::Identity();

    /// The heading frame the body triple defines, expressed in the internal
    /// frame (`internal_from_ecef * ecef_from_enu * frame_enu`). Bundle
    /// adjustment uses it to keep the yaw/roll parametrisation well posed.
    Eigen::Matrix3d heading_frame = Eigen::Matrix3d::Identity();

    /// True when body forward is vertical, so yaw and roll about the heading
    /// frame are not separable from the attitude triple alone.
    bool degenerate_heading = false;
};

/// Rotation carried by a 3x3 block, with a uniform similarity scale divided
/// out. Throws ValueError when the block carries no valid positive scale.
Eigen::Matrix3d rotation_of_similarity(const Eigen::Matrix3d& matrix);

/// The measured rotation of `camera` in the internal frame, or nullopt when the
/// camera has no reference position and no usable orientation (a position at
/// least supplies the ENU origin). `camera` must belong to `chunk`; the caller
/// has already checked that the reference is enabled.
std::optional<ReferenceRotation> measured_rotation(const Chunk& chunk,
                                                   const Camera& camera,
                                                   const Eigen::Matrix4d& internal_to_ecef);

/// The effective master sensor's antenna for `camera` (a sensor with a master
/// resolves its master's mount; the mount is part of the physical body).
const Antenna& effective_antenna(const Chunk& chunk, const Camera& camera);

}  // namespace alphashape::reference
