#include "common/algorithm/reference/reference_orientation.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "common/algorithm/reference/convert_reference.hpp"
#include "common/algorithm/sfm/bundle_adjustment_detail.hpp"
#include "model/camera/antenna.hpp"
#include "model/crs/euler.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape::reference {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;

/// Rz(angle) in the ENU / body axes.
Eigen::Matrix3d rotation_about_z(double degrees) {
    const double angle = degrees * kDegToRad;
    Eigen::Matrix3d m;
    m << std::cos(angle), -std::sin(angle), 0.0,  //
        std::sin(angle), std::cos(angle), 0.0,     //
        0.0, 0.0, 1.0;
    return m;
}

/// Body-to-ENU rotation for a stored orientation triple, in `order`:
/// `R_wb = Rz(90 deg) * to_matrix({x, -y, z})`. The Rz(90) turns the
/// north-pointing zero-heading body frame into the ENU basis; the second
/// component is negated because a positive stored pitch is nose up while
/// `to_matrix` applies a positive rotation about its second axis (see
/// convert_reference.hpp).
Eigen::Matrix3d world_from_body(const Eigen::Vector3d& angles, crs::RotationOrder order) {
    const crs::EulerAngles internal{angles.x(), -angles.y(), angles.z(), order};
    return rotation_about_z(90.0) * crs::to_matrix(internal);
}

}  // namespace

Eigen::Matrix3d rotation_of_similarity(const Eigen::Matrix3d& matrix) {
    const double determinant = matrix.determinant();
    if (!std::isfinite(determinant) || !(determinant > 0.0)) {
        throw ValueError("chunk transform carries no valid rotation");
    }
    return matrix / std::cbrt(determinant);
}

const Antenna& effective_antenna(const Chunk& chunk, const Camera& camera) {
    const std::int32_t effective =
        effective_sensor_id(chunk.sensors(), camera.sensor_id);
    return chunk.sensors()[static_cast<std::size_t>(effective)].antenna;
}

std::optional<ReferenceRotation> measured_rotation(const Chunk& chunk,
                                                   const Camera& camera,
                                                   const Eigen::Matrix4d& internal_to_ecef) {
    if (!camera.reference.orientation_deg.has_value()) {
        return std::nullopt;
    }
    // The ENU frame is taken at the reference position; without one, at the
    // camera's current estimated centre converted to geodetic. The difference
    // only changes the local meridian convergence over the block.
    geo::Geodetic geodetic;
    if (camera.reference.location.has_value()) {
        const Eigen::Vector3d& loc = *camera.reference.location;
        geodetic = geo::Geodetic{loc.x(), loc.y(), std::isfinite(loc.z()) ? loc.z() : 0.0};
    } else {
        const Eigen::Matrix3d r = camera.pose.block<3, 3>(0, 0);
        const Eigen::Vector3d t = camera.pose.block<3, 1>(0, 3);
        const Eigen::Vector3d centre = -(r.inverse() * t);
        geodetic = geo::ecef_to_geodetic(internal_to_ecef.block<3, 3>(0, 0) * centre +
                                        internal_to_ecef.block<3, 1>(0, 3));
    }

    const Eigen::Matrix3d ecef_from_internal =
        rotation_of_similarity(internal_to_ecef.block<3, 3>(0, 0));
    const Eigen::Matrix3d ecef_from_enu = geo::enu_to_ecef(geodetic).block<3, 3>(0, 0);
    const Eigen::Matrix3d r_wb = world_from_body(
        *camera.reference.orientation_deg, reference::rotation_order_of(chunk));

    const Antenna& antenna = effective_antenna(chunk, camera);
    // The stored triple is a BODY attitude. The camera is mounted by the nadir
    // body-to-camera rotation unless the effective master sensor's antenna
    // carries a boresight, which replaces that mount. Camera-to-world is
    // R_wb * B^T, so world->camera is B * R_wb^T.
    const Eigen::Matrix3d r_bc = antenna.boresight_deg.has_value()
                                     ? antenna.body_to_camera()
                                     : body_to_camera_nadir();

    ReferenceRotation result;
    bool degenerate_heading = false;
    const Eigen::Matrix3d frame_enu = sfm::detail::heading_frame_from_body(r_wb, &degenerate_heading);
    result.world_to_camera =
        r_bc * r_wb.transpose() * ecef_from_enu.transpose() * ecef_from_internal;
    result.heading_frame = ecef_from_internal.transpose() * ecef_from_enu * frame_enu;
    result.degenerate_heading = degenerate_heading;
    return result;
}

}  // namespace alphashape::reference
