#include "common/algorithm/markers/project_markers.hpp"

#include <cmath>
#include <optional>

#include <Eigen/Core>

#include "base/utils/error.hpp"

namespace alphashape {

std::optional<Eigen::Vector2d> project_marker(const Eigen::Vector3d& point_local,
                                              const Camera& camera,
                                              const Calibration& calibration,
                                              bool clip_to_image) {
    // An unaligned camera has an identity pose, which would place every point
    // in a fictitious frame; the honest answer is "not observable".
    if (!camera.enabled) {
        return std::nullopt;
    }
    if (!point_local.allFinite() || !camera.pose.allFinite()) {
        return std::nullopt;
    }
    // World -> camera: the same split every other projection site uses.
    const Eigen::Vector3d p_cam = camera.pose.block<3, 3>(0, 0) * point_local +
                                  camera.pose.block<3, 1>(0, 3);
    // Camera convention: +Z is forward. A point at or behind the image plane is
    // not observable in this image, including through a fisheye lens.
    if (!(p_cam.z() > 0.0)) {
        return std::nullopt;
    }
    Eigen::Vector2d pixel;
    try {
        pixel = calibration.project(p_cam);
    } catch (const ValueError&) {
        // The lens model refuses the point (a fisheye at the exact optical
        // centre); there is no meaningful pixel to return.
        return std::nullopt;
    }
    if (!pixel.allFinite()) {
        return std::nullopt;
    }
    if (clip_to_image &&
        !(pixel.x() >= 0.0 && pixel.x() <= static_cast<double>(calibration.width) &&
          pixel.y() >= 0.0 && pixel.y() <= static_cast<double>(calibration.height))) {
        return std::nullopt;
    }
    return pixel;
}

std::optional<Eigen::Vector2d> project_marker(const Marker& marker, const Camera& camera,
                                              const Calibration& calibration,
                                              bool clip_to_image) {
    if (!marker.position.has_value()) {
        return std::nullopt;
    }
    return project_marker(*marker.position, camera, calibration, clip_to_image);
}

}  // namespace alphashape
