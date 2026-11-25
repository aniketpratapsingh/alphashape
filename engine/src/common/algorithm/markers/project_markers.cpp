#include "common/algorithm/markers/project_markers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "common/core/chunk.hpp"
#include "common/io/camera_image.hpp"

#if defined(ALPHASHAPE_USE_PROJ)
#include "common/algorithm/reference/convert_reference.hpp"
#include "model/crs/coordinate_system.hpp"
#include "model/geo/geodesy.hpp"
#endif

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
    // Camera convention: +Z is forward. Only the Frame model rejects a point at
    // or behind the image plane; an equidistant fisheye maps the whole sphere
    // (theta up to pi), so it must see points with z <= 0 and let
    // Calibration::project decide.
    if (calibration.type == CameraType::Frame && !(p_cam.z() > 0.0)) {
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

#if defined(ALPHASHAPE_USE_PROJ)

namespace {

/// True when the chunk carries a real georeferencing: a non-identity local ->
/// ECEF transform AND a CRS that is not the local engineering frame. The
/// transform alone yields ECEF metres, not a chunk frame, and a CRS with an
/// identity transform is only a label (see frame_stats.cpp).
bool is_georeferenced(const Chunk& chunk) {
    if (chunk.transform() == Transform::Identity()) {
        return false;
    }
    const crs::CoordinateSystem cs = chunk.coordinate_system();
    return cs.is_set() && !cs.is_local();
}

/// Geodetic (lon, lat, ellipsoidal height) -> chunk-local, through the inverse
/// of the local -> ECEF transform. Nullopt when the transform is singular or
/// the result is not finite.
std::optional<Eigen::Vector3d> local_from_geodetic(const Chunk& chunk,
                                                   const Eigen::Vector3d& geodetic) {
    if (!geodetic.allFinite()) {
        return std::nullopt;
    }
    const Eigen::Vector3d ecef =
        geo::geodetic_to_ecef(geo::Geodetic{geodetic.x(), geodetic.y(), geodetic.z()});
    if (!ecef.allFinite()) {
        return std::nullopt;
    }
    // The transform maps local -> ECEF; its inverse carries the reference back
    // into the chunk frame. A similarity (a scaled rotation) is inverted
    // generally rather than assumed rigid.
    const Eigen::Matrix4d inverse = chunk.transform().inverse();
    if (!inverse.allFinite()) {
        return std::nullopt;
    }
    const Eigen::Vector3d local = (inverse * ecef.homogeneous()).head<3>();
    if (!local.allFinite()) {
        return std::nullopt;
    }
    return local;
}

}  // namespace

std::optional<Eigen::Vector3d> marker_local_from_reference(const Chunk& chunk,
                                                           std::size_t marker_index) {
    if (marker_index >= chunk.markers().size() || !is_georeferenced(chunk)) {
        return std::nullopt;
    }
    std::vector<std::optional<Eigen::Vector3d>> geodetic;
    try {
        geodetic = reference::geodetic_marker_references(chunk);
    } catch (const std::exception&) {
        // A reference the CRS engine cannot convert is "no prediction", not an
        // error the caller has to handle.
        return std::nullopt;
    }
    if (marker_index >= geodetic.size() || !geodetic[marker_index].has_value()) {
        return std::nullopt;
    }
    return local_from_geodetic(chunk, *geodetic[marker_index]);
}

std::vector<std::optional<Eigen::Vector3d>> marker_locals_from_references(
    const Chunk& chunk) {
    std::vector<std::optional<Eigen::Vector3d>> locals(chunk.markers().size());
    if (!is_georeferenced(chunk)) {
        return locals;
    }
    std::vector<std::optional<Eigen::Vector3d>> geodetic;
    try {
        geodetic = reference::geodetic_marker_references(chunk);
    } catch (const std::exception&) {
        return locals;
    }
    for (std::size_t i = 0; i < locals.size() && i < geodetic.size(); ++i) {
        if (geodetic[i].has_value()) {
            locals[i] = local_from_geodetic(chunk, *geodetic[i]);
        }
    }
    return locals;
}

#else  // !ALPHASHAPE_USE_PROJ

// Without the CRS engine there is no geodetic conversion, so a reference can
// never be turned into a local point. The functions stay defined so callers
// compile in a build without PROJ; they simply never predict.
std::optional<Eigen::Vector3d> marker_local_from_reference(const Chunk& chunk,
                                                           std::size_t marker_index) {
    static_cast<void>(chunk);
    static_cast<void>(marker_index);
    return std::nullopt;
}

std::vector<std::optional<Eigen::Vector3d>> marker_locals_from_references(
    const Chunk& chunk) {
    return std::vector<std::optional<Eigen::Vector3d>>(chunk.markers().size());
}

#endif  // ALPHASHAPE_USE_PROJ

namespace {

/// The calibration a camera is projected through, or nullopt when the chunk
/// has no usable sensor for it. Mirrors what the Photo tab resolves.
std::optional<Calibration> camera_calibration(const Chunk& chunk, const Camera& camera) {
    try {
        if (const Sensor* sensor = find_effective_sensor(chunk, camera)) {
            return sensor_pipeline_calibration(*sensor);
        }
    } catch (const std::exception&) {
        return std::nullopt;
    }
    return std::nullopt;
}

}  // namespace

std::vector<MarkerCameraHit> cameras_seeing_marker(const Chunk& chunk,
                                                   std::size_t marker_index) {
    std::vector<MarkerCameraHit> hits;
    if (marker_index >= chunk.markers().size()) {
        return hits;
    }
    const Marker& marker = chunk.markers()[marker_index];
    // The point to project: the estimated position when there is one, else the
    // surveyed reference carried into the local frame (georeferenced chunks
    // only). A marker with neither is not visible anywhere.
    std::optional<Eigen::Vector3d> point = marker.position;
    if (!point.has_value()) {
        point = marker_local_from_reference(chunk, marker_index);
    }
    if (!point.has_value()) {
        return hits;
    }
    for (const Camera& camera : chunk.cameras()) {
        // A disabled photo is not part of the block; it is not offered as "a photo that sees
        // the marker" even if a stale projection names it.
        if (!camera.enabled) {
            continue;
        }
        const std::optional<Calibration> calibration = camera_calibration(chunk, camera);
        if (!calibration.has_value()) {
            continue;
        }
        // A placed projection is a measurement and wins over a prediction; it
        // is in-image by construction, but the clip check keeps the contract.
        std::optional<Eigen::Vector2d> pixel;
        for (const MarkerProjection& projection : marker.projections) {
            if (projection.camera_id == camera.id) {
                pixel = projection.pixel;
                break;
            }
        }
        if (!pixel.has_value()) {
            pixel = project_marker(*point, camera, *calibration, /*clip_to_image=*/true);
        }
        if (!pixel.has_value()) {
            continue;
        }
        MarkerCameraHit hit;
        hit.camera_id = camera.id;
        hit.pixel = *pixel;
        const double cx = static_cast<double>(calibration->width) / 2.0;
        const double cy = static_cast<double>(calibration->height) / 2.0;
        hit.distance_from_centre = std::hypot(pixel->x() - cx, pixel->y() - cy);
        hits.push_back(hit);
    }
    std::sort(hits.begin(), hits.end(), [](const MarkerCameraHit& a, const MarkerCameraHit& b) {
        return a.distance_from_centre < b.distance_from_centre;
    });
    return hits;
}

}  // namespace alphashape
