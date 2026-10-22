#include "common/algorithm/elevation/surface_frame.hpp"

#include <string>

#include <Eigen/LU>

#include "base/utils/error.hpp"

namespace alphashape::elevation {
namespace {

/// A 4x4 transform is "the identity" for our purposes when it equals the exact
/// identity matrix; Chunk stores Transform::Identity() for unreferenced chunks.
bool is_identity(const Eigen::Matrix4d& m) { return m == Eigen::Matrix4d::Identity(); }

}  // namespace

SurfaceFrame SurfaceFrame::for_chunk(const Chunk& chunk, int epsg) {
    const Eigen::Matrix4d& transform = chunk.transform();
    const bool referenced = !is_identity(transform);

    SurfaceFrame frame;
    if (!referenced) {
        if (epsg != 0) {
            throw ValueError(
                "SurfaceFrame: unreferenced chunk has no projection for EPSG:" +
                std::to_string(epsg));
        }
        frame.projection_ = geo::Projection::from_epsg(0);
        return frame;
    }

    frame.to_ecef_ = transform;
    frame.to_local_ = transform.inverse();

    if (epsg == 0) {
        // Auto-select the UTM zone covering the chunk-local origin.
        const Eigen::Vector4d origin_local(0.0, 0.0, 0.0, 1.0);
        const Eigen::Vector4d origin_ecef = transform * origin_local;
        const geo::Geodetic origin =
            geo::ecef_to_geodetic(origin_ecef.head<3>());
        const geo::UtmZone zone =
            geo::utm_zone_for(origin.longitude_deg, origin.latitude_deg);
        frame.projection_ = geo::Projection::from_epsg(zone.epsg());
        return frame;
    }

    frame.projection_ = geo::Projection::from_epsg(epsg);
    return frame;
}

Eigen::Vector3d SurfaceFrame::to_plane(const Eigen::Vector3d& local) const {
    if (is_local()) {
        return local;
    }
    const Eigen::Vector4d local_h(local.x(), local.y(), local.z(), 1.0);
    const Eigen::Vector4d ecef = to_ecef_ * local_h;
    const geo::Geodetic g = geo::ecef_to_geodetic(ecef.head<3>());
    const Eigen::Vector2d xy = projection_.forward(g);
    return {xy.x(), xy.y(), g.height_m};
}

Eigen::Vector3d SurfaceFrame::to_local(const Eigen::Vector3d& plane) const {
    if (is_local()) {
        return plane;
    }
    const geo::Geodetic g = projection_.inverse(plane.x(), plane.y(), plane.z());
    const Eigen::Vector3d ecef = geo::geodetic_to_ecef(g);
    const Eigen::Vector4d ecef_h(ecef.x(), ecef.y(), ecef.z(), 1.0);
    const Eigen::Vector4d local = to_local_ * ecef_h;
    return local.head<3>();
}

}  // namespace alphashape::elevation
