#pragma once

// SurfaceFrame -- the bridge between a chunk's local coordinate frame and the
// geographic / projected frame a DEM or orthomosaic is written in.
//
// A georeferenced chunk carries a local -> ECEF transform (Chunk::transform()).
// SurfaceFrame composes that with a map projection so that a chunk-local point
// can be expressed as a plane coordinate (easting/northing for UTM, degrees for
// Geographic) plus an ellipsoidal height, and back again:
//
//     local --transform--> ECEF --ecef_to_geodetic--> geodetic
//           --projection.forward--> plane (x, y) + ellipsoidal height
//
// An unreferenced chunk has an identity transform: to_plane/to_local are the
// identity and the frame is Local. Asking an unreferenced chunk for a non-zero
// EPSG is a programming error and throws ValueError.

#include <Eigen/Core>

#include "common/core/chunk.hpp"
#include "model/geo/geodesy.hpp"
#include "model/geo/projection.hpp"

namespace alphashape::elevation {

/// Maps between a chunk's local metric frame and a map projection.
class SurfaceFrame {
public:
    /// Build the frame for `chunk`.
    ///
    /// * Georeferenced chunk (transform() != identity):
    ///     - epsg == 0  -> auto-select the UTM zone covering local (0, 0, 0).
    ///     - otherwise  -> the requested projection (Local/Geographic/UTM).
    /// * Unreferenced chunk (transform() == identity):
    ///     - epsg == 0  -> Local identity frame.
    ///     - epsg != 0  -> throws ValueError (no geographic meaning).
    static SurfaceFrame for_chunk(const Chunk& chunk, int epsg = 0);

    /// Chunk-local (x, y, z) metres -> plane (x, y, ellipsoidal height m). For a
    /// Local frame this is the identity. For Geographic the plane x/y are
    /// longitude/latitude in degrees.
    Eigen::Vector3d to_plane(const Eigen::Vector3d& local) const;

    /// Inverse of to_plane(): plane (x, y, height m) -> chunk-local metres.
    Eigen::Vector3d to_local(const Eigen::Vector3d& plane) const;

    /// The projection this frame writes in.
    const geo::Projection& projection() const { return projection_; }

    /// EPSG code of the projection (0 when Local).
    int epsg() const { return projection_.epsg(); }

    /// Ground metres per plane unit along x (east) and y (north) at the plane
    /// ordinate `plane_y` (northing for a UTM/local plane, latitude in degrees
    /// for a Geographic one). Local and UTM planes are metric, so both
    /// components are exactly 1. A Geographic plane is in degrees, so the
    /// components are the WGS84 arc lengths of one degree of longitude
    /// (prime-vertical radius times cos(latitude)) and latitude (meridian
    /// radius) at that ordinate.
    ///
    /// This is what turns a metric ground sample distance into the units an
    /// output grid is expressed in: dividing by the larger component gives a
    /// plane-unit cell that is never coarser than the metric size on either
    /// axis.
    Eigen::Vector2d metres_per_plane_unit(double plane_y) const;

    /// True when the frame is the Local identity (unreferenced chunk).
    bool is_local() const { return projection_.kind == geo::ProjectionKind::Local; }

private:
    SurfaceFrame() = default;

    geo::Projection projection_{};

    /// local -> ECEF (chunk transform); identity when the frame is Local.
    Eigen::Matrix4d to_ecef_ = Eigen::Matrix4d::Identity();
    /// ECEF -> local (inverse of to_ecef_); identity when the frame is Local.
    Eigen::Matrix4d to_local_ = Eigen::Matrix4d::Identity();
};

}  // namespace alphashape::elevation
