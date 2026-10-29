#pragma once

// SurfaceFrame -- the bridge between a chunk's local coordinate frame and the
// CRS a DEM or orthomosaic is written in (M12 §4.9, WP-12.8).
//
// A georeferenced chunk carries a local -> ECEF transform (Chunk::transform()).
// SurfaceFrame composes that with the CRS engine so a chunk-local point can be
// expressed in any CRS PROJ can relate to ECEF, and back again:
//
//     local --chunk.transform()--> ECEF --LocalFrame--> target CRS
//
// The chain is exact at every point (iterated ECEF->geodetic plus PROJ), not a
// projection series linearised at the origin. An unreferenced chunk has an
// identity transform: to_plane/to_local are the identity and the frame is the
// local engineering CRS. Asking an unreferenced chunk for a geographic or
// projected CRS is a programming error and throws ValueError.
//
// The PROJ-backed state is behind a shared, immutable pimpl so this header
// carries no PROJ include and a `-DALPHASHAPE_USE_PROJ=OFF` build still
// compiles the DEM/ortho code that names the class (with the local identity
// frame supported and every geographic request refused).

#include <memory>

#include <Eigen/Core>

#include "common/core/chunk.hpp"
#include "model/crs/coordinate_system.hpp"

namespace alphashape::elevation {

/// Maps between a chunk's local metric frame and a target CRS.
class SurfaceFrame {
public:
    SurfaceFrame();
    SurfaceFrame(const SurfaceFrame&);
    SurfaceFrame& operator=(const SurfaceFrame&);
    SurfaceFrame(SurfaceFrame&&) noexcept;
    SurfaceFrame& operator=(SurfaceFrame&&) noexcept;
    ~SurfaceFrame();

    /// Build the frame for `chunk` with an EPSG code.
    ///
    /// * Georeferenced chunk (transform() != identity):
    ///     - epsg == 0  -> auto-select the UTM zone covering local (0, 0, 0).
    ///     - otherwise  -> that CRS (any PROJ code, not just UTM/4326).
    /// * Unreferenced chunk (transform() == identity):
    ///     - epsg == 0  -> the local identity frame.
    ///     - epsg != 0  -> throws ValueError (no geographic meaning).
    static SurfaceFrame for_chunk(const Chunk& chunk, int epsg = 0);

    /// Build the frame for `chunk` writing in `target`; an unset `target` means
    /// the chunk's own CRS. The local CRS is only meaningful for an
    /// unreferenced chunk (the identity frame); any other combination with the
    /// local CRS, or an unset CRS on a georeferenced chunk, throws ValueError.
    static SurfaceFrame for_chunk(const Chunk& chunk, const crs::CoordinateSystem& target);

    /// Chunk-local (x, y, z) metres -> target CRS point (east/north/up). For the
    /// local frame this is the identity; for a geographic CRS the horizontal
    /// components are degrees and z is the ellipsoidal height in metres.
    Eigen::Vector3d to_plane(const Eigen::Vector3d& local) const;

    /// Exact inverse of to_plane(): CRS point -> chunk-local metres.
    Eigen::Vector3d to_local(const Eigen::Vector3d& plane) const;

    /// The CRS this frame writes in (the local frame for an unreferenced chunk).
    /// @throws Error in a build without the CRS engine.
    const crs::CoordinateSystem& crs() const;

    /// EPSG code of the target CRS (0 when local or custom).
    int epsg() const;
    /// Ground metres per plane unit along x (east) and y (north) at the plane
    /// ordinate `plane_y` (northing for a projected plane, latitude in degrees
    /// for a geographic one). A projected plane is metric, so both components
    /// are exactly 1 (UTM's ~0.9996 point scale is immaterial for a cell size).
    /// A geographic plane is in degrees, so the components are the WGS84 arc
    /// lengths of one degree of longitude (prime-vertical radius times cos
    /// latitude) and latitude (meridian radius) at that ordinate.
    Eigen::Vector2d metres_per_plane_unit(double plane_y) const;

    /// True when the frame is the local identity (unreferenced chunk).
    bool is_local() const;

private:
    struct Impl;
    /// Null is the local identity frame.
    std::shared_ptr<const Impl> impl_;
};

/// Parse a build task's `crs` parameter (M12 §4.9, WP-12.8): "EPSG:<n>", a bare
/// integer code, a PROJ string ("+proj=utm +zone=33 +datum=WGS84") or WKT. An
/// empty string yields the unset DTO (the local frame). Throws ValueError for
/// text PROJ cannot read, and ValueError in a build without the CRS engine
/// (which cannot express any CRS).
///
/// Returns the project DTO rather than an engine `CoordinateSystem` so that a
/// `-DALPHASHAPE_USE_PROJ=OFF` build, where the engine library does not exist,
/// still compiles the callers in this layer. The DTO converts to the engine type
/// on the way in (`for_options`).
CRS parse_crs_param(const std::string& text);

/// Build the frame for a task's requested CRS, given as the DTO that a task
/// parameter produced. An unset `requested` applies the chunk rules
/// (`for_chunk(chunk, 0)`: auto UTM when georeferenced, identity otherwise);
/// a set one is honoured exactly. In a build without the CRS engine only an
/// unset request is possible and anything else throws ValueError.
SurfaceFrame for_options(const Chunk& chunk, const CRS& requested);

/// True when `requested` names something other than the chunk's own CRS, i.e.
/// the caller asked for a specific output CRS. False for an unset DTO and for a
/// code equal to `chunk.crs().epsg`. Kept here (rather than as a direct `.epsg`
/// read) so the DTO's identity test is one place.
bool requests_other_crs(const Chunk& chunk, const CRS& requested);

}  // namespace alphashape::elevation
