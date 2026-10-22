#pragma once

// georeference.hpp -- place a chunk's reconstruction into a metric ENU frame.
//
// Convention adopted here (and relied on by the rest of the pipeline):
//
//   * After georeferencing, the chunk's INTERNAL (local) frame IS a metric ENU
//     frame -- x east, y north, z up, metres -- at an origin near the cameras.
//   * chunk.transform() is then enu_to_ecef(origin): a rigid 4x4 mapping that
//     local (ENU) coordinates to geocentric ECEF (see model/geo/geodesy.hpp).
//   * chunk.crs().epsg is set to 4326 (WGS84 geographic, the CRS the outputs
//     are advertised in). The WKT string is left for the raster milestone.
//   * Before georeferencing, chunk.transform() is the identity and the local
//     frame is the arbitrary similarity frame produced by SfM.
//
// The estimate is a similarity fit from the aligned cameras' local centres to
// the ENU positions of their GNSS references (estimate_similarity_ransac), so a
// single call recovers scale, rotation and translation at once. Repeating a call
// is stable: once the local frame is metric ENU the fit collapses to (near) the
// identity, and an already-georeferenced chunk is first mapped through its old
// transform so the composition stays exact.
//
// Pose update. A camera pose is X_cam = R_c X + t_c, and the chunk's local
// coordinates are remapped by the similarity X' = s R X + t. Substituting
// X = R^T (X' - t) / s gives X_cam = R_c R^T X' / s + (t_c - R_c R^T t / s),
// i.e. R_c' = R_c R^T and t_c' = (s t_c - R_c' t) / s. Dropping the positive
// overall factor 1/s leaves R_c' = R_c R^T, t_c' = s t_c - R_c' t; because the
// camera projects homogeneous coordinates (the division by z cancels any global
// scale) the pixel projection is unchanged, and the camera centre
// C' = -(R_c')^T t_c' = s R C + t is exactly the similarity applied to C.
//
// Region. Region is an axis-aligned box with no rotation, so the rotated region
// is approximated by the transformed centre (mapped like a point) and the size
// scaled by s.

#include <cstddef>
#include <string>

#include "common/core/chunk.hpp"

namespace alphashape::sfm {

/// Outcome of a georeferencing attempt.
struct GeoreferenceReport {
    bool applied = false;
    std::size_t references_used = 0;   ///< inlier cameras in the centre fit
    std::size_t references_total = 0;  ///< aligned cameras with a usable reference
    double rms_error_m = 0.0;          ///< residual of the camera-centre fit, inliers only
    std::string reason;                ///< why not applied, if not
};

/// Places `chunk` into a metric ENU frame from its aligned cameras' GNSS
/// references.
///
/// If at least three aligned (enabled) cameras carry an enabled reference with
/// finite longitude, latitude and altitude, and their local centres are not
/// collinear, this estimates the similarity local-centre -> ENU (origin = the
/// geodetic of the mean ECEF of those references) with
/// estimate_similarity_ransac (threshold = max(0.5 m, 5 x the median horizontal
/// accuracy)), transforms every aligned camera pose, every track point and the
/// region into the ENU frame, sets chunk.transform() = enu_to_ecef(origin) and
/// chunk.crs().epsg = 4326. Otherwise the chunk is left untouched and `reason`
/// explains why.
///
/// If the chunk is already georeferenced (transform != identity) the existing
/// local frame is first carried through the old transform to ECEF and into the
/// new ENU, so repeated calls are stable.
GeoreferenceReport georeference_chunk(Chunk& chunk);

}  // namespace alphashape::sfm
