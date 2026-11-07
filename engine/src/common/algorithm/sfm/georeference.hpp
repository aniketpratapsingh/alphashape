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
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "common/core/chunk.hpp"

namespace alphashape::sfm {

/// Outcome of a georeferencing attempt.
struct GeoreferenceReport {
    bool applied = false;
    std::size_t references_used = 0;   ///< inlier cameras in the centre fit
    std::size_t references_total = 0;  ///< aligned cameras with a usable reference
    double rms_error_m = 0.0;          ///< residual of the camera-centre fit, inliers only
    std::string reason;                ///< why not applied, if not

    // -- update_transform (WP-13.12) ----------------------------------------
    std::size_t control_points_used = 0;  ///< inlier control markers in the fit
    std::size_t cameras_used = 0;         ///< inlier camera references in the fit
    std::size_t check_points_used = 0;    ///< check points measured after the fit
    double rms_control_m = 0.0;           ///< residual over the control inliers
    double rms_check_m = 0.0;             ///< RMS of the held-out check-point errors
    double max_check_m = 0.0;             ///< largest held-out check-point error
    double scale = 1.0;                   ///< recovered similarity scale
    /// Sum of the 1/sigma^2 weights over the inlier camera references and over
    /// the inlier control markers. The ratio shows which correspondences carry
    /// the datum: a 1 cm GCP outweighs a metre-level reference by four orders.
    double camera_weight = 0.0;
    double control_weight = 0.0;
    /// Held-out check-point errors, (marker id, 3-D error in metres), measured
    /// on the marker's position AFTER the transform is applied.
    std::vector<std::pair<std::int32_t, double>> check_errors;
    /// Ids of control markers that the robust fit rejected as outliers.
    std::vector<std::int32_t> rejected_markers;
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

/// Fits ONE 7-parameter similarity from the chunk's current local frame to the
/// reference frame and applies it uniformly (WP-13.12). Unlike
/// georeference_chunk this uses control markers as well as camera references,
/// and unlike OptimizeCameras it never changes the internal shape of the block.
///
/// Correspondences are (local point -> ENU reference) pairs from
///   * aligned enabled cameras with a full lon/lat/alt reference (centre), and
///   * Regular markers that carry both a local position and a full reference
///     and whose meta role is not "check_point", when the chunk's
///     ReferenceSettings::use_gcp is true.
/// Both kinds share one weighted system: weight = 1/sigma^2, sigma = the mean
/// per-axis accuracy (falling back to 10 m when it is non-finite or non-positive).
/// Check-point markers are measured after the fit and reported, never used.
/// With use_gcp=false camera references still drive the fit; only the marker
/// part of the constraint is dropped.
///
/// The robust fit uses estimate_similarity_ransac with a deterministic seed and
/// threshold = max(camera rule, 5 x median marker sigma, 0.05 m), then refits the
/// inliers with estimate_similarity_weighted. The result is applied to the
/// aligned poses, every track point, the region and every marker position, and
/// the chunk transform/CRS slots are set exactly as georeference_chunk does.
///
/// At least three non-collinear correspondences are required; otherwise
/// `applied` is false and `reason` explains why. A second call changes the chunk
/// by no more than numerical noise.
GeoreferenceReport update_transform(Chunk& chunk);

}  // namespace alphashape::sfm
