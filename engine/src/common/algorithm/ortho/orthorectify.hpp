#pragma once

// True orthorectification of one photograph onto an output raster grid
// (milestone 9, the orthomosaic core).
//
// Everything in this package lives in ONE "plane frame": x east, y north, z up,
// the same metric units as the DEM. Callers convert from the chunk frame; the
// algorithms here never see a projection.
//
// Per output cell the routine
//   1. reads the ground height z from the DEM (bilinear, nodata-aware),
//   2. projects the ground point P = (x, y, z) into the camera,
//   3. keeps the cell only when the pixel is inside the image and P is in front
//      of the camera AND visible: a ray is marched from P toward the camera
//      centre in steps of half a DEM cell and the surface occludes P when the
//      DEM stands more than `occlusion_tolerance` above the ray (Amhar et al.
//      1998, "true orthophoto"). The march stops once the ray climbs above the
//      tallest DEM sample, where occlusion is geometrically impossible.
//   4. colours the cell by bilinear image sampling and scores it by the cosine
//      of the view angle against vertical times a taper that decays toward the
//      image border (border pixels are stretched, blurry and distortion-prone).
//
// The result is one OrthoTile per camera, all shares on the same grid, which
// the seamline and blending stages consume.

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "image/image.hpp"
#include "model/camera/calibration.hpp"
#include "model/raster/geo_raster.hpp"

namespace alphashape::ortho {

/// A camera expressed in the plane frame: X_cam = R X_plane + t.
///
/// The plane frame is not, in general, a rigid transform of the frame the pose
/// lives in: a DEM in UTM carries scale (~0.9996), meridian convergence and
/// Earth curvature (~d^2/2R, 8 cm at 1 km). Fitting an R,t through that is an
/// approximation with a position-dependent error. When `plane_to_local` is set
/// the routine instead maps every ground point it is about to project through
/// it and then applies the chunk-local pose. The occlusion march stays in the
/// plane frame (it compares against DEM heights); the march direction uses the
/// camera centre mapped into the plane frame.
///
/// With `plane_to_local` empty the camera is used exactly as before: R and t
/// are the plane-frame pose and `plane_to_local` is the identity.
struct OrthoCamera {
    Calibration calibration;                 ///< intrinsics (projection model)
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    const Image* image = nullptr;            ///< full-resolution photograph

    /// Optional plane-frame -> chunk-local map (identity when empty).
    std::function<Eigen::Vector3d(const Eigen::Vector3d&)> plane_to_local;

    /// Optional optical centre expressed in the PLANE frame, used as the
    /// target of the occlusion march (which compares against DEM heights and
    /// therefore must stay in the plane frame). Only consulted when
    /// `plane_to_local` is set; otherwise the march target is `centre()`.
    std::optional<Eigen::Vector3d> plane_centre;

    /// Ground metres per plane unit on each horizontal axis of the plane frame
    /// (1, 1) for a metric plane, the per-degree arc lengths for a Geographic
    /// one). The occlusion march needs this: it compares a metric ray against
    /// DEM heights, so its geometry has to be solved in metres even when the
    /// grid's x/y are degrees. Only consulted together with `plane_centre`
    /// (otherwise the pose and the grid are already in the same units).
    Eigen::Vector2d plane_metres_per_unit = Eigen::Vector2d(1.0, 1.0);

    /// The optical centre in the frame R, t act in (chunk-local when
    /// plane_to_local is set, plane frame otherwise): X = -R^T t.
    Eigen::Vector3d centre() const { return -R.transpose() * t; }
};

/// One camera rectified onto the output grid.
struct OrthoTile {
    int camera = -1;                        ///< caller's camera index
    std::vector<std::uint8_t> rgb;          ///< 3 * W * H, row-major
    std::vector<std::uint8_t> valid;        ///< W * H, 1 when visible & inside
    std::vector<float> score;               ///< W * H, quality in (0, 1]; 0 invalid
};

/// Rectify `camera` over `out_grid`. `dem` is sampled at output cell centres;
/// `dem_nodata` marks missing samples. Cells whose ray is blocked by the DEM by
/// more than `occlusion_tolerance` metres are left invalid.
OrthoTile orthorectify(const GeoRaster<float>& dem, float dem_nodata,
                       const RasterGrid& out_grid, const OrthoCamera& camera,
                       int camera_index, double occlusion_tolerance);

}  // namespace alphashape::ortho
