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
//
// MEMORY. A tile is CROPPED to the bounding box of the camera's ground
// footprint on that grid (OrthoTile::x0/y0/width/height) instead of carrying a
// full-grid buffer with a mostly-empty interior. Cell (col, row) of the tile
// grid is stored at (col - x0, row - y0); a cell outside the box is exactly
// equivalent to a cell the camera does not cover (`valid == 0`). On a large
// flight this turns "C x 8 bytes x every cell of the output" into "8 bytes x
// the area each camera actually sees", which for an overlapping block is a
// small multiple of the grid rather than C times it. Consumers must go through
// OrthoTile::contains()/index(); there is no full-grid buffer to index into.

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

    /// Optional chunk-local -> plane-frame map, the inverse of
    /// `plane_to_local` (identity when empty). Only used to place a tile crop
    /// box: the bounding routine works out the camera's footprint in the frame
    /// the pose acts in and has to express it on the output grid. Callers that
    /// set `plane_to_local` should set this too (it is normally the frame's own
    /// to_plane()).
    std::function<Eigen::Vector3d(const Eigen::Vector3d&)> local_to_plane;

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

/// One camera rectified onto the output grid, cropped to its footprint.
struct OrthoTile {
    int camera = -1;                        ///< caller's camera index
    int x0 = 0;                             ///< leftmost output-grid column held
    int y0 = 0;                             ///< topmost output-grid row held
    int width = 0;                          ///< 0 when the camera covers nothing
    int height = 0;
    std::vector<std::uint8_t> rgb;          ///< 3 * width * height, row-major
    std::vector<std::uint8_t> valid;        ///< width * height, 1 when visible
    std::vector<float> score;               ///< width * height, (0, 1]; 0 invalid

    bool empty() const { return width <= 0 || height <= 0; }

    /// True when the tile holds output cell (col, row). A false answer means the
    /// camera does not cover that cell: consumers must treat it as `valid == 0`.
    bool contains(int col, int row) const {
        return col >= x0 && col < x0 + width && row >= y0 && row < y0 + height;
    }

    /// Offset of (col, row) inside the tile's buffers. The caller must have
    /// established contains(col, row) -- or bounds-checked it itself.
    std::size_t index(int col, int row) const {
        return static_cast<std::size_t>(row - y0) * static_cast<std::size_t>(width) +
               static_cast<std::size_t>(col - x0);
    }
};

/// A rectangle of output-grid cells: the tile crop box.
struct TileBounds {
    int x0 = 0;
    int y0 = 0;
    int width = 0;
    int height = 0;
};

/// The output-grid bounding box the camera's footprint can reach, clamped to
/// `out_grid` and padded by `margin_cells` on every side.
///
/// The footprint is bounded from the rays through the image border: a pinhole
/// camera sees a convex cone, and the ground lies between `lowest_height` and
/// `highest_height`, so the intersection of the cone with that slab is the
/// convex hull of the two border quads. Their border points (5 per edge, so a
/// distorted border's bow is bracketed too) bound it. The heights are in the
/// frame the pose acts in (chunk-local when `plane_to_local` is set, plane frame
/// otherwise); the bounds are mapped onto the grid through `local_to_plane`.
/// Returns a zero-width box when the camera has no image.
TileBounds camera_footprint_bounds(const RasterGrid& out_grid, const OrthoCamera& camera,
                                   double lowest_height, double highest_height,
                                   int margin_cells);

/// Rectify `camera` over `out_grid`. `dem` is sampled at output cell centres;
/// `dem_nodata` marks missing samples. Cells whose ray is blocked by the DEM by
/// more than `occlusion_tolerance` metres are left invalid.
///
/// With `crop` (the default) the tile is allocated over its footprint only; see
/// camera_footprint_bounds(). `crop == false` allocates the full output grid and
/// is the reference behaviour the crop is verified against byte for byte.
OrthoTile orthorectify(const GeoRaster<float>& dem, float dem_nodata,
                       const RasterGrid& out_grid, const OrthoCamera& camera,
                       int camera_index, double occlusion_tolerance, bool crop = true);

// -- test-only cost accounting ----------------------------------------------
// The occlusion march's cost is invisible in a tile's contents: a 4x-finer
// output grid must not multiply the DEM samples per output cell, but the tile
// looks the same either way. These accessors expose a process-wide counter so
// a test can assert the bound. They are not part of the production contract.

/// DEM samples taken by occlusion marches since the last reset (all threads).
std::uint64_t occlusion_sample_count();

/// Zero the counter. Call before the run being measured.
void reset_occlusion_sample_count();

}  // namespace alphashape::ortho
