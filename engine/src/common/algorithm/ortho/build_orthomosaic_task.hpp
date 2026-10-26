#pragma once

// BuildOrthomosaic: build the chunk's true orthophoto (milestone 9).
//
// The task adapter over the ortho core (orthorectify.hpp, seamlines.hpp,
// blend.hpp). Every enabled camera is rectified onto one north-up grid in the
// DEM's projection plane, the seamline labeller picks the source camera per
// cell, and the blender combines the tiles. The result is appended as a NEW
// orthomosaic asset (made active), with the chunk camera ids in its per-cell
// label field, so an earlier orthomosaic is never overwritten.
//
// Parameter names follow Metashape's public buildOrthomosaic()
// Python API; the implementation (true orthorectification with a DEM occlusion
// test) is ours.

#include <functional>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include "common/algorithm/elevation/surface_frame.hpp"
#include "common/tasks/task.hpp"
#include "model/raster/geo_raster.hpp"

namespace alphashape {
namespace ortho {

/// One camera's principal axis for the automatic-resolution rule, expressed in
/// a METRIC plane frame (x east, y north, z up, all metres).
struct FootprintCamera {
    Eigen::Vector3d centre = Eigen::Vector3d::Zero();  ///< optical centre, metres
    Eigen::Vector3d direction = Eigen::Vector3d(0, 0, -1);  ///< principal axis
    double focal_px = 0.0;                             ///< focal length in pixels
};

/// Surface height in metres at a horizontal position in the same metric plane
/// frame as the cameras. MUST return NaN (not a finite nodata sentinel) where
/// the surface is unknown: the ray march treats every finite value as ground.
using SurfaceHeight = std::function<double(double x, double y)>;

/// The median ground sample distance (metres) the cameras see at their image
/// centres on the surface sampled by `height`.
///
/// For each camera the principal ray is intersected with the surface and the
/// footprint is taken as
///
///     gsd = slant_range / (focal_px * cos(incidence))
///
/// i.e. the pixel's size perpendicular to the optical axis (slant_range /
/// focal_px) projected onto the terrain along the surface normal (taken as
/// vertical, the plane frame's up). A nadir camera over level ground therefore
/// gives height / focal_px, the classic formula; a 45-degree oblique over level
/// ground gives 2 * height / focal_px, because an image pixel spans
/// `height / focal_px` across the image and `height / (focal_px cos 45)` along
/// the line of sight, so the footprint grows by sec(incidence) on that axis.
///
/// `stride_metres` is the marching step; a surface cell (or one metre) is ample.
/// `lowest_surface_metres` is the smallest height the surface takes (pass NaN to
/// let the routine probe around the camera instead); it bounds the ray march, so
/// a camera on a ridge above a valley still finds the ground.
/// A camera whose principal axis does not reach the surface, or that looks
/// nearly horizontally along it (cos(incidence) < 0.15), contributes nothing.
/// Returns 0 when no camera contributes; the caller then falls back to the
/// surface's own cell.
double median_ground_sample_distance(const std::vector<FootprintCamera>& cameras,
                                     const SurfaceHeight& height, double stride_metres,
                                     double lowest_surface_metres = std::numeric_limits<
                                         double>::quiet_NaN());

}  // namespace ortho

/// Build the plane-frame -> chunk-local map the orthorectifier applies before
/// the chunk-local camera pose.
///
/// Returns an empty function (identity) for a Local frame. Otherwise the local
/// coordinates of the grid's corner NODES are precomputed once (each at the DEM
/// height sampled there) and the query is bilinearly interpolated, with the
/// vertical corrected exactly: the mapping is near-rigid, so the interpolation
/// error over one cell is far below a millimetre, while the exact projection
/// inverse costs a geodesy round trip per call.
///
/// `heights` must be the DEM whose grid `grid` follows; `nodata` marks missing
/// samples at the nodes (their height falls back to the first valid sample).
std::function<Eigen::Vector3d(const Eigen::Vector3d&)> make_plane_to_local(
    const elevation::SurfaceFrame& frame, const RasterGrid& grid,
    const GeoRaster<float>& heights, float nodata);

/// Build the chunk's orthomosaic from its DEM (surface_data = "elevation") or
/// from a DSM derived from Chunk::model() (surface_data = "model"). Throws
/// ValueError when the required surface is missing, when `projection` names a
/// different CRS than the DEM (reproject the DEM first), or for an unknown
/// surface/blending name. Commits with one move, so a failure leaves any
/// previous orthomosaic untouched.
class BuildOrthomosaicTask : public AbstractTask {
public:
    std::string name() const override { return "BuildOrthomosaic"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;

private:
    /// The body of apply(): apply() is the thin wrapper that turns an unexpected
    /// std::bad_alloc into the same actionable ValueError the memory guard
    /// raises.
    void run(Document& document, Params& params, const Progress& progress);
};

}  // namespace alphashape
