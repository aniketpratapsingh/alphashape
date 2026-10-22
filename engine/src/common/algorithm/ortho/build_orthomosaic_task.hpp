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

#include <Eigen/Core>

#include "common/algorithm/elevation/surface_frame.hpp"
#include "common/tasks/task.hpp"
#include "model/raster/geo_raster.hpp"

namespace alphashape {

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
};

}  // namespace alphashape
