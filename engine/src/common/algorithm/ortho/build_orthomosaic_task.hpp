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

#include <cstdint>
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
/// focal_px) projected onto the ground. The projection uses the VERTICAL
/// (|direction.z|, the plane frame's up), not the local surface normal: the
/// footprint therefore assumes level ground at the DEM's median height and does
/// not follow a slope. A nadir camera over level ground gives height / focal_px,
/// the classic formula; a 45-degree oblique over level ground gives
/// 2 * height / focal_px, because an image pixel spans `height / focal_px`
/// across the image and `height / (focal_px cos 45)` along the line of sight, so
/// the footprint grows by sec(incidence) on that axis.
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
///
/// `fallback_height` overrides the nodata fallback: block-wise callers compute
/// the DEM's median valid height ONCE and pass it here, because recomputing the
/// median per block would rescan the whole DEM for every block. NaN (the
/// default) means "compute it from `heights`".
///
/// `lattice`, when given, makes a block's map reproduce the WHOLE grid's map bit
/// for bit inside the block: the node coordinates are evaluated on the global
/// lattice (not recomputed from the block's own origin, which differs in the
/// last bit) and the float offsets share the global anchor, which is what the
/// subtraction rounds against. Without it the map is self-consistent but its
/// values can differ from a whole-grid map in the last float bit, which would
/// make block-wise rectification differ from whole-grid rectification.
struct PlaneToLocalLattice {
    double origin_x = 0.0;   ///< global grid's outer top-left corner
    double origin_y = 0.0;
    int col_offset = 0;      ///< this block's first node, in global node columns
    int row_offset = 0;
    Eigen::Vector3d anchor = Eigen::Vector3d::Zero();  ///< global first-node local
};

/// The local position of a grid's first node at the height sampled there: the
/// anchor every map over that grid shares. Block-wise callers compute it once on
/// the WHOLE grid and put it in PlaneToLocalLattice.
Eigen::Vector3d plane_to_local_anchor(const elevation::SurfaceFrame& frame,
                                     const RasterGrid& grid,
                                     const GeoRaster<float>& heights, float nodata,
                                     double fallback_height);

std::function<Eigen::Vector3d(const Eigen::Vector3d&)> make_plane_to_local(
    const elevation::SurfaceFrame& frame, const RasterGrid& grid,
    const GeoRaster<float>& heights, float nodata,
    double fallback_height = std::numeric_limits<double>::quiet_NaN(),
    const PlaneToLocalLattice* lattice = nullptr);

// -- block plan --------------------------------------------------------------
//
// A full-resolution ortho over a large site does not fit in memory as one
// piece: the rectified tiles alone are "footprint x 8 B/cell", and a flight of
// fully overlapping obliques makes the footprint a large multiple of the grid.
// The task therefore processes the output grid in square blocks, each with a
// margin of cells that overlaps its neighbours, and keeps only one block's
// tiles, seam field and blend planes alive at a time. The assembled result
// (RGBA + seam labels, 8 B/cell) is the only grid-sized object.
//
// Small rasters take exactly the old path: when the whole grid already fits the
// budget the plan asks for ONE block, and the block loop degenerates to the
// previous whole-grid code (no margin, no re-indexing of cells).

/// What build_orthomosaic_task passes to plan_ortho_blocks().
struct OrthoBlockPlanInput {
    int grid_width = 0;
    int grid_height = 0;
    /// Summed area of every camera's tile crop box over the WHOLE grid (what
    /// predicted_footprint_cells() returns).
    double footprint_cells = 0.0;
    /// Pixels of the largest photograph (3 bytes each when decoded).
    double photo_cells = 0.0;
    int camera_count = 0;
    /// True when the chunk is georeferenced (the plane->local node map exists).
    bool has_node_map = false;
    /// The task's byte budget; 0 means "no limit" (one block).
    std::uint64_t budget_bytes = 0;
    /// Expert/test override: process in blocks of this side (0 = automatic).
    int forced_block_cells = 0;
};

/// The block layout a run will use, with the peak it is planned for.
struct OrthoBlockPlan {
    /// Side of a block in cells; 0 means "process the whole grid in one piece".
    int block_cells = 0;
    /// Overlap kept around every block, on each side, in cells.
    int margin_cells = 0;
    int columns = 1;
    int rows = 1;
    /// The mosaic blend's pyramid depth. Pinned to the WHOLE grid's depth (or
    /// as deep as the margin supports) and shared by every block, so all blocks
    /// apply the same operator; -1 would let each block pick its own.
    int levels = 0;
    /// LRU capacity for decoded photographs. A share of the budget, so the
    /// cache can never be the reason a run does not fit.
    std::uint64_t photo_cache_bytes = 0;
    /// The predicted peak of the planned layout (window peak + assembled
    /// output + photo cache), in bytes.
    double estimated_peak_bytes = 0.0;
    /// True when even the smallest block does not fit the budget: the caller
    /// then falls back to the last-resort resolution coarsening.
    bool coarsened = false;
};

/// The margin a block of `block_cells` cells is processed with, at the mosaic
/// pyramid depth `pyramid_levels` the plan pins. It is the blend's own support
/// (126 cells at the 6-level cap) plus a small slack, NOT a fraction of the
/// block: the margin is what makes a block's interior identical to the whole
/// grid's there, so it is set by the operators that read across cells. Returns 0
/// for a non-positive block side.
int block_margin_cells(int block_cells, int pyramid_levels);

/// The predicted peak of one block, in bytes:
/// the window's own estimate (estimate_peak_bytes() on the block+margin grid)
/// plus the assembled output raster and label field (8 B per OUTPUT cell) plus
/// the photo cache. The window and output terms are disjoint: only one window
/// is alive, and the assembled raster grows while the windows come and go.
double estimate_block_peak_bytes(std::uint64_t window_cells, double window_footprint_cells,
                                 bool has_node_map, double photo_cells,
                                 std::uint64_t output_cells,
                                 std::uint64_t photo_cache_bytes);

/// Choose the block layout for a run. One block when the whole grid fits the
/// budget (identical behaviour for small rasters); otherwise the largest block
/// side from 4096 down to 512 whose estimate stays inside a quarter of the
/// budget -- the rest of the budget has to hold the assembled output, the photo
/// cache and the rest of the process.
OrthoBlockPlan plan_ortho_blocks(const OrthoBlockPlanInput& input);

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
