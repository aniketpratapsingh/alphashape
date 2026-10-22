#pragma once

// TV-L1 range-image integration (Zach, Pock & Bischof, "A Duality Based
// Approach to Real-Time TV-L1 Optical Flow" and the 2008 range-image
// formulation): the multi-view depth maps are integrated into a single
// volumetric field by an L1 data term on the per-voxel depth histogram, with a
// total-variation regulariser that carves a piecewise-constant surface out of a
// noisy / outlier-prone measurement field.
//
// Unlike a plain projective TSDF (a weighted mean), the L1 / histogram form is
// robust to the gross outliers a stereo matcher leaves behind, at the cost of a
// convex optimisation per grid. It is solved here with the Chambolle-Pock
// primal-dual algorithm.

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "base/algorithm/voxel_grid.hpp"
#include "common/algorithm/mesh/volume.hpp"
#include "common/core/chunk.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::meshing {

/// One depth map to integrate, together with the pose that places it in the
/// reconstruction volume.
struct FusionView {
    const DepthMap* map = nullptr;
    /// Intrinsics, already scaled to the depth-map resolution.
    Calibration calibration;
    /// world -> camera rotation.
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    /// world -> camera translation.
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

struct TvFusionOptions {
    /// Resolution of the longest bounding-box axis.
    int max_resolution = 128;
    /// The truncation distance eta, in voxels. Measurements are clamped to a
    /// band of +/- truncation_voxels around the observed surface.
    double truncation_voxels = 3.0;
    /// The width of the free-space band in front of the surface, in units of
    /// eta: voxels within this band on the outside are pulled toward free space.
    double front_band = 3.0;
    /// Number of bins in the per-voxel depth histogram (the L1 data term is
    /// computed over these bins).
    int histogram_bins = 8;
    /// Weight of the data term relative to the TV term.
    double lambda = 1.0;
    /// Primal-dual iterations.
    int iterations = 300;
    /// Subsampling stride over each depth map's pixels (1 = every pixel).
    int pixel_step = 1;
};

/// Integrate `views` into a regular grid over [bbox_min, bbox_max] and return
/// the field with its level set and observed mask. See VolumeResult for the
/// sign convention (> iso outside, < iso inside).
VolumeResult tv_l1_fusion(const std::vector<FusionView>& views, const Eigen::Vector3d& bbox_min,
                          const Eigen::Vector3d& bbox_max, const TvFusionOptions& options);

namespace detail {

/// The histogram integration and surface initialisation of tv_l1_fusion: the
/// grid, the per-voxel histograms normalised to probabilities, the dilated
/// observed mask, and the breadth-first nearest-neighbour initial field. Exposed
/// so the GPU port (tv_fusion_gpu.hpp) reuses exactly the CPU's setup instead of
/// duplicating it; not part of the public algorithm surface.
struct TvFusionSetup {
    GridSpec grid;
    double truncation = 0.0;      ///< eta in world units
    int bins = 0;
    std::vector<double> centres;  ///< ascending bin centres over [-1, 1]
    std::vector<double> weights;  ///< voxels * bins; each measured row sums to 1
    std::vector<std::uint8_t> observed;
    std::vector<float> initial_field;
};

/// Validates the inputs and builds the setup. Throws ValueError exactly where
/// tv_l1_fusion does, before any device work.
TvFusionSetup tv_l1_fusion_setup(const std::vector<FusionView>& views,
                                 const Eigen::Vector3d& bbox_min,
                                 const Eigen::Vector3d& bbox_max,
                                 const TvFusionOptions& options);

/// Runs the Chambolle-Pock iterations on a setup produced by
/// tv_l1_fusion_setup. This is the CPU solver path shared by the public entry
/// point and the GPU fallback.
VolumeResult tv_l1_fusion_solve(TvFusionSetup setup, const TvFusionOptions& options);

}  // namespace detail

}  // namespace alphashape::meshing
