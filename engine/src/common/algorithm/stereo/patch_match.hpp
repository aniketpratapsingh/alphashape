#pragma once

// Multi-view PatchMatch depth estimation. For one reference view, iterate over
// depth and normal candidates (propagate + random search, red-black sweeps),
// aggregating a photometric cost over the selected neighbour views. The result
// is a per-pixel depth (distance along the bearing, see DepthMap), an optional
// normal, and the final cost.

#include <cstdint>
#include <functional>
#include <vector>

#include <Eigen/Core>

#include "common/algorithm/stereo/census.hpp"
#include "image/image.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::stereo {

/// Which photometric cost the search minimises.
///
/// Zncc    the intensity cost (1 - ZNCC), in [0, 2] with 1.0 at zero
///         correlation. The original cost; assumes the two views of a surface
///         point differ by a gain and a bias, which real photo sets only
///         approximately do.
/// Census  (1 - ZNCC) / 2 computed on the census descriptor values (census.hpp):
///         each pixel's 16-bit census mask is read as a sample in [0, 1]
///         (mask / 65535) and the existing ZNCC aggregation runs on those
///         values. The half factor puts this cost on its own [0, 1] interval, as
///         the plan specifies, so its max_cost default (0.6) is a fraction of
///         that interval. The descriptor is invariant to any monotone intensity
///         change, so this survives exposure, white-balance and gamma
///         differences between overlapping images and keeps a usable signal on
///         low-contrast ground where an intensity correlation is noise. Reusing
///         the ZNCC core also keeps the sub-pixel bilinear sampling and the
///         variance floor, which a per-sample Hamming mean would lose.
///
/// The two costs have different intervals (intensity [0, 2], census [0, 1]) and
/// therefore different max_cost defaults. Their zero-correlation (chance) points
/// are 1.0 and 0.5 respectively -- half of each interval -- and
/// patch_match_default_max_cost() supplies a threshold above that for each.
enum class CostType {
    Zncc = 0,
    Census = 1,
};

struct PMView {
    const FloatImage* image = nullptr;   // luminance at depth-map resolution
    Calibration calibration;             // matches image resolution (scaled_calibration)
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();  // world->camera
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

/// Called at the end of every refinement sweep with the fraction of the search
/// finished (0..1), so a front end can move a progress bar inside one camera
/// and a caller can cancel between sweeps. The hook may throw (a task's
/// CancelledError); both the CPU and the GPU implementation let it propagate
/// unchanged, which discards the half-swept map.
using SweepProgress = std::function<void(double fraction)>;

struct PatchMatchOptions {
    double depth_min = 0.0;
    double depth_max = 0.0;
    int window_radius = 5;       // window is (2r+1)^2, sampled every window_step px
    int window_step = 2;
    int iterations = 6;          // red-black sweeps
    int best_k = 0;              // neighbours averaged per pixel; 0 = ceil(n/2)
    /// Final pixels whose cost is above this get depth -1. The value is on the
    /// selected cost's interval: intensity (1 - ZNCC) in [0, 2] (1.0 = zero
    /// correlation), census (1 - ZNCC) / 2 in [0, 1]. A task that exposes the
    /// cost as a choice should default it per cost type
    /// (patch_match_default_max_cost()); this struct's 0.5 is the intensity
    /// threshold the search has always used.
    double max_cost = 0.5;
    /// Skip reference windows flatter than this. The gate is measured on the
    /// INTENSITY window for both cost types (a census run reads the same pixels
    /// through the descriptor's transform instead), so this stays an intensity
    /// stddev on [0, 1], roughly one 8-bit level.
    double min_texture_stddev = 1.0 / 255.0;
    /// Photometric cost the search minimises. Zncc by default: selecting census
    /// is an explicit accuracy/robustness trade a caller opts into (it costs one
    /// extra transform per view and a second correlation arithmetic).
    CostType cost_type = CostType::Zncc;
    /// Census transform parameters, used when cost_type == CostType::Census.
    /// The descriptor radius is independent of `window_radius`: it describes a
    /// pixel's own neighbourhood, while `window_radius` is the patch the search
    /// correlates.
    CensusOptions census = {};
    /// Coarse-to-fine search (WP-17.14). 1 = a single level at the reference
    /// resolution (the pre-pyramid behaviour), 0 = automatic (as many halvings
    /// as keep the shorter side at or above `pyramid_min_size`), N > 1 = exactly
    /// N levels, clamped to what the geometry and the iteration budget allow.
    /// See depth_pyramid.hpp for the plan.
    int pyramid_levels = 0;
    /// The shortest side, in pixels, an automatic pyramid stops shrinking at.
    /// Ignored by an explicit level count.
    int pyramid_min_size = 256;
    std::uint64_t seed = 1;
    /// Optional per-sweep progress. Not part of a map's identity and never
    /// persisted; empty means "report nothing".
    SweepProgress on_sweep;
};

/// The iterations each level of the search runs, coarsest first. A run's level
/// count and per-level budget in one place, so a front end, the replica in the
/// GPU path and the tests all read the same plan. `size` is the level count.
/// The finest entry is the reference resolution; its neighbours are
/// geometrically one half further out per step.
std::vector<int> patch_match_iteration_plan(int width, int height,
                                            const PatchMatchOptions& options);

/// The `max_cost` a run should start from for `type`, on that type's own
/// interval (intensity [0, 2], census [0, 1]).
///
/// ZNCC keeps the historical 0.5. Census uses 0.6: the descriptor is coarser
/// than 8-bit intensity, so an otherwise-correct pixel correlates a little less
/// sharply and its true-plane cost sits a little above the intensity cost's
/// (measured ~0.015 against ~0.009 on the noisy plane fixture, both halved),
/// so the default is made more permissive so the stricter descriptor does not
/// cut pixels the intensity run keeps. 0.6 is still above the census cost's
/// chance value of 0.5 (zero correlation) and below its 1.0 rejection sentinel.
/// The figure is our own calibration on our fixtures, not a value taken from a
/// paper.
double patch_match_default_max_cost(CostType type);

struct PatchMatchResult {
    int width = 0, height = 0;
    std::vector<float> depth;    // -1 = none
    std::vector<float> normals;  // 3*N, camera frame of the reference
    std::vector<float> cost;     // N
};

PatchMatchResult run_patch_match(const PMView& reference, const std::vector<PMView>& neighbours,
                                 const PatchMatchOptions& options);

}  // namespace alphashape::stereo
