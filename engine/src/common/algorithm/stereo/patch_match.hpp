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
/// Zncc    the intensity ZNCC normalised to [0, 1]: (1 - z) / 2. The original
///         cost; assumes the two views of a surface point differ by a gain and a
///         bias, which real photo sets only approximately do.
/// Census  the ZNCC computed on the census descriptor values (census.hpp):
///         each pixel's 16-bit census mask is read as a sample in [0, 1]
///         (mask / 65535) and the existing ZNCC aggregation runs on those
///         values. The descriptor is invariant to any monotone intensity
///         change, so this survives exposure, white-balance and gamma
///         differences between overlapping images and keeps a usable signal on
///         low-contrast ground where an intensity correlation is noise. Reusing
///         the ZNCC core also keeps the sub-pixel bilinear sampling and the
///         variance floor, which a per-sample Hamming mean would lose.
///
/// The two costs share the [0, 1] scale and the "0.5 = chance" convention, so
/// `max_cost` means the same thing ("the winning match must be better than
/// chance") whichever type a run selects; the best threshold for a given scene
/// still differs between them and is set per cost type by the caller.
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
    /// selected cost's own [0, 1] scale. A task that exposes the cost as a
    /// choice should default it per cost type (patch_match_default_max_cost());
    /// this struct's 0.5 is the ZNCC threshold the search has always used.
    double max_cost = 0.5;
    double min_texture_stddev = 1.0 / 255.0;  // skip windows flatter than this (intensity 0..1)
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

/// The `max_cost` a run should start from for `type`, on that cost's own [0, 1]
/// scale.
///
/// ZNCC keeps the historical 0.5. Census uses 0.35: a census distance above
/// roughly a third of the bits is what a rejection looks like in practice, and
/// on the dense test scenes the true plane sits far below it while a spurious
/// match sits above. (The exact figure is a calibration of ours, measured on
/// our own fixtures, not a value taken from a paper; it is deliberately stricter
/// than 0.5 so the default does not admit the chance-level plateau at 0.5.)
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
