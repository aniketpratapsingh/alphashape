#pragma once

// Adaptive camera-model selection for self-calibration (ordered forward
// selection). Freeing high-order distortion terms on weak geometry overfits the
// solution: the extra terms absorb noise instead of signal. Adaptive mode starts
// from a minimal, always-justified base set and adds one candidate group at a
// time, keeping a group only when the robust cost drops by more than a
// likelihood-ratio gate (about 3 sigma per freed parameter) AND by a minimum
// relative amount.
//
// The selection is a greedy forward pass, not a global search: a group rejected
// early is never reconsidered, matching how the task documents the mode.

#include <array>

#include "common/algorithm/sfm/bundle_adjustment.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::sfm {

/// Gates for accepting a candidate group. These are our starting values, chosen
/// from the self-calibration literature and the likelihood-ratio test at about
/// 3 sigma; they are expected to be tuned on the benchmark.
struct AdaptiveFitOptions {
    /// Accept only if (C0 - C1) / (C1 / dof) >= lr_threshold_per_param * group_size.
    double lr_threshold_per_param = 9.0;
    /// Accept only if (C0 - C1) / C0 >= min_relative_gain.
    double min_relative_gain = 0.005;
};

struct AdaptiveFitResult {
    /// Report of the last accepted solve.
    BAReport report;
    /// Coefficients that ended up free (all sensors), in the packed order.
    std::array<bool, kCalibrationParamCount> selected = {};
    /// Number of BA solves performed, including the base solve.
    int solves = 0;
};

/// Runs ordered forward selection over `problem` in place.
///
/// `allowed` is the hard upper bound: a coefficient with allowed[k] == false is
/// never freed, and the per-sensor fit masks already stored in
/// problem.fit_intrinsics are further ANDed with it. The base set is
/// {f, cx, cy, k1} intersect allowed; candidate groups, applied in order, are
/// {k2}, {p1,p2}, {k3}, {b1}, {b2}, {k4}, {p3,p4}. Each accepted group is a warm
/// start from the current accepted state; a rejected group restores the
/// problem exactly. Returns the final report, the selected mask and the solve
/// count. Throws Error if the base solve is unusable.
AdaptiveFitResult adaptive_fit(BAProblem& problem,
                               const std::array<bool, kCalibrationParamCount>& allowed,
                               const AdaptiveFitOptions& options = {});

}  // namespace alphashape::sfm
