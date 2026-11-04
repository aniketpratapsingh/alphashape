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
///
/// The gate is an approximate likelihood-ratio test: (C0 - C1) / sigma2 is
/// compared with a chi-square threshold scaled by the group size. It is only
/// approximate because the objective uses robust losses (Cauchy on image
/// residuals, Huber on priors), so the statistic is not exactly chi-square.
/// The reweighting is deliberately not undone: the same robust objective is
/// minimised and compared for every candidate (cf. Förstner & Wrobel,
/// Photogrammetric Computer Vision, 2016, ch. 4).
struct AdaptiveFitOptions {
    /// Accept only if (C0 - C1) / sigma2 >= lr_threshold_per_param * group_size,
    /// where sigma2 is the residual variance estimate of the candidate solve.
    double lr_threshold_per_param = 9.0;
    /// Accept only if (C0 - C1) / C0 >= min_relative_gain.
    double min_relative_gain = 0.005;
    /// Smallest residual RMS, in pixels, used when estimating the candidate's
    /// variance. It keeps a near-zero cost (noise-free or degenerate data) from
    /// making every group look overwhelmingly significant. Our starting value.
    double min_residual_px = 0.05;
    /// Options forwarded to every bundle_adjust solve. Callers that want the
    /// final report's posterior covariance set the covariance flags here; the
    /// default (all off) changes nothing.
    BAOptions solve_options;
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
///
/// Coefficients outside the active set are held at their input values, not
/// zeroed: a group rejected here keeps whatever the camera's calibration already
/// contained.
///
/// Selection is joint across sensors: a group is freed for every sensor that
/// allows it, so one accepted group is one entry in the model shared by all
/// sensors. Per-sensor selection would be a possible follow-up but is not
/// attempted here.
AdaptiveFitResult adaptive_fit(BAProblem& problem,
                               const std::array<bool, kCalibrationParamCount>& allowed,
                               const AdaptiveFitOptions& options = {});

}  // namespace alphashape::sfm
