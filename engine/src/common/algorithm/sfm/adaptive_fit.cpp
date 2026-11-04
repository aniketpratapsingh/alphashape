// Adaptive camera-model selection: ordered forward selection over the
// self-calibration coefficient groups. See adaptive_fit.hpp for the policy.

#include "common/algorithm/sfm/adaptive_fit.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include "base/utils/error.hpp"

namespace alphashape::sfm {
namespace {

/// Robust objective value of a solve. The solver's final cost may only be
/// compared when it actually produced one; there is no regression fallback.
double cost_of(const BAReport& report) { return report.robust_cost; }

/// Writes the free masks for `active` into the problem: a term is free only when
/// the sensor allows it AND it is in the active set.
void set_free_mask(BAProblem& problem,
                   const std::vector<std::array<bool, kCalibrationParamCount>>& sensor_allowed,
                   const std::array<bool, kCalibrationParamCount>& active) {
    problem.fit_intrinsics.assign(
        sensor_allowed.size(), std::array<bool, kCalibrationParamCount>{});
    for (std::size_t s = 0; s < sensor_allowed.size(); ++s) {
        for (int k = 0; k < kCalibrationParamCount; ++k) {
            const std::size_t kk = static_cast<std::size_t>(k);
            problem.fit_intrinsics[s][kk] = sensor_allowed[s][kk] && active[kk];
        }
    }
}

/// Candidate groups in selection order; -1 marks an unused slot. Indices follow
/// the packed order f, cx, cy, b1, b2, k1, k2, k3, k4, p1, p2, p3, p4.
constexpr std::array<std::array<int, 2>, 7> kCandidateGroups = {{
    {{6, -1}},   // k2
    {{9, 10}},   // p1, p2
    {{7, -1}},   // k3
    {{3, -1}},   // b1
    {{4, -1}},   // b2
    {{8, -1}},   // k4
    {{11, 12}},  // p3, p4
}};

}  // namespace

AdaptiveFitResult adaptive_fit(BAProblem& problem,
                               const std::array<bool, kCalibrationParamCount>& allowed,
                               const AdaptiveFitOptions& options) {
    AdaptiveFitResult result;

    // Per-sensor allowed set: the problem's own masks (fit_X and fixed_params
    // already applied by the caller) intersected with the task-level mask.
    std::vector<std::array<bool, kCalibrationParamCount>> sensor_allowed(
        problem.calibrations.size(), std::array<bool, kCalibrationParamCount>{});
    for (std::size_t s = 0; s < sensor_allowed.size(); ++s) {
        for (int k = 0; k < kCalibrationParamCount; ++k) {
            const std::size_t kk = static_cast<std::size_t>(k);
            const bool free_now =
                problem.fit_intrinsics.empty() ? true : problem.fit_intrinsics[s][kk];
            sensor_allowed[s][kk] = free_now && allowed[kk];
        }
    }

    std::array<bool, kCalibrationParamCount> base = {};
    base[0] = base[1] = base[2] = base[5] = true;  // f, cx, cy, k1

    set_free_mask(problem, sensor_allowed, base);
    BAReport report = bundle_adjust(problem, options.solve_options);
    result.solves = 1;
    if (!report.usable || !report.cost_valid) {
        throw Error(
            "AdaptiveFit: base bundle adjustment did not produce a usable solution");
    }
    double accepted_cost = cost_of(report);
    std::array<bool, kCalibrationParamCount> accepted = base;

    // The cost is 0.5 * sum rho(r^2) with r in pixels, so a residual variance
    // floor of min_residual_px^2 contributes 0.5 * min_residual_px^2 per
    // degree of freedom. Without the floor a noise-free cost of zero makes
    // every candidate look infinitely significant.
    const double min_variance =
        0.5 * options.min_residual_px * options.min_residual_px;

    for (const std::array<int, 2>& group : kCandidateGroups) {
        std::array<bool, kCalibrationParamCount> next = accepted;
        std::size_t new_parameters = 0;
        for (const int index : group) {
            if (index < 0) {
                continue;
            }
            const std::size_t kk = static_cast<std::size_t>(index);
            next[kk] = true;
            if (!accepted[kk]) {
                for (const std::array<bool, kCalibrationParamCount>& mask :
                     sensor_allowed) {
                    if (mask[kk]) {
                        ++new_parameters;
                    }
                }
            }
        }
        if (new_parameters == 0) {
            continue;  // no sensor allows anything new in this group
        }

        const BAProblem saved = problem;
        set_free_mask(problem, sensor_allowed, next);
        const BAReport candidate = bundle_adjust(problem, options.solve_options);
        ++result.solves;
        bool accept = false;
        if (candidate.usable && candidate.cost_valid &&
            candidate.residual_count > candidate.free_parameter_count) {
            const double candidate_cost = cost_of(candidate);
            if (candidate_cost < accepted_cost) {
                const std::size_t dof =
                    candidate.residual_count - candidate.free_parameter_count;
                const double variance = std::max(
                    candidate_cost / static_cast<double>(dof), min_variance);
                const double statistic = (accepted_cost - candidate_cost) / variance;
                const bool significant =
                    statistic >= options.lr_threshold_per_param *
                                     static_cast<double>(new_parameters);
                const bool enough_gain =
                    (accepted_cost - candidate_cost) / accepted_cost >=
                    options.min_relative_gain;
                accept = significant && enough_gain;
            }
            if (accept) {
                accepted = next;
                accepted_cost = candidate_cost;
                report = candidate;
            }
        }
        if (!accept) {
            problem = saved;  // exact restore of poses, points and intrinsics
        }
    }

    for (std::size_t s = 0; s < sensor_allowed.size(); ++s) {
        for (int k = 0; k < kCalibrationParamCount; ++k) {
            const std::size_t kk = static_cast<std::size_t>(k);
            if (accepted[kk] && sensor_allowed[s][kk]) {
                result.selected[kk] = true;
            }
        }
    }
    result.report = report;
    return result;
}

}  // namespace alphashape::sfm



