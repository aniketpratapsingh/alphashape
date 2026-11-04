// Adaptive camera-model selection: ordered forward selection over the
// self-calibration coefficient groups. See adaptive_fit.hpp for the policy.

#include "common/algorithm/sfm/adaptive_fit.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "base/utils/error.hpp"

namespace alphashape::sfm {
namespace {

/// Dimension of the residual vector the solver actually minimises: two per
/// non-check observation, three per non-check control point, one per non-check
/// scale bar and one per constrained prior axis.
std::size_t count_residuals(const BAProblem& problem) {
    std::vector<std::uint8_t> is_check(problem.points.size(), 0);
    for (std::size_t i = 0;
         i < problem.checkpoint_point.size() && i < is_check.size(); ++i) {
        if (problem.checkpoint_point[i] != 0) {
            is_check[i] = 1;
        }
    }
    for (const BAControlPoint& control : problem.control_points) {
        if (control.checkpoint && control.point < is_check.size()) {
            is_check[control.point] = 1;
        }
    }
    std::size_t n = 0;
    for (const BAObservation& observation : problem.observations) {
        const bool check = observation.point < is_check.size() &&
                           is_check[observation.point] != 0;
        if (!check) {
            n += 2;
        }
    }
    for (const BACentrePrior& prior : problem.centre_priors) {
        for (int axis = 0; axis < 3; ++axis) {
            const double sigma = prior.sigma[static_cast<Eigen::Index>(axis)];
            if (std::isfinite(sigma) && sigma > 0.0) {
                ++n;
            }
        }
    }
    for (const BAControlPoint& control : problem.control_points) {
        if (!control.checkpoint) {
            n += 3;
        }
    }
    for (const BAScalebar& bar : problem.scalebars) {
        if (!bar.checkpoint) {
            ++n;
        }
    }
    return n;
}

/// Number of free parameters in the problem: free intrinsics, six per free pose,
/// three per point, minus the pinned scale-anchor component.
std::size_t count_free_parameters(const BAProblem& problem) {
    std::size_t n = 0;
    for (const std::array<bool, kCalibrationParamCount>& mask : problem.fit_intrinsics) {
        for (const bool free_term : mask) {
            if (free_term) {
                ++n;
            }
        }
    }
    for (std::size_t i = 0; i < problem.poses.size(); ++i) {
        const bool fixed =
            !problem.fixed_pose.empty() && problem.fixed_pose[i] != 0;
        if (!fixed) {
            n += 6;
        }
    }
    n += 3 * problem.points.size();
    if (problem.scale_anchor_camera >= 0 && n > 0) {
        --n;  // the anchor pins one translation component
    }
    return n;
}

/// Robust objective value of a solve, falling back to the plain squared
/// reprojection error when the solver did not report a cost.
double cost_of(const BAReport& report) {
    if (report.robust_cost > 0.0) {
        return report.robust_cost;
    }
    return 0.5 * report.final_rms_px * report.final_rms_px *
           static_cast<double>(report.observations);
}

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

    const std::size_t residuals = count_residuals(problem);

    set_free_mask(problem, sensor_allowed, base);
    BAReport report = bundle_adjust(problem, BAOptions{});
    result.solves = 1;
    if (!report.usable) {
        throw Error(
            "AdaptiveFit: base bundle adjustment did not produce a usable solution");
    }
    double accepted_cost = cost_of(report);
    std::array<bool, kCalibrationParamCount> accepted = base;

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
        const std::size_t free_parameters = count_free_parameters(problem);
        const std::size_t dof =
            residuals > free_parameters ? residuals - free_parameters : 0;
        const BAReport candidate = bundle_adjust(problem, BAOptions{});
        ++result.solves;
        bool accept = false;
        if (candidate.usable && dof > 0) {
            const double candidate_cost = cost_of(candidate);
            if (candidate_cost > 0.0 && candidate_cost < accepted_cost) {
                const double statistic = (accepted_cost - candidate_cost) /
                                         (candidate_cost / static_cast<double>(dof));
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



