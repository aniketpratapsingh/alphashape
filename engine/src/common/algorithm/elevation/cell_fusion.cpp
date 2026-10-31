#include "common/algorithm/elevation/cell_fusion.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace alphashape::elevation {
namespace {

/// Median of the sorted observations in [begin, end).
double median_of_sorted(const std::vector<HeightObservation>& observations, std::size_t begin,
                        std::size_t end) {
    const std::size_t count = end - begin;
    if (count == 0U) {
        return 0.0;
    }
    const std::size_t middle = begin + count / 2U;
    if (count % 2U == 1U) {
        return observations[middle].height;
    }
    return 0.5 * (observations[middle - 1U].height + observations[middle].height);
}

/// Tukey biweight for one residual at `scale`, multiplied by the observation's
/// confidence. Zero outside c * scale; fully weighted at zero residual.
double tukey_weight(double residual, double scale, double confidence) {
    if (!(confidence > 0.0)) {
        return 0.0;
    }
    const double cutoff = kFusionTukeyC * scale;
    if (!(cutoff > 0.0)) {
        return residual == 0.0 ? confidence : 0.0;
    }
    const double ratio = residual / cutoff;
    if (!(std::abs(ratio) < 1.0)) {
        return 0.0;
    }
    const double shape = 1.0 - ratio * ratio;
    return confidence * shape * shape;
}

/// Median of a vector of deviations (consumes and sorts it).
double median_of_deviations(std::vector<double>& deviations) {
    const std::size_t count = deviations.size();
    if (count == 0U) {
        return 0.0;
    }
    std::sort(deviations.begin(), deviations.end());
    const std::size_t middle = count / 2U;
    if (count % 2U == 1U) {
        return deviations[middle];
    }
    return 0.5 * (deviations[middle - 1U] + deviations[middle]);
}

}  // namespace

FusedCell fuse_cell(std::vector<HeightObservation>& observations, double scale_floor) {
    FusedCell fused;
    if (observations.empty()) {
        return fused;
    }

    // Order independence: height first, then confidence so duplicate heights
    // have a deterministic order too.
    std::sort(observations.begin(), observations.end(),
              [](const HeightObservation& a, const HeightObservation& b) {
                  if (a.height != b.height) {
                      return a.height < b.height;
                  }
                  return a.confidence < b.confidence;
              });

    const std::size_t count = observations.size();
    const double median = median_of_sorted(observations, 0U, count);

    // Robust scale from the median absolute deviation, floored so a cell with
    // many identical samples still has a usable cutoff.
    std::vector<double> deviations;
    deviations.reserve(count);
    for (const HeightObservation& observation : observations) {
        deviations.push_back(std::abs(observation.height - median));
    }
    const double mad = median_of_deviations(deviations);
    double scale = kFusionMadToSigma * mad;
    if (scale < scale_floor) {
        scale = scale_floor;
    }

    // Symmetric rank trim around the median: keep the central fraction, but
    // never below kFusionMinKept and never for small cells.
    std::size_t begin = 0U;
    std::size_t end = count;
    if (count >= kFusionTrimMinCount) {
        std::size_t keep = static_cast<std::size_t>(
            std::lround(static_cast<double>(count) * kFusionKeepFraction));
        if (keep < kFusionMinKept) {
            keep = kFusionMinKept;
        }
        if (keep < count) {
            const std::size_t trim = (count - keep) / 2U;
            begin = trim;
            end = count - trim;
        }
    }
    fused.kept = end - begin;

    // M-estimate over the kept set: start at its median, refine with IRLS.
    const double kept_median = median_of_sorted(observations, begin, end);
    double estimate = kept_median;
    for (int iteration = 0; iteration < kFusionIrlsIterations; ++iteration) {
        double sum_weight = 0.0;
        double sum_weighted_height = 0.0;
        for (std::size_t i = begin; i < end; ++i) {
            const HeightObservation& observation = observations[i];
            const double weight =
                tukey_weight(observation.height - estimate, scale, observation.confidence);
            sum_weight += weight;
            sum_weighted_height += weight * observation.height;
        }
        if (!(sum_weight > 0.0)) {
            estimate = kept_median;
            break;
        }
        estimate = sum_weighted_height / sum_weight;
    }

    // Final per-sample weights at the converged centre: their sum is the cell's
    // data weight before the agreement factor.
    double weight_sum = 0.0;
    double residual_square_sum = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
        const HeightObservation& observation = observations[i];
        const double residual = observation.height - estimate;
        const double weight = tukey_weight(residual, scale, observation.confidence);
        weight_sum += weight;
        residual_square_sum += weight * residual * residual;
    }
    if (!(weight_sum > 0.0)) {
        fused.value = kept_median;  // every observation rejected: keep the median
        fused.weight = 0.0;
        return fused;
    }

    const double weighted_rms = std::sqrt(residual_square_sum / weight_sum);
    const double reference = scale_floor * kFusionAgreementScale;
    double agreement = 1.0;
    if (reference > 0.0) {
        const double ratio = weighted_rms / reference;
        agreement = 1.0 / (1.0 + ratio * ratio);
    }
    fused.value = estimate;
    fused.weight = weight_sum * agreement;
    return fused;
}

}  // namespace alphashape::elevation
