// Robust per-cell height fusion (BuildDem Q3): single-sample pass-through,
// Gaussian efficiency against the plain median, outlier rejection, the
// downsample weight and order independence.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "common/algorithm/elevation/cell_fusion.hpp"

namespace alphashape::elevation {
namespace {

/// One standard-normal variate by Box-Muller. The two mt19937 draws are on
/// separate statements (the test lint allows one draw per statement).
double normal_draw(std::mt19937& rng, double sigma) {
    const auto first = rng();
    const auto second = rng();
    const double uniform_one = (static_cast<double>(first) + 1.0) / 4294967297.0;
    const double uniform_two = static_cast<double>(second) / 4294967296.0;
    const double radius = std::sqrt(-2.0 * std::log(uniform_one));
    const double angle = 6.283185307179586 * uniform_two;
    return sigma * radius * std::cos(angle);
}

double plain_median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    if (values.size() % 2U == 1U) {
        return values[middle];
    }
    return 0.5 * (values[middle - 1U] + values[middle]);
}

TEST(CellFusion, SingleObservationPassesThrough) {
    std::vector<HeightObservation> observations;
    observations.push_back(HeightObservation{12.5, 0.75});
    const FusedCell fused = fuse_cell(observations, 0.25);
    EXPECT_DOUBLE_EQ(fused.value, 12.5);
    EXPECT_DOUBLE_EQ(fused.weight, 0.75);
    EXPECT_EQ(fused.kept, 1U);
}

TEST(CellFusion, GaussianNoiseBeatsThePlainMedian) {
    // The robust estimator uses the cell's redundancy (a trimmed, weighted
    // mean), the median only its middle rank. Over many cells it must be at
    // least 15% better in RMS.
    std::mt19937 rng(123456789U);
    constexpr int kTrials = 200;
    constexpr int kSamples = 40;
    double robust_error = 0.0;
    double median_error = 0.0;
    for (int trial = 0; trial < kTrials; ++trial) {
        std::vector<HeightObservation> observations;
        observations.reserve(kSamples);
        std::vector<double> heights;
        heights.reserve(kSamples);
        for (int sample = 0; sample < kSamples; ++sample) {
            const double height = normal_draw(rng, 0.1);
            observations.push_back(HeightObservation{height, 1.0});
            heights.push_back(height);
        }
        const double median_estimate = plain_median(heights);
        const double robust_estimate = fuse_cell(observations, 0.5).value;
        robust_error += robust_estimate * robust_estimate;
        median_error += median_estimate * median_estimate;
    }
    const double robust_rms = std::sqrt(robust_error / static_cast<double>(kTrials));
    const double median_rms = std::sqrt(median_error / static_cast<double>(kTrials));
    std::printf("[CellFusion.Gaussian] robust rms=%.4f m median rms=%.4f m (%.1f%% lower)\n",
                robust_rms, median_rms, 100.0 * (1.0 - robust_rms / median_rms));
    EXPECT_LT(robust_rms, 0.85 * median_rms);
}

TEST(CellFusion, GrossOutliersAreRejectedAndLoseWeight) {
    std::vector<HeightObservation> contaminated;
    contaminated.reserve(30U);
    for (int inlier = 0; inlier < 21; ++inlier) {
        contaminated.push_back(HeightObservation{10.0, 1.0});
    }
    for (int outlier = 0; outlier < 9; ++outlier) {
        contaminated.push_back(HeightObservation{15.0, 1.0});
    }
    std::vector<HeightObservation> clean(30U, HeightObservation{10.0, 1.0});
    const FusedCell bad = fuse_cell(contaminated, 0.25);
    const FusedCell good = fuse_cell(clean, 0.25);
    EXPECT_NEAR(bad.value, 10.0, 0.05);
    EXPECT_LT(bad.weight, good.weight);
}

TEST(CellFusion, BimodalStepDoesNotAverageToTheMiddle) {
    std::vector<HeightObservation> observations;
    observations.reserve(40U);
    for (int low = 0; low < 24; ++low) {
        observations.push_back(HeightObservation{10.0, 1.0});
    }
    for (int high = 0; high < 16; ++high) {
        observations.push_back(HeightObservation{13.0, 1.0});
    }
    const FusedCell fused = fuse_cell(observations, 0.25);
    // A mean would land near 11.2; the robust estimate must stay on the
    // majority mode.
    EXPECT_NEAR(fused.value, 10.0, 0.1);
}

TEST(CellFusion, WidelySpreadCellWeighsLessThanATightOne) {
    std::vector<HeightObservation> tight;
    std::vector<HeightObservation> wide;
    tight.reserve(20U);
    wide.reserve(20U);
    for (int sample = 0; sample < 20; ++sample) {
        const double sign = sample % 2 == 0 ? 1.0 : -1.0;
        tight.push_back(HeightObservation{10.0 + sign * 0.05, 1.0});
        wide.push_back(HeightObservation{10.0 + sign * 0.5, 1.0});
    }
    const FusedCell tight_fused = fuse_cell(tight, 0.25);
    const FusedCell wide_fused = fuse_cell(wide, 0.25);
    EXPECT_GT(tight_fused.weight, wide_fused.weight);
}

TEST(CellFusion, ShuffledInputIsBitIdentical) {
    std::vector<HeightObservation> original;
    original.reserve(24U);
    for (int sample = 0; sample < 24; ++sample) {
        const double height =
            10.0 + 0.1 * static_cast<double>((sample * 7) % 11 - 5);
        const double confidence = 0.5 + 0.05 * static_cast<double>(sample % 5);
        original.push_back(HeightObservation{height, confidence});
    }
    std::vector<HeightObservation> shuffled = original;
    std::mt19937 rng(987654321U);
    for (std::size_t count = shuffled.size(); count > 1U; --count) {
        const auto draw = rng();
        const std::size_t swap_with = static_cast<std::size_t>(draw) % count;
        std::swap(shuffled[count - 1U], shuffled[swap_with]);
    }
    const FusedCell first = fuse_cell(original, 0.25);
    const FusedCell second = fuse_cell(shuffled, 0.25);
    EXPECT_EQ(first.kept, second.kept);
    EXPECT_EQ(first.value, second.value);
    EXPECT_EQ(first.weight, second.weight);
}

}  // namespace
}  // namespace alphashape::elevation
