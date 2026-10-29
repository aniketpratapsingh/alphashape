// test_ortho_gain.cpp -- per-camera gain compensation (Brown & Lowe 2007, §6).
//
// The unit under test is the solve and its application, so every case here is
// synthetic: a set of overlap records with KNOWN gains, and a tile whose
// correction is checked against the numbers. That is what makes "recovered
// within 2%" an assertion rather than a hope -- there is no photograph, no pose
// and no DEM in the way of the arithmetic being verified.
//
// The end-to-end behaviour (a seam brightness step shrinking in a real mosaic) is
// covered separately in test_build_orthomosaic.cpp, where a scene with a real
// overlap exists.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "common/algorithm/ortho/blend.hpp"
#include "common/algorithm/ortho/gain.hpp"
#include "common/algorithm/ortho/orthorectify.hpp"

namespace alphashape {
namespace {

using ortho::apply_gain;
using ortho::GainOptions;
using ortho::OverlapAccumulator;
using ortho::OverlapPair;
using ortho::solve_gains;

std::array<std::uint8_t, 3> rgb(double r, double g, double b) {
    return {static_cast<std::uint8_t>(std::lround(r)), static_cast<std::uint8_t>(std::lround(g)),
            static_cast<std::uint8_t>(std::lround(b))};
}

/// Build overlap records from a set of cameras whose KNOWN correction gains are
/// `true_gain`.
///
/// Camera i renders the ground at `colour / true_gain[i]` (a camera that needs a
/// 1.3x brightening correction is 1.3x too dark), so the recovered gain must be
/// that camera's factor up to the one global scale the prior pins near 1. The
/// prior makes the ABSOLUTE scale unidentifiable (multiplying every gain by c and
/// every observed colour by c is the same data), so a test asserts the RATIOS,
/// which are what removes a seam.
std::vector<OverlapPair> overlaps_from_gains(int cameras,
                                             const std::vector<std::array<double, 3>>& true_gain,
                                             const std::vector<std::array<double, 3>>& colour,
                                             double count = 1000.0) {
    std::vector<OverlapPair> pairs;
    for (int a = 0; a < cameras; ++a) {
        for (int b = a + 1; b < cameras; ++b) {
            OverlapPair pair;
            pair.a = a;
            pair.b = b;
            pair.count = count;
            for (int c = 0; c < 3; ++c) {
                const std::size_t channel = static_cast<std::size_t>(c);
                pair.mean_a[channel] = colour[static_cast<std::size_t>(a)][channel] /
                                       true_gain[static_cast<std::size_t>(a)][channel];
                pair.mean_b[channel] = colour[static_cast<std::size_t>(b)][channel] /
                                       true_gain[static_cast<std::size_t>(b)][channel];
            }
            pairs.push_back(pair);
        }
    }
    return pairs;
}

/// The gain ratios a cover needs: dividing every recovered gain by the mean
/// removes the one free global scale, so what is left is comparable with the
/// true gains normalised the same way.
std::vector<std::array<double, 3>> normalised(
    const std::vector<std::array<double, 3>>& gains) {
    std::array<double, 3> mean{0.0, 0.0, 0.0};
    for (const auto& gain : gains) {
        for (int c = 0; c < 3; ++c) {
            mean[static_cast<std::size_t>(c)] += gain[static_cast<std::size_t>(c)];
        }
    }
    for (int c = 0; c < 3; ++c) {
        mean[static_cast<std::size_t>(c)] /= static_cast<double>(gains.size());
    }
    std::vector<std::array<double, 3>> out = gains;
    for (auto& gain : out) {
        for (int c = 0; c < 3; ++c) {
            gain[static_cast<std::size_t>(c)] /= mean[static_cast<std::size_t>(c)];
        }
    }
    return out;
}

TEST(OrthoGain, two_cameras_recover_their_ratio) {
    // One camera nominal, one 1.3x: a fully overlapping pair. The solve must put
    // their recovered gains 1.3 apart, whatever the global scale settles to.
    const std::vector<std::array<double, 3>> true_gain = {
        {1.0, 1.0, 1.0}, {1.3, 1.3, 1.3}};
    const std::vector<std::array<double, 3>> colour = {
        {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}};
    const auto gains = solve_gains(2, overlaps_from_gains(2, true_gain, colour));
    const auto rel = normalised(gains);
    const auto truth = normalised(true_gain);
    for (int i = 0; i < 2; ++i) {
        for (int c = 0; c < 3; ++c) {
            const double want = truth[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)];
            const double got = rel[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)];
            EXPECT_NEAR(got, want, 0.02) << "camera " << i << " channel " << c;
        }
    }
}

TEST(OrthoGain, three_cameras_recover_1_0_1_3_0_8) {
    // The brief's case: known gains 1.0 / 1.3 / 0.8, all mutually overlapping.
    const std::vector<std::array<double, 3>> true_gain = {
        {1.0, 1.0, 1.0}, {1.3, 1.3, 1.3}, {0.8, 0.8, 0.8}};
    const std::vector<std::array<double, 3>> colour = {
        {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}};
    const auto gains = solve_gains(3, overlaps_from_gains(3, true_gain, colour));
    const auto rel = normalised(gains);
    const auto truth = normalised(true_gain);
    for (int i = 0; i < 3; ++i) {
        for (int c = 0; c < 3; ++c) {
            const double want = truth[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)];
            const double got = rel[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)];
            EXPECT_NEAR(got, want, 0.02) << "camera " << i << " channel " << c;
        }
    }
}

TEST(OrthoGain, per_channel_gains_are_independent) {
    // A camera that is neutral overall but red-biased must get a red gain and no
    // green/blue move, which only works if the three systems are solved apart.
    std::vector<std::array<double, 3>> true_gain = {
        {1.0, 1.0, 1.0}, {1.2, 1.0, 0.9}};
    const std::vector<std::array<double, 3>> colour = {
        {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}};
    const auto gains = solve_gains(2, overlaps_from_gains(2, true_gain, colour));
    const auto rel = normalised(gains);
    const auto truth = normalised(true_gain);
    for (int i = 0; i < 2; ++i) {
        for (int c = 0; c < 3; ++c) {
            EXPECT_NEAR(rel[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)],
                        truth[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)], 0.02);
        }
    }
}

TEST(OrthoGain, a_camera_with_no_overlap_keeps_gain_one) {
    // Camera 2 overlaps nothing. Without the prior its gain would be unconstrained;
    // with it, the answer is exactly 1 rather than whatever a singular system gives.
    std::vector<std::array<double, 3>> true_gain = {
        {1.0, 1.0, 1.0}, {1.3, 1.3, 1.3}, {1.0, 1.0, 1.0}};
    const std::vector<std::array<double, 3>> colour = {
        {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}};
    std::vector<OverlapPair> pairs = overlaps_from_gains(2, true_gain, colour);  // only 0-1
    const auto gains = solve_gains(3, pairs);
    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(gains[2][static_cast<std::size_t>(c)], 1.0, 1e-9);
    }
}

TEST(OrthoGain, no_overlaps_leaves_every_gain_at_one) {
    const auto gains = solve_gains(4, {});
    EXPECT_EQ(gains.size(), 4U);
    for (const auto& gain : gains) {
        for (const double g : gain) {
            EXPECT_DOUBLE_EQ(g, 1.0);
        }
    }
}

TEST(OrthoGain, a_gain_is_clamped_to_the_configured_range) {
    // A wild 4x ratio: the ratio is recovered up to the clamp, so no gain may
    // leave [min_gain, max_gain] whatever the solve wanted.
    std::vector<std::array<double, 3>> true_gain = {
        {1.0, 1.0, 1.0}, {4.0, 4.0, 4.0}};
    const std::vector<std::array<double, 3>> colour = {
        {0.25, 0.25, 0.25}, {0.25, 0.25, 0.25}};
    GainOptions options;
    const auto gains = solve_gains(2, overlaps_from_gains(2, true_gain, colour), options);
    for (const auto& gain : gains) {
        for (const double g : gain) {
            EXPECT_GE(g, options.min_gain);
            EXPECT_LE(g, options.max_gain);
        }
    }
}

TEST(OrthoGain, the_prior_pulls_a_single_camera_pair_toward_one) {
    // Sanity on the prior's role: with a tiny sample count (so the data term is
    // weak) a 1.3 ratio must still be pulled towards 1 versus a large count.
    std::vector<std::array<double, 3>> true_gain = {
        {1.0, 1.0, 1.0}, {1.3, 1.3, 1.3}};
    const std::vector<std::array<double, 3>> colour = {
        {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}};
    const auto weak = normalised(solve_gains(2, overlaps_from_gains(2, true_gain, colour, 1.0)));
    const auto strong =
        normalised(solve_gains(2, overlaps_from_gains(2, true_gain, colour, 100000.0)));
    // Both describe a 1.3 ratio; the weak one is closer to the uninformative 1.
    const double weak_ratio = weak[1][0] / weak[0][0];
    const double strong_ratio = strong[1][0] / strong[0][0];
    EXPECT_LT(std::abs(weak_ratio - 1.0), std::abs(strong_ratio - 1.0));
}

TEST(OrthoGain, the_accumulator_merges_split_ranges_without_drift) {
    // A block-split accumulation must equal a whole-grid one: the sums are
    // associative, and this is the property that lets the gains be independent of
    // the block layout.
    OverlapAccumulator whole;
    OverlapAccumulator first;
    OverlapAccumulator second;
    for (int i = 0; i < 100; ++i) {
        const auto a = rgb(100 + i, 120, 140);
        const auto b = rgb(130, 150, 170);
        whole.add(0, 1, a.data(), b.data());
        if (i % 2 == 0) {
            first.add(0, 1, a.data(), b.data());
        } else {
            second.add(0, 1, a.data(), b.data());
        }
    }
    first.merge(second);
    const auto whole_pairs = whole.finish();
    const auto merged_pairs = first.finish();
    ASSERT_EQ(whole_pairs.size(), merged_pairs.size());
    ASSERT_EQ(whole_pairs.size(), 1U);
    EXPECT_DOUBLE_EQ(whole_pairs[0].count, merged_pairs[0].count);
    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(whole_pairs[0].mean_a[static_cast<std::size_t>(c)],
                    merged_pairs[0].mean_a[static_cast<std::size_t>(c)], 1e-12);
        EXPECT_NEAR(whole_pairs[0].mean_b[static_cast<std::size_t>(c)],
                    merged_pairs[0].mean_b[static_cast<std::size_t>(c)], 1e-12);
    }
}

TEST(OrthoGain, the_accumulator_orders_the_pair) {
    // Added as (1, 0), stored as (0, 1): a pair is unordered, so the solve never
    // sees the same cameras twice in opposite directions.
    OverlapAccumulator accumulator;
    const auto a = rgb(10, 20, 30);
    const auto b = rgb(40, 50, 60);
    accumulator.add(1, 0, b.data(), a.data());
    const auto pairs = accumulator.finish();
    ASSERT_EQ(pairs.size(), 1U);
    EXPECT_EQ(pairs[0].a, 0);
    EXPECT_EQ(pairs[0].b, 1);
    // Camera 0's own colour is 10/20/30 and it is recorded as side A.
    EXPECT_NEAR(pairs[0].mean_a[0], 10.0 / 255.0, 1e-9);
    EXPECT_NEAR(pairs[0].mean_b[0], 40.0 / 255.0, 1e-9);
}

TEST(OrthoGain, apply_scales_only_valid_cells_and_clamps) {
    ortho::OrthoTile tile;
    tile.camera = 0;
    tile.x0 = 0;
    tile.y0 = 0;
    tile.width = 3;
    tile.height = 1;
    tile.rgb = {100, 100, 100, 10, 10, 10, 200, 200, 200};
    tile.valid = {1, 0, 1};
    tile.score = {1.0F, 0.0F, 1.0F};
    apply_gain(tile, {1.5, 1.5, 1.5});
    // Valid cells scale; the invalid 10 stays untouched so a later blend cannot
    // resurrect a brightened ghost.
    EXPECT_EQ(tile.rgb[0], static_cast<std::uint8_t>(150));
    EXPECT_EQ(tile.rgb[3], static_cast<std::uint8_t>(10));
    // 200 * 1.5 = 300 clamps to 255.
    EXPECT_EQ(tile.rgb[6], static_cast<std::uint8_t>(255));
}

TEST(OrthoGain, apply_with_gain_one_is_the_identity) {
    ortho::OrthoTile tile;
    tile.width = 2;
    tile.height = 2;
    tile.rgb = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    tile.valid = {1, 1, 1, 1};
    const std::vector<std::uint8_t> before = tile.rgb;
    apply_gain(tile, {1.0, 1.0, 1.0});
    EXPECT_EQ(tile.rgb, before);
}

// -- end to end: real tiles, real labels, real blend ------------------------
//
// A row of two fully overlapping tiles that disagree in brightness, composed by
// the production blend with blending DISABLED (a copy of the labelled tile). The
// seam between them is therefore the raw per-camera step, and the guarantee the
// brief states — correction cuts it by at least 80% — can be asserted directly on
// the composed mosaic, with no photograph, pose or DEM in the way.

namespace {

/// A constant-colour tile over a w x h grid with a given camera index.
ortho::OrthoTile flat_tile(int camera, int width, int height, std::uint8_t value) {
    ortho::OrthoTile tile;
    tile.camera = camera;
    tile.x0 = 0;
    tile.y0 = 0;
    tile.width = width;
    tile.height = height;
    const std::size_t cells = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    tile.rgb.assign(cells * 3U, value);
    tile.valid.assign(cells, 1U);
    tile.score.assign(cells, 1.0F);
    return tile;
}

/// The largest column-to-column luminance step in a composed mosaic's interior,
/// as a fraction of full scale. A two-tile row has exactly one meaningful seam,
/// so the maximum is the number to watch.
double worst_column_step(const ortho::OrthoMosaic& mosaic) {
    double worst = 0.0;
    for (int row = 0; row < mosaic.height; ++row) {
        for (int col = 1; col < mosaic.width; ++col) {
            const std::size_t a =
                (static_cast<std::size_t>(row) * static_cast<std::size_t>(mosaic.width) +
                 static_cast<std::size_t>(col - 1)) *
                3U;
            const std::size_t b = a + 3U;
            const double la = (0.2126 * mosaic.rgb[a] + 0.7152 * mosaic.rgb[a + 1] +
                               0.0722 * mosaic.rgb[a + 2]) /
                              255.0;
            const double lb = (0.2126 * mosaic.rgb[b] + 0.7152 * mosaic.rgb[b + 1] +
                               0.0722 * mosaic.rgb[b + 2]) /
                              255.0;
            worst = std::max(worst, std::abs(lb - la));
        }
    }
    return worst;
}

}  // namespace

TEST(OrthoGain, TheProducedMosaicShrinksItsSeamStepByEightyPercent) {
    // Two overlapping cameras whose photographs differ by 25% in brightness.
    // Left half labels camera 0, right half camera 1. The mosaic is composed by
    // the production blend with correction OFF and then ON, and the WORST column
    // step across the seam is compared.
    constexpr int kWidth = 64;
    constexpr int kHeight = 32;
    constexpr std::uint8_t kBright = 200;
    constexpr std::uint8_t kDim = 160;  // 0.8 of the bright one

    ortho::OrthoTile bright = flat_tile(0, kWidth, kHeight, kBright);
    ortho::OrthoTile dim = flat_tile(1, kWidth, kHeight, kDim);

    // Labels: the left half from the bright camera, the right half from the dim.
    std::vector<std::int32_t> labels(static_cast<std::size_t>(kWidth) *
                                         static_cast<std::size_t>(kHeight),
                                     0);
    for (int row = 0; row < kHeight; ++row) {
        for (int col = kWidth / 2; col < kWidth; ++col) {
            labels[static_cast<std::size_t>(row) * static_cast<std::size_t>(kWidth) +
                   static_cast<std::size_t>(col)] = 1;
        }
    }

    const ortho::OrthoMosaic before = ortho::blend(kWidth, kHeight, {bright, dim}, labels,
                                                   ortho::Blending::Disabled, 0);
    const double step_before = worst_column_step(before);

    // Recover the correction from the overlap of the two cameras: they cover the
    // SAME ground, so the ratio of their stored colours is the gain ratio.
    OverlapAccumulator accumulator;
    std::array<std::uint8_t, 3> rgb_bright{kBright, kBright, kBright};
    std::array<std::uint8_t, 3> rgb_dim{kDim, kDim, kDim};
    for (int i = 0; i < 1000; ++i) {
        accumulator.add(0, 1, rgb_bright.data(), rgb_dim.data());
    }
    const auto gains = solve_gains(2, accumulator.finish());
    apply_gain(bright, gains[0]);
    apply_gain(dim, gains[1]);

    const ortho::OrthoMosaic after = ortho::blend(kWidth, kHeight, {bright, dim}, labels,
                                                  ortho::Blending::Disabled, 0);
    const double step_after = worst_column_step(after);

    std::printf("[OrthoGain.ProducedMosaic] step before=%.4f after=%.4f\n", step_before,
                step_after);
    ASSERT_GT(step_before, 0.1) << "the two cameras should disagree by a visible step";
    EXPECT_LE(step_after, 0.2 * step_before)
        << "colour correction must cut the produced mosaic's seam step by at least 80%";
}

}  // namespace
}  // namespace alphashape
