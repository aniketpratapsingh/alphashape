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
#include "common/algorithm/ortho/colour_model.hpp"
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

TEST(OrthoGain, a_camera_with_no_overlap_keeps_gain_near_one) {
    // Camera 2 overlaps nothing. Without the prior its gain would be unconstrained;
    // with it, the answer stays at the prior's 1 (the mean-normalisation of the
    // whole vector can nudge it, but it cannot be dragged to another camera's
    // ratio). Camera 2 must sit between its neighbours, not at either extreme.
    std::vector<std::array<double, 3>> true_gain = {
        {1.0, 1.0, 1.0}, {1.3, 1.3, 1.3}, {1.0, 1.0, 1.0}};
    const std::vector<std::array<double, 3>> colour = {
        {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}};
    std::vector<OverlapPair> pairs = overlaps_from_gains(2, true_gain, colour);  // only 0-1
    const auto gains = solve_gains(3, pairs);
    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(gains[2][static_cast<std::size_t>(c)], 1.0, 0.15);
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

TEST(OrthoGain, a_noisy_flight_does_not_collapse_to_one) {
    // THE BENCH REGRESSION. On the aerial-nadir flight the coarse pass finds
    // thousands of cells per overlapping pair, so the count-weighted data term
    // dwarfs the prior and the raw solution's absolute scale lands far below
    // min_gain (it sits near prior / data, i.e. ~1e-2). A solve that clamped
    // BEFORE normalising collapsed every camera onto the same bound and then
    // divided a constant by itself, returning exactly 1.0 for every camera — a
    // silent no-op that left the mosaic byte-identical to the control.
    //
    // The fixture here is that shape: 30 cameras on a 5x6 grid, adjacent and
    // diagonal overlaps, realistic per-pair sample counts, one camera truly 1.2x
    // too dark, and a small per-side content mismatch (real overlap means do not
    // agree exactly). The assertion is on the RATIOS, which are what removes a
    // seam and what the collapse destroyed.
    constexpr int kRows = 5;
    constexpr int kCols = 6;
    constexpr int kCameras = kRows * kCols;
    constexpr double kTrueDark = 1.2;  // camera 7 needs a 1.2x brightening
    constexpr double kStored = 0.35;   // typical stored colour of the shared cells
    constexpr double kCount = 1500.0;  // samples per overlapping pair
    const auto slot = [](int row, int col) { return row * kCols + col; };

    std::vector<std::array<double, 3>> true_gain(
        static_cast<std::size_t>(kCameras), std::array<double, 3>{1.0, 1.0, 1.0});
    true_gain[7] = {kTrueDark, kTrueDark, kTrueDark};

    // Deterministic per-side content mismatch, +-3%: overlap means computed from
    // different photographs of the same ground never agree exactly, and that
    // residual is what makes the raw solution's scale depart from 1 at all.
    const auto mismatch = [](int a, int b, int side) {
        const std::int64_t h = (static_cast<std::int64_t>(a) * 7919 + b * 104729 +
                                side * 15485863) %
                               101;
        return 1.0 + 0.03 * (static_cast<double>(h) / 100.0 * 2.0 - 1.0);
    };

    std::vector<OverlapPair> pairs;
    const auto add_pair = [&](int a, int b) {
        OverlapPair pair;
        pair.a = std::min(a, b);
        pair.b = std::max(a, b);
        pair.count = kCount;
        for (int c = 0; c < 3; ++c) {
            const std::size_t channel = static_cast<std::size_t>(c);
            pair.mean_a[channel] =
                kStored / true_gain[static_cast<std::size_t>(pair.a)][channel] *
                mismatch(pair.a, pair.b, 0);
            pair.mean_b[channel] =
                kStored / true_gain[static_cast<std::size_t>(pair.b)][channel] *
                mismatch(pair.a, pair.b, 1);
        }
        pairs.push_back(pair);
    };
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            for (const auto& step : {std::pair<int, int>{0, 1}, {1, 0}, {1, 1}, {1, -1}}) {
                const int rr = row + step.first;
                const int cc = col + step.second;
                if (rr < 0 || rr >= kRows || cc < 0 || cc >= kCols) {
                    continue;
                }
                add_pair(slot(row, col), slot(rr, cc));
            }
        }
    }
    ASSERT_GT(pairs.size(), 40U) << "the fixture should be a well-connected flight";

    const auto gains = solve_gains(kCameras, pairs);
    const auto rel = normalised(gains);
    const auto truth = normalised(true_gain);

    // The dark camera must come out brighter than the others by the true ratio.
    // Comparing against the normalised truth is what makes this independent of
    // the (unidentifiable) absolute scale.
    for (int i = 0; i < kCameras; ++i) {
        for (int c = 0; c < 3; ++c) {
            const double want = truth[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)];
            const double got = rel[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)];
            EXPECT_NEAR(got, want, 0.06) << "camera " << i << " channel " << c;
        }
    }
    // The specific failure signature: every camera at exactly 1, which is what
    // clamp-then-normalise produced on the bench.
    double lo = gains.front()[0];
    double hi = gains.front()[0];
    for (const auto& gain : gains) {
        lo = std::min(lo, gain[0]);
        hi = std::max(hi, gain[0]);
    }
    std::printf("[OrthoGain.NoisyFlight] red gain range %.6f .. %.6f\n", lo, hi);
    EXPECT_GT(hi - lo, 0.10)
        << "a 1.2x exposure difference must survive the solve, not collapse to 1";
    EXPECT_GT(gains[7][0], 1.05) << "the 1.2x-dark camera must be brightened";
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


// -- the affine colour model ------------------------------------------------
//
// The model is fitted on paired observations of the SAME surface point, so the
// input a unit test can hand it is exactly the pair list a real flight produces
// once the sampling stage has run. A synthetic scene with KNOWN (a, b) therefore
// asserts "recovered" against arithmetic, not against a hoped-for mosaic.
//
// The scene's ground colour varies over the lattice on purpose: with a single
// constant colour a and b are unidentifiable (a*C + b is one equation in two
// unknowns), and a fit would be free to split the same correction any way it
// liked between them. Real surfaces are never constant, which is why the
// estimator is sound on them and this fixture has to be too.

namespace {

using ortho::apply_colour;
using ortho::CameraColour;
using ortho::ColourObservation;
using ortho::pair_observations;
using ortho::solve_affine_colours;

/// A ground colour (0..1, all channels) that varies over the lattice so `a` and
/// `b` are separable: a fit on a constant colour would be as degenerate as a
/// gain-only model. The band is 0.10..0.65 on purpose -- wide enough that a and
/// b are separable, and narrow enough that neither camera's rendered value
/// clips at 0 or 1 for the gains and offsets these tests use, since a clipped
/// sample would be a fabricated observation rather than a measurement.
double scene_colour(std::size_t point, std::size_t channel) {
    const double phase = static_cast<double>((point * 37U + channel * 11U) % 251U) / 251.0;
    const double ramp = static_cast<double>(point % 97U) / 97.0;
    return 0.10 + 0.55 * (0.55 * phase + 0.45 * ramp);
}

/// Observations of one ground colour seen by every camera, with camera i
/// rendering it as (C - b_i) / a_i so that a_i * I_i + b_i = C.
std::vector<ColourObservation> observations_from_model(
    const std::vector<CameraColour>& truth, std::size_t points,
    const std::vector<std::size_t>& channels = {0, 1, 2}, std::size_t outlier_points = 0,
    std::size_t outlier_camera = 0) {
    std::vector<ColourObservation> observations;
    for (std::size_t p = 0; p < points; ++p) {
        std::array<double, 3> c{scene_colour(p, 0), scene_colour(p, 1), scene_colour(p, 2)};
        const bool outlier = p < outlier_points;
        for (std::size_t i = 0; i < truth.size(); ++i) {
            for (std::size_t j = i + 1; j < truth.size(); ++j) {
                ColourObservation o;
                o.a = static_cast<std::int32_t>(i);
                o.b = static_cast<std::int32_t>(j);
                for (const std::size_t ch : channels) {
                    const double shared = c[ch];
                    // The "moving car": for a minority of points, ONE camera
                    // sees a surface of a different colour (a vehicle over the
                    // tarmac). Perturbing whichever side of the pair that camera
                    // occupies is what an occlusion actually looks like in the
                    // observation list.
                    const double seen_a = (outlier && i == outlier_camera)
                                              ? std::min(1.0, shared + 0.30)
                                              : shared;
                    const double seen_b = (outlier && j == outlier_camera)
                                              ? std::min(1.0, shared + 0.30)
                                              : shared;
                    o.value_a[ch] = static_cast<float>(
                        std::clamp((seen_a - truth[i].offset[ch] / 255.0) / truth[i].gain[ch],
                                   0.0, 1.0));
                    o.value_b[ch] = static_cast<float>(
                        std::clamp((seen_b - truth[j].offset[ch] / 255.0) / truth[j].gain[ch],
                                   0.0, 1.0));
                }
                observations.push_back(o);
            }
        }
    }
    return observations;
}

/// The mean gain and mean offset a solve must land on, so a test can assert the
/// gauge independently of the coefficients.
std::array<double, 3> mean_gain(const std::vector<CameraColour>& colours) {
    std::array<double, 3> mean{0.0, 0.0, 0.0};
    for (const CameraColour& c : colours) {
        for (int ch = 0; ch < 3; ++ch) {
            mean[static_cast<std::size_t>(ch)] += c.gain[static_cast<std::size_t>(ch)];
        }
    }
    for (int ch = 0; ch < 3; ++ch) {
        mean[static_cast<std::size_t>(ch)] /= static_cast<double>(colours.size());
    }
    return mean;
}

}  // namespace

TEST(OrthoAffine, recovers_known_gains_and_offsets_within_tolerance) {
    // Two cameras with a KNOWN affine difference, on a scene whose colour varies
    // over the lattice (see scene_colour: a constant scene cannot separate a from
    // b). The requested tolerance is 2% on the gain and 0.005 of full scale
    // (about one grey level) on the offset, after the gauge is fixed.
    std::vector<CameraColour> truth(2);
    truth[0].gain = {1.2, 1.2, 1.2};
    truth[0].offset = {0.05 * 255.0, 0.05 * 255.0, 0.05 * 255.0};
    truth[1].gain = {0.85, 0.85, 0.85};
    truth[1].offset = {-0.03 * 255.0, -0.03 * 255.0, -0.03 * 255.0};

    const std::vector<ColourObservation> observations = observations_from_model(truth, 4000);
    const std::vector<CameraColour> solved = solve_affine_colours(2, observations);
    ASSERT_EQ(solved.size(), 2U);

    // Gauge: the solve states its answer with mean gain 1 and mean offset 0, so
    // the TRUTH is restated the same way before comparing. This is a property of
    // the model, not a relaxation of the test: (a, b) and (s a, s b + t s) are
    // the same correction up to a global brightness change, and the mosaic's own
    // brightness is what the gauge preserves.
    std::array<double, 3> truth_mean_a{0.0, 0.0, 0.0};
    std::array<double, 3> truth_mean_b{0.0, 0.0, 0.0};
    for (const CameraColour& c : truth) {
        for (int ch = 0; ch < 3; ++ch) {
            truth_mean_a[static_cast<std::size_t>(ch)] += c.gain[static_cast<std::size_t>(ch)];
            truth_mean_b[static_cast<std::size_t>(ch)] += c.offset[static_cast<std::size_t>(ch)];
        }
    }
    for (int ch = 0; ch < 3; ++ch) {
        truth_mean_a[static_cast<std::size_t>(ch)] /= 2.0;
        truth_mean_b[static_cast<std::size_t>(ch)] /= 2.0;
    }
    for (std::size_t i = 0; i < truth.size(); ++i) {
        for (int ch = 0; ch < 3; ++ch) {
            const std::size_t channel = static_cast<std::size_t>(ch);
            const double a = truth[i].gain[channel] / truth_mean_a[channel];
            const double b = (truth[i].offset[channel] - truth_mean_b[channel]) /
                             truth_mean_a[channel] / 255.0;
            const double got_a = solved[i].gain[channel];
            const double got_b = solved[i].offset[channel] / 255.0;
            EXPECT_NEAR(got_a, a, 0.02 * std::max(1.0, std::abs(a)))
                << "camera " << i << " channel " << ch;
            EXPECT_NEAR(got_b, b, 0.005) << "camera " << i << " channel " << ch;
        }
    }
    for (int ch = 0; ch < 3; ++ch) {
        EXPECT_NEAR(mean_gain(solved)[static_cast<std::size_t>(ch)], 1.0, 1e-9);
    }
}

TEST(OrthoAffine, an_outlier_patch_does_not_bias_the_fit) {
    // A moving car: 10% of the lattice points show a different colour in ONE
    // camera. Under an L2 fit those points pull that camera's a and b; under the
    // Cauchy loss they are down-weighted. The recovered coefficients must stay within
    // 1% of the clean fit, which is what "does not bias" has to mean.
    std::vector<CameraColour> truth(3);
    truth[0].gain = {1.10, 1.00, 0.92};
    truth[0].offset = {4.0, 0.0, -3.0};
    truth[1].gain = {1.00, 1.05, 1.00};
    truth[1].offset = {0.0, 2.0, 0.0};
    truth[2].gain = {0.90, 1.00, 1.12};
    truth[2].offset = {-5.0, 1.0, 6.0};

    constexpr std::size_t kPoints = 5000;
    constexpr std::size_t kOutliers = kPoints / 10;
    const std::vector<ColourObservation> clean =
        observations_from_model(truth, kPoints, {0, 1, 2}, 0, 0);
    const std::vector<ColourObservation> dirty =
        observations_from_model(truth, kPoints, {0, 1, 2}, kOutliers, 2);

    const std::vector<CameraColour> clean_solved = solve_affine_colours(3, clean);
    const std::vector<CameraColour> dirty_solved = solve_affine_colours(3, dirty);

    for (std::size_t i = 0; i < truth.size(); ++i) {
        for (int ch = 0; ch < 3; ++ch) {
            const std::size_t channel = static_cast<std::size_t>(ch);
            EXPECT_NEAR(dirty_solved[i].gain[channel], clean_solved[i].gain[channel],
                        0.01 * std::max(1.0, std::abs(clean_solved[i].gain[channel])))
                << "outliers must not move camera " << i << " channel " << ch << "'s gain";
            EXPECT_NEAR(dirty_solved[i].offset[channel], clean_solved[i].offset[channel], 2.5)
                << "outliers must not move camera " << i << " channel " << ch << "'s offset";
        }
    }
}

TEST(OrthoAffine, the_solve_is_deterministic_and_order_independent) {
    // The task samples ONCE over the whole flight and every block applies the
    // same coefficients, so the solve must not depend on how the observations
    // were enumerated. Two properties pin that: the same input twice gives
    // bit-identical coefficients, and a reversed input -- a different order of
    // the same measurements -- moves nothing by more than float rounding.
    std::vector<CameraColour> truth(4);
    truth[0].gain = {1.15, 1.02, 0.95};
    truth[0].offset = {6.0, 0.0, -4.0};
    truth[1].gain = {0.88, 1.00, 1.08};
    truth[1].offset = {-7.0, 2.0, 3.0};
    truth[2].gain = {1.00, 1.20, 1.00};
    truth[2].offset = {1.0, -5.0, 1.0};
    truth[3].gain = {1.05, 0.85, 1.10};
    truth[3].offset = {0.0, 4.0, -2.0};

    const std::vector<ColourObservation> all = observations_from_model(truth, 3000);
    std::vector<ColourObservation> reversed(all.rbegin(), all.rend());

    const std::vector<CameraColour> once = solve_affine_colours(4, all);
    const std::vector<CameraColour> twice = solve_affine_colours(4, all);
    const std::vector<CameraColour> permuted = solve_affine_colours(4, reversed);

    ASSERT_EQ(once.size(), 4U);
    for (std::size_t i = 0; i < once.size(); ++i) {
        for (int ch = 0; ch < 3; ++ch) {
            const std::size_t channel = static_cast<std::size_t>(ch);
            EXPECT_DOUBLE_EQ(once[i].gain[channel], twice[i].gain[channel])
                << "the solve must be deterministic";
            EXPECT_NEAR(permuted[i].gain[channel], once[i].gain[channel], 1e-6)
                << "enumerating the same samples in another order must not move the fit";
            EXPECT_NEAR(permuted[i].offset[channel], once[i].offset[channel], 0.05)
                << "enumerating the same samples in another order must not move the fit";
        }
    }
}

TEST(OrthoAffine, a_camera_with_no_observations_keeps_the_identity) {
    std::vector<CameraColour> truth(2);
    truth[0].gain = {1.2, 1.2, 1.2};
    truth[1].gain = {0.8, 0.8, 0.8};
    const std::vector<ColourObservation> observations = observations_from_model(truth, 500);
    const std::vector<CameraColour> solved = solve_affine_colours(4, observations);
    ASSERT_EQ(solved.size(), 4U);
    for (std::size_t i = 2; i < 4; ++i) {
        for (int ch = 0; ch < 3; ++ch) {
            EXPECT_DOUBLE_EQ(solved[i].gain[static_cast<std::size_t>(ch)], 1.0);
            EXPECT_DOUBLE_EQ(solved[i].offset[static_cast<std::size_t>(ch)], 0.0);
        }
    }
}

TEST(OrthoAffine, no_observations_leaves_every_camera_at_the_identity) {
    const std::vector<CameraColour> solved = solve_affine_colours(3, {});
    ASSERT_EQ(solved.size(), 3U);
    for (const CameraColour& c : solved) {
        for (int ch = 0; ch < 3; ++ch) {
            EXPECT_DOUBLE_EQ(c.gain[static_cast<std::size_t>(ch)], 1.0);
            EXPECT_DOUBLE_EQ(c.offset[static_cast<std::size_t>(ch)], 0.0);
        }
    }
}

TEST(OrthoAffine, apply_is_the_identity_for_an_identity_model) {
    ortho::OrthoTile tile;
    tile.width = 3;
    tile.height = 2;
    tile.valid = {1, 1, 1, 1, 1, 1};
    tile.rgb = {0, 40, 255, 10, 20, 30, 200, 100, 50,
                1, 2, 3,   4,  5,  6,  7,   8,   9};
    const std::vector<std::uint8_t> before = tile.rgb;
    apply_colour(tile, CameraColour{});
    EXPECT_EQ(tile.rgb, before);
}

TEST(OrthoAffine, apply_scales_and_offsets_only_valid_cells_and_clamps) {
    ortho::OrthoTile tile;
    tile.width = 3;
    tile.height = 1;
    tile.valid = {1, 0, 1};
    tile.rgb = {100, 10, 250, 77, 77, 77, 60, 200, 120};
    CameraColour colour;
    colour.gain = {1.5, 0.5, 1.0};
    colour.offset = {20.0, -10.0, 100.0};
    apply_colour(tile, colour);
    // Cell 0: 1.5*100+20 = 170, 0.5*10-10 = -5 -> 0, 1.0*250+100 = 350 -> 255.
    EXPECT_EQ(tile.rgb[0], 170);
    EXPECT_EQ(tile.rgb[1], 0);
    EXPECT_EQ(tile.rgb[2], 255);
    // Cell 1 is invalid: untouched.
    EXPECT_EQ(tile.rgb[3], 77);
    EXPECT_EQ(tile.rgb[4], 77);
    EXPECT_EQ(tile.rgb[5], 77);
    // Cell 2: 1.5*60+20 = 110, 0.5*200-10 = 90, 1.0*120+100 = 220.
    EXPECT_EQ(tile.rgb[6], 110);
    EXPECT_EQ(tile.rgb[7], 90);
    EXPECT_EQ(tile.rgb[8], 220);
}

TEST(OrthoAffine, pair_observations_gives_every_camera_a_share) {
    // A dense flight with a cap small relative to the candidate pairs. A
    // first-pairs-first cap spends it all on cameras 0 and 1 and leaves the high
    // indices with no observation (hence the identity); the per-camera share
    // must reach every camera.
    constexpr std::size_t kCameras = 8;
    constexpr std::size_t kPoints = 1000;
    std::vector<std::vector<int>> points_by_camera(kCameras);
    std::vector<std::vector<std::array<float, 3>>> colours(kCameras);
    for (std::size_t k = 0; k < kCameras; ++k) {
        for (std::size_t p = 0; p < kPoints; ++p) {
            points_by_camera[k].push_back(static_cast<int>(p));
            const double v =
                0.05 + 0.90 * static_cast<double>(p) / static_cast<double>(kPoints - 1);
            // Odd cameras are half as bright, so their true gain is 2: a camera
            // that keeps the identity is visible as a gain of exactly 1.
            const float raw = static_cast<float>(k % 2 == 0 ? v : v * 0.5);
            colours[k].push_back({raw, raw, raw});
        }
    }
    const std::vector<ColourObservation> observations =
        pair_observations(points_by_camera, colours, std::size_t{1600});
    EXPECT_EQ(observations.size(), std::size_t{800});
    std::vector<bool> seen(kCameras, false);
    for (const ColourObservation& o : observations) {
        seen[static_cast<std::size_t>(o.a)] = true;
        seen[static_cast<std::size_t>(o.b)] = true;
    }
    for (std::size_t k = 0; k < kCameras; ++k) {
        EXPECT_TRUE(seen[k]) << "camera " << k << " must not be starved by the cap";
    }
    const std::vector<CameraColour> solved = solve_affine_colours(8, observations);
    for (std::size_t k = 0; k < kCameras; ++k) {
        EXPECT_GT(std::abs(solved[k].gain[0] - 1.0), 0.05)
            << "camera " << k << " kept the identity although its true gain is not 1";
    }
}

TEST(OrthoAffine, clipped_samples_do_not_bias_the_fit) {
    // A clipped 0 or 255 is the sensor's ceiling, not a measurement, and an
    // affine model fitted to it would be fitting the ceiling. Drop those channel
    // samples and the recovered coefficients must match the clean solve.
    std::vector<CameraColour> truth(2);
    truth[0].gain = {1.15, 1.05, 0.95};
    truth[0].offset = {8.0, 0.0, -5.0};
    truth[1].gain = {0.88, 1.00, 1.10};
    truth[1].offset = {-6.0, 3.0, 4.0};
    const std::vector<ColourObservation> clean = observations_from_model(truth, 2000);
    std::vector<ColourObservation> clipped = clean;
    for (std::size_t i = 0; i < clipped.size(); i += 5) {
        clipped[i].value_a[0] = 1.0F;
        clipped[i].value_b[0] = 0.0F;
    }
    const std::vector<CameraColour> expected = solve_affine_colours(2, clean);
    const std::vector<CameraColour> solved = solve_affine_colours(2, clipped);
    for (std::size_t i = 0; i < truth.size(); ++i) {
        EXPECT_NEAR(solved[i].gain[0], expected[i].gain[0], 0.02)
            << "clipped red must not move camera " << i << "'s red gain";
        EXPECT_NEAR(solved[i].offset[0], expected[i].offset[0], 1.5)
            << "clipped red must not move camera " << i << "'s red offset";
        EXPECT_NEAR(solved[i].gain[1], expected[i].gain[1], 1e-9)
            << "green was untouched, so its solve must be identical";
    }
}

TEST(OrthoAffine, a_camera_with_too_few_observations_keeps_the_identity) {
    // Ten observations cannot pin a two-parameter model; the prior alone would
    // otherwise invent a correction. The camera must keep the identity instead.
    std::vector<CameraColour> truth(2);
    truth[0].gain = {1.2, 1.2, 1.2};
    truth[1].gain = {0.8, 0.8, 0.8};
    std::vector<ColourObservation> observations = observations_from_model(truth, 600);
    for (std::size_t p = 0; p < 10; ++p) {
        ColourObservation o;
        o.a = 0;
        o.b = 2;
        for (std::size_t ch = 0; ch < 3; ++ch) {
            o.value_a[ch] = static_cast<float>(scene_colour(p, ch));
            o.value_b[ch] = static_cast<float>(scene_colour(p, ch) * 0.5);
        }
        observations.push_back(o);
    }
    const std::vector<CameraColour> solved = solve_affine_colours(3, observations);
    ASSERT_EQ(solved.size(), 3U);
    for (int ch = 0; ch < 3; ++ch) {
        EXPECT_DOUBLE_EQ(solved[2].gain[static_cast<std::size_t>(ch)], 1.0);
        EXPECT_DOUBLE_EQ(solved[2].offset[static_cast<std::size_t>(ch)], 0.0);
    }
    EXPECT_GT(std::abs(solved[0].gain[0] - 1.0), 0.05)
        << "the well-observed cameras must still be corrected";
}

TEST(OrthoAffine, a_camera_with_a_flat_channel_keeps_the_identity_there) {
    // A constant green value has no contrast, so a green gain and offset cannot
    // be separated. That camera's green channel must stay the identity while its
    // other channels, which do have spread, are still corrected.
    std::vector<CameraColour> truth(3);
    truth[0].gain = {1.10, 1.00, 0.95};
    truth[0].offset = {5.0, 0.0, -3.0};
    truth[1].gain = {0.95, 1.05, 1.00};
    truth[1].offset = {0.0, 2.0, 1.0};
    truth[2].gain = {1.20, 0.90, 1.05};
    truth[2].offset = {2.0, -3.0, 4.0};
    std::vector<ColourObservation> observations = observations_from_model(truth, 800);
    for (ColourObservation& o : observations) {
        if (o.b == 2) {
            o.value_b[1] = 0.5F;
        }
    }
    const std::vector<CameraColour> solved = solve_affine_colours(3, observations);
    ASSERT_EQ(solved.size(), 3U);
    EXPECT_DOUBLE_EQ(solved[2].gain[1], 1.0);
    EXPECT_DOUBLE_EQ(solved[2].offset[1], 0.0);
    EXPECT_GT(std::abs(solved[2].gain[0] - 1.0), 0.02)
        << "the channels with real spread must still be corrected";
}

}  // namespace
}  // namespace alphashape
