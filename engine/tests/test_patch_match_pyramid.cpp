// Coarse-to-fine (pyramid) PatchMatch: WP-17.14.
//
// Two things are tested here and they are deliberately separate:
//
//   * the pyramid PLAN and the CARRY -- the level count, the iteration split,
//     the level seed keys, the hole-aware bilinear upsample and the local search
//     radius. These are pure functions with no device, no threads and no cost
//     model, so they are asserted exactly;
//   * the SEARCH ON A PYRAMID -- that running a synthetic plane coarse-to-fine
//     recovers the same surface as one flat level (the point of the change), that
//     an untextured patch is seeded through the coarse level, and that the whole
//     thing is bit-identical run to run.
//
// The timing test is the one whose bound depends on the machine, so it is gated
// the way the rest of the tree gates timing: ALPHASHAPE_PERF_ASSERTS=0 (the
// shared CI runners) reports the numbers without failing, and a sanitizer build
// skips the comparison outright because a 1.6x wall-clock ratio measured on an
// instrumented binary is not a measurement of anything.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base/utils/error.hpp"
#include "common/algorithm/stereo/depth_pyramid.hpp"
#include "common/algorithm/stereo/patch_match.hpp"
#include "image/image.hpp"
#include "model/camera/calibration.hpp"
#include "perf_gate.hpp"

namespace {

using alphashape::Calibration;
using alphashape::FloatImage;
using alphashape::stereo::DepthHypotheses;
using alphashape::stereo::PMView;
using alphashape::stereo::PatchMatchOptions;
using alphashape::stereo::PatchMatchResult;

// ---------------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------------

TEST(PatchMatchPyramid, LevelCountAutomaticHalvesToMinSize) {
    // 1600x1200 at a 256 floor: 1600 -> 800 -> 400 -> 200 (below 256), so 3
    // levels and a coarsest level of 400 px. The rule halves while the SHORTER
    // side stays at or above min_size.
    EXPECT_EQ(alphashape::stereo::pyramid_level_count(1600, 1200, 0, 256, 5), 3);
    // 512x512 with the same floor: 512 -> 256 (at the floor, allowed), so 2.
    EXPECT_EQ(alphashape::stereo::pyramid_level_count(512, 512, 0, 256, 5), 2);
    // A level count is capped by the geometry: the coarsest level needs an
    // interior pixel, i.e. more than 2 * radius on both sides.
    EXPECT_EQ(alphashape::stereo::pyramid_level_count(1600, 1200, 0, 16, 5), 7);
    // An explicit count is honoured, and 0/1 keep their meanings.
    EXPECT_EQ(alphashape::stereo::pyramid_level_count(1600, 1200, 1, 256, 5), 1);
    EXPECT_EQ(alphashape::stereo::pyramid_level_count(1600, 1200, 2, 256, 5), 2);
    EXPECT_EQ(alphashape::stereo::pyramid_level_count(1600, 1200, 3, 256, 5), 3);
    // A huge request is clamped to what the geometry can do, never rejected.
    EXPECT_EQ(alphashape::stereo::pyramid_level_count(1600, 1200, 99, 256, 5), 7);
    // A tiny image has no pyramid at all.
    EXPECT_EQ(alphashape::stereo::pyramid_level_count(12, 12, 0, 256, 5), 1);

    EXPECT_THROW(alphashape::stereo::pyramid_level_count(1600, 1200, 0, 0, 5),
                 alphashape::ValueError);
    EXPECT_THROW(alphashape::stereo::pyramid_level_count(1600, 1200, -1, 256, 5),
                 alphashape::ValueError);
}

TEST(PatchMatchPyramid, IterationPlanSplitsFinestForty) {
    // The documented schedule, exactly.
    EXPECT_EQ(alphashape::stereo::pyramid_iterations(1, 6), std::vector<int>({6}));
    EXPECT_EQ(alphashape::stereo::pyramid_iterations(2, 6), std::vector<int>({4, 2}));
    EXPECT_EQ(alphashape::stereo::pyramid_iterations(3, 6), std::vector<int>({2, 2, 2}));
    // Two levels and ten sweeps: the finest takes floor(40%) = 4 and the
    // coarsest the remaining 6, so the coarse level (a quarter of the pixels)
    // carries most of the sweeps.
    EXPECT_EQ(alphashape::stereo::pyramid_iterations(2, 10), std::vector<int>({6, 4}));

    // The budget cannot buy the requested count: levels are dropped rather than
    // given less than the minimum, and at least one level always survives.
    EXPECT_EQ(alphashape::stereo::pyramid_iterations(4, 2), std::vector<int>({2}));
    EXPECT_EQ(alphashape::stereo::pyramid_iterations(4, 1), std::vector<int>({1}));
    EXPECT_EQ(alphashape::stereo::pyramid_iterations(9, 6), std::vector<int>({2, 2, 2}));
    // Zero sweeps is a single random initialisation and stays one level.
    EXPECT_EQ(alphashape::stereo::pyramid_iterations(3, 0), std::vector<int>({0}));

    // Whatever the count, the plan sums to the budget and every entry is at
    // least one, which is what makes the length of the plan the effective level
    // count.
    for (int level_count = 1; level_count <= 8; ++level_count) {
        for (int iterations = 0; iterations <= 24; ++iterations) {
            const std::vector<int> plan =
                alphashape::stereo::pyramid_iterations(level_count, iterations);
            ASSERT_FALSE(plan.empty()) << level_count << " levels, " << iterations << " sweeps";
            int sum = 0;
            for (const int per_level : plan) {
                EXPECT_GE(per_level, iterations == 0 ? 0 : 1);
                sum += per_level;
            }
            EXPECT_EQ(sum, iterations) << level_count << " levels, " << iterations << " sweeps";
            EXPECT_LE(static_cast<int>(plan.size()), std::max(level_count, 1));
        }
    }
}

TEST(PatchMatchPyramid, LevelSeedKeepsTheFinestStream) {
    // Level 0 IS the configured seed: a single-level run (and the reference
    // level of a pyramid) must draw exactly the numbers it drew before the
    // pyramid existed. A coarser level must not share that stream, or two levels
    // would explore the same random planes.
    const std::uint64_t seed = 0x0123456789ABCDEFULL;
    EXPECT_EQ(alphashape::stereo::pyramid_level_seed(seed, 0), seed);
    EXPECT_NE(alphashape::stereo::pyramid_level_seed(seed, 1), seed);
    EXPECT_NE(alphashape::stereo::pyramid_level_seed(seed, 1),
              alphashape::stereo::pyramid_level_seed(seed, 2));
    // Deterministic: the same input always gives the same key.
    EXPECT_EQ(alphashape::stereo::pyramid_level_seed(seed, 3),
              alphashape::stereo::pyramid_level_seed(seed, 3));
    // Different seeds stay different, including 0, which a caller may pass.
    EXPECT_NE(alphashape::stereo::pyramid_level_seed(seed, 2),
              alphashape::stereo::pyramid_level_seed(seed + 1, 2));
    EXPECT_EQ(alphashape::stereo::pyramid_level_seed(0, 0), 0ULL);
}

TEST(PatchMatchPyramid, DownsampleIsTheWorkingReduction) {
    // The pyramid's reduction is the same box 2x2 mean the task's downscale
    // uses, so a pyramid over an already reduced image is a further reduction
    // of the original (and not some other filter).
    FloatImage image(4, 2);
    image.at(0, 0) = 0.0F;
    image.at(1, 0) = 1.0F;
    image.at(2, 0) = 2.0F;
    image.at(3, 0) = 3.0F;
    image.at(0, 1) = 10.0F;
    image.at(1, 1) = 11.0F;
    image.at(2, 1) = 12.0F;
    image.at(3, 1) = 13.0F;
    const FloatImage half = alphashape::stereo::pyramid_downsample(image);
    ASSERT_EQ(half.width(), 2);
    ASSERT_EQ(half.height(), 1);
    EXPECT_FLOAT_EQ(half.at(0, 0), 5.5F);   // (0 + 1 + 10 + 11) / 4
    EXPECT_FLOAT_EQ(half.at(1, 0), 7.5F);   // (2 + 3 + 12 + 13) / 4

    // An odd size truncates, exactly as downsample_box does.
    const FloatImage odd(5, 3);
    EXPECT_EQ(alphashape::stereo::pyramid_downsample(odd).width(), 2);
    EXPECT_EQ(alphashape::stereo::pyramid_downsample(odd).height(), 1);
    EXPECT_THROW(alphashape::stereo::pyramid_downsample(FloatImage(1, 1)),
                 alphashape::ValueError);
}

TEST(PatchMatchPyramid, BuiltPyramidIsConsistentWithItsCalibrations) {
    // Every level's image must match its calibration, and scaled_calibration's
    // bearing identity must hold: the coarse pixel's bearing equals the fine
    // index 2j + 0.5's bearing. That identity is what makes the depth carry
    // exact, so it is worth pinning.
    const int width = 320;
    const int height = 240;
    Calibration calibration =
        Calibration::initial(alphashape::CameraType::Frame, width, height, 300.0);
    calibration.cx = 4.5;  // an off-centre principal point, to catch a naive halving
    calibration.cy = -3.25;
    const FloatImage reference(width, height, 0.5F);

    const alphashape::stereo::DepthPyramid pyramid =
        alphashape::stereo::build_depth_pyramid(reference, calibration, 3);
    ASSERT_EQ(pyramid.levels(), 3);
    for (int level = 0; level < pyramid.levels(); ++level) {
        const FloatImage& image = pyramid.image(level);
        const Calibration from_finest = pyramid.calibration_from_finest(
            pyramid.levels() - 1 - level);
        EXPECT_EQ(image.width(), from_finest.width) << "level " << level;
        EXPECT_EQ(image.height(), from_finest.height) << "level " << level;
        // The finest level is the caller's own image, never a copy.
        if (level == pyramid.levels() - 1) {
            EXPECT_EQ(&image, &reference);
        }
    }

    // The bearing identity, one level down: bearing of coarse j == bearing of
    // fine 2j + 0.5, for a grid of pixels.
    for (int level = pyramid.levels() - 2; level >= 0; --level) {
        const Calibration coarse = pyramid.calibration_from_finest(pyramid.levels() - 1 - level);
        const Calibration fine = pyramid.calibration_from_finest(pyramid.levels() - 2 - level);
        for (int y = 0; y < std::min(coarse.height, 20); ++y) {
            for (int x = 0; x < std::min(coarse.width, 20); ++x) {
                const Eigen::Vector3d b_coarse = coarse.unproject(
                    Eigen::Vector2d(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5));
                const Eigen::Vector3d b_fine = fine.unproject(
                    Eigen::Vector2d(static_cast<double>(2 * x) + 1.0,
                                    static_cast<double>(2 * y) + 1.0));
                EXPECT_LT((b_coarse - b_fine).norm(), 1e-12)
                    << "level " << level << " pixel " << x << "," << y;
            }
        }
    }

    EXPECT_THROW(alphashape::stereo::build_depth_pyramid(reference, calibration, 0),
                 alphashape::ValueError);
    EXPECT_THROW(alphashape::stereo::build_depth_pyramid(FloatImage(8, 8), calibration, 2),
                 alphashape::ValueError);
}

TEST(PatchMatchPyramid, OddSizesKeepTheBearingIdentityAtEveryLevel) {
    // The identity "coarse pixel j is the bearing of fine 2j + 0.5" is what makes
    // the depth carry exact, and it has to keep holding when the halved sizes
    // TRUNCATE: 101x77 -> 50x38 -> 25x19 is the case where a one-shot
    // scaled_calibration(full, 4) would differ from two halvings by a quarter
    // pixel, so the level-by-level chain is the one that must be pinned.
    const int width = 101;
    const int height = 77;
    Calibration calibration =
        Calibration::initial(alphashape::CameraType::Frame, width, height, 140.0);
    calibration.cx = 3.75;  // off-centre, so a naive halving of the principal point fails
    calibration.cy = -2.125;
    const FloatImage reference(width, height, 0.5F);

    const alphashape::stereo::DepthPyramid pyramid =
        alphashape::stereo::build_depth_pyramid(reference, calibration, 3);
    ASSERT_EQ(pyramid.levels(), 3);
    // The sizes truncate as the box reduction does.
    EXPECT_EQ(pyramid.calibration_from_finest(0).width, 101);
    EXPECT_EQ(pyramid.calibration_from_finest(1).width, 50);
    EXPECT_EQ(pyramid.calibration_from_finest(2).width, 25);
    EXPECT_EQ(pyramid.calibration_from_finest(0).height, 77);
    EXPECT_EQ(pyramid.calibration_from_finest(1).height, 38);
    EXPECT_EQ(pyramid.calibration_from_finest(2).height, 19);

    // Every level pair, including the coarse pair (levels 2 and 1, i.e. a level
    // two steps from the reference), and EVERY pixel of the coarse level rather
    // than a corner: the principal point error a one-shot scaling would introduce
    // grows with x and y.
    //
    // The bearing to compare is fine INDEX 2j + 0.5 -- the midpoint of the two
    // fine pixels the coarse one covers -- and unproject takes a continuous
    // coordinate, where fine index i is the centre i + 0.5. So the pixel
    // coordinate is (2j + 0.5) + 0.5 = 2j + 1.0.
    for (int from_finest = 2; from_finest >= 1; --from_finest) {
        const Calibration coarse = pyramid.calibration_from_finest(from_finest);
        const Calibration fine = pyramid.calibration_from_finest(from_finest - 1);
        for (int y = 0; y < coarse.height; ++y) {
            for (int x = 0; x < coarse.width; ++x) {
                const Eigen::Vector3d b_coarse = coarse.unproject(
                    Eigen::Vector2d(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5));
                const Eigen::Vector3d b_fine = fine.unproject(
                    Eigen::Vector2d(static_cast<double>(2 * x) + 1.0,
                                    static_cast<double>(2 * y) + 1.0));
                EXPECT_LT((b_coarse - b_fine).norm(), 1e-9)
                    << "from_finest " << from_finest << " pixel " << x << "," << y;
            }
        }
    }

    // And the upsample mapping is the same identity read backwards: fine pixel i
    // samples the coarse map at continuous coarse coordinate (i - 0.5) / 2, i.e.
    // i/2 - 0.25, which is the continuation of the bearing above. Checked
    // against a coarse map that is linear in the coarse x coordinate, so a wrong
    // offset shows up as a wrong value rather than as an interpolation tolerance.
    DepthHypotheses coarse_map;
    coarse_map.width = 25;  // the level-2 size of the chain above
    coarse_map.height = 19;
    const std::size_t coarse_cells = static_cast<std::size_t>(coarse_map.width) *
                                     static_cast<std::size_t>(coarse_map.height);
    coarse_map.depth.assign(coarse_cells, 0.0F);
    coarse_map.normals.assign(3 * coarse_cells, 0.0F);
    for (int y = 0; y < coarse_map.height; ++y) {
        for (int x = 0; x < coarse_map.width; ++x) {
            const std::size_t p = static_cast<std::size_t>(y) *
                                      static_cast<std::size_t>(coarse_map.width) +
                                  static_cast<std::size_t>(x);
            coarse_map.depth[p] = 10.0F + static_cast<float>(x);
            coarse_map.normals[3 * p + 2] = 1.0F;
        }
    }
    const DepthHypotheses fine_map = alphashape::stereo::pyramid_upsample(coarse_map, 50, 38);
    for (int i = 0; i < 50; ++i) {
        // Interior only: the edge clamps the sample, which is a deliberate
        // different behaviour (see pyramid_upsample).
        if (i < 2 || i > 47) {
            continue;
        }
        const double coarse_x = 0.5 * static_cast<double>(i) - 0.25;
        const double expected = 10.0 + coarse_x;
        EXPECT_NEAR(fine_map.depth[static_cast<std::size_t>(i)], expected, 1e-4)
            << "fine x " << i << " should sample coarse " << coarse_x;
    }
}

TEST(PatchMatchPyramid, SingleLevelDoesNotNeedAMinSize) {
    // run_patch_match's own validation accepts `pyramid_levels == 1` with
    // `pyramid_min_size == 0` (there is no halving to stop), so the level count
    // must accept it too rather than contradicting the search it plans for.
    EXPECT_EQ(alphashape::stereo::pyramid_level_count(1600, 1200, 1, 0, 5), 1);
    EXPECT_EQ(alphashape::stereo::pyramid_level_count(1600, 1200, 1, -5, 5), 1);
    // A pyramid still has to say where it stops.
    EXPECT_THROW(alphashape::stereo::pyramid_level_count(1600, 1200, 0, 0, 5),
                 alphashape::ValueError);
    EXPECT_THROW(alphashape::stereo::pyramid_level_count(1600, 1200, 3, 0, 5),
                 alphashape::ValueError);
    // And the plan for such a run is a single level, which is what the search
    // then executes.
    PatchMatchOptions options;
    options.pyramid_levels = 1;
    options.pyramid_min_size = 0;
    options.iterations = 6;
    const std::vector<int> plan =
        alphashape::stereo::patch_match_iteration_plan(1600, 1200, options);
    EXPECT_EQ(plan, std::vector<int>({6}));
}

TEST(PatchMatchPyramid, UpsampleCarriesDepthsAndRenormalisesNormals) {
    // A coarse map with one hole in it. The fine map must interpolate the
    // visible taps only: the cell sitting on the hole must not be dragged toward
    // -1, and the cell whose four taps are all holes must stay a hole.
    DepthHypotheses coarse;
    coarse.width = 3;
    coarse.height = 1;
    coarse.depth = {10.0F, -1.0F, 20.0F};
    coarse.normals = {0.0F, 0.0F, -1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, -1.0F};

    const DepthHypotheses fine = alphashape::stereo::pyramid_upsample(coarse, 6, 2);
    ASSERT_EQ(fine.width, 6);
    ASSERT_EQ(fine.height, 2);
    ASSERT_EQ(fine.depth.size(), 12U);

    // Fine x samples coarse at x/2 - 0.25: x=0 -> -0.25 (clamped to 0), so the
    // left edge is the left coarse value.
    EXPECT_NEAR(fine.depth[0], 10.0, 1e-4);
    // Fine x samples coarse at (x - 0.5) / 2, the bearing identity above:
    // x=0 -> -0.25 (clamped to the left edge), so the left edge is the left
    // coarse value.
    EXPECT_NEAR(fine.depth[0], 10.0, 1e-4);
    // x=2 sits at coarse 0.75, between 0 and 1 -- but 1 is a hole, so only the
    // valid tap contributes and the value stays at the neighbour's, not diluted.
    EXPECT_NEAR(fine.depth[2], 10.0, 1e-4);
    // x=3 sits at coarse 1.25, past the hole and on the way to 20, so it is 20.
    EXPECT_NEAR(fine.depth[3], 20.0, 1e-4);
    // x=4 sits at coarse 1.75, past the hole: 20.
    EXPECT_NEAR(fine.depth[4], 20.0, 1e-4);
    EXPECT_NEAR(fine.depth[5], 20.0, 1e-4);

    // Normals are renormalised unit vectors wherever a depth was carried.
    for (std::size_t p = 0; p < fine.depth.size(); ++p) {
        if (!(fine.depth[p] > 0.0F)) {
            continue;
        }
        const double nx = static_cast<double>(fine.normals[3 * p + 0]);
        const double ny = static_cast<double>(fine.normals[3 * p + 1]);
        const double nz = static_cast<double>(fine.normals[3 * p + 2]);
        EXPECT_NEAR(std::sqrt(nx * nx + ny * ny + nz * nz), 1.0, 1e-5);
    }

    // A coarse map that is all holes produces an all-hole fine map: the finer
    // level then seeds those pixels from its own random draw.
    DepthHypotheses empty;
    empty.width = 2;
    empty.height = 2;
    empty.depth.assign(4, -1.0F);
    empty.normals.assign(12, 0.0F);
    const DepthHypotheses no_hypotheses = alphashape::stereo::pyramid_upsample(empty, 4, 4);
    for (const float d : no_hypotheses.depth) {
        EXPECT_FLOAT_EQ(d, -1.0F);
    }

    EXPECT_THROW(alphashape::stereo::pyramid_upsample(coarse, 0, 4), alphashape::ValueError);
    EXPECT_THROW(alphashape::stereo::pyramid_upsample(DepthHypotheses{}, 4, 4),
                 alphashape::ValueError);
}

TEST(PatchMatchPyramid, SearchSpanMeasuresLocalVariation) {
    // A flat map has no variation, so the span falls to its floor.
    const int width = 64;
    const int height = 64;
    std::vector<float> flat(static_cast<std::size_t>(width) * height, 12.0F);
    EXPECT_DOUBLE_EQ(
        alphashape::stereo::pyramid_search_span(flat, width, height, 3, 0.01), 0.01);

    // A map whose inverse depth varies by 0.02 across every window gives a
    // median spread of about 0.02: check it lands in that band rather than at
    // the floor or the cap.
    std::vector<float> slanted(static_cast<std::size_t>(width) * height, 0.0F);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // inverse depth 1/10 + x * 0.001, i.e. a depth from 10 down to ~9.4.
            const double inv = 0.1 + 0.001 * static_cast<double>(x);
            slanted[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                    static_cast<std::size_t>(x)] = static_cast<float>(1.0 / inv);
        }
    }
    const double span = alphashape::stereo::pyramid_search_span(slanted, width, height, 3, 0.001);
    // Over a radius-3 window the inverse-depth range is 6 * 0.001 = 0.006.
    EXPECT_GT(span, 0.005);
    EXPECT_LT(span, 0.007);

    // An empty or all-hole map has nothing to measure and returns the floor.
    std::vector<float> holes(static_cast<std::size_t>(width) * height, -1.0F);
    EXPECT_DOUBLE_EQ(alphashape::stereo::pyramid_search_span(holes, width, height, 3, 0.02),
                     0.02);

    EXPECT_THROW(
        alphashape::stereo::pyramid_search_span(slanted, width, height, 3, 0.0),
        alphashape::ValueError);
}

// ---------------------------------------------------------------------------
// The search on a pyramid
// ---------------------------------------------------------------------------

/// A textured plane seen by a reference and four neighbours, with the analytic
/// depth along each pixel's bearing. Modelled on test_patch_match.cpp's scene.
struct PlaneScene {
    FloatImage reference;
    std::vector<FloatImage> neighbours;
    Calibration calibration;
    std::vector<Eigen::Matrix3d> rotations;
    std::vector<Eigen::Vector3d> translations;
    double plane_depth = 10.0;
};

float hash2i(int x, int y) {
    std::uint32_t h = static_cast<std::uint32_t>(x) * 374761393U +
                      static_cast<std::uint32_t>(y) * 668265263U;
    h = (h ^ (h >> 13)) * 1274126177U;
    h ^= h >> 16;
    return static_cast<float>(h & 0xFFFFFFU) / static_cast<float>(0xFFFFFFU);
}

/// A band-limited but clearly textured field: the same smooth sines and value
/// noise the CPU suite uses, which survives the box reduction with the coarse
/// level still correlatable.
float plane_texture(double a, double b) {
    const double noise = 0.14 * static_cast<double>(
                                     hash2i(static_cast<int>(std::floor(a * 3.0)),
                                            static_cast<int>(std::floor(b * 3.0))) -
                                     0.5F);
    double value = 0.5 + 0.22 * std::sin(1.7 * a + 0.3) + 0.18 * std::sin(2.3 * b + 1.1) +
                   0.12 * std::sin(1.1 * a + 1.9 * b + 0.7) + noise;
    if (value < 0.0) {
        value = 0.0;
    }
    if (value > 1.0) {
        value = 1.0;
    }
    return static_cast<float>(value);
}

PlaneScene make_plane_scene(int width, int height, double focal) {
    PlaneScene scene;
    scene.calibration = Calibration::initial(alphashape::CameraType::Frame, width, height, focal);
    const Eigen::Vector3d centres[5] = {
        Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.8, 0.0, 0.0),
        Eigen::Vector3d(-0.6, 0.35, 0.0), Eigen::Vector3d(0.4, -0.9, 0.25),
        Eigen::Vector3d(-0.5, -0.45, -0.3)};
    const Eigen::Vector3d target(0.0, 0.0, 10.0);
    const Eigen::Vector3d normal(0.0, 0.0, -1.0);
    for (int v = 0; v < 5; ++v) {
        const Eigen::Vector3d c = centres[v];
        Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
        if (c.norm() > 1e-12) {
            const Eigen::Vector3d forward = (target - c).normalized();
            Eigen::Vector3d x = Eigen::Vector3d(0.0, 1.0, 0.0).cross(forward);
            if (x.norm() < 1e-6) {
                x = Eigen::Vector3d(1.0, 0.0, 0.0);
            }
            x.normalize();
            const Eigen::Vector3d y = forward.cross(x);
            r.row(0) = x;
            r.row(1) = y;
            r.row(2) = forward;
        }
        scene.rotations.push_back(r);
        scene.translations.push_back(-r * c);
        FloatImage image(width, height, 0.0F);
        const Eigen::Matrix3d rt = r.transpose();
        const Eigen::Vector3d centre = -rt * scene.translations.back();
        const double d_plane = normal.dot(Eigen::Vector3d(0.0, 0.0, 10.0));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const Eigen::Vector3d b = scene.calibration.unproject(
                    Eigen::Vector2d(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5));
                const Eigen::Vector3d dir = rt * b;
                const double denom = normal.dot(dir);
                if (std::abs(denom) < 1e-9) {
                    continue;
                }
                const double depth = (d_plane - normal.dot(centre)) / denom;
                if (depth > 0.0) {
                    const Eigen::Vector3d p = centre + depth * dir;
                    image.at(x, y) = plane_texture(p.x(), p.y());
                }
            }
        }
        if (v == 0) {
            scene.reference = image;
        } else {
            scene.neighbours.push_back(image);
        }
    }
    return scene;
}

std::vector<PMView> plane_views(PlaneScene& scene) {
    std::vector<PMView> views;
    for (std::size_t i = 0; i < scene.neighbours.size(); ++i) {
        PMView view;
        view.image = &scene.neighbours[i];
        view.calibration = scene.calibration;
        view.R = scene.rotations[i + 1];
        view.t = scene.translations[i + 1];
        views.push_back(view);
    }
    return views;
}

PMView plane_reference(PlaneScene& scene) {
    PMView reference;
    reference.image = &scene.reference;
    reference.calibration = scene.calibration;
    reference.R = scene.rotations[0];
    reference.t = scene.translations[0];
    return reference;
}

/// Relative depth error over the interior against the analytic plane.
struct Accuracy {
    double coverage = 0.0;
    double median = 0.0;
    double p90 = 0.0;
    std::size_t covered = 0;
};

Accuracy measure(const PlaneScene& scene, const PatchMatchResult& result, int radius) {
    std::vector<double> errors;
    const int width = scene.reference.width();
    const int height = scene.reference.height();
    const Eigen::Matrix3d rt = scene.rotations[0].transpose();
    const Eigen::Vector3d centre = -rt * scene.translations[0];
    const Eigen::Vector3d normal(0.0, 0.0, -1.0);
    const double d_plane = normal.dot(Eigen::Vector3d(0.0, 0.0, 10.0));
    int interior = 0;
    int covered = 0;
    for (int y = radius; y < height - radius; ++y) {
        for (int x = radius; x < width - radius; ++x) {
            const Eigen::Vector3d b = scene.calibration.unproject(
                Eigen::Vector2d(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5));
            const double denom = normal.dot(rt * b);
            if (std::abs(denom) < 1e-9) {
                continue;
            }
            const double expected = (d_plane - normal.dot(centre)) / denom;
            if (!(expected > 0.0)) {
                continue;
            }
            ++interior;
            const std::size_t index =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(x);
            if (!(result.depth[index] > 0.0F)) {
                continue;
            }
            ++covered;
            errors.push_back(std::abs(static_cast<double>(result.depth[index]) - expected) /
                             expected);
        }
    }
    std::sort(errors.begin(), errors.end());
    Accuracy accuracy;
    accuracy.coverage =
        interior > 0 ? static_cast<double>(covered) / static_cast<double>(interior) : 0.0;
    accuracy.covered = static_cast<std::size_t>(covered);
    if (!errors.empty()) {
        accuracy.median = errors[errors.size() / 2];
        accuracy.p90 = errors[(errors.size() * 9) / 10];
    }
    return accuracy;
}

PatchMatchOptions scene_options() {
    PatchMatchOptions options;
    options.depth_min = 5.0;
    options.depth_max = 20.0;
    options.window_radius = 5;
    options.window_step = 2;
    options.iterations = 6;
    options.max_cost = 0.5;
    options.seed = 12345;
    options.pyramid_levels = 1;
    return options;
}

// ---------------------------------------------------------------------------
// Pyramid matches a single level on a smooth scene
// ---------------------------------------------------------------------------

TEST(PatchMatchPyramid, PyramidMatchesSingleLevelOnSmooth) {
    PlaneScene scene = make_plane_scene(320, 240, 300.0);
    const PMView reference = plane_reference(scene);
    const std::vector<PMView> all_views = plane_views(scene);
    // Three neighbours and four sweeps keep this a light test; the plan still
    // spends two sweeps at each of two levels, which is the case that matters
    // (the coarse level has to be right for the fine one to start well).
    const std::vector<PMView> neighbours = {all_views[0], all_views[1], all_views[2]};

    PatchMatchOptions flat = scene_options();
    flat.iterations = 4;
    flat.pyramid_levels = 1;

    PatchMatchOptions pyramid = scene_options();
    pyramid.iterations = 4;
    pyramid.pyramid_levels = 0;   // automatic
    pyramid.pyramid_min_size = 100;  // 240 -> 120, so two levels

    const std::vector<int> plan =
        alphashape::stereo::patch_match_iteration_plan(320, 240, pyramid);
    ASSERT_EQ(plan.size(), 2U) << "the automatic plan should have chosen a pyramid here";

    const PatchMatchResult flat_result = alphashape::stereo::run_patch_match(
        reference, neighbours, flat);
    const PatchMatchResult pyramid_result = alphashape::stereo::run_patch_match(
        reference, neighbours, pyramid);

    const Accuracy flat_accuracy = measure(scene, flat_result, 5);
    const Accuracy pyramid_accuracy = measure(scene, pyramid_result, 5);

    // Median relative depth difference between the two maps, over the pixels
    // valid in both: the spec's 5e-3 bound.
    std::vector<double> differences;
    std::size_t both = 0;
    for (std::size_t p = 0; p < flat_result.depth.size(); ++p) {
        if (!(flat_result.depth[p] > 0.0F) || !(pyramid_result.depth[p] > 0.0F)) {
            continue;
        }
        ++both;
        differences.push_back(std::abs(static_cast<double>(pyramid_result.depth[p]) -
                                       static_cast<double>(flat_result.depth[p])) /
                              static_cast<double>(flat_result.depth[p]));
    }
    std::sort(differences.begin(), differences.end());
    const double median_difference =
        differences.empty() ? 1.0 : differences[differences.size() / 2];
    std::string schedule;
    for (std::size_t i = 0; i < plan.size(); ++i) {
        schedule += (i == 0 ? "" : "/") + std::to_string(plan[i]);
    }
    std::printf("[pyramid smooth] %zu levels %s | flat: cover %.3f median %.5f p90 %.5f | "
                "pyramid: cover %.3f median %.5f p90 %.5f | median |diff| %.5f of %zu\n",
                plan.size(), schedule.c_str(), flat_accuracy.coverage, flat_accuracy.median,
                flat_accuracy.p90, pyramid_accuracy.coverage, pyramid_accuracy.median,
                pyramid_accuracy.p90, median_difference, both);

    ASSERT_GT(both, 1000U) << "the two runs must have a large common support to compare";
    EXPECT_LE(median_difference, 5e-3) << "the pyramid run drifted from the single-level one";
    // Coverage is the property the pyramid could plausibly lose (a hole in the
    // coarse map must not survive into the result), and accuracy must not be
    // worse: the point of the change is the same surface for less work.
    EXPECT_GT(pyramid_accuracy.coverage, 0.85);
    EXPECT_LE(pyramid_accuracy.median, flat_accuracy.median + 1e-3)
        << "the pyramid lost accuracy against the single-level run";
    EXPECT_LE(pyramid_accuracy.p90, flat_accuracy.p90 + 2e-3);
}

// ---------------------------------------------------------------------------
// A hole in the coarse map is filled at the fine level
// ---------------------------------------------------------------------------

TEST(PatchMatchPyramid, PyramidSeededConsistent) {
    // "A hole in the coarse map is filled at the fine level" is asserted against
    // the coarse map itself, not against a guessed premise: the coarse level's
    // search is run through the same public API (the reference and its
    // neighbours reduced, the calibration halved), so the holes counted here are
    // exactly the cells a pyramid would have inherited with no hypothesis.
    //
    // This is deliberately not a synthetic "flat patch" scenario. A patch that
    // is flat at one level and textured at the other cannot be built from a box
    // reduction: a flat coarse window means every 2x2 block mean inside it is
    // equal, which also makes the fine window's stride-2 samples equal. So the
    // holes are taken where the search actually leaves them -- pixels the
    // coarse level's larger window could not support, or whose cost stayed above
    // max_cost -- and the pyramid must recover them at the reference
    // resolution.
    PlaneScene scene = make_plane_scene(320, 240, 300.0);
    const PMView reference = plane_reference(scene);
    const std::vector<PMView> neighbours = plane_views(scene);

    // -- the coarse level's own search, one level, at half resolution --------
    const FloatImage coarse_reference = alphashape::stereo::pyramid_downsample(scene.reference);
    std::vector<FloatImage> coarse_neighbour_images;
    std::vector<PMView> coarse_neighbours;
    for (const PMView& view : neighbours) {
        coarse_neighbour_images.push_back(alphashape::stereo::pyramid_downsample(*view.image));
    }
    for (std::size_t i = 0; i < neighbours.size(); ++i) {
        PMView view;
        view.image = &coarse_neighbour_images[i];
        view.calibration = alphashape::scaled_calibration(neighbours[i].calibration, 2);
        view.R = neighbours[i].R;
        view.t = neighbours[i].t;
        coarse_neighbours.push_back(std::move(view));
    }
    PMView coarse_view;
    coarse_view.image = &coarse_reference;
    coarse_view.calibration = alphashape::scaled_calibration(reference.calibration, 2);
    coarse_view.R = reference.R;
    coarse_view.t = reference.t;

    // The coarse level's budget as the plan gives it (the reference resolution's
    // level runs its own, larger, share).
    PatchMatchOptions options = scene_options();
    options.pyramid_levels = 2;
    options.pyramid_min_size = 32;
    const std::vector<int> plan =
        alphashape::stereo::patch_match_iteration_plan(320, 240, options);
    ASSERT_EQ(plan.size(), 2U) << "this test needs exactly two levels";

    PatchMatchOptions coarse_options = scene_options();
    coarse_options.pyramid_levels = 1;
    coarse_options.iterations = plan[0];
    // The coarse level of a pyramid uses the configured span; it is the coarsest
    // level, so nothing is inherited.
    const PatchMatchResult coarse_result =
        alphashape::stereo::run_patch_match(coarse_view, coarse_neighbours, coarse_options);
    ASSERT_EQ(coarse_result.width, 160);
    ASSERT_EQ(coarse_result.height, 120);

    const PatchMatchResult pyramid_result =
        alphashape::stereo::run_patch_match(reference, neighbours, options);

    // Every coarse hole covers a 2x2 block of fine pixels. Count those blocks,
    // and how many of them the pyramid filled.
    std::size_t coarse_holes = 0;
    std::size_t fine_filled = 0;
    for (int y = 0; y < coarse_result.height; ++y) {
        for (int x = 0; x < coarse_result.width; ++x) {
            const std::size_t c =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(coarse_result.width) +
                static_cast<std::size_t>(x);
            if (coarse_result.depth[c] > 0.0F) {
                continue;
            }
            ++coarse_holes;
            // A fine pixel is "filled" for the hole if any of the block's four is.
            bool filled = false;
            for (int dy = 0; dy < 2 && !filled; ++dy) {
                for (int dx = 0; dx < 2 && !filled; ++dx) {
                    const int fx = 2 * x + dx;
                    const int fy = 2 * y + dy;
                    if (fx >= pyramid_result.width || fy >= pyramid_result.height) {
                        continue;
                    }
                    const std::size_t p =
                        static_cast<std::size_t>(fy) *
                            static_cast<std::size_t>(pyramid_result.width) +
                        static_cast<std::size_t>(fx);
                    filled = pyramid_result.depth[p] > 0.0F;
                }
            }
            fine_filled += filled ? 1U : 0U;
        }
    }

    // Support must not regress at all: everything the coarse level found, the
    // pyramid still has.
    std::size_t coarse_found = 0;
    std::size_t kept = 0;
    for (int y = 0; y < coarse_result.height; ++y) {
        for (int x = 0; x < coarse_result.width; ++x) {
            const std::size_t c =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(coarse_result.width) +
                static_cast<std::size_t>(x);
            if (!(coarse_result.depth[c] > 0.0F)) {
                continue;
            }
            ++coarse_found;
            const int fx = 2 * x;
            const int fy = 2 * y;
            if (fx >= pyramid_result.width || fy >= pyramid_result.height) {
                continue;
            }
            const std::size_t p =
                static_cast<std::size_t>(fy) * static_cast<std::size_t>(pyramid_result.width) +
                static_cast<std::size_t>(fx);
            kept += pyramid_result.depth[p] > 0.0F ? 1U : 0U;
        }
    }

    std::printf("[pyramid seeded] coarse %dx%d: %zu found, %zu holes | pyramid filled %zu of the "
                "holes (%.1f%%), kept %zu of %zu coarse cells\n",
                coarse_result.width, coarse_result.height, coarse_found, coarse_holes,
                fine_filled,
                coarse_holes > 0 ? 100.0 * static_cast<double>(fine_filled) /
                                       static_cast<double>(coarse_holes)
                                 : 100.0,
                kept, coarse_found);

    ASSERT_GT(coarse_holes, 0U) << "the fixture produced no coarse holes to test the fill with";
    // The fine level must actively fill the holes, not merely inherit them: a
    // pyramid that dropped the coarse map's missing cells would leave them all
    // missing.
    EXPECT_GT(fine_filled * 2, coarse_holes)
        << "the fine level did not fill the cells the coarse level left empty";
    EXPECT_GE(kept, static_cast<std::size_t>(0.80 * static_cast<double>(coarse_found)))
        << "the pyramid lost support the coarse level found";
}

// ---------------------------------------------------------------------------
// Progress: the hook describes the whole pyramid, not one level of it
// ---------------------------------------------------------------------------

TEST(PatchMatchPyramid, SweepProgressReportsMonotoneFractions) {
    // A level's share of the bar is its own half-sweeps over the run's, spent
    // across that level's sweeps; the fraction therefore has to be monotone
    // ACROSS level boundaries and reach the end from the sweeps rather than from
    // a single trailing tick. A hook that scaled by the wrong denominator would
    // still be monotone, just permanently far behind, so the values are pinned.
    // A small scene: the values this test asserts are the schedule, and the
    // long run below sweeps 120 times, which has to stay cheap under ASan.
    PlaneScene scene = make_plane_scene(64, 48, 80.0);
    const PMView reference = plane_reference(scene);
    const std::vector<PMView> all_views = plane_views(scene);
    const std::vector<PMView> neighbours = {all_views[0]};

    const auto progress_of = [&](PatchMatchOptions options) {
        std::vector<double> reported;
        options.on_sweep = [&](double fraction) { reported.push_back(fraction); };
        const PatchMatchResult result =
            alphashape::stereo::run_patch_match(reference, neighbours, options);
        EXPECT_FALSE(result.depth.empty()) << "the hook must not disturb the search";
        return reported;
    };
    const auto expect_shape = [](const std::vector<double>& reported, const char* what) {
        ASSERT_GE(reported.size(), 2U) << what << ": the hook must report more than one tick";
        for (std::size_t i = 0; i < reported.size(); ++i) {
            EXPECT_GE(reported[i], 0.0) << what << " report " << i;
            EXPECT_LE(reported[i], 1.0) << what << " report " << i;
            if (i > 0) {
                EXPECT_GE(reported[i], reported[i - 1])
                    << what << " report " << i << " went backwards (" << reported[i - 1]
                    << " -> " << reported[i] << ")";
            }
        }
        EXPECT_DOUBLE_EQ(reported.back(), 1.0) << what << " did not finish at 1.0";
    };

    PatchMatchOptions options = scene_options();
    options.window_radius = 3;
    options.window_step = 3;

    // A single level with six iterations is twelve half-sweeps, so the last
    // report has 11/12 behind it. Well below that means the sweeps were divided
    // by the wrong count.
    options.pyramid_levels = 1;
    options.iterations = 6;
    const std::vector<double> single = progress_of(options);
    expect_shape(single, "single level");
    ASSERT_EQ(single.size(), 12U) << "one report per half-sweep of the only level";
    EXPECT_GE(single[single.size() - 2], 0.8)
        << "last report before 1.0 was " << single[single.size() - 2];

    // Enough sweeps that the second-to-last report is within 1% of the end.
    options.iterations = 60;
    const std::vector<double> long_run = progress_of(options);
    expect_shape(long_run, "long run");
    ASSERT_EQ(long_run.size(), 120U);
    EXPECT_GE(long_run[long_run.size() - 2], 0.99)
        << "last report before 1.0 was " << long_run[long_run.size() - 2];

    // A pyramid reports across its levels: still monotone, still ending at 1.0,
    // and the coarse level spends its own share rather than jumping straight to
    // its base being visible only at the boundary.
    PatchMatchOptions pyramid = options;
    pyramid.iterations = 6;
    pyramid.pyramid_levels = 2;
    const std::vector<int> plan =
        alphashape::stereo::patch_match_iteration_plan(scene.reference.width(),
                                                       scene.reference.height(), pyramid);
    ASSERT_EQ(plan.size(), 2U) << "the two-level request should survive here";
    const std::vector<double> multi = progress_of(pyramid);
    expect_shape(multi, "two levels");
    EXPECT_EQ(multi.size(), 12U) << "one report per half-sweep over the whole pyramid";
    const double coarse_share =
        static_cast<double>(2 * plan[0]) / static_cast<double>(2 * (plan[0] + plan[1]));
    // The coarse level must have reported somewhere strictly inside its share --
    // not only the 0 it starts at and not only its boundary value.
    bool coarse_interior = false;
    for (const double value : multi) {
        if (value > 0.0 && value < coarse_share - 1e-12) {
            coarse_interior = true;
        }
    }
    EXPECT_TRUE(coarse_interior) << "the coarse level reported no progress of its own";
}

TEST(PatchMatchPyramid, SingleLevelSearchNeedsNoMinSize) {
    // The end-to-end half of SingleLevelDoesNotNeedAMinSize: the search itself
    // must accept a single-level run with no floor and produce the flat result,
    // rather than reaching a throw from the plan it asked for.
    PlaneScene scene = make_plane_scene(96, 72, 100.0);
    const PMView reference = plane_reference(scene);
    const std::vector<PMView> all_views = plane_views(scene);
    const std::vector<PMView> neighbours = {all_views[0]};

    PatchMatchOptions options = scene_options();
    options.window_radius = 3;
    options.window_step = 3;
    options.iterations = 2;
    options.pyramid_levels = 1;
    options.pyramid_min_size = 0;
    EXPECT_NO_THROW({
        const PatchMatchResult result =
            alphashape::stereo::run_patch_match(reference, neighbours, options);
        EXPECT_EQ(result.width, 96);
        EXPECT_EQ(result.height, 72);
    });

    // And it really is the flat search: the same options with a defined floor
    // resolve to the same single level and produce the same map.
    PatchMatchOptions with_floor = options;
    with_floor.pyramid_min_size = 256;
    const PatchMatchResult no_floor =
        alphashape::stereo::run_patch_match(reference, neighbours, options);
    const PatchMatchResult floor_run =
        alphashape::stereo::run_patch_match(reference, neighbours, with_floor);
    EXPECT_EQ(no_floor.depth, floor_run.depth);
    EXPECT_EQ(no_floor.normals, floor_run.normals);
    EXPECT_EQ(no_floor.cost, floor_run.cost);
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

TEST(PatchMatchPyramid, PyramidDeterministic) {
    PlaneScene scene = make_plane_scene(200, 160, 200.0);
    const PMView reference = plane_reference(scene);
    const std::vector<PMView> neighbours = plane_views(scene);

    PatchMatchOptions options = scene_options();
    options.iterations = 4;
    options.pyramid_levels = 0;
    options.pyramid_min_size = 40;

    const PatchMatchResult first =
        alphashape::stereo::run_patch_match(reference, neighbours, options);
    const PatchMatchResult second =
        alphashape::stereo::run_patch_match(reference, neighbours, options);
    EXPECT_EQ(first.depth, second.depth);
    EXPECT_EQ(first.normals, second.normals);
    EXPECT_EQ(first.cost, second.cost);

    // And a different seed must actually change the search on the coarse levels:
    // if the level keys were ignored, a pyramid would be seed-independent from
    // the coarse level down and a second seed would reproduce the first map.
    PatchMatchOptions other = options;
    other.seed = options.seed + 12345;
    const PatchMatchResult third =
        alphashape::stereo::run_patch_match(reference, neighbours, other);
    EXPECT_NE(third.depth, first.depth) << "the pyramid ignored the seed";
}

// ---------------------------------------------------------------------------
// Timing: the pyramid is faster on a large map
// ---------------------------------------------------------------------------

TEST(PatchMatchPyramid, PyramidFasterOnLarge) {
#if defined(ALPHASHAPE_SANITIZED_BUILD)
    GTEST_SKIP() << "a wall-clock ratio under ASan/UBSan measures the instrumentation";
#endif
    // A 1 MP plane -- the smallest size at which the reduction's saving is a
    // real measurement rather than scheduler noise, and small enough to run four
    // searches inside the heavy lane's budget. Two neighbours and a coarse
    // window keep the flat run to well under a minute; the pyramid's saving is
    // the share of the pixels it moves to the coarse levels, which does not
    // depend on either.
    PlaneScene scene = make_plane_scene(1024, 1024, 1000.0);
    const PMView reference = plane_reference(scene);
    const std::vector<PMView> all_views = plane_views(scene);
    const std::vector<PMView> neighbours = {all_views[0], all_views[1]};

    PatchMatchOptions flat = scene_options();
    flat.window_radius = 3;
    flat.window_step = 3;
    // Six sweeps, so the three-level request survives the plan's two-sweeps-per-
    // level floor and the schedule is [2, 2, 2].
    flat.iterations = 6;
    flat.pyramid_levels = 1;
    flat.seed = 99;

    PatchMatchOptions pyramid = flat;
    pyramid.pyramid_levels = 0;
    pyramid.pyramid_min_size = 256;  // 1024 -> 512 -> 256, so 3 levels

    const std::vector<int> plan =
        alphashape::stereo::patch_match_iteration_plan(1024, 1024, pyramid);
    ASSERT_EQ(plan.size(), 3U);

    // One warm-up each, so neither side pays a first-call page-in.
    alphashape::stereo::run_patch_match(reference, neighbours, flat);
    alphashape::stereo::run_patch_match(reference, neighbours, pyramid);

    const auto time_of = [&](const PatchMatchOptions& options) {
        const auto start = std::chrono::steady_clock::now();
        const PatchMatchResult result =
            alphashape::stereo::run_patch_match(reference, neighbours, options);
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        return std::make_pair(result, seconds);
    };
    const std::pair<PatchMatchResult, double> flat_run = time_of(flat);
    const std::pair<PatchMatchResult, double> pyramid_run = time_of(pyramid);

    const Accuracy flat_accuracy = measure(scene, flat_run.first, 3);
    const Accuracy pyramid_accuracy = measure(scene, pyramid_run.first, 3);
    const double speedup = pyramid_run.second > 0.0 ? flat_run.second / pyramid_run.second : 0.0;

    std::printf("[pyramid timing] %d levels %d/%d/%d | flat %.2f s (cover %.3f median %.5f) | "
                "pyramid %.2f s (cover %.3f median %.5f) | speed-up %.2fx\n",
                static_cast<int>(plan.size()), plan[0], plan[1], plan[2], flat_run.second,
                flat_accuracy.coverage, flat_accuracy.median, pyramid_run.second,
                pyramid_accuracy.coverage, pyramid_accuracy.median, speedup);
    RecordProperty("flat_seconds", flat_run.second);
    RecordProperty("pyramid_seconds", pyramid_run.second);
    RecordProperty("speedup", speedup);

    // Equal accuracy first: a speed-up bought by losing the surface is not the
    // trade this test is for.
    EXPECT_GT(pyramid_accuracy.coverage, 0.85);
    EXPECT_LT(pyramid_accuracy.median, 0.005);
    EXPECT_LT(pyramid_accuracy.median, flat_accuracy.median * 2.0 + 1e-3);
    if (alphashape::test::perf_asserts_enabled()) {
        EXPECT_GT(speedup, 1.6) << "the pyramid was not 1.6x faster at equal accuracy";
    }
}

}  // namespace
