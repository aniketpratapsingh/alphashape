// Census transform and cost tests (WP-17.13, DP-03).
//
// The transform is checked against a hand-computed descriptor, and the search's
// census cost (the ZNCC of the descriptor values in [0, 1]) is checked for its
// defining property: invariance to an affine intensity change, on descriptors
// and end to end through the search. The noise advantage over the intensity cost
// is PatchMatch.CensusImprovesPlaneNoise in test_patch_match.cpp, which owns the
// synthetic scene helpers.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "common/algorithm/stereo/census.hpp"
#include "common/algorithm/stereo/patch_match.hpp"
#include "image/image.hpp"
#include "model/camera/calibration.hpp"

namespace {

using alphashape::Calibration;
using alphashape::FloatImage;
using alphashape::stereo::CensusOptions;
using alphashape::stereo::CensusImage;
using alphashape::stereo::PMView;
using alphashape::stereo::PatchMatchOptions;
using alphashape::stereo::PatchMatchResult;

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

TEST(Census, TransformKnownPattern) {
    // A 9x9 patch around a centre of 0.5. Every supported sample is brighter
    // (0.6) except four, one per arm at a known bit position:
    //   (0, 4) -> left arm k = 0  -> bit 0
    //   (6, 4) -> right arm k = 1 -> bit 5
    //   (4, 0) -> top arm k = 0   -> bit 8
    //   (4, 8) -> bottom arm k = 3 -> bit 15
    FloatImage image(9, 9, 0.6F);
    image.at(4, 4) = 0.5F;
    image.at(0, 4) = 0.25F;
    image.at(6, 4) = 0.3F;
    image.at(4, 0) = 0.1F;
    image.at(4, 8) = 0.45F;

    CensusOptions options;  // wide, radius 4
    const CensusImage census = alphashape::stereo::census_transform(image, options);
    ASSERT_EQ(census.width, 9);
    ASSERT_EQ(census.height, 9);

    const std::uint16_t expected = static_cast<std::uint16_t>((1U << 0) | (1U << 5) |
                                                             (1U << 8) | (1U << 15));
    EXPECT_EQ(census.at(4, 4), expected);
}

TEST(Census, FlatPatchIsZero) {
    // Every comparison against a constant is a tie, and a tie sets no bit. A
    // flat region is therefore all zeros, not all ones: a saturated white patch
    // cannot masquerade as strongly textured.
    FloatImage image(9, 9, 0.7F);
    const CensusImage census = alphashape::stereo::census_transform(image, CensusOptions{});
    for (const std::uint16_t value : census.values) {
        EXPECT_EQ(value, 0U);
    }
}

TEST(Census, RadianceOrderIsWhatMatters) {
    // A single bright centre against darker neighbours sets every bit; the
    // magnitudes are irrelevant. This is the property the cost relies on.
    FloatImage image(9, 9, 0.2F);
    image.at(4, 4) = 0.8F;
    const CensusImage census = alphashape::stereo::census_transform(image, CensusOptions{});
    EXPECT_EQ(census.at(4, 4), 0xFFFFU);
}

TEST(Census, NarrowCensusFillsLowByteOnly) {
    FloatImage image(3, 3, 0.5F);
    image.at(0, 0) = 0.1F;   // neighbour (-1, -1) -> bit 0
    image.at(2, 2) = 0.9F;   // brighter: sets nothing
    image.at(1, 0) = 0.2F;   // neighbour (0, -1)  -> bit 3

    CensusOptions options;
    options.wide = false;
    options.window_radius = 1;
    const CensusImage census = alphashape::stereo::census_transform(image, options);
    EXPECT_EQ(census.at(1, 1),
              static_cast<std::uint16_t>((1U << 0) | (1U << 3)));
    for (const std::uint16_t value : census.values) {
        EXPECT_EQ(value & 0xFF00U, 0U) << "narrow census must leave the high byte zero";
    }
}

TEST(Census, InvalidOptionsThrow) {
    FloatImage image(9, 9, 0.5F);
    CensusOptions wide_too_big;
    wide_too_big.window_radius = 9;  // 4 arms x 9 = 36 bits
    EXPECT_THROW(alphashape::stereo::census_transform(image, wide_too_big),
                 alphashape::ValueError);

    CensusOptions narrow_wrong_radius;
    narrow_wrong_radius.wide = false;
    narrow_wrong_radius.window_radius = 2;
    EXPECT_THROW(alphashape::stereo::census_transform(image, narrow_wrong_radius),
                 alphashape::ValueError);

    CensusOptions zero;
    zero.window_radius = 0;
    EXPECT_THROW(alphashape::stereo::census_transform(image, zero), alphashape::ValueError);
}

TEST(Census, DistanceIsNormalisedHamming) {
    EXPECT_DOUBLE_EQ(alphashape::stereo::census_distance(0x0000U, 0x0000U), 0.0);
    EXPECT_DOUBLE_EQ(alphashape::stereo::census_distance(0x0000U, 0xFFFFU), 1.0);
    EXPECT_DOUBLE_EQ(alphashape::stereo::census_distance(0x00FFU, 0x0000U), 8.0 / 16.0);
    // Symmetric.
    EXPECT_DOUBLE_EQ(alphashape::stereo::census_distance(0x1234U, 0x4321U),
                     alphashape::stereo::census_distance(0x4321U, 0x1234U));
}

TEST(Census, CacheTransformsEachViewOnce) {
    FloatImage a(16, 16, 0.2F);
    FloatImage b(16, 16, 0.8F);
    a.at(3, 3) = 0.9F;
    alphashape::stereo::CensusCache cache(CensusOptions{});
    const CensusImage& ca = cache.get(a);
    const CensusImage& cb = cache.get(b);
    EXPECT_EQ(cache.size(), 2U);
    // A second request returns the same entry, not a new transform.
    EXPECT_EQ(&cache.get(a), &ca);
    EXPECT_EQ(&cache.get(b), &cb);
    EXPECT_EQ(cache.size(), 2U);
    EXPECT_NE(ca.values, cb.values);
}

// ---------------------------------------------------------------------------
// Invariance to an affine intensity change
// ---------------------------------------------------------------------------

/// A deterministic patch of values on a coarse grid, so every pair of pixels in
/// any 9x9 window is separated by much more than a float rounding step. The
/// affine map below then cannot turn a strict comparison into a tie by rounding.
FloatImage grid_image(int width, int height, float gain, float offset) {
    FloatImage image(width, height, 0.0F);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int level = (x * 7 + y * 13) % 17;  // 17 distinct levels
            const float base = 0.15F + 0.04F * static_cast<float>(level);  // [0.15, 0.79]
            image.at(x, y) = gain * base + offset;
        }
    }
    return image;
}

TEST(Census, InvariantToAffineIntensity) {
    // The census compares ranks, so a positive gain and a bias must leave every
    // descriptor unchanged; the cost is therefore unchanged too.
    const FloatImage original = grid_image(40, 32, 1.0F, 0.0F);
    const FloatImage affine = grid_image(40, 32, 1.3F, 0.05F);

    const CensusImage a = alphashape::stereo::census_transform(original, CensusOptions{});
    const CensusImage b = alphashape::stereo::census_transform(affine, CensusOptions{});

    double total = 0.0;
    std::size_t count = 0;
    for (int y = 0; y < a.height; ++y) {
        for (int x = 0; x < a.width; ++x) {
            total += alphashape::stereo::census_distance(a.at(x, y), b.at(x, y));
            ++count;
        }
    }
    ASSERT_GT(count, 0U);
    EXPECT_LE(total / static_cast<double>(count), 1e-3);
}

TEST(Census, AffineIntensityDoesNotChangePlaneCost) {
    // The same invariance, measured through the search's own cost. Two views of
    // one flat scene, the neighbour's gain and bias changed: a census run must
    // recover the same plane at the same cost.
    const int width = 96;
    const int height = 72;
    const double focal = 100.0;
    const Calibration calib = Calibration::initial(alphashape::CameraType::Frame, width, height,
                                                   focal);
    const Eigen::Vector3d plane_n(0.0, 0.0, -1.0);
    const Eigen::Vector3d plane_p(0.0, 0.0, 8.0);
    const Eigen::Matrix3d r = Eigen::Matrix3d::Identity();

    // Reference at the origin, one neighbour 0.6 to the right, both looking
    // along +Z at the plane z = 8.
    const Eigen::Vector3d ref_c(0.0, 0.0, 0.0);
    const Eigen::Vector3d nb_c(0.6, 0.0, 0.0);
    const Eigen::Matrix3d nb_r = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d nb_t = -nb_r * nb_c;

    const auto render = [&](const Eigen::Vector3d& camera_centre, const Eigen::Matrix3d& rot,
                            float gain, float offset) {
        FloatImage image(width, height, 0.0F);
        const Eigen::Matrix3d rt = rot.transpose();
        // Camera centre in world space: t = -R C, so C = -R^T t = the centre the
        // caller passed.
        const Eigen::Vector3d c = camera_centre;
        const double dw = plane_n.dot(plane_p);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const Eigen::Vector2d pixel(static_cast<double>(x) + 0.5,
                                            static_cast<double>(y) + 0.5);
                const Eigen::Vector3d dir = (rt * calib.unproject(pixel)).normalized();
                const double denom = plane_n.dot(dir);
                const double depth = (dw - plane_n.dot(c)) / denom;
                const Eigen::Vector3d world = c + depth * dir;
                // Ranks only: a smooth ramp plus a checker, all well separated.
                const double u = 0.15 + 0.3 * (0.5 + 0.5 * std::sin(world.x() * 2.0)) +
                                 0.2 * ((static_cast<int>(std::floor(world.x() * 3.0)) +
                                         static_cast<int>(std::floor(world.y() * 3.0))) &
                                        1);
                image.at(x, y) = gain * static_cast<float>(u) + offset;
            }
        }
        return image;
    };

    FloatImage reference = render(ref_c, r, 1.0F, 0.0F);
    // The same neighbour twice: once as rendered, once with a gain and a bias.
    // The census compares ranks, so the two must search to the same plane at the
    // same cost -- which is the invariance claim, measured end to end through
    // the search rather than on descriptors alone.
    const FloatImage neighbour = render(nb_c, nb_r, 1.0F, 0.0F);
    const FloatImage bright = render(nb_c, nb_r, 1.3F, 0.05F);

    PMView ref_view;
    ref_view.image = &reference;
    ref_view.calibration = calib;
    ref_view.R = r;
    ref_view.t = -r * ref_c;

    const auto run_with = [&](const FloatImage& neighbour_image) {
        PMView nb_view;
        nb_view.image = &neighbour_image;
        nb_view.calibration = calib;
        nb_view.R = nb_r;
        nb_view.t = nb_t;
        PatchMatchOptions options;
        options.depth_min = 5.0;
        options.depth_max = 12.0;
        options.window_radius = 5;
        options.window_step = 2;
        options.iterations = 3;
        options.max_cost = 2.0;  // never cut: compare whole fields
        options.pyramid_levels = 1;
        options.cost_type = alphashape::stereo::CostType::Census;
        options.seed = 7;
        return alphashape::stereo::run_patch_match(ref_view, {nb_view}, options);
    };
    const PatchMatchResult plain = run_with(neighbour);
    const PatchMatchResult affine = run_with(bright);

    std::size_t valid = 0;
    std::size_t same_depth = 0;
    double cost_delta = 0.0;
    for (std::size_t i = 0; i < plain.depth.size(); ++i) {
        EXPECT_GE(plain.cost[i], 0.0F);
        EXPECT_LE(plain.cost[i], 1.0F) << "census cost must live in [0, 1]";
        if (plain.depth[i] > 0.0F && affine.depth[i] > 0.0F) {
            ++valid;
            const double rel = std::abs(static_cast<double>(plain.depth[i]) -
                                        static_cast<double>(affine.depth[i])) /
                               static_cast<double>(plain.depth[i]);
            if (rel < 1e-3) {
                ++same_depth;
            }
            cost_delta += std::abs(static_cast<double>(plain.cost[i]) -
                                   static_cast<double>(affine.cost[i]));
        }
    }
    ASSERT_GT(valid, plain.depth.size() / 2);
    // An affine intensity change is invisible to the descriptor, so the search
    // must land on the same plane at the same cost almost everywhere.
    EXPECT_GT(static_cast<double>(same_depth) / static_cast<double>(valid), 0.99);
    EXPECT_LT(cost_delta / static_cast<double>(valid), 1.0 / 65536.0);
}

// ---------------------------------------------------------------------------
// Cost distribution
// ---------------------------------------------------------------------------

/// A reproducible pseudo-random field in [0, 1), so two unrelated images give
/// unrelated descriptors.
float hash_field(int x, int y, int salt) {
    std::uint32_t h = static_cast<std::uint32_t>(x) * 374761393U +
                      static_cast<std::uint32_t>(y) * 668265263U +
                      static_cast<std::uint32_t>(salt) * 2246822519U;
    h = (h ^ (h >> 13)) * 1274126177U;
    h ^= h >> 16;
    return static_cast<float>(h & 0xFFFFFFU) / static_cast<float>(0x1000000U);
}

TEST(Census, ValuesAreNormalisedToUnitRange) {
    // The search correlates mask / 65535, so every transformed value lies in
    // [0, 1]; this is the [0, 1] scale the max_cost cut depends on.
    FloatImage image(24, 20, 0.0F);
    for (int y = 0; y < 20; ++y) {
        for (int x = 0; x < 24; ++x) {
            image.at(x, y) = hash_field(x, y, 3);
        }
    }
    const CensusImage census = alphashape::stereo::census_transform(image, CensusOptions{});
    const std::vector<float> values = alphashape::stereo::census_values(census);
    ASSERT_EQ(values.size(), census.values.size());
    for (const float v : values) {
        EXPECT_GE(v, 0.0F);
        EXPECT_LE(v, 1.0F);
    }
}

TEST(Census, DistanceAtChanceForUnrelatedPatterns) {
    // Unrelated descriptors differ on about half their bits: the Hamming distance
    // of two independent patterns sits at chance, not at agreement.
    FloatImage a(40, 32, 0.0F);
    FloatImage b(40, 32, 0.0F);
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 40; ++x) {
            a.at(x, y) = hash_field(x, y, 1);
            b.at(x, y) = hash_field(x, y, 2);
        }
    }
    const CensusImage ca = alphashape::stereo::census_transform(a, CensusOptions{});
    const CensusImage cb = alphashape::stereo::census_transform(b, CensusOptions{});
    double total = 0.0;
    std::size_t count = 0;
    for (int y = 4; y < 28; ++y) {
        for (int x = 4; x < 36; ++x) {
            const double d = alphashape::stereo::census_distance(ca.at(x, y), cb.at(x, y));
            EXPECT_GE(d, 0.0);
            EXPECT_LE(d, 1.0);
            total += d;
            ++count;
        }
    }
    ASSERT_GT(count, 0U);
    const double mean = total / static_cast<double>(count);
    EXPECT_GT(mean, 0.4) << mean;
    EXPECT_LT(mean, 0.6) << mean;
}

}  // namespace
