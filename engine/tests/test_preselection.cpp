// Tests for camera-pair preselection: exhaustive, sequential, GPS reference,
// and the generic low-resolution prematch.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/algorithm/match/preselection.hpp"

namespace alphashape::match {
namespace {

using Pair = CameraPair;

bool contains(const std::vector<Pair>& pairs, std::int32_t a, std::int32_t b) {
    const Pair want = a < b ? Pair{a, b} : Pair{b, a};
    return std::find(pairs.begin(), pairs.end(), want) != pairs.end();
}

/// Every pair has a<b, the list is strictly increasing (hence sorted and free of
/// duplicates).
bool sorted_unique(const std::vector<Pair>& pairs) {
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        if (pairs[i].a >= pairs[i].b) return false;
        if (i > 0) {
            const bool greater = pairs[i].a > pairs[i - 1].a ||
                                 (pairs[i].a == pairs[i - 1].a && pairs[i].b > pairs[i - 1].b);
            if (!greater) return false;
        }
    }
    return true;
}

// --------------------------------------------------------------------------

TEST(PreselectionExhaustive, CountAndContent) {
    const std::vector<std::int32_t> ids{4, 1, 9, 2};
    const std::vector<Pair> pairs = exhaustive_pairs(ids);
    EXPECT_EQ(pairs.size(), 6U);  // 4*3/2
    EXPECT_TRUE(sorted_unique(pairs));
    EXPECT_TRUE(contains(pairs, 1, 2));
    EXPECT_TRUE(contains(pairs, 1, 4));
    EXPECT_TRUE(contains(pairs, 1, 9));
    EXPECT_TRUE(contains(pairs, 2, 4));
    EXPECT_TRUE(contains(pairs, 2, 9));
    EXPECT_TRUE(contains(pairs, 4, 9));

    EXPECT_TRUE(exhaustive_pairs({}).empty());
    EXPECT_TRUE(exhaustive_pairs({7}).empty());
}

TEST(PreselectionSequential, Window) {
    const std::vector<std::int32_t> ids{10, 20, 30, 40, 50};
    const std::vector<Pair> pairs = sequential_pairs(ids, 2);
    EXPECT_EQ(pairs.size(), 7U);  // (0,1)(0,2)(1,2)(1,3)(2,3)(2,4)(3,4)
    EXPECT_TRUE(sorted_unique(pairs));
    EXPECT_TRUE(contains(pairs, 10, 20));
    EXPECT_TRUE(contains(pairs, 10, 30));
    EXPECT_TRUE(contains(pairs, 20, 30));
    EXPECT_TRUE(contains(pairs, 20, 40));
    EXPECT_TRUE(contains(pairs, 30, 40));
    EXPECT_TRUE(contains(pairs, 30, 50));
    EXPECT_TRUE(contains(pairs, 40, 50));
    EXPECT_FALSE(contains(pairs, 10, 40));

    // A window >= n collapses to the exhaustive set.
    EXPECT_EQ(sequential_pairs(ids, 99).size(), 10U);
}

TEST(PreselectionReference, GridNeighbours) {
    // A 3x3 grid at ~latitude 45, spacing ~10 m. Longitude degrees shrink with
    // cos(lat); at 45 degrees, 10 m is about 1.28e-4 deg in longitude.
    const double lat0 = 45.0;
    const double dlat = 10.0 / 111320.0;                       // ~9e-5 deg
    const double dlon = 10.0 / (111320.0 * std::cos(lat0 * 3.14159265358979323846 / 180.0));

    std::vector<std::pair<std::int32_t, Eigen::Vector3d>> locations;
    std::int32_t id = 0;
    // Row-major grid indices 0..8; store the grid coordinates so we can state
    // expected neighbours explicitly.
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const double lon = col * dlon;
            const double lat = lat0 + row * dlat;
            locations.emplace_back(id, Eigen::Vector3d(lon, lat, 100.0));
            ++id;
        }
    }

    const std::vector<Pair> pairs = reference_pairs(locations, {}, 2);
    EXPECT_TRUE(sorted_unique(pairs));

    // Grid id = row*3 + col. The centre (4) has four orthogonal neighbours at
    // ~10 m: 1 and 7 vertically, 3 and 5 horizontally. The equirectangular
    // projection scales longitude by cos(lat0) using the centroid latitude, which
    // is slightly north of the centre row, so the two horizontal neighbours are
    // marginally nearer (~16 micrometres); with neighbors=2 they are the picks.
    EXPECT_TRUE(contains(pairs, 4, 3));
    EXPECT_TRUE(contains(pairs, 4, 5));
    // The far corner (8) is never any node's nearest.
    EXPECT_FALSE(contains(pairs, 0, 8));
    EXPECT_FALSE(contains(pairs, 4, 8));
    // Corner (0)'s two nearest are its orthogonal neighbours 1 and 3.
    EXPECT_TRUE(contains(pairs, 0, 1));
    EXPECT_TRUE(contains(pairs, 0, 3));
}

TEST(PreselectionReference, NaNAltitudeUsesHorizontal) {
    // Cameras 2 and 3 have unknown (NaN) altitude; camera 1 has a large known
    // altitude. Under the horizontal-only fallback the pair (1,3) is produced
    // (camera 3's nearest is camera 1, tied with 2 and broken by index).
    // If altitude were instead treated as 0, camera 3's nearest would be 2 and
    // (2,3), not (1,3), would appear. So this pins the documented rule.
    std::vector<std::pair<std::int32_t, Eigen::Vector3d>> locations;
    locations.emplace_back(1, Eigen::Vector3d(0.0, 45.0, 1000.0));
    locations.emplace_back(2, Eigen::Vector3d(0.0, 45.0, std::numeric_limits<double>::quiet_NaN()));
    locations.emplace_back(3, Eigen::Vector3d(0.001, 45.0, std::numeric_limits<double>::quiet_NaN()));

    const std::vector<Pair> pairs = reference_pairs(locations, {}, 1);
    EXPECT_TRUE(contains(pairs, 1, 2));   // 0 m horizontally between 1 and 2
    EXPECT_TRUE(contains(pairs, 1, 3));   // 3's nearest is 1 (78 m, index tie)
    EXPECT_FALSE(contains(pairs, 2, 3));
}

TEST(PreselectionReference, UnreferencedPairsWithEveryone) {
    std::vector<std::pair<std::int32_t, Eigen::Vector3d>> locations;
    locations.emplace_back(1, Eigen::Vector3d(0.0, 45.0, 10.0));
    locations.emplace_back(2, Eigen::Vector3d(0.001, 45.0, 10.0));
    locations.emplace_back(3, Eigen::Vector3d(5.0, 45.0, 10.0));

    const std::vector<std::int32_t> unreferenced{9};
    const std::vector<Pair> pairs = reference_pairs(locations, unreferenced, 1);
    // The unreferenced camera links to all three referenced ones.
    EXPECT_TRUE(contains(pairs, 9, 1));
    EXPECT_TRUE(contains(pairs, 9, 2));
    EXPECT_TRUE(contains(pairs, 9, 3));
    EXPECT_TRUE(sorted_unique(pairs));
}

TEST(PreselectionInvalid, Throws) {
    EXPECT_THROW(sequential_pairs({1, 2}, 0), ValueError);
    EXPECT_THROW(sequential_pairs({1, 2}, -3), ValueError);
    EXPECT_THROW(reference_pairs({}, {}, 0), ValueError);
    EXPECT_THROW(exhaustive_pairs({1, 1}), ValueError);
    EXPECT_THROW(sequential_pairs({1, 1}, 1), ValueError);
}

TEST(PreselectionGeneric, KeepsSharedDropsDisjoint) {
    // Three images with 60 keypoints each. {0,1} share many descriptors (1 gets
    // 0's descriptors with light bit flips); {0,2} share none (2 is random).
    std::mt19937 rng(123U);
    constexpr int kCount = 60;
    auto random_set = [&rng](std::int32_t camera_id) {
        KeypointSet set;
        set.camera_id = camera_id;
        for (int i = 0; i < kCount; ++i) {
            Keypoint keypoint;
            keypoint.scale = 1.0F + static_cast<float>(i % 5);
            set.keypoints.push_back(keypoint);
            Descriptor descriptor;
            for (std::uint8_t& byte : descriptor.bytes) {
                byte = static_cast<std::uint8_t>(rng() & 0xFFU);
            }
            set.descriptors.push_back(descriptor);
        }
        return set;
    };

    std::vector<KeypointSet> sets;
    sets.push_back(random_set(0));
    KeypointSet one = random_set(1);
    KeypointSet two = random_set(2);
    // Make 1 share 0's descriptors with a few flips.
    for (int i = 0; i < kCount; ++i) {
        one.descriptors[static_cast<std::size_t>(i)] = sets[0].descriptors[static_cast<std::size_t>(i)];
        one.descriptors[static_cast<std::size_t>(i)].bytes[0] ^= 0x01U;
    }
    sets.push_back(std::move(one));
    sets.push_back(std::move(two));

    const std::vector<Pair> candidates{exhaustive_pairs({0, 1, 2})};
    const GenericOptions options;
    const std::vector<Pair> pairs = generic_pairs(sets, candidates, options);

    EXPECT_TRUE(contains(pairs, 0, 1));    // shared descriptors -> kept
    EXPECT_FALSE(contains(pairs, 0, 2));   // disjoint -> dropped by min_score
    EXPECT_FALSE(contains(pairs, 1, 2));
    EXPECT_TRUE(sorted_unique(pairs));
}

TEST(PreselectionGeneric, MissingKeypointSetContributesNothing) {
    std::mt19937 rng(321U);
    std::vector<KeypointSet> sets;
    KeypointSet one;
    one.camera_id = 1;
    for (int i = 0; i < 40; ++i) {
        one.keypoints.push_back(Keypoint{});
        Descriptor descriptor;
        for (std::uint8_t& byte : descriptor.bytes) byte = static_cast<std::uint8_t>(rng() & 0xFFU);
        one.descriptors.push_back(descriptor);
    }
    sets.push_back(one);

    // Camera 5 has no keypoint set.
    const std::vector<Pair> candidates{{1, 5}};
    const std::vector<Pair> pairs = generic_pairs(sets, candidates, GenericOptions{});
    EXPECT_TRUE(pairs.empty());
}

}  // namespace
}  // namespace alphashape::match
