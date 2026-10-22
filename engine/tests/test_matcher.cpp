// Tests for the Hamming brute-force matcher: synthetic descriptors from a
// deterministic RNG, the ratio test, cross-check, the absolute cap, known-byte
// distance, and thread-count determinism.

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "base/utils/portable_rng.hpp"
#include "common/algorithm/match/hamming_matcher.hpp"

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

namespace alphashape::match {
namespace {

Descriptor make_descriptor(std::mt19937& rng) {
    Descriptor descriptor;
    for (std::uint8_t& byte : descriptor.bytes) {
        byte = static_cast<std::uint8_t>(rng() & 0xFFU);
    }
    return descriptor;
}

/// Flip `bits` distinct bits in a descriptor, leaving it close to the original.
Descriptor flip_bits(const Descriptor& source, int bits, std::mt19937& rng) {
    Descriptor result = source;
    std::set<int> flipped;
    while (static_cast<int>(flipped.size()) < bits) {
        const int bit = static_cast<int>(rng() % (Descriptor::kBytes * 8));
        if (!flipped.insert(bit).second) continue;
        const std::size_t byte = static_cast<std::size_t>(bit) / 8;
        const int offset = bit % 8;
        result.bytes[byte] = static_cast<std::uint8_t>(result.bytes[byte] ^ (1U << offset));
    }
    return result;
}

// --------------------------------------------------------------------------

TEST(MatcherHashing, KnownBytes) {
    Descriptor zero;  // all zero
    Descriptor ones;
    ones.bytes.fill(0xFFU);

    EXPECT_EQ(hamming(zero, zero), 0);
    EXPECT_EQ(hamming(zero, ones), static_cast<int>(Descriptor::kBytes * 8));

    Descriptor a;
    Descriptor b;
    b.bytes[0] = 0b10100001U;  // three differing bits
    EXPECT_EQ(hamming(a, b), 3);

    // A differing bit in the last (63rd) byte must be counted.
    Descriptor c;
    c.bytes[Descriptor::kBytes - 1] = 0b00000001U;
    EXPECT_EQ(hamming(a, c), 1);
}

TEST(MatcherRecall, PermutationWithFlips) {
    std::mt19937 rng(20261017U);
    constexpr int kCount = 500;
    constexpr int kDistractors = 200;

    std::vector<Descriptor> a;
    a.reserve(kCount);
    for (int i = 0; i < kCount; ++i) a.push_back(make_descriptor(rng));

    // b is a random permutation of a with <= 30 random bit flips per descriptor.
    std::vector<int> order(kCount);
    for (int i = 0; i < kCount; ++i) order[static_cast<std::size_t>(i)] = i;
    // Fisher-Yates through the portable index draw: std::shuffle's algorithm is
    // implementation-defined, so libc++ and libstdc++ permuted differently from
    // one seed and the recall figures came from different scenes.
    for (std::size_t i = order.size(); i > 1; --i) {
        const std::size_t j = alphashape::rng::index_below(rng, i);
        std::swap(order[i - 1], order[j]);
    }

    std::vector<Descriptor> b;
    b.reserve(kCount + kDistractors);
    std::vector<std::uint32_t> expected_partner(kCount, 0);
    for (int slot = 0; slot < kCount; ++slot) {
        const int source = order[static_cast<std::size_t>(slot)];
        const int flips = static_cast<int>(rng() % 31);  // 0..30
        b.push_back(flip_bits(a[static_cast<std::size_t>(source)], flips, rng));
        expected_partner[static_cast<std::size_t>(source)] = static_cast<std::uint32_t>(slot);
    }
    for (int i = 0; i < kDistractors; ++i) b.push_back(make_descriptor(rng));

    const std::vector<KeypointMatch> matches =
        match_descriptors(a, b, MatchOptions{.ratio = 0.8F, .cross_check = false, .max_distance = 0});

    int correct = 0;
    for (const KeypointMatch& match : matches) {
        ASSERT_LT(match.index_a, static_cast<std::uint32_t>(kCount));
        EXPECT_EQ(match.index_b, expected_partner[match.index_a])
            << "wrong match for query " << match.index_a;
        if (match.index_b == expected_partner[match.index_a]) ++correct;
    }
    const double recall = static_cast<double>(correct) / static_cast<double>(kCount);
    EXPECT_GT(recall, 0.95);

    // Cross-check must not remove true partners: for these well-separated
    // descriptors each true partner is the unique nearest both ways.
    const std::vector<KeypointMatch> checked =
        match_descriptors(a, b, MatchOptions{.ratio = 0.8F, .cross_check = true, .max_distance = 0});
    EXPECT_GE(checked.size(), static_cast<std::size_t>(correct));
    for (const KeypointMatch& match : checked) {
        EXPECT_EQ(match.index_b, expected_partner[match.index_a]);
    }
}

TEST(MatcherRatio, RejectsAmbiguousQuery) {
    Descriptor query;  // all zero
    Descriptor near_best;
    near_best.bytes[0] = 0x0FU;  // 4 bits set -> distance 4
    Descriptor near_second;
    near_second.bytes[0] = 0x1FU;  // 5 bits set -> distance 5; 4 < 0.8*5 = 4.0 is false

    const std::vector<Descriptor> a{query};
    const std::vector<Descriptor> b{near_best, near_second};
    EXPECT_TRUE(
        match_descriptors(a, b, MatchOptions{.ratio = 0.8F, .cross_check = false, .max_distance = 0})
            .empty());

    // A clearer gap passes: best 1, second 8 -> 1 < 6.4.
    Descriptor clear_best;
    clear_best.bytes[0] = 0x01U;  // distance 1
    Descriptor far_second;
    far_second.bytes[0] = 0xFFU;  // distance 8
    const std::vector<Descriptor> b2{clear_best, far_second};
    const std::vector<KeypointMatch> accepted =
        match_descriptors(a, b2, MatchOptions{.ratio = 0.8F, .cross_check = false, .max_distance = 0});
    ASSERT_EQ(accepted.size(), 1U);
    EXPECT_EQ(accepted[0].index_a, 0U);
    EXPECT_EQ(accepted[0].index_b, 0U);
}

TEST(MatcherRatio, SecondBestZeroFails) {
    // Both candidates are exact; best == second == 0, so best < ratio*second is
    // false and the query is rejected.
    const Descriptor identical;
    const std::vector<Descriptor> a{identical};
    const std::vector<Descriptor> b{identical, identical};
    EXPECT_TRUE(match_descriptors(a, b, MatchOptions{}).empty());
}

TEST(MatcherCrossCheck, RemovesNonMutual) {
    // Distances (byte 0 only):
    //   d(a0,b0)=2  d(a0,b1)=3
    //   d(a1,b0)=1  d(a1,b1)=2
    // Forward a->b: a0->b0 (2<0.8*3) and a1->b0 (1<0.8*2).
    // Reverse b->a: b0->a1 (1<0.8*2) and b1->a1 (2<0.8*3).
    // Cross-check drops a0->b0 (reverse picks a1, not a0) and keeps a1->b0.
    std::vector<Descriptor> a(2);
    std::vector<Descriptor> b(2);
    a[0].bytes[0] = 0x00U;
    a[1].bytes[0] = 0x02U;
    b[0].bytes[0] = 0x03U;
    b[1].bytes[0] = 0x07U;

    const std::vector<KeypointMatch> forward =
        match_descriptors(a, b, MatchOptions{.ratio = 0.8F, .cross_check = false, .max_distance = 0});
    EXPECT_EQ(forward.size(), 2U);

    const std::vector<KeypointMatch> checked =
        match_descriptors(a, b, MatchOptions{.ratio = 0.8F, .cross_check = true, .max_distance = 0});
    ASSERT_EQ(checked.size(), 1U);
    EXPECT_EQ(checked[0].index_a, 1U);
    EXPECT_EQ(checked[0].index_b, 0U);
}

TEST(MatcherMaxDistance, Rejects) {
    // query all-zero; b0 is 8 bits away and b1 is maximally far (504 bits), so
    // the query passes the ratio test (8 < 0.8*504) and matches b0.
    std::vector<Descriptor> a(1);
    std::vector<Descriptor> b(2);
    b[0].bytes[0] = 0xFFU;  // distance 8 from a0
    b[1].bytes.fill(0xFFU);  // distance 504 from a0

    MatchOptions options{.ratio = 0.8F, .cross_check = false, .max_distance = 0};
    const std::vector<KeypointMatch> uncapped = match_descriptors(a, b, options);
    ASSERT_EQ(uncapped.size(), 1U);
    EXPECT_EQ(uncapped[0].index_b, 0U);

    options.max_distance = 4;  // below the best distance of 8
    EXPECT_TRUE(match_descriptors(a, b, options).empty());

    options.max_distance = 8;  // exactly the best distance: still accepted
    EXPECT_EQ(match_descriptors(a, b, options).size(), 1U);
}

TEST(MatcherEdges, EmptyAndSingleCandidate) {
    std::mt19937 rng(11U);
    const std::vector<Descriptor> empty;
    std::vector<Descriptor> two{make_descriptor(rng), make_descriptor(rng)};

    EXPECT_TRUE(match_descriptors(empty, two, MatchOptions{}).empty());
    EXPECT_TRUE(match_descriptors(two, empty, MatchOptions{}).empty());
    EXPECT_TRUE(match_descriptors(empty, empty, MatchOptions{}).empty());

    // A single candidate in b gives no second-best distance, so no match.
    std::vector<Descriptor> one{make_descriptor(rng)};
    EXPECT_TRUE(match_descriptors(two, one, MatchOptions{}).empty());
    EXPECT_TRUE(match_descriptors(one, two, MatchOptions{}).empty());
}

TEST(MatcherDeterminism, ThreadCountIndependent) {
#if defined(ALPHASHAPE_HAVE_OPENMP)
    std::mt19937 rng(99U);
    constexpr int kCount = 1000;
    std::vector<Descriptor> a;
    for (int i = 0; i < kCount; ++i) a.push_back(make_descriptor(rng));
    std::vector<Descriptor> b;
    for (int i = 0; i < kCount; ++i) b.push_back(make_descriptor(rng));
    for (int i = 0; i < 200; ++i) {
        b.push_back(flip_bits(a[static_cast<std::size_t>(i)], 3, rng));
    }

    omp_set_num_threads(1);
    const std::vector<KeypointMatch> single = match_descriptors(a, b, MatchOptions{});
    omp_set_num_threads(4);
    const std::vector<KeypointMatch> multi = match_descriptors(a, b, MatchOptions{});
    EXPECT_EQ(single, multi);
#else
    GTEST_SKIP() << "OpenMP not available";
#endif
}

}  // namespace
}  // namespace alphashape::match
