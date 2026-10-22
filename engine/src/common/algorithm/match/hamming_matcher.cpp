#include "common/algorithm/match/hamming_matcher.hpp"

#include <algorithm>
#include <array>
#include <bit>          // std::popcount
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <vector>

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

namespace alphashape::match {
namespace {

// A 63-byte descriptor is 504 bits; we pack it into 8 64-bit words (512 bits)
// with the final byte zero-padded. Packing happens once per match_descriptors
// call, never per comparison.
constexpr std::size_t kWords = 8;
constexpr std::size_t kBufferBytes = kWords * 8;  // 64 bytes, one pad byte

using Packed = std::array<std::uint64_t, kWords>;

Packed pack(const Descriptor& descriptor) {
    // Copy the 63 stored bytes into a 64-byte zeroed buffer so the trailing
    // byte is 0, then reinterpret as 8 little-endian words. The buffer is
    // byte-aligned, so no aliasing or alignment assumptions are made.
    std::array<std::uint8_t, kBufferBytes> buffer{};
    std::memcpy(buffer.data(), descriptor.bytes.data(), Descriptor::kBytes);
    Packed words{};
    std::memcpy(words.data(), buffer.data(), buffer.size());
    return words;
}

int hamming_packed(const Packed& x, const Packed& y) {
    int distance = 0;
    for (std::size_t word = 0; word < kWords; ++word) {
        distance += std::popcount(x[word] ^ y[word]);
    }
    return distance;
}

/// 2-nearest-neighbour search of every descriptor in `a` against `b`, applying
/// the ratio test and the optional absolute cap. Each query yields at most one
/// match, so the result (ordered by index_a) is already sorted.
///
/// Tie-breaking: candidates are scanned in ascending index order and the best
/// is replaced only on a strictly smaller distance, so at equal distance the
/// lowest index wins. Parallelised over queries; the per-query outputs are
/// combined in index order, making the result independent of thread count.
std::vector<KeypointMatch> ratio_matches(const std::vector<Packed>& a,
                                         const std::vector<Packed>& b,
                                         const MatchOptions& options) {
    std::vector<KeypointMatch> result;
    // Fewer than two candidates in b means there is no second-best distance and
    // the ratio test cannot be evaluated, so nothing matches.
    if (a.empty() || b.size() < 2) return result;

    std::vector<char> accepted(a.size(), 0);
    std::vector<std::uint32_t> partner(a.size(), 0);

#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t query = 0; query < static_cast<std::ptrdiff_t>(a.size()); ++query) {
        const std::size_t i = static_cast<std::size_t>(query);
        int best = std::numeric_limits<int>::max();
        int second = std::numeric_limits<int>::max();
        std::uint32_t best_index = 0;
        for (std::size_t j = 0; j < b.size(); ++j) {
            const int d = hamming_packed(a[i], b[j]);
            if (d < best) {
                second = best;
                best = d;
                best_index = static_cast<std::uint32_t>(j);
            } else if (d < second) {
                second = d;
            }
        }
        // Integer distances, evaluated as floats per Lowe 2004. If the second
        // best is 0 the best is also 0 and the strict inequality fails.
        const bool ratio_ok =
            static_cast<float>(best) < options.ratio * static_cast<float>(second);
        const bool cap_ok = options.max_distance <= 0 || best <= options.max_distance;
        if (ratio_ok && cap_ok) {
            accepted[i] = 1;
            partner[i] = best_index;
        }
    }

    result.reserve(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (accepted[i] != 0) {
            result.push_back(KeypointMatch{static_cast<std::uint32_t>(i), partner[i]});
        }
    }
    return result;
}

std::uint64_t key(std::uint32_t first, std::uint32_t second) {
    return (static_cast<std::uint64_t>(first) << 32) | static_cast<std::uint64_t>(second);
}

}  // namespace

int hamming(const Descriptor& x, const Descriptor& y) {
    return hamming_packed(pack(x), pack(y));
}

std::vector<KeypointMatch> match_descriptors(const std::vector<Descriptor>& a,
                                             const std::vector<Descriptor>& b,
                                             const MatchOptions& options) {
    if (a.empty() || b.empty()) return {};

    std::vector<Packed> packed_a;
    packed_a.reserve(a.size());
    for (const Descriptor& descriptor : a) packed_a.push_back(pack(descriptor));
    std::vector<Packed> packed_b;
    packed_b.reserve(b.size());
    for (const Descriptor& descriptor : b) packed_b.push_back(pack(descriptor));

    std::vector<KeypointMatch> forward = ratio_matches(packed_a, packed_b, options);
    if (!options.cross_check) return forward;

    // Cross-check: match b->a under the same rule and keep only the forward
    // matches whose reverse entry is the exact mutual pair (index_a here is an
    // index into b, index_b an index into a).
    const std::vector<KeypointMatch> reverse = ratio_matches(packed_b, packed_a, options);
    std::unordered_set<std::uint64_t> reverse_keys;
    reverse_keys.reserve(reverse.size() * 2);
    for (const KeypointMatch& match : reverse) {
        reverse_keys.insert(key(match.index_a, match.index_b));
    }

    std::vector<KeypointMatch> mutual;
    mutual.reserve(forward.size());
    for (const KeypointMatch& match : forward) {
        if (reverse_keys.contains(key(match.index_b, match.index_a))) {
            mutual.push_back(match);
        }
    }
    return mutual;
}

}  // namespace alphashape::match
