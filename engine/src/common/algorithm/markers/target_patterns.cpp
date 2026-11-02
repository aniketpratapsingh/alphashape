#include "common/algorithm/markers/target_patterns.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>

namespace alphashape {
namespace {

std::uint32_t bit_mask(int bits) {
    return (bits >= 32) ? 0xFFFFFFFFu : ((1u << static_cast<unsigned>(bits)) - 1u);
}

std::uint32_t rotate_right(std::uint32_t word, int bits) {
    const std::uint32_t mask = bit_mask(bits);
    return ((word >> 1) | ((word & 1u) << static_cast<unsigned>(bits - 1))) & mask;
}

bool is_canonical(std::uint32_t word, int bits) {
    std::uint32_t rotated = word;
    for (int i = 0; i < bits - 1; ++i) {
        rotated = rotate_right(rotated, bits);
        if (rotated < word) {
            return false;
        }
    }
    return true;
}

int minimum_self_distance(std::uint32_t word, int bits) {
    int best = bits;
    std::uint32_t rotated = word;
    for (int i = 0; i < bits - 1; ++i) {
        rotated = rotate_right(rotated, bits);
        best = std::min(best, static_cast<int>(std::popcount(word ^ rotated)));
    }
    return best;
}

/// Mark every word within Hamming distance 2 of `word`. Accepting a table word
/// forbids these words for every later candidate, which is exactly the
/// distance-3 requirement against all rotations of the accepted table.
void mark_radius_two_ball(std::vector<std::uint8_t>& forbidden, std::uint32_t word, int bits) {
    forbidden[word] = 1;
    for (int i = 0; i < bits; ++i) {
        forbidden[word ^ (1u << static_cast<unsigned>(i))] = 1;
    }
    for (int i = 0; i < bits; ++i) {
        for (int j = i + 1; j < bits; ++j) {
            forbidden[word ^ (1u << static_cast<unsigned>(i)) ^
                      (1u << static_cast<unsigned>(j))] = 1;
        }
    }
}

}  // namespace

bool is_supported_circular_bits(int bits) {
    for (int supported : kCircularCodeBits) {
        if (supported == bits) {
            return true;
        }
    }
    return false;
}

std::vector<std::uint32_t> generate_circular_codes(int bits) {
    if (!is_supported_circular_bits(bits)) {
        return {};
    }
    const std::uint32_t size = 1u << static_cast<unsigned>(bits);
    std::vector<std::uint8_t> forbidden(static_cast<std::size_t>(size), 0);
    std::vector<std::uint32_t> accepted;
    for (std::uint32_t word = 1; word + 1 < size; ++word) {
        if (forbidden[word] != 0) continue;
        if (!is_canonical(word, bits)) continue;
        if (minimum_self_distance(word, bits) < 3) continue;
        accepted.push_back(word);
        std::uint32_t rotated = word;
        for (int i = 0; i < bits; ++i) {
            mark_radius_two_ball(forbidden, rotated, bits);
            rotated = rotate_right(rotated, bits);
        }
    }
    return accepted;
}

const std::vector<std::uint32_t>& circular_codes(int bits) {
    static const std::vector<std::uint32_t> empty;
    for (int index = 0; index < kCircularCodeBitCount; ++index) {
        if (kCircularCodeBits[index] != bits) continue;
        static const std::vector<std::vector<std::uint32_t>> tables = [] {
            std::vector<std::vector<std::uint32_t>> generated(
                static_cast<std::size_t>(kCircularCodeBitCount));
            for (int i = 0; i < kCircularCodeBitCount; ++i) {
                generated[static_cast<std::size_t>(i)] = generate_circular_codes(
                    kCircularCodeBits[i]);
            }
            return generated;
        }();
        return tables[static_cast<std::size_t>(index)];
    }
    return empty;
}

std::optional<std::uint32_t> circular_ring_word(int bits, int code) {
    if (code < 0) {
        return std::nullopt;
    }
    const std::vector<std::uint32_t>& table = circular_codes(bits);
    const std::size_t index = static_cast<std::size_t>(code / 2);
    if (index >= table.size()) {
        return std::nullopt;
    }
    return table[index];
}

std::uint32_t canonical_rotation(std::uint32_t word, int bits) {
    if (bits <= 0 || bits > 32) {
        return word;
    }
    std::uint32_t best = word;
    std::uint32_t rotated = word;
    for (int i = 0; i < bits - 1; ++i) {
        rotated = rotate_right(rotated, bits);
        best = std::min(best, rotated);
    }
    return best;
}

int word_hamming(std::uint32_t a, std::uint32_t b) {
    return static_cast<int>(std::popcount(a ^ b));
}

}  // namespace alphashape
