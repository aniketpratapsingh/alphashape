#include "common/algorithm/markers/tag_families.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/algorithm/markers/tag_family_data.hpp"

namespace alphashape {
namespace {

/// Code bit `i` of the layout (most significant published bit first) belongs
/// at bit `bits - 1 - i` of the published integer. Converting once per word
/// lets the table comparison be a plain popcount.
std::uint64_t to_published_word(std::uint64_t layout_word, int bits) {
    std::uint64_t out = 0;
    for (int i = 0; i < bits; ++i) {
        if (((layout_word >> static_cast<unsigned>(i)) & 1u) != 0u) {
            out |= std::uint64_t{1} << static_cast<unsigned>(bits - 1 - i);
        }
    }
    return out;
}

/// Rotation index map for one family: index `i` moves to `map[i]` when the
/// tag turns 90 degrees about its centre. The published layout is symmetric
/// about the border square, so the map is closed for every family.
std::vector<int> build_rotation_map(const TagFamily& family) {
    const int count = static_cast<int>(family.bit_xy.size());
    std::vector<int> map(static_cast<std::size_t>(count), 0);
    const int width = family.width_at_border;
    for (int i = 0; i < count; ++i) {
        const int x = family.bit_xy[static_cast<std::size_t>(i)].first;
        const int y = family.bit_xy[static_cast<std::size_t>(i)].second;
        const int target_x = width - 1 - y;
        const int target_y = x;
        int found = i;
        for (int j = 0; j < count; ++j) {
            if (family.bit_xy[static_cast<std::size_t>(j)].first == target_x &&
                family.bit_xy[static_cast<std::size_t>(j)].second == target_y) {
                found = j;
                break;
            }
        }
        map[static_cast<std::size_t>(i)] = found;
    }
    return map;
}

std::uint64_t reverse_layout(std::uint64_t word, const TagFamily& family) {
    return to_published_word(word, family.bits);
}

std::uint64_t rotate_with_map(std::uint64_t word, const std::vector<int>& map) {
    std::uint64_t out = 0;
    for (std::size_t i = 0; i < map.size(); ++i) {
        if (((word >> i) & 1u) != 0u) {
            assert(map[i] >= 0 && map[i] < 64);
            out |= std::uint64_t{1} << static_cast<unsigned>(map[i]);
        }
    }
    return out;
}

/// Distinct rotated code words, used for the random-word acceptance fraction.
std::size_t distinct_rotations(const TagFamily& family, const std::vector<int>& map) {
    std::vector<std::uint64_t> pool;
    pool.reserve(family.codes.size() * 4);
    for (std::uint64_t code : family.codes) {
        std::uint64_t word = reverse_layout(code, family);
        for (int step = 0; step < 4; ++step) {
            pool.push_back(word);
            word = rotate_with_map(word, map);
        }
    }
    std::sort(pool.begin(), pool.end());
    pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
    return pool.size();
}

/// Largest correction radius whose random-word acceptance is below 1e-4.
/// The balls of radius d around distinct rotated codes are disjoint while
/// d <= (h-1)/2, so the count is exact: |rotations| * sum_{i<=d} C(bits, i).
int compute_default(const TagFamily& family, const std::vector<int>& map) {
    // A family outside 1..64 bits cannot be rotated or matched in a 64-bit
    // word, and its binomial row would not fit; such a table is rejected here
    // rather than reading past the array.
    if (family.bits < 1 || family.bits > 64) {
        return 0;
    }
    const std::size_t rotations = distinct_rotations(family, map);
    const double total = std::ldexp(1.0, family.bits);
    const int allowed = (family.min_hamming - 1) / 2;
    std::array<double, 65> binomial{};
    binomial[0] = 1.0;
    for (int i = 1; i <= family.bits; ++i) {
        binomial[static_cast<std::size_t>(i)] =
            binomial[static_cast<std::size_t>(i - 1)] *
            static_cast<double>(family.bits - i + 1) / static_cast<double>(i);
    }
    int best = 0;
    double ball = 0.0;
    for (int d = 0; d <= allowed; ++d) {
        ball += binomial[static_cast<std::size_t>(d)];
        const double acceptance = static_cast<double>(rotations) * ball / total;
        if (acceptance < 1e-4) best = d;
    }
    return best;
}

struct FamilyCache {
    std::vector<TagFamily> families;
    std::vector<std::vector<int>> rotations;
    std::vector<int> defaults;
    std::vector<int> min_weights;
    std::vector<int> max_weights;
};

FamilyCache build_cache() {
    FamilyCache built;
    for (int index = 0; index < kAprilTagFamilyCount; ++index) {
        const TagFamilyData& data =
            tag_family_data(static_cast<AprilTagFamily>(index));
        TagFamily family;
        family.name = data.name;
        family.bits = data.bits;
        family.min_hamming = data.min_hamming;
        family.total_width = data.total_width;
        family.width_at_border = data.width_at_border;
        family.reversed_border = data.reversed_border;
        family.codes.assign(data.codes.begin(), data.codes.end());
        family.bit_xy.assign(data.bit_xy.begin(), data.bit_xy.end());
        built.families.push_back(std::move(family));
    }
    built.rotations.reserve(built.families.size());
    for (const TagFamily& family : built.families) {
        built.rotations.push_back(build_rotation_map(family));
    }
    for (std::size_t i = 0; i < built.families.size(); ++i) {
        built.defaults.push_back(
            compute_default(built.families[i], built.rotations[i]));
    }
    for (const TagFamily& family : built.families) {
        int lowest = family.bits;
        int highest = 0;
        for (std::uint64_t code : family.codes) {
            const int weight = static_cast<int>(std::popcount(code));
            lowest = std::min(lowest, weight);
            highest = std::max(highest, weight);
        }
        built.min_weights.push_back(lowest);
        built.max_weights.push_back(highest);
    }
    return built;
}

const FamilyCache& cache() {
    static const FamilyCache built = build_cache();
    return built;
}

const TagFamily* lookup(std::string_view name) {
    std::string wanted;
    wanted.reserve(name.size());
    for (char character : name) {
        if (std::isalnum(static_cast<unsigned char>(character)) != 0) {
            wanted.push_back(static_cast<char>(
                std::tolower(static_cast<unsigned char>(character))));
        }
    }
    if (wanted.rfind("tag", 0) == 0 && wanted.size() > 3) {
        wanted.erase(0, 3);
    }
    for (const TagFamily& family : cache().families) {
        std::string candidate;
        candidate.reserve(family.name.size());
        for (char character : family.name) {
            if (std::isalnum(static_cast<unsigned char>(character)) != 0) {
                candidate.push_back(static_cast<char>(
                    std::tolower(static_cast<unsigned char>(character))));
            }
        }
        if (candidate == wanted) {
            return &family;
        }
    }
    return nullptr;
}

}  // namespace

const TagFamily& tag_family(AprilTagFamily family) {
    const int index = static_cast<int>(family);
    static const TagFamily empty;
    if (index < 0 || index >= kAprilTagFamilyCount) {
        return empty;
    }
    return cache().families[static_cast<std::size_t>(index)];
}

std::optional<AprilTagFamily> parse_tag_family(std::string_view name) {
    const TagFamily* family = lookup(name);
    if (family == nullptr) {
        return std::nullopt;
    }
    const TagFamily* first = cache().families.data();
    const std::ptrdiff_t offset = family - first;
    if (offset < 0 || offset >= kAprilTagFamilyCount) {
        return std::nullopt;
    }
    return static_cast<AprilTagFamily>(offset);
}

int default_max_hamming(const TagFamily& family) {
    const std::optional<AprilTagFamily> known = parse_tag_family(family.name);
    if (known) {
        return cache().defaults[static_cast<std::size_t>(*known)];
    }
    return compute_default(family, build_rotation_map(family));
}

int default_max_hamming(AprilTagFamily family) {
    const int index = static_cast<int>(family);
    if (index < 0 || index >= kAprilTagFamilyCount) {
        return 0;
    }
    return cache().defaults[static_cast<std::size_t>(index)];
}

std::uint64_t rotate_tag_word(std::uint64_t word, const TagFamily& family) {
    const std::optional<AprilTagFamily> known = parse_tag_family(family.name);
    if (known) {
        return rotate_with_map(
            word, cache().rotations[static_cast<std::size_t>(*known)]);
    }
    return rotate_with_map(word, build_rotation_map(family));
}

int min_code_weight(const TagFamily& family) {
    const std::optional<AprilTagFamily> known = parse_tag_family(family.name);
    if (known) {
        return cache().min_weights[static_cast<std::size_t>(*known)];
    }
    int lowest = family.bits;
    for (std::uint64_t code : family.codes) {
        lowest = std::min(lowest, static_cast<int>(std::popcount(code)));
    }
    return lowest;
}

int max_code_weight(const TagFamily& family) {
    const std::optional<AprilTagFamily> known = parse_tag_family(family.name);
    if (known) {
        return cache().max_weights[static_cast<std::size_t>(*known)];
    }
    int highest = 0;
    for (std::uint64_t code : family.codes) {
        highest = std::max(highest, static_cast<int>(std::popcount(code)));
    }
    return highest;
}

}  // namespace alphashape
