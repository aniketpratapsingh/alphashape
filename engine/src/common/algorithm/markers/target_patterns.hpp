#pragma once

/// @file target_patterns.hpp
/// @brief Circular coded-target pattern and its code tables (WP-13.6).
///
/// ## Target design (ours)
///
/// The target is a concentric black/white pattern. All radii are in units of
/// the outer radius `R`:
///
/// | region            | radius       | colour |
/// |-------------------|--------------|--------|
/// | centre dot        | `<= 0.25`    | black (parity 1) or white (parity 0) |
/// | white gap         | `0.25 .. 0.5`| white  |
/// | coding ring       | `0.5 .. 0.85`| N equal sectors, bit i = sector i |
/// | white gap         | `0.85 .. 0.9`| white  |
/// | outer border ring | `0.9 .. 1.0` | black  |
/// | quiet zone        | `>= 1.3`     | white  |
///
/// The coding ring is split into `N` equal angular sectors. Sector 0 starts at
/// 12 o'clock (image-up) and sectors advance clockwise in the image. A black
/// sector encodes a 1 bit and a white sector a 0 bit; bit i is sector i.
///
/// A full code id packs the ring word and the centre parity:
/// `code = 2 * ring_index + parity`. `bits` is `N`.
///
/// ## Code construction
///
/// A ring word is a valid table entry when
///
///  * it is not the all-zero or all-one word, and
///  * its Hamming distance to every non-trivial cyclic rotation of itself is
///    at least 3, and
///  * its Hamming distance to every rotation of every other table word is at
///    least 3.
///
/// The table is a greedy lexicode: words are visited in increasing numeric
/// order using the canonical (lexicographically smallest) rotation as the
/// representative, and a word is accepted when it satisfies the constraints
/// against everything accepted so far. The resulting size is whatever the
/// greedy construction yields; it is not forced to a round number.
///
/// With distance 3 between rotated words a single corrupted sector plus an
/// unknown rotation is still uniquely decodable (the decoder additionally
/// requires a two-bit margin to the runner-up ring code).

#include <cstdint>
#include <optional>
#include <vector>

namespace alphashape {

/// Bit widths for which tables exist.
inline constexpr int kCircularCodeBits[] = {12, 14, 16, 20};
inline constexpr int kCircularCodeBitCount = 4;

/// True when `bits` is one of the supported widths.
bool is_supported_circular_bits(int bits);

/// Sorted ring words for `bits`, each packed so that sector 0 is the most
/// significant of the `bits` low bits. Returns an empty vector for an
/// unsupported width.
std::vector<std::uint32_t> generate_circular_codes(int bits);

/// Cached table for `bits` (empty for an unsupported width).
const std::vector<std::uint32_t>& circular_codes(int bits);

/// Packed word for `code`: `code = 2 * ring_index + parity`, so the ring word
/// is `code >> 1` indexed into the table.
std::optional<std::uint32_t> circular_ring_word(int bits, int code);

/// Canonical rotation of `word` (lexicographically smallest rotation).
std::uint32_t canonical_rotation(std::uint32_t word, int bits);

/// Hamming distance between two `bits`-wide words.
int word_hamming(std::uint32_t a, std::uint32_t b);

}  // namespace alphashape
