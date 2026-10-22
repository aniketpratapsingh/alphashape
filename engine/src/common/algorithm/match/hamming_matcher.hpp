#pragma once

// Brute-force binary descriptor matching in Hamming space.
//
// M-LDB descriptors are stored at 63 bytes (504 bits) and matched with a
// 2-nearest-neighbour Hamming search plus Lowe's 2004 ratio test: our own
// implementation, published technique only.

#include <cstdint>
#include <vector>

#include "common/core/types.hpp"

namespace alphashape::match {

/// Matching parameters. The defaults mirror Metashape's public Python API
/// (ratio 0.8, cross-check on) so scripts read the same.
struct MatchOptions {
    float ratio = 0.8F;        ///< Lowe 2004's published value: accept best < ratio * second
    bool cross_check = true;   ///< keep only mutual best matches
    int max_distance = 0;      ///< reject best distance > this; 0 = no absolute cap
};

/// Brute-force 2-NN in Hamming space over 504-bit (63-byte) descriptors.
/// Returns (index_a, index_b) pairs sorted by index_a. A query with fewer than
/// two candidates in `b` cannot pass the ratio test and is never matched.
std::vector<KeypointMatch> match_descriptors(const std::vector<Descriptor>& a,
                                             const std::vector<Descriptor>& b,
                                             const MatchOptions& options);

/// Hamming distance between two descriptors: the number of differing bits.
int hamming(const Descriptor& x, const Descriptor& y);

}  // namespace alphashape::match
