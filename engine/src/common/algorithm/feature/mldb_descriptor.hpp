#pragma once

// M-LDB-style binary descriptor, 63 bytes (504 bits).
//
// We store a binary M-LDB descriptor with a 63-byte stride. M-LDB ("Modified Local Difference Binary") is the AKAZE
// descriptor: Alcantarilla, Nuevo, Bartoli, "Fast Explicit Diffusion for
// Accelerated Features in Nonlinear Scale Spaces", BMVC 2013. It compares mean
// intensity and mean x/y derivatives between pairs of grid cells over a patch
// aligned to the keypoint's orientation.
//
// The 486-bit core follows AKAZE's FULL M-LDB layout: the 2x2, 3x3 and 4x4
// grids, each cell pair compared on three channels (intensity, d/du, d/dv).
//   grids 2x2, 3x3, 4x4 -> 6 + 36 + 120 = 162 cell pairs
//   x 3 channels (intensity, d/du, d/dv)                     = 486 bits
// The remaining 18 bits are OUR OWN addition, not part of AKAZE, present to
// pad the descriptor out to our 63-byte stride:
//   + 3x3 grid, each cell vs the patch mean:
//       intensity (9 bits) and gradient magnitude (9 bits)   =  18 bits
//                                                    total   = 504 bits
// The extra 18 bits carry absolute (cell-vs-patch) contrast, which pairwise
// comparisons alone do not encode. So this is "M-LDB plus 18 bits", NOT
// bit-for-bit AKAZE M-LDB.

#include <vector>

#include "common/algorithm/feature/dog_detector.hpp"
#include "common/core/types.hpp"

namespace alphashape {

struct MldbParams {
    int samples = 24;           ///< samples per side; divisible by 2, 3 and 4
    float patch_radius = 6.0F;  ///< half-width of the patch, in keypoint sigmas (our choice)
};

/// Describe one detection using its octave's Gaussian level.
Descriptor compute_mldb(const ScaleSpace& space, const Detection& detection,
                        const MldbParams& params = {});

std::vector<Descriptor> compute_mldb(const ScaleSpace& space, const std::vector<Detection>& detections,
                                     const MldbParams& params = {});

/// Number of differing bits. The distance a matcher uses on these descriptors.
int hamming_distance(const Descriptor& a, const Descriptor& b);

}  // namespace alphashape
