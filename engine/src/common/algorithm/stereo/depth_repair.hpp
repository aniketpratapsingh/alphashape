#pragma once

// Post-filter repair of small depth-map holes.
//
// The cross-view consistency filter (depth_filter.hpp) only DELETES pixels, so
// it opens holes inside surfaces that were otherwise consistent. This pass puts
// them back where a single plane explains the hole's boundary, and leaves them
// empty where it does not (a hole on a depth discontinuity, e.g. a roof edge
// beside the ground, must never be bridged).
//
// The family is local surface fitting: a robust plane is fitted to the 3-D
// boundary points in camera space (RANSAC, Fischler & Bolles 1981, "Random
// Sample Consensus", CACM 24(6); the segmentation family it belongs to is
// summarised in Zollhofer et al. 2018, "State of the Art on 3D Reconstruction
// with RGB-D Cameras", CGF 37(2)), and each hole pixel is filled by intersecting
// its viewing ray with that plane. Working in 3-D rather than on the depth image
// is what makes a slanted surface fill correctly.

#include <cstddef>

#include "common/core/chunk.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::stereo {

struct RepairOptions {
    /// Largest 4-connected invalid component, in pixels at the map's working
    /// resolution, that may be repaired. Bigger holes are not a filter artefact
    /// and are left empty.
    int max_area = 2000;
};

struct RepairStats {
    std::size_t components = 0;         ///< invalid 4-connected components seen
    std::size_t filled_components = 0;  ///< components a plane explained
    std::size_t filled_pixels = 0;
    std::size_t rejected_area = 0;       ///< components larger than max_area
    std::size_t rejected_boundary = 0;   ///< too few boundary points / open boundary
    std::size_t rejected_plane = 0;      ///< boundary not explained by one plane
    std::size_t unfilled_pixels = 0;     ///< pixels a grazing/off-range ray skipped
};

/// Returns a repaired copy of `map`. Pixels the fill produces get the plane
/// normal and kDepthRepairedFlag set in `flags`; that bit is how a later stage
/// tells a synthetic pixel from a measured one.
DepthMap repair_depth_map(const DepthMap& map, const Calibration& calibration,
                          const RepairOptions& options, RepairStats* stats = nullptr);

/// Confidence written on a repaired pixel. It is ZERO, because a repaired pixel
/// has no measured support, and zero is the only value guaranteed not to exceed
/// a measured pixel's: the filter keeps a pixel with no supporting neighbour at
/// confidence 0, so any positive constant would invert the ordering. Provenance
/// therefore lives in DepthMap::flags (kDepthRepairedFlag), not in this value;
/// the confidence is a fallback for a reader that predates the flag.
constexpr float kRepairedConfidence = 0.0F;

}  // namespace alphashape::stereo
