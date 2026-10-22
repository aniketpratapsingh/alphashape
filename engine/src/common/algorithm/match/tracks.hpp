#pragma once

// Track (tie point) construction: the transitive closure of pairwise keypoint
// matches into multi-image observations, built with a union-find over
// (camera, keypoint) nodes.

#include <cstddef>
#include <vector>

#include "common/core/types.hpp"

namespace alphashape::match {

/// Summary of a build_tracks run.
struct TrackStats {
    std::size_t tracks = 0;
    std::size_t rejected_inconsistent = 0;
};

/// Transitive closure of pairwise matches with union-find over
/// (camera_id, keypoint_index) nodes. A component that contains two different
/// keypoints of the same camera is inconsistent and is dropped whole (counted in
/// rejected_inconsistent). Components of size < 2 cannot occur. Output:
/// observations sorted by camera_id; tracks sorted by their first observation
/// (deterministic).
std::vector<Track> build_tracks(const std::vector<PairMatches>& matches, TrackStats* stats = nullptr);

/// Tie-point limit: for each camera, rank the tracks it observes by length (more
/// observations first; ties by track index) and select its top `per_camera_limit`.
/// Keep any track selected by at least one of its cameras. 0 = no limit. Returns
/// the kept tracks in their original order.
std::vector<Track> limit_tracks(const std::vector<Track>& tracks, std::size_t per_camera_limit);

}  // namespace alphashape::match
