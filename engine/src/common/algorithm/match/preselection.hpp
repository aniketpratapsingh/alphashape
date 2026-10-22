#pragma once

// Camera-pair preselection: deciding which image pairs are worth matching at
// all. Three strategies:
//
//   * exhaustive_pairs   - every unordered pair (small projects)
//   * sequential_pairs   - each image with the next `window` in capture order
//   * reference_pairs    - GPS proximity (each image with its nearest neighbours)
//
// plus generic_pairs, a cheap low-resolution prematch used when no reference
// data exists.
//
// The GenericOptions defaults are OUR engineering choices: a coarse prematch over each image's largest-scale
// keypoints, keeping each image's strongest partners. They are documented as
// ours deliberately (see the spec).

#include <cstdint>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "common/core/types.hpp"

namespace alphashape::match {

/// An unordered camera pair, normalised so a < b.
struct CameraPair {
    std::int32_t a = -1;
    std::int32_t b = -1;

    bool operator==(const CameraPair&) const = default;
};

/// Every unordered pair, sorted lexicographically.
std::vector<CameraPair> exhaustive_pairs(const std::vector<std::int32_t>& camera_ids);

/// Each camera paired with the next `window` cameras in the given order.
std::vector<CameraPair> sequential_pairs(const std::vector<std::int32_t>& camera_ids, int window);

/// GPS neighbours. `locations` = (camera_id, (lon_deg, lat_deg, alt_m)); alt may
/// be NaN. Positions are converted to local metres around the centroid with a
/// local tangent-plane approximation (equirectangular with cos(lat0) and the
/// WGS84 mean radius); if either altitude is NaN the horizontal distance only is
/// used. Each camera is paired with its `neighbors` nearest; the result is the
/// symmetric union, sorted and de-duplicated. Cameras listed in `unreferenced`
/// (no GPS) are paired with EVERY camera, as the safe fallback.
std::vector<CameraPair> reference_pairs(
    const std::vector<std::pair<std::int32_t, Eigen::Vector3d>>& locations,
    const std::vector<std::int32_t>& unreferenced, int neighbors);

/// Generic preselection parameters. OURS: per image, `prematch_keypoints`
/// largest-scale keypoints are the coarse feature set; each image keeps its top
/// `pairs_per_image` partners by prematch score; pairs scoring below `min_score`
/// are dropped.
struct GenericOptions {
    int prematch_keypoints = 1000;  ///< ours: per image, keypoints with largest scale (coarse features)
    int pairs_per_image = 30;       ///< ours: keep each image's top-N partners by prematch score
    int min_score = 5;              ///< ours: pairs scoring below this are dropped
};

/// Generic preselection: cheap low-resolution prematch. For each candidate pair,
/// match only the `prematch_keypoints` largest-scale keypoints of each image
/// (match_descriptors, default options) and score the pair by match count. Each
/// image keeps its top `pairs_per_image` partners. The result is the union over
/// images, a subset of `candidates`. Keypoint sets are looked up by camera id; a
/// camera with no keypoint set contributes no pairs.
std::vector<CameraPair> generic_pairs(const std::vector<KeypointSet>& keypoint_sets,
                                      const std::vector<CameraPair>& candidates,
                                      const GenericOptions& options);

}  // namespace alphashape::match
