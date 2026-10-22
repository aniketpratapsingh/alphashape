#include "common/algorithm/match/preselection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

#include "base/utils/error.hpp"
#include "common/algorithm/match/hamming_matcher.hpp"

namespace alphashape::match {
namespace {

constexpr double kPi = 3.14159265358979323846;
/// WGS84 mean radius, in metres. The local tangent-plane approximation below is
/// documented as such: equirectangular about the scene centroid's latitude.
constexpr double kEarthMeanRadiusM = 6371008.8;

CameraPair make_pair(std::int32_t x, std::int32_t y) {
    return x < y ? CameraPair{x, y} : CameraPair{y, x};
}

void check_unique(const std::vector<std::int32_t>& camera_ids) {
    std::unordered_set<std::int32_t> seen;
    seen.reserve(camera_ids.size() * 2);
    for (const std::int32_t id : camera_ids) {
        if (!seen.insert(id).second) {
            throw ValueError("duplicate camera id " + std::to_string(id) +
                             " in preselection input");
        }
    }
}

/// Sort and de-duplicate a pair vector (it is assumed to be already a<b).
std::vector<CameraPair> normalise(std::vector<CameraPair> pairs) {
    std::sort(pairs.begin(), pairs.end(), [](const CameraPair& x, const CameraPair& y) {
        if (x.a != y.a) return x.a < y.a;
        return x.b < y.b;
    });
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    return pairs;
}

}  // namespace

std::vector<CameraPair> exhaustive_pairs(const std::vector<std::int32_t>& camera_ids) {
    check_unique(camera_ids);
    std::vector<CameraPair> pairs;
    const std::size_t n = camera_ids.size();
    pairs.reserve(n * (n - 1) / 2);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            pairs.push_back(make_pair(camera_ids[i], camera_ids[j]));
        }
    }
    return normalise(std::move(pairs));
}

std::vector<CameraPair> sequential_pairs(const std::vector<std::int32_t>& camera_ids,
                                         int window) {
    check_unique(camera_ids);
    if (window < 1) {
        throw ValueError("sequential_pairs window must be >= 1");
    }
    std::vector<CameraPair> pairs;
    const std::size_t n = camera_ids.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t step = 1; step <= static_cast<std::size_t>(window); ++step) {
            const std::size_t j = i + step;
            if (j >= n) break;
            pairs.push_back(make_pair(camera_ids[i], camera_ids[j]));
        }
    }
    return normalise(std::move(pairs));
}

std::vector<CameraPair> reference_pairs(
    const std::vector<std::pair<std::int32_t, Eigen::Vector3d>>& locations,
    const std::vector<std::int32_t>& unreferenced, int neighbors) {
    if (neighbors < 1) {
        throw ValueError("reference_pairs neighbors must be >= 1");
    }

    std::vector<std::int32_t> all_ids;
    all_ids.reserve(locations.size() + unreferenced.size());
    for (const auto& entry : locations) all_ids.push_back(entry.first);
    for (const std::int32_t id : unreferenced) all_ids.push_back(id);
    check_unique(all_ids);
    check_unique(unreferenced);

    std::vector<CameraPair> pairs;

    // Each unreferenced camera (no GPS) is paired with every other camera: the
    // safe fallback, since we cannot reason about its position.
    for (const std::int32_t id : unreferenced) {
        for (const std::int32_t other : all_ids) {
            if (other != id) pairs.push_back(make_pair(id, other));
        }
    }

    // Referenced cameras: build the local tangent plane about the centroid.
    const std::size_t n = locations.size();
    if (n < 2) return normalise(std::move(pairs));

    double lon_sum = 0.0;
    double lat_sum = 0.0;
    for (const auto& entry : locations) {
        lon_sum += entry.second.x();
        lat_sum += entry.second.y();
    }
    const double lon0 = lon_sum / static_cast<double>(n);
    const double lat0 = lat_sum / static_cast<double>(n);
    // Equirectangular projection: longitude scaled by cos(lat0); WGS84 mean
    // radius. Valid for scenes small relative to the Earth (our use case) and
    // documented here as the required approximation. A scene straddling the
    // antimeridian is not supported by this simple mean.
    const double cos_lat0 = std::cos(lat0 * kPi / 180.0);

    // Local metric coordinates; z is altitude (NaN allowed).
    std::vector<Eigen::Vector3d> local(n);
    for (std::size_t i = 0; i < n; ++i) {
        const Eigen::Vector3d& p = locations[i].second;
        const double x = (p.x() - lon0) * kPi / 180.0 * cos_lat0 * kEarthMeanRadiusM;
        const double y = (p.y() - lat0) * kPi / 180.0 * kEarthMeanRadiusM;
        local[i] = Eigen::Vector3d(x, y, p.z());
    }

    for (std::size_t i = 0; i < n; ++i) {
        std::vector<std::pair<double, std::size_t>> ranked;
        ranked.reserve(n - 1);
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            const double dx = local[i].x() - local[j].x();
            const double dy = local[i].y() - local[j].y();
            // Horizontal distance only when either altitude is unknown.
            double distance = 0.0;
            if (std::isnan(local[i].z()) || std::isnan(local[j].z())) {
                distance = std::sqrt(dx * dx + dy * dy);
            } else {
                const double dz = local[i].z() - local[j].z();
                distance = std::sqrt(dx * dx + dy * dy + dz * dz);
            }
            ranked.emplace_back(distance, j);
        }
        // Nearest first; ties broken by position in `locations` for determinism.
        std::sort(ranked.begin(), ranked.end(),
                  [](const std::pair<double, std::size_t>& x,
                     const std::pair<double, std::size_t>& y) {
                      if (x.first != y.first) return x.first < y.first;
                      return x.second < y.second;
                  });
        const std::size_t take = std::min(ranked.size(), static_cast<std::size_t>(neighbors));
        for (std::size_t k = 0; k < take; ++k) {
            pairs.push_back(make_pair(locations[i].first, locations[ranked[k].second].first));
        }
    }

    return normalise(std::move(pairs));
}

namespace {

/// The coarse feature set for one camera: the indices of its largest-scale
/// keypoints, at most `limit`, ordered by (scale desc, index asc). Ours.
std::vector<std::uint32_t> coarse_indices(const KeypointSet& set, int limit) {
    std::vector<std::uint32_t> order(set.keypoints.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = static_cast<std::uint32_t>(i);
    }
    std::sort(order.begin(), order.end(),
              [&set](std::uint32_t lhs, std::uint32_t rhs) {
                  const float sl = set.keypoints[lhs].scale;
                  const float sr = set.keypoints[rhs].scale;
                  if (sl != sr) return sl > sr;
                  return lhs < rhs;
              });
    if (limit >= 0 && order.size() > static_cast<std::size_t>(limit)) {
        order.resize(static_cast<std::size_t>(limit));
    }
    return order;
}

}  // namespace

std::vector<CameraPair> generic_pairs(const std::vector<KeypointSet>& keypoint_sets,
                                      const std::vector<CameraPair>& candidates,
                                      const GenericOptions& options) {
    std::unordered_map<std::int32_t, const KeypointSet*> by_camera;
    by_camera.reserve(keypoint_sets.size() * 2);
    for (const KeypointSet& set : keypoint_sets) {
        by_camera[set.camera_id] = &set;
    }

    // Precompute each camera's coarse descriptor set once. `coarse_descriptors`
    // holds the reduced descriptors in the same order as `coarse_indices`.
    std::unordered_map<std::int32_t, std::vector<Descriptor>> coarse_descriptors;
    for (const KeypointSet& set : keypoint_sets) {
        const std::vector<std::uint32_t> indices = coarse_indices(set, options.prematch_keypoints);
        std::vector<Descriptor> subset;
        subset.reserve(indices.size());
        for (const std::uint32_t index : indices) {
            subset.push_back(set.descriptors[index]);
        }
        coarse_descriptors.emplace(set.camera_id, std::move(subset));
    }

    const std::size_t pair_count = candidates.size();
    std::vector<int> score(pair_count, 0);

#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(pair_count); ++i) {
        const std::size_t index = static_cast<std::size_t>(i);
        const CameraPair& pair = candidates[index];
        const auto it_a = coarse_descriptors.find(pair.a);
        const auto it_b = coarse_descriptors.find(pair.b);
        if (it_a == coarse_descriptors.end() || it_b == coarse_descriptors.end()) {
            continue;  // a camera with no keypoint set contributes no pairs
        }
        const std::vector<KeypointMatch> matched =
            match_descriptors(it_a->second, it_b->second, MatchOptions{});
        score[index] = static_cast<int>(matched.size());
    }

    // min_score gate, then each camera keeps its top pairs_per_image partners.
    std::vector<char> keep(pair_count, 0);
    std::unordered_map<std::int32_t, std::vector<std::size_t>> by_partner;  // camera -> candidate indices
    by_partner.reserve(keypoint_sets.size() * 4);
    for (std::size_t i = 0; i < pair_count; ++i) {
        if (score[i] < options.min_score) continue;
        by_partner[candidates[i].a].push_back(i);
        by_partner[candidates[i].b].push_back(i);
    }

    for (auto& entry : by_partner) {
        std::vector<std::size_t>& indices = entry.second;
        const std::int32_t self = entry.first;
        const auto partner_of = [&candidates, self](std::size_t index) {
            return candidates[index].a == self ? candidates[index].b : candidates[index].a;
        };
        std::sort(indices.begin(), indices.end(),
                  [&score, &partner_of, &candidates](std::size_t x, std::size_t y) {
                      if (score[x] != score[y]) return score[x] > score[y];
                      const std::int32_t px = partner_of(x);
                      const std::int32_t py = partner_of(y);
                      if (px != py) return px < py;
                      return candidates[x].b < candidates[y].b;
                  });
        std::size_t take = indices.size();
        if (options.pairs_per_image >= 0 && take > static_cast<std::size_t>(options.pairs_per_image)) {
            take = static_cast<std::size_t>(options.pairs_per_image);
        }
        for (std::size_t k = 0; k < take; ++k) keep[indices[k]] = 1;
    }

    std::vector<CameraPair> result;
    for (std::size_t i = 0; i < pair_count; ++i) {
        if (keep[i] != 0) result.push_back(candidates[i]);
    }
    return normalise(std::move(result));
}

}  // namespace alphashape::match
