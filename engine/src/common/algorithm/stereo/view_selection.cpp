#include "common/algorithm/stereo/view_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"

namespace alphashape::stereo {
namespace {

// The valid options range and the depth-band margins live here as named
// constants so the intent is explicit and the tests can cite them.
//
// Depth range: we take the 2nd/98th percentile of the observed depth samples
// and then widen the band outward by these margins so the matcher still has
// headroom for points that sit just beyond the tie-point cloud. The margins are
// ours: a modest 0.75x / 1.33x around the inner
// band, which is wide enough not to clip the cloud and tight enough to exclude
// gross outliers a bad tie point might introduce.
constexpr double kDepthLowMargin = 0.75;
constexpr double kDepthHighMargin = 1.33;

// A neighbour list longer than this is almost certainly a mistake, and the
// header reserves the top of the range, so reject it up front.
constexpr int kMaxNeighborsExclusive = 256;

// Rays shorter than this (in chunk-local units) carry no usable direction and
// the camera centre effectively coincides with the point; skip such a sample.
constexpr double kMinRayLength = 1e-9;

// An ordered (reference, neighbour) pair, used as the accumulation key.
struct PairKey {
    std::int32_t reference = 0;
    std::int32_t neighbour = 0;

    bool operator==(const PairKey& other) const {
        return reference == other.reference && neighbour == other.neighbour;
    }
};

struct PairKeyHash {
    std::size_t operator()(const PairKey& key) const noexcept {
        const auto a = static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.reference));
        const auto b = static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.neighbour));
        return static_cast<std::size_t>((a << 32) ^ b);
    }
};

struct PairStats {
    int count = 0;    // tracks whose triangulation angle fell inside the window
    double score = 0.0;
};

/// Camera centre C = -R^T t for a world->camera pose.
Eigen::Vector3d camera_centre(const Camera& camera) {
    const Eigen::Matrix3d r = camera.pose.topLeftCorner<3, 3>();
    const Eigen::Vector3d t = camera.pose.topRightCorner<3, 1>();
    return -r.transpose() * t;
}

/// Triangulation angle in degrees between the two viewing rays at `point`,
/// or -1 when either ray is degenerate.
double triangulation_angle_deg(const Eigen::Vector3d& centre_a,
                               const Eigen::Vector3d& centre_b,
                               const Eigen::Vector3d& point) {
    const Eigen::Vector3d ray_a = point - centre_a;
    const Eigen::Vector3d ray_b = point - centre_b;
    const double norm_a = ray_a.norm();
    const double norm_b = ray_b.norm();
    if (norm_a < kMinRayLength || norm_b < kMinRayLength) {
        return -1.0;
    }
    double cosine = ray_a.dot(ray_b) / (norm_a * norm_b);
    cosine = std::clamp(cosine, -1.0, 1.0);
    return std::acos(cosine) * 180.0 / 3.14159265358979323846;
}

/// Nearest-rank percentile of `samples` on the (n-1) scaling. Reorders
/// `samples`; `samples` must be non-empty.
double percentile_in_place(std::vector<double>& samples, double fraction) {
    const std::size_t n = samples.size();
    const auto idx = static_cast<std::size_t>(
        std::llround(fraction * static_cast<double>(n - 1)));
    const std::size_t clamped = std::min(idx, n - 1);
    std::nth_element(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(clamped),
                     samples.end());
    return samples[clamped];
}

void validate_options(const ViewSelectionOptions& options) {
    if (options.max_neighbors < 1 || options.max_neighbors >= kMaxNeighborsExclusive) {
        throw ValueError("stereo::select_views: max_neighbors must be in [1, 256)");
    }
    if (options.min_common_points < 1) {
        throw ValueError("stereo::select_views: min_common_points must be >= 1");
    }
    if (!(options.min_angle_deg >= 0.0 && options.min_angle_deg < options.max_angle_deg &&
          options.max_angle_deg <= 180.0)) {
        throw ValueError(
            "stereo::select_views: require 0 <= min_angle_deg < max_angle_deg <= 180");
    }
}

}  // namespace

std::vector<ViewNeighbourhood> select_views(const Chunk& chunk,
                                            const ViewSelectionOptions& options) {
    validate_options(options);

    const std::vector<Camera>& cameras = chunk.cameras();

    // Camera id -> index into `cameras`, plus the centre and depth samples per
    // enabled camera. Only aligned (enabled) cameras participate; a disabled
    // camera is neither an output row nor a candidate neighbour.
    std::unordered_map<std::int32_t, std::size_t> index_of_enabled;
    std::vector<bool> enabled(cameras.size(), false);
    std::vector<Eigen::Vector3d> centre(cameras.size(), Eigen::Vector3d::Zero());
    std::vector<std::vector<double>> depths(cameras.size());

    for (std::size_t i = 0; i < cameras.size(); ++i) {
        if (!cameras[i].enabled) {
            continue;
        }
        enabled[i] = true;
        index_of_enabled.emplace(cameras[i].id, i);
        centre[i] = camera_centre(cameras[i]);
    }

    // Pair accumulation: for every track with a point, every ordered pair of
    // enabled cameras observing it gets a triangulation-angle test, and inside
    // the window the pair's count increments and its score gains a weight. The
    // weight is 1.0 per in-window track ("keep it simple", per the brief): a
    // pair's score is then its in-window shared-point count, which already
    // ranks a pair with more usable overlap above one with less.
    std::unordered_map<PairKey, PairStats, PairKeyHash> pairs;

    const std::vector<Track>& tracks = chunk.tie_points().tracks;
    for (const Track& track : tracks) {
        if (!track.point.has_value()) {
            continue;
        }
        const Eigen::Vector3d point = *track.point;

        // Observations carry a camera id; project each onto our enabled index
        // and keep the ones that are aligned. Depth samples feed the range.
        std::vector<std::pair<std::int32_t, std::size_t>> observed;
        observed.reserve(track.observations.size());
        for (const Observation& observation : track.observations) {
            const auto found = index_of_enabled.find(observation.camera_id);
            if (found == index_of_enabled.end()) {
                continue;
            }
            observed.emplace_back(observation.camera_id, found->second);
        }
        if (observed.size() < 2) {
            continue;
        }

        for (const auto& observation : observed) {
            depths[observation.second].push_back((point - centre[observation.second]).norm());
        }

        for (std::size_t a = 0; a < observed.size(); ++a) {
            for (std::size_t b = 0; b < observed.size(); ++b) {
                if (a == b) {
                    continue;
                }
                const std::int32_t id_a = observed[a].first;
                const std::int32_t id_b = observed[b].first;
                const double angle = triangulation_angle_deg(
                    centre[observed[a].second], centre[observed[b].second], point);
                if (angle < 0.0 || angle < options.min_angle_deg ||
                    angle > options.max_angle_deg) {
                    continue;
                }
                PairStats& stats = pairs[{id_a, id_b}];
                stats.count += 1;
                stats.score += 1.0;  // one in-window track == one unit of weight
            }
        }
    }

    std::vector<ViewNeighbourhood> result;

    for (std::size_t i = 0; i < cameras.size(); ++i) {
        if (!enabled[i]) {
            continue;
        }

        // Depth range for this camera. Fewer than three samples is not enough
        // for a robust inner band, so the camera is skipped entirely.
        std::vector<double> samples = depths[i];
        if (samples.size() < 3) {
            continue;
        }
        const double p_lo = percentile_in_place(samples, 0.02);
        const double p_hi = percentile_in_place(samples, 0.98);
        const double depth_min = kDepthLowMargin * p_lo;
        const double depth_max = kDepthHighMargin * p_hi;
        if (!(depth_min > 0.0)) {
            continue;
        }

        // Candidate neighbours: pairs (this camera, other) with enough shared
        // in-window points. Ties break on ascending neighbour id for
        // determinism, independent of hash-map iteration order.
        std::vector<std::pair<std::int32_t, double>> candidates;
        const std::int32_t self_id = cameras[i].id;
        for (const auto& [key, stats] : pairs) {
            if (key.reference != self_id || stats.count < options.min_common_points) {
                continue;
            }
            candidates.emplace_back(key.neighbour, stats.score);
        }
        if (candidates.empty()) {
            continue;
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& lhs, const auto& rhs) {
                      if (lhs.second != rhs.second) {
                          return lhs.second > rhs.second;  // higher score first
                      }
                      return lhs.first < rhs.first;  // stable, id ascending
                  });
        if (static_cast<int>(candidates.size()) > options.max_neighbors) {
            candidates.resize(static_cast<std::size_t>(options.max_neighbors));
        }

        ViewNeighbourhood neighbourhood;
        neighbourhood.camera_id = self_id;
        neighbourhood.neighbours.reserve(candidates.size());
        neighbourhood.scores.reserve(candidates.size());
        for (const auto& [neighbour_id, score] : candidates) {
            neighbourhood.neighbours.push_back(neighbour_id);
            neighbourhood.scores.push_back(score);
        }
        neighbourhood.depth_min = depth_min;
        neighbourhood.depth_max = depth_max;
        result.push_back(std::move(neighbourhood));
    }

    return result;
}

}  // namespace alphashape::stereo
