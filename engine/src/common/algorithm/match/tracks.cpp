#include "common/algorithm/match/tracks.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

namespace alphashape::match {
namespace {

/// Packing of a (camera_id, keypoint_index) node into one 64-bit key. The
/// keypoint index is 32 bits and the camera id is 32 bits (the id is signed but
/// used as a bit pattern here, which is all we need for map identity).
std::uint64_t node_key(std::int32_t camera_id, std::uint32_t keypoint_index) {
    const std::uint64_t cam = static_cast<std::uint64_t>(static_cast<std::uint32_t>(camera_id));
    return (cam << 32) | static_cast<std::uint64_t>(keypoint_index);
}

/// Union-find with path compression and union by size. Nodes are addressed by
/// their packed 64-bit key; a node is created on first reference.
class DisjointSet {
public:
    std::uint32_t find(std::uint32_t node) {
        std::uint32_t root = node;
        while (parent_[root] != root) root = parent_[root];
        // Path compression.
        while (parent_[node] != root) {
            const std::uint32_t next = parent_[node];
            parent_[node] = root;
            node = next;
        }
        return root;
    }

    void unite(std::uint32_t x, std::uint32_t y) {
        std::uint32_t rx = find(x);
        std::uint32_t ry = find(y);
        if (rx == ry) return;
        if (size_[rx] < size_[ry]) std::swap(rx, ry);
        parent_[ry] = rx;
        size_[rx] += size_[ry];
    }

    /// Intern a packed key, returning its dense node id (creating it if new).
    std::uint32_t intern(std::uint64_t key) {
        const auto [it, inserted] = ids_.try_emplace(key, static_cast<std::uint32_t>(parent_.size()));
        if (inserted) {
            parent_.push_back(it->second);
            size_.push_back(1);
        }
        return it->second;
    }

private:
    std::unordered_map<std::uint64_t, std::uint32_t> ids_;
    std::vector<std::uint32_t> parent_;
    std::vector<std::uint32_t> size_;
};

}  // namespace

std::vector<Track> build_tracks(const std::vector<PairMatches>& matches, TrackStats* stats) {
    DisjointSet sets;

    /// One interned node and the (camera, index) it stands for.
    struct NodeInfo {
        std::int32_t camera_id;
        std::uint32_t keypoint_index;
    };
    std::unordered_map<std::uint32_t, NodeInfo> info;
    const auto remember = [&info, &sets](std::int32_t camera_id, std::uint32_t keypoint_index) {
        const std::uint32_t node = sets.intern(node_key(camera_id, keypoint_index));
        info.emplace(node, NodeInfo{camera_id, keypoint_index});
        return node;
    };

    for (const PairMatches& pair : matches) {
        for (const KeypointMatch& match : pair.matches) {
            const std::uint32_t na = remember(pair.camera_a, match.index_a);
            const std::uint32_t nb = remember(pair.camera_b, match.index_b);
            sets.unite(na, nb);
        }
    }

    if (stats != nullptr) {
        stats->tracks = 0;
        stats->rejected_inconsistent = 0;
    }

    // Group node ids by component root.
    std::map<std::uint32_t, std::vector<std::uint32_t>> components;
    for (const auto& entry : info) {
        components[sets.find(entry.first)].push_back(entry.first);
    }

    std::vector<Track> tracks;
    std::size_t rejected = 0;
    for (auto& component : components) {
        std::vector<NodeInfo> nodes;
        nodes.reserve(component.second.size());
        for (const std::uint32_t node : component.second) {
            nodes.push_back(info.at(node));
        }
        // Sort by camera; a repeated camera id within the component means two
        // keypoints of one image landed in the same track, which is physically
        // impossible, so the whole component is dropped.
        std::sort(nodes.begin(), nodes.end(), [](const NodeInfo& x, const NodeInfo& y) {
            if (x.camera_id != y.camera_id) return x.camera_id < y.camera_id;
            return x.keypoint_index < y.keypoint_index;
        });
        bool inconsistent = false;
        for (std::size_t i = 1; i < nodes.size(); ++i) {
            if (nodes[i].camera_id == nodes[i - 1].camera_id) {
                inconsistent = true;
                break;
            }
        }
        if (inconsistent) {
            ++rejected;
            continue;
        }
        if (nodes.size() < 2) continue;  // cannot occur from matches, but be safe

        Track track;
        track.observations.reserve(nodes.size());
        for (const NodeInfo& node : nodes) {
            track.observations.push_back(Observation{node.camera_id, node.keypoint_index});
        }
        tracks.push_back(std::move(track));
    }

    // Deterministic order: by first observation (camera, then keypoint index).
    // Each keypoint belongs to exactly one component, so first observations are
    // unique and this is a total order.
    std::sort(tracks.begin(), tracks.end(), [](const Track& x, const Track& y) {
        const Observation& fx = x.observations.front();
        const Observation& fy = y.observations.front();
        if (fx.camera_id != fy.camera_id) return fx.camera_id < fy.camera_id;
        return fx.keypoint_index < fy.keypoint_index;
    });

    if (stats != nullptr) {
        stats->tracks = tracks.size();
        stats->rejected_inconsistent = rejected;
    }
    return tracks;
}

std::vector<Track> limit_tracks(const std::vector<Track>& tracks, std::size_t per_camera_limit) {
    if (per_camera_limit == 0) return tracks;

    // For each camera, collect the (index of tracks it observes). Then rank and
    // keep the top per_camera_limit for that camera; a track survives if any
    // camera selects it.
    std::map<std::int32_t, std::vector<std::size_t>> per_camera;
    for (std::size_t i = 0; i < tracks.size(); ++i) {
        for (const Observation& observation : tracks[i].observations) {
            per_camera[observation.camera_id].push_back(i);
        }
    }

    std::vector<char> keep(tracks.size(), 0);
    for (auto& entry : per_camera) {
        std::vector<std::size_t>& indices = entry.second;
        std::sort(indices.begin(), indices.end(), [&tracks](std::size_t x, std::size_t y) {
            if (tracks[x].observations.size() != tracks[y].observations.size()) {
                return tracks[x].observations.size() > tracks[y].observations.size();
            }
            return x < y;
        });
        const std::size_t take = std::min(indices.size(), per_camera_limit);
        for (std::size_t k = 0; k < take; ++k) keep[indices[k]] = 1;
    }

    std::vector<Track> kept;
    for (std::size_t i = 0; i < tracks.size(); ++i) {
        if (keep[i] != 0) kept.push_back(tracks[i]);
    }
    return kept;
}

}  // namespace alphashape::match
