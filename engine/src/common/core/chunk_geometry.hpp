#pragma once

// ChunkGeometry: the part of a chunk that a similarity rewrite replaces.
//
// AlignChunks (M11 §4.8, WP-11.7) moves a chunk into another chunk's frame:
// camera poses, triangulated tie points, marker positions, the region, the
// chunk transform and every geometry asset follow one similarity. The edit has
// to be undoable through the command layer (§4.7), and core must not depend on
// the algorithms library that owns the similarity estimator -- so the task
// computes the post-state, and the command swaps pre-state and post-state.
//
// This struct is that pre/post state. The small members are copied by value.
// Asset payloads are held by shared_ptr: capture is O(1), a replacement never
// mutates the payload it replaces, and the command's retained memory is exactly
// the payloads an undo brings back (see payload_bytes()).

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "common/core/chunk.hpp"

namespace alphashape {

struct ChunkGeometry {
    Transform transform = Transform::Identity();
    Region region;
    /// The chunk CRS. An alignment installs the reference chunk's frame, so the
    /// CRS travels with the transform.
    CRS crs;

    /// One entry per `chunk.cameras()`, in order.
    std::vector<Transform> camera_poses;
    /// One entry per `chunk.tie_points().tracks`, in order (nullopt = untriangulated).
    std::vector<std::optional<Eigen::Vector3d>> track_points;
    /// One entry per `chunk.markers()`, in order.
    std::vector<std::optional<Eigen::Vector3d>> marker_positions;
    /// Vertices of `chunk.shapes().features`, in feature order.
    std::vector<std::vector<Eigen::Vector3d>> shape_vertices;
    /// Positions of `chunk.trajectories()`, in trajectory order.
    std::vector<std::vector<Eigen::Vector3d>> trajectory_positions;
    /// The whole meta list, in order.
    std::vector<std::pair<std::string, std::string>> meta;

    /// One payload per point-cloud / model slot, in order. The slot ids,
    /// labels, groups and per-slot meta are NOT part of the geometry: an
    /// alignment replaces positions, never assets.
    std::vector<std::shared_ptr<PointCloud>> point_clouds;
    std::vector<std::shared_ptr<Mesh>> models;
};

/// Snapshot the transformable state of `chunk`.
ChunkGeometry capture_geometry(const Chunk& chunk);

/// Overwrite the transformable state of `chunk` with `geometry`. Throws
/// ValueError and leaves the chunk untouched when the snapshot does not match
/// the chunk's structure (a different number of cameras, tracks, markers or
/// slots). Never mutates a payload the snapshot shares with another chunk:
/// every payload is cloned before it is installed.
void apply_geometry(Chunk& chunk, const ChunkGeometry& geometry);

/// Bytes retained by a snapshot's asset payloads (point clouds and models).
/// Used with kUndoAssetSpillBytes to decide whether the edit is undoable.
std::size_t payload_bytes(const ChunkGeometry& geometry);

}  // namespace alphashape
