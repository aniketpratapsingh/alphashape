#pragma once

// Frame statistics (WP-11.8): a cheap, read-only summary of a chunk's camera
// block for the UI and scripts. "Internal" coordinates are the chunk-local
// frame the camera poses live in; "world" is that frame after the chunk's
// local-to-world transform, and is reported only when the chunk is
// georeferenced.

#include <cstddef>

#include <Eigen/Core>

namespace alphashape {

class Chunk;

struct FrameStats {
    /// Axis-aligned bounding box of the enabled cameras' centres, chunk-local.
    bool has_extent = false;
    Eigen::Vector3d extent_min = Eigen::Vector3d::Zero();
    Eigen::Vector3d extent_max = Eigen::Vector3d::Zero();

    /// The same box after the chunk's transform, when a CRS is set.
    bool has_world_extent = false;
    Eigen::Vector3d world_extent_min = Eigen::Vector3d::Zero();
    Eigen::Vector3d world_extent_max = Eigen::Vector3d::Zero();

    /// Median nearest-neighbour camera-centre distance times two (an estimate
    /// of the capture distance). 0 when fewer than two cameras are enabled.
    double mean_capture_distance = 0.0;

    std::size_t camera_count = 0;
    std::size_t enabled_count = 0;
    /// Cameras with a pose. `Camera::enabled` is the alignment flag, so this
    /// equals `enabled_count` until a separate "used for processing" flag is
    /// introduced.
    std::size_t aligned_count = 0;
    std::size_t frame_count = 0;
    std::size_t group_count = 0;
};

/// Compute the summary. Never mutates the chunk.
FrameStats compute_frame_stats(const Chunk& chunk);

}  // namespace alphashape
