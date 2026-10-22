#pragma once

// Trajectories: a timed sequence of camera poses (WP-11.1, M11 §4.1). Used for
// animation / video-frame imports and for georeferencing against a GNSS/IMU
// track. Positions are chunk-local; `timestamp_s` is seconds since the track
// origin.

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace alphashape {

/// One sample of a trajectory. `orientation` is only meaningful when
/// `has_orientation` is true; it is the camera-to-world rotation.
struct TrajectoryPoint {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    double timestamp_s = 0.0;
    bool has_orientation = false;

    bool operator==(const TrajectoryPoint& other) const {
        return position == other.position &&
               orientation.coeffs() == other.orientation.coeffs() &&
               timestamp_s == other.timestamp_s &&
               has_orientation == other.has_orientation;
    }
};

/// A labelled track. Points are stored in increasing timestamp order.
struct Trajectory {
    std::int32_t id = -1;
    std::string label;
    std::vector<TrajectoryPoint> points;

    bool operator==(const Trajectory&) const = default;
};

}  // namespace alphashape
