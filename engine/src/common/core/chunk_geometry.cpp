#include "common/core/chunk_geometry.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"

namespace alphashape {

ChunkGeometry capture_geometry(const Chunk& chunk) {
    ChunkGeometry geometry;
    geometry.transform = chunk.transform();
    geometry.region = chunk.region();
    geometry.crs = chunk.crs();
    geometry.meta = chunk.meta().items;

    geometry.camera_poses.reserve(chunk.cameras().size());
    for (const Camera& camera : chunk.cameras()) {
        geometry.camera_poses.push_back(camera.pose);
    }

    geometry.track_points.reserve(chunk.tie_points().tracks.size());
    for (const Track& track : chunk.tie_points().tracks) {
        geometry.track_points.push_back(track.point);
    }

    geometry.marker_positions.reserve(chunk.markers().size());
    for (const Marker& marker : chunk.markers()) {
        geometry.marker_positions.push_back(marker.position);
    }

    geometry.shape_vertices.reserve(chunk.shapes().features.size());
    for (const Shape& shape : chunk.shapes().features) {
        geometry.shape_vertices.push_back(shape.vertices);
    }

    geometry.trajectory_positions.reserve(chunk.trajectories().size());
    for (const Trajectory& trajectory : chunk.trajectories()) {
        std::vector<Eigen::Vector3d> positions;
        positions.reserve(trajectory.points.size());
        for (const TrajectoryPoint& point : trajectory.points) {
            positions.push_back(point.position);
        }
        geometry.trajectory_positions.push_back(std::move(positions));
    }

    geometry.point_clouds.reserve(chunk.point_clouds().size());
    for (const PointCloudAsset& slot : chunk.point_clouds()) {
        geometry.point_clouds.push_back(slot.data);
    }
    geometry.models.reserve(chunk.models().size());
    for (const ModelAsset& slot : chunk.models()) {
        geometry.models.push_back(slot.data);
    }
    return geometry;
}

namespace {

/// Clone-on-install: a payload shared with another chunk (or with an undo
/// record) must never be mutated in place by a later geometry edit.
template <class T>
std::shared_ptr<T> own_copy(const std::shared_ptr<T>& payload) {
    return payload ? std::make_shared<T>(*payload) : payload;
}

void check_size(std::size_t expected, std::size_t actual, const char* what) {
    if (expected != actual) {
        throw ValueError(std::string("chunk geometry: ") + what + " count " +
                         std::to_string(actual) + " does not match the snapshot's " +
                         std::to_string(expected));
    }
}

}  // namespace

void apply_geometry(Chunk& chunk, const ChunkGeometry& geometry) {
    // Validate every dimension before touching the chunk, so a mismatched
    // snapshot can never leave it half rewritten.
    check_size(geometry.camera_poses.size(), chunk.cameras().size(), "camera");
    check_size(geometry.track_points.size(), chunk.tie_points().tracks.size(), "track");
    check_size(geometry.marker_positions.size(), chunk.markers().size(), "marker");
    check_size(geometry.point_clouds.size(), chunk.point_clouds().size(), "point cloud");
    check_size(geometry.models.size(), chunk.models().size(), "model");
    check_size(geometry.shape_vertices.size(), chunk.shapes().features.size(), "shape");
    check_size(geometry.trajectory_positions.size(), chunk.trajectories().size(),
               "trajectory");
    for (std::size_t i = 0; i < geometry.trajectory_positions.size(); ++i) {
        check_size(geometry.trajectory_positions[i].size(),
                   chunk.trajectories()[i].points.size(), "trajectory point");
    }

    chunk.transform() = geometry.transform;
    chunk.region() = geometry.region;
    chunk.crs() = geometry.crs;
    chunk.meta().items = geometry.meta;

    for (std::size_t i = 0; i < geometry.camera_poses.size(); ++i) {
        chunk.cameras()[i].pose = geometry.camera_poses[i];
    }
    for (std::size_t i = 0; i < geometry.track_points.size(); ++i) {
        chunk.tie_points().tracks[i].point = geometry.track_points[i];
    }
    for (std::size_t i = 0; i < geometry.marker_positions.size(); ++i) {
        chunk.markers()[i].position = geometry.marker_positions[i];
    }
    for (std::size_t i = 0; i < geometry.shape_vertices.size(); ++i) {
        chunk.shapes().features[i].vertices = geometry.shape_vertices[i];
    }
    for (std::size_t i = 0; i < geometry.trajectory_positions.size(); ++i) {
        std::vector<TrajectoryPoint>& points = chunk.trajectories()[i].points;
        for (std::size_t j = 0; j < points.size(); ++j) {
            points[j].position = geometry.trajectory_positions[i][j];
        }
    }
    for (std::size_t i = 0; i < geometry.point_clouds.size(); ++i) {
        chunk.point_clouds()[i].data = own_copy(geometry.point_clouds[i]);
    }
    for (std::size_t i = 0; i < geometry.models.size(); ++i) {
        chunk.models()[i].data = own_copy(geometry.models[i]);
    }
}

std::size_t payload_bytes(const ChunkGeometry& geometry) {
    std::size_t bytes = 0;
    for (const auto& cloud : geometry.point_clouds) {
        if (!cloud) {
            continue;
        }
        bytes += cloud->points.size() * sizeof(Eigen::Vector3d);
        bytes += cloud->normals.size() * sizeof(Eigen::Vector3f);
        bytes += cloud->colours.size() * sizeof(std::uint8_t);
        bytes += cloud->confidence.size() * sizeof(float);
    }
    for (const auto& model : geometry.models) {
        if (!model) {
            continue;
        }
        bytes += model->vertices.size() * sizeof(Eigen::Vector3d);
        bytes += model->faces.size() * sizeof(Eigen::Vector3i);
        bytes += model->uvs.size() * sizeof(Eigen::Vector2f);
        bytes += model->uv_faces.size() * sizeof(Eigen::Vector3i);
        bytes += model->colours.size() * sizeof(std::uint8_t);
        bytes += model->confidence.size() * sizeof(float);
    }
    return bytes;
}

}  // namespace alphashape
