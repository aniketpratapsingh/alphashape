// BuildPointCloud: fuse the chunk's depth maps into one dense point cloud.
//
// The adapter between the chunk's data model and stereo::fuse_depth_maps. Every
// depth map whose camera still exists and is enabled is back-projected into
// chunk-local space (using the map's reduced calibration and the camera pose) and
// merged, voxel-decimated, into a single cloud. Full-resolution images are read
// only when point_colors is set; they colour each fused point.
//
// The cloud is built locally, then installed in a NEW point-cloud asset
// (appended and made active), so an error anywhere leaves the chunk untouched and
// an earlier cloud is never overwritten.

#include "common/algorithm/stereo/build_point_cloud_task.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/stereo/fusion.hpp"
#include "common/core/assets.hpp"
#include "common/core/frame.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

/// Resolve a frame's recorded path against the document's directory (the same
/// rule AddPhotos/DetectFeatures use).
std::filesystem::path resolve_frame_path(const Document& document, const std::string& path) {
    const std::filesystem::path recorded(path);
    if (recorded.is_absolute() || document.path().empty()) {
        return recorded;
    }
    return std::filesystem::path(document.path()).parent_path() / recorded;
}

Eigen::Matrix3d pose_R(const Transform& pose) { return pose.block<3, 3>(0, 0); }
Eigen::Vector3d pose_t(const Transform& pose) { return pose.block<3, 1>(0, 3); }

}  // namespace

void BuildPointCloudTask::declare_params(Params& params) const {
    // Defaults follow Metashape's public Python API.
    params.declare({"point_colors", ParamType::Bool, true,
                    "Colour the fused points from the source images."});
    params.declare({"point_confidence", ParamType::Bool, true,
                    "Store a per-point confidence derived from the depth-map support."});
    params.declare({"min_confidence", ParamType::Float, 0.0,
                    "Keep only depth samples whose confidence is at least this value."});
    params.declare({"label", ParamType::String, std::string{},
                    "Label for the new point-cloud asset. Empty selects \"Point Cloud <n>\". "
                    "Each run creates a new asset."});
}

std::vector<Stage> BuildPointCloudTask::stages() const {
    return {Stage{"validate", 0}, Stage{"fuse", 0}, Stage{"finalize", 0}};
}

void BuildPointCloudTask::apply(Document& document, Params& params, const Progress& progress) {
    // -- validate every parameter before touching the chunk ------------------
    const bool point_colors = params.get_bool("point_colors");
    const bool point_confidence = params.get_bool("point_confidence");
    const double min_confidence = params.get_float("min_confidence");

    progress.update("validate", 0.0);
    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("BuildPointCloud needs an active chunk");
    }
    // Read through a CONST view: the non-const legacy accessors lazily create an
    // empty asset, which would leave a phantom cloud behind on a failed run.
    const Chunk& source = *chunk;
    if (source.depth_maps().empty()) {
        throw ValueError("No depth maps");
    }

    // -- gather fuse inputs; skip maps whose camera is gone or disabled ------
    std::unordered_map<std::int32_t, const Frame*> frame_of;
    for (const auto& frame : chunk->frames()) {
        if (frame->camera_id() >= 0) {
            frame_of.emplace(frame->camera_id(), frame.get());
        }
    }
    std::unordered_map<std::int32_t, const Camera*> camera_by_id;
    for (const Camera& camera : chunk->cameras()) {
        camera_by_id[camera.id] = &camera;
    }

    // Full-resolution images, decoded once per camera and kept alive for the
    // duration of the fusion (FusionInput holds a pointer).
    std::unordered_map<std::int32_t, Image> images;

    std::vector<stereo::FusionInput> inputs;
    inputs.reserve(source.depth_maps().size());
    for (const DepthMap& map : source.depth_maps()) {
        const auto camera_it = camera_by_id.find(map.camera_id);
        if (camera_it == camera_by_id.end()) {
            continue;  // camera no longer exists
        }
        const Camera& camera = *camera_it->second;
        if (!camera.enabled) {
            continue;  // not aligned: its map is stale
        }
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk->sensors().size())) {
            throw ValueError("BuildPointCloud: camera " + std::to_string(map.camera_id) +
                             " has no sensor");
        }

        stereo::FusionInput input;
        input.map = &map;
        input.calibration = scaled_calibration(
            chunk->sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration,
            map.downscale);
        input.R = pose_R(camera.pose);
        input.t = pose_t(camera.pose);

        if (point_colors) {
            const auto frame_it = frame_of.find(map.camera_id);
            if (frame_it == frame_of.end()) {
                throw ValueError("BuildPointCloud: camera " + std::to_string(map.camera_id) +
                                 " has no linked frame to colour from");
            }
            auto inserted = images.emplace(map.camera_id, Image{});
            if (inserted.second) {
                inserted.first->second =
                    read_image(resolve_frame_path(document, frame_it->second->path()).string());
            }
            input.image = &inserted.first->second;
        }
        inputs.push_back(std::move(input));
    }
    if (inputs.empty()) {
        throw ValueError("No depth maps");
    }
    progress.update("validate", 1.0);

    // -- fuse ----------------------------------------------------------------
    progress.update("fuse", 0.0);
    stereo::FusionOptions options;
    options.colours = point_colors;
    options.min_confidence = static_cast<float>(min_confidence);
    PointCloud cloud = stereo::fuse_depth_maps(inputs, options);
    progress.update("fuse", 1.0);

    // -- commit atomically ---------------------------------------------------
    progress.update("finalize", 0.0);
    if (!point_confidence) {
        cloud.confidence.clear();
    }
    const std::size_t point_count = cloud.points.size();
    // A new point-cloud asset per run: the previous cloud is kept.
    PointCloudAsset& asset = chunk->add_point_cloud(params.get_string("label"));
    *asset.data = std::move(cloud);
    progress.update("finalize", 1.0);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "BuildPointCloud: " << point_count << " points from "
                            << inputs.size() << " depth maps"
                            << (point_colors ? " (coloured)" : "")
                            << (point_confidence ? "" : " (no confidence)") << " -> asset "
                            << asset.id << " '" << asset.label << "'";
    }
}

}  // namespace alphashape
