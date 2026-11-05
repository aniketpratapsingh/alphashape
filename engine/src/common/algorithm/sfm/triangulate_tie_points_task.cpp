#include "common/algorithm/sfm/triangulate_tie_points_task.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/feature/detect_features_task.hpp"  // keypoint_to_pixel
#include "common/algorithm/match/tracks.hpp"
#include "common/algorithm/sfm/absolute_pose.hpp"
#include "common/algorithm/sfm/triangulation.hpp"
#include "common/commands/command.hpp"
#include "common/commands/model_commands.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;

/// Set (or replace) one key in a captured meta list, preserving key order.
void set_meta(std::vector<std::pair<std::string, std::string>>& items,
              const std::string& key, const std::string& value) {
    for (auto& item : items) {
        if (item.first == key) {
            item.second = value;
            return;
        }
    }
    items.emplace_back(key, value);
}

/// Everything triangulation needs about one aligned camera.
struct View {
    std::int32_t camera_id = -1;
    const KeypointSet* keypoints = nullptr;
    const Calibration* calibration = nullptr;
    sfm::AbsolutePose pose;
};

}  // namespace

void TriangulateTiePointsTask::declare_params(Params& params) const {
    params.declare({"max_reprojection_error_px", ParamType::Float, 4.0,
                    "Drop an observation whose reprojection error exceeds this."});
    params.declare({"min_angle_deg", ParamType::Float, 1.0,
                    "Minimum pairwise parallax of an accepted tie point."});
    params.declare({"min_views", ParamType::Int, std::int64_t{2},
                    "Minimum observations a tie point must keep."});
}

std::vector<Stage> TriangulateTiePointsTask::stages() const {
    return {Stage{"prepare", 0}, Stage{"triangulate", 0}};
}

void TriangulateTiePointsTask::apply(Document& document, Params& params,
                                     const Progress& progress) {
    progress.update("prepare", 0.0);
    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("TriangulateTiePoints needs an active chunk");
    }
    const double max_error_px = params.get_float("max_reprojection_error_px");
    const double min_angle_deg = params.get_float("min_angle_deg");
    const std::int64_t min_views = params.get_int("min_views");
    if (!std::isfinite(max_error_px) || !(max_error_px > 0.0)) {
        throw ValueError("TriangulateTiePoints: max_reprojection_error_px must be finite "
                         "and positive");
    }
    if (!std::isfinite(min_angle_deg) || min_angle_deg < 0.0) {
        throw ValueError("TriangulateTiePoints: min_angle_deg must be finite and >= 0");
    }
    if (min_views < 2) {
        throw ValueError("TriangulateTiePoints: min_views must be at least 2");
    }
    const std::size_t min_view_count = static_cast<std::size_t>(min_views);

    const TiePoints& tie_points = chunk->tie_points();
    if (tie_points.matches.empty()) {
        throw ValueError("TriangulateTiePoints: no matches \u2014 run MatchPhotos first");
    }
    std::unordered_map<std::int32_t, View> views;
    views.reserve(chunk->cameras().size());
    for (const Camera& camera : chunk->cameras()) {
        const KeypointSet* set = tie_points.find_by_camera(camera.id);
        if (!camera.enabled || set == nullptr) {
            continue;
        }
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk->sensors().size())) {
            continue;
        }
        View view;
        view.camera_id = camera.id;
        view.keypoints = set;
        view.calibration = &chunk->sensors()[static_cast<std::size_t>(camera.sensor_id)]
                                .calibration;
        view.pose.R = camera.pose.block<3, 3>(0, 0);
        view.pose.t = camera.pose.block<3, 1>(0, 3);
        views.emplace(camera.id, view);
    }
    if (views.empty()) {
        throw Error("TriangulateTiePoints: no aligned camera has keypoints");
    }

    std::vector<PairMatches> usable_matches;
    usable_matches.reserve(tie_points.matches.size());
    for (const PairMatches& pair : tie_points.matches) {
        if (views.find(pair.camera_a) != views.end() &&
            views.find(pair.camera_b) != views.end()) {
            usable_matches.push_back(pair);
        }
    }
    progress.update("prepare", 1.0);

    progress.update("triangulate", 0.0);
    match::TrackStats stats;
    const std::vector<Track> tracks = match::build_tracks(usable_matches, &stats);
    const double min_parallax_rad = min_angle_deg * kDegToRad;

    std::size_t triangulated = 0;
    std::size_t rejected_views = 0;
    std::size_t rejected_parallax = 0;
    std::size_t rejected_degenerate = 0;
    std::size_t rejected_reprojection = 0;
    std::size_t observations_dropped = 0;
    std::vector<Track> accepted;
    accepted.reserve(tracks.size());

    // Resolve observations to poses, bearings, calibrations and pixels. False
    // when an observation no longer resolves (unknown or unaligned camera, a
    // keypoint index out of range).
    const auto gather = [&views](const std::vector<Observation>& observations,
                                 std::vector<sfm::AbsolutePose>& poses,
                                 std::vector<Eigen::Vector3d>& bearings,
                                 std::vector<const Calibration*>& calibrations,
                                 std::vector<Eigen::Vector2d>& pixels) {
        poses.clear();
        bearings.clear();
        calibrations.clear();
        pixels.clear();
        for (const Observation& observation : observations) {
            const auto it = views.find(observation.camera_id);
            if (it == views.end()) {
                return false;
            }
            const KeypointSet& set = *it->second.keypoints;
            if (static_cast<std::size_t>(observation.keypoint_index) >=
                set.keypoints.size()) {
                return false;
            }
            const Eigen::Vector2d pixel =
                keypoint_to_pixel(set.keypoints[observation.keypoint_index]);
            poses.push_back(it->second.pose);
            bearings.push_back(it->second.calibration->unproject(pixel));
            calibrations.push_back(it->second.calibration);
            pixels.push_back(pixel);
        }
        return true;
    };

    for (const Track& track : tracks) {
        std::vector<Observation> kept = track.observations;
        if (kept.size() < min_view_count) {
            ++rejected_views;
            continue;
        }
        std::vector<sfm::AbsolutePose> poses;
        std::vector<Eigen::Vector3d> bearings;
        std::vector<const Calibration*> calibrations;
        std::vector<Eigen::Vector2d> pixels;
        if (!gather(kept, poses, bearings, calibrations, pixels)) {
            ++rejected_degenerate;
            continue;
        }
        // First pass without the parallax floor: a nullopt now means the rays
        // do not meet in front of every camera (or are numerically degenerate).
        sfm::TriangulationOptions open_options;
        open_options.min_parallax_rad = 0.0;
        open_options.max_angle_error_rad = 0.0;
        std::optional<Eigen::Vector3d> point =
            sfm::triangulate_multiview(poses, bearings, open_options);
        if (!point.has_value() || !point->allFinite()) {
            ++rejected_degenerate;
            continue;
        }
        if (sfm::max_parallax(poses, *point) < min_parallax_rad) {
            ++rejected_parallax;
            continue;
        }
        // Drop every observation that disagrees with the triangulated point,
        // then re-triangulate from the survivors.
        std::vector<Observation> remaining;
        remaining.reserve(kept.size());
        for (std::size_t i = 0; i < kept.size(); ++i) {
            const Eigen::Vector3d x_cam = poses[i].R * (*point) + poses[i].t;
            const double error = (calibrations[i]->project(x_cam) - pixels[i]).norm();
            if (std::isfinite(error) && error <= max_error_px) {
                remaining.push_back(kept[i]);
            } else {
                ++observations_dropped;
            }
        }
        if (remaining.size() != kept.size()) {
            if (remaining.size() < min_view_count) {
                ++rejected_reprojection;
                continue;
            }
            if (!gather(remaining, poses, bearings, calibrations, pixels)) {
                ++rejected_reprojection;
                continue;
            }
            sfm::TriangulationOptions closed_options;
            closed_options.min_parallax_rad = min_parallax_rad;
            closed_options.max_angle_error_rad = 0.0;
            point = sfm::triangulate_multiview(poses, bearings, closed_options);
            if (!point.has_value() || !point->allFinite()) {
                ++rejected_reprojection;
                continue;
            }
            kept = std::move(remaining);
        }
        Track out;
        out.observations = std::move(kept);
        out.point = *point;
        out.covariance.reset();
        accepted.push_back(std::move(out));
        ++triangulated;
    }
    progress.update("triangulate", 1.0);

    // -- one undoable commit -------------------------------------------------
    ChunkPoseState state = capture_chunk_pose_state(*chunk);
    const std::size_t track_count = tracks.size();
    state.tracks = std::move(accepted);
    set_meta(state.meta, "triangulate_tie_points.tracks", std::to_string(track_count));
    set_meta(state.meta, "triangulate_tie_points.triangulated",
             std::to_string(triangulated));
    set_meta(state.meta, "triangulate_tie_points.observations_dropped",
             std::to_string(observations_dropped));
    set_meta(state.meta, "triangulate_tie_points.rejected_views",
             std::to_string(rejected_views));
    set_meta(state.meta, "triangulate_tie_points.rejected_parallax",
             std::to_string(rejected_parallax));
    set_meta(state.meta, "triangulate_tie_points.rejected_degenerate",
             std::to_string(rejected_degenerate));
    set_meta(state.meta, "triangulate_tie_points.rejected_reprojection",
             std::to_string(rejected_reprojection));

    CommandTransaction transaction("Triangulate Tie Points");
    transaction.add(make_set_chunk_pose_state_command(
        chunk->id(), std::move(state), "Triangulate Tie Points"));
    transaction.apply(document);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "TriangulateTiePoints: " << triangulated << "/"
                            << track_count << " tracks, " << observations_dropped
                            << " observations dropped (" << rejected_views
                            << " too few views, " << rejected_parallax
                            << " parallax, " << rejected_degenerate << " degenerate, "
                            << rejected_reprojection << " reprojection)";
    }
}

}  // namespace alphashape

