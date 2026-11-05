#include "common/algorithm/reference/align_from_reference_task.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/reference/convert_reference.hpp"
#include "common/algorithm/reference/reference_orientation.hpp"
#include "common/commands/command.hpp"
#include "common/commands/model_commands.hpp"
#include "model/camera/antenna.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape {
namespace {

/// True when the transform maps internal coordinates to ECEF (not identity).
bool has_transform(const Transform& transform) {
    return (transform - Transform::Identity()).cwiseAbs().maxCoeff() > 1e-12;
}

/// Camera ids requested by the `cameras` list. Empty means every camera.
std::optional<std::unordered_set<std::int32_t>> requested_ids(const std::vector<std::string>& ids,
                                                              const Chunk& chunk) {
    if (ids.empty()) {
        return std::nullopt;
    }
    std::unordered_set<std::int32_t> requested;
    requested.reserve(ids.size());
    for (const std::string& text : ids) {
        std::int32_t id = -1;
        try {
            std::size_t used = 0;
            const long value = std::stol(text, &used);
            if (used != text.size() || value < 0 ||
                value > static_cast<long>(INT32_MAX)) {
                throw std::invalid_argument("range");
            }
            id = static_cast<std::int32_t>(value);
        } catch (const std::exception&) {
            throw ValueError("AlignFromReference: bad camera id '" + text + "'");
        }
        bool known = false;
        for (const Camera& camera : chunk.cameras()) {
            if (camera.id == id) {
                known = true;
                break;
            }
        }
        if (!known) {
            throw ValueError("AlignFromReference: unknown camera " + text);
        }
        requested.insert(id);
    }
    return requested;
}

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

}  // namespace

void AlignFromReferenceTask::declare_params(Params& params) const {
    params.declare({"cameras", ParamType::StringList, std::vector<std::string>{},
                    "Camera ids to place; empty places every enabled camera."});
    params.declare({"reset_unreferenced", ParamType::Bool, false,
                    "Reset skipped cameras to the unaligned state instead of keeping "
                    "their current pose."});
}

std::vector<Stage> AlignFromReferenceTask::stages() const {
    return {Stage{"prepare", 0}, Stage{"place", 0}};
}

void AlignFromReferenceTask::apply(Document& document, Params& params,
                                   const Progress& progress) {
    progress.update("prepare", 0.0);
    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("AlignFromReference needs an active chunk");
    }
    // The antenna mount resolves through a sensor's master, so an invalid link
    // is a hard error before any camera is placed.
    validate_sensor_masters(chunk->sensors());
    const std::vector<std::string> id_texts = params.get_string_list("cameras");
    const bool reset_unreferenced = params.get_bool("reset_unreferenced");
    const std::optional<std::unordered_set<std::int32_t>> requested =
        requested_ids(id_texts, *chunk);

    // Cameras the task may place: enabled, explicitly selected (or every one
    // when the list is empty), with an enabled reference. A requested camera
    // whose reference is disabled is reset below when reset_unreferenced is set.
    std::vector<std::size_t> candidates;
    std::vector<std::size_t> disabled_reference;
    Eigen::Vector3d mean_ecef = Eigen::Vector3d::Zero();
    std::size_t located = 0;
    for (std::size_t i = 0; i < chunk->cameras().size(); ++i) {
        const Camera& camera = chunk->cameras()[i];
        const bool requested_here =
            !requested.has_value() || requested->find(camera.id) != requested->end();
        if (requested.has_value() && requested_here && !camera.reference.enabled) {
            disabled_reference.push_back(i);
        }
        if (!camera.enabled || !camera.reference.enabled || !requested_here) {
            continue;
        }
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk->sensors().size())) {
            throw ValueError("AlignFromReference: camera " + std::to_string(camera.id) +
                             " has no sensor");
        }
        candidates.push_back(i);
        if (!camera.reference.location.has_value()) {
            continue;
        }
        const Eigen::Vector3d& loc = *camera.reference.location;
        if (!std::isfinite(loc.x()) || !std::isfinite(loc.y()) || !std::isfinite(loc.z())) {
            continue;
        }
        // References are lon/lat/ellipsoidal height. A projected camera CRS
        // cannot be consumed; refusing here matches update_transform, before
        // any conversion or mutation.
        if (!crs_is_geographic(chunk->camera_crs())) {
            throw ValueError("AlignFromReference: camera reference CRS is not geographic; "
                             "convert the camera references to lon/lat first");
        }
        mean_ecef += geo::geodetic_to_ecef(geo::Geodetic{loc.x(), loc.y(), loc.z()});
        ++located;
    }
    if (candidates.empty()) {
        if (requested.has_value()) {
            throw ValueError("AlignFromReference: no selected camera has an enabled "
                             "reference");
        }
        // With no list and no enabled reference anywhere there is nothing to
        // place. Leaving the chunk untouched is the no-op the task-schema
        // parity harness expects from every task.
        progress.update("prepare", 1.0);
        progress.update("place", 1.0);
        return;
    }
    if (located == 0) {
        throw ValueError("AlignFromReference: no selected camera has a finite reference "
                         "position");
    }
    mean_ecef /= static_cast<double>(located);
    progress.update("prepare", 1.0);

    // -- the internal frame --------------------------------------------------
    progress.update("place", 0.0);
    ChunkPoseState state = capture_chunk_pose_state(*chunk);
    Transform transform = chunk->transform();
    CRS crs = chunk->crs();
    std::optional<CRS> camera_crs = chunk->explicit_camera_crs();
    std::optional<CRS> world_crs = chunk->explicit_world_crs();
    const bool built_frame = !has_transform(transform);
    if (built_frame) {
        // Internal coordinates become metres in the local ENU at the centroid
        // of the measured positions; transform = enu_to_ecef(centroid).
        const geo::Geodetic origin = geo::ecef_to_geodetic(mean_ecef);
        transform = geo::enu_to_ecef(origin);
        CRS geographic;
        geographic.epsg = 4326;
        crs = geographic;
        camera_crs = geographic;
        world_crs = geographic;
    }
    const Transform inverse = transform.inverse();

    const auto reset_camera = [](Camera& camera) {
        camera.pose = Transform::Identity();
        camera.enabled = false;
        camera.location_covariance.setZero();
        camera.rotation_covariance.setZero();
    };
    if (reset_unreferenced) {
        for (const std::size_t index : disabled_reference) {
            reset_camera(state.cameras[index]);
        }
    }

    std::size_t placed = 0;
    std::size_t skipped = 0;
    std::size_t skipped_no_altitude = 0;
    std::vector<bool> is_placed(state.cameras.size(), false);
    for (const std::size_t index : candidates) {
        Camera& camera = state.cameras[index];
        const std::optional<Eigen::Vector3d>& loc = camera.reference.location;
        const bool finite_xy =
            loc.has_value() && std::isfinite(loc->x()) && std::isfinite(loc->y());
        const bool usable_position = finite_xy && std::isfinite(loc->z());
        if (!usable_position) {
            // No position, or a GNSS fix without altitude: keep skipping (a
            // wrong height would shift the whole block), but count the missing
            // altitude separately so it is visible in the run metadata.
            if (finite_xy) {
                ++skipped_no_altitude;
            } else {
                ++skipped;
            }
            if (reset_unreferenced) {
                reset_camera(camera);
            }
            continue;
        }
        // A rotation cannot be derived without an attitude, so a camera whose
        // reference carries none is always skipped.
        const std::optional<reference::ReferenceRotation> measured =
            reference::measured_rotation(*chunk, camera, transform);
        if (!measured.has_value()) {
            ++skipped;
            if (reset_unreferenced) {
                reset_camera(camera);
            }
            continue;
        }
        const Eigen::Vector3d antenna = geo::geodetic_to_ecef(
            geo::Geodetic{loc->x(), loc->y(), loc->z()});
        const Eigen::Matrix3d r_wc = measured->world_to_camera;
        const Eigen::Vector3d lever =
            reference::effective_antenna(*chunk, camera).camera_offset();
        // The measured rotation maps chunk-internal coordinates to the camera,
        // so the lever arm is rotated into the internal frame; a similarity
        // scale converts its metric length. A = C + R_wc^T * lever, so
        // C = A - R_wc^T * lever, both in internal units.
        const double scale = std::cbrt(transform.block<3, 3>(0, 0).determinant());
        const Eigen::Vector3d antenna_internal =
            (inverse * antenna.homogeneous()).head<3>();
        const Eigen::Vector3d centre =
            antenna_internal - r_wc.transpose() * lever / scale;
        camera.pose = Transform::Identity();
        camera.pose.block<3, 3>(0, 0) = r_wc;
        camera.pose.block<3, 1>(0, 3) = -r_wc * centre;
        camera.enabled = true;
        camera.location_covariance.setZero();
        camera.rotation_covariance.setZero();
        is_placed[index] = true;
        ++placed;
    }

    if (placed == 0) {
        throw Error("AlignFromReference: no camera could be placed from its reference (" +
                    std::to_string(skipped) + " skipped, " +
                    std::to_string(skipped_no_altitude) + " without altitude)");
    }
    // A built frame replaces the arbitrary one, so every pose and estimated
    // marker position expressed in the old frame is stale. Cameras the task did
    // not place are disabled rather than left in the old frame; the region is
    // refitted to the placed centres. A reused frame keeps every other pose.
    std::size_t disabled = 0;
    std::vector<std::int32_t> cleared_markers;
    if (built_frame) {
        for (std::size_t i = 0; i < state.cameras.size(); ++i) {
            if (!is_placed[i]) {
                reset_camera(state.cameras[i]);
                ++disabled;
            }
        }
        std::vector<Eigen::Vector3d> centres;
        centres.reserve(placed);
        for (std::size_t i = 0; i < state.cameras.size(); ++i) {
            if (!is_placed[i]) {
                continue;
            }
            const Camera& camera = state.cameras[i];
            centres.push_back(-camera.pose.block<3, 3>(0, 0).transpose() *
                              camera.pose.block<3, 1>(0, 3));
        }
        state.region = region_from_points(centres, state.region);
        for (const Marker& marker : chunk->markers()) {
            if (marker.position.has_value()) {
                cleared_markers.push_back(marker.id);
            }
        }
    }
    const std::size_t cleared_tracks = state.tracks.size();
    state.tracks.clear();  // they belong to the old poses
    state.transform = transform;
    state.crs = crs;
    state.camera_crs = camera_crs;
    state.world_crs = world_crs;
    set_meta(state.meta, "align_from_reference.placed", std::to_string(placed));
    set_meta(state.meta, "align_from_reference.skipped", std::to_string(skipped));
    set_meta(state.meta, "align_from_reference.skipped_no_altitude",
             std::to_string(skipped_no_altitude));
    set_meta(state.meta, "align_from_reference.disabled", std::to_string(disabled));
    set_meta(state.meta, "align_from_reference.tie_points_cleared",
             std::to_string(cleared_tracks));
    set_meta(state.meta, "align_from_reference.markers_cleared",
             std::to_string(cleared_markers.size()));

    // -- one undoable commit -------------------------------------------------
    CommandTransaction transaction("Align From Reference");
    transaction.add(make_set_chunk_pose_state_command(
        chunk->id(), std::move(state), "Align From Reference"));
    for (const std::int32_t marker_id : cleared_markers) {
        transaction.add(make_set_marker_position_command(chunk->id(), marker_id,
                                                         std::nullopt));
    }
    transaction.apply(document);
    progress.update("place", 1.0);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "AlignFromReference: placed " << placed << "/"
                            << candidates.size() << " cameras, " << skipped
                            << " skipped (" << skipped_no_altitude
                            << " without altitude), " << disabled << " disabled, "
                            << cleared_tracks << " tie points and "
                            << cleared_markers.size() << " marker positions cleared, "
                            << "local frame " << (built_frame ? "built" : "reused");
    }
}

}  // namespace alphashape
