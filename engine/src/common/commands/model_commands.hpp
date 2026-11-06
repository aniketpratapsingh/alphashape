#pragma once

// Model commands: one factory per document mutation (M11 §4.7).
//
// WP-11.5a declares the whole surface and implements only the base layer
// (AbstractCommand, StagedMutation, CommandTransaction, UndoStack). The
// factories below are implemented by WP-11.5b (doc/chunk, markers, references,
// groups) and WP-11.5c (assets). Each is named "<Verb> <object>" so the
// user-facing label is "Undo <name>".
//
// Every factory follows the same contract:
//   * the returned command validates its target in apply() and throws
//     ValueError on a missing id or an illegal edit;
//   * a failed apply leaves the document untouched (StagedMutation);
//   * apply/revert/apply yields the state of the first apply.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Core>

#include "common/commands/command.hpp"
#include "common/core/chunk_geometry.hpp"
#include "common/core/document.hpp"
#include "common/core/marker.hpp"
#include "common/core/scalebar.hpp"

namespace alphashape {

/// Which asset container a generic asset command addresses.
enum class AssetKind {
    PointCloud,
    Model,
    DepthMaps,
    Elevation,
    Orthomosaic,
    Panorama,
    TiledModel,
};

/// Which grouping container a group command addresses.
enum class GroupKind { Marker, PointCloud, Model, Camera };

// --- Chunk and workspace (WP-11.5b) ----------------------------------------

std::unique_ptr<AbstractCommand> make_add_chunk_command(
    Workspace& workspace, std::string label, std::int32_t insert_index);
std::unique_ptr<AbstractCommand> make_remove_chunk_command(std::int32_t chunk_id);
std::unique_ptr<AbstractCommand> make_duplicate_chunk_command(
    std::int32_t chunk_id, std::string new_label);
std::unique_ptr<AbstractCommand> make_reorder_chunks_command(
    std::vector<std::int32_t> new_order);
std::unique_ptr<AbstractCommand> make_set_chunk_transform_command(
    std::int32_t chunk_id, Transform transform, Region region);

/// Set the chunk CRS. §4.7 declares this over `crs::CoordinateSystem`, which
/// does not exist until M12 WP-12.11; this takes the existing `CRS` descriptor
/// so the undo wiring ships now and M12 only widens the argument type.
std::unique_ptr<AbstractCommand> make_set_chunk_crs_command(std::int32_t chunk_id,
                                                            CRS crs);

/// Which per-object CRS slot a command addresses (M12 §4.4).
enum class CrsSlot { Camera, Marker, World };

/// Set or clear (nullopt = follow the chunk CRS) one per-object CRS slot.
/// Undo restores the previous slot exactly, absence included.
std::unique_ptr<AbstractCommand> make_set_chunk_crs_slot_command(
    std::int32_t chunk_id, CrsSlot slot, std::optional<CRS> value);

/// Replace a chunk's reference settings (WP-13.1). Undoable: revert restores the
/// previous settings. Every accuracy must be finite and strictly positive;
/// `use_gcp` is unconstrained. Throws ValueError on a bad accuracy or an
/// unknown chunk.
std::unique_ptr<AbstractCommand> make_set_reference_settings_command(
    std::int32_t chunk_id, ReferenceSettings settings);

/// Install an already-computed geometry state on a chunk (camera poses, tie
/// points, marker positions, region, transform, meta and asset payloads).
/// AlignChunks (§4.8) is the caller: it estimates the similarity, builds the
/// post-state, and hands it here so the whole alignment is one undoable step.
/// The command captures its own pre-state, so apply -> revert -> apply
/// reproduces exactly the first apply.
std::unique_ptr<AbstractCommand> make_align_chunks_command(std::int32_t chunk_id,
                                                           ChunkGeometry after);

/// The pose/tie-point part of a chunk that AlignFromReference and
/// TriangulateTiePoints rewrite. Unlike ChunkGeometry, the track list itself
/// may change (observations are dropped), and camera enabled flags are part of
/// the state, so the tasks use this narrower snapshot.
struct ChunkPoseState {
    std::vector<Camera> cameras;
    std::vector<Track> tracks;
    Transform transform = Transform::Identity();
    Region region;
    std::vector<std::pair<std::string, std::string>> meta;
    CRS crs;
    std::optional<CRS> camera_crs;
    std::optional<CRS> world_crs;
};

ChunkPoseState capture_chunk_pose_state(const Chunk& chunk);

/// Install a computed pose/tie-point state in one undoable step. The command
/// captures its own pre-state, so apply -> revert restores the chunk exactly
/// and apply -> revert -> apply reproduces the first apply.
std::unique_ptr<AbstractCommand> make_set_chunk_pose_state_command(
    std::int32_t chunk_id, ChunkPoseState after, std::string label);

/// Set one chunk-meta property, restoring the previous value (or removing the
/// key when it was absent) on revert. Used for records that live beside a
/// geometry edit, e.g. AlignChunks writing the applied similarity into the
/// reference chunk.
std::unique_ptr<AbstractCommand> make_set_chunk_meta_command(std::int32_t chunk_id,
                                                             std::string key,
                                                             std::string value);

// --- Markers (WP-11.5b) ----------------------------------------------------

std::unique_ptr<AbstractCommand> make_add_marker_command(
    std::int32_t chunk_id, Marker marker);
std::unique_ptr<AbstractCommand> make_remove_marker_command(
    std::int32_t chunk_id, std::int32_t marker_id);
/// Rename one marker. Undo restores the previous label verbatim. An empty
/// label is rejected, and so is one another marker in the same chunk carries.
std::unique_ptr<AbstractCommand> make_set_marker_label_command(
    std::int32_t chunk_id, std::int32_t marker_id, std::string label);
std::unique_ptr<AbstractCommand> make_set_marker_reference_command(
    std::int32_t chunk_id, std::int32_t marker_id,
    std::optional<Eigen::Vector3d> reference, Eigen::Vector3d accuracy);
/// Set a marker's enabled flag (`Marker::valid`). Undo restores it.
std::unique_ptr<AbstractCommand> make_set_marker_enabled_command(
    std::int32_t chunk_id, std::int32_t marker_id, bool enabled);
/// Set a marker's pinned flag (interactively placed). Undo restores it.
std::unique_ptr<AbstractCommand> make_set_marker_pinned_command(
    std::int32_t chunk_id, std::int32_t marker_id, bool pinned);
/// Set a marker's reference role (control point / check point). Undo restores
/// the previous role. Check points are excluded from the adjustment.
std::unique_ptr<AbstractCommand> make_set_marker_role_command(
    std::int32_t chunk_id, std::int32_t marker_id, MarkerReferenceRole role);
std::unique_ptr<AbstractCommand> make_move_marker_projection_command(
    std::int32_t chunk_id, std::int32_t marker_id, std::int32_t camera_id,
    Eigen::Vector2d pixel);
std::unique_ptr<AbstractCommand> make_remove_marker_projection_command(
    std::int32_t chunk_id, std::int32_t marker_id, std::int32_t camera_id);

// --- Marker placement (WP-13.4) --------------------------------------------

/// Create a marker. A nullopt `label` generates the lowest free "point N"
/// (N >= 1); an explicit label that another marker already carries is a
/// ValueError. `position`, when given, is the chunk-local marker position.
std::unique_ptr<AbstractCommand> make_add_marker_command(
    std::int32_t chunk_id, std::optional<std::string> label,
    std::optional<Eigen::Vector3d> position);

/// A marker reference for placement edits. The alternatives are resolved
/// independently: an id names a marker by id only, a string names one by label
/// only. Keeping them apart means a marker labelled "5" can never capture an
/// edit addressed to marker id 5 (and vice versa).
using MarkerRef = std::variant<std::int32_t, std::string>;

/// Set or replace one projection of the referenced marker in `camera_id`.
///
/// With an id the marker must already exist; with a label a missing marker is
/// created with that label. A newly created marker is pinned when `pinned` and
/// receives `meta` (target provenance), so a later refinement can tell the
/// target kind.
std::unique_ptr<AbstractCommand> make_add_marker_with_projection_command(
    std::int32_t chunk_id, MarkerRef marker, std::int32_t camera_id,
    Eigen::Vector2d uv, bool pinned = true, const Meta& meta = Meta{});

/// Remove one projection. When this was the marker's last projection and
/// `delete_marker` is set, the marker itself is removed; otherwise it stays
/// (with no projections).
std::unique_ptr<AbstractCommand> make_remove_marker_projection_command(
    std::int32_t chunk_id, std::int32_t marker_id, std::int32_t camera_id,
    bool delete_marker);

/// Set (or clear, with nullopt) the marker's chunk-local position.
std::unique_ptr<AbstractCommand> make_set_marker_position_command(
    std::int32_t chunk_id, std::int32_t marker_id,
    std::optional<Eigen::Vector3d> position);

// --- Scale bars (WP-11.5b) -------------------------------------------------

std::unique_ptr<AbstractCommand> make_add_scalebar_command(
    std::int32_t chunk_id, Scalebar scalebar);
std::unique_ptr<AbstractCommand> make_remove_scalebar_command(
    std::int32_t chunk_id, std::int32_t scalebar_id);
std::unique_ptr<AbstractCommand> make_set_scalebar_reference_command(
    std::int32_t chunk_id, std::int32_t scalebar_id, double length_m,
    double accuracy_m);

// --- Groups (WP-11.5b) -----------------------------------------------------

std::unique_ptr<AbstractCommand> make_add_group_command(
    GroupKind kind, std::int32_t chunk_id, std::string label);
std::unique_ptr<AbstractCommand> make_remove_group_command(
    GroupKind kind, std::int32_t chunk_id, std::int32_t group_id);
std::unique_ptr<AbstractCommand> make_move_to_group_command(
    GroupKind kind, std::int32_t chunk_id, std::vector<std::int32_t> ids,
    std::int32_t group_id);

// --- Assets (WP-11.5c) -----------------------------------------------------

/// Add an empty slot of `kind` with `label` (empty -> the next default name,
/// e.g. "Elevation 2"). The slot is made active; the caller fills its payload
/// through the chunk's accessors right after execute, which undo/redo preserve
/// because the command and the document share the same slot object.
std::unique_ptr<AbstractCommand> make_add_asset_command(
    AssetKind kind, std::int32_t chunk_id, std::string label);

/// Typed add overloads carrying the payload a producing task returned (§4.7
/// "a task does not push itself; the caller wraps its result"). The kind is
/// implied by the payload type. The payload is moved in, never copied.
std::unique_ptr<AbstractCommand> make_add_asset_command(
    std::int32_t chunk_id, PointCloud cloud, std::string label);
std::unique_ptr<AbstractCommand> make_add_asset_command(
    std::int32_t chunk_id, ModelAsset model, std::string label);
std::unique_ptr<AbstractCommand> make_add_asset_command(
    std::int32_t chunk_id, std::vector<DepthMap> maps, std::string label);
std::unique_ptr<AbstractCommand> make_add_asset_command(
    std::int32_t chunk_id, Elevation elevation, std::string label);
std::unique_ptr<AbstractCommand> make_add_asset_command(
    std::int32_t chunk_id, Orthomosaic orthomosaic, std::string label);
std::unique_ptr<AbstractCommand> make_add_asset_command(
    std::int32_t chunk_id, TiledModel model, std::string label);
std::unique_ptr<AbstractCommand> make_add_asset_command(
    std::int32_t chunk_id, Image panorama, std::string label);

std::unique_ptr<AbstractCommand> make_remove_asset_command(
    AssetKind kind, std::int32_t chunk_id, std::int32_t asset_id);
std::unique_ptr<AbstractCommand> make_set_active_asset_command(
    AssetKind kind, std::int32_t chunk_id, std::int32_t asset_id);
std::unique_ptr<AbstractCommand> make_rename_asset_command(
    AssetKind kind, std::int32_t chunk_id, std::int32_t asset_id,
    std::string new_label);

// --- Cameras (WP-11.5b) ----------------------------------------------------

std::unique_ptr<AbstractCommand> make_set_camera_enabled_command(
    std::int32_t chunk_id, std::vector<std::int32_t> camera_ids, bool enabled);
/// Set one camera's reference flag (`Camera::reference.enabled`), the flag
/// OptimizeCameras gates its GNSS prior on. Undo restores it.
std::unique_ptr<AbstractCommand> make_set_camera_reference_enabled_command(
    std::int32_t chunk_id, std::int32_t camera_id, bool enabled);
/// Set one camera's GNSS reference (location and per-axis accuracy). Undo
/// restores both. The accuracy must be finite and positive, and a location
/// must be finite.
std::unique_ptr<AbstractCommand> make_set_camera_reference_command(
    std::int32_t chunk_id, std::int32_t camera_id,
    std::optional<Eigen::Vector3d> location, Eigen::Vector3d accuracy);
/// Set (or clear, with nullopt) one camera's per-axis attitude accuracy in
/// degrees, the sigma trio the adjustment weights that camera's rotation
/// prior by. Undo restores the previous value, absence included. Every set
/// component must be finite and non-negative.
std::unique_ptr<AbstractCommand>
make_set_camera_reference_rotation_accuracy_command(
    std::int32_t chunk_id, std::int32_t camera_id,
    std::optional<Eigen::Vector3d> accuracy_deg);
std::unique_ptr<AbstractCommand> make_set_camera_selected_command(
    std::int32_t chunk_id, std::vector<std::int32_t> camera_ids, bool selected);

// --- Sensors (WP-13.16) ----------------------------------------------------

/// Replace one sensor's calibration; undo restores the previous calibration,
/// including its covariance. The sensor's type, width and height follow the
/// calibration, so the two never disagree. Throws ValueError for an unknown
/// sensor, a non-finite/non-positive calibration, or one that breaks the
/// sensor's master link (a slave must match its master's type and size).
std::unique_ptr<AbstractCommand> make_set_sensor_calibration_command(
    std::int32_t chunk_id, std::int32_t sensor_index, Calibration calibration);

/// Point one sensor at a master (`master_index >= 0`) or detach it (-1).
/// Throws ValueError when the link is invalid under the same rules as doc.xml
/// loading: in range, not itself, the master is not a slave, and the image
/// sizes match.
std::unique_ptr<AbstractCommand> make_set_sensor_master_command(
    std::int32_t chunk_id, std::int32_t sensor_index, std::int32_t master_index);

/// Replace one sensor's film/fiducial configuration (WP-13.18a). Undo restores
/// the previous settings verbatim. Throws ValueError for an unknown sensor, an
/// empty or duplicate fiducial label, or a non-finite film coordinate. Camera
/// fiducial measurements are not edited; the caller re-detects after a layout
/// change.
std::unique_ptr<AbstractCommand> make_set_sensor_film_command(
    std::int32_t chunk_id, std::int32_t sensor_index, FilmSettings film);

/// Replace one sensor's GNSS antenna mount (WP-13.19). Undo restores the
/// previous offset and boresight. Throws ValueError for an unknown sensor or a
/// non-finite offset; a boresight, when present, must be finite too.
std::unique_ptr<AbstractCommand> make_set_sensor_antenna_command(
    std::int32_t chunk_id, std::int32_t sensor_index, Antenna antenna);

/// Replace one sensor's "held fixed during adjustment" coefficient names.
/// Every name must be a calibration parameter (kCalibrationParamNames) and no
/// name may repeat. Undo restores the previous list.
std::unique_ptr<AbstractCommand> make_set_sensor_fixed_params_command(
    std::int32_t chunk_id, std::int32_t sensor_index,
    std::vector<std::string> fixed_params);

/// Replace one sensor's physical geometry (pixel pitch and focal length as
/// recorded from EXIF/spec). Both must be finite and non-negative; 0 means
/// unknown. Undo restores the previous values.
std::unique_ptr<AbstractCommand> make_set_sensor_physical_command(
    std::int32_t chunk_id, std::int32_t sensor_index, double pixel_size_um,
    double focal_length_mm);

// --- Cross-document (WP-11.5b) ---------------------------------------------

std::unique_ptr<AbstractCommand> make_append_document_command(
    std::shared_ptr<Document> source, std::string source_path);

}  // namespace alphashape
