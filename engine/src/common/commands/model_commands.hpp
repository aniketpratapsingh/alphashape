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

/// Install an already-computed geometry state on a chunk (camera poses, tie
/// points, marker positions, region, transform, meta and asset payloads).
/// AlignChunks (§4.8) is the caller: it estimates the similarity, builds the
/// post-state, and hands it here so the whole alignment is one undoable step.
/// The command captures its own pre-state, so apply -> revert -> apply
/// reproduces exactly the first apply.
std::unique_ptr<AbstractCommand> make_align_chunks_command(std::int32_t chunk_id,
                                                           ChunkGeometry after);

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
std::unique_ptr<AbstractCommand> make_set_marker_reference_command(
    std::int32_t chunk_id, std::int32_t marker_id,
    std::optional<Eigen::Vector3d> reference, Eigen::Vector3d accuracy);
std::unique_ptr<AbstractCommand> make_move_marker_projection_command(
    std::int32_t chunk_id, std::int32_t marker_id, std::int32_t camera_id,
    Eigen::Vector2d pixel);
std::unique_ptr<AbstractCommand> make_remove_marker_projection_command(
    std::int32_t chunk_id, std::int32_t marker_id, std::int32_t camera_id);

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
std::unique_ptr<AbstractCommand> make_set_camera_selected_command(
    std::int32_t chunk_id, std::vector<std::int32_t> camera_ids, bool selected);

// --- Cross-document (WP-11.5b) ---------------------------------------------

std::unique_ptr<AbstractCommand> make_append_document_command(
    std::shared_ptr<Document> source, std::string source_path);

}  // namespace alphashape
