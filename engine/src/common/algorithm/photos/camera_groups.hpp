#pragma once

// Camera grouping (WP-11.8): rebuild a chunk's camera groups from the capture
// station or the source folder, or clear the grouping. The bulk edit is exposed
// as an undoable command so the GUI and Python share one implementation.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/commands/command.hpp"

namespace alphashape {

class Chunk;
struct CameraGroup;

/// How `group_cameras` partitions the cameras.
enum class GroupType {
    Station,  ///< by sensor serial when known, otherwise make + model
    Folder,   ///< by the parent directory of the camera's frame
    None      ///< remove the automatic (station and folder) groups
};

/// Stable text spelling used by the Python surface and diagnostics.
const char* group_type_name(GroupType type);

/// Parse "station", "folder" or "none". @throws ValueError for anything else.
GroupType parse_group_type(const std::string& name);

/// Rebuild `chunk`'s camera groups in place.
///
/// Station and Folder replace only groups of that same type, leaving user
/// groups and the other automatic kind alone; None removes the automatic
/// (station/folder) groups and keeps user/Fixed groups. Membership lists are
/// sorted and unique, and group ids come from `Chunk::allocate_id()`.
///
/// Folder grouping leaves a camera with no frame out of every group and logs
/// how many were skipped.
void group_cameras(Chunk& chunk, GroupType type);

/// Give every folder-type group a label that is unique among the folder
/// groups: the base name, extended with parent path components (shortest
/// trailing suffix first) when two folders share a base name, with a " (n)"
/// suffix as the last resort. Non-folder groups are untouched. AddPhotos calls
/// this after creating folder groups; `group_cameras(Folder)` calls it itself.
void disambiguate_folder_labels(std::vector<CameraGroup>& groups);

/// Undoable wrapper around `group_cameras`. Undo restores the exact previous
/// group list (members, labels, types and folders) and the chunk's id
/// allocator, so a redo allocates the same group ids.
std::unique_ptr<AbstractCommand> make_group_cameras_command(std::int32_t chunk_id,
                                                            GroupType type);

}  // namespace alphashape
