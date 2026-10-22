#pragma once

// Grouping containers (WP-11.1, M11 §4.3). Markers, point clouds and models are
// grouped by a plain labelled group; cameras use a richer group that records
// how Metashape would have created it (folder import, station, fixed rig).

#include <cstdint>
#include <string>
#include <vector>

namespace alphashape {

namespace group_tags {
struct MarkerTag;
struct PointCloudTag;
struct ModelTag;
}  // namespace group_tags

/// A labelled set of members of one kind. `members` holds the ids of the
/// grouped objects; they are kept ascending and unique.
template <class Tag>
struct Group {
    std::int32_t id = -1;
    std::string label;
    std::vector<std::int32_t> members;

    bool operator==(const Group&) const = default;
};

using MarkerGroup = Group<group_tags::MarkerTag>;
using PointCloudGroup = Group<group_tags::PointCloudTag>;
using ModelGroup = Group<group_tags::ModelTag>;

/// How a camera group came into existence. `None` is a plain user group;
/// `Folder` records the import folder in `folder`; `Station` groups cameras
/// that share a camera station; `Fixed` groups a rigid multi-camera rig.
enum class CameraGroupType { None, Folder, Station, Fixed };

/// A camera group. Unlike `Group<Tag>` this carries the group type and, for
/// `CameraGroupType::Folder`, the source folder path.
struct CameraGroup {
    std::int32_t id = -1;
    std::string label;
    std::vector<std::int32_t> members;
    CameraGroupType type = CameraGroupType::None;
    std::string folder;

    bool operator==(const CameraGroup&) const = default;
};

}  // namespace alphashape
