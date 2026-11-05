#include "common/algorithm/photos/camera_groups.hpp"

#include <algorithm>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/utils/error.hpp"
#include "common/core/document.hpp"
#include "common/core/chunk.hpp"
#include "common/core/frame.hpp"
#include "common/core/groups.hpp"

namespace alphashape {

namespace {

namespace fs = std::filesystem;

/// One group's contents before a group id is allocated.
struct Bucket {
    std::string label;
    std::string folder;
    std::vector<std::int32_t> members;
};

void insert_member(std::vector<std::int32_t>& members, std::int32_t id) {
    const auto slot = std::lower_bound(members.begin(), members.end(), id);
    if (slot == members.end() || *slot != id) {
        members.insert(slot, id);
    }
}

/// A sensor's station identity: the body serial when it is known, otherwise
/// the make and model. The separator cannot appear in EXIF text, so the two
/// forms cannot collide.
std::string station_key(const Sensor& sensor) {
    if (!sensor.serial.empty()) {
        return "serial\x1f" + sensor.serial;
    }
    return "model\x1f" + sensor.make + "\x1f" + sensor.model;
}

std::string station_label(const Sensor& sensor) {
    if (!sensor.serial.empty()) {
        return sensor.serial;
    }
    const std::string model =
        sensor.model.empty() ? sensor.make : sensor.make + " " + sensor.model;
    return model.empty() ? "Unknown" : model;
}

std::map<std::string, Bucket> build_buckets(const Chunk& chunk, GroupType type) {
    std::map<std::string, Bucket> buckets;
    if (type == GroupType::Station) {
        for (const Camera& camera : chunk.cameras()) {
            if (camera.sensor_id < 0 ||
                static_cast<std::size_t>(camera.sensor_id) >= chunk.sensors().size()) {
                throw ValueError("group cameras: camera " + std::to_string(camera.id) +
                                 " has an out-of-range sensor id");
            }
            const Sensor& sensor = chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)];
            Bucket& bucket = buckets[station_key(sensor)];
            bucket.label = station_label(sensor);
            insert_member(bucket.members, camera.id);
        }
        return buckets;
    }
    std::map<std::int32_t, std::string> parents;
    for (const std::shared_ptr<Frame>& frame : chunk.frames()) {
        parents[frame->camera_id()] =
            fs::path(frame->path()).parent_path().lexically_normal().string();
    }
    for (const Camera& camera : chunk.cameras()) {
        const auto parent = parents.find(camera.id);
        if (parent == parents.end()) {
            continue;
        }
        Bucket& bucket = buckets[parent->second];
        bucket.folder = parent->second;
        const std::string name = fs::path(parent->second).filename().string();
        bucket.label = name.empty() ? parent->second : name;
        insert_member(bucket.members, camera.id);
    }
    return buckets;
}

}  // namespace

const char* group_type_name(GroupType type) {
    switch (type) {
        case GroupType::Station: return "station";
        case GroupType::Folder: return "folder";
        case GroupType::None: return "none";
    }
    return "none";
}

GroupType parse_group_type(const std::string& name) {
    if (name == "station") {
        return GroupType::Station;
    }
    if (name == "folder") {
        return GroupType::Folder;
    }
    if (name == "none") {
        return GroupType::None;
    }
    throw ValueError("group type must be 'station', 'folder' or 'none', got '" + name +
                     "'");
}

void group_cameras(Chunk& chunk, GroupType type) {
    std::vector<CameraGroup>& groups = chunk.camera_groups();
    if (type == GroupType::None) {
        groups.clear();
        return;
    }
    const CameraGroupType wanted =
        type == GroupType::Station ? CameraGroupType::Station : CameraGroupType::Folder;
    groups.erase(std::remove_if(groups.begin(), groups.end(),
                                [wanted](const CameraGroup& group) {
                                    return group.type == wanted;
                                }),
                 groups.end());
    for (auto& [key, bucket] : build_buckets(chunk, type)) {
        (void)key;
        CameraGroup group;
        group.id = chunk.allocate_id();
        group.label = std::move(bucket.label);
        group.type = wanted;
        group.folder = std::move(bucket.folder);
        group.members = std::move(bucket.members);
        std::sort(group.members.begin(), group.members.end());
        group.members.erase(std::unique(group.members.begin(), group.members.end()),
                            group.members.end());
        groups.push_back(std::move(group));
    }
}

namespace {

/// Stage the whole camera-group list and rebuild it on commit, so undo restores
/// the previous grouping exactly (including ids and labels).
class GroupCamerasCommand : public StagedMutation {
public:
    GroupCamerasCommand(std::int32_t chunk_id, GroupType type)
        : chunk_id_(chunk_id), type_(type) {}

    std::string name() const override { return "Group Cameras"; }

protected:
    void validate(Document& document) override { (void)target(document); }
    void stage(Document& document) override {
        previous_ = target(document).camera_groups();
    }
    void commit(Document& document) override { group_cameras(target(document), type_); }
    void restore(Document& document) override {
        target(document).camera_groups() = previous_;
    }

private:
    Chunk& target(Document& document) {
        Chunk* chunk = document.find_chunk(chunk_id_);
        if (chunk == nullptr) {
            throw ValueError("no chunk with id " + std::to_string(chunk_id_));
        }
        return *chunk;
    }

    std::int32_t chunk_id_;
    GroupType type_;
    std::vector<CameraGroup> previous_;
};

}  // namespace

std::unique_ptr<AbstractCommand> make_group_cameras_command(std::int32_t chunk_id,
                                                            GroupType type) {
    return std::make_unique<GroupCamerasCommand>(chunk_id, type);
}

}  // namespace alphashape
