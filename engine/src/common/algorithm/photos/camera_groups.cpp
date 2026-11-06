#include "common/algorithm/photos/camera_groups.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
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

/// EXIF text often carries padding (a serial written into a fixed-width field,
/// or a trailing newline from a maker note). Trim it before comparing: two
/// bodies whose serials differ only by whitespace are the same body.
std::string trim(const std::string& value) {
    const std::size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const std::size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

/// A sensor's station identity: the body serial when it is known, otherwise
/// the make and model, otherwise the sensor's own index. The separator cannot
/// appear in EXIF text, so the forms cannot collide, and the index fallback
/// keeps N unknown sensors in N groups instead of one shared "Unknown" bucket.
std::string station_key(const Sensor& sensor, std::size_t sensor_id) {
    const std::string serial = trim(sensor.serial);
    if (!serial.empty()) {
        return "serial\x1f" + serial;
    }
    const std::string make = trim(sensor.make);
    const std::string model = trim(sensor.model);
    if (make.empty() && model.empty()) {
        return "sensor\x1f" + std::to_string(sensor_id);
    }
    return "model\x1f" + make + "\x1f" + model;
}

std::string station_label(const Sensor& sensor) {
    const std::string serial = trim(sensor.serial);
    if (!serial.empty()) {
        return serial;
    }
    const std::string make = trim(sensor.make);
    const std::string model = trim(sensor.model);
    if (!make.empty() && !model.empty()) {
        return make + " " + model;
    }
    if (!make.empty()) {
        return make;
    }
    return model.empty() ? "Unknown" : model;
}

std::map<std::string, Bucket> build_buckets(const Chunk& chunk, GroupType type,
                                            std::size_t& ungrouped) {
    std::map<std::string, Bucket> buckets;
    ungrouped = 0;
    if (type == GroupType::Station) {
        for (const Camera& camera : chunk.cameras()) {
            if (camera.sensor_id < 0 ||
                static_cast<std::size_t>(camera.sensor_id) >= chunk.sensors().size()) {
                throw ValueError("group cameras: camera " + std::to_string(camera.id) +
                                 " has an out-of-range sensor id");
            }
            const std::size_t index = static_cast<std::size_t>(camera.sensor_id);
            const Sensor& sensor = chunk.sensors()[index];
            Bucket& bucket = buckets[station_key(sensor, index)];
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
            ++ungrouped;
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

std::size_t component_count(const std::string& folder) {
    std::size_t count = 0;
    for (const fs::path& part : fs::path(folder)) {
        (void)part;
        ++count;
    }
    return count;
}

/// The last `count` components of `folder`, joined with '/'. Used to grow a
/// colliding base name into the shortest trailing path that is still unique.
std::string trailing_suffix(const std::string& folder, std::size_t count) {
    const std::size_t total = component_count(folder);
    const std::size_t first = total > count ? total - count : 0;
    std::size_t seen = 0;
    fs::path out;
    for (const fs::path& part : fs::path(folder)) {
        if (seen++ >= first) {
            out /= part;
        }
    }
    return out.generic_string();
}

}  // namespace

void disambiguate_folder_labels(std::vector<CameraGroup>& groups) {
    std::map<std::string, std::vector<std::size_t>> by_label;
    for (std::size_t i = 0; i < groups.size(); ++i) {
        if (groups[i].type == CameraGroupType::Folder) {
            by_label[groups[i].label].push_back(i);
        }
    }
    for (auto& entry : by_label) {
        std::vector<std::size_t>& indices = entry.second;
        if (indices.size() < 2) {
            continue;
        }
        std::size_t max_parts = 1;
        for (const std::size_t i : indices) {
            max_parts = std::max(max_parts, component_count(groups[i].folder));
        }
        bool resolved = false;
        for (std::size_t take = 2; take <= max_parts && !resolved; ++take) {
            std::vector<std::string> labels;
            std::set<std::string> unique;
            labels.reserve(indices.size());
            for (const std::size_t i : indices) {
                labels.push_back(trailing_suffix(groups[i].folder, take));
                unique.insert(labels.back());
            }
            if (unique.size() == indices.size()) {
                for (std::size_t k = 0; k < indices.size(); ++k) {
                    groups[indices[k]].label = labels[k];
                }
                resolved = true;
            }
        }
        if (!resolved) {
            // Only reachable when one folder path is a suffix of another (or
            // duplicated paths in a hand-edited project). Number them in path
            // order so the result is deterministic.
            std::sort(indices.begin(), indices.end(),
                      [&groups](std::size_t a, std::size_t b) {
                          return groups[a].folder < groups[b].folder;
                      });
            for (std::size_t k = 0; k < indices.size(); ++k) {
                groups[indices[k]].label = entry.first + " (" + std::to_string(k + 1) + ")";
            }
        }
    }
}

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
        // None clears the automatic grouping only. User groups ("Fixed", and
        // any group created by hand with an unknown type) are not ours to
        // remove; Station and Folder likewise leave them alone.
        groups.erase(std::remove_if(groups.begin(), groups.end(),
                                    [](const CameraGroup& group) {
                                        return group.type == CameraGroupType::Station ||
                                               group.type == CameraGroupType::Folder;
                                    }),
                     groups.end());
        return;
    }
    const CameraGroupType wanted =
        type == GroupType::Station ? CameraGroupType::Station : CameraGroupType::Folder;
    groups.erase(std::remove_if(groups.begin(), groups.end(),
                                [wanted](const CameraGroup& group) {
                                    return group.type == wanted;
                                }),
                 groups.end());
    std::size_t ungrouped = 0;
    for (auto& [key, bucket] : build_buckets(chunk, type, ungrouped)) {
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
    if (type == GroupType::Folder) {
        disambiguate_folder_labels(groups);
        if (ungrouped > 0) {
            // A camera with no frame has no folder to group by. Keep it out of
            // every folder group rather than inventing a home for it, but say
            // so: the number is the operator's cue that a frame is missing.
            log(LogLevel::Warn) << "group cameras: " << ungrouped
                                << " camera(s) have no frame and were left ungrouped";
        }
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
        Chunk& chunk = target(document);
        previous_ = chunk.camera_groups();
        // The commit allocates group ids. Restore the counter with the list so
        // a redo hands out the same ids (the same contract AddChunk and
        // DuplicateChunk follow), instead of burning a fresh range per redo.
        previous_next_id_ = chunk.next_id();
    }
    void commit(Document& document) override { group_cameras(target(document), type_); }
    void restore(Document& document) override {
        Chunk& chunk = target(document);
        chunk.camera_groups() = previous_;
        chunk.set_next_id(previous_next_id_);
    }

private:
    Chunk& target(Document& document) {
        Chunk* chunk = document.workspace().find_chunk(chunk_id_);
        if (chunk == nullptr) {
            throw ValueError("no chunk with id " + std::to_string(chunk_id_));
        }
        return *chunk;
    }

    std::int32_t chunk_id_;
    GroupType type_;
    std::vector<CameraGroup> previous_;
    std::int32_t previous_next_id_ = 0;
};

}  // namespace

std::unique_ptr<AbstractCommand> make_group_cameras_command(std::int32_t chunk_id,
                                                            GroupType type) {
    return std::make_unique<GroupCamerasCommand>(chunk_id, type);
}

}  // namespace alphashape
