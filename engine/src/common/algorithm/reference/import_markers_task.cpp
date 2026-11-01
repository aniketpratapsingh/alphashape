#include "common/algorithm/reference/import_markers_task.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/reference/markers_io.hpp"
#include "common/commands/command.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/marker.hpp"
#include "common/core/types.hpp"

namespace alphashape {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw Error("ImportMarkers: cannot open '" + path + "'");
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    if (in.bad()) {
        throw Error("ImportMarkers: failed while reading '" + path + "'");
    }
    return buffer.str();
}

std::string line_prefix(std::size_t line) {
    return line == 0 ? std::string{} : "line " + std::to_string(line) + ": ";
}

/// One validated marker, with camera labels already resolved to ids.
struct ImportMarkerPlan {
    std::string label;
    std::optional<Eigen::Vector3d> position;
    std::vector<std::pair<std::int32_t, Eigen::Vector2d>> projections;
};

const Camera* find_camera_by_label(const Chunk& chunk, const std::string& label) {
    for (const Camera& camera : chunk.cameras()) {
        if (camera.label == label) {
            return &camera;
        }
    }
    return nullptr;
}

/// The number of cameras carrying `label`. A file projection names its camera
/// by label only, so a label shared by two cameras is ambiguous.
std::size_t count_cameras_by_label(const Chunk& chunk, const std::string& label) {
    std::size_t count = 0;
    for (const Camera& camera : chunk.cameras()) {
        if (camera.label == label) {
            ++count;
        }
    }
    return count;
}

const Camera* find_camera_by_id(const Chunk& chunk, std::int32_t id) {
    for (const Camera& camera : chunk.cameras()) {
        if (camera.id == id) {
            return &camera;
        }
    }
    return nullptr;
}

Marker* find_marker_by_label(Chunk& chunk, const std::string& label) {
    for (Marker& marker : chunk.markers()) {
        if (marker.label == label) {
            return &marker;
        }
    }
    return nullptr;
}

/// Insert a projection keeping the container sorted by camera id (the model's
/// invariant); a repeated camera replaces the existing pixel.
void set_projection(Marker& marker, std::int32_t camera_id, const Eigen::Vector2d& uv) {
    for (MarkerProjection& projection : marker.projections) {
        if (projection.camera_id == camera_id) {
            projection.pixel = uv;
            return;
        }
    }
    MarkerProjection projection;
    projection.camera_id = camera_id;
    projection.pixel = uv;
    const auto it = std::lower_bound(
        marker.projections.begin(), marker.projections.end(), camera_id,
        [](const MarkerProjection& item, std::int32_t id) {
            return item.camera_id < id;
        });
    marker.projections.insert(it, projection);
}

class ImportMarkersCommand : public StagedMutation {
public:
    ImportMarkersCommand(std::int32_t chunk_id, std::vector<ImportMarkerPlan> markers)
        : chunk_id_(chunk_id), markers_(std::move(markers)) {}

    std::string name() const override { return "Import Markers"; }

protected:
    void validate(Document& document) override {
        Chunk& chunk = require_chunk(document);
        for (const ImportMarkerPlan& marker : markers_) {
            if (marker.label.empty()) {
                throw ValueError("import markers: empty marker label");
            }
            if (marker.position.has_value() && !marker.position->allFinite()) {
                throw ValueError("import markers: marker '" + marker.label +
                                 "' has a non-finite position");
            }
            for (const auto& [camera_id, uv] : marker.projections) {
                if (find_camera_by_id(chunk, camera_id) == nullptr) {
                    throw ValueError("import markers: no camera with id " +
                                     std::to_string(camera_id));
                }
                if (!uv.allFinite()) {
                    throw ValueError("import markers: marker '" + marker.label +
                                     "' has a non-finite projection");
                }
            }
        }
    }

    void stage(Document& document) override {
        Chunk& chunk = require_chunk(document);
        markers_before_ = chunk.markers();
        next_id_ = chunk.next_id();
    }

    void commit(Document& document) override {
        apply_plan(require_chunk(document));
    }

    void restore(Document& document) override {
        Chunk& chunk = require_chunk(document);
        chunk.markers() = markers_before_;
        chunk.set_next_id(next_id_);
    }

private:
    Chunk& require_chunk(Document& document) const {
        for (auto& workspace : document.workspaces()) {
            if (Chunk* chunk = workspace->find_chunk(chunk_id_)) {
                return *chunk;
            }
        }
        throw ValueError("import markers: no chunk with id " +
                         std::to_string(chunk_id_));
    }

    void apply_plan(Chunk& chunk) const {
        for (const ImportMarkerPlan& item : markers_) {
            Marker* marker = find_marker_by_label(chunk, item.label);
            if (marker == nullptr) {
                Marker created;
                created.id = chunk.allocate_id();
                created.label = item.label;
                auto it = std::lower_bound(
                    chunk.markers().begin(), chunk.markers().end(), created.id,
                    [](const Marker& existing, std::int32_t id) {
                        return existing.id < id;
                    });
                it = chunk.markers().insert(it, std::move(created));
                marker = &*it;
            }
            if (item.position.has_value()) {
                marker->position = item.position;
            }
            for (const auto& [camera_id, uv] : item.projections) {
                set_projection(*marker, camera_id, uv);
            }
        }
    }

    std::int32_t chunk_id_;
    std::vector<ImportMarkerPlan> markers_;
    std::vector<Marker> markers_before_;
    std::int32_t next_id_ = 0;
};

}  // namespace
}  // namespace alphashape

namespace alphashape {

void ImportMarkersTask::declare_params(Params& params) const {
    params.declare({"path", ParamType::String, std::string{}, "Input CSV or XML file. Required."});
    ParamSpec format{"format", ParamType::String, std::string{},
                     "csv or xml. Empty infers it from the path extension."};
    format.choices = {"", "csv", "xml"};
    params.declare(format);
}

std::vector<Stage> ImportMarkersTask::stages() const {
    return {Stage{"parse", 0}, Stage{"apply", 0}};
}

void ImportMarkersTask::apply(Document& document, Params& params,
                              const Progress& progress) {
    progress.update("parse", 0.0);
    const std::string path = params.get_string("path");
    if (path.empty()) {
        throw ValueError("ImportMarkers: path is required");
    }
    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("ImportMarkers needs an active chunk");
    }
    const reference::MarkerIoFormat format =
        reference::marker_io_format(params.get_string("format"), path, "ImportMarkers");
    std::string text = read_file(path);
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEFU &&
        static_cast<unsigned char>(text[1]) == 0xBBU &&
        static_cast<unsigned char>(text[2]) == 0xBFU) {
        text.erase(0, 3);  // UTF-8 BOM
    }
    const std::vector<reference::MarkerIoMarker> parsed =
        format == reference::MarkerIoFormat::Csv
            ? reference::parse_marker_csv(text)
            : reference::parse_marker_xml(text);

    // Pass 1: resolve every camera and validate every row. Nothing is written
    // until this loop has accepted the whole file.
    std::vector<ImportMarkerPlan> plan;
    plan.reserve(parsed.size());
    std::size_t projection_count = 0;
    std::size_t created_count = 0;
    for (const reference::MarkerIoMarker& marker : parsed) {
        ImportMarkerPlan item;
        item.label = marker.label;
        item.position = marker.position;
        if (find_marker_by_label(*chunk, marker.label) == nullptr) {
            ++created_count;
        }
        for (const reference::MarkerIoProjection& projection : marker.projections) {
            if (count_cameras_by_label(*chunk, projection.camera) > 1) {
                throw ValueError("ImportMarkers: " + line_prefix(projection.line) +
                                 "camera label '" + projection.camera +
                                 "' matches more than one camera");
            }
            const Camera* camera = find_camera_by_label(*chunk, projection.camera);
            if (camera == nullptr) {
                throw ValueError("ImportMarkers: " + line_prefix(projection.line) +
                                 "no camera with label '" + projection.camera + "'");
            }
            if (!projection.uv.allFinite()) {
                throw ValueError("ImportMarkers: " + line_prefix(projection.line) +
                                 "projection of marker '" + marker.label +
                                 "' is not finite");
            }
            item.projections.emplace_back(camera->id, projection.uv);
            ++projection_count;
        }
        plan.push_back(std::move(item));
    }
    progress.update("parse", 1.0);

    // Pass 2: one undoable mutation. A transaction puts the import on the undo
    // stack exactly like Import Reference; applying the staged command directly
    // would change the document without a history entry.
    progress.update("apply", 0.0);
    if (!plan.empty()) {
        CommandTransaction transaction("Import Markers");
        transaction.add(std::make_unique<ImportMarkersCommand>(chunk->id(),
                                                               std::move(plan)));
        transaction.apply(document);
    }
    progress.update("apply", 1.0);
    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "ImportMarkers: " << projection_count
                            << " projection(s) across " << parsed.size()
                            << " marker(s)";
        if (created_count > 0) {
            log(LogLevel::Info) << "ImportMarkers: created " << created_count
                                << " new marker(s)";
        }
    }
}

}  // namespace alphashape
