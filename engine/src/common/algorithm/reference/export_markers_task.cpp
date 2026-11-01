#include "common/algorithm/reference/export_markers_task.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/reference/markers_io.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/marker.hpp"
#include "common/core/types.hpp"

namespace alphashape {
namespace {

const Camera* find_camera_by_id(const Chunk& chunk, std::int32_t id) {
    for (const Camera& camera : chunk.cameras()) {
        if (camera.id == id) {
            return &camera;
        }
    }
    return nullptr;
}

/// Full double round-trip precision for the CSV writer.
constexpr int kExportPrecision = 17;

}  // namespace

void ExportMarkersTask::declare_params(Params& params) const {
    params.declare({"path", ParamType::String, std::string{}, "Output file. Required."});
    ParamSpec format{"format", ParamType::String, std::string{},
                     "csv or xml. Empty infers it from the path extension."};
    format.choices = {"", "csv", "xml"};
    params.declare(format);
}

std::vector<Stage> ExportMarkersTask::stages() const {
    return {Stage{"build", 0}, Stage{"write", 0}};
}

void ExportMarkersTask::apply(Document& document, Params& params,
                              const Progress& progress) {
    progress.update("build", 0.0);
    const std::string path = params.get_string("path");
    if (path.empty()) {
        throw ValueError("ExportMarkers: path is required");
    }
    const Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("ExportMarkers needs an active chunk");
    }
    const reference::MarkerIoFormat format =
        reference::marker_io_format(params.get_string("format"), path, "ExportMarkers");

    std::vector<reference::MarkerIoMarker> markers;
    markers.reserve(chunk->markers().size());
    std::size_t projection_count = 0;
    // The file names a camera by label only, so two cameras that share a label
    // would be written as the same camera and silently merged on re-import.
    // Remember which camera id each written label came from.
    std::map<std::string, std::int32_t> camera_by_label;
    // Two markers with one label would come back as one: the XML reader rejects
    // a duplicate and the CSV reader merges the rows. Refuse to write the file.
    std::set<std::string> marker_labels;
    for (const Marker& marker : chunk->markers()) {
        if (!marker_labels.insert(marker.label).second) {
            throw ValueError("ExportMarkers: duplicate marker label '" +
                             marker.label + "'");
        }
        reference::MarkerIoMarker out;
        out.label = marker.label;
        out.position = marker.position;
        for (const MarkerProjection& projection : marker.projections) {
            const Camera* camera = find_camera_by_id(*chunk, projection.camera_id);
            if (camera == nullptr) {
                throw ValueError("ExportMarkers: marker '" + marker.label +
                                 "' projects into missing camera " +
                                 std::to_string(projection.camera_id));
            }
            const auto [seen, inserted] =
                camera_by_label.emplace(camera->label, camera->id);
            if (!inserted && seen->second != camera->id) {
                throw ValueError("ExportMarkers: camera label '" + camera->label +
                                 "' is used by more than one camera");
            }
            reference::MarkerIoProjection item;
            item.camera = camera->label;
            item.uv = projection.pixel;
            out.projections.push_back(std::move(item));
            ++projection_count;
        }
        markers.push_back(std::move(out));
    }
    progress.update("build", 1.0);

    const std::string text =
        format == reference::MarkerIoFormat::Csv
            ? reference::write_marker_csv(markers, kExportPrecision)
            : reference::write_marker_xml(markers);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw Error("ExportMarkers: cannot open '" + path + "' for writing");
    }
    out << text;
    out.close();
    if (!out) {
        throw Error("ExportMarkers: failed while writing '" + path + "'");
    }
    progress.update("write", 1.0);
    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "ExportMarkers: " << projection_count
                            << " projection(s) across " << markers.size()
                            << " marker(s) -> " << path;
    }
}

}  // namespace alphashape
