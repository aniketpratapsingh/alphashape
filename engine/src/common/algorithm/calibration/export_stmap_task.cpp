#include "common/algorithm/calibration/export_stmap_task.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "base/utils/error.hpp"
#include "common/algorithm/calibration/stmap.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"

namespace alphashape {

void ExportSTMapTask::declare_params(Params& params) const {
    params.declare({"path", ParamType::String, std::string{},
                    "Destination path for the 32-bit float STMap TIFF."});
    ParamSpec sensor{"sensor", ParamType::Int, std::int64_t{0},
                     "Sensor index in the active chunk."};
    sensor.minimum = std::int64_t{0};
    params.declare(sensor);
    ParamSpec width{"width", ParamType::Int, std::int64_t{0},
                    "Map width in pixels; 0 keeps the calibration size."};
    width.minimum = std::int64_t{0};
    params.declare(width);
    ParamSpec height{"height", ParamType::Int, std::int64_t{0},
                     "Map height in pixels; 0 keeps the calibration size."};
    height.minimum = std::int64_t{0};
    params.declare(height);
}

std::vector<Stage> ExportSTMapTask::stages() const {
    return {Stage{"export", 0}};
}

void ExportSTMapTask::apply(Document& document, Params& params,
                            const Progress& progress) {
    const std::string path = params.get_string("path");
    if (path.empty()) {
        throw ValueError("ExportSTMap: path is required");
    }
    const int sensor_index = static_cast<int>(params.get_int("sensor"));
    const int width = static_cast<int>(params.get_int("width"));
    const int height = static_cast<int>(params.get_int("height"));

    const Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("ExportSTMap needs an active chunk");
    }
    if (sensor_index < 0 ||
        sensor_index >= static_cast<int>(chunk->sensors().size())) {
        throw ValueError("ExportSTMap: sensor index " +
                         std::to_string(sensor_index) +
                         " is out of range for this chunk");
    }

    progress.update("export", 0.0);
    const Calibration& calibration =
        chunk->sensors()[static_cast<std::size_t>(sensor_index)].calibration;
    const STMap map = compute_stmap(calibration, width, height);
    write_stmap_tiff(path, map);
    progress.update("export", 1.0);
}

}  // namespace alphashape
