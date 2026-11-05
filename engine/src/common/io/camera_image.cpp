#include "common/io/camera_image.hpp"

#include <cstddef>
#include <memory>

#include "base/utils/error.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "image/io/image_io.hpp"
#include "model/camera/film_geometry.hpp"

namespace alphashape {
namespace {

std::string camera_name(const Camera& camera) {
    if (!camera.label.empty()) {
        return "camera '" + camera.label + "'";
    }
    return "camera #" + std::to_string(camera.id);
}

}  // namespace

std::filesystem::path resolve_frame_path(const Document& document,
                                         const std::string& path) {
    const std::filesystem::path recorded(path);
    if (recorded.is_absolute() || document.path().empty()) {
        return recorded;
    }
    return std::filesystem::path(document.path()).parent_path() / recorded;
}

const Sensor* find_effective_sensor(const Chunk& chunk, const Camera& camera) {
    if (camera.sensor_id < 0 ||
        static_cast<std::size_t>(camera.sensor_id) >= chunk.sensors().size()) {
        return nullptr;
    }
    const Sensor& sensor =
        chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)];
    if (sensor.master_sensor_id < 0) {
        return &sensor;
    }
    if (static_cast<std::size_t>(sensor.master_sensor_id) >=
        chunk.sensors().size()) {
        throw ValueError(camera_name(camera) +
                         " names an out-of-range master sensor");
    }
    return &chunk.sensors()[static_cast<std::size_t>(sensor.master_sensor_id)];
}

const Sensor& effective_sensor(const Chunk& chunk, const Camera& camera) {
    const Sensor* sensor = find_effective_sensor(chunk, camera);
    if (sensor == nullptr) {
        throw ValueError(camera_name(camera) + " has no valid sensor");
    }
    return *sensor;
}

// With the film-normalised invariant enforced on the sensor itself
// (normalise_sensor_film), the pipeline frame IS the stored sensor frame.
std::pair<int, int> sensor_pipeline_size(const Sensor& sensor) {
    return {sensor.width, sensor.height};
}

Calibration sensor_pipeline_calibration(const Sensor& sensor) {
    return sensor.calibration;
}

Image load_camera_image(const Document& document, const Chunk& chunk,
                        const Camera& camera) {
    const std::shared_ptr<Frame>* linked = nullptr;
    for (const std::shared_ptr<Frame>& frame : chunk.frames()) {
        if (frame != nullptr && frame->camera_id() == camera.id) {
            linked = &frame;
            break;
        }
    }
    if (linked == nullptr) {
        throw Error(camera_name(camera) + " has no linked frame");
    }
    const Image scan =
        read_image(resolve_frame_path(document, (*linked)->path()).string());
    // A digital camera with no sensor (or a chunk built without sensors, as
    // some digital-only tests do) has no film geometry, so the raw frame
    // stands. A FILM camera without a usable sensor link must never fall back
    // to the raw scan: its keypoints would be in the wrong pixel space.
    const Sensor* sensor = find_effective_sensor(chunk, camera);
    if (sensor == nullptr) {
        if (camera.film_transform.has_value()) {
            throw Error(camera_name(camera) +
                        " is a film camera but its sensor link is missing");
        }
        return scan;
    }
    if (!sensor->film.enabled) {
        return scan;
    }
    if (!camera.film_transform.has_value()) {
        throw Error(camera_name(camera) +
                    " is a film camera without a film transform; run "
                    "DetectFiducials first");
    }
    if (!film_geometry_complete(sensor->film)) {
        throw Error(camera_name(camera) +
                    " has no film-normalised geometry; run CalibrateFiducials "
                    "with a resolution first");
    }
    return resample_film_image(scan, *camera.film_transform, sensor->film);
}

}  // namespace alphashape
