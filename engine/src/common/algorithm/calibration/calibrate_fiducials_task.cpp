#include "common/algorithm/calibration/calibrate_fiducials_task.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/calibration/fiducials.hpp"
#include "common/core/chunk.hpp"
#include "common/core/types.hpp"

namespace alphashape {

std::vector<Fiducial> derive_film_layout(const std::vector<Fiducial>& current,
                                         const std::vector<Eigen::Vector2d>& pixels,
                                         double resolution_mm_per_px) {
    if (!std::isfinite(resolution_mm_per_px) || !(resolution_mm_per_px > 0.0)) {
        throw ValueError(
            "CalibrateFiducials: resolution must be finite and positive");
    }
    if (current.empty() || current.size() != pixels.size()) {
        throw ValueError(
            "CalibrateFiducials: every listed fiducial needs a measured pixel");
    }
    Eigen::Vector2d minimum = Eigen::Vector2d::Constant(
        std::numeric_limits<double>::infinity());
    Eigen::Vector2d maximum = Eigen::Vector2d::Constant(
        -std::numeric_limits<double>::infinity());
    for (const Eigen::Vector2d& pixel : pixels) {
        if (!pixel.allFinite()) {
            throw ValueError("CalibrateFiducials: a fiducial pixel is not finite");
        }
        minimum = minimum.cwiseMin(pixel);
        maximum = maximum.cwiseMax(pixel);
    }
    const Eigen::Vector2d centre = 0.5 * (minimum + maximum);
    std::vector<Fiducial> layout = current;
    for (std::size_t i = 0; i < layout.size(); ++i) {
        layout[i].film_mm = Eigen::Vector2d(
            (pixels[i].x() - centre.x()) * resolution_mm_per_px,
            -(pixels[i].y() - centre.y()) * resolution_mm_per_px);
    }
    return layout;
}

namespace {

/// True when every listed fiducial has a measured pixel on this camera.
bool complete_scan(const Camera& camera, std::size_t count,
                   std::vector<Eigen::Vector2d>& pixels) {
    if (camera.fiducial_pixels.size() != count) {
        return false;
    }
    pixels.clear();
    pixels.reserve(count);
    for (const std::optional<Eigen::Vector2d>& pixel : camera.fiducial_pixels) {
        if (!pixel.has_value()) {
            pixels.clear();
            return false;
        }
        pixels.push_back(*pixel);
    }
    return true;
}

/// Re-fit every camera on one sensor from its measured marks against the given
/// layout. A camera without enough marks gets no transform.
void refit_camera_transforms(Chunk& chunk, std::int32_t sensor_index,
                             const FilmSettings& film) {
    const std::size_t minimum =
        film.transform == FilmTransformType::Affine ? 3U : 4U;
    for (Camera& camera : chunk.cameras()) {
        if (camera.sensor_id != sensor_index) {
            continue;
        }
        std::vector<Eigen::Vector2d> film_points;
        std::vector<Eigen::Vector2d> image_points;
        const std::size_t count =
            std::min(camera.fiducial_pixels.size(), film.fiducials.size());
        for (std::size_t i = 0; i < count; ++i) {
            if (!camera.fiducial_pixels[i].has_value()) {
                continue;
            }
            film_points.push_back(film.fiducials[i].film_mm);
            image_points.push_back(*camera.fiducial_pixels[i]);
        }
        if (film_points.size() < minimum) {
            camera.film_transform.reset();
            continue;
        }
        Eigen::Matrix3d transform;
        double rms = 0.0;
        if (fit_film_transform(film_points, image_points, film.transform, transform,
                               rms)) {
            camera.film_transform = transform;
        } else {
            camera.film_transform.reset();
        }
    }
}

/// Calibrate one sensor into `staged`. Returns false when the sensor has no
/// film layout or no complete reference scan. Throws ValueError when the
/// caller explicitly named a camera on this sensor that is missing marks.
bool calibrate_one_sensor(const Chunk& chunk, Chunk& staged, std::size_t sensor_index,
                          const CalibrateFiducialsOptions& options) {
    const Sensor& sensor = chunk.sensors()[sensor_index];
    if (!sensor.film.enabled || sensor.film.fiducials.empty()) {
        return false;
    }
    const std::size_t count = sensor.film.fiducials.size();
    int reference = -1;
    bool belongs = false;
    std::vector<Eigen::Vector2d> pixels;
    for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
        const Camera& camera = chunk.cameras()[i];
        if (camera.sensor_id != static_cast<std::int32_t>(sensor_index)) {
            continue;
        }
        if (options.camera >= 0) {
            if (camera.id != options.camera) {
                continue;
            }
            belongs = true;
        }
        if (complete_scan(camera, count, pixels)) {
            reference = static_cast<int>(i);
            break;
        }
    }
    if (reference < 0) {
        if (belongs) {
            throw ValueError(
                "CalibrateFiducials: the reference camera is missing fiducials");
        }
        return false;
    }
    FilmSettings film = sensor.film;
    film.fiducials = derive_film_layout(sensor.film.fiducials, pixels,
                                        options.resolution_mm_per_px);
    staged.sensors()[sensor_index].film = film;
    refit_camera_transforms(staged, static_cast<std::int32_t>(sensor_index), film);
    const std::string prefix =
        "calibrate_fiducials/" + std::to_string(sensor_index) + "/";
    staged.meta().set(prefix + "reference_camera",
                      std::to_string(
                          chunk.cameras()[static_cast<std::size_t>(reference)].id));
    staged.meta().set(prefix + "resolution",
                      std::to_string(options.resolution_mm_per_px));
    return true;
}

}  // namespace

void CalibrateFiducialsTask::declare_params(Params& params) const {
    params.declare({"resolution", ParamType::Float, 0.0,
                    "Millimetres per scan pixel (must be greater than zero "
                    "when a sensor has film data)."});
    params.declare({"camera", ParamType::Int, std::int64_t{-1},
                    "Reference camera id, or -1 for the first complete scan."});
}

std::vector<Stage> CalibrateFiducialsTask::stages() const {
    return {Stage{"calibrate", 0}, Stage{"apply", 0}};
}

void CalibrateFiducialsTask::apply(Document& document, Params& params,
                                   const Progress& progress) {
    CalibrateFiducialsOptions options;
    options.resolution_mm_per_px = params.get_float("resolution");
    options.camera = static_cast<std::int32_t>(params.get_int("camera"));
    if (options.camera < -1) {
        throw ValueError("CalibrateFiducials: camera must be -1 or a camera id");
    }
    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("CalibrateFiducials needs an active chunk");
    }
    bool has_film = false;
    for (const Sensor& sensor : chunk->sensors()) {
        if (sensor.film.enabled && !sensor.film.fiducials.empty()) {
            has_film = true;
            break;
        }
    }
    if (!has_film) {
        // Nothing to calibrate (digital-only project): a no-op run.
        progress.update("calibrate", 1.0);
        progress.update("apply", 1.0);
        return;
    }
    if (!std::isfinite(options.resolution_mm_per_px) ||
        !(options.resolution_mm_per_px > 0.0)) {
        throw ValueError(
            "CalibrateFiducials: resolution must be finite and positive");
    }
    progress.update("calibrate", 0.0);
    Chunk staged = *chunk;
    int calibrated = 0;
    for (std::size_t s = 0; s < chunk->sensors().size(); ++s) {
        if (calibrate_one_sensor(*chunk, staged, s, options)) {
            ++calibrated;
        }
    }
    if (calibrated == 0) {
        throw Error(
            "CalibrateFiducials: no film sensor with a complete reference scan");
    }
    progress.update("calibrate", 1.0);
    progress.update("apply", 0.0);
    *chunk = std::move(staged);
    progress.update("apply", 1.0);
    log(LogLevel::Info) << "CalibrateFiducials: derived the layout of "
                        << calibrated << " sensors";
}

}  // namespace alphashape
