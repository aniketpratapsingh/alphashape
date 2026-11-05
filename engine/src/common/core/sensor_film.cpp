#include "common/core/sensor_film.hpp"

#include <cmath>
#include <cstddef>

#include <Eigen/Core>

#include "model/camera/film_geometry.hpp"

namespace alphashape {

SensorFilmNormalisation normalise_sensor_film(Sensor& sensor,
                                              const std::vector<Camera>& cameras,
                                              std::int32_t sensor_index) {
    SensorFilmNormalisation result;
    if (!sensor.film.enabled || !film_geometry_complete(sensor.film)) {
        return result;  // no canonical frame yet: the raw image still stands
    }
    const FilmSize size = film_normalised_size(sensor.film);
    if (!sensor.film.calibration_normalised) {
        // Mean linear part of the sensor's per-scan film transforms. A scan
        // maps film millimetres to pixels with A, so sqrt|det A| is scan pixels
        // per millimetre. s converts a raw pixel length into normalised pixels.
        Eigen::Matrix2d linear = Eigen::Matrix2d::Zero();
        int count = 0;
        for (const Camera& camera : cameras) {
            if (camera.sensor_id != sensor_index || !camera.film_transform.has_value()) {
                continue;
            }
            const Eigen::Matrix2d part = camera.film_transform->topLeftCorner<2, 2>();
            if (!part.allFinite()) {
                continue;
            }
            linear += part;
            ++count;
        }
        if (count > 0) {
            linear /= static_cast<double>(count);
            const double determinant =
                linear(0, 0) * linear(1, 1) - linear(0, 1) * linear(1, 0);
            const double denominator =
                sensor.film.resolution_mm_per_px * std::sqrt(std::abs(determinant));
            if (std::isfinite(denominator) && denominator > 0.0) {
                const double scale = 1.0 / denominator;
                Calibration& calibration = sensor.calibration;
                calibration.f *= scale;
                calibration.cx *= scale;
                calibration.cy *= scale;
                calibration.b1 *= scale;
                calibration.b2 *= scale;
                sensor.film.calibration_normalised = true;
                result.coefficients_rescaled = true;
                result.coefficient_scale = scale;
            }
        }
    }
    if (sensor.width != size.first || sensor.height != size.second ||
        sensor.calibration.width != size.first ||
        sensor.calibration.height != size.second) {
        sensor.width = size.first;
        sensor.height = size.second;
        sensor.calibration.width = size.first;
        sensor.calibration.height = size.second;
        result.sizes_changed = true;
    }
    return result;
}

}  // namespace alphashape
