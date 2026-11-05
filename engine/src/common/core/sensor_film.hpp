#pragma once

/// @file sensor_film.hpp
/// @brief Enforce the film-normalised sensor invariant (WP-13.18b).
///
/// When a sensor's film geometry is complete the pipeline runs in
/// film-NORMALISED pixels: the sensor's width/height and calibration geometry
/// ARE the normalised size and the pixel-unit coefficients (f, cx, cy, b1, b2)
/// are in those pixels. This helper is the single place that establishes the
/// invariant, so every mutation path (commands, calibration tasks, project
/// load) agrees on it.

#include <cstdint>
#include <vector>

#include "common/core/types.hpp"

namespace alphashape {

/// What normalise_sensor_film() changed.
struct SensorFilmNormalisation {
    /// The sizes were rewritten to the normalised image size.
    bool sizes_changed = false;
    /// The pixel-unit coefficients were rescaled into normalised pixels.
    bool coefficients_rescaled = false;
    /// The factor applied to f, cx, cy, b1, b2 (1.0 when nothing changed).
    double coefficient_scale = 1.0;
};

/// Establish the invariant on one sensor. Does nothing while its film geometry
/// is incomplete. When complete, sets width/height and the calibration's
/// width/height to the normalised size, and — the first time only, on a sensor
/// whose coefficients are still in raw scan pixels — rescales f, cx, cy, b1,
/// b2 by
///
///     s = (raw scan pixel size) / (normalised pixel size)
///       = 1 / (resolution_mm_per_px * sqrt|det(A)|)
///
/// where A is the mean 2x2 linear part of the film transforms of `cameras`
/// whose sensor_id is `sensor_index`. Without any such transform the
/// coefficients are left alone (only the sizes are set) and the flag stays
/// false; a later call converts once a transform exists. Radial/tangential
/// terms are dimensionless on normalised image coordinates and never change.
SensorFilmNormalisation normalise_sensor_film(Sensor& sensor,
                                              const std::vector<Camera>& cameras,
                                              std::int32_t sensor_index);

}  // namespace alphashape
