#pragma once

/// @file camera_image.hpp
/// @brief Load a camera's frame in the pixel frame the pipeline runs in.
///
/// A digital camera's frame is decoded as-is. A scanned-film camera
/// (WP-13.18b) is resampled once into the canonical film-normalised frame
/// described in model/camera/film_geometry.hpp, so every later stage can use
/// an ordinary Image and pixel Calibration. The raw scan stays available to
/// DetectFiducials and CalibrateFiducials, which must measure scan pixels.

#include <filesystem>
#include <string>
#include <utility>

#include "common/core/types.hpp"
#include "image/image.hpp"

namespace alphashape {

class Document;
class Chunk;

/// Resolve a frame's recorded path against the document's directory. A frame
/// path may be absolute; an unsaved document has no anchor, so the path is
/// taken as given. Shared by the pipeline tasks so they all resolve identically.
std::filesystem::path resolve_frame_path(const Document& document,
                                         const std::string& path);

/// The sensor whose film settings and calibration govern this camera: the
/// camera's sensor, or that sensor's master when it is a slave (WP-13.16).
/// Returns null when the camera has no usable sensor, which a digital-only
/// pipeline may legitimately allow; throws ValueError when the index exists
/// but the master link is broken.
const Sensor* find_effective_sensor(const Chunk& chunk, const Camera& camera);

/// As above, but a missing sensor is an error (callers that need a lens).
const Sensor& effective_sensor(const Chunk& chunk, const Camera& camera);

/// Pixel size the pipeline uses for a sensor. A film sensor with complete film
/// geometry reports its normalised image size; every other sensor reports its
/// stored width and height unchanged.
std::pair<int, int> sensor_pipeline_size(const Sensor& sensor);

/// The sensor's calibration with its width and height set to the pipeline
/// pixel size. The lens coefficients are relative to the image centre, so a
/// film sensor's calibration is valid in the normalised frame once the size is
/// swapped; every other sensor is returned unchanged. Pipeline stages that
/// project into or scale a loaded image must use this, not `sensor.calibration`.
Calibration sensor_pipeline_calibration(const Sensor& sensor);

/// Decode the frame linked to `camera`. For a film camera returns the
/// film-normalised resampling; throws Error when the film transform or the
/// normalised geometry is missing (the caller must run DetectFiducials and
/// CalibrateFiducials first), naming the camera.
Image load_camera_image(const Document& document, const Chunk& chunk,
                        const Camera& camera);

}  // namespace alphashape
