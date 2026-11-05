#pragma once

/// @file detect_fiducials_task.hpp
/// @brief DetectFiducials: per-scan fiducial marks and film transform
/// (WP-13.18a).
///
/// For every camera whose sensor has `film.enabled`, the frame is decoded,
/// the exposed-frame quad is found, the requested marks are detected, ordered
/// and fitted, and the successful results are written into a staged chunk copy
/// (`Camera::fiducial_pixels`, `Camera::film_transform`). A camera that fails
/// keeps its previous data and is reported in the chunk metadata and the task
/// log; the task throws only when every camera failed.

#include <cstdint>
#include <string>
#include <vector>

#include "common/core/document.hpp"
#include "common/tasks/task.hpp"

namespace alphashape {

/// Resolved task parameters.
struct DetectFiducialsOptions {
    /// "" means "use the sensor's detector"; otherwise circle, corner or
    /// frame_corner.
    std::string detector;
    double min_radius_px = 3.0;
    double max_radius_px = 40.0;
    double max_residual_px = 3.0;
    /// Largest allowed |sigma1/sigma2 - 1| of the fitted film -> scan map.
    double max_anisotropy = 0.02;
    /// Orientation prior used when a dihedrally symmetric layout leaves more
    /// than one plausible relabeling (see FiducialDetectOptions).
    std::string scan_orientation = "0";
    /// Camera ids to process. Empty means every camera on a film sensor.
    std::vector<std::int32_t> cameras;
};

class DetectFiducialsTask : public AbstractTask {
public:
    std::string name() const override { return "DetectFiducials"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
