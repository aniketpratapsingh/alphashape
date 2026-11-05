#pragma once

/// @file calibrate_camera_task.hpp
/// @brief CalibrateCamera: chessboard calibration per sensor (WP-13.15a).
///
/// For every sensor in the active chunk, each camera's frame is decoded, a
/// chessboard is detected in it, and one shared interior orientation is solved
/// together with one pose per view. Alignment is not required. The result is
/// written back to the sensor in a staged chunk copy when `apply` is true, so
/// an exception leaves the document untouched.
///
/// Sensors with fewer than three usable boards are skipped and reported in the
/// chunk metadata; if no sensor could be calibrated the task throws.

#include <cstdint>
#include <string>
#include <vector>

#include "common/algorithm/calibration/calibrate.hpp"
#include "common/core/document.hpp"
#include "common/tasks/task.hpp"

namespace alphashape {

/// Everything the task reads from `Params`, resolved and converted.
struct CalibrateCameraOptions {
    int board_cols = 9;
    int board_rows = 6;
    double square_size = 0.025;
    CalibrationFitFlags fit = default_calibration_fit_flags();
    bool apply = true;
    /// Empty disables the report. `report_format` is "html" or "pdf"; any
    /// other value is rejected before solving.
    std::string report_path;
    std::string report_format = "html";
};

struct SensorCalibration {
    std::int32_t sensor_id = -1;
    std::string label;
    bool solved = false;
    std::string skipped;  ///< Reason when `solved` is false.
    CalibrationResult result;
};

/// The staged outcome of one run. Pure with respect to the document.
struct CalibrateCameraPlan {
    std::vector<SensorCalibration> sensors;
    int solved = 0;
};

CalibrateCameraPlan plan_calibrate_camera(const Document& document,
                                          std::int32_t chunk_id,
                                          const CalibrateCameraOptions& options);

class CalibrateCameraTask : public AbstractTask {
public:
    std::string name() const override { return "CalibrateCamera"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params,
               const Progress& progress) override;
};

}  // namespace alphashape
