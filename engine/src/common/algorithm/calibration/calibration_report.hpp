#pragma once

/// @file calibration_report.hpp
/// @brief Human-readable calibration reports: distortion plot, HTML and PDF.
///
/// A report renders one solved sensor, or explains why a sensor was skipped.
/// Every renderer formats numbers with snprintf-family helpers, never a
/// locale-dependent stream, so a report is identical regardless of the process
/// locale's decimal separator.

#include <cstdint>
#include <string>
#include <vector>

#include "common/algorithm/calibration/calibrate.hpp"
#include "image/image.hpp"

namespace alphashape {

/// One sensor's section of a report.
struct CalibrationReportInput {
    std::string sensor_label;
    CalibrationResult result;
    /// View camera labels, parallel to `result.views`. A short list leaves the
    /// remaining rows labelled by camera id.
    std::vector<std::string> view_labels;
    /// When non-empty the sensor was not calibrated and the report prints this
    /// reason as a one-line section instead of any tables.
    std::string skipped;
};

/// Plot of the lens distortion: white 8-bit RGB, a 17 x 13 lattice of ideal
/// positions, and an arrow from each ideal position to where the lens actually
/// maps it. Arrows are magnified so the longest is about 8% of the plot width.
/// Nothing else is drawn.
Image render_distortion_plot(const Calibration& calibration, int width_px,
                             int height_px);

/// Self-contained HTML5 report with inline CSS and a base64 PNG plot per
/// sensor. Every label is HTML-escaped.
std::string calibration_report_html(
    const std::vector<CalibrationReportInput>& sensors);

/// A4 PDF with the same content as text lines and the distortion plot.
std::vector<std::uint8_t> calibration_report_pdf(
    const std::vector<CalibrationReportInput>& sensors);

}  // namespace alphashape
