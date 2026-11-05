#pragma once

// Calibration file I/O (WP-13.16): save and load one sensor calibration in a
// small set of interchange formats.
//
// Formats:
//   * Xml         -- our own <calibration> document element (lossless).
//   * Json        -- {"format":"alphashape-calibration","version":1,...}.
//   * OpenCvXml   -- the OpenCV FileStorage XML layout.
//   * OpenCvYaml  -- the OpenCV FileStorage YAML layout.
//
// The OpenCV mapping. OpenCV's pinhole model puts the centre of the top-left
// pixel at (0, 0) and uses
//
//     u = fx x'' + s y'' + cx_o,   v = fy y'' + cy_o
//     x'' = x (1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 P1 x y + P2 (r^2 + 2 x^2)
//     y'' = y (1 + k1 r^2 + k2 r^4 + k3 r^6) + P1 (r^2 + 2 y^2) + 2 P2 x y
//
// Our model (see model/camera/calibration.hpp) has the same radial part but
// multiplies (r^2 + 2x^2) by p1 and 2xy by p2 -- the names are crossed -- so
// OpenCV's P1 is our p2 and OpenCV's P2 is our p1. The affine terms map as
// fx = f + b1, s = b2, fy = f. Our top-left pixel centre is (0.5, 0.5), so a
// pixel coordinate u corresponds to OpenCV's u - 0.5 and the principal point
// offset becomes cx_o = W/2 + cx - 0.5 (cy_o likewise).
//
// On save the OpenCV formats warn about (and omit) k4, p3 and p4, which the
// exported 5-term distortion cannot express, and refuse Fisheye sensors. On
// load only the first five distortion values are used; any non-zero value
// beyond k3 (rational, thin-prism or tilt terms) is discarded with a warning.

#include <string>
#include <vector>

#include "common/core/types.hpp"

namespace alphashape {

enum class CalibrationFormat { Xml, Json, OpenCvXml, OpenCvYaml };

struct CalibrationFileResult {
    Calibration calibration;
    std::vector<std::string> warnings;
};

/// Write `c` to `path` in `format`. Returns non-fatal warnings.
/// @throws Error on I/O failure, ValueError on unrepresentable content.
std::vector<std::string> save_calibration(const std::string& path, const Calibration& c,
                                          CalibrationFormat format);

/// Read a calibration from `path`. `fallback_width`/`fallback_height` supply
/// the image size for the OpenCV formats when the file does not record one.
/// @throws Error on I/O failure, ValueError on malformed content.
CalibrationFileResult load_calibration(const std::string& path, CalibrationFormat format,
                                       int fallback_width = 0,
                                       int fallback_height = 0);

}  // namespace alphashape
