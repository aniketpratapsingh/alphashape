#pragma once

/// @file marker_detector.hpp
/// @brief Automatic marker detection (WP-13.5).
///
/// Finds printed circular and square targets in a grayscale image and returns
/// subpixel image measurements. Only image measurements are produced here;
/// calibration and pose estimation are separate layers.
///
/// Pixel convention: results use the engine's calibration convention, where the
/// centre of the top-left pixel is (0.5, 0.5) and the image spans
/// [0, W] x [0, H] — the same convention as `project_marker()` and
/// `Calibration`. `FloatImage::sample()` addresses pixel centres at integer
/// coordinates, so the detector converts by half a pixel at the API boundary.
/// `gray` is expected in [0, 1].
///
/// Circular targets are located with a scale-normalised Hessian determinant
/// blob detector (Steger 1998), refined by fitting an ellipse (Fitzgibbon,
/// Pilu & Fisher 1999, numerically stable variant) to subpixel boundary points
/// found along radial gradient rays. Square targets are found from an adaptive
/// threshold, contour tracing, Douglas-Peucker corner approximation and a
/// gradient-weighted line fit per side.

#include <optional>
#include <vector>

#include <Eigen/Core>

#include "image/image.hpp"

namespace alphashape {

/// One detected target. `corners` is empty for circular targets and holds four
/// points (counter-clockwise from the top-left) for square targets.
struct MarkerCandidate {
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    double radius_px = 0.0;    ///< Ellipse semi-major axis, or square half-diagonal.
    double angle_rad = 0.0;    ///< Ellipse major-axis angle or square rotation.
    double axis_ratio = 1.0;   ///< Ellipse minor/major axis (1 for squares).
    double response = 0.0;     ///< Detector strength, larger is stronger.
    std::vector<Eigen::Vector2d> corners;
};

struct BlobParams {
    int min_radius = 3;
    int max_radius = 64;
    float threshold = 0.02F;
    bool dark_on_light = true;
    double min_axis_ratio = 0.3;
};

struct QuadParams {
    int min_side = 8;
    int max_side = 512;
    float threshold = 0.02F;
    bool dark_on_light = true;
};

/// Conic parameters of a fitted ellipse. `a >= b`; `angle_rad` is the
/// major-axis direction in image coordinates (X right, Y down).
struct Ellipse {
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    double a = 0.0;
    double b = 0.0;
    double angle_rad = 0.0;
};

/// Detect circular targets. Results are sorted by centre y then x.
std::vector<MarkerCandidate> detect_blobs(const FloatImage& gray, const BlobParams& params = {});

/// Detect square targets. Results are sorted by centre y then x.
std::vector<MarkerCandidate> detect_quads(const FloatImage& gray, const QuadParams& params = {});

/// Re-estimate the centre of an already detected ellipse by iterating the
/// radial edge search and refit. Returns nullopt if the refit fails.
std::optional<Eigen::Vector2d> refine_ellipse_center(const FloatImage& gray,
                                                     const MarkerCandidate& seed,
                                                     bool dark_on_light);

/// Direct least-squares ellipse fit (Fitzgibbon/Pilu/Fisher, numerically
/// stable split form). Returns nullopt for fewer than 5 points, degenerate or
/// non-elliptic conics.
std::optional<Ellipse> fit_ellipse(const std::vector<Eigen::Vector2d>& points);

}  // namespace alphashape
