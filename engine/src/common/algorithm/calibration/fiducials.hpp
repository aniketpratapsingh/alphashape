#pragma once

/// @file fiducials.hpp
/// @brief Fiducial-mark detection and ordering on scanned analogue film
/// (WP-13.18a).
///
/// Pipeline, all our own implementations:
///   1. frame quad: Sobel gradient magnitude on a downsampled copy (longest
///      side <= 1500 px), first strong edge per image side, RANSAC line per
///      side (Fischler & Bolles 1981) and a least-squares refit on inliers;
///      adjacent lines are intersected into TL, TR, BR, BL.
///   2. search windows: four corner windows pulled inward along the diagonal
///      and four side windows pulled inward along the inward normal. The
///      pull/half-size fractions are ours (documented in the .cpp).
///   3. marks: circle centres by gradient-direction Hough voting (Hough 1962;
///      Ballard 1981), L/frame corners by a structure-tensor corner response
///      (Förstner & Gülch 1987). A shared least-squares sub-pixel refinement
///      solves sum w (n_i . (x - x_i))^2 with w = |g|^2; for circles the
///      normals are the gradient rotated 90 degrees, so the gradient lines
///      pass through the centre.
///   4. ordering: the eight dihedral relabelings of the sensor layout (four
///      rotations x mirror) are tried; the assignment with the lowest RMS wins.
///   5. transform: affine by linear least squares (>= 3 marks) or normalised
///      DLT homography (Hartley & Zisserman 2004, 4.1; >= 4 marks).
///
/// Coordinates are the engine's continuous image convention: (0, 0) is the
/// centre of the top-left pixel, y grows downwards. Film millimetres use
/// x right and y up, origin at the film centre.

#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "image/image.hpp"
#include "model/camera/film.hpp"

namespace alphashape {

/// Window a mark was searched in. Canonical order below is the order the
/// marks are reported in: TL, T, TR, R, BR, B, BL, L.
enum class FiducialWindow { TopLeft, Top, TopRight, Right, BottomRight, Bottom, BottomLeft, Left };

const char* fiducial_window_name(FiducialWindow window);

/// The exposed-frame quad, ordered top-left, top-right, bottom-right,
/// bottom-left in image coordinates.
struct FrameQuad {
    Eigen::Vector2d top_left = Eigen::Vector2d::Zero();
    Eigen::Vector2d top_right = Eigen::Vector2d::Zero();
    Eigen::Vector2d bottom_right = Eigen::Vector2d::Zero();
    Eigen::Vector2d bottom_left = Eigen::Vector2d::Zero();
};

/// Detector choices. The search flags normally come from `FilmSettings`; the
/// caller passes them here after resolving the task's overrides.
struct FiducialDetectOptions {
    FiducialDetector detector = FiducialDetector::Circle;
    bool search_corners = true;
    bool search_sides = false;
    double min_radius_px = 3.0;   ///< circle Hough radius range, pixels
    double max_radius_px = 40.0;
    /// Ordering rejects a best assignment whose RMS exceeds this, pixels.
    double max_residual_px = 3.0;
};

/// One mark, with the window that produced it.
struct FiducialMark {
    FiducialWindow window = FiducialWindow::TopLeft;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
};

/// Result of one scan. `failure` is empty on success.
struct FiducialDetection {
    std::vector<FiducialMark> marks;  ///< canonical window order
    /// Parallel to `FilmSettings::fiducials` when those coordinates exist;
    /// empty when the sensor still has no calibrated layout.
    std::vector<std::optional<Eigen::Vector2d>> by_fiducial;
    std::optional<Eigen::Matrix3d> transform;  ///< film mm -> scan pixel
    double rms_px = 0.0;
    std::vector<double> residuals_px;  ///< per assigned fiducial
    std::string failure;
};

/// Detect the exposed-frame quad. Returns nullopt when a side has too few
/// inliers or the four corners are not convex.
std::optional<FrameQuad> detect_frame_quad(const FloatImage& gray);

/// Detect, order and fit one scan. Detection failures are reported in
/// `FiducialDetection::failure` rather than thrown; invalid options throw
/// ValueError.
FiducialDetection detect_fiducials(const FloatImage& gray,
                                   const FilmSettings& settings,
                                   const FiducialDetectOptions& options = {});

/// Fit film mm -> pixel. Affine needs >= 3 correspondences, projective >= 4.
/// Returns false for too few points, non-finite input or a singular system.
/// `rms_px` is the root-mean-square reprojection error; `residuals` (when
/// non-null) receives the per-point errors.
bool fit_film_transform(const std::vector<Eigen::Vector2d>& film_mm,
                        const std::vector<Eigen::Vector2d>& pixels,
                        FilmTransformType type, Eigen::Matrix3d& transform,
                        double& rms_px,
                        std::vector<double>* residuals = nullptr);

}  // namespace alphashape
