#pragma once

/// @file board_detector.hpp
/// @brief Chessboard calibration-target detection (WP-13.14).
///
/// Finds a printed planar chessboard in a photo, verifies its inner X-junctions
/// and labels them with integer (col, row) lattice coordinates. A later work
/// package (WP-13.15) turns these labels into a Zhang-style camera calibration.
///
/// Pixel convention: results use the calibration convention, where the centre
/// of the top-left pixel is (0.5, 0.5) and the image spans [0, W] x [0, H] --
/// the same convention as `Calibration`. `FloatImage::sample()` addresses pixel
/// centres at integer coordinates, so the detector converts by half a pixel.
///
/// Partial views carry a per-view unknown lattice origin: labels are relative
/// and, once the minimum on each axis is shifted to zero, are only defined up
/// to a translation (WP-13.15 resolves the per-view offset). The label axes
/// are always right-handed in image coordinates (X right, Y down) and are
/// never transposed relative to the spec extents, so lattice differences are
/// comparable across views. A calibration solve that mixes partial views must
/// account for the per-view origin.
///
/// Only a complete view (every inner corner of a `cols` x `rows` board) has
/// canonical labels. The origin is the corner whose diagonally-outside square
/// is dark. When `cols + rows` is odd the board is colour-asymmetric under a
/// 180 degree rotation, so that rule pins the labelling uniquely; when
/// `cols + rows` is even the board looks identical after 180 degrees, so the
/// tie is broken deterministically by choosing whichever of the two diagonal
/// corners (0, 0) / (cols - 1, rows - 1) lies nearer the image top-left.
/// Square boards (`cols == rows`) are additionally ambiguous under 90 degrees;
/// they use the same deterministic tie-break and are therefore not guaranteed
/// to agree across views. Every complete view is asserted to span exactly
/// `[0, cols) x [0, rows)`.

#include <optional>
#include <vector>

#include <Eigen/Core>

#include "image/image.hpp"

namespace alphashape {

/// Physical board layout. `cols` and `rows` count INNER corners.
struct BoardSpec {
    int cols = 9;
    int rows = 6;
    double square_size_m = 0.025;
};

/// Tunables. All pixel distances are in the calibration convention (top-left
/// pixel centre at (0.5, 0.5)).
struct BoardDetectOptions {
    double sigma_px = 1.5;         ///< Gaussian smoothing before the response.
    double min_distance_px = 5.0;  ///< NMS suppression radius between candidates.
    double response_fraction = 0.1; ///< Cut relative to the 99th-percentile response.
    double ring_radius_px = 5.0;   ///< Radius of the X-junction verification ring.
    double angle_tol_deg = 15.0;   ///< Edge-direction agreement tolerance.
    double spacing_tol = 0.35;     ///< Relative tolerance on the lattice spacing.
    int refine_radius = 4;         ///< Half-size of the saddle-refinement window.
    int min_corners = 12;          ///< Minimum number of labelled corners.
};

/// One labelled inner corner. `pixel` is in calibration coordinates.
struct BoardCorner {
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
    int col = 0;
    int row = 0;
    double response = 0.0;  ///< Detector strength, larger is stronger.
};

/// A detected board. `complete` is true when every inner corner of the spec was
/// found, in which case the labels are canonical; otherwise the labels are
/// relative to an arbitrary per-view origin.
struct BoardView {
    std::vector<BoardCorner> corners;
    bool complete = false;
    double square_size_m = 0.025;  ///< Used by board_points().

    /// Board-plane coordinates (col * square, row * square) in metres, z = 0,
    /// in the same order as `corners`.
    std::vector<Eigen::Vector2d> board_points() const;
};

/// Detect a chessboard. Returns nullopt when no valid lattice of at least
/// `min_corners` corners is found. Throws ValueError for invalid options or a
/// spec smaller than 2 x 2.
std::optional<BoardView> detect_board(const FloatImage& gray, const BoardSpec& spec,
                                      const BoardDetectOptions& options = {});

}  // namespace alphashape
