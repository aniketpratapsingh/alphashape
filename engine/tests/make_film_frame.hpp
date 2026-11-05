#pragma once

/// @file make_film_frame.hpp
/// @brief Synthetic analogue film scan for the WP-13.18a detector tests.
///
/// A dark scan background carries a bright exposed frame (an affine image of
/// the film rectangle, optionally rotated and mirrored) and dark fiducial marks
/// at chosen film millimetre positions. Rendering supersamples 4x4 per output
/// pixel and adds mild Gaussian noise from a fixed seed, so every test sees the
/// same image and a known film-mm -> pixel transform.

#include <array>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "image/image.hpp"
#include "model/camera/film.hpp"

namespace alphashape::test {

struct FilmFrameSpec {
    int width = 1600;
    int height = 1600;
    double frame_half_mm = 50.0;
    double scale_px_per_mm = 15.0;  ///< 1500 px across the frame
    double rotation_deg = 3.0;
    bool mirror = false;
    std::vector<Eigen::Vector2d> marks_mm;
    FiducialDetector mark_type = FiducialDetector::Circle;
    double mark_radius_mm = 0.4;
    double mark_arm_mm = 1.2;
    double background = 0.12;
    double frame = 0.85;
    double mark = 0.12;
    double noise = 0.01;
    std::uint32_t seed = 20261105u;
};

struct FilmFrame {
    FloatImage image;
    Eigen::Matrix3d transform = Eigen::Matrix3d::Identity();
    std::array<Eigen::Vector2d, 4> frame_corners{};  ///< TL, TR, BR, BL
    std::vector<Eigen::Vector2d> mark_pixels;        ///< parallel to marks_mm
};

FilmFrame render_film_frame(const FilmFrameSpec& spec);

}  // namespace alphashape::test
