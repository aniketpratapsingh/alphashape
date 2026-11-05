#pragma once

/// @file film_geometry.hpp
/// @brief The canonical film-normalised pixel frame (WP-13.18b).
///
/// Each scanned film is resampled once, at load time, into a frame whose pixel
/// grid is metric. Pixel (0.5, 0.5) is the centre of the top-left pixel (the
/// engine's image convention), x grows right, y grows down, and one pixel spans
/// `resolution_mm_per_px` millimetres. The film origin is the layout centre:
///
///     x = (u - W/2) * res,   y = -(v - H/2) * res
///
/// Downstream every stage sees an ordinary Image and an ordinary pixel
/// Calibration; the scan's affine/projective film_transform stays outside the
/// lens model. A film scan cannot be used by the pipeline until both the
/// resolution and the extent are known.

#include <optional>
#include <utility>

#include <Eigen/Core>

#include "image/image.hpp"
#include "model/camera/film.hpp"

namespace alphashape {

/// Pixel dimensions of the normalised image (width, height).
using FilmSize = std::pair<int, int>;

/// True when `film` carries a positive resolution and a positive extent, so a
/// normalised image can be built.
bool film_geometry_complete(const FilmSettings& film);

/// Normalised-image extent in millimetres: the stored size when set, otherwise
/// the bounding box of the fiducial layout. Zero when neither is available.
Eigen::Vector2d film_extent_mm(const FilmSettings& film);

/// Normalised image size: round(extent / resolution) per axis. Throws
/// ValueError when the settings are incomplete or the result is empty.
FilmSize film_normalised_size(const FilmSettings& film);

/// Normalised pixel -> film millimetres (x right, y up).
Eigen::Vector2d film_mm_from_normalised(const FilmSettings& film,
                                        const Eigen::Vector2d& normalised);

/// Film millimetres -> normalised pixel.
Eigen::Vector2d normalised_from_film_mm(const FilmSettings& film,
                                        const Eigen::Vector2d& film_mm);

/// Normalised pixel -> scan pixel through the per-scan film transform, with the
/// homogeneous divide. Returns nullopt when the projected point is not a finite
/// Euclidean point (a projective map's horizon).
std::optional<Eigen::Vector2d> scan_pixel_from_normalised(
    const Eigen::Matrix3d& film_transform, const FilmSettings& film,
    const Eigen::Vector2d& normalised);

/// Resample a scan into the normalised frame with bilinear interpolation.
/// A downscale of more than 1.5 scan px per normalised px is prefiltered with a
/// Gaussian of sigma = 0.5 * factor to stop aliasing; samples outside the scan
/// are 0. The output keeps the scan's channel count.
Image resample_film_image(const Image& scan, const Eigen::Matrix3d& film_transform,
                          const FilmSettings& film);

}  // namespace alphashape
