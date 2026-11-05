#pragma once

/// @file film.hpp
/// @brief Fiducial marks on scanned analogue aerial film (WP-13.18a).
///
/// A metric film camera exposes fiducial marks at known positions on the film
/// (millimetres). A scan places those marks at pixel positions; a per-image
/// 2-D transform from film millimetres to scan pixels is the classical
/// analytical interior orientation (Kraus 2007, "Photogrammetry", section 3;
/// ASPRS Manual of Photogrammetry). This work package detects, orders, fits
/// and stores that transform. Applying it inside projection is WP-13.18b.
///
/// Film coordinates are millimetres with x to the right and y up, origin at
/// the centre of the fiducial layout (the film centre). Pixel coordinates are
/// in the engine's continuous image convention, where (0, 0) is the centre of
/// the top-left pixel and y grows downwards.

#include <string>
#include <vector>

#include <Eigen/Core>

namespace alphashape {

/// Shape of the per-scan film -> pixel map.
enum class FilmTransformType { Affine, Projective };

/// Kind of mark found on the film.
enum class FiducialDetector {
    FrameCorner,  ///< the four corners of the exposed frame itself
    Circle,       ///< a dot or ring mark
    Corner        ///< a right-angle L mark
};

/// One calibrated fiducial position on the film.
struct Fiducial {
    std::string label;
    Eigen::Vector2d film_mm = Eigen::Vector2d::Zero();

    bool operator==(const Fiducial& other) const {
        return label == other.label && film_mm == other.film_mm;
    }
};

/// Per-sensor film configuration.
struct FilmSettings {
    bool enabled = false;
    FilmTransformType transform = FilmTransformType::Affine;
    FiducialDetector detector = FiducialDetector::Circle;
    /// Search the four corner windows, the four side mid-point windows, or both.
    bool search_corners = true;
    bool search_sides = false;
    std::vector<Fiducial> fiducials;

    bool operator==(const FilmSettings& other) const {
        return enabled == other.enabled && transform == other.transform &&
               detector == other.detector &&
               search_corners == other.search_corners &&
               search_sides == other.search_sides && fiducials == other.fiducials;
    }
};

/// A named starting layout. Presets fill the detector, the search flags and
/// placeholder labels only: the coordinates are a nominal 55 mm ring and MUST
/// be replaced from the camera's calibration certificate or by
/// CalibrateFiducials. Throws ValueError for an unknown name.
FilmSettings film_preset(const std::string& name);

}  // namespace alphashape
