#include "model/camera/film.hpp"

#include <string>
#include <utility>
#include <vector>

#include "base/utils/error.hpp"

namespace alphashape {
namespace {

/// Nominal half-pitch of the starting presets, millimetres. It is deliberately
/// a round number, not a camera-specific one: a preset is a labelled scaffold
/// that the user replaces with the certificate values (or that
/// CalibrateFiducials derives from a scan).
constexpr double kPresetHalfMm = 55.0;

Fiducial mark(std::string label, double x, double y) {
    Fiducial fiducial;
    fiducial.label = std::move(label);
    fiducial.film_mm = Eigen::Vector2d(x, y);
    return fiducial;
}

std::vector<Fiducial> corner_marks() {
    return {mark("corner_tl", -kPresetHalfMm, kPresetHalfMm),
            mark("corner_tr", kPresetHalfMm, kPresetHalfMm),
            mark("corner_br", kPresetHalfMm, -kPresetHalfMm),
            mark("corner_bl", -kPresetHalfMm, -kPresetHalfMm)};
}

std::vector<Fiducial> side_marks() {
    return {mark("side_top", 0.0, kPresetHalfMm),
            mark("side_right", kPresetHalfMm, 0.0),
            mark("side_bottom", 0.0, -kPresetHalfMm),
            mark("side_left", -kPresetHalfMm, 0.0)};
}

}  // namespace

FilmSettings film_preset(const std::string& name) {
    FilmSettings settings;
    settings.enabled = true;
    if (name == "corners4") {
        settings.detector = FiducialDetector::Corner;
        settings.search_corners = true;
        settings.search_sides = false;
        settings.fiducials = corner_marks();
        return settings;
    }
    if (name == "sides4") {
        settings.detector = FiducialDetector::Circle;
        settings.search_corners = false;
        settings.search_sides = true;
        settings.fiducials = side_marks();
        return settings;
    }
    if (name == "all8") {
        settings.detector = FiducialDetector::Circle;
        settings.search_corners = true;
        settings.search_sides = true;
        const std::vector<Fiducial> corners = corner_marks();
        const std::vector<Fiducial> sides = side_marks();
        // Canonical window order TL, T, TR, R, BR, B, BL, L.
        settings.fiducials = {corners[0], sides[0], corners[1], sides[1],
                              corners[2], sides[2], corners[3], sides[3]};
        return settings;
    }
    throw ValueError("film_preset: unknown preset '" + name + "'");
}

}  // namespace alphashape
