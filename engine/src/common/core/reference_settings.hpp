#pragma once

/// @file reference_settings.hpp
/// @brief Chunk-level observation accuracies for georeferencing (WP-13.1, M13).
///
/// The settings are weights, not measurements: they say how much to trust each
/// class of observation in the adjustment and they supply the per-row sigma the
/// reference table reports when an object has no accuracy of its own. They live
/// on the chunk (`Chunk::reference_settings()`) and persist under a
/// `<reference>` element in doc.xml.
///
/// Defaults are ours, chosen for a typical consumer-GNSS drone flight and a
/// surveyed handful of control points. The attribute names are the public
/// Python API spellings so scripts that set them keep working.
///
/// This header sits in `core` because `Chunk` owns the value; the reference
/// table and command live one layer up and include it from here.

namespace alphashape {

struct ReferenceSettings {
    /// 1-sigma position accuracy of an unweighted camera reference, metres.
    double camera_location_accuracy_m = 10.0;
    /// 1-sigma rotation accuracy of an unweighted camera reference, degrees.
    double camera_rotation_accuracy_deg = 10.0;
    /// 1-sigma position accuracy of an unweighted marker reference, metres.
    double marker_location_accuracy_m = 0.005;
    /// 1-sigma image-space accuracy of a marker projection, pixels.
    double marker_projection_accuracy_px = 0.5;
    /// 1-sigma length accuracy of a scale bar, metres.
    double scalebar_accuracy_m = 0.001;
    /// 1-sigma tie-point accuracy, pixels.
    double tiepoint_accuracy_px = 1.0;
    /// Whether marker references act as control points (false turns every
    /// marker into a check point for the adjustment).
    bool use_gcp = true;

    /// True when every field equals the default, i.e. when the writer may
    /// omit the `<reference>` element entirely.
    bool is_default() const { return *this == ReferenceSettings{}; }

    bool operator==(const ReferenceSettings&) const = default;
};

}  // namespace alphashape
