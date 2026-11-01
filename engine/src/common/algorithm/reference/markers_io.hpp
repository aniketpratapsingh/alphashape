#pragma once

/// @file markers_io.hpp
/// @brief CSV and XML codecs for marker projections (WP-13.4).
///
/// The CSV form is one row per projection:
///
///     marker,camera,u,v
///     GCP1,IMG_0001,1234.5,987.25
///
/// `camera` names a camera by its exact label. The XML form nests projections
/// inside their marker and carries the optional chunk-local position:
///
///     <markers>
///       <marker label="GCP1" x="1" y="2" z="3">
///         <projection camera="IMG_0001" u="1234.5" v="987.25"/>
///       </marker>
///     </markers>
///
/// Neither codec resolves camera labels or touches a chunk; that is the
/// importer's/exporters' job. Numbers use the same locale-independent
/// conversion as the reference import/export (WP-13.2), and the writer emits
/// enough digits for an exact double round-trip.

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace alphashape::reference {

enum class MarkerIoFormat { Csv, Xml };

/// Resolve `format` (empty infers from `path`'s extension). `task` prefixes the
/// error message, e.g. "ImportMarkers".
MarkerIoFormat marker_io_format(const std::string& format, const std::string& path,
                                const char* task);

/// One image measurement, with the physical line it came from (0 when the
/// format has no line information, as with XML).
struct MarkerIoProjection {
    std::string camera;
    Eigen::Vector2d uv = Eigen::Vector2d::Zero();
    std::size_t line = 0;
};

/// A marker with all of its projections, grouped from the flat CSV or read from
/// one <marker> element.
struct MarkerIoMarker {
    std::string label;
    std::optional<Eigen::Vector3d> position;
    std::vector<MarkerIoProjection> projections;
    std::size_t line = 0;
};

/// Parse the CSV form. A header row naming marker/camera/u/v in any order is
/// accepted; without one the four columns are read in that fixed order.
///
/// @throws ValueError naming the line for a malformed row, a short row or a
///         non-finite coordinate.
std::vector<MarkerIoMarker> parse_marker_csv(const std::string& text);

/// Parse the XML form.
///
/// @throws ValueError on a wrong root element, a missing required attribute, a
///         duplicate marker label or a non-finite coordinate.
std::vector<MarkerIoMarker> parse_marker_xml(const std::string& text);

/// Serialise the CSV form. `precision` is the number of fractional digits; 17
/// is the exact-round-trip choice.
std::string write_marker_csv(const std::vector<MarkerIoMarker>& markers, int precision);

/// Serialise the XML form with two-space indentation and LF line endings.
std::string write_marker_xml(const std::vector<MarkerIoMarker>& markers);

}  // namespace alphashape::reference
