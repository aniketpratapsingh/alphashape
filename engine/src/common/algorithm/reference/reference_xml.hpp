#pragma once

/// @file reference_xml.hpp
/// @brief The reference XML document used by ImportReference/ExportReference.
///
/// The format is the common photogrammetry reference XML:
///
///     <reference>
///       <cameras>
///         <camera label="IMG_0001" x="..." y="..." z="..." sxyz="..." enabled="1"/>
///       </cameras>
///       <markers>
///         <marker label="GCP1" x="..." y="..." z="..." sxyz="..." enabled="1"/>
///       </markers>
///       <scalebars>
///         <scalebar label="S1" distance="..." accuracy="..." enabled="1"/>
///       </scalebars>
///     </reference>
///
/// Only `label`, `x`, `y` and `z` are required on a camera/marker; `sxyz` is a
/// single 1-sigma accuracy and `enabled` defaults to true. Scale bars are
/// optional and carry a measured `distance` (metres) and `accuracy`.
///
/// Coordinates are in whichever CRS the caller's `crs` argument names; this
/// module does not transform anything.

#include <string>
#include <vector>

#include <Eigen/Core>

namespace alphashape::reference {

struct XmlReferencePosition {
    std::string label;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    /// One 1-sigma figure shared by all three axes; 0 means "not stated", in
    /// which case the importer keeps the chunk's default accuracy.
    double accuracy = 0.0;
    bool enabled = true;
};

struct XmlReferenceScalebar {
    std::string label;
    double distance = 0.0;
    double accuracy = 0.0;
    bool enabled = true;
};

struct XmlReferenceDocument {
    std::vector<XmlReferencePosition> cameras;
    std::vector<XmlReferencePosition> markers;
    std::vector<XmlReferenceScalebar> scalebars;
};

/// Parse a reference XML document.
///
/// @throws ValueError when the root is not <reference>, a required attribute is
///         absent, or a number is malformed.
XmlReferenceDocument parse_reference_xml(const std::string& text);

/// Serialise `document` with two-space indentation and LF line endings.
std::string write_reference_xml(const XmlReferenceDocument& document);

}  // namespace alphashape::reference
