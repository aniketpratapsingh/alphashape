#pragma once

// Cross-view depth-map consistency filter. Each depth of the reference map is
// reprojected into the neighbour views; it survives when enough neighbours
// agree (within a relative depth tolerance and a reprojection tolerance), and
// small unsupported islands are removed by a speckle pass. The retained count
// becomes the pixel's confidence.

#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "common/core/chunk.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::stereo {

enum class FilterMode { Disabled, Mild, Moderate, Aggressive };

/// "disabled"/"mild"/"moderate"/"aggressive" (case-insensitive); anything else
/// throws ValueError.
FilterMode parse_filter_mode(const std::string& text);
std::string to_string(FilterMode mode);

struct DepthView {
    const DepthMap* map = nullptr;
    Calibration calibration;      // scaled to the map's resolution
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

struct FilterOptions {
    /// Mirrors the BuildDepthMaps default: "moderate" is what the benchmark
    /// settled on (21% less hole area than "mild" at no time cost), and a
    /// default-constructed policy must not quietly ask for the weaker filter.
    FilterMode mode = FilterMode::Moderate;
    double relative_depth_tolerance = 0.01;
    double reprojection_tolerance_px = 1.0;
};

struct FilterStats { std::size_t input = 0, supported = 0, rejected_support = 0, rejected_speckle = 0; };

/// Returns a filtered copy of *reference.map; confidence[] set to the support count.
DepthMap filter_depth_map(const DepthView& reference, const std::vector<DepthView>& neighbours,
                          const FilterOptions& options, FilterStats* stats = nullptr);

}  // namespace alphashape::stereo
