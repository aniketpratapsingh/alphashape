#pragma once

// Stereo view selection: for each aligned camera, choose the neighbours that
// make useful depth-map references. A neighbour is useful when it shares enough
// tie points with the reference and its baseline subtends a triangulation angle
// inside the configured window — too small gives no depth precision, too large
// gives poor overlap and matching.

#include <cstdint>
#include <vector>

#include "common/core/chunk.hpp"

namespace alphashape::stereo {

struct ViewSelectionOptions {
    int max_neighbors = 8;
    int min_common_points = 10;
    double min_angle_deg = 2.0;    // triangulation-angle window for a useful neighbour
    double max_angle_deg = 45.0;
};

struct ViewNeighbourhood {
    std::int32_t camera_id = -1;
    std::vector<std::int32_t> neighbours;  // camera ids, best first
    std::vector<double> scores;            // parallel to neighbours
    double depth_min = 0.0;                // range along the bearing, chunk-local units
    double depth_max = 0.0;
};

/// One entry per aligned (enabled) camera that has >= 1 neighbour, in camera order.
std::vector<ViewNeighbourhood> select_views(const Chunk& chunk, const ViewSelectionOptions& options);

}  // namespace alphashape::stereo
