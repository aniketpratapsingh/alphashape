#include "base/algorithm/voxel_grid.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace alphashape {

GridSpec make_grid(const Eigen::Vector3d& bbox_min, const Eigen::Vector3d& bbox_max,
                   int max_resolution) {
    if (max_resolution < 2) {
        throw ValueError("make_grid: max_resolution must be at least 2, got " +
                         std::to_string(max_resolution));
    }
    const Eigen::Vector3d extent = bbox_max - bbox_min;
    if (!(extent.x() > 0.0) || !(extent.y() > 0.0) || !(extent.z() > 0.0)) {
        throw ValueError("make_grid: the bounding box is empty or inverted");
    }

    // The longest axis gets exactly `max_resolution` voxels; the resulting cubic
    // voxel size then fixes the resolution of the other two axes.
    const double longest = std::max({extent.x(), extent.y(), extent.z()});
    const double voxel = longest / static_cast<double>(max_resolution);

    const auto axis_count = [voxel](double length) {
        const int n = static_cast<int>(std::ceil(length / voxel));
        return std::max(n, 1);
    };

    GridSpec spec;
    spec.origin = bbox_min;
    spec.voxel = voxel;
    spec.nx = axis_count(extent.x());
    spec.ny = axis_count(extent.y());
    spec.nz = axis_count(extent.z());
    return spec;
}

}  // namespace alphashape
