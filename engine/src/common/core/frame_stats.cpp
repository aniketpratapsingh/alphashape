#include "common/core/frame_stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "common/core/chunk.hpp"

#if defined(ALPHASHAPE_USE_PROJ)
#include "model/crs/local_frame.hpp"
#endif

namespace alphashape {

namespace {

#if defined(ALPHASHAPE_USE_PROJ)
bool is_identity(const Eigen::Matrix4d& m) { return m == Eigen::Matrix4d::Identity(); }
#endif

/// The camera centre C = -A^-1 b for a pose stored as [A | b], where
/// X_camera = A X_world + b. The inverse is general because AlignChunks can
/// scale A with a similarity.
Eigen::Vector3d camera_centre(const Camera& camera) {
    const Eigen::Matrix3d a = camera.pose.block<3, 3>(0, 0);
    const Eigen::Vector3d b = camera.pose.block<3, 1>(0, 3);
    return -(a.inverse() * b);
}

double median(std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid),
                     values.end());
    const double upper = values[mid];
    if (values.size() % 2 != 0) {
        return upper;
    }
    const double lower =
        *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid));
    return 0.5 * (lower + upper);
}

}  // namespace

FrameStats compute_frame_stats(const Chunk& chunk) {
    FrameStats stats;
    stats.camera_count = chunk.cameras().size();
    stats.frame_count = chunk.frames().size();
    stats.group_count = chunk.camera_groups().size();

    std::vector<Eigen::Vector3d> centres;
    centres.reserve(chunk.cameras().size());
    for (const Camera& camera : chunk.cameras()) {
        if (!camera.enabled) {
            continue;
        }
        ++stats.enabled_count;
        ++stats.aligned_count;
        centres.push_back(camera_centre(camera));
    }

    if (!centres.empty()) {
        stats.has_extent = true;
        stats.extent_min = centres.front();
        stats.extent_max = centres.front();
        for (const Eigen::Vector3d& centre : centres) {
            stats.extent_min = stats.extent_min.cwiseMin(centre);
            stats.extent_max = stats.extent_max.cwiseMax(centre);
        }
#if defined(ALPHASHAPE_USE_PROJ)
        // A georeferenced chunk has a non-identity local->ECEF transform and a
        // CRS. Go through the same LocalFrame the reference table uses: the
        // transform alone yields ECEF metres, not the chunk CRS, and a chunk
        // with a CRS but an identity transform is still chunk-local (the CRS
        // is only a label until the chunk is georeferenced).
        const Eigen::Matrix4d& transform = chunk.transform();
        const crs::CoordinateSystem cs = chunk.coordinate_system();
        if (!is_identity(transform) && cs.is_set() && !cs.is_local()) {
            const crs::LocalFrame frame =
                crs::LocalFrame::with_rotation(cs, transform.block<3, 1>(0, 3),
                                               transform.block<3, 3>(0, 0));
            for (std::size_t i = 0; i < centres.size(); ++i) {
                const crs::CrsPoint point = frame.to_crs(centres[i]);
                const Eigen::Vector3d world(point.x, point.y, point.z);
                if (i == 0) {
                    stats.world_extent_min = world;
                    stats.world_extent_max = world;
                } else {
                    stats.world_extent_min = stats.world_extent_min.cwiseMin(world);
                    stats.world_extent_max = stats.world_extent_max.cwiseMax(world);
                }
            }
            stats.has_world_extent = true;
        }
#else
        (void)chunk;
#endif
    }

    if (centres.size() >= 2) {
        std::vector<double> nearest;
        nearest.reserve(centres.size());
        for (std::size_t i = 0; i < centres.size(); ++i) {
            double best = std::numeric_limits<double>::infinity();
            for (std::size_t j = 0; j < centres.size(); ++j) {
                if (i == j) {
                    continue;
                }
                best = std::min(best, (centres[i] - centres[j]).norm());
            }
            nearest.push_back(best);
        }
        stats.mean_capture_distance = 2.0 * median(nearest);
    }
    return stats;
}

}  // namespace alphashape
