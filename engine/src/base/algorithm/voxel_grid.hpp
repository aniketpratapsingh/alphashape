#pragma once

// A regular voxel grid over an axis-aligned box — the common substrate for the
// milestone-7 volumetric stages (marching tetrahedra, TV-L1 fusion, screened
// Poisson).
//
// Indexing convention: values live at voxel CENTRES. The centre of voxel
// (x, y, z) is
//
//     centre = origin + voxel * (x + 0.5, y + 0.5, z + 0.5)
//
// and x is the fastest-varying axis, so a flat index is (z*ny + y)*nx + x. The
// origin is the *corner* of the box, i.e. voxel (0,0,0) is centred half a voxel
// inside it.

#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"

namespace alphashape {

/// The geometry of a regular voxel grid: an origin corner, a cubic voxel size,
/// and a resolution along each axis.
struct GridSpec {
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    double voxel = 1.0;  ///< edge length of a cubic voxel
    int nx = 0;
    int ny = 0;
    int nz = 0;

    std::size_t size() const {
        return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
               static_cast<std::size_t>(nz);
    }

    /// Flat index of voxel (x, y, z); x fastest.
    std::size_t index(int x, int y, int z) const {
        return (static_cast<std::size_t>(z) * static_cast<std::size_t>(ny) +
                static_cast<std::size_t>(y)) *
                   static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(x);
    }

    /// World-space centre of voxel (x, y, z).
    Eigen::Vector3d centre(int x, int y, int z) const {
        return origin + voxel * Eigen::Vector3d(static_cast<double>(x) + 0.5,
                                                static_cast<double>(y) + 0.5,
                                                static_cast<double>(z) + 0.5);
    }

    bool contains(int x, int y, int z) const {
        return x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz;
    }
};

/// Build a grid of cubic voxels covering [bbox_min, bbox_max]. The longest
/// bounding-box axis is divided into `max_resolution` voxels; each other axis
/// gets ceil(extent / voxel) voxels, at least one. Throws ValueError for a
/// resolution below 2 or an empty / inverted box.
GridSpec make_grid(const Eigen::Vector3d& bbox_min, const Eigen::Vector3d& bbox_max,
                   int max_resolution);

/// A GridSpec plus one value per voxel, in flat (x fastest) order.
template <typename T>
struct VoxelGrid {
    GridSpec spec;
    std::vector<T> data;  ///< spec.size() entries

    VoxelGrid() = default;
    explicit VoxelGrid(const GridSpec& s, T fill = T{}) : spec(s), data(s.size(), fill) {}

    T& at(int x, int y, int z) { return data[spec.index(x, y, z)]; }
    const T& at(int x, int y, int z) const { return data[spec.index(x, y, z)]; }
};

}  // namespace alphashape
