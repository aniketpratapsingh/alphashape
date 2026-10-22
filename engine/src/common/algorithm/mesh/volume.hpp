#pragma once

// The shared result type of the milestone-7 volumetric reconstruction stages
// (TV-L1 fusion, screened Poisson). Both produce a scalar field sampled on a
// regular grid plus a per-voxel "observed" mask; marching tetrahedra turns that
// pair into a mesh.
//
// SIGN CONVENTION (shared by every field producer): the field is NEGATIVE inside
// the surface and POSITIVE outside, so the surface is the `iso` level set and
// marching-tetrahedra faces are oriented with their normals pointing toward
// INCREASING field value — i.e. OUT of the surface.

#include <cstdint>
#include <vector>

#include "base/algorithm/voxel_grid.hpp"

namespace alphashape::meshing {

/// A reconstructed scalar field on a voxel grid, with the level set that defines
/// the surface and a mask of the voxels that data actually supports.
struct VolumeResult {
    /// Field values at voxel centres. > iso is outside / free space, < iso is
    /// inside the surface.
    VoxelGrid<float> field;
    /// The iso level defining the surface.
    float iso = 0.0F;
    /// spec.size(); 1 where the data supports the field, 0 where the value is an
    /// extrapolation that should be trimmed before meshing.
    std::vector<std::uint8_t> observed;
};

}  // namespace alphashape::meshing
