#pragma once

// Marching tetrahedra: extract the iso-surface of a scalar field sampled at
// voxel centres.
//
// Each cube of eight neighbouring centres is split into six tetrahedra around a
// consistent main diagonal, and each tetrahedron is polygonised independently.
// Splitting every cube the same way means shared faces are cut identically by
// both of their cubes, so the surface is crack-free. New vertices lie on tetra
// edge segments by linear interpolation of the field and are SHARED between the
// cubes that meet along a grid edge (one vertex per grid edge crossing).
//
// Orientation: face normals point toward INCREASING field value, so with the
// VolumeResult convention (positive = outside) the outward normal is correct.

#include <cstdint>
#include <vector>

#include "base/algorithm/voxel_grid.hpp"
#include "model/mesh/mesh.hpp"

namespace alphashape {

/// Extract the `iso` level set of `field` as a triangle mesh.
///
/// `mask`, when non-null, must have spec.size() entries; a cube contributes only
/// when all eight of its corner voxels have a non-zero mask value. Use it to
/// trim extrapolated regions flagged unobserved by a fusion stage.
Mesh extract_isosurface(const VoxelGrid<float>& field, float iso,
                        const std::vector<std::uint8_t>* mask = nullptr);

}  // namespace alphashape
