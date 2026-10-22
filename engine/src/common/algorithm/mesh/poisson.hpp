#pragma once

// Screened Poisson surface reconstruction (Kazhdan & Hoppe, "Screened Poisson
// Surface Reconstruction", TOG 2013): the oriented point cloud defines an
// indicator function whose gradient matches the vector field formed by the
// points' normals, and the surface is the indicator's iso level set. Poisson
// solves a global Poisson equation on a regular grid, so it closes holes and
// resists noise far better than a local triangulation, while the screening term
// keeps the solution anchored to the input points.

#include <cstddef>

#include <Eigen/Core>

#include "common/algorithm/mesh/volume.hpp"
#include "model/mesh/point_cloud.hpp"

namespace alphashape::meshing {

struct PoissonOptions {
    /// Resolution of the longest bounding-box axis.
    int max_resolution = 128;
    /// Weight of the point-value screening term (relative to the divergence /
    /// gradient term).
    double screening = 4.0;
    /// Cap on conjugate-gradient iterations.
    int max_iterations = 500;
    /// Relative residual at which the conjugate-gradient solve stops.
    double tolerance = 1e-6;
    /// A voxel is marked observed when a point lies within this many voxels of it.
    double density_radius_voxels = 2.0;
};

/// Diagnostics of the most recent solve. The caller passes a pointer to the
/// trailing `stats` argument of screened_poisson() to receive them; nothing is
/// kept in global state, so two concurrent solves can never race on the result.
struct PoissonStats {
    int iterations = 0;              ///< conjugate-gradient iterations performed
    double relative_residual = 0.0;  ///< final ||r|| / ||b|| of the solve
};

/// Reconstruct a surface from `cloud` over [bbox_min, bbox_max]. Requires
/// cloud.normals (outward; empty normals are an error). The returned field
/// follows VolumeResult's sign convention (> iso outside, < iso inside).
/// When `stats` is non-null it receives the solve's convergence diagnostics.
VolumeResult screened_poisson(const PointCloud& cloud, const Eigen::Vector3d& bbox_min,
                              const Eigen::Vector3d& bbox_max, const PoissonOptions& options,
                              PoissonStats* stats = nullptr);

}  // namespace alphashape::meshing
