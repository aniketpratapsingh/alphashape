#pragma once

// Private seam between the CPU screened-Poisson solver (poisson.cpp) and its
// GPU port (poisson_gpu.cpp). Both paths must solve the SAME discrete system
// with the SAME right-hand side, and turn the solution into the SAME
// VolumeResult, so the assembly and the result finalisation live here rather
// than in either solver. Nothing outside src/common/algorithm/mesh includes
// this header.

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "base/algorithm/voxel_grid.hpp"
#include "common/algorithm/mesh/poisson.hpp"
#include "model/mesh/point_cloud.hpp"

namespace alphashape::meshing::detail {

/// The screened Poisson normal equations (G^T G + alpha_s W) g = rhs on the
/// voxel grid, everything the solvers need and nothing they do not.
struct PoissonSystem {
    GridSpec spec;
    std::vector<double> weight;  ///< per-voxel splat weight W (the screening term)
    std::vector<float> weight_f;  ///< weight as float, for the GPU stencil
    std::vector<double> rhs;     ///< G^T V = -div V
    std::vector<float> rhs_f;    ///< rhs as float, for the GPU vectors
    std::vector<double> diag;    ///< operator diagonal (Jacobi preconditioner)
    std::vector<float> inv_diag_f;  ///< 1 / max(diag, floor) as float
    std::vector<std::size_t> covered;  ///< flat indices of voxels with a splat
    double inv_h2 = 1.0;               ///< 1 / h^2
    double alpha_s = 0.0;              ///< screening weight
    double rhs_norm = 0.0;             ///< ||rhs||_2 in double

    /// out = A x with the 7-point Neumann stencil, in place-safe (out may not
    /// alias x). Deterministic under OpenMP.
    void apply(const std::vector<double>& x, std::vector<double>& out) const;

    /// Deterministic dot product: one z-slice per thread, added in z order.
    double dot(const std::vector<double>& a, const std::vector<double>& b) const;
};

/// Validates the inputs exactly as screened_poisson() does.
void validate_poisson_inputs(const PointCloud& cloud, const PoissonOptions& options);

/// Splats the normals, forms the right-hand side and the screening operator.
PoissonSystem assemble_poisson_system(const PointCloud& cloud,
                                      const Eigen::Vector3d& bbox_min,
                                      const Eigen::Vector3d& bbox_max,
                                      const PoissonOptions& options);

/// Field, iso level and observed mask from a solution vector.
VolumeResult finalize_poisson_result(const PoissonSystem& system,
                                     const std::vector<double>& solution,
                                     const PointCloud& cloud, const PoissonOptions& options);

/// Diagnostics of one solve (the same fields PoissonStats exposes publicly).
struct PoissonCgStats {
    int iterations = 0;
    double relative_residual = 0.0;
};

}  // namespace alphashape::meshing::detail
