// Screened Poisson surface reconstruction (Kazhdan, Bolitho & Hoppe 2006;
// Kazhdan & Hoppe 2013), implemented from scratch on a regular voxel grid.
//
// DISCRETISATION
// --------------
// The oriented points define a vector field V (splatted trilinearly into the
// voxel grid, 3 doubles per voxel) whose direction is the surface normal and
// whose integral over the surface layer is 1. We look for the scalar field g
// minimising
//
//     E(g) = sum_voxels h^3 |(G g)_i - V_i|^2 + alpha_s * sum_voxels h^3 W_i g_i^2
//
// where G is the forward-difference gradient and W the per-voxel splat weight.
// The normal equations are
//
//     (G^T G + alpha_s W) g = G^T V = -div V
//
// with div the BACKWARD difference, so G^T G = -div grad is exactly the
// 7-point Neumann Laplacian L g = (n_i g_i - sum_j g_j) / h^2 over in-grid
// neighbours only (the natural zero-flux boundary of Poisson reconstruction).
// The point-free lowest frequency of L is fixed by the screening term, and the
// screened operator is symmetric positive definite.
//
// The solution g is positive outside the surface and negative inside it, i.e. it
// follows VolumeResult's sign convention with iso being the mean of g sampled at
// the input points.

#include "common/algorithm/mesh/poisson.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "base/algorithm/voxel_grid.hpp"
#include "base/utils/error.hpp"
#include "common/algorithm/mesh/poisson_internal.hpp"
#include "common/algorithm/mesh/poisson_gpu.hpp"

namespace alphashape::meshing {
namespace detail {

/// Reject inputs the reconstruction cannot handle.
void validate_poisson_inputs(const PointCloud& cloud, const PoissonOptions& options) {
    if (cloud.points.empty()) {
        throw ValueError("screened_poisson: point cloud is empty");
    }
    if (cloud.normals.size() != cloud.points.size()) {
        throw ValueError("screened_poisson: cloud.normals must hold one normal per point");
    }
    if (options.max_resolution < 4) {
        throw ValueError("screened_poisson: max_resolution must be at least 4");
    }
    if (options.screening < 0.0) {
        throw ValueError("screened_poisson: screening must not be negative");
    }
    if (options.max_iterations < 1) {
        throw ValueError("screened_poisson: max_iterations must be at least 1");
    }
    if (options.tolerance <= 0.0) {
        throw ValueError("screened_poisson: tolerance must be positive");
    }
    if (options.density_radius_voxels < 0.0) {
        throw ValueError("screened_poisson: density_radius_voxels must not be negative");
    }
}

namespace {

/// The trilinearly splatted vector field: `value` accumulates the outward
/// normals of the points that fall in each voxel's neighbourhood and `weight`
/// accumulates the splat weights (the per-voxel point density). Each point
/// distributes its normal over the 8 voxel centres surrounding it with the
/// trilinear weights (which sum to 1), so `weight` is ~1 per fully covered
/// voxel. Both sums are plain doubles, and the loop is serial and in input
/// order, so the field is bit-for-bit reproducible.
struct SplatField {
    std::vector<Eigen::Vector3d> value;  ///< sum_p w_p * n_p, per voxel
    std::vector<double> weight;          ///< sum_p w_p, per voxel
    double mean_weight = 0.0;            ///< mean of weight over voxels with weight > 0
    std::vector<std::size_t> covered;    ///< flat indices of voxels with weight > 0
};

SplatField splat_points(const GridSpec& spec, const PointCloud& cloud) {
    SplatField field;
    field.value.assign(spec.size(), Eigen::Vector3d::Zero());
    field.weight.assign(spec.size(), 0.0);
    const double inv_h = 1.0 / spec.voxel;
    for (std::size_t p = 0; p < cloud.points.size(); ++p) {
        // Continuous voxel-centre coordinates: voxel (x,y,z) sits at q = (x,y,z).
        const Eigen::Vector3d q =
            (cloud.points[p] - spec.origin) * inv_h - Eigen::Vector3d::Constant(0.5);
        const Eigen::Vector3d base_d = q.array().floor().matrix();
        const Eigen::Vector3d frac = q - base_d;
        const int bx = static_cast<int>(base_d.x());
        const int by = static_cast<int>(base_d.y());
        const int bz = static_cast<int>(base_d.z());
        const double wx[2] = {1.0 - frac.x(), frac.x()};
        const double wy[2] = {1.0 - frac.y(), frac.y()};
        const double wz[2] = {1.0 - frac.z(), frac.z()};
        const Eigen::Vector3d normal = cloud.normals[p].cast<double>();
        for (int dz = 0; dz < 2; ++dz) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const int x = bx + dx;
                    const int y = by + dy;
                    const int z = bz + dz;
                    if (!spec.contains(x, y, z)) {
                        continue;
                    }
                    const double w = wx[dx] * wy[dy] * wz[dz];
                    const std::size_t idx = spec.index(x, y, z);
                    field.value[idx] += w * normal;
                    field.weight[idx] += w;
                }
            }
        }
    }
    double total = 0.0;
    for (std::size_t i = 0; i < field.weight.size(); ++i) {
        if (field.weight[i] > 0.0) {
            total += field.weight[i];
            field.covered.push_back(i);
        }
    }
    if (!field.covered.empty()) {
        field.mean_weight = total / static_cast<double>(field.covered.size());
    }
    return field;
}

/// b = div V, a BACKWARD difference divided by the voxel size, with V = 0
/// outside the grid (V[-1] = 0 on every axis). This is the negative of the
/// gradient operator's adjoint, i.e. b = -G^T V, which is exactly the right-hand
/// side of the screened normal equations (G^T G + alpha W) g = G^T V.
std::vector<double> backward_divergence(const GridSpec& spec,
                                        const std::vector<Eigen::Vector3d>& v) {
    const double inv_h = 1.0 / spec.voxel;
    const std::size_t plane = static_cast<std::size_t>(spec.nx) * static_cast<std::size_t>(spec.ny);
    std::vector<double> b(spec.size(), 0.0);
    for (int z = 0; z < spec.nz; ++z) {
        for (int y = 0; y < spec.ny; ++y) {
            for (int x = 0; x < spec.nx; ++x) {
                const std::size_t idx = spec.index(x, y, z);
                const Eigen::Vector3d& here = v[idx];
                const double left = x > 0 ? v[idx - 1].x() : 0.0;
                const double back =
                    y > 0 ? v[idx - static_cast<std::size_t>(spec.nx)].y() : 0.0;
                const double down = z > 0 ? v[idx - plane].z() : 0.0;
                b[idx] = ((here.x() - left) + (here.y() - back) + (here.z() - down)) * inv_h;
            }
        }
    }
    return b;
}

/// Neighbour count of voxel (x, y, z) inside the grid (3 .. 6).
int neighbour_count(const GridSpec& spec, int x, int y, int z) {
    int n = 0;
    if (x > 0) ++n;
    if (x + 1 < spec.nx) ++n;
    if (y > 0) ++n;
    if (y + 1 < spec.ny) ++n;
    if (z > 0) ++n;
    if (z + 1 < spec.nz) ++n;
    return n;
}

}  // namespace

/// The screened Poisson operator A = -Delta + alpha_s * W applied matrix-free on
/// the flattened voxel grid. `-Delta` is the 7-point Neumann Laplacian
/// (n_i g_i - sum_j g_j) / h^2 summed over the in-grid neighbours j of i, so a
/// voxel whose neighbourhood leaves the grid simply has fewer neighbours (the
/// zero-flux boundary condition of Poisson reconstruction). W is the per-voxel
/// splat weight. A is symmetric positive definite for any alpha_s >= 0.
void PoissonSystem::apply(const std::vector<double>& g, std::vector<double>& out) const {
    const GridSpec& s = spec;
    const std::vector<double>& w = weight;
    const std::size_t plane = static_cast<std::size_t>(s.nx) * static_cast<std::size_t>(s.ny);
    const int nx = s.nx;
    const int ny = s.ny;
    const int nz = s.nz;
    const double a = alpha_s;
    const double ih2 = inv_h2;
    out.resize(s.size());
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const std::size_t idx = s.index(x, y, z);
                const double g0 = g[idx];
                int n = 0;
                double sum = 0.0;
                if (x > 0) {
                    sum += g[idx - 1];
                    ++n;
                }
                if (x + 1 < nx) {
                    sum += g[idx + 1];
                    ++n;
                }
                if (y > 0) {
                    sum += g[idx - static_cast<std::size_t>(nx)];
                    ++n;
                }
                if (y + 1 < ny) {
                    sum += g[idx + static_cast<std::size_t>(nx)];
                    ++n;
                }
                if (z > 0) {
                    sum += g[idx - plane];
                    ++n;
                }
                if (z + 1 < nz) {
                    sum += g[idx + plane];
                    ++n;
                }
                out[idx] = (static_cast<double>(n) * g0 - sum) * ih2 + a * w[idx] * g0;
            }
        }
    }
}

/// Deterministic dot product: each z-slice is summed by a single thread into its
/// own slot, then the slots are added in z order.
double PoissonSystem::dot(const std::vector<double>& a, const std::vector<double>& b) const {
    const std::size_t plane = static_cast<std::size_t>(spec.nx) * static_cast<std::size_t>(spec.ny);
    const int nz = spec.nz;
    std::vector<double> partial(static_cast<std::size_t>(nz), 0.0);
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int z = 0; z < nz; ++z) {
        const std::size_t begin = static_cast<std::size_t>(z) * plane;
        const std::size_t end = begin + plane;
        double s = 0.0;
        for (std::size_t i = begin; i < end; ++i) {
            s += a[i] * b[i];
        }
        partial[static_cast<std::size_t>(z)] = s;
    }
    double total = 0.0;
    for (int z = 0; z < nz; ++z) {
        total += partial[static_cast<std::size_t>(z)];
    }
    return total;
}

namespace {

/// Jacobi-preconditioned conjugate gradients for A g = rhs. Deterministic: all
/// reductions go through PoissonSystem::dot.
std::vector<double> conjugate_gradients(const PoissonSystem& a, const std::vector<double>& rhs,
                                        double tolerance, int max_iterations,
                                        PoissonCgStats& stats) {
    const std::size_t n = rhs.size();
    std::vector<double> x(n, 0.0);
    std::vector<double> r = rhs;
    std::vector<double> z(n, 0.0);
    std::vector<double> p(n, 0.0);
    std::vector<double> q(n, 0.0);
    const double rhs_norm = std::sqrt(a.dot(rhs, rhs));
    stats.iterations = 0;
    stats.relative_residual = rhs_norm > 0.0 ? 1.0 : 0.0;
    if (rhs_norm == 0.0) {
        return x;
    }
    const double floor_diag = 1e-30 * a.inv_h2;
    auto precondition = [&]() {
        for (std::size_t i = 0; i < n; ++i) {
            z[i] = r[i] / std::max(a.diag[i], floor_diag);
        }
    };
    precondition();
    p = z;
    double rho = a.dot(r, z);
    for (int it = 0; it < max_iterations; ++it) {
        a.apply(p, q);
        const double pq = a.dot(p, q);
        if (!(pq > 0.0)) {
            break;  // p is (numerically) in the operator's null space.
        }
        const double alpha = rho / pq;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        const double r_norm = std::sqrt(a.dot(r, r));
        stats.iterations = it + 1;
        stats.relative_residual = r_norm / rhs_norm;
        if (stats.relative_residual < tolerance) {
            break;
        }
        precondition();
        const double rho_next = a.dot(r, z);
        const double beta = rho_next / rho;
        for (std::size_t i = 0; i < n; ++i) {
            p[i] = z[i] + beta * p[i];
        }
        rho = rho_next;
    }
    return x;
}

/// Trilinear sample of the field at a world-space position, clamping to the
/// nearest voxel centre outside the grid. Voxel (x,y,z) sits at continuous
/// coordinate (x,y,z), so the sample coordinate is q = (p - origin)/h - 0.5.
double sample_trilinear(const GridSpec& spec, const std::vector<double>& g,
                        const Eigen::Vector3d& p) {
    const Eigen::Vector3d q = (p - spec.origin) / spec.voxel - Eigen::Vector3d::Constant(0.5);
    const Eigen::Vector3d base_d = q.array().floor().matrix();
    const Eigen::Vector3d frac = q - base_d;
    const int bx = static_cast<int>(base_d.x());
    const int by = static_cast<int>(base_d.y());
    const int bz = static_cast<int>(base_d.z());
    const double wx[2] = {1.0 - frac.x(), frac.x()};
    const double wy[2] = {1.0 - frac.y(), frac.y()};
    const double wz[2] = {1.0 - frac.z(), frac.z()};
    double acc = 0.0;
    for (int dz = 0; dz < 2; ++dz) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                const int x = std::min(std::max(bx + dx, 0), spec.nx - 1);
                const int y = std::min(std::max(by + dy, 0), spec.ny - 1);
                const int z = std::min(std::max(bz + dz, 0), spec.nz - 1);
                acc += wx[dx] * wy[dy] * wz[dz] * g[spec.index(x, y, z)];
            }
        }
    }
    return acc;
}

}  // namespace

PoissonSystem assemble_poisson_system(const PointCloud& cloud, const Eigen::Vector3d& bbox_min,
                                      const Eigen::Vector3d& bbox_max,
                                      const PoissonOptions& options) {
    PoissonSystem system;
    system.spec = make_grid(bbox_min, bbox_max, options.max_resolution);
    const SplatField splat = splat_points(system.spec, cloud);
    const GridSpec& spec = system.spec;
    const double h = spec.voxel;
    system.inv_h2 = 1.0 / (h * h);
    const std::vector<double> b = backward_divergence(spec, splat.value);

    // Screening scaling: alpha_s = screening / (h^2 * mean splat weight), so the
    // screening term matches the Laplacian's scale on the data-supported voxels
    // regardless of resolution or point density.
    const double mean_weight = splat.mean_weight > 0.0 ? splat.mean_weight : 1.0;
    system.alpha_s = options.screening / (h * h * mean_weight);

    system.weight = splat.weight;
    system.covered = splat.covered;

    // Normal equations (G^T G + alpha_s W) g = G^T V, and G^T V = -div V.
    system.rhs.resize(spec.size());
    for (std::size_t i = 0; i < system.rhs.size(); ++i) {
        system.rhs[i] = -b[i];
    }

    // Diagonal for the Jacobi preconditioner, from the same stencil apply() runs.
    system.diag.assign(spec.size(), 0.0);
    for (int z = 0; z < spec.nz; ++z) {
        for (int y = 0; y < spec.ny; ++y) {
            for (int x = 0; x < spec.nx; ++x) {
                const std::size_t idx = spec.index(x, y, z);
                system.diag[idx] =
                    static_cast<double>(neighbour_count(spec, x, y, z)) * system.inv_h2 +
                    system.alpha_s * system.weight[idx];
            }
        }
    }
    system.rhs_norm = std::sqrt(system.dot(system.rhs, system.rhs));

    // float32 mirrors for the GPU path (M1 has no fp64 shaders). The vectors are
    // float there; the double copies above stay authoritative for the residual.
    const double floor_diag = 1e-30 * system.inv_h2;
    system.weight_f.resize(system.weight.size());
    system.rhs_f.resize(system.rhs.size());
    system.inv_diag_f.resize(system.diag.size());
    for (std::size_t i = 0; i < system.diag.size(); ++i) {
        system.weight_f[i] = static_cast<float>(system.weight[i]);
        system.rhs_f[i] = static_cast<float>(system.rhs[i]);
        system.inv_diag_f[i] = static_cast<float>(1.0 / std::max(system.diag[i], floor_diag));
    }
    return system;
}

VolumeResult finalize_poisson_result(const PoissonSystem& system,
                                     const std::vector<double>& solution,
                                     const PointCloud& cloud, const PoissonOptions& options) {
    const GridSpec& spec = system.spec;
    VolumeResult result;
    result.field.spec = spec;
    result.field.data.assign(spec.size(), 0.0F);
    for (std::size_t i = 0; i < solution.size(); ++i) {
        result.field.data[i] = static_cast<float>(solution[i]);
    }

    // iso = mean of the field sampled at the input points (the points lie on the
    // surface, so this centres the level set on the data).
    double iso_sum = 0.0;
    for (const Eigen::Vector3d& p : cloud.points) {
        iso_sum += sample_trilinear(spec, solution, p);
    }
    result.iso = static_cast<float>(iso_sum / static_cast<double>(cloud.points.size()));

    // Observed mask: voxels covered by a splat, box-dilated by the requested
    // radius (Chebyshev distance), so isolated coverage still yields a usable
    // band for marching tetrahedra.
    result.observed.assign(spec.size(), 0);
    const int radius = static_cast<int>(std::ceil(options.density_radius_voxels));
    for (const std::size_t flat : system.covered) {
        const int x = static_cast<int>(flat % static_cast<std::size_t>(spec.nx));
        const int y = static_cast<int>((flat / static_cast<std::size_t>(spec.nx)) %
                                       static_cast<std::size_t>(spec.ny));
        const int z = static_cast<int>(flat / (static_cast<std::size_t>(spec.nx) *
                                               static_cast<std::size_t>(spec.ny)));
        for (int dz = -radius; dz <= radius; ++dz) {
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int xx = x + dx;
                    const int yy = y + dy;
                    const int zz = z + dz;
                    if (spec.contains(xx, yy, zz)) {
                        result.observed[spec.index(xx, yy, zz)] = 1;
                    }
                }
            }
        }
    }
    return result;
}

}  // namespace detail

std::vector<double> poisson_cpu_solve(const detail::PoissonSystem& system, double tolerance,
                                      int max_iterations, detail::PoissonCgStats& stats) {
    return detail::conjugate_gradients(system, system.rhs, tolerance, max_iterations, stats);
}

VolumeResult screened_poisson(const PointCloud& cloud, const Eigen::Vector3d& bbox_min,
                              const Eigen::Vector3d& bbox_max, const PoissonOptions& options,
                              PoissonStats* stats) {
    detail::validate_poisson_inputs(cloud, options);
    const detail::PoissonSystem system =
        detail::assemble_poisson_system(cloud, bbox_min, bbox_max, options);

    detail::PoissonCgStats solve_stats;
    const std::vector<double> g = poisson_solve_with_fallback(system, options, solve_stats);
    if (stats != nullptr) {
        stats->iterations = solve_stats.iterations;
        stats->relative_residual = solve_stats.relative_residual;
    }
    return detail::finalize_poisson_result(system, g, cloud, options);
}

}  // namespace alphashape::meshing
