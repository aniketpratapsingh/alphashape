#pragma once

// GPU port of the screened-Poisson linear solve (M10).
//
// screened_poisson() assembles the discretised system exactly as before and then
// calls poisson_solve_with_fallback(), which runs the Jacobi-preconditioned
// conjugate-gradient iteration on the GPU through gpu::with_fallback and falls
// back to the CPU on any device failure. The assembly and the result
// finalisation are shared with the CPU path, so both solvers see the same
// operator, right-hand side and output conventions.
//
// The device is M1-class hardware with float32 only, so the vectors and the
// stencil live in float32. Every reduction is a subgroup shuffle tree plus one
// per-workgroup combine in float; the per-workgroup partials are read back and
// re-summed in compensated double on the host, and the convergence decision is
// made in double there. A true residual recomputed in double on the CPU
// guarantees the returned field meets the CPU tolerance (see poisson_gpu.cpp
// for the scheme).

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/algorithm/mesh/poisson.hpp"
#include "common/algorithm/mesh/poisson_internal.hpp"
#include "model/mesh/point_cloud.hpp"

namespace alphashape::meshing {

/// True when the GPU solver may run: the layer is available and enabled, and the
/// system size fits the kernels' 32-bit indexing.
bool poisson_gpu_supported(const detail::PoissonSystem& system);

/// Solves system.rhs = A g on the device. Throws gpu::Error when the device is
/// unavailable or a dispatch fails; poisson_solve_with_fallback() catches it.
/// `stats` receives the double-precision convergence diagnostics.
std::vector<double> poisson_solve_device(const detail::PoissonSystem& system,
                                         const PoissonOptions& options,
                                         detail::PoissonCgStats& stats);

/// The production entry point used by screened_poisson(): GPU when supported and
/// working, CPU otherwise.
std::vector<double> poisson_solve_with_fallback(const detail::PoissonSystem& system,
                                                const PoissonOptions& options,
                                                detail::PoissonCgStats& stats);

/// CPU solve with the same diagnostics type, exposed so tests can compare.
std::vector<double> poisson_cpu_solve(const detail::PoissonSystem& system, double tolerance,
                                      int max_iterations, detail::PoissonCgStats& stats);

/// Test hooks. force_failure makes the next poisson_solve_device() throw after
/// the caller validates its inputs, so the fallback path can be exercised.
void poisson_gpu_force_failure(bool value);

/// Diagnostics of the last device solve: submissions, CG iterations and the
/// refinement rounds float32 needed. All zero for the CPU path.
int poisson_gpu_last_submissions();
int poisson_gpu_last_iterations();
int poisson_gpu_last_refinements();

}  // namespace alphashape::meshing
