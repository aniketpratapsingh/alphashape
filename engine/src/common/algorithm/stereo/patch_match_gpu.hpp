#pragma once

// GPU PatchMatch depth estimation (M10).
//
// Same inputs, outputs and options as stereo::run_patch_match, executed on the
// accepted gpu:: layer when a device is available and enabled, with an
// automatic per-image CPU fallback. See patch_match_gpu.cpp for the scheme.
//
// The search itself is the published red-black checkerboard propagation of
// Galliani, Lasinger and Schindler (2015), "Massively parallel multiview
// stereopsis by surface normal diffusion": each half-sweep updates one colour
// of the checkerboard from its spatial neighbours' hypotheses, then applies the
// random refinement. The cost is the CPU's multi-view ZNCC, including the
// k_eff visibility vote rule; only the propagation order (red-black instead of
// the CPU's sequential sweep) and the arithmetic precision (float32) differ,
// so the result matches the CPU in quality, not bit for bit.

#include <cstdint>
#include <string>
#include <vector>

#include "common/algorithm/stereo/patch_match.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::stereo {

/// Capabilities of the GPU path. False when the layer is unavailable, disabled,
/// or the inputs exceed a kernel limit (window samples, neighbour count).
bool patch_match_gpu_supported(const PMView& reference, const std::vector<PMView>& neighbours,
                               const PatchMatchOptions& options);

/// Maximum window samples (offsets) and neighbours the GPU kernels handle.
int patch_match_gpu_max_samples();
int patch_match_gpu_max_neighbours();

/// True when PatchMatch's working buffers stay on host-visible memory. The
/// pre-decided fallback from the M10 plan: every pixel is seeded from the host
/// and the whole depth/normal/cost grid is read back each call, so the staging
/// round trips of DeviceLocal storage would add cost without changing the
/// kernel. Poisson's large vectors do use DeviceLocal. Exposed so the choice is
/// observable in diagnostics and tests.
bool patch_match_gpu_prefers_host_memory();

/// Runs PatchMatch through gpu::with_fallback: the GPU kernel when available
/// and enabled, the CPU implementation otherwise or after a device failure.
/// Throws ValueError exactly where run_patch_match does.
PatchMatchResult run_patch_match_device(const PMView& reference,
                                        const std::vector<PMView>& neighbours,
                                        const PatchMatchOptions& options);

/// Forces the GPU path to fail once, at the next call that would run it. Test
/// hook: the failure is absorbed by the fallback and counted by gpu::failure_count().
void patch_match_gpu_force_failure(bool value);

/// True when the next run_patch_match_device() call will attempt the GPU.
bool patch_match_gpu_would_run(const PMView& reference, const std::vector<PMView>& neighbours,
                               const PatchMatchOptions& options);

}  // namespace alphashape::stereo
