#pragma once

// GPU TV-L1 range-image fusion (M10).
//
// Same inputs, outputs and options as meshing::tv_l1_fusion, executed on the
// accepted gpu:: layer when a device is available and enabled, with an
// automatic CPU fallback. See tv_fusion_gpu.cpp for the scheme.
//
// The histogram integration and the surface initialisation stay on the CPU: the
// ray-march scatter writes V x bins counts through atomics and runs once per
// call, while the two Chambolle-Pock sweeps (the hot part, 200-300 iterations)
// move to the device. The iteration loop is batched: one gpu::run carries a
// whole batch of dual/primal dispatch pairs, and a per-workgroup partial sum of
// |u_new - u_old| is reduced on the CPU in double between batches to stop at
// the fixed point.

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "common/algorithm/mesh/tv_fusion.hpp"

namespace alphashape::gpu {
class Context;
}  // namespace alphashape::gpu

namespace alphashape::meshing {

/// True when the device path can run: the layer is available and enabled *and*
/// the selected device is not a software rasteriser. The dual kernel packs the
/// dual field as fp16, which llvmpipe cannot compile (its LLVM backend aborts
/// the process), so a software device declines here, before any tvl1 pipeline
/// is created, and the callers run the CPU solver instead. Everything else is
/// input-independent: the setup and iteration kernels have no input-size
/// limits, so no input-dependent test is needed beyond the CPU validation.
bool tv_l1_fusion_gpu_supported();

/// Runs TV-L1 fusion through gpu::with_fallback: the GPU kernels when available
/// and enabled, the CPU implementation otherwise or after a device failure.
/// Validates exactly where tv_l1_fusion does, before touching a device.
VolumeResult tv_l1_fusion_device(const std::vector<FusionView>& views,
                                 const Eigen::Vector3d& bbox_min,
                                 const Eigen::Vector3d& bbox_max,
                                 const TvFusionOptions& options);

/// Forces the GPU path to fail once, at the next call that would run it. Test
/// hook: the failure is absorbed by the fallback and counted by
/// gpu::failure_count().
void tv_fusion_gpu_force_failure(bool value);

/// Wall-clock breakdown of the most recent GPU solve (zero when the CPU path
/// ran). setup_seconds is the CPU histogram integration, upload_seconds the row
/// compaction plus buffer creation, iterate_seconds the dispatch batches
/// including their partial read-backs, readback_seconds the final field read.
struct TvFusionGpuProfile {
    double setup_seconds = 0.0;
    double upload_seconds = 0.0;
    double iterate_seconds = 0.0;
    double readback_seconds = 0.0;
    int batches = 0;
};
TvFusionGpuProfile tv_fusion_gpu_last_profile();

/// True when the next tv_l1_fusion_device() call will attempt the GPU.
bool tv_l1_fusion_gpu_would_run();

/// True when the context's driver can compile the fp16 storage used by the
/// tvl1_dual projection step (shaderFloat16 + storageBuffer16BitAccess). It is
/// false for a software device even where the extension is advertised: llvmpipe
/// reports VK_KHR_shader_float16_int8, but its LLVM backend aborts the process
/// when it compiles the pattern ("Cannot select: ... anyext from i16"). The
/// remaining accumulations stay fp32 either way; the fp16 packing only shrinks
/// the dual stream.
bool tv_l1_fusion_gpu_fp16_available(const gpu::Context& context);

}  // namespace alphashape::meshing
