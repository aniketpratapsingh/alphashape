// Shared declarations for the TV-L1 Chambolle-Pock kernels (tvl1_dual.comp,
// tvl1_primal.comp, tvl1_delta.comp, tvl1_reduce.comp): the push-constant
// block, the shared constants and the bin-centre convention.
//
// No `#version` line here: this file is included from each kernel after that
// kernel's own `#version 450`, and GLSL requires the directive to be the very
// first token of a translation unit.

// --- push constants ---------------------------------------------------------
// Byte-for-byte Tvl1Params in src/common/algorithm/mesh/tv_fusion_gpu.cpp.
// Keep the declaration order and types in step with that struct: the static
// asserts there pin the offsets, and a drift here is a silent wrong answer.
// Fields the kernels do not read stay declared so the block size (56 bytes)
// matches what the host pushes.
layout(push_constant) uniform Tvl1Params {
    uint nx;            // grid dimensions, x fastest
    uint ny;
    uint nz;
    uint bins;          // histogram bins per measured voxel
    uint row_count;     // number of (y, z) rows; block_list length
    uint iterations;    // unused by the kernels; read by the host only
    float tau;          // primal step
    float sigma;        // dual step (== tau, so tau * sigma * ||K||^2 < 1)
    float data_weight;  // tau * lambda
    float tolerance;    // unused by the kernels; read by the host only
    uint voxel_count;   // total voxels, the reduction's loop bound
    uint reduce_groups; // workgroup count of the delta reduction
    uint x_blocks;      // 128-voxel x-blocks per grid row
    uint reserved;      // pads the block to 56 bytes
} pc;

// --- shared constants -------------------------------------------------------
// Upper bound on the histogram bins the GPU path accepts; a wider histogram
// takes the CPU path (kMaxBins in the host code reads the same number).
const uint TVL1_MAX_BINS = 16u;

// Sentinel for "no row": returned when the second grid dimension overshoots the
// row count, so a group past the tail does no work. Nothing is culled at
// runtime any more (the block list is the identity over rows), but the launch
// dimensions are rounded up per kernel shape, so the tail guard is required.
const uint kCulled = 0xFFFFFFFFu;

// --- bin geometry -----------------------------------------------------------
// Bin centre c_k uniformly covering the clamped signed-distance interval
// [-1, 1]: exactly the CPU's bin_centre() in tv_fusion.cpp. Bit-identical to
// the host's double expression for the bin counts in use (2..16).
float tvl1_centre(uint k, uint bins) {
    if (bins <= 1u) {
        return 0.0;
    }
    return -1.0 + 2.0 * float(k) / float(bins - 1u);
}
