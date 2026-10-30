#pragma once

// Per-camera gain compensation for the orthomosaic (Brown & Lowe 2007,
// "Automatic Panoramic Image Stitching using Invariant Features", section 6).
//
// Two photographs of the same ground rarely agree in brightness: exposure,
// vignetting and the camera's own gain differ, and the difference becomes a
// visible step where the seamline hands a cell from one photograph to the next.
// A single gain per camera per channel removes most of it.
//
// The formulation, per channel:
//
//     minimise  sum over overlapping pairs (i, j) of
//                 N_ij * [ (g_i * I_ij - g_j * I_ji)^2 / sigma_N^2
//                          + (1 - g_i)^2 / sigma_g^2 ]
//
// where I_ij is the mean of camera i's colour over the cells it shares with j,
// I_ji the same for j, and N_ij the number of shared cells sampled. The first
// term drives overlapping cameras to agree; the second is a prior that keeps a
// camera's gain near 1 so a camera that overlaps nothing, or a flight where
// every camera drifts together, does not disappear to zero.
//
// Setting the derivative to zero gives one linear equation per camera:
//
//     g_i * [ sum_j N_ij * I_ij^2 / sigma_N^2 + 1 / sigma_g^2 ]
//       - sum_j [ N_ij * I_ij * I_ji / sigma_N^2 ] * g_j  =  1 / sigma_g^2
//
// a symmetric positive-definite system (the prior makes it non-singular even
// for a disconnected overlap graph), solved here by Cholesky.
//
// COLOUR SPACE. The engine's Image is 8-bit sRGB-encoded and no stage decodes
// it, so I_ij, the solve and the application are all on STORED values. That is
// the right space for this problem rather than a compromise: the defect being
// removed is a step in the stored mosaic, so matching stored means is exactly
// the objective. A linear-light gain would require a decode round trip the
// pipeline does not otherwise perform, and would target a quantity nobody
// displays.

#include <array>
#include <cstdint>
#include <map>
#include <vector>

#include "common/algorithm/ortho/orthorectify.hpp"

namespace alphashape::ortho {

/// How much a fitted gain may deviate from 1. Clamping keeps a pathological
/// overlap (two nearly black frames, say) from producing an absurd multiplier
/// that would blow the mosaic out.
struct GainOptions {
    /// Per-sample noise in the stored [0, 1] scale: Brown & Lowe's 10 of 255
    /// (about 4% of full scale) is a plausible sensor + resampling noise floor
    /// and is what makes a small overlap count weigh less than a large one.
    double sigma_n = 10.0 / 255.0;
    /// Prior width on the gain: 0.1 says "expect a camera within 10% of
    /// nominal"; the paper's value, and conservative enough that a real
    /// exposure difference is still corrected because N_ij / sigma_n^2
    /// dominates it on any sizeable overlap.
    double sigma_g = 0.1;
    double min_gain = 0.5;
    double max_gain = 2.0;
};

/// One ordered-pair-free overlap record between two cameras, all channels.
///
/// `mean_a` is the mean of camera `a`'s stored colour over the cells both
/// cover, `mean_b` the same for `b`, both in [0, 1]; `count` the number of such
/// cells. One record per overlapping pair, with `a < b` by construction so the
/// solve never sees the same pair twice.
struct OverlapPair {
    int a = -1;
    int b = -1;
    double count = 0.0;
    std::array<double, 3> mean_a{0.0, 0.0, 0.0};
    std::array<double, 3> mean_b{0.0, 0.0, 0.0};
};

/// Accumulates overlap statistics incrementally, so a caller can feed a whole
/// grid, or blocks of it, and merge the results.
///
/// Per-pair sums live in a flat map keyed by (a, b); merging two accumulators
/// of the same grid, split anywhere, reproduces the whole-grid result exactly
/// (the sums are associative), which is what lets the estimate be computed
/// once and be independent of how the mosaic is later blocked.
class OverlapAccumulator {
public:
    /// Add one co-observed cell: the stored RGB (0..255) each camera rendered
    /// there. A pair is recorded with the lower camera index first.
    void add(int a, int b, const std::uint8_t rgb_a[3], const std::uint8_t rgb_b[3]);

    /// Fold `other` into this accumulator.
    void merge(const OverlapAccumulator& other);

    /// The finished records, pairs with no overlap dropped, sorted by (a, b)
    /// so the solve is deterministic.
    std::vector<OverlapPair> finish() const;

    bool empty() const { return sums_.empty(); }

private:
    struct Sum {
        double count = 0.0;
        /// Six running totals per pair: the sum of each camera's stored value on
        /// each of the three channels, in 0..255.
        std::array<double, 6> totals{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    };
    /// Key is ``a * stride + b`` with ``a < b``; a flat index keeps one lookup
    /// per cell and the sorted iteration in finish() makes the order
    /// deterministic. The stride lives in the .cpp (kKeyStride).
    std::map<std::int64_t, Sum> sums_;
};

/// Solve the per-channel gains. Returns one {g_r, g_g, g_b} per camera, in
/// [min_gain, max_gain]; a camera with no overlap gets exactly 1.
std::vector<std::array<double, 3>> solve_gains(int camera_count,
                                               const std::vector<OverlapPair>& pairs,
                                               const GainOptions& options = {});

/// Multiply `tile`'s stored colour by `gain`, in place, only where it is valid.
/// Values clamp to 0..255 (a gain above 1 can saturate a bright cell, which is
/// what the camera itself would have done).
void apply_gain(OrthoTile& tile, const std::array<double, 3>& gain);

}  // namespace alphashape::ortho
