#pragma once

// Coarse-to-fine (pyramid) support for the PatchMatch depth search (WP-17.14).
//
// A depth search at the working resolution starts from a random plane, so much
// of its budget is spent walking away from nonsense. Running the same search
// first on a reduced copy of the block gives a cheaper way to find the basin: at
// half resolution the same plane window covers twice the scene, so a good answer
// there costs a quarter of the pixels, and the finer level then starts from that
// answer instead of from scratch.
//
// The geometry that makes the transfer exact is in scaled_calibration(): a
// calibration scaled by 2 has half the focal length and the principal point
// expressed about the halved image centre, so the unit bearing of coarse pixel
// j is exactly the bearing of fine index 2j + 0.5 -- the midpoint of the two
// fine pixels the coarse one covers. A depth measured along a bearing is
// therefore the same number at both levels, and a normal lives in the reference
// camera frame at every level. Depth and normal cross a level boundary
// unchanged: no scale factor, no rotation. That identity is also why the
// upsample below samples the coarse map at fine/2 - 0.25 (see the .cpp).
//
// Everything here is a pure function of its inputs -- no randomness, no
// scheduling -- so a seeded pyramid run reproduces bit for bit on every platform
// and thread count. Only the *plan* and the *carry* live here; the search itself
// is run_patch_match() at each level.

#include <cstdint>
#include <vector>

#include "image/image.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::stereo {

/// Number of levels a search over a `width` x `height` reference may run at.
///
/// `levels` is the caller's request: 1 means a single level (the pre-WP-17.14
/// behaviour, the reference resolution only), N > 1 asks for exactly N, and 0
/// means automatic -- as many halvings as keep the shorter side at or above
/// `min_size`. Two caps always apply, in both cases:
///
///   * geometry: the coarsest level needs an interior, i.e. more than
///     2 * window_radius pixels on its shorter side;
///   * the plan: pyramid_iterations() never gives a level fewer than one sweep,
///     and prefers at least two, so it may reduce this count further. The number
///     of iterations actually used is the length of that plan.
///
/// Throws ValueError for a non-positive `min_size` or a negative `levels`.
int pyramid_level_count(int width, int height, int levels, int min_size, int window_radius);

/// Iterations per level, coarsest first, summing to `iterations`.
///
/// The finest level takes 40% of the budget and the remaining 60% is split the
/// same way among the coarser levels: with two levels and six iterations the
/// plan is [4, 2], with three it is [2, 2, 2]. The length of the result is the
/// effective level count -- level_count is reduced when the budget cannot give
/// every level at least `kMinIterationsPerLevel` sweeps (one is always given)
/// and never exceeds `iterations`.
///
/// Why 40/60: a level's cost is proportional to its pixel count, and each
/// halving costs a quarter of the previous level, so two levels cost 1.25x a
/// flat run's pixels. Spending 60% of the sweeps at the coarse level (where 75%
/// of the pixels are) and 40% at the fine one is what turns a 1.25x pixel cost
/// into a ~1.8x speed-up at equal quality -- see test_patch_match_pyramid.cpp.
constexpr int kMinIterationsPerLevel = 2;

std::vector<int> pyramid_iterations(int level_count, int iterations);

/// The random stream key of one level, `level_from_finest` = 0 being the
/// reference resolution. Level 0 returns `seed` unchanged, so a single-level
/// run -- and the finest level of a pyramid, given the same budget -- draws
/// exactly the numbers it drew before the pyramid existed. Coarser levels get a
/// mixed key, which keeps a level's draws independent of how many levels the run
/// happened to use.
std::uint64_t pyramid_level_seed(std::uint64_t seed, int level_from_finest);

/// One level down: the box 2x2 mean, the filter downsample_box() applies for the
/// working-resolution reduction, so a pyramid over an already reduced image is
/// a further reduction of the original.
FloatImage pyramid_downsample(const FloatImage& src);

/// The reference image and calibration at every level of a pyramid, coarsest
/// first. Level 0 is the coarsest; `levels() - 1` is the reference resolution,
/// whose image is the caller's own (never copied).
struct DepthPyramid {
    std::vector<FloatImage> reduced;         // levels() - 1 entries, coarsest first
    std::vector<Calibration> calibrations;   // levels() entries, coarsest first
    const FloatImage* reference = nullptr;   // the finest level, as given

    int levels() const { return static_cast<int>(calibrations.size()); }

    /// Level `level` counted from the COARSEST (0) to the reference (last).
    const FloatImage& image(int level) const {
        return level == levels() - 1 ? *reference : reduced[static_cast<std::size_t>(level)];
    }
    /// Level `level` counted from the reference resolution (0) to the coarsest.
    const Calibration& calibration_from_finest(int level_from_finest) const {
        return calibrations[calibrations.size() - 1 - static_cast<std::size_t>(level_from_finest)];
    }
};

/// Builds the pyramid for `reference` at `calibration`. Level `k` from the
/// finest is image k reduced by `pyramid_downsample` k times, and its
/// calibration is `scaled_calibration` applied k times to `calibration` -- the
/// chain anchored at the reference resolution, so the two stay consistent with
/// the images the search actually sees (including odd widths, where the halved
/// size truncates).
///
/// Throws ValueError when `level_count` is less than 1.
DepthPyramid build_depth_pyramid(const FloatImage& reference, const Calibration& calibration,
                                 int level_count);

/// A depth/normal map in flight between two levels.
struct DepthHypotheses {
    int width = 0;
    int height = 0;
    std::vector<float> depth;    // width * height, -1 = no hypothesis
    std::vector<float> normals;  // 3 * width * height, camera frame of the reference
};

/// Bilinear upsample by two, to `width` x `height` (normally twice the coarse
/// size). Fine pixel x samples the coarse map at x/2 - 0.25, the continuation of
/// the bearing identity above (the .cpp derives it in two lines).
///
/// Invalid (negative) depths carry no weight, and the weights of the remaining
/// taps are renormalised, so a hole in the coarse map never drags a neighbour
/// toward -1. A cell whose four taps are all holes is itself a hole -- the case
/// the finer level fills from its own random draw. Normals are interpolated
/// component-wise and renormalised; a zero result is written as a zero normal,
/// which the search treats as "no inherited plane".
///
/// Throws ValueError for a non-positive fine size or an empty coarse map.
DepthHypotheses pyramid_upsample(const DepthHypotheses& coarse, int width, int height);

/// The inverse-depth radius the random search starts from at a warm-started
/// level: the median, over the interior pixels with a depth, of the local
/// inverse-depth spread in `depth` -- the range of 1/depth over the
/// (2 * radius + 1)-square window around each pixel.
///
/// A flat run perturbs in inverse depth by the whole configured range at the
/// first sweep; at a fine level that means a candidate nowhere near the warm
/// plane, which is simply rejected, so the sweeps waste their draws. Starting
/// from the measured local variation makes them a real local search around the
/// inherited answer, and the existing per-iteration schedule halves it. The
/// result is clamped to [floor_value, 64 * floor_value] so a degenerate map
/// (one depth, or none at all) still searches sensibly.
double pyramid_search_span(const std::vector<float>& depth, int width, int height, int radius,
                           double floor_value);

}  // namespace alphashape::stereo
