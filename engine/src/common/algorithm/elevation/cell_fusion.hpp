#pragma once

// Per-cell height fusion for BuildDem (phase-2 quality item Q3).
//
// A DEM cell fed by a depth-map union usually holds many samples, one per
// overlapping view. The plain median discards that redundancy (its noise stays
// at the single-sample level) and treats a wide, disagreement-heavy cell like a
// tight one. `fuse_cell` replaces it with a small robust estimator:
//
//   1. sort the cell's observations (height, then confidence) so the result is
//      independent of rasterisation order;
//   2. centre at the median, scale s = 1.4826 * MAD, floored at the caller's
//      scale_floor (a fraction of the cell size);
//   3. with at least kFusionTrimMinCount observations, keep at least the central
//      kFusionKeepFraction by rank around the median (at least kFusionMinKept),
//      then extend the set outward while the biweight still accepts the
//      observation at c * scale: the rank trim is a floor, so a clean cell
//      keeps all its redundancy while a gross outlier stays outside;
//   4. refine the centre with kFusionIrlsIterations IRLS steps of a Tukey
//      biweight M-estimate, each observation weighted by its confidence;
//   5. report the summed final weight as the cell's data weight, reduced by an
//      agreement factor when the kept set's weighted RMS residual is large
//      relative to the scale floor.
//
// Published basis: Tukey (1977) biweight and Hampel (1974) robust statistics;
// Huber (1964) M-estimation; per-view confidence fusion as in Furukawa & Ponce
// (2010) and Schoenberger et al. (2016).

#include <cstddef>
#include <vector>

namespace alphashape::elevation {

/// One measured sample in a DEM cell.
struct HeightObservation {
    double height = 0.0;
    double confidence = 0.0;  ///< per-view fusion weight, >= 0
};

/// The result of fusing one cell.
struct FusedCell {
    double value = 0.0;    ///< fused height
    double weight = 0.0;   ///< data weight for the TGV pass
    std::size_t kept = 0;  ///< observations with a nonzero final weight
};

/// Scale floor as a fraction of the DEM cell size: the smallest robust spread
/// (in metres) a cell is allowed to have. A tunable pending the benchmark.
constexpr double kFusionScaleFloorFactor = 0.25;
/// Fraction of observations kept by the symmetric rank trim.
constexpr double kFusionKeepFraction = 0.7;
/// Never trim a cell below this many observations.
constexpr std::size_t kFusionMinKept = 3;
/// Trimming only applies at or above this observation count.
constexpr std::size_t kFusionTrimMinCount = 5;
/// Tukey biweight tuning constant (95% asymptotic efficiency under Gaussian
/// noise).
constexpr double kFusionTukeyC = 4.685;
/// IRLS refinement steps.
constexpr int kFusionIrlsIterations = 2;
/// MAD to Gaussian sigma consistency factor.
constexpr double kFusionMadToSigma = 1.482602218505602;
/// The agreement reference is this multiple of the scale floor.
constexpr double kFusionAgreementScale = 2.0;

/// Fuse one cell's measured observations. Sorts `observations` in place.
/// Repaired samples must not be passed here: they never mix with measurements.
/// Returns a zero weight for an empty input.
FusedCell fuse_cell(std::vector<HeightObservation>& observations, double scale_floor);

}  // namespace alphashape::elevation
