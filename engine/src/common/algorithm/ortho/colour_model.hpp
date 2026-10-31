#pragma once

// Per-camera, per-channel AFFINE colour model for the orthomosaic.
//
// A single gain per camera (gain.hpp) removes a brightness step but cannot
// remove a *contrast* difference: two photographs of one roof can disagree by
// different amounts in shadow and in sun, and no multiplier fixes both ends.
// An affine model per camera per channel,
//
//     I' = a_c * I + b_c
//
// has the two degrees of freedom that need: a matches the local contrast, b the
// black level (haze, flare, sensor black offset). The joint estimation problem,
// the robust loss and the gauge are the ones used by published joint
// radiometric alignment work (Brown & Lowe 2007 for the gain form; Goldman &
// Chen 2005 for a per-camera affine/vignetting model; Xiong & Pulli 2010 for
// robust colour matching of image sequences), restated here in our own terms:
//
//   * observations are the SAME 3-D surface point seen in two cameras, never
//     per-pair means of an output grid: a mean throws away exactly the spread
//     (dark vs bright parts of the shared area) that separates a from b;
//   * the residual is (a_i I_i + b_i) - (a_j I_j + b_j), i.e. the corrected
//     colours disagree;
//   * a robust redescending loss (Cauchy, as in the IRLS literature Huber 1964
//     formalised) keeps occlusions, moving objects and specular highlights --
//     all of which are a large residual on a minority of points -- from dragging
//     the fit. Huber's bounded weight was measured on our own 10%-occluder case
//     and left a 4.4% gain bias; Cauchy's leaves 0.6%;
//   * a weak prior pulls a toward 1 and b toward 0, so a camera with little or
//     no overlap cannot wander;
//   * the model has a two-parameter gauge: a' = s a, b' = s b + t leaves every
//     residual unchanged. The gauge is fixed AFTER the solve, to mean gain 1 and
//     mean offset 0, and only then clamped (clamping first would project the
//     whole vector onto the same bound and destroy its ratios -- the failure the
//     gain solve shipped with once).
//
// COLOUR SPACE. As in gain.hpp: the pipeline never decodes to linear light and
// the defect being removed is a step in stored sRGB, so the model is fitted and
// applied on STORED values.
//
// UNITS. The solver and its options work in NORMALISED stored units, 0..1, the
// same scale as GainOptions (a value divided by 255), which keeps the normal
// equations well conditioned. The one exception is CameraColour::offset, which
// is in stored GREY LEVELS because apply_colour adds it straight to an 8-bit
// pixel; the solve multiplies by 255 on the way out.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/algorithm/ortho/orthorectify.hpp"

namespace alphashape::ortho {

/// The correction one camera applies to its stored colour, per channel:
/// I' = gain[c] * I + offset[c], I in stored 0..255, offset in stored GREY
/// LEVELS. The default is the identity, so an unsolved camera is left alone.
struct CameraColour {
    std::array<double, 3> gain{1.0, 1.0, 1.0};
    std::array<double, 3> offset{0.0, 0.0, 0.0};
};

/// Constants of the robust affine fit. Every number here is a compromise
/// between "trust the data" and "do not invent a correction"; the comments say
/// which and why.
struct AffineOptions {
    /// Scale of the robust (Cauchy) loss, in stored units (0..1). 10/255 (~4%
    /// of full scale) is the sensor + resampling noise floor GainOptions::
    /// sigma_n uses and roughly the residual a small genuine exposure
    /// difference produces, so a residual at that size keeps half its weight. A
    /// tighter scale would call ordinary texture noise an outlier; a much looser
    /// one (4x this, measured) stops down-weighting a 10% occluder enough to
    /// keep the fit inside its bias budget. The loss itself is Cauchy rather
    /// than Huber because the contamination is a small fraction of LARGE
    /// residuals -- an occluder -- which is the case a redescending weight
    /// handles and a merely bounded one does not; see cauchy_weight() in the
    /// .cpp for the measurement.
    double loss_scale = 10.0 / 255.0;
    /// Prior width on (a_i - 1): "expect a camera within a factor of two of
    /// nominal". Deliberately loose -- the data term on any real overlap
    /// hundreds of times outvotes it -- because its job is only to keep the
    /// two gauge directions (which the data term does not observe exactly)
    /// finite, not to bias a genuine difference.
    double sigma_gain = 0.5;
    /// Prior width on b_i, in stored units: "expect a black level within a
    /// quarter of full scale". Also loose, same reason.
    double sigma_offset = 0.25;
    /// Clamps applied AFTER the gauge is fixed (see the file comment).
    double min_gain = 0.5;
    double max_gain = 2.0;
    double max_offset = 0.5;
    /// IRLS iterations. The problem is linear up to the weights, and the
    /// weights settle after a handful of passes on real data; 8 leaves margin
    /// without making the (cheap) loop the stage's cost.
    int iterations = 8;
};

/// One paired observation: the same surface point's stored colour (0..1, all
/// three channels) as cameras `a` and `b` render it. `a < b` by construction.
struct ColourObservation {
    std::int32_t a = -1;
    std::int32_t b = -1;
    std::array<float, 3> value_a{0.0F, 0.0F, 0.0F};
    std::array<float, 3> value_b{0.0F, 0.0F, 0.0F};
};

/// Extract paired observations from the sampler's per-camera point lists.
///
/// `points_by_camera[k][i]` is camera k's i-th shared surface point, ascending,
/// and `colours[k][i]` is that camera's colour there (same index). Two cameras
/// contribute one observation wherever their lists name the same point.
///
/// The global `max_observations` cap is spent FAIRLY: each camera may contribute
/// at most max(max_observations / camera_count, 2) observations, and the pairs
/// are walked in ROUNDS so every camera's first observation is placed before any
/// camera's second. A plain first-pairs-first cap lets a densely overlapping
/// flight fill the budget with the low-numbered pairs and leave high-numbered
/// cameras with none, which silently pins them at the identity. The walk is
/// deterministic: cameras and points are visited in ascending order only.
std::vector<ColourObservation> pair_observations(
    const std::vector<std::vector<int>>& points_by_camera,
    const std::vector<std::vector<std::array<float, 3>>>& colours,
    std::size_t max_observations);

/// Solve one (a, b) pair per camera per channel from paired observations.
///
/// Returns `camera_count` identities when there are no observations. Cameras
/// that appear in no observation keep the identity (the prior is not enough to
/// invent a correction). The solve is sparse: cameras only couple where they
/// were observed together.
std::vector<CameraColour> solve_affine_colours(
    int camera_count, const std::vector<ColourObservation>& observations,
    const AffineOptions& options = {});

/// Apply `colour` to `tile`'s stored pixels in place, only where it is valid.
/// Each channel is clamped to 0..255 and rounded -- the same integer the camera
/// itself would have stored.
void apply_colour(OrthoTile& tile, const CameraColour& colour);

}  // namespace alphashape::ortho
