#pragma once

// Second-order total generalised variation (TGV^2) denoising / inpainting of a
// height field, and its first-order (TV) special case.
//
// TGV^2 (Bredies, Kunisch & Pock 2010, "Total Generalized Variation", SIAM
// J. Imaging Sci. 3(3)) penalises a symmetric-gradient-aware measure of the
// second derivative instead of the plain total variation, so piecewise-affine
// surfaces (planes, ramps -- exactly what a DEM looks like over flat terrain)
// are reconstructed without the staircase artefacts TV produces on slopes.
//
// The model, on a W x H grid with the forward difference operator grad and the
// symmetrised gradient eps (see the discretisation note in the .cpp):
//
//   min_{u,w}  a1*|grad u - w| + a0*|eps(w)| + D(u, f, c)
//
// with the data term D an L1 or squared-L2 residual against the observations f,
// weighted per cell by a confidence c >= 0. Cells with c = 0 carry no
// information and are filled by the regulariser alone, i.e. the model inpaints.
//
// Solved with the first-order primal-dual algorithm of Chambolle & Pock 2011
// (JMIV 40(1), Algorithm 1) -- the same family as the TV-L1 voxel fusion in
// meshing/tv_fusion.cpp.

#include <vector>

namespace alphashape::elevation {

/// Which data-fidelity term is used for the residual u - f.
enum class TgvDataTerm {
    /// Sum_i lambda*c_i*|u_i - f_i|: robust to salt-and-pepper outliers, the
    /// default because elevation samples from stereo matching are heavy-tailed.
    L1,
    /// (lambda/2)*sum_i c_i*(u_i - f_i)^2: smooth, faster to converge, but a
    /// single outlier biases its whole neighbourhood.
    L2,
};

struct TgvOptions {
    /// Weight of the first-order term |grad u - w|.
    double alpha1 = 1.0;
    /// Weight of the second-order term |eps(w)|. Default 2.0: TGV^2 favours
    /// piecewise *affine* reconstruction, and the ratio a0/a1 = 2 keeps
    /// piecewise-linear (0.5*x^2) ramps and flat plateaus both stationary
    /// without over-smoothing a genuine ridge (Bredies-Kunisch-Pock use
    /// a0/a1 in [1, 3] for piecewise-affine test images).
    double alpha0 = 2.0;
    /// Weight of the data term relative to the regulariser.
    double lambda = 1.0;
    TgvDataTerm data_term = TgvDataTerm::L1;
    /// Primal-dual iterations.
    int max_iterations = 500;
    /// Stop when RMS change of u per iteration / (RMS |u| + 1) < tolerance,
    /// checked every 10 iterations.
    double tolerance = 1e-5;
};

struct TgvResult {
    /// Primal-dual iterations actually run.
    int iterations = 0;
    /// Whether the tolerance criterion fired before max_iterations.
    bool converged = false;
    /// Value of the primal objective a1|grad u - w|_1 + a0|eps(w)|_1 + D(u)
    /// at the returned iterate.
    double primal_energy = 0.0;
};

/// TGV^2 denoise of the W*H height field `f` (row-major, index y*width + x) with
/// per-cell `confidence`. Heights are in any unit; the caller scales them so one
/// grid step is one unit (heights / cell size), which makes slopes
/// dimensionless.
///
/// `u` is in/out. On input it may hold an initial guess of size width*height; if
/// it is empty it is initialised from f where c > 0 and the mean of the observed
/// f elsewhere. On return it holds the solution.
///
/// Throws alphashape::ValueError when width or height < 2, when f or confidence
/// has the wrong size, when any c_i is negative, when f is non-finite where
/// c_i > 0, when all c_i are zero, or when alpha1/alpha0/lambda/max_iterations
/// is not positive.
TgvResult tgv2_denoise(int width, int height, const std::vector<float>& f,
                       const std::vector<float>& confidence, std::vector<float>& u,
                       const TgvOptions& options = {});

/// First-order total variation (w == 0) with the same data term. Kept as the
/// TV baseline and as a fallback for callers that do not want the second-order
/// field. Same arguments and validation as tgv2_denoise.
TgvResult tv_denoise(int width, int height, const std::vector<float>& f,
                     const std::vector<float>& confidence, std::vector<float>& u,
                     const TgvOptions& options = {});

}  // namespace alphashape::elevation
