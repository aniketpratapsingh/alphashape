#include "common/algorithm/ortho/colour_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "base/utils/log.hpp"

namespace alphashape::ortho {
namespace {

/// Cauchy IRLS weight for a residual `r` and scale `delta`:
/// w = 1 / (1 + (r/delta)^2). A redescending weight: an ordinary residual is
/// nearly 1 (plain least squares), a residual at `delta` is 0.5, and a wild one
/// falls towards 0 instead of levelling off at a bounded but still large
/// influence the way Huber's delta/|r| does.
///
/// WHY CAUCHY AND NOT HUBER. The brief allows either. Measured on our own
/// 10%-occluder case (a whole patch of the scene showing a different surface in
/// one camera), Huber's bounded influence still leaves a 4.4% bias in that
/// camera's gain, because a bounded weight on 10% of the data is still a
/// systematic pull; Cauchy's redescending weight leaves 0.6%, inside the 1%
/// the requirement asks for. Huber would be the right choice for a heavier tail
/// of SMALL residuals (Gaussian noise with contamination); an occluder is a
/// small fraction of LARGE residuals, which is exactly Cauchy's regime.
double cauchy_weight(double r, double delta) {
    const double scaled = r / delta;
    return 1.0 / (1.0 + scaled * scaled);
}

/// The two unknowns per camera: a at 2*i, b at 2*i+1.
int gain_index(int camera) { return 2 * camera; }
int offset_index(int camera) { return 2 * camera + 1; }

/// One camera pair's six weighted totals, the sufficient statistics of the
/// normal equations for that pair. Accumulating these instead of a matrix entry
/// per observation is what keeps the solve O(pairs) in memory: a pair's 4x4
/// block is an analytic function of these sums.
struct PairSums {
    double w = 0.0;        ///< sum of weights
    double wa = 0.0;       ///< sum w * I_a
    double wb = 0.0;       ///< sum w * I_b
    double waa = 0.0;      ///< sum w * I_a^2
    double wbb = 0.0;      ///< sum w * I_b^2
    double wab = 0.0;      ///< sum w * I_a * I_b
};

/// Key for the deterministic pair map (std::map iterates sorted).
std::int64_t pair_key(int a, int b) {
    if (a > b) {
        std::swap(a, b);
    }
    return static_cast<std::int64_t>(a) * (std::int64_t{1} << 32) + static_cast<std::int64_t>(b);
}

/// Observations grouped by camera pair, each group holding the indices of its
/// observations in the caller's vector. Grouping once costs a single pass and
/// lets the IRLS loop recompute a pair's sufficient statistics without a map
/// lookup per observation. Groups come out sorted by (a, b), so a solve is
/// reproducible run to run.
///
/// An observation naming a camera outside [0, camera_count) is dropped: it can
/// only come from a caller bug, and letting it index the solution vector would
/// be undefined behaviour.
struct ObservationGroup {
    int a = -1;  ///< lower camera index
    int b = -1;  ///< higher camera index
    std::vector<std::size_t> index;
};

std::vector<ObservationGroup> group_observations(
    int camera_count, const std::vector<ColourObservation>& observations) {
    std::map<std::int64_t, ObservationGroup> by_pair;
    for (std::size_t i = 0; i < observations.size(); ++i) {
        const ColourObservation& o = observations[i];
        if (o.a < 0 || o.b < 0 || o.a == o.b || o.a >= camera_count ||
            o.b >= camera_count) {
            continue;
        }
        ObservationGroup& group = by_pair[pair_key(o.a, o.b)];
        if (group.index.empty()) {
            group.a = std::min(o.a, o.b);
            group.b = std::max(o.a, o.b);
        }
        group.index.push_back(i);
    }
    std::vector<ObservationGroup> groups;
    groups.reserve(by_pair.size());
    for (auto& entry : by_pair) {  // std::map: deterministic order
        groups.push_back(std::move(entry.second));
    }
    return groups;
}

/// The current model's coefficients for one channel.
struct ChannelState {
    std::vector<double> a;  ///< per camera
    std::vector<double> b;  ///< per camera
};

/// Cauchy-weighted sufficient statistics of one camera pair's observations for
/// one channel, evaluated at the CURRENT coefficients. The residual is the
/// disagreement of the two corrected colours:
///
///     r = (a_i I_i + b_i) - (a_j I_j + b_j)
///
/// and the observation enters the normal equations through its Jacobian
/// (I_i, 1, -I_j, -1), so the six totals below are all a pair contributes.
PairSums weighted_pair_sums(const ObservationGroup& group,
                            const std::vector<ColourObservation>& observations, int channel,
                            const ChannelState& state, double delta) {
    PairSums sums;
    const std::size_t ch = static_cast<std::size_t>(channel);
    const std::size_t ia_cam = static_cast<std::size_t>(group.a);
    const std::size_t ib_cam = static_cast<std::size_t>(group.b);
    for (const std::size_t index : group.index) {
        const ColourObservation& o = observations[index];
        const double ia = static_cast<double>(o.value_a[ch]);
        const double ib = static_cast<double>(o.value_b[ch]);
        // A stored value of 0 or 255 is CLIPPED by the camera, not measured: it
        // may stand for any darker/brighter true colour, so an affine model
        // fitted to it would be fitting the sensor's ceiling. Drop the channel
        // for this observation rather than let the clipped end pull the fit.
        if (!(ia > 0.0 && ia < 1.0) || !(ib > 0.0 && ib < 1.0)) {
            continue;
        }
        const double residual =
            (state.a[ia_cam] * ia + state.b[ia_cam]) - (state.a[ib_cam] * ib + state.b[ib_cam]);
        const double w = cauchy_weight(residual, delta);
        sums.w += w;
        sums.wa += w * ia;
        sums.wb += w * ib;
        sums.waa += w * ia * ia;
        sums.wbb += w * ib * ib;
        sums.wab += w * ia * ib;
    }
    return sums;
}

/// One IRLS Gauss-Newton step for one channel: assemble the sparse normal
/// equations with the Cauchy weights frozen at their current values, and take the
/// step. Returns false when the factorisation fails, leaving `state` untouched.
///
/// The residual r = (a_i I_i + b_i) - (a_j I_j + b_j) is HOMOGENEOUS in the
/// coefficients: it carries no measured target, because the objective is
/// "the two corrected colours agree", not "match a reference". So
/// J^T W r = (J^T W J) x, and the Gauss-Newton step H dx = -g is
/// (H_data + H_prior) dx = -(H_data x + g_prior). Writing it as a step rather
/// than a solve keeps the iterate on the same scale as the identity it starts
/// from, which is what makes the robust weights meaningful and the numbers well
/// conditioned.
///
/// Multiplying every (a_i, b_i) by s is the same solution (the residual is
/// unchanged), so the data term alone has a two-dimensional null space: a global
/// gain and a global offset. The prior fills it, and fix_gauge() then restates
/// the solution in the gauge the mosaic wants.
bool irls_step(const std::vector<ObservationGroup>& groups,
               const std::vector<ColourObservation>& observations, int channel,
               const std::vector<char>& usable, const AffineOptions& options, double delta,
               ChannelState& state) {
    const int n = static_cast<int>(state.a.size());
    const int unknowns = 2 * n;

    // The prior: lambda_a (a - 1)^2 + lambda_b b^2, widths in normalised stored
    // units. Weak by design (see AffineOptions), so it fixes the gauge and does
    // not bias a genuine correction.
    const double lambda_a = 1.0 / (options.sigma_gain * options.sigma_gain);
    const double lambda_b = 1.0 / (options.sigma_offset * options.sigma_offset);

    Eigen::SparseMatrix<double> A(unknowns, unknowns);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(unknowns);

    // Both triangles are pushed: SimplicialLDLT expects a self-adjoint matrix,
    // and writing each mirrored pair explicitly leaves nothing to infer.
    // Duplicate entries (a camera's diagonal across its several pairs) are
    // summed by setFromTriplets, which is exactly the accumulation wanted.
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(groups.size() * 16U + static_cast<std::size_t>(unknowns));

    for (const ObservationGroup& group : groups) {
        // A camera the channel cannot support (too little data or no usable
        // intensity spread) contributes nothing, so it cannot drag the cameras it
        // shares pairs with. It keeps the identity, set by the caller.
        if (usable[static_cast<std::size_t>(group.a)] == 0 ||
            usable[static_cast<std::size_t>(group.b)] == 0) {
            continue;
        }
        const PairSums sums = weighted_pair_sums(group, observations, channel, state, delta);
        if (!(sums.w > 0.0)) {
            continue;  // every observation of this pair was rejected outright
        }
        const int ga = gain_index(group.a);
        const int oa = offset_index(group.a);
        const int gb = gain_index(group.b);
        const int ob = offset_index(group.b);

        // J = (I_a, 1, -I_b, -1), so J^T W J is this 4x4 block, which depends
        // on the pair only through the six weighted totals.
        const double w = sums.w;
        const double ia = sums.wa;
        const double ib = sums.wb;
        const double block[4][4] = {
            {sums.waa, ia, -sums.wab, -ia},
            {ia, w, -ib, -w},
            {-sums.wab, -ib, sums.wbb, ib},
            {-ia, -w, ib, w},
        };
        const int axis[4] = {ga, oa, gb, ob};
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                triplets.push_back({axis[r], axis[c], block[r][c]});
            }
        }

        // J^T W r for THIS pair, which is the block applied to the pair's own
        // (a_i, b_i, a_j, b_j). Accumulated with a minus sign: the step solves
        // H dx = -g.
        const double a_a = state.a[static_cast<std::size_t>(group.a)];
        const double b_a = state.b[static_cast<std::size_t>(group.a)];
        const double a_b = state.a[static_cast<std::size_t>(group.b)];
        const double b_b = state.b[static_cast<std::size_t>(group.b)];
        const double local[4] = {a_a, b_a, a_b, b_b};
        for (int r = 0; r < 4; ++r) {
            double g = 0.0;
            for (int c = 0; c < 4; ++c) {
                g += block[r][c] * local[c];
            }
            rhs[axis[r]] -= g;
        }
    }

    for (int i = 0; i < n; ++i) {
        const std::size_t camera = static_cast<std::size_t>(i);
        triplets.push_back({gain_index(i), gain_index(i), lambda_a});
        triplets.push_back({offset_index(i), offset_index(i), lambda_b});
        // -g_prior, with g_prior = lambda_a (a - 1) and lambda_b b.
        rhs[gain_index(i)] -= lambda_a * (state.a[camera] - 1.0);
        rhs[offset_index(i)] -= lambda_b * state.b[camera];
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        return false;
    }
    const Eigen::VectorXd step = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
        return false;
    }
    for (int i = 0; i < n; ++i) {
        const std::size_t camera = static_cast<std::size_t>(i);
        const double a = state.a[camera] + step[gain_index(i)];
        const double b = state.b[camera] + step[offset_index(i)];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            return false;
        }
        state.a[camera] = a;
        state.b[camera] = b;
    }
    return true;
}

/// Fix the two-parameter gauge: rescale so the mean gain is exactly 1 and shift
/// so the mean offset is exactly 0, taken over the cameras the data actually
/// observed. A camera that appears in no observation is the identity by
/// construction and must not weigh on the gauge -- counting it would pull every
/// real camera towards 1 and shrink a genuine correction.
///
/// This runs EVERY iteration, not just at the end: it keeps the iterate in a
/// physical scale so the Cauchy scale is applied to residuals of the size the
/// final model will produce. It is a gauge transformation, not a clamp -- no
/// ratio is destroyed, and no coefficient is bounded here.
void fix_gauge(ChannelState& state, const std::vector<bool>& observed) {
    const std::size_t n = state.a.size();
    if (n == 0) {
        return;
    }
    double sum_a = 0.0;
    double sum_b = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!observed[i]) {
            continue;
        }
        sum_a += state.a[i];
        sum_b += state.b[i];
        ++count;
    }
    if (count == 0) {
        return;
    }
    const double mean_a = sum_a / static_cast<double>(count);
    const double mean_b = sum_b / static_cast<double>(count);
    if (!(std::isfinite(mean_a)) || !(std::isfinite(mean_b)) || std::abs(mean_a) < 1e-12) {
        return;  // a degenerate iterate: leave the previous one in place
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!observed[i]) {
            continue;
        }
        state.a[i] /= mean_a;
        state.b[i] = (state.b[i] - mean_b) / mean_a;
    }
}

}  // namespace

std::vector<CameraColour> solve_affine_colours(
    int camera_count, const std::vector<ColourObservation>& observations,
    const AffineOptions& options) {
    std::vector<CameraColour> colours(
        static_cast<std::size_t>(std::max(camera_count, 0)));
    if (camera_count <= 0 || observations.empty()) {
        return colours;
    }

    const std::vector<ObservationGroup> groups = group_observations(camera_count, observations);
    if (groups.empty()) {
        return colours;
    }

    // -- per-camera, per-channel data strength ------------------------------
    // Few observations, or a tiny intensity spread, cannot separate a gain from
    // an offset: the normal equations are near singular and the prior would
    // quietly invent a correction. Such a camera keeps the identity for that
    // channel (logged below). Clipped samples (stored 0 or 255) count as seen
    // but not usable, matching the fit's own rejection of them.
    struct ChannelSupport {
        std::size_t seen = 0;    ///< observations naming the camera
        std::size_t usable = 0;  ///< observations with unclipped values
        double lo = 1.0;
        double hi = 0.0;
    };
    constexpr std::size_t kMinObservations = 50;
    constexpr double kMinRangeStored = 8.0 / 255.0;
    std::vector<std::array<ChannelSupport, 3>> support(
        static_cast<std::size_t>(camera_count));
    for (const ObservationGroup& group : groups) {
        ChannelSupport* const a = support[static_cast<std::size_t>(group.a)].data();
        ChannelSupport* const b = support[static_cast<std::size_t>(group.b)].data();
        for (const std::size_t index : group.index) {
            const ColourObservation& o = observations[index];
            for (std::size_t ch = 0; ch < 3; ++ch) {
                ++a[ch].seen;
                ++b[ch].seen;
                const double va = static_cast<double>(o.value_a[ch]);
                const double vb = static_cast<double>(o.value_b[ch]);
                if (va > 0.0 && va < 1.0) {
                    ++a[ch].usable;
                    a[ch].lo = std::min(a[ch].lo, va);
                    a[ch].hi = std::max(a[ch].hi, va);
                }
                if (vb > 0.0 && vb < 1.0) {
                    ++b[ch].usable;
                    b[ch].lo = std::min(b[ch].lo, vb);
                    b[ch].hi = std::max(b[ch].hi, vb);
                }
            }
        }
    }

    for (int channel = 0; channel < 3; ++channel) {
        // Which cameras this channel can support, and hence which take part in
        // the gauge. A camera that appears nowhere keeps the identity: the
        // prior alone would say "1, 0" anyway, but leaving it out of the gauge
        // means the mean is taken over the cameras the data supports.
        const std::size_t ch = static_cast<std::size_t>(channel);
        std::vector<bool> observed(static_cast<std::size_t>(camera_count), false);
        std::vector<char> usable(static_cast<std::size_t>(camera_count), 0);
        std::ostringstream weak;
        bool any_weak = false;
        for (std::size_t i = 0; i < observed.size(); ++i) {
            const ChannelSupport& s = support[i][ch];
            const bool ok = s.usable >= kMinObservations && (s.hi - s.lo) >= kMinRangeStored;
            observed[i] = ok;
            usable[i] = static_cast<char>(ok ? 1 : 0);
            if (!ok && s.seen > 0) {
                if (any_weak) {
                    weak << ", ";
                }
                any_weak = true;
                weak << i << " (" << s.usable << " usable, range "
                     << (s.usable > 0 ? (s.hi - s.lo) * 255.0 : 0.0) << " grey)";
            }
        }
        if (any_weak && Log::enabled(LogLevel::Info)) {
            static const char* const kNames[3] = {"r", "g", "b"};
            log(LogLevel::Info) << "affine colour camera(s) " << weak.str() << " channel "
                                << kNames[channel] << ": too weak to fit, using the identity";
        }
        // Every channel starts at the identity: the correction the model
        // applies when there is nothing to learn. Fixing the gauge here is a
        // no-op on the identity, so it costs one pass and makes the very first
        // residual evaluation operate at physical scale.
        ChannelState state;
        state.a.assign(static_cast<std::size_t>(camera_count), 1.0);
        state.b.assign(static_cast<std::size_t>(camera_count), 0.0);

        // IRLS converges as soon as a step stops moving the coefficients; the
        // iteration cap is the guard for a pathological flight. The tolerance is
        // a stored unit / 1000, i.e. far below the 8-bit quantisation the model
        // is applied at, so a converged solve is indistinguishable from its
        // last iterate.
        const double tolerance = 1.0 / (255.0 * 1000.0);
        int iteration = 0;
        for (; iteration < options.iterations; ++iteration) {
            ChannelState next = state;
            if (!irls_step(groups, observations, channel, usable, options, options.loss_scale,
                           next)) {
                break;  // singular or non-finite: keep the last usable iterate
            }
            fix_gauge(next, observed);
            double change = 0.0;
            for (std::size_t i = 0; i < next.a.size(); ++i) {
                change = std::max(change, std::abs(next.a[i] - state.a[i]));
                change = std::max(change, std::abs(next.b[i] - state.b[i]));
            }
            state = std::move(next);
            if (change < tolerance) {
                break;
            }
        }

        // Gauge FIRST (already applied inside the loop), CLAMP second. The order
        // is the one the gain solver had to learn the hard way: clamping the raw
        // solution can map every camera onto the same bound, at which point the
        // gauge fix divides a constant by itself and the whole correction
        // silently becomes the identity. Clamping a properly gauged model only
        // ever trims a genuinely pathological camera.
        //
        // The offset travels with the gain. The gauge fix divides b by the mean
        // gain, so a camera whose gain was just clipped has an offset that is
        // still expressed against the unclipped scale; rescaling it by
        // clamped_a / a keeps the corrected colour (a I + b) on the same place
        // for the camera's typical intensity. That residual inconsistency is at
        // most a few grey levels and only on a camera already outside the
        // physical range, so it is not worth a second solve.
        for (int i = 0; i < camera_count; ++i) {
            const std::size_t camera = static_cast<std::size_t>(i);
            if (!observed[camera]) {
                continue;  // identity already
            }
            const double raw_a = state.a[camera];
            const double a = std::clamp(raw_a, options.min_gain, options.max_gain);
            const double scale = (std::abs(raw_a) > 1e-12) ? a / raw_a : 1.0;
            double b = state.b[camera] * scale;
            // Normalised -> stored grey levels, the units apply_colour works in.
            b *= 255.0;
            b = std::clamp(b, -options.max_offset * 255.0, options.max_offset * 255.0);
            colours[camera].gain[static_cast<std::size_t>(channel)] = a;
            colours[camera].offset[static_cast<std::size_t>(channel)] = b;
        }
    }
    return colours;
}

std::vector<ColourObservation> pair_observations(
    const std::vector<std::vector<int>>& points_by_camera,
    const std::vector<std::vector<std::array<float, 3>>>& colours,
    std::size_t max_observations) {
    std::vector<ColourObservation> observations;
    const std::size_t camera_count = points_by_camera.size();
    if (camera_count < 2 || max_observations == 0 || colours.size() != camera_count) {
        return observations;
    }
    // Fair share: every camera may appear in at most this many observations, so
    // the global cap can never be exhausted by the low-numbered pairs alone.
    // The floor of two matters: with a quota of one, an odd set of cameras
    // cannot be covered by disjoint pairs and its last camera would starve.
    const std::size_t per_camera =
        std::max<std::size_t>(max_observations / camera_count, std::size_t{2});
    std::vector<std::size_t> emitted(camera_count, 0);
    // Every pair of cameras that share at least one point, in ascending order.
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    for (std::size_t k = 0; k < camera_count; ++k) {
        if (points_by_camera[k].empty() || colours[k].size() != points_by_camera[k].size()) {
            continue;
        }
        for (std::size_t other = k + 1; other < camera_count; ++other) {
            if (points_by_camera[other].empty() ||
                colours[other].size() != points_by_camera[other].size()) {
                continue;
            }
            pairs.emplace_back(k, other);
        }
    }
    observations.reserve(std::min<std::size_t>(
        max_observations, per_camera * camera_count / std::size_t{2} + std::size_t{1}));
    // Each pair's merge cursor: both lists are ascending in point id, so a match
    // is a straight walk, and the cursor makes the walk resumable across rounds.
    std::vector<std::size_t> cursor_i(pairs.size(), 0);
    std::vector<std::size_t> cursor_j(pairs.size(), 0);
    std::vector<char> touched(camera_count, 0);
    while (observations.size() < max_observations) {
        std::fill(touched.begin(), touched.end(), static_cast<char>(0));
        bool any = false;
        for (std::size_t p = 0; p < pairs.size() && observations.size() < max_observations;
             ++p) {
            const std::size_t k = pairs[p].first;
            const std::size_t other = pairs[p].second;
            // One observation per camera per round: every camera's FIRST
            // observation is placed before any camera's second, so high indices
            // cannot be crowded out by the low-numbered pairs.
            if (touched[k] != 0 || touched[other] != 0 || emitted[k] >= per_camera ||
                emitted[other] >= per_camera) {
                continue;
            }
            const std::vector<int>& ka = points_by_camera[k];
            const std::vector<int>& kb = points_by_camera[other];
            bool matched = false;
            while (cursor_i[p] < ka.size() && cursor_j[p] < kb.size()) {
                const int a = ka[cursor_i[p]];
                const int b = kb[cursor_j[p]];
                if (a < b) {
                    ++cursor_i[p];
                } else if (b < a) {
                    ++cursor_j[p];
                } else {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                continue;
            }
            ColourObservation observation;
            observation.a = static_cast<std::int32_t>(k);
            observation.b = static_cast<std::int32_t>(other);
            observation.value_a = colours[k][cursor_i[p]];
            observation.value_b = colours[other][cursor_j[p]];
            ++cursor_i[p];
            ++cursor_j[p];
            observations.push_back(observation);
            ++emitted[k];
            ++emitted[other];
            touched[k] = 1;
            touched[other] = 1;
            any = true;
        }
        if (!any) {
            break;
        }
    }
    return observations;
}

void apply_colour(OrthoTile& tile, const CameraColour& colour) {
    if (tile.rgb.empty()) {
        return;
    }
    const std::size_t cells = tile.valid.size();
    for (std::size_t cell = 0; cell < cells; ++cell) {
        if (tile.valid[cell] == 0) {
            continue;
        }
        for (int c = 0; c < 3; ++c) {
            const std::size_t channel = static_cast<std::size_t>(c);
            const double value = static_cast<double>(tile.rgb[cell * 3 + channel]) *
                                     colour.gain[channel] +
                                 colour.offset[channel];
            tile.rgb[cell * 3 + channel] =
                static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0, 255.0)));
        }
    }
}

}  // namespace alphashape::ortho
