#include "common/algorithm/stereo/patch_match.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base/utils/error.hpp"
#include "common/algorithm/stereo/depth_pyramid.hpp"
#include "image/image.hpp"
#include "model/camera/calibration.hpp"

// Multi-view PatchMatch on slanted planes.
//
// Model. For each reference pixel p with unit bearing b_p (camera frame of the
// reference) a hypothesis is a plane n.X = rho, with rho = d * (n . b_p) so the
// plane passes through the point at distance d along b_p. Depth is the distance
// along the bearing (see DepthMap), not the camera Z. Sampling a pixel q of the
// window with the plane gives depth_q = rho / (n . b_q); the 3D point is
// depth_q * b_q, and a neighbour view is reached through the rigid transform
// X_cam = R X_world + t, i.e. X_n = (Rn Rref^T) X_ref + (tn - (Rn Rref^T) tref).
//
// Search. Red-black sweeps (a checkerboard half of the pixels per pass) with a
// propagation step (planes of opposite-colour neighbours, adjacent and at
// distance 3) and two perturbation steps (a fresh random plane; the current
// plane perturbed in inverse depth and by a random normal rotation). Because a
// half-sweep reads only pixels of the other colour, and every random draw is a
// pure function of (seed, pixel, iteration, stream), the result is independent
// of the OpenMP thread count and bit-identical run to run.
//
// Coarse to fine. run_patch_match() runs the search above once per level of a
// depth pyramid, coarsest first, and hands each finer level the plane the level
// beneath found (depth_pyramid.hpp). Every level is the same code path -- the
// same window, the same cost, the same draw schedule -- with its own image
// resolution, its own stream key and its own budget; the only additions are the
// warm-start evaluation at init and the per-level perturbation radius, both
// described where they happen.

namespace alphashape::stereo {
namespace {

// ---------------------------------------------------------------------------
// Stateless counter-based randomness.
// ---------------------------------------------------------------------------

inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/// 53-bit uniform in [0, 1).
inline double uniform01(std::uint64_t bits) {
    return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
}

/// A draw sequence keyed only by (seed, pixel, iteration, stream). No shared
/// state, so a pixel sees the same numbers whatever thread runs it.
class RngStream {
public:
    RngStream(std::uint64_t seed, std::int64_t pixel, std::int64_t iteration,
              std::int64_t stream) {
        std::uint64_t h = splitmix64(seed + 0x9E3779B97F4A7C15ULL);
        h = splitmix64(h ^ (static_cast<std::uint64_t>(pixel) * 0x100000001B3ULL));
        h = splitmix64(h ^ (static_cast<std::uint64_t>(iteration) * 0xC2B2AE3D27D4EB4FULL));
        h = splitmix64(h ^ (static_cast<std::uint64_t>(stream) * 0x9E3779B97F4A7C15ULL));
        state_ = h;
    }

    /// Uniform in [0, 1).
    double next() {
        state_ = splitmix64(state_);
        return uniform01(state_);
    }

private:
    std::uint64_t state_ = 0;
};

constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kConeHalfAngle = kPi / 3.0;         // 60 degree init cone around -b_p
constexpr double kMaxNormalRefineAngle = kPi / 3.0;  // 60 degree refinement ceiling
constexpr double kNoMatchCost = 2.0;                 // ZNCC = -1 (no agreement)
constexpr double kFacingEps = 1e-6;                  // n . b must be < -eps
constexpr double kCameraZEps = 1e-9;                 // neighbour z > eps (a guard
                                                     // against division blow-up)

/// Iteration key used for the initial random draw, kept distinct from the sweep
/// iterations 0..iterations-1 so the streams never collide.
constexpr std::int64_t kInitIteration = -1;

inline std::size_t si(int v) { return static_cast<std::size_t>(v); }

/// Uniform direction on the unit sphere.
inline Eigen::Vector3d random_unit_vector(RngStream& rng) {
    const double z = 2.0 * rng.next() - 1.0;
    const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
    const double phi = kTwoPi * rng.next();
    return Eigen::Vector3d(r * std::cos(phi), r * std::sin(phi), z);
}

/// Unit normal uniformly inside the 60 degree cone about -b_p, so it faces the
/// reference camera: n . b_p <= -cos(60 deg) < 0.
inline Eigen::Vector3d random_cone_normal(RngStream& rng, const Eigen::Vector3d& b_p) {
    const Eigen::Vector3d axis = (-b_p).normalized();
    const double cos_min = std::cos(kConeHalfAngle);  // 0.5
    const double ct = cos_min + (1.0 - cos_min) * rng.next();
    const double st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
    const double phi = kTwoPi * rng.next();
    const Eigen::Vector3d u = axis.unitOrthogonal();
    const Eigen::Vector3d v = axis.cross(u);
    return (st * std::cos(phi) * u + st * std::sin(phi) * v + ct * axis).normalized();
}

/// n rotated by a random angle in [0, max_angle] about a random axis.
inline Eigen::Vector3d perturb_normal(RngStream& rng, const Eigen::Vector3d& n, double max_angle) {
    const double angle = max_angle * rng.next();
    Eigen::Vector3d axis = random_unit_vector(rng);
    if (std::abs(axis.dot(n)) > 0.999) {
        axis = n.unitOrthogonal();
    }
    const Eigen::Matrix3d rot = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    return (rot * n).normalized();
}

/// Depth sampled uniformly in inverse depth over [depth_min, depth_max].
inline double random_depth(RngStream& rng, double depth_min, double depth_max) {
    const double lo = 1.0 / depth_max;
    const double hi = 1.0 / depth_min;
    return 1.0 / (lo + (hi - lo) * rng.next());
}

/// A neighbour reduced to what the inner loop needs: the composed transform
/// X_n = M X_ref + c (M = Rn Rref^T), the calibration parameters already packed
/// in the order try_project_point() expects, and a raw view of its image.
///
/// The projection is the shared non-throwing forward model, so one general path
/// covers every sensor (frame or fisheye, with or without distortion) — the
/// throw-free try_project_point<double> inlines into the pixel loop, so there
/// is no distortion-free special case to keep in step with the model.
struct NeighbourView {
    const float* data = nullptr;   // image pixels, row-major
    int width = 0;
    int height = 0;
    float max_x = 0.0F;            // width - 1
    float max_y = 0.0F;            // height - 1
    CameraType type = CameraType::Frame;
    double params[kCalibrationParamCount] = {};
    Eigen::Matrix3d M = Eigen::Matrix3d::Identity();
    Eigen::Vector3d c = Eigen::Vector3d::Zero();
};

/// Bilinear sample at integer-centred coordinates. Identical, operation for
/// operation, to FloatImage::sample() but on a raw pointer: the two lower
/// indices are clamped to the image independently (so a coordinate in
/// [-0.5, 0) replicates the edge pixel, exactly as at_clamped() does).
inline float sample_image(const float* data, int width, int height, float x, float y) {
    const float fx = std::floor(x);
    const float fy = std::floor(y);
    const int xr = static_cast<int>(fx);
    const int yr = static_cast<int>(fy);
    const float ax = x - fx;
    const float ay = y - fy;
    int xa = xr < 0 ? 0 : xr;
    int xb = xr + 1 < 0 ? 0 : xr + 1;
    xa = xa > width - 1 ? width - 1 : xa;
    xb = xb > width - 1 ? width - 1 : xb;
    int ya = yr < 0 ? 0 : yr;
    int yb = yr + 1 < 0 ? 0 : yr + 1;
    ya = ya > height - 1 ? height - 1 : ya;
    yb = yb > height - 1 ? height - 1 : yb;
    const std::size_t row_a = si(ya) * si(width);
    const std::size_t row_b = si(yb) * si(width);
    const float v00 = data[row_a + si(xa)];
    const float v10 = data[row_a + si(xb)];
    const float v01 = data[row_b + si(xa)];
    const float v11 = data[row_b + si(xb)];
    const float top = v00 * (1.0F - ax) + v10 * ax;
    const float bottom = v01 * (1.0F - ax) + v11 * ax;
    return top * (1.0F - ay) + bottom * ay;
}

/// Everything the photometric cost needs that is fixed for a whole run. The
/// reference window values and the per-pixel bearings are stored flat and read
/// straight from these arrays (never re-derived inside the candidate loop).
struct CostContext {
    int width = 0;
    int height = 0;
    const float* ref_values = nullptr;           // count * n_samples window values
    const Eigen::Vector3d* bearings = nullptr;   // count bearings, row-major
    const Eigen::Vector2i* offsets = nullptr;    // window offsets
    std::vector<NeighbourView> neighbours;
    int best_k = 1;
};

/// Per-pixel window state, reused by every candidate evaluation of one pixel
/// (~10 candidates x N neighbours per iteration). `b_store` holds the window
/// bearings, `ref_store` the reference window values (both fixed for the pixel),
/// and `mb_store` holds M * b_q for every neighbour, so the inner loop never
/// re-reads global memory nor re-applies the 3x3 transform. All storage lives
/// in the vectors so a per-thread instance allocates once for the whole image.
struct WindowCache {
    int n = 0;
    int n_neighbours = 0;
    std::vector<Eigen::Vector3d> b_store;   // n
    std::vector<float> ref_store;           // n
    std::vector<Eigen::Vector3d> mb_store;  // n_neighbours * n
    std::vector<double> dq;                 // n, candidate-level plane depths
};

/// Plane depth at every window sample for the current candidate: depth_q =
/// rho / (n . b_q). Facing and in-front validity depend only on the plane and
/// the sample (never on the neighbour), so this runs once per candidate instead
/// of once per candidate per neighbour. A non-positive entry marks a sample that
/// is edge-on or behind the reference camera and is skipped for every view.
/// `count` receives how many samples survive, which bounds the validity quorum
/// every neighbour is then judged against.
void prepare_candidate(const WindowCache& wc, double rho, const Eigen::Vector3d& n,
                       std::vector<double>& depth_q_out, int& count) {
    const Eigen::Vector3d* const b = wc.b_store.data();
    double* const dq = depth_q_out.data();
    const double n0 = n.x();
    const double n1 = n.y();
    const double n2 = n.z();
    int valid = 0;
    for (int k = 0; k < wc.n; ++k) {
        const Eigen::Vector3d& bk = b[si(k)];
        const double nb = n0 * bk.x() + n1 * bk.y() + n2 * bk.z();
        double d = 0.0;
        if (nb < -kFacingEps && rho / nb > 0.0) {
            d = rho / nb;
            ++valid;
        }
        dq[si(k)] = d;
    }
    count = valid;
}

/// ZNCC over the window for one neighbour view, clamped to [0, 2] (2 = no
/// agreement). A window with fewer than 75% valid samples, or a neighbour patch
/// with essentially no variance, scores 2. The reference patch and the
/// per-sample plane depths `dq` are candidate-level and passed precomputed, and
/// `mb` is M * b_q per sample, so the loop only does the neighbour-specific work:
/// X_n = dq_q * (M b_q) + c, projection, sample, accumulate.
///
/// `valid_out` receives the number of window samples that projected inside the
/// neighbour image and were accumulated into the correlation. It is what tells a
/// *genuine* mismatch (a textured patch that correlates badly) from a view that
/// simply cannot see this pixel (too little of the window projects into it):
/// only the former is a vote. In a wide-baseline block — a nadir grid, where two
/// cameras 22-24 m apart at 60 m altitude share only part of their footprint —
/// most neighbour views fall into the latter class, so the distinction matters.
double neighbour_cost(const NeighbourView& view, const float* ref, const double* dq,
                      const Eigen::Vector3d* mb, int n_samples, int min_valid,
                      int& valid_out) {
    double ref_sum = 0.0, ref_sq = 0.0, nb_sum = 0.0, nb_sq = 0.0, cross = 0.0;
    int valid_count = 0;

    for (int k = 0; k < n_samples; ++k) {
        const double depth_q = dq[si(k)];
        if (!(depth_q > 0.0)) {
            continue;  // edge-on or behind the reference camera for every view
        }
        // Even if every remaining sample were valid, a window that has already
        // lost too many cannot reach the 75% quorum: stop early.
        if (valid_count + (n_samples - k) < min_valid) {
            valid_out = valid_count;
            return kNoMatchCost;
        }
        // X_n = M * (depth_q * b_q) + c = depth_q * (M b_q) + c.
        const Eigen::Vector3d x_n = depth_q * mb[si(k)] + view.c;
        const double z_n = x_n.z();
        if (!(z_n > kCameraZEps)) {
            continue;  // behind the neighbour camera
        }
        // One general path for every sensor: the shared throw-free forward model
        // inlines here, so a distorted lens costs no more than a clean one.
        double pixel[2] = {0.0, 0.0};
        const double p_cam[3] = {x_n.x(), x_n.y(), z_n};
        if (!try_project_point<double>(view.type, view.width, view.height, view.params, p_cam,
                                       pixel)) {
            continue;  // undefined projection (Frame z <= 0): invalid sample
        }
        // Convention bridge: project() puts the centre of the top-left pixel at
        // (0.5, 0.5), while sample_image() puts it at (0, 0) — integer
        // coordinates are pixel centres there. So the half-pixel shift below is
        // required; feeding the raw projection to the sampler biases every match.
        const float sx = static_cast<float>(pixel[0] - 0.5);
        const float sy = static_cast<float>(pixel[1] - 0.5);
        if (sx < -0.5F || sy < -0.5F || sx > view.max_x + 0.5F || sy > view.max_y + 0.5F) {
            continue;  // out-of-bounds sample is invalid
        }
        const double r = static_cast<double>(ref[si(k)]);
        const double m = static_cast<double>(
            sample_image(view.data, view.width, view.height, sx, sy));
        ref_sum += r;
        ref_sq += r * r;
        nb_sum += m;
        nb_sq += m * m;
        cross += r * m;
        ++valid_count;
    }

    valid_out = valid_count;
    if (valid_count < min_valid) {
        return kNoMatchCost;
    }

    const double inv = 1.0 / static_cast<double>(valid_count);
    const double ref_mean = ref_sum * inv;
    const double nb_mean = nb_sum * inv;
    const double ref_var = std::max(0.0, ref_sq - ref_sum * ref_mean);
    const double nb_var = std::max(0.0, nb_sq - nb_sum * nb_mean);
    if (ref_var <= 1e-12 || nb_var <= 1e-12) {
        return kNoMatchCost;  // flat patch carries no correlation
    }
    // sum (r - mr)(m - mn) = cross - ref_sum * nb_mean.
    const double covariance = cross - ref_sum * nb_mean;
    double z = covariance / std::sqrt(ref_var * nb_var);
    z = std::clamp(z, -1.0, 1.0);
    return 1.0 - z;
}

/// Fill the per-pixel window cache: the window bearings, the reference window
/// values (copied once so the candidate loop never touches the full arrays) and
/// M * b_q for every neighbour. Called once per pixel per half-sweep, i.e. once
/// for the ~10 candidates that follow.
void fill_window_cache(const CostContext& ctx, WindowCache& wc, int px, int py) {
    Eigen::Vector3d* const b = wc.b_store.data();
    float* const ref = wc.ref_store.data();
    // ref_values[.] block for pixel p is exactly p's window (image at p + off_k),
    // so the window values need no recomputation; only the bearings, which live
    // at the offset pixels p + off_k, are gathered.
    const std::size_t p = si(py) * si(ctx.width) + si(px);
    const float* const window = ctx.ref_values + p * si(wc.n);
    for (int k = 0; k < wc.n; ++k) {
        const Eigen::Vector2i off = ctx.offsets[si(k)];
        b[si(k)] = ctx.bearings[p + si(off.y()) * si(ctx.width) + si(off.x())];
        ref[si(k)] = window[si(k)];
    }
    Eigen::Vector3d* const mb = wc.mb_store.data();
    for (int j = 0; j < wc.n_neighbours; ++j) {
        const Eigen::Matrix3d& m = ctx.neighbours[si(j)].M;
        for (int k = 0; k < wc.n; ++k) {
            mb[si(j) * si(wc.n) + si(k)] = m * b[si(k)];
        }
    }
}

/// Multi-view cost: the mean of the `best_k` smallest per-neighbour costs, taken
/// over the neighbours that can actually see this pixel.
///
/// A neighbour is a *vote* only when at least 75% of the reference window
/// projects into its image (valid_count >= min_valid) — i.e. its own view of the
/// surface here is real. A neighbour that cannot see enough of the window (a
/// wide-baseline block: at 22-24 m spacing and 60 m altitude two nadir cameras
/// share only part of their footprint, so most of a pixel's neighbourhood cannot
/// map its window at all) is not a vote; folding its no-match cost (2.0) into the
/// average forces every partially-observed pixel above `max_cost` and erases it
/// from the map. Clamping `best_k` to the number of votes is the other half of
/// the same point: "average the best k of n" is a robustness rule against
/// outlier neighbours, but it presupposes n votes exist — here k must never
/// exceed the votes there are, or the average is diluted by phantom costs.
///
/// A neighbour that *does* see the window but correlates badly still returns 2.0
/// and still counts as a vote, because that is a genuine rejection of the
/// hypothesis. Only geometric invisibility is discounted.
double multiview_cost(const CostContext& ctx, WindowCache& wc, double rho,
                      const Eigen::Vector3d& n) {
    const int nn = wc.n_neighbours;
    const int min_valid = (3 * wc.n + 3) / 4;  // ceil(0.75 * n_samples)
    int candidate_valid = 0;
    prepare_candidate(wc, rho, n, wc.dq, candidate_valid);
    if (candidate_valid < min_valid) {
        return kNoMatchCost;  // not even the whole plane covers 75% of the window
    }

    // Keep the `best_k` smallest *voting* costs, with their running sum. votes
    // counts how many neighbours see the pixel; k_eff = min(best_k, votes) is the
    // divisor once all neighbours are in.
    const int want = ctx.best_k;  // already in [1, nn]
    double best_store[64];
    double sum = 0.0;
    int filled = 0;  // how many of the k smallest are accumulated into `sum`
    int votes = 0;

    // Dozens of neighbours: rare, so a vector + partial_sort instead of the
    // fixed 64-slot insertion array.
    std::vector<double> many;
    const bool many_mode = want > 64;
    if (many_mode) {
        many.reserve(si(nn));
    }

    for (int j = 0; j < nn; ++j) {
        int valid = 0;
        const double c = neighbour_cost(ctx.neighbours[si(j)], wc.ref_store.data(), wc.dq.data(),
                                        wc.mb_store.data() + si(j) * si(wc.n), wc.n, min_valid,
                                        valid);
        if (valid < min_valid) {
            continue;  // this neighbour cannot see the window: not a vote
        }
        ++votes;
        if (many_mode) {
            many.push_back(c);
            continue;
        }
        if (filled < want) {
            // Insert into the sorted prefix of the k smallest.
            int i = filled;
            while (i > 0 && best_store[i - 1] > c) {
                best_store[i] = best_store[i - 1];
                --i;
            }
            best_store[i] = c;
            ++filled;
        } else if (c < best_store[want - 1]) {
            // Replace the largest kept cost; k is tiny so a shift is cheapest.
            int i = want - 1;
            while (i > 0 && best_store[i - 1] > c) {
                best_store[i] = best_store[i - 1];
                --i;
            }
            best_store[i] = c;
        }
    }

    if (votes == 0) {
        return kNoMatchCost;  // no neighbour sees this pixel
    }
    const int k_eff = std::min(want, votes);
    if (many_mode) {
        std::partial_sort(many.begin(), many.begin() + k_eff, many.end());
        double total = 0.0;
        for (int i = 0; i < k_eff; ++i) {
            total += many[si(i)];
        }
        return total / static_cast<double>(k_eff);
    }
    for (int t = 0; t < k_eff; ++t) {
        sum += best_store[t];
    }
    return sum / static_cast<double>(k_eff);
}

/// The best hypothesis found for one pixel. A candidate replaces `best` only
/// when strictly cheaper, so the survivor is independent of the order the
/// candidates are tried in.
struct Best {
    double depth = -1.0;
    Eigen::Vector3d normal = Eigen::Vector3d(0.0, 0.0, -1.0);
    double cost = std::numeric_limits<double>::infinity();
};

void evaluate(const CostContext& ctx, WindowCache& wc, double rho, double depth,
              const Eigen::Vector3d& n, Best& best) {
    // A candidate equal to the current incumbent reproduces its cost exactly, so
    // it can never be strictly cheaper: skip it. Propagation hands over the same
    // plane many times (adjacent neighbours converge to identical floats), so
    // this drops a large share of the inner evaluations without changing a digit.
    if (depth == best.depth && n == best.normal) {
        return;
    }
    const double c = multiview_cost(ctx, wc, rho, n);
    if (c < best.cost) {
        best.cost = c;
        best.depth = depth;
        best.normal = n;
    }
}

/// Snap a hypothesis (depth, normal) into a valid plane and evaluate it.
/// Rejects a depth outside the range or a normal not facing the camera.
void evaluate_hypothesis(const CostContext& ctx, WindowCache& wc, double depth,
                         const Eigen::Vector3d& n, double depth_min, double depth_max,
                         const Eigen::Vector3d& b_p, Best& best) {
    if (!(depth > depth_min && depth < depth_max)) {
        return;
    }
    const double nb = n.dot(b_p);
    if (nb >= -kFacingEps) {
        return;  // not facing the reference camera
    }
    const double rho = depth * nb;
    evaluate(ctx, wc, rho, depth, n, best);
}

/// Propagate a neighbour pixel's (depth, normal) to pixel p, transferring the
/// plane: the neighbour's plane through its own point, re-expressed at p's
/// bearing, yields depth_p = rho' / (n . b_p).
void propagate(const CostContext& ctx, WindowCache& wc, int qx, int qy, double depth_min,
               double depth_max, const std::vector<float>& depth,
               const std::vector<float>& normals, const Eigen::Vector3d& b_p, Best& best) {
    const std::size_t q = si(qy) * si(ctx.width) + si(qx);
    const float qd = depth[q];
    if (!(qd > 0.0F)) {
        return;
    }
    const std::size_t base = q * 3;
    const Eigen::Vector3d n(static_cast<double>(normals[base + 0]),
                            static_cast<double>(normals[base + 1]),
                            static_cast<double>(normals[base + 2]));
    if (n.squaredNorm() < 1e-12) {
        return;
    }
    const double rho_q = static_cast<double>(qd) * n.dot(ctx.bearings[q]);  // through point q
    const double denom = n.dot(b_p);
    if (denom >= -kFacingEps) {
        return;  // plane faces away at p
    }
    const double depth_p = rho_q / denom;
    evaluate_hypothesis(ctx, wc, depth_p, n, depth_min, depth_max, b_p, best);
}

/// Everything one level of a pyramid search needs. The search core below is
/// level-agnostic: it is handed a reference and a neighbour list already
/// reduced to this level, its own seed and budget, and (for every level but the
/// coarsest) the hypotheses upsampled from the level beneath.
struct LevelPlan {
    const PMView* reference = nullptr;
    const std::vector<PMView>* neighbours = nullptr;
    std::uint64_t seed = 1;
    int iterations = 0;
    /// Warm-start hypotheses at this level's resolution, or null for the
    /// coarsest level's random initialisation (the pre-pyramid behaviour).
    const DepthHypotheses* warm = nullptr;
    /// The inverse-depth radius the per-iteration perturbation starts from.
    /// The coarsest level uses the configured range's span, as before; a
    /// warm-started level uses the local variation of its inherited map, so its
    /// first sweeps search around the inherited plane instead of far from it.
    double span = 0.0;
    /// Called after every completed half-sweep with (done, total) counted in
    /// this level's own sweeps, so a caller can map them onto the whole
    /// pyramid's. May throw (a task's cancellation); the throw propagates and
    /// the partial level is discarded.
    std::function<void(int, int)> on_sweeps;
};

/// The red-black search at one level. `plan.reference` and `plan.neighbours`
/// must be consistent (calibrations matching their images) and the options
/// validated; run_patch_match() does both once for the whole pyramid.
PatchMatchResult run_level(const LevelPlan& plan, const PatchMatchOptions& options) {
    const PMView& reference = *plan.reference;
    const std::vector<PMView>& neighbours = *plan.neighbours;
    const int width = reference.image->width();
    const int height = reference.image->height();
    const std::size_t count = si(width) * si(height);
    if (count == 0) {
        throw ValueError("stereo::run_patch_match: reference image is empty");
    }

    // ---- Static setup ------------------------------------------------------
    const int r = options.window_radius;

    // Bearings of the reference pixels: the centre of pixel (i, j) is (i+0.5,
    // j+0.5), unproject() returns a unit camera-frame ray for it.
    std::vector<Eigen::Vector3d> bearings(count);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bearings[si(y) * si(width) + si(x)] =
                reference.calibration.unproject(Eigen::Vector2d(static_cast<double>(x) + 0.5,
                                                                static_cast<double>(y) + 0.5));
        }
    }

    // Window offsets: multiples of window_step in [-r, r], always including the
    // axes' endpoints and the centre.
    std::vector<Eigen::Vector2i> offsets;
    {
        std::vector<int> axis;
        for (int v = -r; v <= r; v += options.window_step) {
            axis.push_back(v);
        }
        if (axis.empty() || axis.back() != r) {
            axis.push_back(r);
        }
        if (std::find(axis.begin(), axis.end(), 0) == axis.end()) {
            axis.push_back(0);
            std::sort(axis.begin(), axis.end());
        }
        offsets.reserve(axis.size() * axis.size());
        for (int dy : axis) {
            for (int dx : axis) {
                offsets.emplace_back(dx, dy);
            }
        }
    }

    // Neighbour transforms: X_n = M X_ref + c with M = Rn Rref^T. The
    // calibration is packed once in the order try_project_point() expects, and
    // the image is exposed as a raw row-major pointer so the hot loop never
    // calls through Calibration or FloatImage.
    std::vector<NeighbourView> neighbour_views;
    neighbour_views.reserve(neighbours.size());
    const Eigen::Matrix3d rref_t = reference.R.transpose();
    for (const PMView& view : neighbours) {
        NeighbourView nv;
        const Calibration& c = view.calibration;
        nv.data = view.image->pixels().data();
        nv.width = c.width;
        nv.height = c.height;
        nv.max_x = static_cast<float>(c.width - 1);
        nv.max_y = static_cast<float>(c.height - 1);
        nv.type = c.type;
        nv.params[0] = c.f;
        nv.params[1] = c.cx;
        nv.params[2] = c.cy;
        nv.params[3] = c.b1;
        nv.params[4] = c.b2;
        nv.params[5] = c.k1;
        nv.params[6] = c.k2;
        nv.params[7] = c.k3;
        nv.params[8] = c.k4;
        nv.params[9] = c.p1;
        nv.params[10] = c.p2;
        nv.M = view.R * rref_t;
        nv.c = view.t - nv.M * reference.t;
        neighbour_views.push_back(nv);
    }

    const int n_samples = static_cast<int>(offsets.size());
    const int n_neighbours = static_cast<int>(neighbour_views.size());

    // A pixel is usable when its full window lies inside the image.
    const int interior_x0 = r;
    const int interior_x1 = width - 1 - r;
    const int interior_y0 = r;
    const int interior_y1 = height - 1 - r;

    // The reference window values for every interior pixel, gathered once: the
    // candidate loop then reads them straight from this flat array instead of
    // re-indexing the image per candidate. Indexed like the bearings, at
    // (y * width + x) * n_samples + k, so a window sample is one multiply away.
    std::vector<float> ref_values(count * si(n_samples), 0.0F);
    {
        const std::vector<float>& pixels = reference.image->pixels();
        for (int y = interior_y0; y <= interior_y1; ++y) {
            for (int x = interior_x0; x <= interior_x1; ++x) {
                const std::size_t q = si(y) * si(width) + si(x);
                float* const dst = ref_values.data() + q * si(n_samples);
                for (int k = 0; k < n_samples; ++k) {
                    const Eigen::Vector2i off = offsets[si(k)];
                    dst[si(k)] = pixels[si(y + off.y()) * si(width) + si(x + off.x())];
                }
            }
        }
    }

    CostContext ctx;
    ctx.width = width;
    ctx.height = height;
    ctx.ref_values = ref_values.data();
    ctx.bearings = bearings.data();
    ctx.offsets = offsets.data();
    ctx.neighbours = std::move(neighbour_views);
    {
        int k = options.best_k == 0 ? (n_neighbours + 1) / 2 : options.best_k;  // 0 = ceil(n/2)
        ctx.best_k = std::clamp(k, 1, n_neighbours);
    }

    const double depth_min = options.depth_min;
    const double depth_max = options.depth_max;
    int sweeps_done = 0;

    PatchMatchResult result;
    result.width = width;
    result.height = height;
    result.depth.assign(count, -1.0F);
    result.normals.assign(count * 3, 0.0F);
    result.cost.assign(count, static_cast<float>(kNoMatchCost));

    std::vector<std::uint8_t> active(count, 0);

    // Texture mask and the initial random hypotheses. The whole (interior) row
    // range is walked so every interior pixel has its window cache and bearings
    // filled; `active` records which ones carry a usable hypothesis.
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel
#endif
    {
        WindowCache local;
        local.n = n_samples;
        local.n_neighbours = n_neighbours;
        local.b_store.resize(si(n_samples));
        local.ref_store.resize(si(n_samples));
        local.mb_store.resize(si(n_neighbours) * si(n_samples));
        local.dq.resize(si(n_samples));

#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int y = interior_y0; y <= interior_y1; ++y) {
            Best best;
            for (int x = interior_x0; x <= interior_x1; ++x) {
                const std::size_t p = si(y) * si(width) + si(x);
                fill_window_cache(ctx, local, x, y);
                // Reference window mean / stddev over the sampled offsets.
                const float* const ref = local.ref_store.data();
                double sum = 0.0, sq = 0.0;
                for (int k = 0; k < n_samples; ++k) {
                    const double v = static_cast<double>(ref[si(k)]);
                    sum += v;
                    sq += v * v;
                }
                const double inv = 1.0 / static_cast<double>(n_samples);
                const double mean = sum * inv;
                const double var = std::max(0.0, sq - sum * mean);
                if (std::sqrt(var * inv) < options.min_texture_stddev) {
                    continue;  // too flat: no depth
                }
                active[p] = 1;
                const Eigen::Vector3d& b_p = bearings[p];
                best = Best{};
                // Warm start: the plane inherited from the level beneath, at this
                // level's geometry. It is evaluated here (rather than carrying
                // the coarse level's cost over) because a cost is only
                // comparable within one level: the window is a different size in
                // scene terms, so a coarse cost would rank the inherited plane
                // against nothing. The evaluation also rejects an inherited
                // plane whose normal no longer faces the reference, which leaves
                // the fresh random draw below to seed that pixel.
                if (plan.warm != nullptr) {
                    const std::size_t base = p * 3;
                    const Eigen::Vector3d inherited(
                        static_cast<double>(plan.warm->normals[base + 0]),
                        static_cast<double>(plan.warm->normals[base + 1]),
                        static_cast<double>(plan.warm->normals[base + 2]));
                    if (inherited.squaredNorm() > 1e-12) {
                        evaluate_hypothesis(ctx, local, static_cast<double>(plan.warm->depth[p]),
                                            inherited.normalized(), depth_min, depth_max, b_p,
                                            best);
                    }
                }
                // The coarsest level's initialisation, unchanged: one random
                // depth then one random cone normal, drawn in that order so the
                // keyed stream is not reordered (see the sweep loop's note).
                if (best.depth <= 0.0) {
                    RngStream rng(plan.seed, static_cast<std::int64_t>(p), kInitIteration, 0);
                    const double depth = random_depth(rng, depth_min, depth_max);
                    const Eigen::Vector3d normal = random_cone_normal(rng, b_p);
                    evaluate_hypothesis(ctx, local, depth, normal, depth_min, depth_max, b_p,
                                        best);
                }
                if (best.depth > 0.0) {
                    result.depth[p] = static_cast<float>(best.depth);
                    const std::size_t base = p * 3;
                    result.normals[base + 0] = static_cast<float>(best.normal.x());
                    result.normals[base + 1] = static_cast<float>(best.normal.y());
                    result.normals[base + 2] = static_cast<float>(best.normal.z());
                    result.cost[p] = static_cast<float>(best.cost);
                }
            }
        }
    }

    // ---- Red-black sweeps --------------------------------------------------
    // Refinement schedule: at iteration `it` the perturbation scale is
    //     s = 0.5^(it+1)          (0.5 at the first iteration, halving after)
    // the depth is perturbed in inverse depth by +/- plan.span * s, and the
    // normal by a random rotation of up to 60 degrees * s. plan.span is the
    // configured inverse-depth span at the coarsest level and the local
    // variation of the inherited map at a warm-started one.
    const int adj[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    const int far3[4][2] = {{-3, 0}, {3, 0}, {0, -3}, {0, 3}};

    const int iterations = plan.iterations;
    for (int it = 0; it < iterations; ++it) {
        const double s = std::pow(0.5, static_cast<double>(it) + 1.0);
        const double depth_delta = plan.span * s;
        const double normal_angle = kMaxNormalRefineAngle * s;

        for (int colour = 0; colour < 2; ++colour) {
            // Every pixel of this colour is updated using only opposite-colour
            // sources (adjacent and distance-3 neighbours: 3 is odd), so the
            // half-sweep is a fixed point of the parallel update and the result
            // is independent of the thread count. Rows are allocated to threads
            // in chunks of 4 so a row's workload (which depends on how many
            // pixels are `active`) does not force a fully static split.
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel
#endif
            {
                WindowCache local;
                local.n = n_samples;
                local.n_neighbours = n_neighbours;
                local.b_store.resize(si(n_samples));
                local.ref_store.resize(si(n_samples));
                local.mb_store.resize(si(n_neighbours) * si(n_samples));
                local.dq.resize(si(n_samples));

#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp for schedule(dynamic, 4)
#endif
                for (int py = interior_y0; py <= interior_y1; ++py) {
                    // The two colours of one row are the interleaved pixels; walk
                    // them as one contiguous double-step run (both parities of a
                    // row belong to the same iteration and a pixel never reads
                    // itself, so order within the row is irrelevant).
                    int px = interior_x0 + ((((interior_x0 + py) & 1) != colour) ? 1 : 0);
                    for (; px <= interior_x1; px += 2) {
                        const std::size_t p = si(py) * si(width) + si(px);
                        if (active[p] == 0) {
                            continue;
                        }
                        fill_window_cache(ctx, local, px, py);
                        const std::size_t base = p * 3;
                        const Eigen::Vector3d& b_p = bearings[p];
                        Best best;
                        best.depth = static_cast<double>(result.depth[p]);
                        best.normal =
                            Eigen::Vector3d(static_cast<double>(result.normals[base + 0]),
                                            static_cast<double>(result.normals[base + 1]),
                                            static_cast<double>(result.normals[base + 2]));
                        best.cost = static_cast<double>(result.cost[p]);
                        if (!(best.depth > 0.0)) {
                            // No incumbent yet: seed from a fresh random plane.
                            RngStream rng(plan.seed, static_cast<std::int64_t>(p),
                                          static_cast<std::int64_t>(it), 0);
                            // Sequenced draws: two calls in one argument list are
                            // evaluated in an unspecified order (clang left to
                            // right, g++ right to left), so the same seed consumed
                            // the stream in a different order and the platforms
                            // explored different planes. That one flip showed up as
                            // 2000/43200 differing depth-map pixels, a different
                            // mesh (19999 vs 20000 faces) and texture statistics on
                            // opposite sides of a calibrated bound.
                            const double depth = random_depth(rng, depth_min, depth_max);
                            const Eigen::Vector3d normal = random_cone_normal(rng, b_p);
                            evaluate_hypothesis(ctx, local, depth, normal, depth_min, depth_max,
                                                b_p, best);
                        } else {
                            // Propagation: 4 adjacent + 4 distance-3 neighbours.
                            for (const auto& d : adj) {
                                const int qx = px + d[0];
                                const int qy = py + d[1];
                                if (qx < 0 || qx >= width || qy < 0 || qy >= height) {
                                    continue;
                                }
                                propagate(ctx, local, qx, qy, depth_min, depth_max, result.depth,
                                          result.normals, b_p, best);
                            }
                            for (const auto& d : far3) {
                                const int qx = px + d[0];
                                const int qy = py + d[1];
                                if (qx < 0 || qx >= width || qy < 0 || qy >= height) {
                                    continue;
                                }
                                propagate(ctx, local, qx, qy, depth_min, depth_max, result.depth,
                                          result.normals, b_p, best);
                            }
                            // Refinement: a fresh random plane, then the current
                            // plane perturbed in inverse depth and normal.
                            RngStream rng(plan.seed, static_cast<std::int64_t>(p),
                                          static_cast<std::int64_t>(it), 0);
                            const double fresh_depth = random_depth(rng, depth_min, depth_max);
                            const Eigen::Vector3d fresh_normal = random_cone_normal(rng, b_p);
                            evaluate_hypothesis(ctx, local, fresh_depth, fresh_normal, depth_min,
                                                depth_max, b_p, best);

                            RngStream rng_d(plan.seed, static_cast<std::int64_t>(p),
                                            static_cast<std::int64_t>(it), 1);
                            const double inv_d = 1.0 / best.depth;
                            const double perturbed_inv =
                                inv_d + (2.0 * rng_d.next() - 1.0) * depth_delta;
                            if (perturbed_inv > 0.0) {
                                RngStream rng_n(plan.seed, static_cast<std::int64_t>(p),
                                                static_cast<std::int64_t>(it), 2);
                                const Eigen::Vector3d perturbed =
                                    perturb_normal(rng_n, best.normal, normal_angle);
                                evaluate_hypothesis(ctx, local, 1.0 / perturbed_inv, perturbed,
                                                    depth_min, depth_max, b_p, best);
                            }
                        }
                        if (best.depth > 0.0) {
                            result.depth[p] = static_cast<float>(best.depth);
                            result.normals[base + 0] = static_cast<float>(best.normal.x());
                            result.normals[base + 1] = static_cast<float>(best.normal.y());
                            result.normals[base + 2] = static_cast<float>(best.normal.z());
                            result.cost[p] = static_cast<float>(best.cost);
                        }
                    }
                }
            }
            if (plan.on_sweeps) {
                ++sweeps_done;
                plan.on_sweeps(sweeps_done, 2 * iterations);
            }
        }
    }

    // ---- Finalisation ------------------------------------------------------
    // A zero-iteration search has no sweeps to report, so the caller still gets
    // one "finished" tick rather than a progress bar that never leaves 0.
    if (plan.on_sweeps && iterations == 0) {
        plan.on_sweeps(1, 1);
    }
    for (std::size_t p = 0; p < count; ++p) {
        if (result.depth[p] <= 0.0F || static_cast<double>(result.cost[p]) > options.max_cost) {
            result.depth[p] = -1.0F;
            result.normals[p * 3 + 0] = 0.0F;
            result.normals[p * 3 + 1] = 0.0F;
            result.normals[p * 3 + 2] = 0.0F;
        }
    }

    return result;
}

/// Validates a whole run (reference, neighbours, options) before any level is
/// touched, so a bad call is rejected identically whether it resolves to one
/// level or five.
void validate_run(const PMView& reference, const std::vector<PMView>& neighbours,
                  const PatchMatchOptions& options) {
    if (reference.image == nullptr) {
        throw ValueError("stereo::run_patch_match: reference image is null");
    }
    if (reference.image->width() != reference.calibration.width ||
        reference.image->height() != reference.calibration.height) {
        throw ValueError("stereo::run_patch_match: reference calibration does not match image size");
    }
    if (neighbours.empty()) {
        throw ValueError("stereo::run_patch_match: no neighbour views");
    }
    for (std::size_t i = 0; i < neighbours.size(); ++i) {
        if (neighbours[i].image == nullptr) {
            throw ValueError("stereo::run_patch_match: neighbour image is null");
        }
        if (neighbours[i].image->width() != neighbours[i].calibration.width ||
            neighbours[i].image->height() != neighbours[i].calibration.height) {
            throw ValueError(
                "stereo::run_patch_match: neighbour calibration does not match image size");
        }
    }
    if (!(options.depth_min > 0.0) || !(options.depth_max > options.depth_min)) {
        throw ValueError("stereo::run_patch_match: depth range must satisfy 0 < min < max");
    }
    if (options.window_radius < 0) {
        throw ValueError("stereo::run_patch_match: window_radius must be non-negative");
    }
    if (options.window_step < 1) {
        throw ValueError("stereo::run_patch_match: window_step must be at least 1");
    }
    if (options.iterations < 0) {
        throw ValueError("stereo::run_patch_match: iterations must be non-negative");
    }
    if (options.best_k < 0) {
        throw ValueError("stereo::run_patch_match: best_k must be non-negative");
    }
    if (options.pyramid_levels < 0) {
        throw ValueError("stereo::run_patch_match: pyramid_levels must be non-negative");
    }
    if (options.pyramid_levels != 1 && options.pyramid_min_size <= 0) {
        throw ValueError("stereo::run_patch_match: pyramid_min_size must be positive");
    }
}

}  // namespace

std::vector<int> patch_match_iteration_plan(int width, int height,
                                            const PatchMatchOptions& options) {
    const int levels = pyramid_level_count(width, height, options.pyramid_levels,
                                           options.pyramid_min_size, options.window_radius);
    return pyramid_iterations(levels, options.iterations);
}

PatchMatchResult run_patch_match(const PMView& reference, const std::vector<PMView>& neighbours,
                                 const PatchMatchOptions& options) {
    validate_run(reference, neighbours, options);

    const int width = reference.image->width();
    const int height = reference.image->height();
    const std::vector<int> iterations_by_level = patch_match_iteration_plan(width, height, options);
    const int level_count = static_cast<int>(iterations_by_level.size());

    // Half-sweeps over the WHOLE pyramid, so the fraction a caller sees is
    // monotone across level boundaries and reaches 1.0 only at the end.
    int total_sweeps = 0;
    for (const int iterations : iterations_by_level) {
        total_sweeps += 2 * std::max(iterations, 0);
    }
    int done_before = 0;  // half-sweeps finished by the coarser levels

    // The pyramid's images. A single-level run allocates nothing: `image(0)` of
    // a one-level DepthPyramid is the caller's reference image.
    DepthPyramid pyramid;
    pyramid.reference = reference.image;
    pyramid.calibrations.push_back(reference.calibration);
    if (level_count > 1) {
        pyramid = build_depth_pyramid(*reference.image, reference.calibration, level_count);
    }
    // Neighbour images below the reference resolution, indexed
    // [level_from_finest - 1][camera]. Each entry is the previous level reduced
    // once more -- the same chain the reference pyramid follows -- and its
    // calibration is folded the same way, one halving at a time. Both matter
    // when a size truncates: floor(floor(W/2)/2) equals floor(W/4), but the
    // principal-point formula's `- W/2 * 0.5` term does not, so a one-shot
    // scaled_calibration(W, 4) can differ from two halvings by a quarter pixel.
    std::vector<std::vector<FloatImage>> reduced_neighbours(si(level_count - 1));
    std::vector<std::vector<Calibration>> reduced_calibrations(si(level_count - 1));
    for (int from_finest = 1; from_finest < level_count; ++from_finest) {
        std::vector<FloatImage>& level_images = reduced_neighbours[si(from_finest - 1)];
        std::vector<Calibration>& level_calibrations = reduced_calibrations[si(from_finest - 1)];
        level_images.reserve(neighbours.size());
        level_calibrations.reserve(neighbours.size());
        for (std::size_t i = 0; i < neighbours.size(); ++i) {
            const FloatImage& source =
                from_finest == 1 ? *neighbours[i].image
                                 : reduced_neighbours[si(from_finest - 2)][i];
            const Calibration source_calibration =
                from_finest == 1 ? neighbours[i].calibration
                                 : reduced_calibrations[si(from_finest - 2)][i];
            level_images.push_back(pyramid_downsample(source));
            level_calibrations.push_back(scaled_calibration(source_calibration, 2));
        }
    }

    // Coarsest first: every level beyond the first inherits the level beneath.
    DepthHypotheses inherited;
    PatchMatchResult result;
    for (int level = 0; level < level_count; ++level) {
        const int from_finest = level_count - 1 - level;  // 0 = the reference resolution
        const int iterations = iterations_by_level[si(level)];

        PMView level_reference;
        level_reference.calibration = pyramid.calibration_from_finest(from_finest);
        level_reference.R = reference.R;
        level_reference.t = reference.t;
        level_reference.image = &pyramid.image(level);

        std::vector<PMView> level_neighbours;
        level_neighbours.reserve(neighbours.size());
        for (std::size_t i = 0; i < neighbours.size(); ++i) {
            PMView view;
            view.calibration = from_finest == 0
                                   ? neighbours[i].calibration
                                   : reduced_calibrations[si(from_finest - 1)][i];
            view.R = neighbours[i].R;
            view.t = neighbours[i].t;
            view.image = from_finest == 0 ? neighbours[i].image
                                          : &reduced_neighbours[si(from_finest - 1)][i];
            level_neighbours.push_back(std::move(view));
        }

        // The radius the random search perturbs from. The coarsest level keeps
        // the configured inverse-depth range (the pre-pyramid behaviour). A
        // warm-started level starts from the local variation of its inherited
        // map instead -- floored so a degenerate map still searches, and capped
        // at the configured range so it can never explore further than a flat
        // run's first sweep would.
        const double configured_span = 1.0 / options.depth_min - 1.0 / options.depth_max;
        double span = configured_span;
        if (level > 0 && inherited.width > 0) {
            const double local_span =
                pyramid_search_span(inherited.depth, inherited.width, inherited.height,
                                    options.window_radius, configured_span / 64.0);
            span = std::min(configured_span, local_span);
        }

        LevelPlan level_plan;
        level_plan.reference = &level_reference;
        level_plan.neighbours = &level_neighbours;
        level_plan.seed = pyramid_level_seed(options.seed, from_finest);
        level_plan.iterations = iterations;
        level_plan.warm = inherited.width > 0 ? &inherited : nullptr;
        level_plan.span = span;
        if (options.on_sweep) {
            level_plan.on_sweeps = [&](int done, int level_total) {
                const int level_done = done_before + std::min(done, level_total);
                const double fraction =
                    total_sweeps > 0 ? static_cast<double>(level_done) /
                                           static_cast<double>(total_sweeps)
                                     : 1.0;
                options.on_sweep(std::min(fraction, 1.0));
            };
        }

        result = run_level(level_plan, options);
        done_before += 2 * std::max(iterations, 0);

        if (level + 1 < level_count) {
            // Carry the map up to the NEXT level's resolution. Depth along a
            // bearing and a reference-frame normal both survive a level change
            // unchanged (see depth_pyramid.hpp), so this is a bilinear
            // interpolation with the hole-aware weights, nothing else.
            const int next_from_finest = level_count - 2 - level;
            const Calibration& next_calibration =
                pyramid.calibration_from_finest(next_from_finest);
            DepthHypotheses coarse;
            coarse.width = result.width;
            coarse.height = result.height;
            coarse.depth = std::move(result.depth);
            coarse.normals = std::move(result.normals);
            inherited = pyramid_upsample(coarse, next_calibration.width, next_calibration.height);
        }
    }
    return result;
}

}  // namespace alphashape::stereo
