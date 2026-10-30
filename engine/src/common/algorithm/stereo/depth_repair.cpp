#include "common/algorithm/stereo/depth_repair.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include "base/utils/error.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::stereo {

namespace {

// --- tuned constants -------------------------------------------------------

// A boundary point explains the hole's plane when it is within this FRACTION of
// the boundary's mean depth. 1% matches the cross-view filter's own relative
// depth tolerance, so a filled pixel is about as trustworthy as one the filter
// kept: the filter already accepted cross-view agreement to 1% as consistency.
constexpr double kPlaneRelativeTolerance = 0.01;

// Share of the boundary that must lie on the plane for the fill to be accepted.
// 0.7 is chosen to make a depth discontinuity fail: a hole straddling a step
// edge has its ring split between the two surfaces, so no single plane reaches
// 0.7. A surface that owns 70% of its ring is a hole inside that surface, and
// the remaining 30% is the ragged edge the filter left behind.
constexpr double kPlaneInlierFraction = 0.7;

// A run of outliers this long, walked along the boundary ring, is a second
// surface rather than filter noise. The inlier fraction alone cannot see it: a
// step whose far side owns 20% of the ring leaves 80% on the near plane, which
// passes 0.7 -- and the fill would then bridge the step with the near plane.
// Scattered ragged-edge outliers are isolated or come in pairs; a surface
// boundary is one contiguous arc. 10% is the tolerance: it is a small fraction
// of the ring (a few pixels on a ring of tens) yet far below the 30% the
// inlier rule already permits, so a step that owns between 10% and 30% of the
// ring is caught here. Two adjacent outliers are always tolerated, because on
// the smallest rings (6 points) 10% would be less than one pixel of slack.
constexpr double kMaxOutlierArcFraction = 0.1;
constexpr std::size_t kMaxOutlierArcFloor = 2;

// Share of the outliers that must sit on the plane's FAR side for the split to
// read as noise rather than as a one-sided offset. Random speckle scatters
// roughly evenly; a step is an offset, so its ring points land almost entirely
// on one side. 0.25 tolerates the handful of speckles a real ragged boundary
// leaves on the opposite side while still rejecting a coherent one-sided step.
constexpr double kOutlierSideBalance = 0.25;
constexpr std::size_t kMinOutliersForSideTest = 3;

// Share of a component's perimeter that must lie on MEASURED pixels. A
// 4-connected invalid component never has an invalid 4-neighbour (it would be
// part of the same component), so every uncovered perimeter edge is an image
// border edge: the fraction is exactly "at most 25% of the hole's edge may be
// the image boundary". A hole enclosed by measurements scores 1.0; one cut off
// by the image edge (typically where the depth range runs out) does not, and a
// plane fitted to the remainder would be extrapolated over ground nothing
// measured.
constexpr double kBoundaryCoverageFraction = 0.75;

// Below this many distinct boundary points a plane is not determined well enough
// to extrapolate across the hole: 3 points define a plane exactly, so fewer
// than this many (6) leaves no redundancy to notice a bad fit.
constexpr std::size_t kMinBoundaryPoints = 6;

// RANSAC (Fischler & Bolles 1981) draws; fixed so a map repairs identically on
// every run and platform. The ring is small (tens of points), so a few dozen
// draws find the dominant plane with near-certainty.
constexpr int kRansacIterations = 64;
constexpr std::uint64_t kRansacSeed = 0x243F6A8885A308D3ULL;

// A filled depth must lie within the span its own ring MEASURED. The fill is an
// interpolation across a small gap between measured samples, not an
// extrapolation; a plane that implies a depth outside [min, max] of the ring
// (widened by the same 1% relative tolerance the fit uses) has been evaluated
// where no measurement constrains it -- on real aerial blocks that is how a
// near-edge-on plane produces a depth of hundreds of metres inside a surface
// that spans 10. Rejecting those pixels keeps a repaired map from moving the
// DEM's height range at all.
constexpr double kDepthSpanMargin = 0.01;

// |cos| between the viewing ray and the plane normal below which the
// intersection is too grazing to trust. A small cosine means the ray meets the
// plane almost edge-on, so the depth divides by a number near zero and pixel
// noise in the fitted normal becomes metres of depth. 0.05 is ~87 degrees from
// the normal (3 degrees from the plane): at that incidence the plane is barely
// sampled in a depth image at all, so a fill there is not trustworthy either.
constexpr double kMinRayCosine = 0.05;

/// A plane n . x = offset in camera space; `normal` is unit length.
struct Plane {
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    double offset = 0.0;
};

/// Plane through three points; false when they are (nearly) collinear, where
/// the normal is dominated by rounding noise rather than by the geometry.
bool plane_from_points(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c,
                       Plane& out) {
    const Eigen::Vector3d ab = b - a;
    const Eigen::Vector3d ac = c - a;
    const Eigen::Vector3d n = ab.cross(ac);
    const double scale = ab.norm() * ac.norm();
    if (!(scale > 0.0) || n.norm() <= 1e-9 * scale) {
        return false;
    }
    out.normal = n.normalized();
    out.offset = out.normal.dot(a);
    return true;
}

/// Least-squares plane through the selected points: the eigenvector of their
/// covariance with the smallest eigenvalue, plus the centroid's offset.
bool fit_plane_least_squares(const std::vector<Eigen::Vector3d>& points,
                             const std::vector<std::size_t>& selected, Plane& out) {
    if (selected.size() < 3) {
        return false;
    }
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const std::size_t i : selected) {
        mean += points[i];
    }
    mean /= static_cast<double>(selected.size());
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const std::size_t i : selected) {
        const Eigen::Vector3d d = points[i] - mean;
        cov += d * d.transpose();
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    if (solver.info() != Eigen::Success) {
        return false;
    }
    const Eigen::Vector3d normal = solver.eigenvectors().col(0).normalized();
    // A ring whose points are all on one line has no plane: its middle
    // eigenvalue vanishes too, and the "normal" would be an arbitrary
    // perpendicular. A genuine (even noisy) surface always spreads in two
    // directions. The RANSAC inlier fraction is the real acceptance test; this
    // guard only keeps the refit numerically meaningful.
    const Eigen::Vector3d eigenvalues = solver.eigenvalues();
    if (!(eigenvalues(2) > 0.0) || !(eigenvalues(1) > 1e-12 * eigenvalues(2))) {
        return false;
    }
    out.normal = normal;
    out.offset = normal.dot(mean);
    return true;
}

/// Longest run of outliers along the ring, walking it as a closed loop. The
/// ring is ordered by the angle its points make around their centroid in the
/// plane (an orthonormal basis of that plane), which follows the boundary for
/// the small, simply-connected holes this pass accepts. The order is only used
/// as a conservative test: a mis-ordered ring can split a true arc, never join
/// two unrelated ones, so it may reject a fill it should have kept but will not
/// accept one it should have refused.
std::size_t longest_outlier_arc(const std::vector<Eigen::Vector3d>& ring,
                                const std::vector<std::uint8_t>& outlier, const Plane& plane) {
    const std::size_t n = ring.size();
    if (n == 0) {
        return 0;
    }
    // An orthonormal basis (u, v) of the plane, built from the axis least
    // aligned with the normal so the cross product is well conditioned.
    Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
    if (std::abs(plane.normal.x()) <= std::abs(plane.normal.y()) &&
        std::abs(plane.normal.x()) <= std::abs(plane.normal.z())) {
        axis = Eigen::Vector3d::UnitX();
    } else if (std::abs(plane.normal.y()) <= std::abs(plane.normal.z())) {
        axis = Eigen::Vector3d::UnitY();
    }
    const Eigen::Vector3d u = plane.normal.cross(axis).normalized();
    const Eigen::Vector3d v = plane.normal.cross(u).normalized();
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d& p : ring) {
        centroid += p;
    }
    centroid /= static_cast<double>(n);

    std::vector<std::size_t> order(n);
    std::vector<double> angle(n);
    for (std::size_t i = 0; i < n; ++i) {
        const Eigen::Vector3d d = ring[i] - centroid;
        angle[i] = std::atan2(d.dot(v), d.dot(u));
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&angle](std::size_t a, std::size_t b) {
        if (angle[a] != angle[b]) {
            return angle[a] < angle[b];
        }
        return a < b;  // total order, so the result is reproducible
    });

    // Walk the circle starting just after an INLIER (an outlier run must not
    // span the angular seam), so a run that crosses it is counted whole.
    std::size_t start = n;
    for (std::size_t i = 0; i < n; ++i) {
        if (outlier[order[i]] == 0) {
            start = i;
            break;
        }
    }
    if (start == n) {
        return n;  // every point is an outlier
    }
    std::size_t best = 0;
    std::size_t run = 0;
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t idx = order[(start + k) % n];
        if (outlier[idx] != 0) {
            ++run;
            best = std::max(best, run);
        } else {
            run = 0;
        }
    }
    return best;
}

/// True when the ring's outlier set reads as noise rather than as a second
/// surface: no contiguous outlier arc longer than kMaxOutlierArcFraction of the
/// ring, and no coherent one-sided offset. See the constants for the reasoning.
bool outlier_distribution_is_noise(const std::vector<Eigen::Vector3d>& ring,
                                   const std::vector<std::uint8_t>& outlier, const Plane& plane) {
    const std::size_t arc = longest_outlier_arc(ring, outlier, plane);
    const double allowed_arc =
        std::max(static_cast<double>(kMaxOutlierArcFloor),
                 kMaxOutlierArcFraction * static_cast<double>(ring.size()));
    if (static_cast<double>(arc) > allowed_arc) {
        return false;
    }
    // Signed residuals: points of a parallel far surface all fall on the same
    // side of the fitted plane, while speckle straddles it.
    std::size_t above = 0;
    std::size_t below = 0;
    for (std::size_t i = 0; i < ring.size(); ++i) {
        if (outlier[i] == 0) {
            continue;
        }
        const double residual = plane.normal.dot(ring[i]) - plane.offset;
        if (residual > 0.0) {
            ++above;
        } else {
            ++below;
        }
    }
    const std::size_t outliers = above + below;
    if (outliers < kMinOutliersForSideTest) {
        return true;  // too few to judge a side; the arc test already passed
    }
    const std::size_t minority = std::min(above, below);
    return static_cast<double>(minority) >= kOutlierSideBalance * static_cast<double>(outliers);
}

/// Plane through the ring: RANSAC for the dominant plane, then a least-squares
/// refit on its inliers. `relative_tolerance` is a fraction of the ring's mean
/// depth, so the same number means the same physical slack at any scale.
/// `outlier_mask` is filled in for the RETURNED plane, so callers see the
/// residual classification the acceptance decision was made on.
/// Returns false when no plane reaches the required inlier fraction, when the
/// refit stops explaining the ring, or when the outliers are a coherent second
/// surface rather than noise.
bool fit_ring_plane(const std::vector<Eigen::Vector3d>& ring, double mean_depth,
                    double relative_tolerance, double required_fraction, Plane& out,
                    std::vector<std::uint8_t>& outlier_mask) {
    if (ring.size() < kMinBoundaryPoints) {
        return false;
    }
    const double tolerance = relative_tolerance * mean_depth;
    if (!(tolerance > 0.0)) {
        return false;
    }
    // splitmix64 (the same mixing depth_map_fingerprint uses) drives RANSAC
    // directly: no std:: distribution, so a fixed seed picks the same triples on
    // every implementation (see base/utils/portable_rng.hpp for why).
    std::uint64_t state = kRansacSeed;
    const auto next = [&state]() {
        std::uint64_t x = (state += 0x9E3779B97F4A7C15ULL);
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    };
    std::vector<std::size_t> scratch(ring.size());
    std::size_t triple[3] = {0, 0, 0};
    std::vector<std::size_t> best_inliers;
    Plane best;
    for (int iteration = 0; iteration < kRansacIterations; ++iteration) {
        // Partial Fisher-Yates: three DISTINCT ring indices, uniform and
        // reproducible from the splitmix stream alone.
        for (std::size_t i = 0; i < ring.size(); ++i) {
            scratch[i] = i;
        }
        for (std::size_t i = 0; i < 3; ++i) {
            const std::size_t j = i + static_cast<std::size_t>(next() % (ring.size() - i));
            std::swap(scratch[i], scratch[j]);
            triple[i] = scratch[i];
        }
        Plane candidate;
        if (!plane_from_points(ring[triple[0]], ring[triple[1]], ring[triple[2]], candidate)) {
            continue;
        }
        std::vector<std::size_t> inliers;
        inliers.reserve(ring.size());
        for (std::size_t i = 0; i < ring.size(); ++i) {
            if (std::abs(candidate.normal.dot(ring[i]) - candidate.offset) <= tolerance) {
                inliers.push_back(i);
            }
        }
        if (inliers.size() > best_inliers.size()) {
            best_inliers = std::move(inliers);
            best = candidate;
        }
    }
    if (best_inliers.size() < kMinBoundaryPoints ||
        static_cast<double>(best_inliers.size()) <
            required_fraction * static_cast<double>(ring.size())) {
        return false;
    }
    Plane refined;
    if (fit_plane_least_squares(ring, best_inliers, refined)) {
        best = refined;
    }
    // Re-classify against the FINAL plane instead of trusting the RANSAC
    // inlier set: the TLS refit can tilt the plane enough that some of the
    // points that chose it no longer lie on it. The acceptance decision uses
    // the same residual predicate the caller's later stages do, so the fraction
    // below, the arc test and the span all describe one plane.
    outlier_mask.assign(ring.size(), 1U);
    std::size_t inliers = 0;
    for (std::size_t i = 0; i < ring.size(); ++i) {
        if (std::abs(best.normal.dot(ring[i]) - best.offset) <= tolerance) {
            outlier_mask[i] = 0U;
            ++inliers;
        }
    }
    if (inliers < kMinBoundaryPoints ||
        static_cast<double>(inliers) < required_fraction * static_cast<double>(ring.size())) {
        return false;
    }
    if (!outlier_distribution_is_noise(ring, outlier_mask, best)) {
        return false;
    }
    out = best;
    return true;
}

}  // namespace

DepthMap repair_depth_map(const DepthMap& map, const Calibration& calibration,
                          const RepairOptions& options, RepairStats* stats) {
    if (map.width < 0 || map.height < 0) {
        throw ValueError("repair_depth_map: map has negative size");
    }
    const std::size_t pixels =
        static_cast<std::size_t>(map.width) * static_cast<std::size_t>(map.height);
    if (map.depths.size() != pixels) {
        throw ValueError("repair_depth_map: depths array size does not match map size");
    }
    if (!map.normals.empty() && map.normals.size() != 3 * pixels) {
        throw ValueError("repair_depth_map: normals array size does not match map size");
    }
    if (!map.confidence.empty() && map.confidence.size() != pixels) {
        throw ValueError("repair_depth_map: confidence array size does not match map size");
    }
    if (!map.flags.empty() && map.flags.size() != pixels) {
        throw ValueError("repair_depth_map: flags array size does not match map size");
    }
    if (calibration.width != map.width || calibration.height != map.height) {
        throw ValueError("repair_depth_map: calibration size does not match the map");
    }
    if (options.max_area < 0) {
        throw ValueError("repair_depth_map: max_area must not be negative");
    }

    DepthMap out = map;  // a copy: the caller's map is never modified
    RepairStats local;
    if (options.max_area == 0 || pixels == 0 || map.width == 0 || map.height == 0) {
        if (stats != nullptr) {
            *stats = local;
        }
        return out;
    }
    if (out.flags.empty()) {
        // Materialise the provenance array only when the pass will run: an
        // empty flag vector means "all measured", so a map that is never
        // touched costs nothing.
        out.flags.assign(pixels, 0U);
    }
    const int width = map.width;
    const int height = map.height;
    const std::size_t stride = static_cast<std::size_t>(width);
    const bool have_normals = !out.normals.empty();
    const bool have_confidence = !out.confidence.empty();

    // 4-connected invalid components. Uncovered perimeter edges are image-border
    // edges (an invalid 4-neighbour would share the component), so
    // kBoundaryCoverageFraction bounds how much of a hole may be cut off by the
    // map edge: a plane fitted to a partially unobserved ring would be
    // extrapolated over ground nothing measured.
    std::vector<std::uint8_t> visited(pixels, 0);
    std::vector<std::size_t> stack;
    std::vector<std::size_t> component;
    std::vector<std::size_t> ring_indices;    // valid 4-neighbours of the component
    std::vector<Eigen::Vector3d> ring_points;  // their camera-space positions
    std::vector<std::uint8_t> ring_mark(pixels, 0);
    std::vector<std::size_t> fill_pixels;

    const auto invalid = [&](std::size_t idx) { return !(out.depths[idx] > 0.0F); };

    for (std::size_t start = 0; start < pixels; ++start) {
        if (visited[start] != 0 || !invalid(start)) {
            continue;
        }
        component.clear();
        stack.clear();
        ring_indices.clear();
        std::size_t border_edges = 0;     // perimeter edges that leave the image
        std::size_t measured_edges = 0;   // perimeter edges onto a measured pixel
        stack.push_back(start);
        visited[start] = 1;
        while (!stack.empty()) {
            const std::size_t cur = stack.back();
            stack.pop_back();
            component.push_back(cur);

            const int ci = static_cast<int>(cur % stride);
            const int cj = static_cast<int>(cur / stride);
            const int di[4] = {1, -1, 0, 0};
            const int dj[4] = {0, 0, 1, -1};
            for (int k = 0; k < 4; ++k) {
                const int ni = ci + di[k];
                const int nj = cj + dj[k];
                if (ni < 0 || nj < 0 || ni >= width || nj >= height) {
                    ++border_edges;
                    continue;  // an uncovered edge: the ring is open here
                }
                const std::size_t nidx = static_cast<std::size_t>(nj) * stride +
                                         static_cast<std::size_t>(ni);
                if (invalid(nidx)) {
                    if (visited[nidx] == 0) {
                        visited[nidx] = 1;
                        stack.push_back(nidx);
                    }
                } else {
                    ++measured_edges;
                    if (ring_mark[nidx] == 0) {
                        ring_mark[nidx] = 1;
                        ring_indices.push_back(nidx);
                    }
                }
            }
        }
        for (const std::size_t idx : ring_indices) {
            ring_mark[idx] = 0;  // restore for the next component
        }

        ++local.components;
        if (static_cast<int>(component.size()) > options.max_area) {
            ++local.rejected_area;
            continue;
        }

        // Ring coverage: the share of the component's perimeter that lies on a
        // measured pixel. Every perimeter edge is either measured (an invalid
        // 4-neighbour would have merged into this component) or an image-border
        // edge, so the two counts' sum is the whole perimeter. Below the
        // threshold the hole is cut off by the map edge, not enclosed by a
        // surface, and the plane would be extrapolated over unmeasured ground.
        const double coverage =
            (measured_edges + border_edges) == 0
                ? 0.0
                : static_cast<double>(measured_edges) /
                      static_cast<double>(measured_edges + border_edges);
        if (coverage < kBoundaryCoverageFraction) {
            ++local.rejected_boundary;
            continue;
        }

        // Back-project the ring to camera space and fit one plane to it.
        ring_points.clear();
        ring_points.reserve(ring_indices.size());
        double depth_sum = 0.0;
        double depth_min = std::numeric_limits<double>::max();
        double depth_max = 0.0;
        for (const std::size_t idx : ring_indices) {
            const int i = static_cast<int>(idx % stride);
            const int j = static_cast<int>(idx / stride);
            const Eigen::Vector2d centre(static_cast<double>(i) + 0.5,
                                         static_cast<double>(j) + 0.5);
            const double depth = static_cast<double>(out.depths[idx]);
            ring_points.push_back(calibration.unproject(centre) * depth);
            depth_sum += depth;
            depth_min = std::min(depth_min, depth);
            depth_max = std::max(depth_max, depth);
        }
        // The span a fill is allowed to produce: the ring's own measured range,
        // widened by the same relative tolerance the plane fit uses. A fill
        // outside it is an extrapolation nothing measured, so it is refused
        // (this is what keeps a near-edge-on plane from writing a 100 m depth
        // into a 10 m surface).
        const double allowed_min = depth_min * (1.0 - kDepthSpanMargin);
        const double allowed_max = depth_max * (1.0 + kDepthSpanMargin);
        const double mean_depth = depth_sum / static_cast<double>(ring_points.size());
        Plane plane;
        std::vector<std::uint8_t> outlier_mask;
        if (!fit_ring_plane(ring_points, mean_depth, kPlaneRelativeTolerance,
                            kPlaneInlierFraction, plane, outlier_mask)) {
            ++local.rejected_plane;
            continue;
        }

        // Fill: intersect each component pixel's viewing ray with the plane.
        fill_pixels.clear();
        for (const std::size_t idx : component) {
            const int i = static_cast<int>(idx % stride);
            const int j = static_cast<int>(idx / stride);
            const Eigen::Vector2d centre(static_cast<double>(i) + 0.5,
                                         static_cast<double>(j) + 0.5);
            const Eigen::Vector3d bearing = calibration.unproject(centre);
            const double denom = plane.normal.dot(bearing);
            if (std::abs(denom) < kMinRayCosine) {
                ++local.unfilled_pixels;  // grazing ray: the plane is edge-on
                continue;
            }
            const double depth = plane.offset / denom;
            if (!(depth > 0.0) || !std::isfinite(depth)) {
                ++local.unfilled_pixels;
                continue;
            }
            if (depth < allowed_min || depth > allowed_max) {
                ++local.unfilled_pixels;  // extrapolated beyond what the ring measured
                continue;
            }
            fill_pixels.push_back(idx);
            out.depths[idx] = static_cast<float>(depth);
            // Provenance is the flag, not the confidence: a MEASURED pixel can
            // legitimately hold any confidence including 0 (the filter keeps a
            // pixel with no supporting neighbour), so a magic confidence could
            // not be told apart from a measured one. The confidence value below
            // is only a fallback for readers that predate the flag.
            out.flags[idx] = static_cast<std::uint8_t>(out.flags[idx] | kDepthRepairedFlag);
            if (have_confidence) {
                out.confidence[idx] = kRepairedConfidence;
            }
            if (have_normals) {
                // The map's convention is a normal pointing TOWARD the camera,
                // i.e. n . bearing < 0 (see patch_match's cone normal). The
                // least-squares sign is arbitrary, so orient it here.
                const Eigen::Vector3d n =
                    (plane.normal.dot(bearing) > 0.0 ? -plane.normal : plane.normal);
                out.normals[3 * idx + 0] = static_cast<float>(n.x());
                out.normals[3 * idx + 1] = static_cast<float>(n.y());
                out.normals[3 * idx + 2] = static_cast<float>(n.z());
            }
        }
        if (!fill_pixels.empty()) {
            ++local.filled_components;
            local.filled_pixels += fill_pixels.size();
        }
    }

    if (stats != nullptr) {
        *stats = local;
    }
    return out;
}

}  // namespace alphashape::stereo
