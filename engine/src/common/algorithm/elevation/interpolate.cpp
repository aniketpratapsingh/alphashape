// Elevation raster hole interpolation.
//
// DISTANCE TRANSFORM
// ------------------
// nearest_sites() computes the exact Euclidean distance transform of a
// validity mask with the separable algorithm of Felzenszwalb & Huttenlocher
// (2012), "Distance Transforms of Sampled Functions", Theory of Computing 8(1).
// A one-dimensional squared distance transform D(i) = min_q (i-q)^2 + f(q) is
// the lower envelope of parabolas, built in O(n) with a stack of active
// parabola vertices v[k] and their break points z[k]. We run it on the rows
// first (where f is trivially 0 on valid cells and +inf elsewhere, so a simple
// two-sweep scan suffices) and then, carrying the whole row's best source
// along as the parabola payload, on the columns. The payload propagates the
// arg-min linearly-indexed data cell, giving exact distances in cells and a
// deterministic tie-broken nearest-site index.
//
// HOLE FILLING
// ------------
// Nearest: propagate the value of the nearest data cell.
// IDW: inverse-distance weighting over the k nearest data cells.
// NaturalNeighbour: discrete Sibson interpolation, Park, Linsen, Kreylos,
// Owens & Hamann (2006), "Discrete Sibson Interpolation", IEEE TVCG 12(2).
// It is a SCATTER, not a gather. The natural-neighbour weight of a data site s
// at a query x is the area stolen from s's Voronoi cell when x is inserted into
// the diagram, i.e. the open set of points strictly closer to x than to s,
// which on the raster is the discrete set
//     { p : |p - x| < r(p) and s(p) = s },
// where s(p) is the data site nearest to p and r(p) = |p - s(p)| — both read
// straight off the EDT. So every cell p votes the value f(s(p)) of its own
// nearest site into every query cell strictly within r(p) of p, and the value
// at a hole cell q is the mean of the votes it collects,
//     value(q) = sum_{p : |q-p| < r(p)} f(s(p)) / #{p : |q-p| < r(p)},
// the count for a given site being its (discrete) stolen area at q. Note the
// mean is over the votes received, NOT over a ball B(q, r(q)) around q: r(p)
// varies from cell to cell, so a gather over a fixed ball is a different and
// much weaker operator. Valid data cells are excluded from voting: r = 0 there,
// and their value is already correct and must stay untouched. Since a fillable
// hole cell p has r(p) >= 1, its own vote always lands (|p-p| = 0 < r(p)^2), so
// every hole cell p ranges over receives at least one vote and no count is 0.
//
// Determinism: votes land in one private buffer per FIXED band of source rows
// and the bands are merged in band order. Neither the per-band accumulation
// order (fixed raster order within the band) nor the merge order depends on the
// thread count, and the counts are exact integers, so the output is
// bit-identical for any thread count.

#include "common/algorithm/elevation/interpolate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "base/utils/error.hpp"
#include "common/algorithm/elevation/sibson.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace alphashape::elevation {
namespace {

/// floor(sqrt(v)) for integer-valued v >= 0 (exact for v < 2^52).
int isqrt(double v) {
    if (v <= 0.0) return 0;
    int r = static_cast<int>(std::sqrt(v));
    while (static_cast<double>(r + 1) * static_cast<double>(r + 1) <= v) ++r;
    while (r > 0 && static_cast<double>(r) * static_cast<double>(r) > v) --r;
    return r;
}

/// One-dimensional squared distance transform: out[i] = min_q (i-q)^2 + f[q],
/// out_src[i] = src[q*], where q* is the minimiser. Ties are resolved by the
/// lower envelope itself (deterministic). Every f[q] must be finite and >= 0,
/// or +inf to mark the sample unusable. `vp`, `vs`, `zb` are scratch of H+1
/// entries. Returns false when every sample is +inf.
bool lower_envelope(const double* f, const std::int32_t* src, int n,
                    double* out, std::int32_t* out_src, double* vp,
                    std::int32_t* vs, double* zb) {
    const double inf = std::numeric_limits<double>::infinity();
    int k = -1;
    for (int q = 0; q < n; ++q) {
        if (!(f[q] < inf)) continue;
        const double qd = static_cast<double>(q);
        if (k < 0) {
            k = 0;
            vp[0] = qd;
            vs[0] = src[q];
            zb[0] = -inf;
            zb[1] = inf;
            continue;
        }
        double s = 0.0;
        for (;;) {
            const double vd = vp[k];
            const double fv = f[static_cast<int>(vd)];
            s = ((f[q] + qd * qd) - (fv + vd * vd)) / (2.0 * qd - 2.0 * vd);
            if (s > zb[k] || k == 0) break;
            --k;
        }
        ++k;
        vp[k] = qd;
        vs[k] = src[q];
        zb[k] = s;
        zb[k + 1] = inf;
    }
    if (k < 0) {
        for (int i = 0; i < n; ++i) {
            out[i] = inf;
            out_src[i] = -1;
        }
        return false;
    }
    int kk = 0;
    for (int i = 0; i < n; ++i) {
        const double id = static_cast<double>(i);
        while (zb[kk + 1] < id) ++kk;
        const double vd = vp[kk];
        const double diff = id - vd;
        out[i] = diff * diff + f[static_cast<int>(vd)];
        out_src[i] = vs[kk];
    }
    return true;
}

/// Marks the invalid cells reachable from the raster border by a 4-connected
/// walk that only steps on invalid cells. Used by `interior_only`.
std::vector<std::uint8_t> border_reachable(
    int w, int h, const std::vector<std::uint8_t>& valid) {
    const std::size_t n =
        static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    std::vector<std::uint8_t> reached(n, 0);
    std::vector<int> stack;
    const auto push = [&](int x, int y) {
        const std::size_t i = static_cast<std::size_t>(y) *
                                  static_cast<std::size_t>(w) +
                              static_cast<std::size_t>(x);
        if (valid[i] == 0 && reached[i] == 0) {
            reached[i] = 1;
            stack.push_back(x);
            stack.push_back(y);
        }
    };
    for (int x = 0; x < w; ++x) {
        push(x, 0);
        push(x, h - 1);
    }
    for (int y = 0; y < h; ++y) {
        push(0, y);
        push(w - 1, y);
    }
    while (!stack.empty()) {
        const int y = stack.back();
        stack.pop_back();
        const int x = stack.back();
        stack.pop_back();
        if (x > 0) push(x - 1, y);
        if (x + 1 < w) push(x + 1, y);
        if (y > 0) push(x, y - 1);
        if (y + 1 < h) push(x, y + 1);
    }
    return reached;
}

/// Inverse-distance-weighted value at (cx, cy) from the k nearest valid cells,
/// found by expanding rings in Chebyshev radius out to `idw_search_radius`.
/// Ring boundaries (never mid-ring) decide when enough neighbours were found,
/// so the candidate set does not depend on the scan order. Returns false when
/// no valid cell lies within the search radius.
bool idw_value(int w, int h, const std::vector<std::uint8_t>& valid,
               const std::vector<float>& values, int cx, int cy,
               const HoleFillOptions& o, float& out) {
    struct Cand {
        double d2;
        std::int32_t idx;
    };
    std::vector<Cand> cand;
    const double rmax = o.idw_search_radius;
    const double rmax2 = rmax * rmax;
    const int rings = static_cast<int>(std::ceil(rmax));
    const auto try_cell = [&](int x, int y) {
        if (x < 0 || x >= w || y < 0 || y >= h) return;
        const int dx = x - cx;
        const int dy = y - cy;
        const double d2 = static_cast<double>(dx) * dx +
                          static_cast<double>(dy) * dy;
        if (d2 > rmax2) return;
        const std::size_t i = static_cast<std::size_t>(y) *
                                  static_cast<std::size_t>(w) +
                              static_cast<std::size_t>(x);
        if (valid[i] == 0) return;
        cand.push_back(Cand{d2, static_cast<std::int32_t>(i)});
    };
    for (int r = 1; r <= rings; ++r) {
        const int yl = cy - r;
        const int yh = cy + r;
        for (int x = cx - r; x <= cx + r; ++x) {
            try_cell(x, yl);
            if (yh != yl) try_cell(x, yh);
        }
        for (int y = cy - r + 1; y <= cy + r - 1; ++y) {
            try_cell(cx - r, y);
            if (cx + r != cx - r) try_cell(cx + r, y);
        }
        if (static_cast<int>(cand.size()) >= o.idw_neighbours) break;
    }
    if (cand.empty()) return false;
    // Keep the k nearest, tie-broken by linear index for determinism.
    const auto cmp = [](const Cand& a, const Cand& b) {
        if (a.d2 != b.d2) return a.d2 < b.d2;
        return a.idx < b.idx;
    };
    if (static_cast<int>(cand.size()) > o.idw_neighbours) {
        const auto mid = cand.begin() + o.idw_neighbours;
        std::partial_sort(cand.begin(), mid, cand.end(), cmp);
        cand.resize(static_cast<std::size_t>(o.idw_neighbours));
    } else {
        std::sort(cand.begin(), cand.end(), cmp);
    }
    double wsum = 0.0;
    double vsum = 0.0;
    for (const Cand& c : cand) {
        const double d = std::sqrt(c.d2);  // >= 1: holes never coincide
        const double weight = 1.0 / std::pow(d, o.idw_power);
        wsum += weight;
        vsum += weight *
                static_cast<double>(values[static_cast<std::size_t>(c.idx)]);
    }
    if (wsum <= 0.0) return false;
    out = static_cast<float>(vsum / wsum);
    return true;
}

}  // namespace

NearestSites nearest_sites(int width, int height,
                           const std::vector<std::uint8_t>& valid) {
    if (width <= 0 || height <= 0) {
        throw ValueError("nearest_sites: width and height must be positive");
    }
    const std::size_t n = static_cast<std::size_t>(width) *
                          static_cast<std::size_t>(height);
    if (valid.size() != n) {
        throw ValueError(
            "nearest_sites: valid mask size does not match width*height");
    }
    const int W = width;
    const int H = height;
    const int N = W * H;
    const float finf = std::numeric_limits<float>::infinity();
    const double dinf = std::numeric_limits<double>::infinity();

    NearestSites res;
    res.width = W;
    res.height = H;
    res.distance.assign(static_cast<std::size_t>(N), finf);
    res.site.assign(static_cast<std::size_t>(N), -1);

    std::vector<double> d1(static_cast<std::size_t>(N), dinf);
    std::vector<std::int32_t> s1(static_cast<std::size_t>(N), -1);

    // Horizontal pass: nearest valid cell within each row.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < H; ++y) {
        const int base = y * W;
        int last = -1;
        for (int x = 0; x < W; ++x) {
            const int idx = base + x;
            if (valid[static_cast<std::size_t>(idx)] != 0) last = x;
            if (last >= 0) {
                const double d = static_cast<double>(x - last);
                d1[static_cast<std::size_t>(idx)] = d * d;
                s1[static_cast<std::size_t>(idx)] = base + last;
            }
        }
        int next = -1;
        for (int x = W - 1; x >= 0; --x) {
            const int idx = base + x;
            if (valid[static_cast<std::size_t>(idx)] != 0) next = x;
            if (next < 0) continue;
            const double d = static_cast<double>(next - x);
            const double dd = d * d;
            if (dd < d1[static_cast<std::size_t>(idx)]) {
                d1[static_cast<std::size_t>(idx)] = dd;
                s1[static_cast<std::size_t>(idx)] = base + next;
            }
        }
    }

    // Vertical pass: lower envelope over the rows, carrying the site payload.
    const std::size_t hs = static_cast<std::size_t>(H);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<double> f(hs);
        std::vector<std::int32_t> src(hs);
        std::vector<double> out(hs);
        std::vector<std::int32_t> outs(hs);
        std::vector<double> vp(hs + 1);
        std::vector<std::int32_t> vs(hs + 1);
        std::vector<double> zb(hs + 1);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int x = 0; x < W; ++x) {
            for (int y = 0; y < H; ++y) {
                const std::size_t idx =
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(W) +
                    static_cast<std::size_t>(x);
                f[static_cast<std::size_t>(y)] = d1[idx];
                src[static_cast<std::size_t>(y)] = s1[idx];
            }
            lower_envelope(f.data(), src.data(), H, out.data(), outs.data(),
                           vp.data(), vs.data(), zb.data());
            for (int y = 0; y < H; ++y) {
                const std::size_t idx =
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(W) +
                    static_cast<std::size_t>(x);
                res.distance[idx] =
                    static_cast<float>(std::sqrt(out[static_cast<std::size_t>(y)]));
                res.site[idx] = outs[static_cast<std::size_t>(y)];
            }
        }
    }
    return res;
}

HoleFillReport fill_holes(int width, int height, std::vector<float>& values,
                          std::vector<std::uint8_t>& valid,
                          const HoleFillOptions& options) {
    if (width <= 0 || height <= 0) {
        throw ValueError("fill_holes: width and height must be positive");
    }
    const std::size_t n = static_cast<std::size_t>(width) *
                          static_cast<std::size_t>(height);
    if (values.size() != n || valid.size() != n) {
        throw ValueError("fill_holes: buffer size does not match width*height");
    }
    if (options.idw_power < 0.0 || !std::isfinite(options.idw_power)) {
        throw ValueError("fill_holes: idw_power must be finite and >= 0");
    }
    if (options.idw_neighbours <= 0) {
        throw ValueError("fill_holes: idw_neighbours must be positive");
    }
    if (!(options.idw_search_radius > 0.0)) {
        throw ValueError("fill_holes: idw_search_radius must be positive");
    }
    if (options.max_distance < 0.0) {
        throw ValueError("fill_holes: max_distance must be >= 0");
    }
    if (options.sibson_exact_max_sites < 0) {
        throw ValueError("fill_holes: sibson_exact_max_sites must be >= 0");
    }

    const int W = width;
    const int H = height;
    const std::size_t total = n;

    const NearestSites edt = nearest_sites(W, H, valid);
    std::vector<std::uint8_t> blocked(total, 0);
    if (options.interior_only) blocked = border_reachable(W, H, valid);

    // Gate per hole, before filling: not too far, and (optionally) not on a
    // border-connected region.
    std::vector<std::uint8_t> fill(total, 0);
    for (std::size_t i = 0; i < total; ++i) {
        if (valid[i] != 0) continue;
        if (blocked[i] != 0) continue;
        const double d = static_cast<double>(edt.distance[i]);
        if (!std::isfinite(d)) continue;  // no data cell anywhere
        if (!(d <= options.max_distance)) continue;
        fill[i] = 1;
    }

    std::vector<float> result(total, 0.0f);

    // Nearest and IDW: each output cell is a pure function of the immutable
    // inputs (values and the EDT), so a plain parallel-for over the output cells
    // is deterministic by construction. Discrete Sibson is a scatter over the
    // source cells and is implemented separately below.
    if (options.method != HoleFill::NaturalNeighbour) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(total); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            if (fill[i] == 0) continue;

            if (options.method == HoleFill::Nearest) {
                result[i] = values[static_cast<std::size_t>(edt.site[i])];
                continue;
            }
            const int cx = static_cast<int>(i % static_cast<std::size_t>(W));
            const int cy = static_cast<int>(i / static_cast<std::size_t>(W));
            float v = 0.0f;
            if (idw_value(W, H, valid, values, cx, cy, options, v)) {
                result[i] = v;
            }
        }
    }

    if (options.method == HoleFill::NaturalNeighbour) {
        // Discrete Sibson as a scatter (Park et al. 2006, IEEE TVCG 12(2)).
        // A source cell p contributes its nearest site's value f(s(p)) to every
        // query cell q with |q - p| < r(p); the value at a hole cell is the
        // mean of the contributions it receives. Sources are exactly the cells
        // the caller is going to fill: every fillable hole cell p (r(p) may be
        // anything, including inside another fill region) and, with
        // interior_only, no border-connected hole cell. Data cells are NOT
        // sources — r = 0 would add f(s) to the data cell itself, which is
        // already correct and must stay bit-identical. Because fill[.] only
        // depends on the input mask, the set of sources is independent of the
        // thread count.
        std::vector<std::uint8_t> is_source = fill;

        // Bounding box of the target cells; votes are accumulated into private
        // per-band buffers covering just this box to keep the working set small.
        int min_x = W;
        int max_x = -1;
        int min_y = H;
        int max_y = -1;
        for (int y = 0; y < H; ++y) {
            const std::size_t base = static_cast<std::size_t>(y) *
                                     static_cast<std::size_t>(W);
            for (int x = 0; x < W; ++x) {
                if (fill[base + static_cast<std::size_t>(x)] == 0) continue;
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
        if (max_x >= 0) {
            const int bw = max_x - min_x + 1;
            const int bh = max_y - min_y + 1;
            // One accumulator per band of source rows. The band layout is
            // fixed (BANDS independent of the thread count), each non-empty
            // band owns a contiguous slice of the accumulator array, and bands
            // are merged in band order, so the floating-point summation order
            // does not depend on the thread count.
            const int nbands = std::min(64, H);
            const int rows_per_band = (H + nbands - 1) / nbands;
            std::vector<int> band_start(static_cast<std::size_t>(nbands),
                                        H);
            std::vector<int> band_end(static_cast<std::size_t>(nbands), H);
            std::vector<int> band_slot(static_cast<std::size_t>(nbands), -1);
            int nslots = 0;
            for (int b = 0; b < nbands; ++b) {
                int lo = b * rows_per_band;
                int hi = std::min(H, lo + rows_per_band);
                if (lo >= H) {
                    lo = H;
                    hi = H;
                }
                bool has = false;
                for (int y = lo; y < hi && !has; ++y) {
                    const std::size_t base = static_cast<std::size_t>(y) *
                                             static_cast<std::size_t>(W);
                    for (int x = 0; x < W; ++x) {
                        if (is_source[base + static_cast<std::size_t>(x)] != 0) {
                            has = true;
                            break;
                        }
                    }
                }
                band_start[static_cast<std::size_t>(b)] = lo;
                band_end[static_cast<std::size_t>(b)] = hi;
                if (has) {
                    band_slot[static_cast<std::size_t>(b)] = nslots;
                    ++nslots;
                }
            }

            const std::size_t slots = static_cast<std::size_t>(nslots);

            // MEMORY. The accumulators are `slots x (hole bbox)` cells: with 64
            // bands over a full-frame hole that is 12 bytes x 64 x the whole
            // grid -- tens of GB on a large orthomosaic. The target grid is
            // therefore processed in horizontal BANDS: only the rows of the
            // current band are accumulated, so the working set is
            // slots x bw x band_rows, capped at kSibsonBandBudgetBytes.
            //
            // Determinism and exactness are preserved by construction: within a
            // target band every source still scatters into its own slot in the
            // same (band, row, col) order as before, and the per-target merge
            // still runs slots in band order, so every sum is bit-identical to
            // the unbanded algorithm.
            constexpr std::size_t kSibsonBandBudgetBytes = 64u * 1024u * 1024u;
            constexpr std::size_t kBytesPerCell = sizeof(double) + sizeof(std::int32_t);
            const std::size_t per_row = slots * static_cast<std::size_t>(bw) * kBytesPerCell;
            std::size_t band_rows = per_row > 0 ? kSibsonBandBudgetBytes / per_row : 0;
            if (band_rows < 1) {
                band_rows = 1;  // one band row is the floor; nothing else fits
            }
            if (band_rows > static_cast<std::size_t>(bh)) {
                band_rows = static_cast<std::size_t>(bh);
            }

            // The largest scatter radius, so a target band only reads the source
            // rows that can reach it.
            int max_radius = 0;
            for (std::size_t i = 0; i < total; ++i) {
                if (is_source[i] == 0) continue;
                const double d = static_cast<double>(edt.distance[i]);
                if (!std::isfinite(d)) continue;
                const int ri = isqrt(static_cast<double>(
                    static_cast<std::int64_t>(std::llround(d * d))));
                if (ri > max_radius) max_radius = ri;
            }

            std::vector<double> sums(slots * static_cast<std::size_t>(bw) * band_rows, 0.0);
            std::vector<std::int32_t> counts(sums.size(), 0);

            for (std::size_t band_lo = 0; band_lo < static_cast<std::size_t>(bh);
                 band_lo += band_rows) {
                const int tlo = min_y + static_cast<int>(band_lo);
                const int thi = std::min(max_y + 1, tlo + static_cast<int>(band_rows));
                const std::size_t rows =
                    static_cast<std::size_t>(std::max(0, thi - tlo));
                const std::size_t bsz = static_cast<std::size_t>(bw) * rows;
                std::fill(sums.begin(), sums.end(), 0.0);
                std::fill(counts.begin(), counts.end(), 0);

                // Only source rows whose scatter ball can reach this band.
                const int src_lo = std::max(0, tlo - max_radius);
                const int src_hi = std::min(H, thi + max_radius);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int b = 0; b < nbands; ++b) {
                    const int slot = band_slot[static_cast<std::size_t>(b)];
                    if (slot < 0) continue;
                    double* const sbuf =
                        sums.data() + static_cast<std::size_t>(slot) * bsz;
                    std::int32_t* const cbuf =
                        counts.data() + static_cast<std::size_t>(slot) * bsz;
                    for (int py = std::max(band_start[static_cast<std::size_t>(b)], src_lo);
                         py < std::min(band_end[static_cast<std::size_t>(b)], src_hi); ++py) {
                        const std::size_t pbase = static_cast<std::size_t>(py) *
                                                  static_cast<std::size_t>(W);
                        for (int px = 0; px < W; ++px) {
                            const std::size_t pi = pbase +
                                                   static_cast<std::size_t>(px);
                            if (is_source[pi] == 0) continue;
                            const std::int32_t s = edt.site[pi];
                            if (s < 0) continue;
                            const double d = static_cast<double>(edt.distance[pi]);
                            if (!std::isfinite(d)) continue;
                            // r^2 is an exact integer (the EDT stores sqrt of one).
                            const std::int64_t r2 =
                                static_cast<std::int64_t>(std::llround(d * d));
                            const int ri = isqrt(static_cast<double>(r2));
                            const double f = static_cast<double>(
                                values[static_cast<std::size_t>(s)]);
                            const int ylo = std::max(tlo, py - ri);
                            const int yhi = std::min(thi - 1, py + ri);
                            const int xlo = std::max(min_x, px - ri);
                            const int xhi = std::min(max_x, px + ri);
                            if (ylo > yhi) continue;
                            for (int qy = ylo; qy <= yhi; ++qy) {
                                const std::int64_t dy = qy - py;
                                const std::int64_t dy2 = dy * dy;
                                double* const srow = sbuf +
                                    static_cast<std::size_t>(qy - tlo) *
                                        static_cast<std::size_t>(bw);
                                std::int32_t* const crow = cbuf +
                                    static_cast<std::size_t>(qy - tlo) *
                                        static_cast<std::size_t>(bw);
                                for (int qx = xlo; qx <= xhi; ++qx) {
                                    const std::int64_t dx = qx - px;
                                    // Strict interior of the ball: a cell exactly on
                                    // the sphere lies on the boundary of the site's
                                    // cell and is not stolen from it.
                                    if (dx * dx + dy2 >= r2) continue;
                                    const std::size_t bidx = static_cast<std::size_t>(
                                        qx - min_x);
                                    srow[bidx] += f;
                                    ++crow[bidx];
                                }
                            }
                        }
                    }
                }

                // Merge in fixed band order, then divide. A target always receives
                // its own vote (0 < r(p)^2 for a hole cell), so no count is zero.
                for (int y = tlo; y < thi; ++y) {
                    const std::size_t base = static_cast<std::size_t>(y) *
                                             static_cast<std::size_t>(W);
                    const std::size_t brow =
                        static_cast<std::size_t>(y - tlo) *
                        static_cast<std::size_t>(bw);
                    for (int x = min_x; x <= max_x; ++x) {
                        const std::size_t i = base + static_cast<std::size_t>(x);
                        if (fill[i] == 0) continue;
                        const std::size_t bidx =
                            brow + static_cast<std::size_t>(x - min_x);
                        double sum = 0.0;
                        std::int64_t cnt = 0;
                        for (int b = 0; b < nbands; ++b) {
                            const int slot = band_slot[static_cast<std::size_t>(b)];
                            if (slot < 0) continue;
                            const std::size_t o =
                                static_cast<std::size_t>(slot) * bsz + bidx;
                            sum += sums[o];
                            cnt += static_cast<std::int64_t>(counts[o]);
                        }
                        if (cnt > 0) {
                            result[i] =
                                static_cast<float>(sum / static_cast<double>(cnt));
                        }
                    }
                }
            }
        }
    }

    if (options.method == HoleFill::NaturalNeighbour && options.sibson_exact_max_sites > 0) {
        exact_sibson_fill(W, H, values, valid, fill, result, options.sibson_exact_max_sites);
    }

    std::size_t filled = 0;
    std::size_t remaining = 0;
    for (std::size_t i = 0; i < total; ++i) {
        if (valid[i] != 0) continue;
        if (fill[i] == 0) {  // gated out by max_distance / interior_only
            ++remaining;
            continue;
        }
        values[i] = result[i];
        valid[i] = 1;
        ++filled;
    }
    return HoleFillReport{filled, remaining};
}

}  // namespace alphashape::elevation
