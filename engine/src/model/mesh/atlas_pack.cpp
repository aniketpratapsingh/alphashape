#include "model/mesh/atlas_pack.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"

namespace alphashape {
namespace {

using Vec2 = Eigen::Vector2d;

/// Two world-space points closer than this count as the same hull vertex.
constexpr double kHullEps = 1e-12;

/// A chart flattened into its own rotated, axis-aligned frame.
///
/// `local[i]` is corner i (same order as Chart::corner_uv) expressed in a frame
/// whose axes are (ax, ay) — a right-handed pair, so mapping into it is a
/// rotation and never a reflection: signed areas keep their sign. The rectangle
/// is landscape (`width >= height`) and has its lower-left corner at (0, 0).
struct LocalChart {
    double width = 0.0;   ///< rectangle extent along the frame's +x; >= height.
    double height = 0.0;  ///< rectangle extent along the frame's +y.
    std::vector<Vec2> local;  ///< per-corner frame coords, each in [0,w]x[0,h].
};

/// Convex hull of `pts` (monotone chain), counter-clockwise, collinear interior
/// points dropped and duplicates merged. Fewer than 3 points come back for
/// degenerate input (a single point, or a segment).
std::vector<Vec2> convex_hull(std::vector<Vec2> pts) {
    std::sort(pts.begin(), pts.end(), [](const Vec2& a, const Vec2& b) {
        if (a.x() != b.x()) return a.x() < b.x();
        return a.y() < b.y();
    });
    pts.erase(std::unique(pts.begin(), pts.end(),
                          [](const Vec2& a, const Vec2& b) {
                              return (a - b).squaredNorm() <= kHullEps * kHullEps;
                          }),
              pts.end());
    const std::size_t n = pts.size();
    if (n <= 2) return pts;
    std::vector<Vec2> hull(2 * n);
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {  // lower hull, keeping left turns
        while (k >= 2) {
            const Vec2 d1 = hull[k - 1] - hull[k - 2];
            const Vec2 d2 = pts[i] - hull[k - 1];
            if (d1.x() * d2.y() - d1.y() * d2.x() > 0.0) break;
            --k;
        }
        hull[k++] = pts[i];
    }
    const std::size_t lower = k + 1;
    for (std::size_t i = n - 1; i-- > 0;) {  // upper hull
        while (k >= lower) {
            const Vec2 d1 = hull[k - 1] - hull[k - 2];
            const Vec2 d2 = pts[i] - hull[k - 1];
            if (d1.x() * d2.y() - d1.y() * d2.x() > 0.0) break;
            --k;
        }
        hull[k++] = pts[i];
    }
    hull.resize(k - 1);  // the closing point repeats the first
    return hull;
}

/// Express `pts` in a frame whose axes are the orthonormal pair (ax, ay),
/// choosing the landscape orientation so the rect is at least as wide as tall.
LocalChart frame_from_axes(const std::vector<Vec2>& pts, Vec2 ax, Vec2 ay) {
    double min_a = std::numeric_limits<double>::max();
    double max_a = std::numeric_limits<double>::lowest();
    double min_b = std::numeric_limits<double>::max();
    double max_b = std::numeric_limits<double>::lowest();
    for (const Vec2& p : pts) {
        const double a = p.dot(ax);
        const double b = p.dot(ay);
        min_a = std::min(min_a, a);
        max_a = std::max(max_a, a);
        min_b = std::min(min_b, b);
        max_b = std::max(max_b, b);
    }
    Vec2 fax = ax;
    Vec2 fay = ay;
    if (max_b - min_b > max_a - min_a) {
        // Taller than wide: take the same rectangle in the frame rotated +90°
        // degrees ((a, b) -> (b, -a)), which is landscape by construction.
        fax = ay;
        fay = -ax;
        const double la = min_b;
        const double ha = max_b;
        min_b = -max_a;
        max_b = -min_a;
        min_a = la;
        max_a = ha;
    }
    LocalChart out;
    out.width = max_a - min_a;
    out.height = max_b - min_b;
    out.local.reserve(pts.size());
    for (const Vec2& p : pts) {
        out.local.push_back(Vec2(p.dot(fax) - min_a, p.dot(fay) - min_b));
    }
    return out;
}

/// Minimum-area enclosing rectangle by rotating calipers over the hull's edge
/// directions, returned in its own landscape frame.
LocalChart fit_min_area_rect(const std::vector<Vec2>& pts) {
    if (pts.size() < 3) {
        // Degenerate input: align with the segment (or x for a point); the
        // rectangle collapses to a line and contributes no area.
        Vec2 dir = Vec2::UnitX();
        if (pts.size() == 2) {
            const Vec2 d = pts[1] - pts[0];
            const double len = d.norm();
            if (len > 0.0) dir = d / len;
        }
        return frame_from_axes(pts, dir, Vec2(-dir.y(), dir.x()));
    }
    const std::vector<Vec2> hull = convex_hull(pts);
    if (hull.size() < 3) return frame_from_axes(pts, Vec2::UnitX(), Vec2::UnitY());

    Vec2 best_ax = Vec2::UnitX();
    Vec2 best_ay = Vec2::UnitY();
    double best_area = -1.0;
    for (std::size_t i = 0; i < hull.size(); ++i) {
        const Vec2 edge = hull[(i + 1) % hull.size()] - hull[i];
        const double len = edge.norm();
        if (len <= kHullEps) continue;
        const Vec2 dir = edge / len;
        const Vec2 perp(-dir.y(), dir.x());
        double min_a = std::numeric_limits<double>::max();
        double max_a = std::numeric_limits<double>::lowest();
        double min_b = std::numeric_limits<double>::max();
        double max_b = std::numeric_limits<double>::lowest();
        for (const Vec2& p : hull) {
            const double a = p.dot(dir);
            const double b = p.dot(perp);
            min_a = std::min(min_a, a);
            max_a = std::max(max_a, a);
            min_b = std::min(min_b, b);
            max_b = std::max(max_b, b);
        }
        const double area = (max_a - min_a) * (max_b - min_b);
        if (best_area < 0.0 || area < best_area) {
            best_area = area;
            best_ax = dir;
            best_ay = perp;
        }
    }
    return frame_from_axes(pts, best_ax, best_ay);
}

/// A rectangle to place, already rounded up to whole texels. Rectangles are
/// stored at their chart's index, so a placement answer is addressed by chart.
struct PackRect {
    int width = 0;   ///< texels
    int height = 0;  ///< texels
};

/// One constant-height run of the skyline, covering [x, next.x). The list is
/// kept contiguous from x = 0 and adjacent runs never share a height, so the
/// run count is bounded by the number of rectangles placed so far.
struct SkyRun {
    int x = 0;
    int height = 0;
};

/// Bottom-left skyline packer on a `page x page` texel square.
///
/// `order` lists the rectangles to place (charts sorted by height desc, width
/// desc, index), so the tall ones land first and the skyline stays flat. Each
/// rectangle spans a range of skyline runs and rests on the tallest run it
/// touches, exactly as an array-based skyline would, but the skyline is kept as
/// a compact list of constant-height runs so the cost is O(runs) per item
/// rather than O(page). Candidates are the run starts (the left-most texel of a
/// flat stretch) and their exact-fit partners (a right edge landing on a run
/// start), which is enough for bottom-left on integer texels. Returns the
/// lower-left texel corner of each rectangle (padding included), or an empty
/// vector when a rectangle does not fit.
///
/// Deterministic by construction: no hashing, no pointer ordering, and ties
/// (equal rest height) keep the earliest scanned candidate x.
std::vector<Eigen::Vector2i> skyline_pack(const std::vector<PackRect>& rects,
                                          const std::vector<int>& order, int page) {
    std::vector<SkyRun> runs;
    runs.push_back(SkyRun{0, 0});
    std::vector<int> candidates;  // reused across items to keep allocations flat
    std::vector<Eigen::Vector2i> origin(rects.size(), Eigen::Vector2i::Zero());
    for (const int idx : order) {
        const PackRect& r = rects[static_cast<std::size_t>(idx)];
        if (r.width > page || r.height > page) return {};

        // Candidate left edges: every run start (the left-most texel of a flat
        // stretch) and every position whose right edge lands on a run start (an
        // exact fit into a gap). Both sequences are ascending, so a plain merge
        // keeps the scan in x order — which is what makes "the earliest
        // candidate wins a tie" deterministic.
        candidates.clear();
        for (const SkyRun& run : runs) candidates.push_back(run.x);
        const std::size_t starts = candidates.size();
        for (const SkyRun& run : runs) candidates.push_back(run.x - r.width);

        int lowest = page + 1;
        for (const SkyRun& run : runs) lowest = std::min(lowest, run.height);

        int best_x = -1;
        int best_y = page + 1;
        std::size_t hit = 0;  // run covering the candidate under consideration
        std::size_t ia = 0;
        std::size_t ib = starts;
        int previous_x = std::numeric_limits<int>::min();
        while (ia < starts || ib < candidates.size()) {
            int x = 0;
            if (ia == starts) {
                x = candidates[ib++];
            } else if (ib == candidates.size()) {
                x = candidates[ia++];
            } else if (candidates[ia] <= candidates[ib]) {
                x = candidates[ia++];
            } else {
                x = candidates[ib++];
            }
            if (x == previous_x) continue;  // a run start that is also an exact fit
            previous_x = x;
            if (x < 0 || x + r.width > page) continue;
            // The rectangle rests on the tallest run its span touches. The run
            // covering x is the first one starting at or before it.
            while (hit + 1 < runs.size() && runs[hit + 1].x <= x) ++hit;
            int rest = runs[hit].height;
            for (std::size_t j = hit + 1; j < runs.size() && runs[j].x < x + r.width; ++j) {
                rest = std::max(rest, runs[j].height);
            }
            if (rest + r.height > page) continue;
            if (rest < best_y) {
                best_y = rest;
                best_x = x;
                // Nothing rests lower than the skyline's global minimum, so no
                // later candidate can beat this height; the earliest x wins.
                if (rest == lowest) break;
            }
        }
        if (best_x < 0) return {};
        origin[static_cast<std::size_t>(idx)] = Eigen::Vector2i(best_x, best_y);

        // Raise [best_x, end_x) to the new top. The region to the right of the
        // rectangle is untouched, so the run that was covering end_x must be
        // re-emitted there: the list is only a *finite* set of steps, and the
        // skyline continues at the last step's height past the last step start.
        const int top = best_y + r.height;
        const int end_x = best_x + r.width;
        int tail_height = runs.back().height;
        for (const SkyRun& run : runs) {
            if (run.x > end_x) break;  // runs ascend, so this was the last one
            tail_height = run.height;
        }
        std::vector<SkyRun> next;
        next.reserve(runs.size() + 2);
        std::size_t i = 0;
        for (; i < runs.size() && runs[i].x < best_x; ++i) next.push_back(runs[i]);
        next.push_back(SkyRun{best_x, top});
        // Runs starting inside [best_x, end_x) are absorbed by the rectangle.
        while (i < runs.size() && runs[i].x < end_x) ++i;
        if ((i == runs.size() || runs[i].x > end_x) && tail_height != top) {
            next.push_back(SkyRun{end_x, tail_height});
        }
        for (; i < runs.size(); ++i) next.push_back(runs[i]);
        // Merge neighbours that now share a height so the run count stays small.
        runs.clear();
        runs.reserve(next.size());
        for (const SkyRun& run : next) {
            if (!runs.empty() && runs.back().height == run.height) continue;
            runs.push_back(run);
        }
    }
    return origin;
}

/// Try to place every chart at a single uniform scale `s` (texels per world
/// unit). Each chart's rectangle is its content box grown by `pad` texels and
/// rounded UP to whole texels, so the gap between two placed charts is a whole
/// number of texels, never less than `pad`. Returns false when any rectangle
/// misses the page.
bool pack_at_scale(const std::vector<LocalChart>& local, const std::vector<int>& order,
                   double s, int pad, int page, std::vector<Eigen::Vector2i>& out) {
    if (!(s > 0.0) || !std::isfinite(s)) return false;
    std::vector<PackRect> rects;
    rects.reserve(local.size());
    for (std::size_t i = 0; i < local.size(); ++i) {
        const LocalChart& c = local[i];
        // The rectangle is the content box grown by `pad` texels and rounded UP
        // to whole texels, so two neighbours are separated by at least `pad`.
        const double w = std::ceil(c.width * s) + static_cast<double>(pad);
        const double h = std::ceil(c.height * s) + static_cast<double>(pad);
        // A padded rectangle wider than twice the page is hopeless at any scale,
        // and the bound keeps the conversions below in range.
        if (w > static_cast<double>(page) * 2.0 || h > static_cast<double>(page) * 2.0) {
            return false;
        }
        PackRect r;
        r.width = static_cast<int>(w);
        r.height = static_cast<int>(h);
        if (r.width > page || r.height > page) return false;
        rects.push_back(r);
    }
    std::vector<Eigen::Vector2i> placed = skyline_pack(rects, order, page);
    if (placed.size() != rects.size()) return false;
    out = std::move(placed);
    return true;
}

}  // namespace

PackResult pack_atlas(Mesh& mesh, const std::vector<Chart>& charts, const PackOptions& options) {
    const int page = options.texture_size;
    const int pad = options.padding_px;
    if (page < 16) {
        throw ValueError("pack_atlas: texture_size must be at least 16");
    }
    if (pad < 0) {
        throw ValueError("pack_atlas: padding_px must be non-negative");
    }

    PackResult result;
    if (mesh.faces.empty()) {
        // Nothing to texture. A chart that names a face is then inconsistent;
        // an empty chart list (or all-empty charts) is a clean no-op.
        for (std::size_t ci = 0; ci < charts.size(); ++ci) {
            if (!charts[ci].faces.empty()) {
                throw ValueError("pack_atlas: chart " + std::to_string(ci) +
                                 " references a face of an empty mesh");
            }
            if (!charts[ci].corner_uv.empty()) {
                throw ValueError("pack_atlas: chart " + std::to_string(ci) +
                                 " has corner coordinates but no faces");
            }
        }
        mesh.uvs.clear();
        mesh.uv_faces.clear();
        return result;
    }

    const int face_count = static_cast<int>(mesh.faces.size());
    const int vertex_count = static_cast<int>(mesh.vertices.size());

    // ---- Stage 1: validate the chart partition ---------------------------
    std::vector<int> owner(static_cast<std::size_t>(face_count), -1);
    for (std::size_t ci = 0; ci < charts.size(); ++ci) {
        const Chart& chart = charts[ci];
        const std::size_t nf = chart.faces.size();
        if (chart.corner_uv.size() != 3 * nf) {
            throw ValueError("pack_atlas: chart " + std::to_string(ci) + " has " +
                             std::to_string(chart.corner_uv.size()) +
                             " corner coordinates for " + std::to_string(nf) + " faces");
        }
        for (const int f : chart.faces) {
            if (f < 0 || f >= face_count) {
                throw ValueError("pack_atlas: chart " + std::to_string(ci) +
                                 " references face " + std::to_string(f) +
                                 " which is out of range");
            }
            if (owner[static_cast<std::size_t>(f)] == static_cast<int>(ci)) {
                throw ValueError("pack_atlas: face " + std::to_string(f) +
                                 " appears twice in chart " + std::to_string(ci));
            }
            if (owner[static_cast<std::size_t>(f)] >= 0) {
                throw ValueError("pack_atlas: face " + std::to_string(f) +
                                 " appears in more than one chart");
            }
            owner[static_cast<std::size_t>(f)] = static_cast<int>(ci);
        }
    }
    for (int f = 0; f < face_count; ++f) {
        if (owner[static_cast<std::size_t>(f)] < 0) {
            throw ValueError("pack_atlas: face " + std::to_string(f) +
                             " is not in any chart");
        }
        const Eigen::Vector3i& tri = mesh.faces[static_cast<std::size_t>(f)];
        if (tri.minCoeff() < 0 || tri.maxCoeff() >= vertex_count) {
            throw ValueError("pack_atlas: face " + std::to_string(f) +
                             " references a vertex out of range");
        }
    }

    // ---- Stage 2: rotate each chart to its minimum-area rectangle --------
    std::vector<std::size_t> chart_of;  // flattened index -> index into `charts`
    std::vector<LocalChart> local;
    for (std::size_t ci = 0; ci < charts.size(); ++ci) {
        if (charts[ci].faces.empty()) continue;  // an empty chart occupies nothing
        local.push_back(fit_min_area_rect(charts[ci].corner_uv));
        chart_of.push_back(ci);
    }
    if (local.empty()) {
        mesh.uvs.clear();
        mesh.uv_faces.clear();
        return result;
    }

    // Placement order: tallest first, then widest, then chart index. Greedy
    // bottom-left packers do best when the tall items go down first, and the
    // index tie-break keeps the whole pass deterministic.
    std::vector<int> order(local.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&local](int a, int b) {
        const LocalChart& x = local[static_cast<std::size_t>(a)];
        const LocalChart& y = local[static_cast<std::size_t>(b)];
        if (x.height != y.height) return x.height > y.height;
        if (x.width != y.width) return x.width > y.width;
        return a < b;
    });

    std::vector<Eigen::Vector2i> placed;
    const auto feasible = [&](double s) {
        std::vector<Eigen::Vector2i> attempt;
        if (!pack_at_scale(local, order, s, pad, page, attempt)) return false;
        placed = std::move(attempt);
        return true;
    };

    // ---- Stage 3: one shared scale, largest that still fits --------------
    double total_rect_area = 0.0;
    for (const LocalChart& c : local) total_rect_area += c.width * c.height;
    constexpr double kFillTarget = 0.7;
    const double start = (total_rect_area > 0.0)
                             ? static_cast<double>(page) *
                                   std::sqrt(kFillTarget / total_rect_area)
                             : 1.0;
    const double seed = (start > 0.0 && std::isfinite(start)) ? start : 1.0;

    // Bracket the answer with a known-fitting `lo` and a known-failing `hi`.
    // Rectangles only grow with the scale and the page is fixed, so feasibility
    // is monotone in s and a bisection finds the largest fitting scale; if even
    // a scale far below the seed cannot be placed, no scale can, which is the
    // one genuinely impossible case (a chart whose padding alone exceeds the
    // page, or more charts than the page has room for).
    double lo = seed;
    double hi = 0.0;
    if (feasible(seed)) {
        hi = seed * 2.0;
        constexpr double kScaleCeiling = 1e15;
        while (hi < kScaleCeiling && feasible(hi)) hi *= 2.0;
    } else {
        hi = seed;
        bool found = false;
        for (int i = 0; i < 200 && !found; ++i) {
            lo *= 0.5;
            found = (lo > 0.0) && feasible(lo);
        }
        if (!found) {
            throw ValueError("pack_atlas: chart does not fit the texture page");
        }
    }
    for (int i = 0; i < 64; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (mid <= lo || mid >= hi) break;
        if (feasible(mid)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const double scale = lo;
    // Re-run at the winning scale so `placed` is unambiguously that attempt.
    if (!feasible(scale)) {
        throw ValueError("pack_atlas: internal packing failure at the chosen scale");
    }

    // ---- Stage 4: write the texture coordinates --------------------------
    mesh.uvs.clear();
    mesh.uv_faces.assign(static_cast<std::size_t>(face_count), Eigen::Vector3i::Zero());
    const double inv_page = 1.0 / static_cast<double>(page);
    const double half_pad = 0.5 * static_cast<double>(pad);
    double chart_area_texels = 0.0;

    for (std::size_t i = 0; i < local.size(); ++i) {
        const std::size_t ci = chart_of[i];
        const Chart& chart = charts[ci];
        const LocalChart& lc = local[i];
        const Eigen::Vector2i origin = placed[i];
        const double ox = static_cast<double>(origin.x()) + half_pad;
        const double oy = static_cast<double>(origin.y()) + half_pad;

        // Corners sharing a mesh vertex inside one chart share a pool entry, so
        // the interior edges of a chart cost no extra coordinate. The table is
        // per-chart: across charts the surface is cut, so the two sides of a
        // seam must keep independent coordinates.
        std::vector<int> vertex_to_pool;

        const std::size_t nf = chart.faces.size();
        for (std::size_t k = 0; k < nf; ++k) {
            const int f = chart.faces[k];
            const Eigen::Vector3i& tri = mesh.faces[static_cast<std::size_t>(f)];
            // Chart::corner_uv and Mesh::faces list (v0, v1, v2) in the same
            // order, so corner `corner` of the face uses corner_uv[3k + corner].
            // The frame-local table is in that same corner order, so both index
            // identically.
            Eigen::Vector3i pool_index;
            for (int corner = 0; corner < 3; ++corner) {
                const int vid = tri[corner];
                int entry = -1;
                if (vid >= 0) {
                    if (static_cast<std::size_t>(vid) >= vertex_to_pool.size()) {
                        vertex_to_pool.resize(static_cast<std::size_t>(vid) + 1, -1);
                    }
                    entry = vertex_to_pool[static_cast<std::size_t>(vid)];
                }
                if (entry < 0) {
                    const Vec2& p =
                        lc.local[3 * k + static_cast<std::size_t>(corner)];
                    // Frame coords -> texels -> normalised page. v grows down,
                    // and the local frame is y-down as well, so no flip is
                    // needed.
                    const double tx = ox + p.x() * scale;
                    const double ty = oy + p.y() * scale;
                    entry = static_cast<int>(mesh.uvs.size());
                    mesh.uvs.emplace_back(static_cast<float>(tx * inv_page),
                                          static_cast<float>(ty * inv_page));
                    if (vid >= 0) {
                        vertex_to_pool[static_cast<std::size_t>(vid)] = entry;
                    }
                }
                pool_index[corner] = entry;
            }
            mesh.uv_faces[static_cast<std::size_t>(f)] = pool_index;
            // The chart's interior area in texels, for the fill ratio. Corner
            // coordinates are world units, so scaling the area by s^2 gives it.
            const Vec2& a = chart.corner_uv[3 * k];
            const Vec2& b = chart.corner_uv[3 * k + 1];
            const Vec2& c = chart.corner_uv[3 * k + 2];
            const double doubled = (b - a).x() * (c - a).y() - (b - a).y() * (c - a).x();
            chart_area_texels += std::abs(doubled) * 0.5 * scale * scale;
        }
    }

    result.texels_per_unit = scale;
    const double page_area = static_cast<double>(page) * static_cast<double>(page);
    result.fill_ratio = (page_area > 0.0) ? std::min(1.0, chart_area_texels / page_area) : 0.0;
    return result;
}

}  // namespace alphashape
