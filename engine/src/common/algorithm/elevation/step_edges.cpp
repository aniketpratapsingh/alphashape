#include "common/algorithm/elevation/step_edges.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "base/utils/error.hpp"

namespace alphashape::elevation {
namespace {

/// cos(15 degrees): the orientation tolerance for chaining candidates.
constexpr double kCos15 = 0.96592582628906831;

/// A fitted plane h = a*x + b*y + c over (cell-column, cell-row) coordinates.
struct PlaneFit {
    bool ok = false;
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double rms = 0.0;
};

/// One accepted discontinuity: a line (point + unit direction), the height
/// range along it, and the plane on each side (left = negative signed
/// distance, right = positive).
struct Segment {
    double px = 0.0;
    double py = 0.0;
    double dx = 1.0;
    double dy = 0.0;
    double tmin = 0.0;
    double tmax = 0.0;
    PlaneFit left;
    PlaneFit right;
    double lo = 0.0;
    double hi = 0.0;
};

/// Solve the 3x3 system m*x = rhs by Gaussian elimination with partial
/// pivoting. Returns false when the matrix is numerically singular.
bool solve3(double m[3][3], double rhs[3], double out[3]) {
    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 3; ++row) {
            if (std::abs(m[row][col]) > std::abs(m[pivot][col])) {
                pivot = row;
            }
        }
        if (std::abs(m[pivot][col]) < 1e-12) {
            return false;
        }
        if (pivot != col) {
            for (int k = col; k < 3; ++k) {
                std::swap(m[col][k], m[pivot][k]);
            }
            std::swap(rhs[col], rhs[pivot]);
        }
        const double inv = 1.0 / m[col][col];
        for (int row = col + 1; row < 3; ++row) {
            const double f = m[row][col] * inv;
            if (f == 0.0) {
                continue;
            }
            for (int k = col; k < 3; ++k) {
                m[row][k] -= f * m[col][k];
            }
            rhs[row] -= f * rhs[col];
        }
    }
    for (int row = 2; row >= 0; --row) {
        double sum = rhs[row];
        for (int k = row + 1; k < 3; ++k) {
            sum -= m[row][k] * out[k];
        }
        out[row] = sum / m[row][row];
    }
    return true;
}

/// Weighted least-squares plane fit. An empty `w` means unit weights.
bool solve_plane(const std::vector<double>& x, const std::vector<double>& y,
                 const std::vector<double>& h, const std::vector<double>& w, double out[3]) {
    double a[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    double rhs[3] = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double wi = w.empty() ? 1.0 : w[i];
        const double row[3] = {x[i], y[i], 1.0};
        for (int r = 0; r < 3; ++r) {
            rhs[r] += wi * row[r] * h[i];
            for (int c = 0; c < 3; ++c) {
                a[r][c] += wi * row[r] * row[c];
            }
        }
    }
    return solve3(a, rhs, out);
}

double median_of(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid),
                     values.end());
    double value = values[mid];
    if (values.size() % 2 == 0) {
        const auto lower_end =
            values.begin() + static_cast<std::ptrdiff_t>(mid);
        value = 0.5 * (value + *std::max_element(values.begin(), lower_end));
    }
    return value;
}

/// Linear-interpolated percentile `q` in [0, 1]. Sorts the passed copy.
double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double clamped = std::min(std::max(q, 0.0), 1.0);
    const double position =
        clamped * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(position));
    const std::size_t hi = std::min(lo + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lo);
    return values[lo] + (values[hi] - values[lo]) * fraction;
}

/// Huber IRLS plane fit (Huber 1964). The robust scale is the MAD of the
/// residuals with a floor, so an exact plane does not get an infinite weight.
PlaneFit fit_plane_huber(const std::vector<double>& x, const std::vector<double>& y,
                         const std::vector<double>& h) {
    PlaneFit fit;
    const std::size_t n = x.size();
    if (n < 6) {
        return fit;
    }
    double p[3] = {0.0, 0.0, 0.0};
    if (!solve_plane(x, y, h, {}, p)) {
        return fit;
    }
    std::vector<double> w(n, 1.0);
    std::vector<double> r(n, 0.0);
    for (int iter = 0; iter < 8; ++iter) {
        std::vector<double> abs_r(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            r[i] = p[0] * x[i] + p[1] * y[i] + p[2] - h[i];
            abs_r[i] = std::abs(r[i]);
        }
        const double scale = std::max(0.05, 1.4826 * median_of(abs_r));
        const double delta = 2.0 * scale;
        bool changed = false;
        for (std::size_t i = 0; i < n; ++i) {
            const double next = std::abs(r[i]) <= delta ? 1.0 : delta / std::abs(r[i]);
            changed = changed || std::abs(next - w[i]) > 1e-6;
            w[i] = next;
        }
        if (!solve_plane(x, y, h, w, p)) {
            return fit;
        }
        if (!changed) {
            break;
        }
    }
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double ri = p[0] * x[i] + p[1] * y[i] + p[2] - h[i];
        sum_sq += ri * ri;
    }
    fit.ok = true;
    fit.a = p[0];
    fit.b = p[1];
    fit.c = p[2];
    fit.rms = std::sqrt(sum_sq / static_cast<double>(n));
    return fit;
}

/// Read-only view of the height field used by the segment builder.
struct Grid {
    int width = 0;
    int height = 0;
    double cell_size_x = 1.0;
    double cell_size_y = 1.0;
    const std::vector<double>* h = nullptr;
    const std::vector<std::uint8_t>* valid = nullptr;

    bool inside(int col, int row) const {
        return col >= 0 && row >= 0 && col < width && row < height;
    }
    bool is_valid(int col, int row) const {
        return inside(col, row) &&
               (*valid)[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                        static_cast<std::size_t>(col)] != 0U;
    }
    double height_at(int col, int row) const {
        return (*h)[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                    static_cast<std::size_t>(col)];
    }
};

double plane_value(const PlaneFit& plane, double x, double y) {
    return plane.a * x + plane.b * y + plane.c;
}

/// Turn one orientation-grown component into a validated segment: a total
/// least-squares line (PCA of the cell centres) and one Huber plane per side.
bool build_segment(std::span<const int> cells, const Grid& grid, double min_height,
                   double min_length, int band, Segment& out) {
    if (cells.size() < 6U) {
        return false;
    }
    double mx = 0.0;
    double my = 0.0;
    for (const int index : cells) {
        mx += static_cast<double>(index % grid.width);
        my += static_cast<double>(index / grid.width);
    }
    const double n = static_cast<double>(cells.size());
    mx /= n;
    my /= n;
    double sxx = 0.0;
    double syy = 0.0;
    double sxy = 0.0;
    for (const int index : cells) {
        const double dx = static_cast<double>(index % grid.width) - mx;
        const double dy = static_cast<double>(index / grid.width) - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    const double angle = 0.5 * std::atan2(2.0 * sxy, sxx - syy);
    const double dx = std::cos(angle);
    const double dy = std::sin(angle);
    const double nx = -dy;
    const double ny = dx;
    // The extent comes from tight inliers only: candidates on a perpendicular
    // edge are up to a few cells off this line, and letting them set tmin/tmax
    // made the snap paint past the measured segment.
    std::vector<double> ts;
    ts.reserve(cells.size());
    for (const int index : cells) {
        const double px = static_cast<double>(index % grid.width) - mx;
        const double py = static_cast<double>(index / grid.width) - my;
        if (std::abs(px * nx + py * ny) > 1.0) {
            continue;
        }
        ts.push_back(px * dx + py * dy);
    }
    if (ts.size() < 5U) {
        return false;
    }
    // Straightness: a merged clutter component (e.g. a wavy natural edge)
    // leaves many candidates away from the fitted line.
    if (static_cast<double>(ts.size()) < 0.9 * static_cast<double>(cells.size())) {
        return false;
    }
    const double tmin = percentile(ts, 0.05);
    const double tmax = percentile(ts, 0.95);
    if ((tmax - tmin) *
            std::hypot(dx * grid.cell_size_x, dy * grid.cell_size_y) <
        min_length) {
        return false;
    }
    std::vector<double> lx;
    std::vector<double> ly;
    std::vector<double> lh;
    std::vector<double> rx;
    std::vector<double> ry;
    std::vector<double> rh;
    double lo = std::numeric_limits<double>::infinity();
    double hi = -lo;
    // Walk the segment in 1-cell tangential steps, visiting only the band on
    // each side, instead of scanning the component's bounding box (which a
    // winding component makes quadratic). Two (t, s) samples can round onto
    // the same cell, so deduplicate; the result is the same cell set up to
    // summation order.
    std::vector<int> visited;
    const int t_first = static_cast<int>(std::floor(tmin)) - 1;
    const int t_last = static_cast<int>(std::ceil(tmax)) + 1;
    for (int ti = t_first; ti <= t_last; ++ti) {
        const double bx = mx + dx * static_cast<double>(ti);
        const double by = my + dy * static_cast<double>(ti);
        for (int si = -band; si <= band; ++si) {
            const int col = static_cast<int>(std::lround(bx + nx * static_cast<double>(si)));
            const int row = static_cast<int>(std::lround(by + ny * static_cast<double>(si)));
            if (grid.inside(col, row)) {
                visited.push_back(row * grid.width + col);
            }
        }
    }
    std::sort(visited.begin(), visited.end());
    visited.erase(std::unique(visited.begin(), visited.end()), visited.end());
    for (const int index : visited) {
        const int col = index % grid.width;
        const int row = index / grid.width;
        if (!grid.is_valid(col, row)) {
            continue;
        }
        const double px = static_cast<double>(col) - mx;
        const double py = static_cast<double>(row) - my;
        const double s = px * nx + py * ny;
        const double t = px * dx + py * dy;
        if (t < tmin - 0.5 || t > tmax + 0.5 || std::abs(s) > static_cast<double>(band)) {
            continue;
        }
        const double value = grid.height_at(col, row);
        lo = std::min(lo, value);
        hi = std::max(hi, value);
        // The cell at the line plus its two neighbours still carry most of the
        // blur ramp: fitting them would pull both planes toward each other,
        // leaving soft shoulders on the rebuilt step.
        if (std::abs(s) < 3.0) {
            continue;
        }
        if (s < 0.0) {
            lx.push_back(static_cast<double>(col));
            ly.push_back(static_cast<double>(row));
            lh.push_back(value);
        } else {
            rx.push_back(static_cast<double>(col));
            ry.push_back(static_cast<double>(row));
            rh.push_back(value);
        }
    }
    const PlaneFit left = fit_plane_huber(lx, ly, lh);
    const PlaneFit right = fit_plane_huber(rx, ry, rh);
    if (!left.ok || !right.ok || !(hi > lo)) {
        return false;
    }
    // The sides of a genuine step are planar; a curved roof or rough natural
    // terrain is not, and its fitted planes would smear the snap. Compare the
    // rms with the robust step between the side medians: the full band range
    // would let roughness inflate its own tolerance.
    const double measured_step = std::abs(median_of(rh) - median_of(lh));
    const double residual_limit = std::max(0.05, 0.15 * measured_step);
    if (!(left.rms <= residual_limit) || !(right.rms <= residual_limit)) {
        return false;
    }
    int sign = 0;
    for (int k = 0; k < 3; ++k) {
        const double t = tmin + 0.5 * static_cast<double>(k) * (tmax - tmin);
        const double px = mx + dx * t;
        const double py = my + dy * t;
        const double step = plane_value(left, px, py) - plane_value(right, px, py);
        // A real discontinuity cannot be taller than the whole band's height
        // range; a larger extrapolated step means the two plane fits latched
        // onto different features. The step must also keep the same sign and
        // clear step_min_height at both ends and the middle.
        if (!(std::abs(step) >= min_height) || std::abs(step) > hi - lo) {
            return false;
        }
        const int this_sign = step > 0.0 ? 1 : -1;
        if (sign == 0) {
            sign = this_sign;
        } else if (sign != this_sign) {
            return false;
        }
    }
    out.px = mx;
    out.py = my;
    out.dx = dx;
    out.dy = dy;
    out.tmin = tmin;
    out.tmax = tmax;
    out.left = left;
    out.right = right;
    out.lo = lo;
    out.hi = hi;
    return true;
}

/// Record one accepted segment's snapped values. `best_dist` keeps the nearest
/// segment per cell, so overlaps (building corners) resolve without fighting.
void snap_segment(const Segment& segment, const Grid& grid, int band,
                  std::vector<float>& best_h, std::vector<float>& best_dist) {
    const double nx = -segment.dy;
    const double ny = segment.dx;
    const double s_max = static_cast<double>(band - 1);
    // Walk the segment in 1-cell steps and scan only the perpendicular band,
    // rather than every cell of the (possibly winding) bounding box.
    const int t_first = static_cast<int>(std::floor(segment.tmin));
    const int t_last = static_cast<int>(std::ceil(segment.tmax));
    for (int ti = t_first; ti <= t_last; ++ti) {
        const double cx = segment.px + segment.dx * static_cast<double>(ti);
        const double cy = segment.py + segment.dy * static_cast<double>(ti);
        for (int si = -band; si <= band; ++si) {
            const int col = static_cast<int>(std::lround(cx + nx * static_cast<double>(si)));
            const int row = static_cast<int>(std::lround(cy + ny * static_cast<double>(si)));
            if (!grid.is_valid(col, row)) {
                continue;
            }
            const double px = static_cast<double>(col) - segment.px;
            const double py = static_cast<double>(row) - segment.py;
            const double s = px * nx + py * ny;
            const double t = px * segment.dx + py * segment.dy;
            const double as = std::abs(s);
            if (t < segment.tmin || t > segment.tmax || as > s_max) {
                continue;
            }
            const PlaneFit& plane = s < 0.0 ? segment.left : segment.right;
            double value = plane_value(plane, static_cast<double>(col), static_cast<double>(row));
            value = std::min(std::max(value, segment.lo), segment.hi);
            // One-cell linear blend at the outer edge: no new discontinuity.
            double weight = as > s_max - 1.0 ? std::max(0.0, s_max - as) : 1.0;
            // Fade over the first and last cell along the segment, so the snap
            // cannot paint a full-height ridge beyond its measured extent.
            const double along =
                std::min(t - segment.tmin, segment.tmax - t);
            weight *= std::min(std::max(along, 0.0), 1.0);
            if (!(weight > 0.0)) {
                continue;
            }
            const double original = grid.height_at(col, row);
            const double blended = weight * value + (1.0 - weight) * original;
            const std::size_t index = static_cast<std::size_t>(row) *
                                          static_cast<std::size_t>(grid.width) +
                                      static_cast<std::size_t>(col);
            if (as < static_cast<double>(best_dist[index])) {
                best_dist[index] = static_cast<float>(as);
                best_h[index] = static_cast<float>(blended);
            }
        }
    }
}

}  // namespace

int refine_step_edges(std::vector<float>& heights, const std::vector<std::uint8_t>& valid,
                      int width, int height, double cell_size_x, double cell_size_y,
                      const StepEdgeOptions& options) {
    if (width < 3 || height < 3) {
        throw ValueError("refine_step_edges: the grid must be at least 3x3 cells");
    }
    if (!(cell_size_x > 0.0) || !std::isfinite(cell_size_x) ||
        !(cell_size_y > 0.0) || !std::isfinite(cell_size_y)) {
        throw ValueError("refine_step_edges: cell sizes must be positive and finite");
    }
    const std::size_t count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (heights.size() != count || valid.size() != count) {
        throw ValueError("refine_step_edges: heights and valid must hold width*height cells");
    }
    // Component labels are int (one label per cell); beyond this the labelled
    // pass cannot run. Such a DEM (2^31 cells) is far outside the product.
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return 0;
    }
    const double min_height = std::max(1e-6, options.min_height);
    const double min_length = std::max(0.0, options.min_length);
    const int band = std::max(2, options.band_half_width);
    const auto at = [width](int col, int row) {
        return static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
               static_cast<std::size_t>(col);
    };

    // -- 1. gradient and step evidence ---------------------------------------
    std::vector<double> h(count, 0.0);
    // A non-finite height is not evidence: it would poison every plane fit it
    // touches (NaN comparisons are all false, so no later gate would catch it).
    std::vector<std::uint8_t> usable = valid;
    for (std::size_t i = 0; i < count; ++i) {
        h[i] = static_cast<double>(heights[i]);
        if (!std::isfinite(h[i])) {
            usable[i] = 0U;
        }
    }
    // The gradient planes are float: their precision is far beyond what the
    // candidate threshold and orientation tests need, and they are released
    // before the plane stage.
    std::vector<float> gx(count, 0.0F);
    std::vector<float> gy(count, 0.0F);
    std::vector<float> gm(count, 0.0F);
    for (int row = 1; row + 1 < height; ++row) {
        for (int col = 1; col + 1 < width; ++col) {
            bool complete = true;
            for (int dy = -1; dy <= 1 && complete; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (usable[at(col + dx, row + dy)] == 0U) {
                        complete = false;
                        break;
                    }
                }
            }
            if (!complete) {
                continue;
            }
            // Sobel divided by 8: a derivative in metres per CELL.
            gx[at(col, row)] = static_cast<float>(
                (h[at(col + 1, row - 1)] + 2.0 * h[at(col + 1, row)] +
                 h[at(col + 1, row + 1)] - h[at(col - 1, row - 1)] -
                 2.0 * h[at(col - 1, row)] - h[at(col - 1, row + 1)]) /
                8.0);
            gy[at(col, row)] = static_cast<float>(
                (h[at(col - 1, row + 1)] + 2.0 * h[at(col, row + 1)] +
                 h[at(col + 1, row + 1)] - h[at(col - 1, row - 1)] -
                 2.0 * h[at(col, row - 1)] - h[at(col + 1, row - 1)]) /
                8.0);
            gm[at(col, row)] = static_cast<float>(std::hypot(
                static_cast<double>(gx[at(col, row)]),
                static_cast<double>(gy[at(col, row)])));
        }
    }

    std::vector<std::uint8_t> candidate(count, 0U);
    for (int row = 1; row + 1 < height; ++row) {
        for (int col = 1; col + 1 < width; ++col) {
            const std::size_t index = at(col, row);
            // Slope threshold min_height / (2 cells), expressed per cell. The
            // negated test also rejects NaN: both `gm < t` and `gm >= t` are
            // false for NaN, so only this form cannot let it through.
            if (!(static_cast<double>(gm[index]) >= 0.5 * min_height) ||
                !std::isfinite(h[index])) {
                continue;
            }
            const double magnitude = static_cast<double>(gm[index]);
            const double ux = static_cast<double>(gx[index]) / magnitude;
            const double uy = static_cast<double>(gy[index]) / magnitude;
            double negative_sum = 0.0;
            double positive_sum = 0.0;
            int negative_n = 0;
            int positive_n = 0;
            for (int k = 2; k <= 3; ++k) {
                for (int sign = -1; sign <= 1; sign += 2) {
                    const int sx = col + static_cast<int>(std::lround(sign * k * ux));
                    const int sy = row + static_cast<int>(std::lround(sign * k * uy));
                    if (sx < 0 || sy < 0 || sx >= width || sy >= height ||
                        usable[at(sx, sy)] == 0U) {
                        continue;
                    }
                    if (sign < 0) {
                        negative_sum += h[at(sx, sy)];
                        ++negative_n;
                    } else {
                        positive_sum += h[at(sx, sy)];
                        ++positive_n;
                    }
                }
            }
            if (negative_n == 0 || positive_n == 0) {
                continue;
            }
            const double step = positive_sum / positive_n - negative_sum / negative_n;
            if (std::abs(step) >= min_height) {
                candidate[index] = 1U;
            }
        }
    }
    // -- 2. orientation region growing (+-15 degrees) ------------------------
    std::vector<int> label(count, -1);
    std::vector<int> counts;
    std::vector<int> stack;
    for (std::size_t seed = 0; seed < count; ++seed) {
        if (candidate[seed] == 0U || label[seed] >= 0) {
            continue;
        }
        const int id = static_cast<int>(counts.size());
        counts.push_back(0);
        label[seed] = id;
        // Grow against the SEED orientation, not the neighbour's: at a blurred
        // corner the gradient rotates continuously, and a neighbour-to-
        // neighbour test would walk around the whole building as one curve.
        const double seed_magnitude = static_cast<double>(gm[seed]);
        const double seed_x = static_cast<double>(gx[seed]) / seed_magnitude;
        const double seed_y = static_cast<double>(gy[seed]) / seed_magnitude;
        stack.assign(1, static_cast<int>(seed));
        while (!stack.empty()) {
            const int current = stack.back();
            stack.pop_back();
            ++counts[static_cast<std::size_t>(id)];
            const int col = current % width;
            const int row = current / width;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    const int nc = col + dx;
                    const int nr = row + dy;
                    if (nc < 0 || nr < 0 || nc >= width || nr >= height) {
                        continue;
                    }
                    const std::size_t ni = at(nc, nr);
                    if (candidate[ni] == 0U || label[ni] >= 0) {
                        continue;
                    }
                    const double neighbour_magnitude = static_cast<double>(gm[ni]);
                    const double vx = static_cast<double>(gx[ni]) / neighbour_magnitude;
                    const double vy = static_cast<double>(gy[ni]) / neighbour_magnitude;
                    if (vx * seed_x + vy * seed_y < kCos15) {
                        continue;
                    }
                    label[ni] = id;
                    stack.push_back(static_cast<int>(ni));
                }
            }
        }
    }

    // -- 3. two-plane fit, acceptance, snap ----------------------------------
    // One CSR layout (offsets plus a single cell array) instead of a vector
    // per component. Within a component the cells are in row-major order.
    std::vector<int> offsets(counts.size() + 1, 0);
    for (std::size_t c = 0; c < counts.size(); ++c) {
        offsets[c + 1] = offsets[c] + counts[c];
    }
    std::vector<int> order(static_cast<std::size_t>(offsets.back()), 0);
    std::vector<int> cursor(offsets.begin(), offsets.end() - 1);
    for (std::size_t i = 0; i < count; ++i) {
        const int id = label[i];
        if (id >= 0) {
            order[static_cast<std::size_t>(cursor[static_cast<std::size_t>(id)]++)] =
                static_cast<int>(i);
        }
    }
    // Release the working arrays the plane stage does not read before
    // allocating its per-cell best-height arrays.
    std::vector<int>().swap(label);
    std::vector<int>().swap(cursor);
    std::vector<std::uint8_t>().swap(candidate);
    std::vector<float>().swap(gx);
    std::vector<float>().swap(gy);
    std::vector<float>().swap(gm);

    Grid grid;
    grid.width = width;
    grid.height = height;
    grid.cell_size_x = cell_size_x;
    grid.cell_size_y = cell_size_y;
    grid.h = &h;
    grid.valid = &usable;
    std::vector<float> best_h(count, 0.0F);
    std::vector<float> best_dist(count, std::numeric_limits<float>::infinity());
    int accepted = 0;
    for (std::size_t c = 0; c < counts.size(); ++c) {
        Segment segment;
        const std::span<const int> cells(order.data() + offsets[c],
                                         static_cast<std::size_t>(counts[c]));
        if (!build_segment(cells, grid, min_height, min_length, band, segment)) {
            continue;
        }
        snap_segment(segment, grid, band, best_h, best_dist);
        ++accepted;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (std::isfinite(best_dist[i])) {
            heights[i] = best_h[i];
        }
    }
    return accepted;
}

}  // namespace alphashape::elevation
