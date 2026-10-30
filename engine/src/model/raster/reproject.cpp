// Raster reprojection implementation (M12 §4.10, WP-12.9). See reproject.hpp
// for the design: inverse mapping from the target grid, a coarse lattice of
// exact transforms with bilinear interpolation and a midpoint error probe, and
// nodata-aware bilinear (float) / nearest (u8) resampling.
//
// The whole translation unit compiles without PROJ. When the engine is built
// with ALPHASHAPE_USE_PROJ=OFF the public functions keep their signatures and
// throw ValueError for anything but a same-plane resample, which is what the
// tasks in that build can ask for.

#include "model/raster/reproject.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "base/platform/memory.hpp"
#include "base/utils/error.hpp"
#include "model/crs/transform.hpp"

#if defined(ALPHASHAPE_USE_PROJ)

namespace alphashape {
namespace {

namespace crsc = alphashape::crs;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
/// Rows per band. One band's lattice is at most a few thousand nodes wide and
/// its interpolation buffers a few hundred kilobytes.
constexpr int kBandRows = 256;

/// A grid's own geometry must be usable before any index walk.
void require_grid(const RasterGrid& grid) {
    if (grid.width <= 0 || grid.height <= 0) {
        throw ValueError("reproject: the source raster has no cells");
    }
    if (!(grid.pixel_size_x > 0.0) || !(grid.pixel_size_y > 0.0)) {
        throw ValueError("reproject: the source grid has a non-positive cell size");
    }
}

/// The raster's declared geometry and its buffer must agree before any indexing
/// walk: a mismatch is a caller bug, not a bad file.
void require_consistent(const RasterGrid& grid, int channels, std::size_t size) {
    require_grid(grid);
    if (channels <= 0) {
        throw ValueError("reproject: the source raster has no channels");
    }
    const std::size_t want = static_cast<std::size_t>(grid.width) *
                             static_cast<std::size_t>(grid.height) *
                             static_cast<std::size_t>(channels);
    if (size != want) {
        throw ValueError("reproject: the source buffer does not match its grid");
    }
}

/// Refuse a CRS pair the engine cannot relate. Two local frames share the same
/// plane (a pure resample); a local frame mixed with a real CRS has no transform
/// at all, and an unset CRS names nothing.
void require_transformable_pair(const crsc::CoordinateSystem& src,
                                const crsc::CoordinateSystem& dst) {
    if (src.is_local() && dst.is_local()) {
        return;
    }
    if (src.is_local() || dst.is_local() || !src.is_set() || !dst.is_set()) {
        throw ValueError(
            "reproject: a raster can only be reprojected between two real coordinate "
            "systems, or resampled inside the local frame; one side here is the local "
            "frame or unset");
    }
}

/// The transform between the two planes, held in the SOURCE -> TARGET
/// direction: `forward_batch` maps extent and cell probes, `inverse_batch` maps
/// a target pixel back to the source. `identity` means the planes coincide, so
/// the mapping is the plain affine pixel-to-pixel one.
struct Projection {
    bool identity = false;
    crsc::CrsTransform source_to_target;
};

Projection make_projection(const crsc::CoordinateSystem& src_crs,
                           const crsc::CoordinateSystem& dst_crs) {
    Projection projection;
    if (src_crs.is_local() && dst_crs.is_local()) {
        projection.identity = true;
        return projection;
    }
    // An identical CRS needs no pipeline at all; `create()` would short-circuit
    // it to a no-op anyway, but skipping it keeps the copy path free of PROJ.
    if (src_crs == dst_crs) {
        projection.identity = true;
        return projection;
    }
    projection.source_to_target = crsc::CrsTransform::create(src_crs, dst_crs);
    return projection;
}

/// Transform plane points in place, in `forward` (source -> target) or inverse
/// (target -> source) direction; a refused point becomes NaN. A no-op when the
/// planes coincide.
void map_points(const Projection& projection, bool forward, std::vector<double>& xs,
                std::vector<double>& ys) {
    if (projection.identity) {
        return;
    }
    std::vector<crsc::CrsPoint> points(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        points[i] = crsc::CrsPoint{xs[i], ys[i], 0.0};
    }
    const crsc::CrsTransform::Batch batch = forward
                                                ? projection.source_to_target.forward_batch(points)
                                                : projection.source_to_target.inverse_batch(points);
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const bool ok = batch.is_valid(i);
        xs[i] = ok ? batch.points[i].x : kNaN;
        ys[i] = ok ? batch.points[i].y : kNaN;
    }
}

/// Source-plane points -> continuous source pixel coordinates, where the centre
/// of source pixel (c, r) is at (c + 0.5, r + 0.5), matching RasterGrid.
void to_source_pixels(const RasterGrid& src, std::vector<double>& xs,
                      std::vector<double>& ys) {
    for (std::size_t i = 0; i < xs.size(); ++i) {
        xs[i] = (xs[i] - src.origin_x) / src.pixel_size_x;
        ys[i] = (src.origin_y - ys[i]) / src.pixel_size_y;
    }
}

/// One target pixel centre's exact source pixel coordinate. Used for the
/// per-pixel fallback in cells a NaN lattice node poisons; the single-point
/// path keeps that fallback free of a per-pixel allocation.
Eigen::Vector2d exact_source_pixel(const Projection& projection, const RasterGrid& src,
                                   const RasterGrid& target, int col, int row) {
    double x = target.pixel_center(col, row).x();
    double y = target.pixel_center(col, row).y();
    if (!projection.identity) {
        const std::optional<crsc::CrsPoint> mapped = projection.source_to_target.inverse(
            crsc::CrsPoint{x, y, 0.0});
        if (!mapped.has_value()) {
            return Eigen::Vector2d(kNaN, kNaN);
        }
        x = mapped->x;
        y = mapped->y;
    }
    return Eigen::Vector2d((x - src.origin_x) / src.pixel_size_x,
                           (src.origin_y - y) / src.pixel_size_y);
}

/// A lattice of target pixels, each node carrying the EXACT source pixel
/// coordinate of that target pixel's centre. Nodes sit every `step` pixels with
/// an extra node on the last row/column when the run does not divide evenly, so
/// every target pixel lies inside a real cell (no extrapolation anywhere).
struct Lattice {
    int cols = 0, rows = 0;
    std::vector<int> tx, ty;   ///< target pixel column of each node, row of each
    std::vector<double> x, y;  ///< source pixel coordinates, row-major
};

std::size_t lattice_index(const Lattice& l, int i, int j) {
    return static_cast<std::size_t>(j) * static_cast<std::size_t>(l.cols) +
           static_cast<std::size_t>(i);
}

/// Node positions along one axis: every `step` from `first`, plus the last index
/// when it is not already a node.
std::vector<int> node_positions(int first, int last, int step) {
    std::vector<int> nodes;
    for (int v = first; v < last; v += step) {
        nodes.push_back(v);
    }
    if (nodes.empty() || nodes.back() != last) {
        nodes.push_back(last);
    }
    return nodes;
}

/// Build one band's lattice: exact transforms at the node target pixel centres,
/// converted to source pixels.
Lattice build_lattice(const Projection& projection, const RasterGrid& src,
                      const RasterGrid& target, int row0, int band_rows, int step) {
    Lattice lattice;
    lattice.tx = node_positions(0, target.width - 1, step);
    lattice.ty = node_positions(row0, row0 + band_rows - 1, step);
    lattice.cols = static_cast<int>(lattice.tx.size());
    lattice.rows = static_cast<int>(lattice.ty.size());
    const std::size_t count =
        static_cast<std::size_t>(lattice.cols) * static_cast<std::size_t>(lattice.rows);
    lattice.x.resize(count);
    lattice.y.resize(count);
    for (int j = 0; j < lattice.rows; ++j) {
        for (int i = 0; i < lattice.cols; ++i) {
            const Eigen::Vector2d centre =
                target.pixel_center(lattice.tx[static_cast<std::size_t>(i)],
                                    lattice.ty[static_cast<std::size_t>(j)]);
            const std::size_t at = lattice_index(lattice, i, j);
            lattice.x[at] = centre.x();
            lattice.y[at] = centre.y();
        }
    }
    map_points(projection, /*forward=*/false, lattice.x, lattice.y);
    to_source_pixels(src, lattice.x, lattice.y);
    return lattice;
}

/// Worst source-coordinate error (source pixels) of a spacing's bilinear
/// lattice, measured at up to `kMaxProbes` lattice-cell midpoints spread over
/// the whole target grid. A midpoint is the worst case for bilinear
/// interpolation (nodes are exact by construction).
double probe_lattice_error(const Projection& projection, const RasterGrid& src,
                           const RasterGrid& target, int step) {
    const int cells_x = target.width / step;
    const int cells_y = target.height / step;
    if (projection.identity || cells_x < 1 || cells_y < 1) {
        return 0.0;
    }
    constexpr int kMaxProbes = 400;
    const int stride = std::max(1, (cells_x * cells_y) / kMaxProbes);
    double worst = 0.0;
    for (int j = 0; j < cells_y; ++j) {
        for (int i = 0; i < cells_x; ++i) {
            if (step > 1 && ((j * cells_x) + i) % stride != 0) {
                continue;
            }
            // The exact source coordinate at the four lattice corners.
            std::vector<double> cx(4);
            std::vector<double> cy(4);
            int k = 0;
            for (int dj = 0; dj < 2; ++dj) {
                for (int di = 0; di < 2; ++di) {
                    const Eigen::Vector2d centre =
                        target.pixel_center((i + di) * step, (j + dj) * step);
                    cx[static_cast<std::size_t>(k)] = centre.x();
                    cy[static_cast<std::size_t>(k)] = centre.y();
                    ++k;
                }
            }
            map_points(projection, /*forward=*/false, cx, cy);
            to_source_pixels(src, cx, cy);
            // The exact source coordinate at the cell midpoint.
            std::vector<double> xs(1);
            std::vector<double> ys(1);
            const Eigen::Vector2d mid =
                target.pixel_center(i * step + step / 2, j * step + step / 2);
            xs[0] = mid.x();
            ys[0] = mid.y();
            map_points(projection, /*forward=*/false, xs, ys);
            to_source_pixels(src, xs, ys);
            if (!std::isfinite(xs[0]) || !std::isfinite(ys[0])) {
                continue;  // outside the projection's domain: nothing to measure
            }
            // Bilinear at (0.5, 0.5) is the mean of the four corners.
            const double ix = 0.25 * (cx[0] + cx[1] + cx[2] + cx[3]);
            const double iy = 0.25 * (cy[0] + cy[1] + cy[2] + cy[3]);
            worst = std::max(worst, std::hypot(ix - xs[0], iy - ys[0]));
        }
    }
    return worst;
}

/// The spacing the mapping uses: the caller's pin, else the coarsest spacing
/// from `kLatticeStep` down whose error fits `kMaxLatticeError`, else 1 (the
/// exact per-pixel mapping).
int choose_step(const Projection& projection, const RasterGrid& src,
                const RasterGrid& target, int requested) {
    const int limit = std::max(1, std::max(target.width, target.height));
    if (requested > 0) {
        return std::min(requested, limit);
    }
    if (projection.identity) {
        return kLatticeStep;
    }
    for (int step = kLatticeStep; step >= 2; step /= 2) {
        if (probe_lattice_error(projection, src, target, step) <= kMaxLatticeError) {
            return step;
        }
    }
    return 1;
}

/// Nearest source pixel index for a continuous pixel coordinate (pixel centres
/// are at half-integers, so flooring picks the containing pixel).
int nearest_index(double continuous) { return static_cast<int>(std::floor(continuous)); }

/// Bilinear, nodata-aware float sample at continuous source pixel coordinate
/// (px, py). Out-of-bounds, `nodata_in` and non-finite taps carry no weight; the
/// rest renormalise. Returns false when the valid weight is below
/// `kMinValidWeight`, and the caller writes `nodata_out`.
bool sample_bilinear(const GeoRaster<float>& src, double px, double py, float nodata_in,
                     double* out_sum, double* out_weight) {
    const double cx = px - 0.5;  // bilinear lattice corner in index space
    const double cy = py - 0.5;
    const int c0 = static_cast<int>(std::floor(cx));
    const int r0 = static_cast<int>(std::floor(cy));
    const double fx = cx - static_cast<double>(c0);
    const double fy = cy - static_cast<double>(r0);
    double sum = 0.0;
    double weight = 0.0;
    for (int dr = 0; dr <= 1; ++dr) {
        for (int dc = 0; dc <= 1; ++dc) {
            const double w = (dc == 0 ? 1.0 - fx : fx) * (dr == 0 ? 1.0 - fy : fy);
            if (w <= 0.0) {
                continue;
            }
            const int col = c0 + dc;
            const int row = r0 + dr;
            if (!src.grid.contains_pixel(col, row)) {
                continue;  // an edge tap is missing mass, not a zero measurement
            }
            const double v = static_cast<double>(src.at(col, row));
            if (!std::isfinite(v) || v == static_cast<double>(nodata_in)) {
                continue;
            }
            sum += w * v;
            weight += w;
        }
    }
    *out_sum = sum;
    *out_weight = weight;
    return weight >= kMinValidWeight;
}

/// The lattice cell containing target pixel column `col` (resp. row `row`) and
/// the fraction into it. Binary search because the node list carries an extra
/// tail node; a scan would be O(nodes) per pixel.
int cell_of(const std::vector<int>& nodes, int v, double* fraction) {
    // nodes is ascending; find the last one <= v.
    const auto it = std::upper_bound(nodes.begin(), nodes.end(), v);
    const int hi = static_cast<int>(it - nodes.begin());
    const int a = std::clamp(hi - 1, 0, static_cast<int>(nodes.size()) - 1);
    const int b = std::clamp(hi, 0, static_cast<int>(nodes.size()) - 1);
    const double span = static_cast<double>(nodes[static_cast<std::size_t>(b)] -
                                            nodes[static_cast<std::size_t>(a)]);
    *fraction = span > 0.0
                    ? (static_cast<double>(v) - static_cast<double>(nodes[static_cast<std::size_t>(a)])) /
                          span
                    : 0.0;
    return a;
}

/// One band's precomputed axis cells: for the target columns (rows) in range,
/// the lattice node below them and the fraction into the cell. Precomputing
/// keeps the per-pixel inner loop free of a search.
struct AxisCells {
    std::vector<int> node;
    std::vector<double> fraction;

    static AxisCells for_range(const std::vector<int>& nodes, int first, int count) {
        AxisCells cells;
        cells.node.resize(static_cast<std::size_t>(count));
        cells.fraction.resize(static_cast<std::size_t>(count));
        for (int k = 0; k < count; ++k) {
            double f = 0.0;
            const int at = cell_of(nodes, first + k, &f);
            cells.node[static_cast<std::size_t>(k)] = at;
            cells.fraction[static_cast<std::size_t>(k)] = f;
        }
        return cells;
    }
};

/// True when any of the four lattice nodes around cell (i, j) carries a NaN
/// coordinate: one refused transform poisons the whole cell, so the caller must
/// take the exact per-pixel path there rather than interpolate across it.
bool cell_has_nan(const Lattice& l, int i, int j) {
    const int i1 = std::min(i + 1, l.cols - 1);
    const int j1 = std::min(j + 1, l.rows - 1);
    const std::size_t n = lattice_index(l, i, j);
    const std::size_t a = lattice_index(l, i1, j);
    const std::size_t b = lattice_index(l, i, j1);
    const std::size_t c = lattice_index(l, i1, j1);
    return !std::isfinite(l.x[n]) || !std::isfinite(l.y[n]) ||
           !std::isfinite(l.x[a]) || !std::isfinite(l.y[a]) ||
           !std::isfinite(l.x[b]) || !std::isfinite(l.y[b]) ||
           !std::isfinite(l.x[c]) || !std::isfinite(l.y[c]);
}

/// Bilinear source pixel coordinate at target pixel (col, row) from the band's
/// lattice: the exact nodes around it, interpolated.
Eigen::Vector2d lattice_lookup(const Lattice& l, int i, double fx, int j, double fy) {
    const Eigen::Vector2d a(l.x[lattice_index(l, i, j)], l.y[lattice_index(l, i, j)]);
    if (fx == 0.0 && fy == 0.0) {
        return a;
    }
    const int i1 = std::min(i + 1, l.cols - 1);
    const int j1 = std::min(j + 1, l.rows - 1);
    const Eigen::Vector2d b(l.x[lattice_index(l, i1, j)], l.y[lattice_index(l, i1, j)]);
    const Eigen::Vector2d c(l.x[lattice_index(l, i, j1)], l.y[lattice_index(l, i, j1)]);
    const Eigen::Vector2d d(l.x[lattice_index(l, i1, j1)], l.y[lattice_index(l, i1, j1)]);
    return (1.0 - fx) * (1.0 - fy) * a + fx * (1.0 - fy) * b + (1.0 - fx) * fy * c +
           fx * fy * d;
}

/// Fill the target rows [row0, row0 + band_rows) of `out` from `src`: every
/// target pixel centre's source coordinate comes from the lattice, bilinearly
/// interpolated for pixels between nodes. A NaN source coordinate (the point
/// failed the transform) or a sample the source does not cover leaves the cell
/// at `missing`.
template <typename T>
void fill_band(const GeoRaster<T>& src, const Projection& projection, const Lattice& lattice,
               bool bilinear, float nodata_in, T missing, int row0, int band_rows,
               GeoRaster<T>& out) {
    const int channels = out.channels;
    const AxisCells cols = AxisCells::for_range(lattice.tx, 0, out.grid.width);
    const AxisCells rows = AxisCells::for_range(lattice.ty, row0, band_rows);
    for (int k_row = 0; k_row < band_rows; ++k_row) {
        const int row = row0 + k_row;
        const int j = rows.node[static_cast<std::size_t>(k_row)];
        const double fy = rows.fraction[static_cast<std::size_t>(k_row)];
        for (int col = 0; col < out.grid.width; ++col) {
            const int i = cols.node[static_cast<std::size_t>(col)];
            const double fx = cols.fraction[static_cast<std::size_t>(col)];
            Eigen::Vector2d p = lattice_lookup(lattice, i, fx, j, fy);
            if (cell_has_nan(lattice, i, j)) {
                // A refused lattice node on this cell's corner poisons the whole
                // cell for interpolation (no finite weight to renormalise over),
                // so take the EXACT transform at this pixel instead. That keeps
                // the domain's genuine interior while only the true edge stays
                // nodata.
                p = exact_source_pixel(projection, src.grid, out.grid, col, row);
            }
            std::size_t base = (static_cast<std::size_t>(row) *
                                static_cast<std::size_t>(out.grid.width) +
                                static_cast<std::size_t>(col)) *
                               static_cast<std::size_t>(channels);
            if (!std::isfinite(p.x()) || !std::isfinite(p.y())) {
                for (int ch = 0; ch < channels; ++ch) {
                    out.data[base + static_cast<std::size_t>(ch)] = missing;
                }
                continue;
            }
            if constexpr (std::is_floating_point_v<T>) {
                if (bilinear) {
                    double sum = 0.0;
                    double weight = 0.0;
                    const bool ok =
                        sample_bilinear(src, p.x(), p.y(), nodata_in, &sum, &weight);
                    const T value = ok ? static_cast<T>(sum / weight) : missing;
                    for (int ch = 0; ch < channels; ++ch) {
                        out.data[base + static_cast<std::size_t>(ch)] = value;
                    }
                    continue;
                }
            }
            const int scol = nearest_index(p.x());
            const int srow = nearest_index(p.y());
            if (!src.grid.contains_pixel(scol, srow)) {
                for (int ch = 0; ch < channels; ++ch) {
                    out.data[base + static_cast<std::size_t>(ch)] = missing;
                }
                continue;
            }
            for (int ch = 0; ch < channels; ++ch) {
                out.data[base + static_cast<std::size_t>(ch)] = src.at(scol, srow, ch);
            }
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// suggest_grid
// ---------------------------------------------------------------------------

namespace {

/// Points per edge when densifying the source boundary. A projection's edge
/// bows, and four corners alone would clip the raster at mid-edge; 21 samples
/// put the worst mid-segment sag well under a cell for the projections the
/// engine targets (a UTM/Web-Mercator pair sags a few tenths of a cell over a
/// kilometre at 60 degrees of longitude).
constexpr int kBoundaryPointsPerEdge = 21;

}  // namespace

RasterGrid suggest_grid(const RasterGrid& src, const crsc::CoordinateSystem& src_crs,
                        const crsc::CoordinateSystem& dst_crs, double resolution) {
    require_grid(src);
    require_transformable_pair(src_crs, dst_crs);
    const Projection projection = make_projection(src_crs, dst_crs);

    // -- the transformed, densified boundary in the target plane -------------
    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(4U * static_cast<std::size_t>(kBoundaryPointsPerEdge));
    ys.reserve(xs.capacity());
    const double left = src.origin_x;
    const double top = src.origin_y;
    const double right = src.origin_x + static_cast<double>(src.width) * src.pixel_size_x;
    const double bottom = src.origin_y - static_cast<double>(src.height) * src.pixel_size_y;
    const int n = kBoundaryPointsPerEdge - 1;
    for (int k = 0; k <= n; ++k) {
        const double f = static_cast<double>(k) / static_cast<double>(n);
        xs.push_back(left + f * (right - left));
        ys.push_back(top);  // north edge
    }
    for (int k = 1; k <= n; ++k) {
        const double f = static_cast<double>(k) / static_cast<double>(n);
        xs.push_back(right);
        ys.push_back(top + f * (bottom - top));  // east edge
    }
    for (int k = 1; k <= n; ++k) {
        const double f = static_cast<double>(k) / static_cast<double>(n);
        xs.push_back(right + f * (left - right));
        ys.push_back(bottom);  // south edge
    }
    for (int k = 1; k < n; ++k) {
        const double f = static_cast<double>(k) / static_cast<double>(n);
        xs.push_back(left);
        ys.push_back(bottom + f * (top - bottom));  // west edge
    }
    map_points(projection, /*forward=*/true, xs, ys);
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -min_x;
    double min_y = min_x;
    double max_y = -min_x;
    std::size_t covered = 0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        if (!std::isfinite(xs[i]) || !std::isfinite(ys[i])) {
            continue;
        }
        ++covered;
        min_x = std::min(min_x, xs[i]);
        max_x = std::max(max_x, xs[i]);
        min_y = std::min(min_y, ys[i]);
        max_y = std::max(max_y, ys[i]);
    }
    if (covered < 3) {
        throw ValueError(
            "suggest_grid: the transform covers too little of the source raster to "
            "define a target grid");
    }

    // -- the cell: the caller's, or the source GSD at the raster centre ------
    double cell = resolution;
    if (!(cell > 0.0)) {
        // The centre in the target plane, and the point one source pixel east of
        // it, both transformed: the distance between them is one source pixel's
        // ground length in target units at the centre (degrees for a geographic
        // target). This is the GSD at the centre, by construction.
        const Eigen::Vector2d centre = src.pixel_center(src.width / 2, src.height / 2);
        std::vector<double> px{centre.x(), centre.x() + src.pixel_size_x};
        std::vector<double> py{centre.y(), centre.y()};
        map_points(projection, /*forward=*/true, px, py);
        const double dx = px[1] - px[0];
        const double dy = py[1] - py[0];
        cell = std::hypot(dx, dy);
        if (!(cell > 0.0) || !std::isfinite(cell)) {
            // A degenerate x step (the transform refused the offset, or the
            // source is one pixel wide): fall back to the y direction.
            std::vector<double> qx{centre.x(), centre.x()};
            std::vector<double> qy{centre.y(), centre.y() - src.pixel_size_y};
            map_points(projection, /*forward=*/true, qx, qy);
            cell = std::hypot(qx[1] - qx[0], qy[1] - qy[0]);
        }
        if (!(cell > 0.0) || !std::isfinite(cell)) {
            throw ValueError("suggest_grid: cannot measure the source cell in the target "
                             "CRS; pass resolution explicitly");
        }
    }

    // -- snap the origin outward to a whole cell ----------------------------
    // Row 0 is northernmost: the origin is the north-west OUTER corner, so it
    // is rounded DOWN on x and UP on y, and the grid is large enough to cover
    // max_x/max_y from there. A half-cell guard absorbs the rounding of
    // `ceil` on a span that is an exact multiple of the cell.
    const double origin_x = std::floor(min_x / cell) * cell;
    const double origin_y = std::ceil(max_y / cell) * cell;
    const double cols = std::ceil((max_x - origin_x) / cell - 1e-9);
    const double rows = std::ceil((origin_y - min_y) / cell - 1e-9);
    const double cells = cols * rows;
    if (!(cols >= 1.0) || !(rows >= 1.0) || !std::isfinite(cells) ||
        cells > 1.0e9) {
        throw ValueError("suggest_grid: the target grid came out degenerate (" +
                         std::to_string(cols) + " x " + std::to_string(rows) +
                         " cells); the source extent or CRS pair is not usable");
    }
    RasterGrid grid;
    grid.origin_x = origin_x;
    grid.origin_y = origin_y;
    grid.pixel_size_x = cell;
    grid.pixel_size_y = cell;
    grid.width = static_cast<int>(cols);
    grid.height = static_cast<int>(rows);
    return grid;
}

RasterGrid resample_grid(const RasterGrid& src, double cell) {
    require_grid(src);
    if (!(cell > 0.0) || !std::isfinite(cell)) {
        return src;  // "keep the source cell" is the source grid itself
    }
    const bool same_x = std::abs(cell / src.pixel_size_x - 1.0) <= 1e-9;
    const bool same_y = std::abs(cell / src.pixel_size_y - 1.0) <= 1e-9;
    if (same_x && same_y) {
        return src;
    }
    // The source's own outer corner is the anchor: no snapping, so a real
    // georeference is preserved and the resample cannot introduce a blank
    // border. One extra row/column covers an extent that does not divide.
    const double cols = std::ceil(static_cast<double>(src.width) * src.pixel_size_x / cell - 1e-9);
    const double rows =
        std::ceil(static_cast<double>(src.height) * src.pixel_size_y / cell - 1e-9);
    if (!(cols >= 1.0) || !(rows >= 1.0) || !std::isfinite(cols) || !std::isfinite(rows) ||
        cols * rows > 1.0e9) {
        throw ValueError("resample_grid: the target grid came out degenerate (" +
                         std::to_string(cols) + " x " + std::to_string(rows) +
                         " cells); choose a coarser resolution");
    }
    RasterGrid grid;
    grid.origin_x = src.origin_x;
    grid.origin_y = src.origin_y;
    grid.pixel_size_x = cell;
    grid.pixel_size_y = cell;
    grid.width = static_cast<int>(cols);
    grid.height = static_cast<int>(rows);
    return grid;
}

// ---------------------------------------------------------------------------
// reproject (float)
// ---------------------------------------------------------------------------

namespace {

/// Refuse an output that cannot fit before allocating it. The raster is the
/// product the caller keeps, so it is allocated whole; a DEM far beyond the
/// machine's budget should fail with an explanation, not a bare bad_alloc.
void require_output_fits(const RasterGrid& target, int channels, std::size_t bytes_per_sample,
                         const char* what) {
    const std::uint64_t cells = static_cast<std::uint64_t>(target.width) *
                                static_cast<std::uint64_t>(target.height);
    const std::uint64_t bytes = cells * static_cast<std::uint64_t>(channels) *
                                static_cast<std::uint64_t>(bytes_per_sample);
    const std::uint64_t budget = platform::memory_budget_bytes();
    if (budget != 0 && bytes > budget) {
        throw ValueError(std::string("reproject: the ") + what + " target is " +
                         std::to_string(target.width) + "x" +
                         std::to_string(target.height) + " cells = " +
                         platform::format_gb(static_cast<double>(bytes)) +
                         ", which does not fit this machine's budget of " +
                         platform::format_gb(static_cast<double>(budget)) +
                         "; choose a coarser resolution");
    }
}

}  // namespace

namespace {

template <typename T>
GeoRaster<T> reproject_raster(const GeoRaster<T>& src, const crsc::CoordinateSystem& src_crs,
                              const crsc::CoordinateSystem& dst_crs,
                              const RasterGrid& target, const ReprojectOptions& options,
                              bool bilinear, T missing) {
    require_consistent(src.grid, src.channels, src.data.size());
    require_transformable_pair(src_crs, dst_crs);
    if (target.width <= 0 || target.height <= 0) {
        throw ValueError("reproject: the target raster has no cells");
    }
    if (!(target.pixel_size_x > 0.0) || !(target.pixel_size_y > 0.0)) {
        throw ValueError("reproject: the target grid has a non-positive cell size");
    }
    require_output_fits(target, src.channels, sizeof(T),
                        sizeof(T) == 1 ? "orthomosaic" : "elevation");

    GeoRaster<T> out;
    out.grid = target;
    out.channels = src.channels;
    out.data.assign(static_cast<std::size_t>(target.width) *
                        static_cast<std::size_t>(target.height) *
                        static_cast<std::size_t>(src.channels),
                    missing);

    // An identical plane and grid is a copy: no transform, no resample, and the
    // bits are the source's own (the identity contract).
    const Projection projection = make_projection(src_crs, dst_crs);
    if (projection.identity && src.grid == target) {
        out.data = src.data;
        return out;
    }

    const int step = choose_step(projection, src.grid, target, options.lattice_step);
    for (int row0 = 0; row0 < target.height; row0 += kBandRows) {
        const int band_rows = std::min(kBandRows, target.height - row0);
        const Lattice lattice =
            build_lattice(projection, src.grid, target, row0, band_rows, step);
        fill_band(src, projection, lattice, bilinear, options.nodata_in, missing, row0,
                  band_rows, out);
    }
    return out;
}

}  // namespace

GeoRaster<float> reproject(const GeoRaster<float>& src, const crsc::CoordinateSystem& src_crs,
                           const crsc::CoordinateSystem& dst_crs, const RasterGrid& target,
                           const ReprojectOptions& options) {
    const bool bilinear = options.resampling != Resampling::Nearest;
    return reproject_raster(src, src_crs, dst_crs, target, options, bilinear,
                            options.nodata_out);
}

GeoRaster<std::uint8_t> reproject(const GeoRaster<std::uint8_t>& src,
                                  const crsc::CoordinateSystem& src_crs,
                                  const crsc::CoordinateSystem& dst_crs,
                                  const RasterGrid& target,
                                  const ReprojectOptions& options) {
    // 8-bit is always nearest: a blended colour was never photographed, and the
    // orthomosaic's own seam blend is the only place colours are mixed on
    // purpose. Uncovered cells are zeros, which is alpha 0 for RGBA.
    return reproject_raster(src, src_crs, dst_crs, target, options, /*bilinear=*/false,
                            static_cast<std::uint8_t>(0));
}

}  // namespace alphashape

#else  // !ALPHASHAPE_USE_PROJ

namespace alphashape {

// Without the CRS engine there is no CoordinateSystem to call (the library does
// not exist in this build), so every operation here needs PROJ. The tasks never
// call into this file in that configuration: an export passes the raster
// through unchanged when no CRS is requested, and requesting one throws here.
namespace {

[[noreturn]] void require_proj() {
    throw ValueError(
        "reproject: raster reprojection needs a build with PROJ (ALPHASHAPE_USE_PROJ=ON)");
}

}  // namespace

RasterGrid suggest_grid(const RasterGrid&, const crs::CoordinateSystem&,
                        const crs::CoordinateSystem&, double) {
    require_proj();
}

RasterGrid resample_grid(const RasterGrid& src, double cell) {
    // A re-cell needs no CRS engine: it is the source plane's own geometry.
    if (src.width <= 0 || src.height <= 0) {
        throw ValueError("resample_grid: the source raster has no cells");
    }
    if (!(cell > 0.0) || !std::isfinite(cell) ||
        (std::abs(cell / src.pixel_size_x - 1.0) <= 1e-9 &&
         std::abs(cell / src.pixel_size_y - 1.0) <= 1e-9)) {
        return src;
    }
    RasterGrid grid;
    grid.origin_x = src.origin_x;
    grid.origin_y = src.origin_y;
    grid.pixel_size_x = cell;
    grid.pixel_size_y = cell;
    grid.width = static_cast<int>(
        std::ceil(static_cast<double>(src.width) * src.pixel_size_x / cell - 1e-9));
    grid.height = static_cast<int>(
        std::ceil(static_cast<double>(src.height) * src.pixel_size_y / cell - 1e-9));
    return grid;
}

GeoRaster<float> reproject(const GeoRaster<float>&, const crs::CoordinateSystem&,
                           const crs::CoordinateSystem&, const RasterGrid&,
                           const ReprojectOptions&) {
    require_proj();
}

GeoRaster<std::uint8_t> reproject(const GeoRaster<std::uint8_t>&,
                                  const crs::CoordinateSystem&,
                                  const crs::CoordinateSystem&, const RasterGrid&,
                                  const ReprojectOptions&) {
    require_proj();
}

}  // namespace alphashape

#endif  // ALPHASHAPE_USE_PROJ
