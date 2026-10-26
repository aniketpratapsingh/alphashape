// True orthorectification: see orthorectify.hpp for the contract.

#include "common/algorithm/ortho/orthorectify.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace alphashape::ortho {
namespace {

/// Bilinear DEM height at the plane-frame point (x, y), nodata-aware: nodata
/// neighbours are dropped and the remaining weights renormalised. Returns
/// `nodata` when nothing valid contributes, or the point is off the grid (the
/// outer half cell is treated as nodata so we do not extrapolate).
double sample_dem(const GeoRaster<float>& dem, float nodata, double x, double y) {
    const RasterGrid& grid = dem.grid;
    const Eigen::Vector2d c = grid.to_pixel(x, y);
    const double cf = c.x() - 0.5;  // integer at a cell centre
    const double rf = c.y() - 0.5;
    const int c0 = static_cast<int>(std::floor(cf));
    const int r0 = static_cast<int>(std::floor(rf));
    const double fx = cf - static_cast<double>(c0);
    const double fy = rf - static_cast<double>(r0);

    double sum = 0.0;
    double weight = 0.0;
    for (int dr = 0; dr <= 1; ++dr) {
        for (int dc = 0; dc <= 1; ++dc) {
            const int col = c0 + dc;
            const int row = r0 + dr;
            if (!grid.contains_pixel(col, row)) {
                continue;  // do not extrapolate past the edge
            }
            const float h = dem.at(col, row);
            if (h == nodata || std::isnan(h)) {
                continue;
            }
            const double w = (dc == 0 ? 1.0 - fx : fx) * (dr == 0 ? 1.0 - fy : fy);
            sum += w * static_cast<double>(h);
            weight += w;
        }
    }
    if (weight <= 0.0) {
        return static_cast<double>(nodata);
    }
    return sum / weight;
}

/// The tallest valid DEM sample, the height above which the marching ray can no
/// longer be occluded.
double dem_max(const GeoRaster<float>& dem, float nodata) {
    double best = -std::numeric_limits<double>::infinity();
    for (const float h : dem.data) {
        if (h != nodata && !std::isnan(h)) {
            best = std::max(best, static_cast<double>(h));
        }
    }
    return best;
}

/// Bilinear colour sample of an 8-bit 1- or 3-channel image at continuous pixel
/// coordinates (0.5, 0.5) = centre of the top-left pixel. Edges clamp.
void sample_rgb(const Image& image, double u, double v, std::uint8_t out[3]) {
    const int w = image.width();
    const int h = image.height();
    const int ch = image.channels();
    const double xc = u - 0.5;
    const double yc = v - 0.5;
    const int x0 = static_cast<int>(std::floor(xc));
    const int y0 = static_cast<int>(std::floor(yc));
    const double fx = xc - static_cast<double>(x0);
    const double fy = yc - static_cast<double>(y0);
    for (int c = 0; c < 3; ++c) {
        const int channel = (ch >= 3) ? c : 0;  // grey replicates to RGB
        double acc = 0.0;
        for (int dy = 0; dy <= 1; ++dy) {
            for (int dx = 0; dx <= 1; ++dx) {
                const int sx = std::clamp(x0 + dx, 0, w - 1);
                const int sy = std::clamp(y0 + dy, 0, h - 1);
                const double wgt = (dx == 0 ? 1.0 - fx : fx) * (dy == 0 ? 1.0 - fy : fy);
                acc += wgt * static_cast<double>(image.at(sx, sy, channel));
            }
        }
        out[c] = static_cast<std::uint8_t>(std::lround(std::clamp(acc, 0.0, 255.0)));
    }
}

/// Fill `params` from a Calibration in the kCalibrationParamCount order.
void calibration_params(const Calibration& calibration, double params[kCalibrationParamCount]) {
    params[0] = calibration.f;
    params[1] = calibration.cx;
    params[2] = calibration.cy;
    params[3] = calibration.b1;
    params[4] = calibration.b2;
    params[5] = calibration.k1;
    params[6] = calibration.k2;
    params[7] = calibration.k3;
    params[8] = calibration.k4;
    params[9] = calibration.p1;
    params[10] = calibration.p2;
}

/// True when the surface at `z` blocks the segment P->Q by more than `tol`.
bool blocked(const GeoRaster<float>& dem, float nodata, double tol, double x, double y,
             double z) {
    const double surface = sample_dem(dem, nodata, x, y);
    return surface != static_cast<double>(nodata) && surface > z + tol;
}

/// A safety stop against a pathological (or NaN-driven) loop, NOT a geometric
/// bound. The march's real bound is metric: it stops as soon as the ray climbs
/// above `dem_max + tol`, at which height occlusion is impossible. How far that
/// takes is a property of the scene, not of the output grid, which is what the
/// per-output-cell cost contract relies on (see the comment at the march).
///
/// The previous fixed cap of 4096 samples was an implicit geometry assumption:
/// 4096 half-DEM-cell steps do not always reach the crest of a long slope viewed
/// at a grazing angle, so a genuine occluder could be dropped. The cap is now
/// only what turns "this ray never got above the terrain" into a finite loop.
constexpr std::size_t kMaxOcclusionSamplesPerRay = 1000000;

/// DEM samples taken by the occlusion march since the last reset. Test-only:
/// the cost contract (samples per output cell must not follow the output grid)
/// is otherwise unobservable from a tile's contents.
std::atomic<std::uint64_t> g_occlusion_samples{0};

}  // namespace

std::uint64_t occlusion_sample_count() {
    return g_occlusion_samples.load(std::memory_order_relaxed);
}

void reset_occlusion_sample_count() {
    g_occlusion_samples.store(0, std::memory_order_relaxed);
}

TileBounds camera_footprint_bounds(
    const RasterGrid& out_grid, const OrthoCamera& camera, double lowest_height,
    double highest_height, int margin_cells) {
    TileBounds full;
    full.x0 = 0;
    full.y0 = 0;
    full.width = out_grid.width;
    full.height = out_grid.height;
    // The sensor geometry comes from the decoded photograph when there is one;
    // the calibration carries the same dimensions, so a caller estimating the
    // footprint before decoding (the memory guard) can pass a camera with no
    // image and still get a bound.
    const int image_width = camera.image != nullptr && !camera.image->empty()
                                ? camera.image->width()
                                : camera.calibration.width;
    const int image_height = camera.image != nullptr && !camera.image->empty()
                                 ? camera.image->height()
                                 : camera.calibration.height;
    if (image_width <= 0 || image_height <= 0 || out_grid.width <= 0 ||
        out_grid.height <= 0 || !(highest_height >= lowest_height)) {
        return full;
    }
    // The bounds are computed in the frame the pose acts in. When that is the
    // chunk-local frame (plane_to_local is set) the result has to be carried
    // back onto the plane grid; without the inverse there is no safe box.
    if (camera.plane_to_local && !camera.local_to_plane) {
        return full;
    }
    // The cone-bound argument below needs both heights finite; a DEM with a NaN
    // range is not something to crop against.
    if (!std::isfinite(lowest_height) || !std::isfinite(highest_height)) {
        return full;
    }

    const Eigen::Vector3d centre = camera.centre();
    // The farthest any on-grid ground point can be from the optical centre: the
    // largest corner distance from the camera's own position on the grid, plus
    // the vertical relief. Clipping each border ray there keeps a near-horizontal
    // border ray (a wide-angle or fisheye edge) from producing an unbounded
    // intersection without losing anything that could be on the grid.
    const Eigen::Vector3d centre_plane =
        camera.local_to_plane ? camera.local_to_plane(centre) : centre;
    const double east = out_grid.origin_x +
                        static_cast<double>(out_grid.width) * out_grid.pixel_size_x;
    const double south =
        out_grid.origin_y - static_cast<double>(out_grid.height) * out_grid.pixel_size_y;
    double reach = std::abs(centre.z() - lowest_height) + 1.0;
    for (const double x : {out_grid.origin_x, east}) {
        for (const double y : {out_grid.origin_y, south}) {
            reach = std::max(reach, std::hypot(x - centre_plane.x(), y - centre_plane.y()));
        }
    }
    reach += std::abs(centre.z() - highest_height) + 1.0;

    const TileBounds none;  // zero size: nothing accumulated yet
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = min_x;
    double max_y = max_x;
    bool any = false;

    // The image border, 5 samples per edge: lens distortion bows the border away
    // from the straight pinhole edge, and the intermediate samples bracket it.
    constexpr int kSamplesPerEdge = 5;  // 0, 1/4, 1/2, 3/4, 1 of each edge
    const double iw = static_cast<double>(image_width);
    const double ih = static_cast<double>(image_height);
    auto accumulate = [&](const Eigen::Vector3d& local) {
        const Eigen::Vector3d plane =
            camera.local_to_plane ? camera.local_to_plane(local) : local;
        if (!std::isfinite(plane.x()) || !std::isfinite(plane.y())) {
            return;
        }
        min_x = std::min(min_x, plane.x());
        max_x = std::max(max_x, plane.x());
        min_y = std::min(min_y, plane.y());
        max_y = std::max(max_y, plane.y());
        any = true;
    };
    for (int edge = 0; edge < 4; ++edge) {
        for (int k = 0; k < kSamplesPerEdge; ++k) {
            const double f = static_cast<double>(k) / static_cast<double>(kSamplesPerEdge - 1);
            double u = 0.0;
            double v = 0.0;
            switch (edge) {
                case 0:  // top
                    u = f * iw;
                    v = 0.0;
                    break;
                case 1:  // right
                    u = iw;
                    v = f * ih;
                    break;
                case 2:  // bottom
                    u = f * iw;
                    v = ih;
                    break;
                default:  // left
                    u = 0.0;
                    v = f * ih;
                    break;
            }
            // unproject() never throws: a degenerate calibration returns a
            // straight-ahead ray, and a pixel that cannot be inverted would only
            // make the box smaller, so treat a non-finite ray as "no bound".
            const Eigen::Vector3d d_cam = camera.calibration.unproject(Eigen::Vector2d(u, v));
            if (!d_cam.allFinite()) {
                return full;
            }
            const Eigen::Vector3d direction = camera.R.transpose() * d_cam;
            if (!(std::abs(direction.z()) > 1e-6)) {
                continue;  // parallel to the ground: no bounded intersection
            }
            // Where the ray crosses the lowest and the highest ground height.
            const double t_lo = (lowest_height - centre.z()) / direction.z();
            const double t_hi = (highest_height - centre.z()) / direction.z();
            double s0 = std::min(t_lo, t_hi);
            double s1 = std::max(t_lo, t_hi);
            s0 = std::max(s0, 0.0);
            s1 = std::min(s1, reach);
            if (!(s1 >= s0)) {
                continue;  // the slab is not on the visible side of the camera
            }
            accumulate(centre + s0 * direction);
            accumulate(centre + s1 * direction);
        }
    }
    if (!any) {
        return full;  // nothing boundable: keep the conservative full grid
    }

    const int margin = std::max(0, margin_cells);
    // Distortion bows the true image border away from the polyline through the
    // sampled border points. Inflating the box by 1% of its own extent absorbs
    // any realistic bow (a ~5% distortion moves the border by a few percent of
    // the footprint at most), on top of the fixed cell slack below.
    const double span_x = max_x - min_x;
    const double span_y = max_y - min_y;
    min_x -= 0.01 * span_x;
    max_x += 0.01 * span_x;
    min_y -= 0.01 * span_y;
    max_y += 0.01 * span_y;

    const double px = out_grid.pixel_size_x;
    const double py = out_grid.pixel_size_y;
    // Cell centre i sits at origin + (i + 0.5) * pixel. Floor/ceil with a one
    // cell slack on each side turns the point box into a cell box.
    const int col_lo = static_cast<int>(std::floor((min_x - out_grid.origin_x) / px)) - 1 -
                       margin;
    const int col_hi = static_cast<int>(std::ceil((max_x - out_grid.origin_x) / px)) + 1 + margin;
    const int row_lo = static_cast<int>(std::floor((out_grid.origin_y - max_y) / py)) - 1 - margin;
    const int row_hi = static_cast<int>(std::ceil((out_grid.origin_y - min_y) / py)) + 1 + margin;

    TileBounds bounds;
    bounds.x0 = std::max(0, col_lo);
    bounds.y0 = std::max(0, row_lo);
    const int x1 = std::min(out_grid.width, col_hi + 1);
    const int y1 = std::min(out_grid.height, row_hi + 1);
    bounds.width = std::max(0, x1 - bounds.x0);
    bounds.height = std::max(0, y1 - bounds.y0);
    if (bounds.width <= 0 || bounds.height <= 0) {
        return none;  // the footprint misses the grid entirely
    }
    return bounds;
}

OrthoTile orthorectify(const GeoRaster<float>& dem, float dem_nodata,
                       const RasterGrid& out_grid, const OrthoCamera& camera,
                       int camera_index, double occlusion_tolerance, bool crop) {
    OrthoTile tile;
    tile.camera = camera_index;
    if (camera.image == nullptr || camera.image->empty() || dem.empty()) {
        return tile;  // empty: no buffers, covers nothing
    }

    // The tile buffer. Cropped, this is the camera's footprint only; otherwise it
    // is the full grid, which is exactly the same data with the uncovered border
    // made explicit.
    if (crop) {
        double lowest = std::numeric_limits<double>::infinity();
        double highest = -std::numeric_limits<double>::infinity();
        for (const float h : dem.data) {
            if (h == dem_nodata || std::isnan(h)) {
                continue;
            }
            lowest = std::min(lowest, static_cast<double>(h));
            highest = std::max(highest, static_cast<double>(h));
        }
        // The pose acts in the chunk-local frame, so clip the rays against the
        // LOCAL height range. The DEM carries plane heights; when a plane->local
        // map is in force the vertical offset is near-constant over the grid, so
        // carrying the two extremes through the map at the grid centre is enough
        // (the exact per-point map is what the rectifier applies afterwards).
        double local_low = lowest;
        double local_high = highest;
        if (camera.plane_to_local && std::isfinite(lowest) && std::isfinite(highest)) {
            const Eigen::Vector2d mid = out_grid.pixel_center(out_grid.width / 2,
                                                              out_grid.height / 2);
            const Eigen::Vector3d lo = camera.plane_to_local(Eigen::Vector3d(mid.x(), mid.y(), lowest));
            const Eigen::Vector3d hi = camera.plane_to_local(Eigen::Vector3d(mid.x(), mid.y(), highest));
            local_low = std::min(lo.z(), hi.z());
            local_high = std::max(lo.z(), hi.z());
        }
        const TileBounds bounds =
            camera_footprint_bounds(out_grid, camera, local_low, local_high, 2);
        tile.x0 = bounds.x0;
        tile.y0 = bounds.y0;
        tile.width = bounds.width;
        tile.height = bounds.height;
    } else {
        tile.width = out_grid.width;
        tile.height = out_grid.height;
    }
    if (tile.empty()) {
        return tile;
    }
    const std::size_t cells =
        static_cast<std::size_t>(tile.width) * static_cast<std::size_t>(tile.height);
    tile.rgb.assign(cells * 3, 0);
    tile.valid.assign(cells, 0);
    tile.score.assign(cells, 0.0F);

    double params[kCalibrationParamCount];
    calibration_params(camera.calibration, params);
    const Image& image = *camera.image;
    const double iw = static_cast<double>(image.width());
    const double ih = static_cast<double>(image.height());
    const double max_height = dem_max(dem, dem_nodata);

    // The occlusion march compares against DEM heights, so it must run in the
    // plane frame. When the pose lives in the chunk-local frame, the caller
    // supplies the centre already mapped into the plane frame; without a
    // plane_to_local map the pose itself is the plane-frame pose.
    const Eigen::Vector3d centre =
        camera.plane_to_local && camera.plane_centre.has_value() ? *camera.plane_centre
                                                                 : camera.centre();

    // The march step is the DEM cell, NOT the output cell: sampling the height
    // field finer than its own resolution cannot reveal an occluder that a
    // half-DEM-cell step would miss, so a user asking for an ortho finer than
    // the DEM (the normal case) does not pay for a march proportional to the
    // ortho cell. Taking the max with the output cell here used to make a
    // 4x-finer ortho walk 4x more samples per output cell, i.e. 16x the work
    // per unit area.
    //
    // When the plane frame is a Geographic one its x/y are degrees while the
    // DEM heights are metres; the march direction and the step are then built
    // in metres (mx, my scale the horizontal axes) so the ray stays a straight
    // line in real space and the step means the same thing on both.
    const double mx = camera.plane_metres_per_unit.x();
    const double my = camera.plane_metres_per_unit.y();
    const bool metric_march = mx != 1.0 || my != 1.0;
    const double step =
        0.5 * std::max(dem.grid.pixel_size_x, dem.grid.pixel_size_y) *
        (metric_march ? std::max(mx, my) : 1.0);
    const double min_dim = static_cast<double>(std::min(image.width(), image.height()));
    const double taper_span = 0.1 * min_dim;
    const double epsilon = 1e-6;

    // Rows and columns of the TILE, which index output cells x0 + col, y0 + row.
    const int width = tile.width;
    const int height = tile.height;
    const int col_start = tile.x0;
    const int row_start = tile.y0;

    // Cells are independent and each writes only its own slot, so the result
    // is bit-identical for any thread count. The inner work (a DEM sample, an
    // occlusion march, a projection and a bilinear image read) dominates the
    // stage, and a full ortho rectifies every camera over the same grid.
    std::uint64_t march_samples = 0;  // test-only cost accounting (see the header)
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : march_samples)
#endif
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const Eigen::Vector2d centre_xy =
                out_grid.pixel_center(col_start + col, row_start + row);
            const double x = centre_xy.x();
            const double y = centre_xy.y();
            const double z = sample_dem(dem, dem_nodata, x, y);
            if (z == static_cast<double>(dem_nodata)) {
                continue;  // no ground here
            }
            const Eigen::Vector3d P(x, y, z);
            // Compute the ray in metres when the plane is geographic: the x/y
            // components of `to_cam` and `to_local` (below) are degrees there.
            const Eigen::Vector3d to_cam =
                metric_march ? Eigen::Vector3d((centre.x() - x) * mx, (centre.y() - y) * my,
                                               centre.z() - z)
                             : centre - P;
            const double distance = to_cam.norm();
            if (distance <= 0.0) {
                continue;
            }
            const Eigen::Vector3d dir = to_cam / distance;

            // -- occlusion: march from P toward the camera centre -----------
            // The stop is METRIC and exact: a ray can only be blocked while it
            // is at or below the tallest DEM sample, so it terminates at the
            // point where its height reaches `max_height + tolerance`
            // (pz = P.z + dir.z * s, solved for s). That bound comes from the
            // DEM's own statistics, so it is a property of the scene and not of
            // the output grid -- the per-output-cell cost stays constant however
            // fine the ortho is. When the ray is not climbing (dir.z <= 0) or
            // the DEM has no valid sample, the ray end at `distance` is the only
            // bound. kMaxOcclusionSamplesPerRay is then purely a guard against a
            // NaN-driven loop that never advances.
            double march_end = distance;
            if (dir.z() > 0.0 && std::isfinite(max_height)) {
                const double rise =
                    (max_height + occlusion_tolerance) - P.z();
                if (rise <= 0.0) {
                    march_end = 0.0;  // already above every sample: no occluder
                } else {
                    march_end = std::min(march_end, rise / dir.z());
                }
            }
            bool occluded = false;
            std::size_t samples = 0;
            for (double s = step;
                 s <= march_end && samples < kMaxOcclusionSamplesPerRay; s += step) {
                ++samples;
                // Step in metres, then map back to plane units for the DEM read.
                const double px = P.x() + dir.x() * s / mx;
                const double py = P.y() + dir.y() * s / my;
                const double pz = P.z() + dir.z() * s;
                if (pz > max_height + occlusion_tolerance) {
                    break;  // above every surface sample: cannot be occluded
                }
                if (blocked(dem, dem_nodata, occlusion_tolerance, px, py, pz)) {
                    occluded = true;
                    break;
                }
            }
            march_samples += samples;
            if (occluded) {
                continue;
            }

            // -- project -----------------------------------------------------
            // The pose (R, t) is chunk-local when a plane_to_local map is in
            // force: map the ground point first. With no map this is the
            // historical plane-frame path untouched.
            const Eigen::Vector3d posed =
                camera.plane_to_local ? camera.plane_to_local(P) : P;
            const Eigen::Vector3d p_cam = camera.R * posed + camera.t;
            double pixel[2] = {0.0, 0.0};
            const double p_cam_arr[3] = {p_cam.x(), p_cam.y(), p_cam.z()};
            if (!try_project_point(camera.calibration.type, camera.calibration.width,
                                   camera.calibration.height, params, p_cam_arr, pixel)) {
                continue;
            }
            const double u = pixel[0];
            const double v = pixel[1];
            if (
                !(u >= 0.0 && u <= iw && v >= 0.0 && v <= ih)) {  // inside the image
                continue;
            }

            // -- score: cos(view vs vertical) * border taper ----------------
            const double cos_view = std::clamp(dir.z(), 0.0, 1.0);  // plane-frame vertical
            const double edge = std::min(std::min(u, iw - u), std::min(v, ih - v));
            const double taper = std::clamp(edge / taper_span, 0.0, 1.0);
            const double score = std::max(epsilon, cos_view * taper);

            // -- colour -----------------------------------------------------
            std::uint8_t rgb[3];
            sample_rgb(image, u, v, rgb);
            const std::size_t cell =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(col);
            tile.rgb[cell * 3 + 0] = rgb[0];
            tile.rgb[cell * 3 + 1] = rgb[1];
            tile.rgb[cell * 3 + 2] = rgb[2];
            tile.valid[cell] = 1;
            tile.score[cell] = static_cast<float>(score);
        }
    }
    // One atomic per tile, not per cell: the counter is test-only, but the
    // rectifier is on the task's hot path and must not pay for it.
    g_occlusion_samples.fetch_add(march_samples, std::memory_order_relaxed);
    return tile;
}

}  // namespace alphashape::ortho
