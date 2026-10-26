// True orthorectification: see orthorectify.hpp for the contract.

#include "common/algorithm/ortho/orthorectify.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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

}  // namespace

OrthoTile orthorectify(const GeoRaster<float>& dem, float dem_nodata,
                       const RasterGrid& out_grid, const OrthoCamera& camera,
                       int camera_index, double occlusion_tolerance) {
    OrthoTile tile;
    tile.camera = camera_index;
    const int width = out_grid.width;
    const int height = out_grid.height;
    const std::size_t cells =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    tile.rgb.assign(cells * 3, 0);
    tile.valid.assign(cells, 0);
    tile.score.assign(cells, 0.0F);
    if (camera.image == nullptr || camera.image->empty() || dem.empty()) {
        return tile;
    }

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

    // The march step is bounded below by the DEM cell: sampling the height
    // field finer than its own resolution cannot reveal an occluder that a
    // half-cell step would miss. A user asking for an ortho finer than the DEM
    // (the normal case) therefore does not pay for a march proportional to the
    // ortho cell.
    //
    // When the plane frame is a Geographic one its x/y are degrees while the
    // DEM heights are metres; the march direction and the step are then built
    // in metres (mx, my scale the horizontal axes) so the ray stays a straight
    // line in real space and the step means the same thing on both.
    const double mx = camera.plane_metres_per_unit.x();
    const double my = camera.plane_metres_per_unit.y();
    const bool metric_march = mx != 1.0 || my != 1.0;
    const double step =
        0.5 * std::max(out_grid.pixel_size_x, dem.grid.pixel_size_x) *
        (metric_march ? std::max(mx, my) : 1.0);
    const double min_dim = static_cast<double>(std::min(image.width(), image.height()));
    const double taper_span = 0.1 * min_dim;
    const double epsilon = 1e-6;

    // Cells are independent and each writes only its own slot, so the result
    // is bit-identical for any thread count. The inner work (a DEM sample, an
    // occlusion march, a projection and a bilinear image read) dominates the
    // stage, and a full ortho rectifies every camera over the same grid.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const Eigen::Vector2d centre_xy = out_grid.pixel_center(col, row);
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
            bool occluded = false;
            for (double s = step; s < distance; s += step) {
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
    return tile;
}

}  // namespace alphashape::ortho
