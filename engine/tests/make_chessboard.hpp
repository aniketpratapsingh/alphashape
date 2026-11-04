#pragma once

// Synthetic chessboard renderer for the WP-13.14 detector tests.
//
// A planar chessboard of (cols + 1) x (rows + 1) squares is rendered through a
// pinhole camera with a known rotation, translation, focal length and radial
// distortion k1. Each pixel is supersampled 4x4 with a box filter, so square
// borders are anti-aliased rather than hard-edged. The background is mid-grey
// with a mild low-frequency texture; the squares use flat black and white.
//
// Coordinates use the calibration convention (the centre of the top-left pixel
// is (0.5, 0.5), matching Calibration and BoardCorner). The board plane has its
// inner-corner lattice at (c * square, r * square, 0), so the returned truth
// for inner corner (c, r) is the projection of that point -- the same physical
// point a detector labels (col, row).
//
// Rendering inverts the camera model itself (a Newton solve of the k1 radial
// polynomial) instead of calling Calibration::unproject per subsample, which is
// far too slow at 20 million samples; forward projection uses Calibration, so
// truth pixels and rendered pixels follow exactly the same convention.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include "image/image.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::test {

struct ChessboardPose {
    int width = 1280;
    int height = 960;
    int cols = 9;   ///< Inner corners across.
    int rows = 6;   ///< Inner corners down.
    double square_m = 0.025;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();  ///< world -> camera
    Eigen::Vector3d translation = Eigen::Vector3d(0.0, 0.0, 0.55);
    double f = 1000.0;  ///< Focal length in pixels.
    double k1 = 0.0;
    double white = 0.88;
    double black = 0.12;
    double background = 0.5;
    double texture = 0.03;  ///< Amplitude of the background texture.
    int supersample = 4;
};

struct ChessboardRender {
    FloatImage image;
    std::vector<Eigen::Vector2d> corners;  ///< Truth pixel per inner corner, row-major.
    int cols = 0;
    int rows = 0;

    Eigen::Vector2d corner(int col, int row) const {
        return corners[static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
                       static_cast<std::size_t>(col)];
    }
};

/// Camera-frame ray through a calibration-convention pixel, with radial k1.
inline Eigen::Vector3d ray_through_pixel(double u, double v, int width, int height,
                                         double f, double k1) {
    const double xd = (u - 0.5 * static_cast<double>(width)) / f;
    const double yd = (v - 0.5 * static_cast<double>(height)) / f;
    const double rd = std::hypot(xd, yd);
    double ru = rd;
    for (int iteration = 0; iteration < 6; ++iteration) {
        const double g = ru * (1.0 + k1 * ru * ru) - rd;
        const double slope = 1.0 + 3.0 * k1 * ru * ru;
        if (std::abs(slope) < 1e-9) {
            break;
        }
        ru -= g / slope;
    }
    const double scale = rd > 1e-12 ? ru / rd : 1.0;
    return Eigen::Vector3d(xd * scale, yd * scale, 1.0);
}

inline double background_at(double u, double v, const ChessboardPose& pose) {
    return pose.background + pose.texture * std::sin(0.037 * u + 0.9) *
                                 std::cos(0.029 * v - 0.4);
}

/// Cell parity of a board-plane point given in square units.
inline bool board_cell_black(double u, double v) {
    const int cell_x = static_cast<int>(std::floor(u));
    const int cell_y = static_cast<int>(std::floor(v));
    return ((cell_x + cell_y) % 2) != 0;
}

inline ChessboardRender render_chessboard(const ChessboardPose& pose) {
    ChessboardRender render;
    render.cols = pose.cols;
    render.rows = pose.rows;
    const int w = pose.width;
    const int h = pose.height;
    if (w <= 0 || h <= 0) {
        return render;
    }
    const double sq = pose.square_m;
    const int cols = pose.cols;
    const int rows = pose.rows;
    Calibration cal;
    cal.type = CameraType::Frame;
    cal.width = w;
    cal.height = h;
    cal.f = pose.f;
    cal.k1 = pose.k1;
    render.image = FloatImage(w, h, 0.0F);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            render.image.at(x, y) = static_cast<float>(
                background_at(static_cast<double>(x) + 0.5,
                              static_cast<double>(y) + 0.5, pose));
        }
    }
    const double xs[2] = {-sq, static_cast<double>(cols) * sq};
    const double ys[2] = {-sq, static_cast<double>(rows) * sq};
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -min_x;
    double min_y = min_x;
    double max_y = -min_x;
    bool projected = false;
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            const Eigen::Vector3d world(xs[a], ys[b], 0.0);
            const Eigen::Vector3d camera = pose.rotation * world + pose.translation;
            if (!(camera.z() > 0.0)) {
                continue;
            }
            const Eigen::Vector2d pixel = cal.project(camera);
            min_x = std::min(min_x, pixel.x());
            max_x = std::max(max_x, pixel.x());
            min_y = std::min(min_y, pixel.y());
            max_y = std::max(max_y, pixel.y());
            projected = true;
        }
    }

    const double margin = 12.0;
    if (projected) {
        const int ix0 = std::max(0, static_cast<int>(std::floor(min_x - margin)));
        const int ix1 = std::min(w - 1, static_cast<int>(std::ceil(max_x + margin)));
        const int iy0 = std::max(0, static_cast<int>(std::floor(min_y - margin)));
        const int iy1 = std::min(h - 1, static_cast<int>(std::ceil(max_y + margin)));
        const Eigen::Vector3d normal = pose.rotation.col(2);
        const double offset = normal.dot(pose.translation);
        const int samples = std::max(1, pose.supersample);
        const double total = static_cast<double>(samples * samples);
        for (int y = iy0; y <= iy1; ++y) {
            for (int x = ix0; x <= ix1; ++x) {
                double sum = 0.0;
                for (int sy = 0; sy < samples; ++sy) {
                    for (int sx = 0; sx < samples; ++sx) {
                        const double su =
                            static_cast<double>(x) +
                            (static_cast<double>(sx) + 0.5) / static_cast<double>(samples);
                        const double sv =
                            static_cast<double>(y) +
                            (static_cast<double>(sy) + 0.5) / static_cast<double>(samples);
                        const Eigen::Vector3d ray =
                            ray_through_pixel(su, sv, w, h, pose.f, pose.k1);
                        const double denom = normal.dot(ray);
                        double colour = background_at(su, sv, pose);
                        if (std::abs(denom) > 1e-12) {
                            const double lambda = offset / denom;
                            if (lambda > 0.0) {
                                const Eigen::Vector3d world =
                                    pose.rotation.transpose() *
                                    (lambda * ray - pose.translation);
                                const double board_u = world.x() / sq;
                                const double board_v = world.y() / sq;
                                if (board_u >= -1.0 &&
                                    board_u <= static_cast<double>(cols) &&
                                    board_v >= -1.0 &&
                                    board_v <= static_cast<double>(rows)) {
                                    colour = board_cell_black(board_u, board_v) ? pose.black
                                                                                : pose.white;
                                }
                            }
                        }
                        sum += colour;
                    }
                }
                render.image.at(x, y) = static_cast<float>(sum / total);
            }
        }
    }

    render.corners.reserve(static_cast<std::size_t>(cols * rows));
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const Eigen::Vector3d world(static_cast<double>(col) * sq,
                                        static_cast<double>(row) * sq, 0.0);
            const Eigen::Vector3d camera = pose.rotation * world + pose.translation;
            if (!(camera.z() > 0.0)) {
                render.corners.emplace_back(
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN());
                continue;
            }
            render.corners.push_back(cal.project(camera));
        }
    }
    return render;
}

}  // namespace alphashape::test
