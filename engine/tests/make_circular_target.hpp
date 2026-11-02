#pragma once

// Synthetic circular coded targets (WP-13.6 tests).
//
// The concentric pattern of target_patterns.hpp is evaluated analytically per
// subsample in the target frame, so anti-aliased edges are represented by the
// supersampled black fraction of each pixel. The affine `transform` maps the
// unit target (u, v) frame into image coordinates; a rotation plus an
// anisotropic scale models a target seen under tilt. `cx`, `cy` use the
// calibration convention (pixel x spans the continuous interval [x, x + 1)),
// matching the analytic ellipse renderer in make_marker_sheet.hpp. The decoder
// and the detector's MarkerCandidate use the same convention; FloatImage::sample
// addresses pixel centres at integer coordinates, half a pixel lower.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

#include <Eigen/Core>

#include "image/image.hpp"

namespace alphashape::test {

struct CircularTruth {
    double cx = 0.0;
    double cy = 0.0;
    double radius = 1.0;  ///< Semi-major outer radius R in pixels.
    Eigen::Matrix2d transform = Eigen::Matrix2d::Identity();
    int bits = 12;
    std::uint32_t word = 0;  ///< Ring word, sector 0 in the high bit.
    int parity = 1;          ///< 1 = black centre dot, 0 = white.
};

inline CircularTruth make_circular_truth(double cx, double cy, double radius, int bits,
                                         std::uint32_t word, int parity, double rotation = 0.0,
                                         double axis_ratio = 1.0) {
    CircularTruth truth;
    truth.cx = cx;
    truth.cy = cy;
    truth.radius = radius;
    truth.bits = bits;
    truth.word = word;
    truth.parity = parity;
    const double c = std::cos(rotation);
    const double s = std::sin(rotation);
    Eigen::Matrix2d rotate;
    rotate << c, -s, s, c;
    Eigen::Matrix2d scale = Eigen::Matrix2d::Zero();
    scale(0, 0) = radius;
    scale(1, 1) = radius * axis_ratio;
    truth.transform = rotate * scale;
    return truth;
}

/// Bounding half-extent of the target in x and y (max absolute row sum).
inline double circular_extent(const CircularTruth& truth) {
    return std::max(truth.transform.cwiseAbs().row(0).sum(),
                    truth.transform.cwiseAbs().row(1).sum());
}

/// True when a point in the target frame is painted black.
inline bool circular_black_at(const Eigen::Vector2d& local, const CircularTruth& truth,
                              double sector_width) {
    const double rho = local.norm();
    if (rho > 1.0) return false;
    if (rho <= 0.25) return truth.parity != 0;
    if (rho <= 0.5) return false;
    if (rho <= 0.85) {
        double relative = std::atan2(local.y(), local.x()) +
                          0.5 * std::numbers::pi_v<double> + 2.0 * std::numbers::pi_v<double>;
        relative = std::fmod(relative, 2.0 * std::numbers::pi_v<double>);
        int sector = static_cast<int>(std::floor(relative / sector_width));
        if (sector < 0) sector = 0;
        sector %= truth.bits;
        const int shift = truth.bits - 1 - sector;
        return ((truth.word >> static_cast<unsigned>(shift)) & 1u) != 0u;
    }
    if (rho <= 0.9) return false;
    return true;
}

/// Paint one target into an existing image (background already filled).
inline void draw_circular_target(FloatImage& image, const CircularTruth& truth,
                                 float background, float foreground, int supersample) {
    const int samples = std::max(1, supersample);
    const double total = static_cast<double>(samples) * static_cast<double>(samples);
    const double step = 1.0 / static_cast<double>(samples);
    const double sector_width = 2.0 * std::numbers::pi_v<double> /
                                static_cast<double>(truth.bits);
    const Eigen::Matrix2d inverse = truth.transform.inverse();
    const double extent = circular_extent(truth) + 1.0;
    const int x0 = std::max(0, static_cast<int>(std::floor(truth.cx - extent)));
    const int y0 = std::max(0, static_cast<int>(std::floor(truth.cy - extent)));
    const int x1 = std::min(image.width() - 1, static_cast<int>(std::ceil(truth.cx + extent)));
    const int y1 = std::min(image.height() - 1, static_cast<int>(std::ceil(truth.cy + extent)));
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            int black = 0;
            for (int sy = 0; sy < samples; ++sy) {
                for (int sx = 0; sx < samples; ++sx) {
                    const double px =
                        static_cast<double>(x) + (static_cast<double>(sx) + 0.5) * step;
                    const double py =
                        static_cast<double>(y) + (static_cast<double>(sy) + 0.5) * step;
                    const Eigen::Vector2d offset(px - truth.cx, py - truth.cy);
                    const Eigen::Vector2d local = inverse * offset;
                    if (circular_black_at(local, truth, sector_width)) ++black;
                }
            }
            const double coverage = static_cast<double>(black) / total;
            image.at(x, y) = static_cast<float>(
                static_cast<double>(background) +
                (static_cast<double>(foreground) - static_cast<double>(background)) * coverage);
        }
    }
}

inline FloatImage render_circular_targets(int width, int height,
                                          const std::vector<CircularTruth>& targets,
                                          float background = 0.95F, float foreground = 0.05F,
                                          int supersample = 8) {
    FloatImage image(width, height, background);
    for (const CircularTruth& target : targets) {
        draw_circular_target(image, target, background, foreground, supersample);
    }
    return image;
}

inline FloatImage render_circular_target(int width, int height, const CircularTruth& target,
                                         float background = 0.95F, float foreground = 0.05F,
                                         int supersample = 8) {
    return render_circular_targets(width, height, {target}, background, foreground, supersample);
}

}  // namespace alphashape::test
