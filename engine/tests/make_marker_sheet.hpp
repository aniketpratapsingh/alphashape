#pragma once

// Synthetic marker sheets for the marker-detector tests.
//
// Filled ellipses and squares are rasterised with 16x16 supersampling per pixel
// inside each target's bounding box, so the analytic area coverage of an
// anti-aliased target is represented to 1/256 of a pixel and the ground-truth
// centre, axes and corners are exact. Test code then applies a Gaussian blur and
// additive Gaussian noise with the engine's helpers.
//
// The JPEG-quality variant is a real 8x8 block DCT with the standard luminance
// quantisation table scaled for the requested quality, followed by rounding and
// the inverse DCT. The tree only encodes PNG and decodes JPEG (codec.hpp exposes
// no JPEG encoder), so the test simulates the compression instead of going
// through a codec: no chroma subsampling and no colour conversion, because the
// input is single-channel.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <random>
#include <vector>

#include <Eigen/Core>

#include "base/utils/portable_rng.hpp"
#include "image/image.hpp"

namespace alphashape::test {

struct EllipseTruth {
    double cx = 0.0;
    double cy = 0.0;
    double a = 1.0;
    double b = 1.0;
    double angle_rad = 0.0;
};

struct SquareTruth {
    double cx = 0.0;
    double cy = 0.0;
    double side = 1.0;
    double angle_rad = 0.0;
};

inline FloatImage render_ellipses(int width, int height, const std::vector<EllipseTruth>& targets,
                                  float background, float foreground, int supersample = 16) {
    FloatImage image(width, height, background);
    const int samples = std::max(1, supersample);
    const double total = static_cast<double>(samples) * static_cast<double>(samples);
    const double step = 1.0 / static_cast<double>(samples);
    for (const EllipseTruth& target : targets) {
        const double c = std::cos(target.angle_rad);
        const double s = std::sin(target.angle_rad);
        const double extent = std::max(target.a, target.b) + 1.0;
        const int x0 = std::max(0, static_cast<int>(std::floor(target.cx - extent)));
        const int y0 = std::max(0, static_cast<int>(std::floor(target.cy - extent)));
        const int x1 = std::min(width - 1, static_cast<int>(std::ceil(target.cx + extent)));
        const int y1 = std::min(height - 1, static_cast<int>(std::ceil(target.cy + extent)));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                int inside = 0;
                for (int sy = 0; sy < samples; ++sy) {
                    for (int sx = 0; sx < samples; ++sx) {
                        const double px = static_cast<double>(x) +
                                          (static_cast<double>(sx) + 0.5) * step;
                        const double py = static_cast<double>(y) +
                                          (static_cast<double>(sy) + 0.5) * step;
                        const double dx = px - target.cx;
                        const double dy = py - target.cy;
                        const double u = dx * c + dy * s;
                        const double v = -dx * s + dy * c;
                        if ((u * u) / (target.a * target.a) + (v * v) / (target.b * target.b) <=
                            1.0) {
                            ++inside;
                        }
                    }
                }
                const double coverage = static_cast<double>(inside) / total;
                const double value =
                    static_cast<double>(background) +
                    (static_cast<double>(foreground) - static_cast<double>(background)) * coverage;
                image.at(x, y) = static_cast<float>(value);
            }
        }
    }
    return image;
}

inline FloatImage render_squares(int width, int height, const std::vector<SquareTruth>& targets,
                                 float background, float foreground, int supersample = 16) {
    FloatImage image(width, height, background);
    const int samples = std::max(1, supersample);
    const double total = static_cast<double>(samples) * static_cast<double>(samples);
    const double step = 1.0 / static_cast<double>(samples);
    for (const SquareTruth& target : targets) {
        const double c = std::cos(target.angle_rad);
        const double s = std::sin(target.angle_rad);
        const double half = 0.5 * target.side;
        const double extent = half * std::sqrt(2.0) + 1.0;
        const int x0 = std::max(0, static_cast<int>(std::floor(target.cx - extent)));
        const int y0 = std::max(0, static_cast<int>(std::floor(target.cy - extent)));
        const int x1 = std::min(width - 1, static_cast<int>(std::ceil(target.cx + extent)));
        const int y1 = std::min(height - 1, static_cast<int>(std::ceil(target.cy + extent)));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                int inside = 0;
                for (int sy = 0; sy < samples; ++sy) {
                    for (int sx = 0; sx < samples; ++sx) {
                        const double px = static_cast<double>(x) +
                                          (static_cast<double>(sx) + 0.5) * step;
                        const double py = static_cast<double>(y) +
                                          (static_cast<double>(sy) + 0.5) * step;
                        const double dx = px - target.cx;
                        const double dy = py - target.cy;
                        const double u = dx * c + dy * s;
                        const double v = -dx * s + dy * c;
                        if (std::abs(u) <= half && std::abs(v) <= half) {
                            ++inside;
                        }
                    }
                }
                const double coverage = static_cast<double>(inside) / total;
                const double value =
                    static_cast<double>(background) +
                    (static_cast<double>(foreground) - static_cast<double>(background)) * coverage;
                image.at(x, y) = static_cast<float>(value);
            }
        }
    }
    return image;
}

/// The four corners of a square truth, in the same order as the renderer's
/// local frame: (-h,-h), (h,-h), (h,h), (-h,h) rotated and translated.
inline std::array<Eigen::Vector2d, 4> square_corners(const SquareTruth& target) {
    const double half = 0.5 * target.side;
    const double c = std::cos(target.angle_rad);
    const double s = std::sin(target.angle_rad);
    const double local[4][2] = {{-half, -half}, {half, -half}, {half, half}, {-half, half}};
    std::array<Eigen::Vector2d, 4> corners{};
    for (std::size_t i = 0; i < corners.size(); ++i) {
        const double u = local[i][0];
        const double v = local[i][1];
        corners[i] = Eigen::Vector2d(target.cx + u * c - v * s, target.cy + u * s + v * c);
    }
    return corners;
}

inline void add_noise(FloatImage& image, double sigma, std::uint32_t seed) {
    std::mt19937 engine(seed);
    std::vector<float>& pixels = image.pixels();
    for (float& value : pixels) {
        const double noise = rng::normal(engine, 0.0, sigma);
        value = static_cast<float>(static_cast<double>(value) + noise);
    }
}

inline double cosine_table_entry(int x, int u) {
    return std::cos((2.0 * static_cast<double>(x) + 1.0) * static_cast<double>(u) *
                    std::numbers::pi_v<double> / 16.0);
}

inline const std::array<std::array<double, 8>, 8>& cosine_table() {
    static const std::array<std::array<double, 8>, 8> table = [] {
        std::array<std::array<double, 8>, 8> values{};
        for (int x = 0; x < 8; ++x) {
            for (int u = 0; u < 8; ++u) {
                values[static_cast<std::size_t>(x)][static_cast<std::size_t>(u)] =
                    cosine_table_entry(x, u);
            }
        }
        return values;
    }();
    return table;
}

inline double dct_scale(int frequency) {
    return frequency == 0 ? std::sqrt(0.5) : 1.0;
}

inline std::size_t block_index(int row, int column) {
    return static_cast<std::size_t>(row * 8 + column);
}

inline void dct8x8(const std::array<double, 64>& input, std::array<double, 64>& output) {
    const auto& table = cosine_table();
    for (int u = 0; u < 8; ++u) {
        for (int v = 0; v < 8; ++v) {
            double sum = 0.0;
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    sum += input[block_index(y, x)] *
                           table[static_cast<std::size_t>(x)][static_cast<std::size_t>(u)] *
                           table[static_cast<std::size_t>(y)][static_cast<std::size_t>(v)];
                }
            }
            output[block_index(u, v)] = 0.25 * dct_scale(u) * dct_scale(v) * sum;
        }
    }
}

inline void idct8x8(const std::array<double, 64>& input, std::array<double, 64>& output) {
    const auto& table = cosine_table();
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            double sum = 0.0;
            for (int u = 0; u < 8; ++u) {
                for (int v = 0; v < 8; ++v) {
                    sum += dct_scale(u) * dct_scale(v) * input[block_index(u, v)] *
                           table[static_cast<std::size_t>(x)][static_cast<std::size_t>(u)] *
                           table[static_cast<std::size_t>(y)][static_cast<std::size_t>(v)];
                }
            }
            output[block_index(y, x)] = 0.25 * sum;
        }
    }
}

inline std::array<int, 64> jpeg_quantisation_table(int quality) {
    static const std::array<int, 64> base = {
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};
    const int clamped_quality = std::clamp(quality, 1, 100);
    const int scale =
        clamped_quality < 50 ? 5000 / clamped_quality : 200 - 2 * clamped_quality;
    std::array<int, 64> table{};
    for (std::size_t i = 0; i < table.size(); ++i) {
        table[i] = std::clamp((base[i] * scale + 50) / 100, 1, 255);
    }
    return table;
}

/// Quantise each 8x8 block's DCT coefficients with the scaled luminance table
/// and invert. This is the lossy part of a baseline JPEG encoder without chroma
/// subsampling or a colour transform, which is the right model for a
/// single-channel target image.
inline FloatImage degrade_like_jpeg(const FloatImage& input, int quality = 70) {
    const int w = input.width();
    const int h = input.height();
    FloatImage output(w, h, 0.0F);
    const std::array<int, 64> quantisation = jpeg_quantisation_table(quality);
    for (int block_y = 0; block_y < h; block_y += 8) {
        for (int block_x = 0; block_x < w; block_x += 8) {
            std::array<double, 64> block{};
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    const int sx = std::min(w - 1, block_x + x);
                    const int sy = std::min(h - 1, block_y + y);
                    block[block_index(y, x)] =
                        static_cast<double>(input.at(sx, sy)) * 255.0 - 128.0;
                }
            }
            std::array<double, 64> coefficients{};
            dct8x8(block, coefficients);
            for (std::size_t i = 0; i < coefficients.size(); ++i) {
                const double step = static_cast<double>(quantisation[i]);
                coefficients[i] = std::round(coefficients[i] / step) * step;
            }
            std::array<double, 64> restored{};
            idct8x8(coefficients, restored);
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    const int px = block_x + x;
                    const int py = block_y + y;
                    if (px >= w || py >= h) continue;
                    const double value = (restored[block_index(y, x)] + 128.0) / 255.0;
                    output.at(px, py) = static_cast<float>(std::clamp(value, 0.0, 1.0));
                }
            }
        }
    }
    return output;
}

}  // namespace alphashape::test
