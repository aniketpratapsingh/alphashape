// test_ortho_texture_fill.cpp -- tests for exemplar-based hole completion.

#include "perf_gate.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <gtest/gtest.h>

#include "common/algorithm/ortho/texture_fill.hpp"

namespace alphashape {
namespace {

constexpr int kSize = 160;
constexpr int kPeriod = 6;
constexpr int kHole = 20;
constexpr int kHoleX = (kSize - kHole) / 2;
constexpr int kHoleY = (kSize - kHole) / 2;

/// A square checker (3-cell squares, so period 6) with a deterministic
/// per-pixel perturbation, and a channel-dependent offset so the three
/// channels are all non-constant.
std::uint8_t texture_byte(int x, int y, int channel) {
    const int square = ((x / (kPeriod / 2)) + (y / (kPeriod / 2))) % 2;
    const int base = square == 0 ? 80 : 180;
    const int wave = static_cast<int>((x * 13 + y * 7 + channel * 29) % 17) - 8;
    const int value = base + channel * 12 + wave;
    return static_cast<std::uint8_t>(std::clamp(value, 0, 255));
}

std::vector<std::uint8_t> make_texture() {
    std::vector<std::uint8_t> rgba(
        static_cast<std::size_t>(kSize) * static_cast<std::size_t>(kSize) * 4U, 0);
    for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
            const std::size_t i = (static_cast<std::size_t>(y) * static_cast<std::size_t>(kSize) +
                static_cast<std::size_t>(x)) * 4U;
            for (int c = 0; c < 3; ++c) {
                rgba[i + static_cast<std::size_t>(c)] = texture_byte(x, y, c);
            }
            rgba[i + 3U] = 255;
        }
    }
    return rgba;
}

struct Stats {
    std::array<double, 3> mean{};
    std::array<double, 3> stddev{};
};

/// Mean and population standard deviation of a kSize-strided region.
Stats region_stats(const std::vector<std::uint8_t>& rgba, int x0, int y0, int w, int h) {
    Stats stats;
    std::array<double, 3> sum{};
    std::array<double, 3> sum_sq{};
    const double n = static_cast<double>(w) * static_cast<double>(h);
    for (int y = y0; y < y0 + h; ++y) {
        for (int x = x0; x < x0 + w; ++x) {
            const std::size_t i = (static_cast<std::size_t>(y) * static_cast<std::size_t>(kSize) +
                static_cast<std::size_t>(x)) * 4U;
            for (int c = 0; c < 3; ++c) {
                const double v = static_cast<double>(rgba[i + static_cast<std::size_t>(c)]);
                sum[static_cast<std::size_t>(c)] += v;
                sum_sq[static_cast<std::size_t>(c)] += v * v;
            }
        }
    }
    for (int c = 0; c < 3; ++c) {
        const std::size_t ci = static_cast<std::size_t>(c);
        stats.mean[ci] = sum[ci] / n;
        const double variance = sum_sq[ci] / n - stats.mean[ci] * stats.mean[ci];
        stats.stddev[ci] = std::sqrt(std::max(variance, 0.0));
    }
    return stats;
}

/// Mean absolute RGB error over a region between two same-sized images.
double region_mae(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b, int x0,
                  int y0, int w, int h) {
    double total = 0.0;
    double count = 0.0;
    for (int y = y0; y < y0 + h; ++y) {
        for (int x = x0; x < x0 + w; ++x) {
            const std::size_t i = (static_cast<std::size_t>(y) * static_cast<std::size_t>(kSize) +
                static_cast<std::size_t>(x)) * 4U;
            for (int c = 0; c < 3; ++c) {
                total += std::abs(static_cast<double>(a[i + static_cast<std::size_t>(c)]) -
                                  static_cast<double>(b[i + static_cast<std::size_t>(c)]));
                count += 1.0;
            }
        }
    }
    return count > 0.0 ? total / count : 0.0;
}

}  // namespace

TEST(TextureFill, ReproducesTexture) {
    const std::vector<std::uint8_t> truth = make_texture();
    std::vector<std::uint8_t> started = truth;
    std::array<double, 3> mean{};
    for (int y = kHoleY; y < kHoleY + kHole; ++y) {
        for (int x = kHoleX; x < kHoleX + kHole; ++x) {
            const std::size_t i = (static_cast<std::size_t>(y) * static_cast<std::size_t>(kSize) +
                static_cast<std::size_t>(x)) * 4U;
            for (int c = 0; c < 3; ++c) {
                mean[static_cast<std::size_t>(c)] +=
                    static_cast<double>(truth[i + static_cast<std::size_t>(c)]);
            }
        }
    }
    for (double& value : mean) {
        value /= static_cast<double>(kHole) * static_cast<double>(kHole);
    }
    const std::size_t cells =
        static_cast<std::size_t>(kSize) * static_cast<std::size_t>(kSize);
    std::vector<std::uint8_t> mask(cells, 0);
    for (int y = kHoleY; y < kHoleY + kHole; ++y) {
        for (int x = kHoleX; x < kHoleX + kHole; ++x) {
            const std::size_t cell = static_cast<std::size_t>(y) * static_cast<std::size_t>(kSize) +
                static_cast<std::size_t>(x);
            const std::size_t i = cell * 4U;
            for (int c = 0; c < 3; ++c) {
                started[i + static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(
                    std::lround(mean[static_cast<std::size_t>(c)]));
            }
            started[i + 3U] = 0;
            mask[cell] = 1;
        }
    }
    const std::vector<std::uint8_t> mean_only = started;

    const auto started_clock = std::chrono::steady_clock::now();
    const ortho::TextureFillReport report =
        ortho::texture_fill(kSize, kSize, started, mask, {});
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started_clock).count();

    EXPECT_EQ(report.components, 1U);
    EXPECT_EQ(report.filled_cells, static_cast<std::size_t>(kHole * kHole));
    const Stats inside = region_stats(started, kHoleX, kHoleY, kHole, kHole);
    const Stats texture = region_stats(truth, 0, 0, kSize, kSize);
    const Stats before = region_stats(mean_only, kHoleX, kHoleY, kHole, kHole);
    const double texture_mae = region_mae(started, truth, kHoleX, kHoleY, kHole, kHole);
    const double mean_mae = region_mae(mean_only, truth, kHoleX, kHoleY, kHole, kHole);
    for (int c = 0; c < 3; ++c) {
        const std::size_t ci = static_cast<std::size_t>(c);
        EXPECT_GE(inside.stddev[ci], 0.6 * texture.stddev[ci])
            << "channel " << c << " lost the texture";
        EXPECT_LT(before.stddev[ci], 0.05 * texture.stddev[ci])
            << "the mean-colour initialisation should be almost flat";
    }
    EXPECT_LT(texture_mae, mean_mae);
    std::printf("[texture fill] std in %.2f %.2f %.2f vs out %.2f %.2f %.2f; "
                "MAE texture %.2f vs mean %.2f; %.3f s\n",
                inside.stddev[0], inside.stddev[1], inside.stddev[2], texture.stddev[0],
                texture.stddev[1], texture.stddev[2], texture_mae, mean_mae, seconds);
    if (test::perf_asserts_enabled()) {
        EXPECT_LT(seconds, 2.0);
    }
}

namespace {

/// Set the centre hole to the mean colour of the true texture and return the
/// cell mask, the starting state every completion test shares.
std::vector<std::uint8_t> make_centre_hole(std::vector<std::uint8_t>& image) {
    std::array<double, 3> mean{};
    for (int y = kHoleY; y < kHoleY + kHole; ++y) {
        for (int x = kHoleX; x < kHoleX + kHole; ++x) {
            const std::size_t i = (static_cast<std::size_t>(y) * static_cast<std::size_t>(kSize) +
                static_cast<std::size_t>(x)) * 4U;
            for (int c = 0; c < 3; ++c) {
                mean[static_cast<std::size_t>(c)] +=
                    static_cast<double>(image[i + static_cast<std::size_t>(c)]);
            }
        }
    }
    for (double& value : mean) {
        value /= static_cast<double>(kHole) * static_cast<double>(kHole);
    }
    std::vector<std::uint8_t> mask(
        static_cast<std::size_t>(kSize) * static_cast<std::size_t>(kSize), 0);
    for (int y = kHoleY; y < kHoleY + kHole; ++y) {
        for (int x = kHoleX; x < kHoleX + kHole; ++x) {
            const std::size_t cell = static_cast<std::size_t>(y) * static_cast<std::size_t>(kSize) +
                static_cast<std::size_t>(x);
            for (int c = 0; c < 3; ++c) {
                image[cell * 4U + static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(
                    std::lround(mean[static_cast<std::size_t>(c)]));
            }
            image[cell * 4U + 3U] = 0;
            mask[cell] = 1;
        }
    }
    return mask;
}

}  // namespace

TEST(TextureFill, Deterministic) {
    const std::vector<std::uint8_t> truth = make_texture();
    std::vector<std::uint8_t> first = truth;
    std::vector<std::uint8_t> second = truth;
    const std::vector<std::uint8_t> mask = make_centre_hole(first);
    static_cast<void>(make_centre_hole(second));
    const ortho::TextureFillReport a = ortho::texture_fill(kSize, kSize, first, mask, {});
    const ortho::TextureFillReport b = ortho::texture_fill(kSize, kSize, second, mask, {});
    EXPECT_EQ(a.filled_cells, b.filled_cells);
    EXPECT_EQ(first, second);
}

TEST(TextureFill, WindowCapSkipsThinDiagonalComponent) {
    // A diagonal is the bounding-box trap the old estimate missed: 64 cells, but
    // its margin-dilated window is the whole raster. The window cap, not the
    // cell count, decides, so this component must be skipped.
    constexpr int width = 64;
    constexpr int height = 64;
    std::vector<std::uint8_t> rgba(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4U, 120);
    std::vector<std::uint8_t> mask(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
    for (std::size_t i = 0; i < mask.size(); ++i) {
        rgba[i * 4U + 3U] = 255;
    }
    for (int k = 0; k < width; ++k) {
        mask[static_cast<std::size_t>(k) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(k)] = 1;
    }
    const std::vector<std::uint8_t> before = rgba;
    ortho::TextureFillOptions options;
    options.max_window_cells = 1000;  // below the clamped 64x64 = 4096 window
    const ortho::TextureFillReport report =
        ortho::texture_fill(width, height, rgba, mask, options);
    EXPECT_EQ(report.components, 1U);
    EXPECT_EQ(report.skipped_large, 1U);
    EXPECT_EQ(report.filled_cells, 0U);
    EXPECT_EQ(rgba, before);
}

TEST(TextureFill, LargeComponentSkipped) {
    constexpr int width = 40;
    constexpr int height = 40;
    std::vector<std::uint8_t> rgba(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4U, 120);
    std::vector<std::uint8_t> mask(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
    for (std::size_t i = 0; i < mask.size(); ++i) {
        rgba[i * 4U + 3U] = 255;
    }
    for (int y = 5; y < 35; ++y) {
        for (int x = 5; x < 35; ++x) {
            mask[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(x)] = 1;
        }
    }
    const std::vector<std::uint8_t> before = rgba;
    ortho::TextureFillOptions options;
    options.max_component_cells = 100;
    const ortho::TextureFillReport report =
        ortho::texture_fill(width, height, rgba, mask, options);
    EXPECT_EQ(report.components, 1U);
    EXPECT_EQ(report.skipped_large, 1U);
    EXPECT_EQ(report.filled_cells, 0U);
    EXPECT_EQ(rgba, before);
}

TEST(TextureFill, ValidCellsUntouched) {
    std::vector<std::uint8_t> image = make_texture();
    const std::vector<std::uint8_t> mask = make_centre_hole(image);
    const std::vector<std::uint8_t> before = image;
    static_cast<void>(ortho::texture_fill(kSize, kSize, image, mask, {}));
    std::size_t mismatched = 0;
    for (std::size_t cell = 0; cell < mask.size(); ++cell) {
        if (mask[cell] != 0) {
            continue;
        }
        for (std::size_t c = 0; c < 4; ++c) {
            if (image[cell * 4U + c] != before[cell * 4U + c]) {
                ++mismatched;
            }
        }
    }
    EXPECT_EQ(mismatched, 0U);
}

TEST(TextureFill, NoSourceSkips) {
    constexpr int width = 64;
    constexpr int height = 64;
    std::vector<std::uint8_t> rgba(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4U, 0);
    std::vector<std::uint8_t> mask(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
    for (int y = 20; y < 40; ++y) {
        for (int x = 20; x < 40; ++x) {
            const std::size_t cell = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(x);
            for (std::size_t c = 0; c < 3; ++c) {
                rgba[cell * 4U + c] = 90;
            }
            mask[cell] = 1;
        }
    }
    const std::vector<std::uint8_t> before = rgba;
    const ortho::TextureFillReport report =
        ortho::texture_fill(width, height, rgba, mask, {});
    EXPECT_EQ(report.components, 1U);
    EXPECT_EQ(report.filled_cells, 0U);
    EXPECT_EQ(report.skipped_large, 1U);
    EXPECT_EQ(rgba, before);
}

}  // namespace alphashape
