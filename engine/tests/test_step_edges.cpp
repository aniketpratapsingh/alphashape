// test_step_edges.cpp -- the opt-in DEM step-edge refinement pass (BuildDem).
//
// The synthetic scene has three rectangular buildings over a gently sloping
// ground plane: two flat roofs and one gable roof, one building rotated 30
// degrees. The truth surface is blurred with a Gaussian of sigma = 1.5 cells
// to mimic stereo smearing and then carries 0.05 m of noise. The tests measure
// the height error in a 3-cell band around the true outlines before and after
// the pass, and check that far-from-outline cells are untouched.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "common/algorithm/elevation/build_dem_task.hpp"
#include "common/algorithm/elevation/step_edges.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/tasks/params.hpp"
#include "common/tasks/registry.hpp"
#include "model/camera/calibration.hpp"
#include "perf_gate.hpp"

namespace alphashape {
namespace {

using elevation::refine_step_edges;
using elevation::StepEdgeOptions;

constexpr double kCell = 0.25;  ///< metres per cell in the synthetic scenes
constexpr double kPi = 3.14159265358979323846;

/// Deterministic standard normal from a xorshift32 plus Box-Muller, so the
/// scene does not depend on a standard library's distribution implementation.
struct Noise {
    std::uint32_t state = 0x9e3779b9U;

    double uniform() {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        return (static_cast<double>(state) + 0.5) / 4294967296.0;
    }
    double normal() {
        const double u1 = uniform();
        const double u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
    }
};

/// An oriented rectangle footprint in metres (centre, half extents, rotation).
struct Rect {
    double cx = 0.0;
    double cy = 0.0;
    double hx = 1.0;
    double hy = 1.0;
    double angle = 0.0;
};

/// A flat-roofed or gable-roofed building. The gable's ridge runs along the
/// rectangle's local x axis at local y = 0.
struct Building {
    Rect rect;
    bool gable = false;
    double flat_height = 0.0;  ///< metres above ground, flat roof
    double eave_height = 0.0;  ///< metres above ground, gable eaves
    double ridge_height = 0.0; ///< metres above ground, gable ridge
};

/// The ground: a shallow plane sloping along x.
double ground_height(double x) {
    return 2.0 + 0.05 * x;
}

/// Local (dx, dy) of a point in a rectangle's frame.
void local_xy(const Rect& rect, double x, double y, double& dx, double& dy) {
    const double c = std::cos(rect.angle);
    const double s = std::sin(rect.angle);
    const double px = x - rect.cx;
    const double py = y - rect.cy;
    dx = c * px + s * py;
    dy = -s * px + c * py;
}

/// Height of a building's surface at (x, y), or NaN outside its footprint.
double building_height(const Building& building, double x, double y) {
    double dx = 0.0;
    double dy = 0.0;
    local_xy(building.rect, x, y, dx, dy);
    if (std::abs(dx) > building.rect.hx || std::abs(dy) > building.rect.hy) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (!building.gable) {
        return ground_height(x) + building.flat_height;
    }
    const double rise = (building.ridge_height - building.eave_height) *
                        (1.0 - std::abs(dy) / building.rect.hy);
    return ground_height(x) + building.eave_height + rise;
}

/// Distance from (x, y) in METRES to a rectangle's outline.
double outline_distance(const Rect& rect, double x, double y) {
    double dx = 0.0;
    double dy = 0.0;
    local_xy(rect, x, y, dx, dy);
    const double qx = std::abs(dx) - rect.hx;
    const double qy = std::abs(dy) - rect.hy;
    const double outside = std::hypot(std::max(qx, 0.0), std::max(qy, 0.0));
    const double inside = std::min(std::max(qx, qy), 0.0);
    return std::abs(outside + inside);
}

/// Three buildings on a 40 m x 40 m grid: two flat roofs (one rotated 30
/// degrees) and one gable roof. Coordinates are metres.
std::vector<Building> make_buildings() {
    Building flat;
    flat.rect = Rect{8.75, 8.75, 3.75, 2.5, 0.0};
    flat.flat_height = 5.0;

    Building gable;
    gable.rect = Rect{26.25, 8.75, 5.0, 3.75, 0.0};
    gable.gable = true;
    gable.eave_height = 4.0;
    gable.ridge_height = 7.0;

    Building rotated;
    rotated.rect = Rect{20.0, 26.25, 4.5, 2.5, 30.0 * kPi / 180.0};
    rotated.flat_height = 5.0;

    return {flat, gable, rotated};
}

/// The noise-free, unblurred scene: ground, with each building's surface
/// clamped on top.
std::vector<double> truth_surface(const std::vector<Building>& buildings, int width,
                                  int height) {
    std::vector<double> truth(static_cast<std::size_t>(width) *
                              static_cast<std::size_t>(height));
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const double x = static_cast<double>(col) * kCell;
            const double y = static_cast<double>(row) * kCell;
            double value = ground_height(x);
            for (const Building& building : buildings) {
                const double roof = building_height(building, x, y);
                if (std::isfinite(roof)) {
                    value = roof;
                }
            }
            truth[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                  static_cast<std::size_t>(col)] = value;
        }
    }
    return truth;
}

/// Separable Gaussian blur with clamped edges.
std::vector<double> gaussian_blur(const std::vector<double>& in, int width, int height,
                                  double sigma) {
    const int radius = static_cast<int>(std::ceil(3.0 * sigma));
    std::vector<double> kernel(static_cast<std::size_t>(2 * radius + 1), 0.0);
    double sum = 0.0;
    for (int d = -radius; d <= radius; ++d) {
        const double value = std::exp(-0.5 * static_cast<double>(d) * static_cast<double>(d) /
                                      (sigma * sigma));
        kernel[static_cast<std::size_t>(d + radius)] = value;
        sum += value;
    }
    for (double& value : kernel) {
        value /= sum;
    }
    const auto clamp_index = [](int value, int limit) {
        return std::min(std::max(value, 0), limit - 1);
    };
    std::vector<double> horizontal(in.size(), 0.0);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            double acc = 0.0;
            for (int d = -radius; d <= radius; ++d) {
                const int sc = clamp_index(col + d, width);
                acc += kernel[static_cast<std::size_t>(d + radius)] *
                       in[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                          static_cast<std::size_t>(sc)];
            }
            horizontal[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                       static_cast<std::size_t>(col)] = acc;
        }
    }
    std::vector<double> out(in.size(), 0.0);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            double acc = 0.0;
            for (int d = -radius; d <= radius; ++d) {
                const int sr = clamp_index(row + d, height);
                acc += kernel[static_cast<std::size_t>(d + radius)] *
                       horizontal[static_cast<std::size_t>(sr) * static_cast<std::size_t>(width) +
                                  static_cast<std::size_t>(col)];
            }
            out[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(col)] = acc;
        }
    }
    return out;
}

/// Smooth deterministic value noise in [-1, 1].
double value_noise(double x, double y, std::uint32_t seed) {
    const auto hash = [seed](int ix, int iy) {
        std::uint32_t h = static_cast<std::uint32_t>(ix) * 374761393U +
                          static_cast<std::uint32_t>(iy) * 668265263U + seed;
        h = (h ^ (h >> 13U)) * 1274126177U;
        h ^= h >> 16U;
        return static_cast<double>(h & 0xffffU) / 32767.5 - 1.0;
    };
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);
    const double sx = fx * fx * (3.0 - 2.0 * fx);
    const double sy = fy * fy * (3.0 - 2.0 * fy);
    const double a = hash(x0, y0) + (hash(x0 + 1, y0) - hash(x0, y0)) * sx;
    const double b =
        hash(x0, y0 + 1) + (hash(x0 + 1, y0 + 1) - hash(x0, y0 + 1)) * sx;
    return a + (b - a) * sy;
}

/// Three octaves of value noise; `x` and `y` are in base-feature cells.
double fractal_noise(double x, double y, std::uint32_t seed) {
    double sum = 0.0;
    double amplitude = 1.0;
    double norm = 0.0;
    for (int octave = 0; octave < 3; ++octave) {
        sum += amplitude * value_noise(x, y, seed + static_cast<std::uint32_t>(octave) * 7919U);
        norm += amplitude;
        x *= 2.0;
        y *= 2.0;
        amplitude *= 0.5;
    }
    return sum / norm;
}

/// One synthetic observation: the truth surface, the blurred-and-noisy input
/// the pass sees, and the true outlines for the error bands.
struct Scene {
    std::vector<double> truth;
    std::vector<float> input;
    std::vector<Rect> outlines;
};

Scene make_scene(int width, int height, double sigma, double noise_amplitude) {
    Scene scene;
    const std::vector<Building> buildings = make_buildings();
    for (const Building& building : buildings) {
        scene.outlines.push_back(building.rect);
    }
    scene.truth = truth_surface(buildings, width, height);
    const std::vector<double> blurred = gaussian_blur(scene.truth, width, height, sigma);
    Noise noise;
    scene.input.resize(blurred.size());
    for (std::size_t i = 0; i < blurred.size(); ++i) {
        scene.input[i] = static_cast<float>(blurred[i] + noise_amplitude * noise.normal());
    }
    return scene;
}

/// Mean absolute height error in the 3-cell band around the true outlines and
/// far from them (more than 6 cells).
struct BandErrors {
    double band = 0.0;
    double away = 0.0;
};

BandErrors measure_band_error(const std::vector<float>& values, const Scene& scene, int width,
                              int height) {
    BandErrors errors;
    double band_sum = 0.0;
    double away_sum = 0.0;
    std::size_t band_cells = 0;
    std::size_t away_cells = 0;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const double x = static_cast<double>(col) * kCell;
            const double y = static_cast<double>(row) * kCell;
            double distance = std::numeric_limits<double>::infinity();
            for (const Rect& rect : scene.outlines) {
                distance = std::min(distance, outline_distance(rect, x, y));
            }
            const std::size_t index = static_cast<std::size_t>(row) *
                                          static_cast<std::size_t>(width) +
                                      static_cast<std::size_t>(col);
            const double error =
                std::abs(static_cast<double>(values[index]) - scene.truth[index]);
            if (distance <= 3.0 * kCell) {
                band_sum += error;
                ++band_cells;
            } else if (distance > 6.0 * kCell) {
                away_sum += error;
                ++away_cells;
            }
        }
    }
    errors.band = band_cells == 0U ? 0.0 : band_sum / static_cast<double>(band_cells);
    errors.away = away_cells == 0U ? 0.0 : away_sum / static_cast<double>(away_cells);
    return errors;
}

/// All-valid mask.
std::vector<std::uint8_t> all_valid(std::size_t count) {
    return std::vector<std::uint8_t>(count, 1U);
}

// ---------------------------------------------------------------------------
// 1. The headline case: blurred, noisy roof outlines become sharp again.
// ---------------------------------------------------------------------------

TEST(StepEdges, RebuildsBuildingOutlines) {
    const int width = 160;
    const int height = 160;
    const Scene scene = make_scene(width, height, 1.5, 0.05);
    std::vector<float> values = scene.input;
    const std::vector<std::uint8_t> valid = all_valid(values.size());

    const BandErrors before = measure_band_error(values, scene, width, height);
    StepEdgeOptions options;
    options.min_height = 1.5;
    options.min_length = 3.0;
    const int segments = refine_step_edges(values, valid, width, height, kCell, options);
    const BandErrors after = measure_band_error(values, scene, width, height);

    // The bracketed lowercase tag surfaces this line from the CI test log.
    std::printf("[step edges] band-error %.4f -> %.4f m (%.1f%%), away %.4f -> %.4f m\n",
                before.band, after.band, 100.0 * (1.0 - after.band / before.band), before.away,
                after.away);
    EXPECT_GE(segments, 6);
    EXPECT_LT(after.band, 0.5 * before.band);
    EXPECT_LE(after.away, before.away * 1.01 + 1e-6);
}

// ---------------------------------------------------------------------------
// 2. A smooth hill has no discontinuity: nothing is accepted, nothing changes.
// ---------------------------------------------------------------------------

TEST(StepEdges, SmoothTerrainIsUntouched) {
    const int width = 120;
    const int height = 120;
    std::vector<float> values(static_cast<std::size_t>(width) *
                              static_cast<std::size_t>(height));
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const double x = static_cast<double>(col) * kCell;
            const double y = static_cast<double>(row) * kCell;
            const double r2 = (x - 15.0) * (x - 15.0) + (y - 15.0) * (y - 15.0);
            const double hill = 8.0 * std::exp(-r2 / (2.0 * 6.0 * 6.0));
            values[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(col)] = static_cast<float>(ground_height(x) + hill);
        }
    }
    const std::vector<float> original = values;
    const std::vector<std::uint8_t> valid = all_valid(values.size());
    StepEdgeOptions options;
    const int segments = refine_step_edges(values, valid, width, height, kCell, options);
    EXPECT_EQ(segments, 0);
    EXPECT_TRUE(values == original);
}

// ---------------------------------------------------------------------------
// 3. A step below step_min_height is evidence too weak to accept.
// ---------------------------------------------------------------------------

TEST(StepEdges, SubThresholdStepIsLeftAlone) {
    const int width = 80;
    const int height = 80;
    std::vector<float> values(static_cast<std::size_t>(width) *
                              static_cast<std::size_t>(height));
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const double x = static_cast<double>(col) * kCell;
            const double step = x >= 10.0 ? 1.2 : 0.0;
            values[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(col)] =
                static_cast<float>(ground_height(x) + step);
        }
    }
    const std::vector<float> original = values;
    const std::vector<std::uint8_t> valid = all_valid(values.size());
    StepEdgeOptions options;
    options.min_height = 1.5;
    const int segments = refine_step_edges(values, valid, width, height, kCell, options);
    EXPECT_EQ(segments, 0);
    EXPECT_TRUE(values == original);
}

// ---------------------------------------------------------------------------
// 4. An edge shorter than step_min_length is rejected even with a tall step.
// ---------------------------------------------------------------------------

TEST(StepEdges, ShortSegmentIsRejected) {
    const int width = 80;
    const int height = 80;
    std::vector<float> values(static_cast<std::size_t>(width) *
                              static_cast<std::size_t>(height));
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const double x = static_cast<double>(col) * kCell;
            const double y = static_cast<double>(row) * kCell;
            // A 2 m x 2 m roof: its edges are shorter than the 3 m default.
            const bool inside = std::abs(x - 10.0) <= 1.0 && std::abs(y - 10.0) <= 1.0;
            values[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(col)] =
                static_cast<float>(ground_height(x) + (inside ? 4.0 : 0.0));
        }
    }
    const std::vector<float> original = values;
    const std::vector<std::uint8_t> valid = all_valid(values.size());
    StepEdgeOptions options;
    options.min_height = 1.5;
    options.min_length = 3.0;
    const int segments = refine_step_edges(values, valid, width, height, kCell, options);
    EXPECT_EQ(segments, 0);
    EXPECT_TRUE(values == original);
}

// ---------------------------------------------------------------------------
// 5. Nodata (and, for BuildDem, hole-filled) cells are never evidence and are
//    never rewritten.
// ---------------------------------------------------------------------------

TEST(StepEdges, NodataCellsAreNeverTouched) {
    const int width = 100;
    const int height = 100;
    std::vector<float> values(static_cast<std::size_t>(width) *
                              static_cast<std::size_t>(height));
    std::vector<std::uint8_t> valid(values.size(), 1U);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const double x = static_cast<double>(col) * kCell;
            values[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(col)] =
                static_cast<float>(ground_height(x) + (x >= 12.5 ? 4.0 : 0.0));
        }
    }
    // A nodata block straddling the step, plus one isolated nodata cell away
    // from it: the pass must neither read nor write any of them.
    for (int row = 40; row <= 60; ++row) {
        for (int col = 48; col <= 52; ++col) {
            valid[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                  static_cast<std::size_t>(col)] = 0U;
        }
    }
    valid[10U * static_cast<std::size_t>(width) + 10U] = 0U;
    const std::vector<float> original = values;

    StepEdgeOptions options;
    const int segments = refine_step_edges(values, valid, width, height, kCell, options);
    EXPECT_GE(segments, 1);
    int changed_valid = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (valid[i] == 0U) {
            EXPECT_EQ(values[i], original[i]) << "cell " << i << " is nodata";
        } else if (values[i] != original[i]) {
            ++changed_valid;
        }
    }
    EXPECT_GT(changed_valid, 0) << "the pass must still refine the measured part of the step";
}

// ---------------------------------------------------------------------------
// 6. At a building corner two segments overlap: the snapped height must stay
//    inside the original band's range, never extrapolating past the roof or
//    below the ground.
// ---------------------------------------------------------------------------

TEST(StepEdges, CornersStayWithinTheBandRange) {
    const int width = 100;
    const int height = 100;
    Building flat;
    flat.rect = Rect{12.5, 12.5, 4.0, 3.0, 0.0};
    flat.flat_height = 5.0;
    const std::vector<double> truth =
        truth_surface(std::vector<Building>{flat}, width, height);
    const std::vector<double> blurred = gaussian_blur(truth, width, height, 1.5);
    Noise noise;
    std::vector<float> values(blurred.size());
    for (std::size_t i = 0; i < blurred.size(); ++i) {
        values[i] = static_cast<float>(blurred[i] + 0.05 * noise.normal());
    }
    const std::vector<float> original = values;
    const std::vector<std::uint8_t> valid = all_valid(values.size());
    StepEdgeOptions options;
    const int segments = refine_step_edges(values, valid, width, height, kCell, options);
    EXPECT_GT(segments, 0);

    int changed = 0;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) *
                                          static_cast<std::size_t>(width) +
                                      static_cast<std::size_t>(col);
            if (values[index] == original[index]) {
                continue;
            }
            ++changed;
            double lo = std::numeric_limits<double>::infinity();
            double hi = -lo;
            for (int dr = -5; dr <= 5; ++dr) {
                for (int dc = -5; dc <= 5; ++dc) {
                    const int nr = row + dr;
                    const int nc = col + dc;
                    if (nr < 0 || nc < 0 || nr >= height || nc >= width) {
                        continue;
                    }
                    const double value = static_cast<double>(
                        original[static_cast<std::size_t>(nr) * static_cast<std::size_t>(width) +
                                 static_cast<std::size_t>(nc)]);
                    lo = std::min(lo, value);
                    hi = std::max(hi, value);
                }
            }
            EXPECT_GE(static_cast<double>(values[index]), lo - 1e-3);
            EXPECT_LE(static_cast<double>(values[index]), hi + 1e-3);
        }
    }
    EXPECT_GT(changed, 0);
}

// ---------------------------------------------------------------------------
// 7. A wall edge that ends on flat ground must stop at its ends: no full-height
//    ridge is painted past the measured segment.
// ---------------------------------------------------------------------------

TEST(StepEdges, SegmentEndsAreNotExtended) {
    const int width = 160;
    const int height = 160;
    Building wall;
    wall.rect = Rect{14.0, 15.0, 1.5, 5.0, 0.0};  // 3 m x 10 m, ends at y = 10, 20
    wall.flat_height = 4.0;
    const std::vector<double> truth =
        truth_surface(std::vector<Building>{wall}, width, height);
    const std::vector<double> blurred = gaussian_blur(truth, width, height, 1.5);
    Noise noise;
    std::vector<float> values(blurred.size());
    for (std::size_t i = 0; i < blurred.size(); ++i) {
        values[i] = static_cast<float>(blurred[i] + 0.05 * noise.normal());
    }
    const std::vector<float> original = values;
    const std::vector<std::uint8_t> valid = all_valid(values.size());
    StepEdgeOptions options;
    // The 3 m end faces are shorter than this; only the 10 m sides qualify.
    options.min_length = 4.0;
    const int segments = refine_step_edges(values, valid, width, height, kCell, options);
    EXPECT_GE(segments, 1);

    int changed = 0;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) *
                                          static_cast<std::size_t>(width) +
                                      static_cast<std::size_t>(col);
            if (values[index] == original[index]) {
                continue;
            }
            ++changed;
            const double y = static_cast<double>(row) * kCell;
            EXPECT_GE(y, 10.0 - kCell - 1e-4) << "row " << row;
            EXPECT_LE(y, 20.0 + kCell + 1e-4) << "row " << row;
        }
    }
    EXPECT_GT(changed, 0);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 8. A natural cliff is wavy and rough, not a straight two-plane step. The
//    tightened gates must reject every segment and leave the surface alone.
// ---------------------------------------------------------------------------

TEST(StepEdges, NaturalCliffIsUntouched) {
    const int width = 160;
    const int height = 160;
    const std::size_t count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<double> rough(count, 0.0);
    double mean = 0.0;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const std::size_t index =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(col);
            rough[index] = fractal_noise(static_cast<double>(col) / 8.0,
                                         static_cast<double>(row) / 8.0, 17U);
            mean += rough[index];
        }
    }
    mean /= static_cast<double>(count);
    double variance = 0.0;
    for (const double value : rough) {
        variance += (value - mean) * (value - mean);
    }
    const double sigma = std::sqrt(variance / static_cast<double>(count));
    std::vector<double> surface(count, 0.0);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const std::size_t index =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(col);
            const double x = static_cast<double>(col) * kCell;
            const double y = static_cast<double>(row) * kCell;
            // The cliff line waves by 3 cells with a 12 m wavelength.
            const double cliff_x = 20.0 + 3.0 * kCell * std::sin(2.0 * kPi * y / 12.0);
            const double step = x < cliff_x ? 1.5 : 0.0;
            const double noise = 0.30 * (rough[index] - mean) / sigma;  // sigma 0.3 m
            surface[index] = ground_height(x) + step + noise;
        }
    }
    const std::vector<double> blurred = gaussian_blur(surface, width, height, 1.0);
    std::vector<float> values(blurred.begin(), blurred.end());
    const std::vector<float> original = values;
    const std::vector<std::uint8_t> valid = all_valid(values.size());
    StepEdgeOptions options;
    options.min_height = 1.0;  // the cliff step is 1.5 m
    const int segments = refine_step_edges(values, valid, width, height, kCell, options);
    EXPECT_EQ(segments, 0);
    EXPECT_TRUE(values == original);
}

// ---------------------------------------------------------------------------
// 9. A barrel-vault roof meets the ground with a curved, vertical-tangent
//    side. Its upper "plane" is not a plane, so the rms gate rejects it.
// ---------------------------------------------------------------------------

TEST(StepEdges, CurvedRoofEdgeIsRejectedByRms) {
    const int width = 200;
    const int height = 120;
    std::vector<double> surface(static_cast<std::size_t>(width) *
                                static_cast<std::size_t>(height), 0.0);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const double x = static_cast<double>(col) * kCell;
            double value = ground_height(x);
            // A 10 m radius semicircular vault spanning x = 20..40 m.
            if (x >= 20.0 && x <= 40.0) {
                const double u = x - 30.0;
                value += std::sqrt(std::max(0.0, 100.0 - u * u));
            }
            surface[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                    static_cast<std::size_t>(col)] = value;
        }
    }
    const std::vector<double> blurred = gaussian_blur(surface, width, height, 1.5);
    std::vector<float> values(blurred.begin(), blurred.end());
    const std::vector<float> original = values;
    const std::vector<std::uint8_t> valid = all_valid(values.size());
    StepEdgeOptions options;
    options.min_height = 1.5;
    // A wide fit band sees the curvature the default 5-cell band misses.
    options.band_half_width = 40;
    const int segments = refine_step_edges(values, valid, width, height, kCell, options);
    EXPECT_EQ(segments, 0);
    EXPECT_TRUE(values == original);
}

// 10. BuildDem integration: the parameter is opt-in, and when enabled it only
//    rewrites the band around a detected step.
// ---------------------------------------------------------------------------

TEST(StepEdges, BuildDemRefineEdgesIsOptIn) {
    // mode 0 = parameter absent, 1 = explicit false, 2 = true.
    const auto build = [](int mode) {
        Document document;
        Chunk& chunk = document.workspace().add_chunk("Chunk 1");
        document.set_active_chunk_id(chunk.id());
        PointCloud& cloud = chunk.point_cloud();
        for (double x = -10.0; x <= 10.0 + 1e-9; x += 0.1) {
            for (double y = -10.0; y <= 10.0 + 1e-9; y += 0.1) {
                // A 1 m wide ramp: the point cloud itself is not a step, so the
                // refined DEM differs from the unrefined one.
                const double t = std::min(std::max((x + 0.5) / 1.0, 0.0), 1.0);
                cloud.points.push_back(Eigen::Vector3d(x, y, 4.0 * t));
            }
        }
        Params params = make_params(builtin_registry().require("BuildDem"));
        params.set("source_data", ParamValue{std::string("point_cloud")});
        params.set("interpolation", ParamValue{std::string("disabled")});
        params.set("resolution", ParamValue{0.25});
        params.set("fill_method", ParamValue{std::string("nearest")});
        params.set("smoothing", ParamValue{std::string("none")});
        if (mode != 0) {
            params.set("refine_edges", ParamValue{mode == 2});
        }
        run_task(builtin_registry(), document, "BuildDem", params);
        return document.active_chunk()->elevation();
    };

    const Elevation absent = build(0);
    const Elevation disabled = build(1);
    const Elevation enabled = build(2);
    ASSERT_FALSE(absent.empty());
    ASSERT_FALSE(enabled.empty());
    // Off by default is exactly the historical path.
    EXPECT_TRUE(absent.raster.data == disabled.raster.data);
    ASSERT_EQ(enabled.raster.grid.width, disabled.raster.grid.width);
    ASSERT_EQ(enabled.raster.grid.height, disabled.raster.grid.height);

    const RasterGrid& grid = disabled.raster.grid;
    int changed = 0;
    for (int row = 0; row < grid.height; ++row) {
        for (int col = 0; col < grid.width; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) *
                                          static_cast<std::size_t>(grid.width) +
                                      static_cast<std::size_t>(col);
            if (enabled.raster.data[index] == disabled.raster.data[index]) {
                continue;
            }
            ++changed;
            // Only the band around x = 0 may change (6 cells is generous: the
            // snap band is 4 cells on each side).
            EXPECT_LE(std::abs(grid.pixel_center(col, row).x()), 6.0 * kCell);
        }
    }
    EXPECT_GT(changed, 0);
}

// ---------------------------------------------------------------------------
// 11. A 45-degree step across a 2000x2000 DEM: the walk must stay linear in
//     the number of cells, not quadratic in a component's bounding box.
// ---------------------------------------------------------------------------

TEST(StepEdges, LargeGridIsFast) {
    const int width = 2000;
    const int height = 2000;
    std::vector<float> values(static_cast<std::size_t>(width) *
                              static_cast<std::size_t>(height));
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            values[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(col)] = col >= row ? 5.0F : 0.0F;
        }
    }
    const std::vector<std::uint8_t> valid = all_valid(values.size());
    StepEdgeOptions options;
    const auto start = std::chrono::steady_clock::now();
    const int segments = refine_step_edges(values, valid, width, height, kCell, options);
    const auto stop = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();
    std::printf("[step edges] 2000x2000 45-degree step: %d segment(s) in %.3f s\n", segments,
                seconds);
    EXPECT_GE(segments, 1);
    if (test::perf_asserts_enabled()) {
        EXPECT_LT(seconds, 2.0);
    }
}

}  // namespace
}  // namespace alphashape
