#include "common/algorithm/stereo/depth_pyramid.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "base/utils/error.hpp"

namespace alphashape::stereo {
namespace {

/// splitmix64, the same mixing depth_map_fingerprint and the PatchMatch stream
/// use. Duplicated rather than shared: this file's only use is to derive a
/// level's stream key, and a test asserts the key of level 0 IS the seed.
inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

std::size_t si(int v) { return static_cast<std::size_t>(v); }

/// Levels a search can run at before the geometry runs out: the coarsest level
/// must still have an interior pixel, i.e. more than 2 * radius on both sides.
int geometry_level_cap(int width, int height, int window_radius) {
    int short_side = std::min(width, height);
    int cap = 1;
    while (short_side >= 2 && short_side > 2 * window_radius) {
        // A further halving is only possible (and useful) if the halved level
        // still has an interior.
        const int halved = short_side / 2;
        if (halved <= 2 * window_radius) {
            break;
        }
        short_side = halved;
        ++cap;
    }
    return cap;
}

}  // namespace

int pyramid_level_count(int width, int height, int levels, int min_size, int window_radius) {
    if (levels < 0) {
        throw ValueError("stereo::pyramid_level_count: levels must be non-negative");
    }
    if (window_radius < 0) {
        throw ValueError("stereo::pyramid_level_count: window_radius must be non-negative");
    }
    // An explicit single level does not consult min_size at all: there is no
    // halving to stop, so a caller asking for one level is not required to
    // supply (or even define) the floor an automatic pyramid needs. Checked
    // before min_size so `levels == 1, min_size == 0` is a valid request, which
    // is exactly what run_patch_match's own validation allows.
    if (levels == 1) {
        return 1;  // explicit single level: the pre-pyramid behaviour
    }
    if (min_size <= 0) {
        throw ValueError("stereo::pyramid_level_count: min_size must be positive");
    }
    if (width <= 0 || height <= 0) {
        return 1;
    }

    const int cap = geometry_level_cap(width, height, window_radius);

    int wanted = 1;
    if (levels == 0) {
        // Automatic: halve while the SHORTER side stays at or above min_size, so
        // the coarsest level is the smallest one that is still about min_size.
        int short_side = std::min(width, height);
        while (short_side / 2 >= min_size) {
            short_side /= 2;
            ++wanted;
            if (wanted >= cap) {
                break;
            }
        }
    } else {
        wanted = levels;
    }
    return std::clamp(wanted, 1, cap);
}

std::vector<int> pyramid_iterations(int level_count, int iterations) {
    if (level_count < 1) {
        level_count = 1;
    }
    if (iterations < 0) {
        iterations = 0;
    }
    // A zero-iteration search is a single random initialisation: there is
    // nothing to distribute, and splitting it across levels would change what
    // "no sweeps" means, so it stays at the reference resolution.
    if (iterations == 0) {
        return {0};
    }
    // Every level needs at least kMinIterationsPerLevel sweeps to do anything
    // with its inherited planes; if the budget cannot pay for that, use fewer
    // levels rather than starving them.
    int levels = std::min(level_count, std::max(1, iterations / kMinIterationsPerLevel));
    levels = std::clamp(levels, 1, 32);

    // Finest first while allocating, so the 40% share of the remaining budget
    // that the rule names is applied to the budget that is actually left.
    std::vector<int> from_finest;
    from_finest.reserve(si(levels));
    int budget = iterations;
    int remaining_levels = levels;
    while (remaining_levels > 1) {
        int share = std::max(kMinIterationsPerLevel, (2 * budget) / 5);  // floor(40%)
        // Leave enough for every coarser level.
        share = std::min(share, budget - kMinIterationsPerLevel * (remaining_levels - 1));
        share = std::max(share, kMinIterationsPerLevel);
        from_finest.push_back(share);
        budget -= share;
        --remaining_levels;
    }
    from_finest.push_back(budget);  // the coarsest level takes the rest

    std::reverse(from_finest.begin(), from_finest.end());
    return from_finest;  // coarsest first
}

std::uint64_t pyramid_level_seed(std::uint64_t seed, int level_from_finest) {
    if (level_from_finest <= 0) {
        return seed;
    }
    std::uint64_t h = splitmix64(seed);
    // The literal is narrowed to std::uint64_t explicitly: on Linux that is
    // `unsigned long` and the suffix makes the literal `unsigned long long`, so
    // the product would otherwise widen and trip -Wsign-conversion.
    const std::uint64_t level_mix = static_cast<std::uint64_t>(level_from_finest) *
                                    static_cast<std::uint64_t>(0xC2B2AE3D27D4EB4FULL);
    h = splitmix64(h ^ level_mix);
    return splitmix64(h);
}

FloatImage pyramid_downsample(const FloatImage& src) {
    if (src.width() < 2 || src.height() < 2) {
        throw ValueError("stereo::pyramid_downsample: image is smaller than one 2x2 block");
    }
    const int out_w = src.width() / 2;
    const int out_h = src.height() / 2;
    FloatImage out(out_w, out_h);
    for (int oy = 0; oy < out_h; ++oy) {
        for (int ox = 0; ox < out_w; ++ox) {
            const double sum = static_cast<double>(src.at(2 * ox, 2 * oy)) +
                               static_cast<double>(src.at(2 * ox + 1, 2 * oy)) +
                               static_cast<double>(src.at(2 * ox, 2 * oy + 1)) +
                               static_cast<double>(src.at(2 * ox + 1, 2 * oy + 1));
            out.at(ox, oy) = static_cast<float>(sum * 0.25);
        }
    }
    return out;
}

DepthPyramid build_depth_pyramid(const FloatImage& reference, const Calibration& calibration,
                                 int level_count) {
    if (level_count < 1) {
        throw ValueError("stereo::build_depth_pyramid: level_count must be at least 1");
    }
    if (reference.width() != calibration.width || reference.height() != calibration.height) {
        throw ValueError(
            "stereo::build_depth_pyramid: calibration does not match the reference image");
    }
    DepthPyramid pyramid;
    pyramid.reference = &reference;
    // Coarsest first, so image(i) and calibrations[i] are indexed the same way.
    std::vector<FloatImage> from_finest;
    std::vector<Calibration> calibrations_from_finest;
    from_finest.reserve(si(level_count - 1));
    calibrations_from_finest.reserve(si(level_count - 1));
    calibrations_from_finest.push_back(calibration);
    const FloatImage* previous = &reference;
    Calibration previous_calibration = calibration;
    for (int level = 1; level < level_count; ++level) {
        // Fold level by level rather than by scaled_calibration(full, 2^level):
        // the halved sizes truncate, and each step must describe the image the
        // search actually sees (an odd width halves to floor(W/2), then again).
        from_finest.push_back(pyramid_downsample(*previous));
        previous_calibration = scaled_calibration(previous_calibration, 2);
        calibrations_from_finest.push_back(previous_calibration);
        previous = &from_finest.back();
        if (previous->width() != previous_calibration.width ||
            previous->height() != previous_calibration.height) {
            throw ValueError(
                "stereo::build_depth_pyramid: a reduced level disagrees with its calibration");
        }
    }
    pyramid.reduced.assign(from_finest.rbegin(), from_finest.rend());
    pyramid.calibrations.assign(calibrations_from_finest.rbegin(),
                                calibrations_from_finest.rend());
    return pyramid;
}

DepthHypotheses pyramid_upsample(const DepthHypotheses& coarse, int width, int height) {
    if (width <= 0 || height <= 0) {
        throw ValueError("stereo::pyramid_upsample: target size must be positive");
    }
    if (coarse.width <= 0 || coarse.height <= 0 ||
        coarse.depth.size() < si(coarse.width) * si(coarse.height) ||
        coarse.normals.size() < 3 * si(coarse.width) * si(coarse.height)) {
        throw ValueError("stereo::pyramid_upsample: coarse map is empty or malformed");
    }

    DepthHypotheses fine;
    fine.width = width;
    fine.height = height;
    fine.depth.assign(si(width) * si(height), -1.0F);
    fine.normals.assign(3 * si(width) * si(height), 0.0F);

    const int cw = coarse.width;
    const int ch = coarse.height;
    for (int y = 0; y < height; ++y) {
        // Fine pixel y sits at coarse coordinate y/2 - 0.25: with the shared
        // bearing identity of scaled_calibration(), coarse pixel j is the
        // bearing of fine 2j + 0.5, so fine y samples coarse y' = (y - 0.5)/2.
        const float fy = 0.5F * static_cast<float>(y) - 0.25F;
        const int y0 = static_cast<int>(std::floor(fy));
        const float ay = fy - static_cast<float>(y0);
        const int ya = std::clamp(y0, 0, ch - 1);
        const int yb = std::clamp(y0 + 1, 0, ch - 1);
        for (int x = 0; x < width; ++x) {
            const float fx = 0.5F * static_cast<float>(x) - 0.25F;
            const int x0 = static_cast<int>(std::floor(fx));
            const float ax = fx - static_cast<float>(x0);
            const int xa = std::clamp(x0, 0, cw - 1);
            const int xb = std::clamp(x0 + 1, 0, cw - 1);

            const std::size_t taps[4] = {
                si(ya) * si(cw) + si(xa), si(ya) * si(cw) + si(xb),
                si(yb) * si(cw) + si(xa), si(yb) * si(cw) + si(xb)};
            // Braced initialiser: the clauses are sequenced left to right, so
            // the weight list has no compiler-chosen order issue.
            const float weights[4] = {(1.0F - ax) * (1.0F - ay), ax * (1.0F - ay),
                                      (1.0F - ax) * ay, ax * ay};
            double weight_sum = 0.0;
            double depth_sum = 0.0;
            double nx = 0.0, ny = 0.0, nz = 0.0;
            for (int t = 0; t < 4; ++t) {
                const std::size_t q = taps[t];
                const float d = coarse.depth[q];
                if (!(d > 0.0F) || weights[t] <= 0.0F) {
                    continue;  // a hole carries no weight
                }
                const double w = static_cast<double>(weights[t]);
                weight_sum += w;
                depth_sum += w * static_cast<double>(d);
                nx += w * static_cast<double>(coarse.normals[3 * q + 0]);
                ny += w * static_cast<double>(coarse.normals[3 * q + 1]);
                nz += w * static_cast<double>(coarse.normals[3 * q + 2]);
            }
            if (!(weight_sum > 0.0)) {
                continue;  // every tap is a hole: this cell is a hole too
            }
            const std::size_t p = si(y) * si(width) + si(x);
            fine.depth[p] = static_cast<float>(depth_sum / weight_sum);
            const double norm = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (norm > 1e-9) {
                fine.normals[3 * p + 0] = static_cast<float>(nx / norm);
                fine.normals[3 * p + 1] = static_cast<float>(ny / norm);
                fine.normals[3 * p + 2] = static_cast<float>(nz / norm);
            }
        }
    }
    return fine;
}

double pyramid_search_span(const std::vector<float>& depth, int width, int height, int radius,
                           double floor_value) {
    if (!(floor_value > 0.0)) {
        throw ValueError("stereo::pyramid_search_span: floor_value must be positive");
    }
    if (width <= 0 || height <= 0) {
        return floor_value;
    }
    if (depth.size() < si(width) * si(height)) {
        throw ValueError("stereo::pyramid_search_span: depth map is smaller than the image");
    }
    const int r = std::max(radius, 0);
    const int x0 = r;
    const int x1 = width - 1 - r;
    const int y0 = r;
    const int y1 = height - 1 - r;
    if (x0 > x1 || y0 > y1) {
        return floor_value;  // no interior to measure
    }

    // A full window scan over a 4 MP map at radius 5 is half a billion reads,
    // which would cost more than the sweeps it is meant to save. Sample a
    // strided grid instead: the median of a systematic sample is as stable as
    // the full median for a smooth map, and the stride keeps the work bounded
    // and, being a pure function of the geometry, deterministic.
    const std::size_t interior = si(x1 - x0 + 1) * si(y1 - y0 + 1);
    // Sample at most ~200k windows: the median of a systematic sample is as
    // stable as the full median for a smooth map, and the cap keeps the cost
    // below the sweeps it exists to save. `stride` is an int (it is used as a
    // step on int coordinates), so the division is done in std::size_t.
    constexpr std::size_t kMaxWindows = 200000;
    int stride = 1;
    while (interior / (si(stride) * si(stride)) > kMaxWindows) {
        ++stride;
    }

    std::vector<double> spreads;
    spreads.reserve(interior / (si(stride) * si(stride)) + si(1));
    for (int y = y0; y <= y1; y += stride) {
        for (int x = x0; x <= x1; x += stride) {
            if (!(depth[si(y) * si(width) + si(x)] > 0.0F)) {
                continue;
            }
            double low = std::numeric_limits<double>::infinity();
            double high = 0.0;
            int seen = 0;
            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    const float d = depth[si(y + dy) * si(width) + si(x + dx)];
                    if (!(d > 0.0F)) {
                        continue;
                    }
                    const double inv = 1.0 / static_cast<double>(d);
                    low = std::min(low, inv);
                    high = std::max(high, inv);
                    ++seen;
                }
            }
            if (seen >= 2 && high > low) {
                spreads.push_back(high - low);
            }
        }
    }
    if (spreads.empty()) {
        return floor_value;  // one depth everywhere, or none: nothing to measure
    }
    const std::size_t mid = spreads.size() / 2;
    std::nth_element(spreads.begin(), spreads.begin() + static_cast<std::ptrdiff_t>(mid),
                     spreads.end());
    const double median = spreads[mid];
    return std::clamp(median, floor_value, 64.0 * floor_value);
}

}  // namespace alphashape::stereo
