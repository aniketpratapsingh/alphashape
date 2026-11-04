#include "common/algorithm/ortho/texture_fill.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "base/utils/error.hpp"

namespace alphashape::ortho {
namespace {

/// One pyramid level of one component's work window. `rgb` is three bytes per
/// cell, `hole` marks the cells this component must complete, and `valid` marks
/// cells that may appear in a source patch (known and not part of any hole).
struct Level {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;
    std::vector<std::uint8_t> hole;
    std::vector<std::uint8_t> valid;
};

std::size_t cell_index(int width, int x, int y) {
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x);
}

std::uint8_t clamp_byte(double value) {
    const double rounded = std::floor(value + 0.5);
    if (rounded <= 0.0) {
        return 0;
    }
    if (rounded >= 255.0) {
        return 255;
    }
    return static_cast<std::uint8_t>(rounded);
}

/// Copy the component's window out of the full raster. `window_hole` is the
/// window-sized membership mask of this component; `fill_mask` is the full
/// raster's mask, used to keep every other hole out of the source set too.
Level make_window_level(int source_width, const std::vector<std::uint8_t>& rgba,
                        const std::vector<std::uint8_t>& fill_mask,
                        const std::vector<std::uint8_t>& window_hole, int wx0, int wy0, int ww,
                        int wh) {
    Level level;
    level.width = ww;
    level.height = wh;
    const std::size_t cells = static_cast<std::size_t>(ww) * static_cast<std::size_t>(wh);
    level.rgb.assign(cells * 3U, 0);
    level.hole = window_hole;
    level.valid.assign(cells, 0);
    for (int y = 0; y < wh; ++y) {
        for (int x = 0; x < ww; ++x) {
            const std::size_t local = cell_index(ww, x, y);
            const std::size_t global =
                cell_index(source_width, wx0 + x, wy0 + y);
            for (int c = 0; c < 3; ++c) {
                level.rgb[local * 3U + static_cast<std::size_t>(c)] =
                    rgba[global * 4U + static_cast<std::size_t>(c)];
            }
            level.valid[local] =
                rgba[global * 4U + 3U] != 0 && fill_mask[global] == 0 ? 1 : 0;
        }
    }
    return level;
}

/// Box-downsample by two. A coarse cell is a hole when any child is (OR), and
/// a usable source only when every child is (AND), so source patches never
/// straddle a hole or a nodata cell.
Level downsample(const Level& fine) {
    Level coarse;
    coarse.width = (fine.width + 1) / 2;
    coarse.height = (fine.height + 1) / 2;
    const std::size_t cells =
        static_cast<std::size_t>(coarse.width) * static_cast<std::size_t>(coarse.height);
    coarse.rgb.assign(cells * 3U, 0);
    coarse.hole.assign(cells, 0);
    coarse.valid.assign(cells, 0);
    for (int y = 0; y < coarse.height; ++y) {
        for (int x = 0; x < coarse.width; ++x) {
            const std::size_t dst = cell_index(coarse.width, x, y);
            int sums[3] = {0, 0, 0};
            int count = 0;
            bool any_hole = false;
            bool all_valid = true;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const int sx = x * 2 + dx;
                    const int sy = y * 2 + dy;
                    if (sx >= fine.width || sy >= fine.height) {
                        all_valid = false;
                        continue;
                    }
                    const std::size_t src = cell_index(fine.width, sx, sy);
                    for (int c = 0; c < 3; ++c) {
                        sums[c] += fine.rgb[src * 3U + static_cast<std::size_t>(c)];
                    }
                    ++count;
                    any_hole = any_hole || fine.hole[src] != 0;
                    all_valid = all_valid && fine.valid[src] != 0;
                }
            }
            for (int c = 0; c < 3; ++c) {
                coarse.rgb[dst * 3U + static_cast<std::size_t>(c)] =
                    count > 0 ? static_cast<std::uint8_t>(
                                    (sums[c] + count / 2) / count)
                              : 0;
            }
            coarse.hole[dst] = any_hole ? 1 : 0;
            coarse.valid[dst] = all_valid ? 1 : 0;
        }
    }
    return coarse;
}

/// Finest-first pyramid. Deeper levels are only built while both sides are at
/// least four patches wide, below which a coarse solve has too few distinct
/// source positions to be meaningful.
std::vector<Level> build_pyramid(Level finest, const TextureFillOptions& options) {
    std::vector<Level> levels;
    levels.push_back(std::move(finest));
    const int wanted = std::max(1, options.pyramid_levels);
    const int patch = 2 * options.patch_radius + 1;
    const int minimum = 4 * patch;
    while (static_cast<int>(levels.size()) < wanted) {
        Level next = downsample(levels.back());
        if (next.width < minimum || next.height < minimum) {
            break;
        }
        levels.push_back(std::move(next));
    }
    return levels;
}

/// Seed the finer level's holes from the completed coarse level. Valid cells
/// keep their original photograph; only the cells we are inventing move.
void upsample_holes(const Level& coarse, Level& fine) {
    for (int y = 0; y < fine.height; ++y) {
        for (int x = 0; x < fine.width; ++x) {
            const std::size_t dst = cell_index(fine.width, x, y);
            if (fine.hole[dst] == 0) {
                continue;
            }
            const std::size_t src =
                cell_index(coarse.width, std::min(x / 2, coarse.width - 1),
                           std::min(y / 2, coarse.height - 1));
            for (int c = 0; c < 3; ++c) {
                fine.rgb[dst * 3U + static_cast<std::size_t>(c)] =
                    coarse.rgb[src * 3U + static_cast<std::size_t>(c)];
            }
        }
    }
}

/// Flood-fill one 4-connected component of `mask` from `seed_cell`, marking
/// `visited`. Returns the cells in discovery order (irrelevant to the result;
/// the component's raster-first cell is the caller's seed) and its inclusive
/// bounding box.
std::vector<std::size_t> collect_component(int width, int height,
                                           const std::vector<std::uint8_t>& mask,
                                           std::vector<std::uint8_t>& visited,
                                           std::size_t seed_cell, int& min_x, int& max_x,
                                           int& min_y, int& max_y) {
    std::vector<std::size_t> component;
    std::vector<std::size_t> stack;
    stack.push_back(seed_cell);
    visited[seed_cell] = 1;
    const int seed_x = static_cast<int>(seed_cell % static_cast<std::size_t>(width));
    const int seed_y = static_cast<int>(seed_cell / static_cast<std::size_t>(width));
    min_x = max_x = seed_x;
    min_y = max_y = seed_y;
    while (!stack.empty()) {
        const std::size_t cell = stack.back();
        stack.pop_back();
        component.push_back(cell);
        const int x = static_cast<int>(cell % static_cast<std::size_t>(width));
        const int y = static_cast<int>(cell / static_cast<std::size_t>(width));
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
        const int nx[4] = {x - 1, x + 1, x, x};
        const int ny[4] = {y, y, y - 1, y + 1};
        for (int k = 0; k < 4; ++k) {
            if (nx[k] < 0 || nx[k] >= width || ny[k] < 0 || ny[k] >= height) {
                continue;
            }
            const std::size_t neighbour = cell_index(width, nx[k], ny[k]);
            if (mask[neighbour] != 0 && visited[neighbour] == 0) {
                visited[neighbour] = 1;
                stack.push_back(neighbour);
            }
        }
    }
    return component;
}

/// Sum of squared RGB differences between a target patch and a source patch.
/// On the first pass only the target's known pixels are compared, so the initial
/// smooth values cannot steer the match; later passes compare every target
/// pixel, which ties neighbouring completed regions together.
double patch_cost(const Level& level, int target_x, int target_y, int source_x, int source_y,
                  int radius, bool known_only) {
    double sum = 0.0;
    bool compared = false;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int ty = target_y + dy;
        if (ty < 0 || ty >= level.height) {
            continue;
        }
        for (int dx = -radius; dx <= radius; ++dx) {
            const int tx = target_x + dx;
            if (tx < 0 || tx >= level.width) {
                continue;
            }
            const std::size_t target = cell_index(level.width, tx, ty);
            if (known_only && level.valid[target] == 0) {
                continue;
            }
            const std::size_t source =
                cell_index(level.width, source_x + dx, source_y + dy);
            for (int c = 0; c < 3; ++c) {
                const double delta =
                    static_cast<double>(level.rgb[target * 3U + static_cast<std::size_t>(c)]) -
                    static_cast<double>(level.rgb[source * 3U + static_cast<std::size_t>(c)]);
                sum += delta * delta;
            }
            compared = true;
        }
    }
    return compared ? sum : 0.0;
}

/// All patch centres whose whole patch lies on usable source pixels. The
/// source patch must be fully known, so a candidate is rejected as soon as one
/// cell is a hole or nodata.
std::vector<std::size_t> source_centres(const Level& level, int radius) {
    std::vector<std::size_t> centres;
    for (int y = radius; y < level.height - radius; ++y) {
        for (int x = radius; x < level.width - radius; ++x) {
            bool usable = true;
            for (int dy = -radius; dy <= radius && usable; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (level.valid[cell_index(level.width, x + dx, y + dy)] == 0) {
                        usable = false;
                        break;
                    }
                }
            }
            if (usable) {
                centres.push_back(cell_index(level.width, x, y));
            }
        }
    }
    return centres;
}

/// PatchMatch over one level: alternate the scan direction so each sweep can
/// propagate a good match from both sides, then refine the current best with a
/// shrinking uniform random search.
void refine_nnf(Level& level, std::vector<std::int32_t>& nnf,
                const std::vector<std::uint8_t>& source_mask,
                const std::vector<std::size_t>& sources,
                const std::vector<std::size_t>& targets, int radius, int sweeps,
                bool known_only, std::mt19937& rng) {
    const std::size_t cells =
        static_cast<std::size_t>(level.width) * static_cast<std::size_t>(level.height);
    for (std::size_t t = 0; t < targets.size(); ++t) {
        const std::size_t cell = targets[t];
        if (nnf[cell] < 0 || nnf[cell] >= static_cast<std::int32_t>(cells) ||
            source_mask[static_cast<std::size_t>(nnf[cell])] == 0) {
            const std::size_t pick = static_cast<std::size_t>(rng()) % sources.size();
            nnf[cell] = static_cast<std::int32_t>(sources[pick]);
        }
    }
    const int max_step = std::max(level.width, level.height);
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        const bool forward = (sweep % 2) == 0;
        for (std::size_t k = 0; k < targets.size(); ++k) {
            const std::size_t index = forward ? k : targets.size() - 1 - k;
            const std::size_t cell = targets[index];
            const int tx = static_cast<int>(cell % static_cast<std::size_t>(level.width));
            const int ty = static_cast<int>(cell / static_cast<std::size_t>(level.width));
            int best_x = static_cast<int>(nnf[cell] % level.width);
            int best_y = static_cast<int>(nnf[cell] / level.width);
            double best_cost = patch_cost(level, tx, ty, best_x, best_y, radius, known_only);
            const int step_x[2] = {forward ? -1 : 1, 0};
            const int step_y[2] = {0, forward ? -1 : 1};
            for (int n = 0; n < 2; ++n) {
                const int nx = tx + step_x[n];
                const int ny = ty + step_y[n];
                if (nx < 0 || nx >= level.width || ny < 0 || ny >= level.height) {
                    continue;
                }
                const std::int32_t neighbour = nnf[cell_index(level.width, nx, ny)];
                if (neighbour < 0) {
                    continue;
                }
                const int sx = static_cast<int>(neighbour % level.width) - step_x[n];
                const int sy = static_cast<int>(neighbour / level.width) - step_y[n];
                if (sx < radius || sx >= level.width - radius || sy < radius ||
                    sy >= level.height - radius ||
                    source_mask[cell_index(level.width, sx, sy)] == 0) {
                    continue;
                }
                const double cost = patch_cost(level, tx, ty, sx, sy, radius, known_only);
                if (cost < best_cost) {
                    best_cost = cost;
                    best_x = sx;
                    best_y = sy;
                }
            }
            for (int step = max_step; step >= 1; step /= 2) {
                const std::uint32_t span = static_cast<std::uint32_t>(2 * step + 1);
                const int offset_x = static_cast<int>(rng() % span) - step;
                const int offset_y = static_cast<int>(rng() % span) - step;
                const int sx = std::clamp(best_x + offset_x, radius, level.width - 1 - radius);
                const int sy = std::clamp(best_y + offset_y, radius, level.height - 1 - radius);
                if (source_mask[cell_index(level.width, sx, sy)] == 0) {
                    continue;
                }
                const double cost = patch_cost(level, tx, ty, sx, sy, radius, known_only);
                if (cost < best_cost) {
                    best_cost = cost;
                    best_x = sx;
                    best_y = sy;
                }
            }
            nnf[cell] = static_cast<std::int32_t>(cell_index(level.width, best_x, best_y));
        }
    }
}

/// Carry a coarse nearest-neighbour field to the next finer level: a match at
/// (2c, 2c) stays put, and the finer EM pass refines it from there.
std::vector<std::int32_t> upsample_nnf(const Level& coarse,
                                       const std::vector<std::int32_t>& coarse_nnf,
                                       const Level& fine, int radius) {
    std::vector<std::int32_t> nnf(
        static_cast<std::size_t>(fine.width) * static_cast<std::size_t>(fine.height), -1);
    for (int y = 0; y < fine.height; ++y) {
        for (int x = 0; x < fine.width; ++x) {
            const std::size_t coarse_cell = cell_index(
                coarse.width, std::min(x / 2, coarse.width - 1),
                std::min(y / 2, coarse.height - 1));
            const std::int32_t source = coarse_nnf[coarse_cell];
            if (source < 0) {
                continue;
            }
            const int sx = std::min(
                static_cast<int>(source % coarse.width) * 2, fine.width - 1 - radius);
            const int sy = std::min(
                static_cast<int>(source / coarse.width) * 2, fine.height - 1 - radius);
            nnf[cell_index(fine.width, x, y)] =
                static_cast<std::int32_t>(cell_index(fine.width, sx, sy));
        }
    }
    return nnf;
}

/// Every hole pixel becomes the mean of the source pixels that the overlapping
/// patches (each at its own best match) place on it. Overlap is what makes the
/// seams between neighbouring patches disappear.
void vote(Level& level, const std::vector<std::int32_t>& nnf,
          const std::vector<std::size_t>& targets, int radius) {
    const std::size_t cells =
        static_cast<std::size_t>(level.width) * static_cast<std::size_t>(level.height);
    std::vector<double> sums(cells * 3U, 0.0);
    std::vector<std::uint32_t> counts(cells, 0);
    for (const std::size_t cell : targets) {
        const std::size_t source = static_cast<std::size_t>(nnf[cell]);
        const int sx = static_cast<int>(source % static_cast<std::size_t>(level.width));
        const int sy = static_cast<int>(source / static_cast<std::size_t>(level.width));
        const int tx = static_cast<int>(cell % static_cast<std::size_t>(level.width));
        const int ty = static_cast<int>(cell / static_cast<std::size_t>(level.width));
        for (int dy = -radius; dy <= radius; ++dy) {
            const int y = ty + dy;
            if (y < 0 || y >= level.height) {
                continue;
            }
            for (int dx = -radius; dx <= radius; ++dx) {
                const int x = tx + dx;
                if (x < 0 || x >= level.width) {
                    continue;
                }
                const std::size_t target = cell_index(level.width, x, y);
                if (level.hole[target] == 0) {
                    continue;
                }
                const std::size_t from = cell_index(level.width, sx + dx, sy + dy);
                for (int c = 0; c < 3; ++c) {
                    sums[target * 3U + static_cast<std::size_t>(c)] +=
                        static_cast<double>(level.rgb[from * 3U + static_cast<std::size_t>(c)]);
                }
                ++counts[target];
            }
        }
    }
    for (std::size_t i = 0; i < cells; ++i) {
        if (counts[i] == 0) {
            continue;
        }
        for (int c = 0; c < 3; ++c) {
            level.rgb[i * 3U + static_cast<std::size_t>(c)] = clamp_byte(
                sums[i * 3U + static_cast<std::size_t>(c)] /
                static_cast<double>(counts[i]));
        }
    }
}

/// Run the search/vote rounds on one pyramid level and return its nearest-
/// neighbour field. An empty result means the level was skipped: it had no
/// full source patch, or no hole cell to complete.
std::vector<std::int32_t> process_level(Level& level, const Level* coarser,
                                        const std::vector<std::int32_t>& carried, int radius,
                                        int em_iterations, int patchmatch_iterations,
                                        bool& first_pass, std::mt19937& rng) {
    const std::size_t cells =
        static_cast<std::size_t>(level.width) * static_cast<std::size_t>(level.height);
    const std::vector<std::size_t> sources = source_centres(level, radius);
    if (sources.empty()) {
        return {};
    }
    std::vector<std::uint8_t> source_mask(cells, 0);
    for (const std::size_t source : sources) {
        source_mask[source] = 1;
    }
    std::vector<std::size_t> targets;
    targets.reserve(cells / 4U + 1U);
    for (std::size_t i = 0; i < cells; ++i) {
        if (level.hole[i] != 0) {
            targets.push_back(i);
        }
    }
    if (targets.empty()) {
        return {};
    }
    std::vector<std::int32_t> nnf(cells, -1);
    if (coarser != nullptr && !carried.empty()) {
        nnf = upsample_nnf(*coarser, carried, level, radius);
    }
    for (int em = 0; em < em_iterations; ++em) {
        refine_nnf(level, nnf, source_mask, sources, targets, radius, patchmatch_iterations,
                   first_pass, rng);
        vote(level, nnf, targets, radius);
        first_pass = false;
    }
    return nnf;
}

/// Coarse-to-fine completion of one component's window. Returns false when the
/// finest level has no full valid source patch, in which case the caller keeps
/// the smooth values it wrote.
bool complete_component(int source_width, std::size_t first_cell, Level finest, int wx0,
                        int wy0, const TextureFillOptions& options,
                        std::vector<std::uint8_t>& rgba) {
    std::vector<Level> levels = build_pyramid(std::move(finest), options);
    // One generator per component, seeded from the component's raster-first
    // cell: processing order and threading cannot change the result.
    std::mt19937 rng(static_cast<std::mt19937::result_type>(
        options.seed ^ (static_cast<std::uint32_t>(first_cell) * 0x9E3779B9u)));
    std::vector<std::int32_t> carried;
    bool first_pass = true;
    bool has_source = false;
    for (int li = static_cast<int>(levels.size()) - 1; li >= 0; --li) {
        Level& level = levels[static_cast<std::size_t>(li)];
        Level* coarser = li + 1 < static_cast<int>(levels.size())
                             ? &levels[static_cast<std::size_t>(li + 1)]
                             : nullptr;
        if (coarser != nullptr) {
            upsample_holes(*coarser, level);
        }
        carried = process_level(level, coarser, carried, options.patch_radius,
                                options.em_iterations, options.patchmatch_iterations,
                                first_pass, rng);
        if (li == 0) {
            has_source = !carried.empty();
        }
    }
    if (!has_source) {
        return false;
    }
    const Level& fine = levels.front();
    for (int y = 0; y < fine.height; ++y) {
        for (int x = 0; x < fine.width; ++x) {
            const std::size_t local = cell_index(fine.width, x, y);
            if (fine.hole[local] == 0) {
                continue;
            }
            const std::size_t global = cell_index(source_width, wx0 + x, wy0 + y);
            for (int c = 0; c < 3; ++c) {
                rgba[global * 4U + static_cast<std::size_t>(c)] =
                    fine.rgb[local * 3U + static_cast<std::size_t>(c)];
            }
            rgba[global * 4U + 3U] = 255;
        }
    }
    return true;
}

}  // namespace

TextureFillReport texture_fill(int width, int height, std::vector<std::uint8_t>& rgba,
                               const std::vector<std::uint8_t>& fill_mask,
                               const TextureFillOptions& options) {
    if (width <= 0 || height <= 0) {
        throw ValueError("texture_fill: width and height must be positive");
    }
    const std::size_t cells =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (rgba.size() != cells * 4U || fill_mask.size() != cells) {
        throw ValueError("texture_fill: buffer sizes do not match width*height");
    }
    if (options.patch_radius < 0 || options.max_component_cells < 0 ||
        options.source_margin < 0 || options.pyramid_levels < 0 ||
        options.em_iterations < 0 || options.patchmatch_iterations < 0) {
        throw ValueError("texture_fill: options must be non-negative");
    }

    TextureFillReport report;
    const int patch = 2 * options.patch_radius + 1;
    const int margin = options.source_margin;
    std::vector<std::uint8_t> visited(cells, 0);
    for (std::size_t first = 0; first < cells; ++first) {
        if (fill_mask[first] == 0 || visited[first] != 0) {
            continue;
        }
        int min_x = 0;
        int max_x = 0;
        int min_y = 0;
        int max_y = 0;
        const std::vector<std::size_t> component = collect_component(
            width, height, fill_mask, visited, first, min_x, max_x, min_y, max_y);
        ++report.components;
        if (component.size() >
            static_cast<std::size_t>(options.max_component_cells)) {
            ++report.skipped_large;
            continue;
        }
        const int wx0 = std::max(0, min_x - margin);
        const int wy0 = std::max(0, min_y - margin);
        const int wx1 = std::min(width - 1, max_x + margin);
        const int wy1 = std::min(height - 1, max_y + margin);
        const int ww = wx1 - wx0 + 1;
        const int wh = wy1 - wy0 + 1;
        if (ww < patch || wh < patch) {
            ++report.skipped_large;
            continue;
        }
        std::vector<std::uint8_t> window_hole(
            static_cast<std::size_t>(ww) * static_cast<std::size_t>(wh), 0);
        for (const std::size_t cell : component) {
            const int x = static_cast<int>(cell % static_cast<std::size_t>(width));
            const int y = static_cast<int>(cell / static_cast<std::size_t>(width));
            window_hole[cell_index(ww, x - wx0, y - wy0)] = 1;
        }
        Level finest = make_window_level(width, rgba, fill_mask, window_hole, wx0, wy0, ww,
                                         wh);
        if (complete_component(width, first, std::move(finest), wx0, wy0, options, rgba)) {
            report.filled_cells += component.size();
        } else {
            ++report.skipped_large;
        }
    }
    return report;
}

}  // namespace alphashape::ortho
