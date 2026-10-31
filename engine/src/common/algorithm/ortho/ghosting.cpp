// Multi-view consensus ghosting filter -- see ghosting.hpp for the contract.

#include "common/algorithm/ortho/ghosting.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace alphashape::ortho {
namespace {

/// A tile-local binary mask view: `w` x `h`, row-major.
struct MaskView {
    int w = 0;
    int h = 0;
    std::uint8_t* data = nullptr;

    std::size_t index(int col, int row) const {
        return static_cast<std::size_t>(row) * static_cast<std::size_t>(w) +
               static_cast<std::size_t>(col);
    }
    std::uint8_t at(int col, int row) const {
        return col >= 0 && row >= 0 && col < w && row < h ? data[index(col, row)] : 0;
    }
};

/// One binary morphology pass. `grow` dilates (a cell becomes 1 when any cell
/// inside the square radius is 1); otherwise it erodes (a cell stays 1 only
/// when every cell inside the radius is 1). Outside the mask reads as 0, so an
/// erode pulls the border inward and a dilate never invents border pixels.
void morph(MaskView mask, int radius, bool grow) {
    if (radius <= 0 || mask.w <= 0 || mask.h <= 0) {
        return;
    }
    std::vector<std::uint8_t> out(static_cast<std::size_t>(mask.w) *
                                      static_cast<std::size_t>(mask.h),
                                  0);
    for (int row = 0; row < mask.h; ++row) {
        for (int col = 0; col < mask.w; ++col) {
            std::uint8_t value = grow ? 0 : 1;
            for (int dy = -radius; dy <= radius && value != (grow ? 1 : 0); ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const std::uint8_t v = mask.at(col + dx, row + dy);
                    if (grow && v != 0) {
                        value = 1;
                        break;
                    }
                    if (!grow && v == 0) {
                        value = 0;
                        break;
                    }
                }
            }
            out[static_cast<std::size_t>(row) * static_cast<std::size_t>(mask.w) +
                static_cast<std::size_t>(col)] = value;
        }
    }
    std::copy(out.begin(), out.end(), mask.data);
}

/// Median of `values` (which is reordered). An even count takes the mean of the
/// two middle values.
double median_of(std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid),
                     values.end());
    const double upper = values[mid];
    if (values.size() % 2 != 0) {
        return upper;
    }
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid - 1),
                     values.begin() + static_cast<std::ptrdiff_t>(mid));
    return 0.5 * (values[mid - 1] + upper);
}

/// Drop 4-connected marked components smaller than `min_cells`. Scan order and
/// the neighbour order are fixed, so the result is deterministic.
void drop_small_components(MaskView mask, int min_cells) {
    if (min_cells <= 1 || mask.w <= 0 || mask.h <= 0) {
        return;
    }
    std::vector<std::uint8_t> seen(static_cast<std::size_t>(mask.w) *
                                       static_cast<std::size_t>(mask.h),
                                   0);
    std::vector<std::size_t> queue;
    std::vector<std::size_t> component;
    for (int row = 0; row < mask.h; ++row) {
        for (int col = 0; col < mask.w; ++col) {
            const std::size_t start = mask.index(col, row);
            if (mask.data[start] == 0 || seen[start] != 0) {
                continue;
            }
            component.clear();
            queue.clear();
            queue.push_back(start);
            seen[start] = 1;
            for (std::size_t head = 0; head < queue.size(); ++head) {
                const std::size_t cell = queue[head];
                component.push_back(cell);
                const int c = static_cast<int>(cell % static_cast<std::size_t>(mask.w));
                const int r = static_cast<int>(cell / static_cast<std::size_t>(mask.w));
                const int steps[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
                for (const auto& step : steps) {
                    const int nc = c + step[0];
                    const int nr = r + step[1];
                    if (nc < 0 || nr < 0 || nc >= mask.w || nr >= mask.h) {
                        continue;
                    }
                    const std::size_t ni = mask.index(nc, nr);
                    if (mask.data[ni] != 0 && seen[ni] == 0) {
                        seen[ni] = 1;
                        queue.push_back(ni);
                    }
                }
            }
            if (static_cast<int>(component.size()) < min_cells) {
                for (const std::size_t cell : component) {
                    mask.data[cell] = 0;
                }
            }
        }
    }
}

/// Mark every camera deviant at one scale. `factor` is the box-downsample step
/// (1, 4, 16, ...). Rows are parallel; each thread owns its own scratch value
/// buffers and writes only its own cells of the per-tile masks.
void mark_level(int width, int height, int factor, const std::vector<OrthoTile>& tiles,
                const GhostOptions& options,
                std::vector<std::vector<std::uint8_t>>& masks) {
    const int step = std::max(1, factor);
    const int cw = (width + step - 1) / step;
    const int ch = (height + step - 1) / step;
    if (cw <= 0 || ch <= 0 || tiles.empty()) {
        return;
    }
    const std::size_t tile_count = tiles.size();
    const double k = options.k > 0.0 ? options.k : 1.0;
    const double floor_spread = std::max(0.0, options.min_abs_diff) / k;
    const int min_cameras = std::max(1, options.min_cameras);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::array<std::vector<double>, 3> values;
        std::array<std::vector<double>, 3> scratch;
        std::vector<std::size_t> cameras;
        for (auto& v : values) {
            v.reserve(tile_count);
        }
        for (auto& v : scratch) {
            v.reserve(tile_count);
        }
        cameras.reserve(tile_count);

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int crow = 0; crow < ch; ++crow) {
            for (int ccol = 0; ccol < cw; ++ccol) {
                for (auto& v : values) {
                    v.clear();
                }
                cameras.clear();
                for (std::size_t t = 0; t < tile_count; ++t) {
                    const OrthoTile& tile = tiles[t];
                    if (tile.empty()) {
                        continue;
                    }
                    double sum[3] = {0.0, 0.0, 0.0};
                    int count = 0;
                    for (int dy = 0; dy < step; ++dy) {
                        const int row = crow * step + dy;
                        if (row >= height) {
                            break;
                        }
                        for (int dx = 0; dx < step; ++dx) {
                            const int col = ccol * step + dx;
                            if (col >= width || !tile.contains(col, row)) {
                                continue;
                            }
                            const std::size_t buffer = tile.index(col, row);
                            if (tile.valid[buffer] == 0) {
                                continue;
                            }
                            for (int c = 0; c < 3; ++c) {
                                sum[c] += static_cast<double>(
                                    tile.rgb[buffer * 3 + static_cast<std::size_t>(c)]);
                            }
                            ++count;
                        }
                    }
                    if (count == 0) {
                        continue;
                    }
                    const double inv = 1.0 / static_cast<double>(count);
                    for (int c = 0; c < 3; ++c) {
                        values[static_cast<std::size_t>(c)].push_back(sum[c] * inv);
                    }
                    cameras.push_back(t);
                }
                if (static_cast<int>(cameras.size()) < min_cameras) {
                    continue;  // no consensus exists here
                }
                double consensus[3] = {0.0, 0.0, 0.0};
                double spread[3] = {0.0, 0.0, 0.0};
                for (int c = 0; c < 3; ++c) {
                    scratch[static_cast<std::size_t>(c)] = values[static_cast<std::size_t>(c)];
                    consensus[c] = median_of(scratch[static_cast<std::size_t>(c)]);
                    for (std::size_t i = 0; i < values[static_cast<std::size_t>(c)].size(); ++i) {
                        scratch[static_cast<std::size_t>(c)][i] =
                            std::abs(values[static_cast<std::size_t>(c)][i] - consensus[c]);
                    }
                    const double mad = median_of(scratch[static_cast<std::size_t>(c)]);
                    spread[c] = std::max(1.4826 * mad, floor_spread);
                }
                for (std::size_t i = 0; i < cameras.size(); ++i) {
                    bool deviant = false;
                    for (int c = 0; c < 3; ++c) {
                        const double value = values[static_cast<std::size_t>(c)][i];
                        const double difference = std::abs(value - consensus[c]);
                        if (difference > k * spread[c] &&
                            difference > options.min_abs_diff) {
                            deviant = true;
                            break;
                        }
                    }
                    if (!deviant) {
                        continue;
                    }
                    const OrthoTile& tile = tiles[cameras[i]];
                    for (int dy = 0; dy < step; ++dy) {
                        const int row = crow * step + dy;
                        if (row >= height) {
                            break;
                        }
                        for (int dx = 0; dx < step; ++dx) {
                            const int col = ccol * step + dx;
                            if (col >= width || !tile.contains(col, row)) {
                                continue;
                            }
                            const std::size_t buffer = tile.index(col, row);
                            if (tile.valid[buffer] != 0) {
                                masks[cameras[i]][buffer] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

std::vector<std::vector<std::uint8_t>> detect_ghosts(
    int width, int height, const std::vector<OrthoTile>& tiles,
    const GhostOptions& options) {
    std::vector<std::vector<std::uint8_t>> masks(tiles.size());
    for (std::size_t t = 0; t < tiles.size(); ++t) {
        const OrthoTile& tile = tiles[t];
        if (!tile.empty()) {
            masks[t].assign(static_cast<std::size_t>(tile.width) *
                                static_cast<std::size_t>(tile.height),
                            0);
        }
    }
    if (width <= 0 || height <= 0 || tiles.empty()) {
        return masks;
    }
    const int levels = std::max(1, options.levels);
    int factor = 1;
    for (int level = 0; level < levels; ++level) {
        mark_level(width, height, factor, tiles, options, masks);
        factor *= 4;
        if (factor > (std::numeric_limits<int>::max)() / 4) {
            break;
        }
    }
    for (std::size_t t = 0; t < tiles.size(); ++t) {
        const OrthoTile& tile = tiles[t];
        if (tile.empty()) {
            continue;
        }
        MaskView view{tile.width, tile.height, masks[t].data()};
        // Close (dilate, erode) then open (erode, dilate): the close joins a
        // broken object across a one-cell gap, the open removes single cells.
        morph(view, options.morph_radius, true);
        morph(view, options.morph_radius, false);
        morph(view, options.morph_radius, false);
        morph(view, options.morph_radius, true);
        drop_small_components(view, options.min_component_cells);
    }
    return masks;
}

int filter_label_speckles(int width, int height, const std::vector<OrthoTile>& tiles,
                          std::vector<std::int32_t>& labels,
                          const std::vector<std::uint8_t>* pinned, int min_component_cells) {
    const std::size_t cells =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (width <= 0 || height <= 0 || labels.size() < cells) {
        return 0;
    }
    const bool have_pins = pinned != nullptr && pinned->size() >= cells;
    const int min_cells = std::max(1, min_component_cells);
    std::vector<std::uint8_t> seen(cells, 0);
    std::vector<std::size_t> queue;
    std::vector<std::size_t> component;
    std::vector<std::int32_t> candidates;
    std::vector<std::size_t> counts;
    int removed = 0;
    const int steps[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const std::size_t start =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(col);
            const std::int32_t label = labels[start];
            if (label < 0 || seen[start] != 0) {
                continue;
            }
            component.clear();
            queue.clear();
            queue.push_back(start);
            seen[start] = 1;
            for (std::size_t head = 0; head < queue.size(); ++head) {
                const std::size_t cell = queue[head];
                component.push_back(cell);
                const int c = static_cast<int>(cell % static_cast<std::size_t>(width));
                const int r = static_cast<int>(cell / static_cast<std::size_t>(width));
                for (const auto& step : steps) {
                    const int nc = c + step[0];
                    const int nr = r + step[1];
                    if (nc < 0 || nr < 0 || nc >= width || nr >= height) {
                        continue;
                    }
                    const std::size_t ni =
                        static_cast<std::size_t>(nr) * static_cast<std::size_t>(width) +
                        static_cast<std::size_t>(nc);
                    if (labels[ni] == label && seen[ni] == 0) {
                        seen[ni] = 1;
                        queue.push_back(ni);
                    }
                }
            }
            if (static_cast<int>(component.size()) >= min_cells) {
                continue;
            }
            if (have_pins) {
                bool touches_pin = false;
                for (const std::size_t cell : component) {
                    if ((*pinned)[cell] != 0) {
                        touches_pin = true;
                        break;
                    }
                }
                if (touches_pin) {
                    continue;
                }
            }
            candidates.clear();
            counts.clear();
            for (const std::size_t cell : component) {
                const int c = static_cast<int>(cell % static_cast<std::size_t>(width));
                const int r = static_cast<int>(cell / static_cast<std::size_t>(width));
                for (const auto& step : steps) {
                    const int nc = c + step[0];
                    const int nr = r + step[1];
                    if (nc < 0 || nr < 0 || nc >= width || nr >= height) {
                        continue;
                    }
                    const std::size_t ni =
                        static_cast<std::size_t>(nr) * static_cast<std::size_t>(width) +
                        static_cast<std::size_t>(nc);
                    const std::int32_t neighbour = labels[ni];
                    if (neighbour < 0 || neighbour == label ||
                        neighbour >= static_cast<std::int32_t>(tiles.size())) {
                        continue;
                    }
                    const OrthoTile& tile = tiles[static_cast<std::size_t>(neighbour)];
                    bool valid_everywhere = true;
                    for (const std::size_t member : component) {
                        const int mc = static_cast<int>(member % static_cast<std::size_t>(width));
                        const int mr = static_cast<int>(member / static_cast<std::size_t>(width));
                        if (!tile.contains(mc, mr) || tile.valid[tile.index(mc, mr)] == 0) {
                            valid_everywhere = false;
                            break;
                        }
                    }
                    if (!valid_everywhere) {
                        continue;
                    }
                    const auto it = std::find(candidates.begin(), candidates.end(), neighbour);
                    if (it == candidates.end()) {
                        candidates.push_back(neighbour);
                        counts.push_back(1);
                    } else {
                        ++counts[static_cast<std::size_t>(
                            std::distance(candidates.begin(), it))];
                    }
                }
            }
            if (candidates.empty()) {
                continue;
            }
            std::size_t best = 0;
            for (std::size_t i = 1; i < candidates.size(); ++i) {
                if (counts[i] > counts[best] ||
                    (counts[i] == counts[best] && candidates[i] < candidates[best])) {
                    best = i;
                }
            }
            for (const std::size_t cell : component) {
                labels[cell] = candidates[best];
            }
            ++removed;
        }
    }
    return removed;
}

}  // namespace alphashape::ortho
