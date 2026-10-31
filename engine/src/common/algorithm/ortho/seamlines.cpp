// Seamline labelling (multi-scale alpha-expansion) -- see seamlines.hpp.

#include "common/algorithm/ortho/seamlines.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace alphashape::ortho {
namespace {

constexpr double kGreyRange = 255.0;  // per-channel byte range
constexpr double kFlowEps = 1e-11;    // capacities below this are zero
constexpr double kForbid = 1e9;       // "this label cannot be taken here"

/// A per-tile field on one grid. `score`, `rgb` and `valid` point at the owning
/// storage (the tiles' cropped buffers at full resolution, a block-mean copy on
/// a coarse level); `best` is the highest score of any tile valid at the cell.
///
/// The tile buffers are CROPPED to the camera footprint, so every read of tile
/// `t` at cell (col, row) is translated through `x0/y0` and the tile's own
/// `stride`. A cell outside the crop reads as invalid without touching the
/// buffer, which is exactly the "not covered" case the full-grid layout padded
/// with zeros.
struct Field {
    int width = 0;
    int height = 0;
    std::vector<const float*> score;
    std::vector<const std::uint8_t*> rgb;
    std::vector<const std::uint8_t*> valid;
    /// Optional pin mask (nullptr when every cell is free): a non-zero entry
    /// fixes the cell's label for the whole optimisation.
    const std::uint8_t* pinned = nullptr;
    /// The pinned label values, consulted only where `pinned` is non-zero.
    const std::int32_t* pinned_label = nullptr;
    /// Tile crop boxes and buffer row strides, in FIELD cells (a coarse level
    /// scales them down with its own grid).
    std::vector<int> x0;
    std::vector<int> y0;
    std::vector<int> tile_width;
    std::vector<int> tile_height;
    std::vector<int> stride;
    /// Optional per-tile deviant mask (see SeamOptions::avoid); nullptr entry
    /// means that tile has no mask. Read through index(t, col, row).
    std::vector<const std::uint8_t*> avoid;
    std::vector<double> best;
    std::vector<std::uint8_t> any_valid;
    /// Number of tiles valid at each cell: a lone valid camera is never
    /// penalised for being deviant, because dropping it would make a hole.
    std::vector<std::int32_t> valid_count;
    std::vector<std::size_t> parity[2];

    std::size_t cells() const {
        return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    }
    std::size_t tiles() const { return score.size(); }

    /// True when tile `t` holds field cell (col, row).
    bool has(std::size_t t, int col, int row) const {
        return col >= x0[t] && col < x0[t] + tile_width[t] && row >= y0[t] &&
               row < y0[t] + tile_height[t];
    }

    /// Linear index of field cell (col, row) in tile `t`'s buffers; the caller
    /// must have established has(t, col, row).
    std::size_t index(std::size_t t, int col, int row) const {
        return static_cast<std::size_t>(row - y0[t]) * static_cast<std::size_t>(stride[t]) +
               static_cast<std::size_t>(col - x0[t]);
    }

    /// Tile `t`'s validity at field cell (col, row), false when out of crop.
    bool valid_at(std::size_t t, int col, int row) const {
        return has(t, col, row) && valid[t][index(t, col, row)] != 0;
    }

    float score_at(std::size_t t, int col, int row) const {
        return score[t][index(t, col, row)];
    }

    const std::uint8_t* rgb_at(std::size_t t, int col, int row) const {
        return rgb[t] + index(t, col, row) * 3;
    }
};

/// Fill `best`, `any_valid` and the checkerboard parity lists from the tiles.
void finalize(Field& field) {
    const std::size_t cells = field.cells();
    field.best.assign(cells, 0.0);
    field.any_valid.assign(cells, 0);
    field.valid_count.assign(cells, 0);
    for (std::size_t t = 0; t < field.tiles(); ++t) {
        const int tw = field.tile_width[t];
        const int th = field.tile_height[t];
        const std::uint8_t* valid = field.valid[t];
        const float* score = field.score[t];
        const int stride = field.stride[t];
        for (int row = 0; row < th; ++row) {
            const std::size_t cell_row =
                static_cast<std::size_t>(field.y0[t] + row) *
                static_cast<std::size_t>(field.width);
            const std::size_t buf_row = static_cast<std::size_t>(row) *
                                        static_cast<std::size_t>(stride);
            for (int col = 0; col < tw; ++col) {
                const std::size_t b = buf_row + static_cast<std::size_t>(col);
                if (valid[b] == 0) {
                    continue;
                }
                const std::size_t cell =
                    cell_row + static_cast<std::size_t>(field.x0[t] + col);
                field.any_valid[cell] = 1;
                ++field.valid_count[cell];
                field.best[cell] = std::max(field.best[cell], static_cast<double>(score[b]));
            }
        }
    }
    field.parity[0].clear();
    field.parity[1].clear();
    for (int row = 0; row < field.height; ++row) {
        for (int col = 0; col < field.width; ++col) {
            field.parity[(row + col) & 1].push_back(
                static_cast<std::size_t>(row) * static_cast<std::size_t>(field.width) +
                static_cast<std::size_t>(col));
        }
    }
}

bool covers(const Field& field, int label, std::size_t cell) {
    if (label < 0) {
        return false;
    }
    const int col = static_cast<int>(cell % static_cast<std::size_t>(field.width));
    const int row = static_cast<int>(cell / static_cast<std::size_t>(field.width));
    return field.valid_at(static_cast<std::size_t>(label), col, row);
}

/// The data cost of label `label` at field cell (col, row): the score shortfall
/// against the best tile there, scaled by data_weight. -1 is free where no tile
/// covers the cell; a tile that does not cover the cell is forbidden.
double data_cost_at(const Field& field, int label, std::size_t cell, int col, int row,
                    const SeamOptions& options) {
    if (label < 0) {
        return field.any_valid[cell] != 0 ? kForbid : 0.0;
    }
    const std::size_t t = static_cast<std::size_t>(label);
    if (!field.valid_at(t, col, row)) {
        return kForbid;
    }
    const double shortfall = field.best[cell] - static_cast<double>(field.score_at(t, col, row));
    double cost = options.data_weight * std::max(0.0, shortfall);
    // Ghosting: prefer a camera the multi-view consensus agrees with. A lone
    // valid camera is exempt, so a marked cell is never cut into a hole.
    if (t < field.avoid.size() && field.avoid[t] != nullptr &&
        field.avoid[t][field.index(t, col, row)] != 0 && field.valid_count[cell] > 1) {
        cost += options.avoid_cost;
    }
    return cost;
}

/// The same, addressed by a linear cell index.
double data_cost(const Field& field, int label, std::size_t cell, const SeamOptions& options) {
    const int col = static_cast<int>(cell % static_cast<std::size_t>(field.width));
    const int row = static_cast<int>(cell / static_cast<std::size_t>(field.width));
    return data_cost_at(field, label, cell, col, row, options);
}

/// Sum of the per-channel absolute colour differences the two tiles show at
/// field cell (col, row), or 0 when either does not cover it.
double colour_sum_at(const Field& field, int a, int b, int col, int row) {
    if (a < 0 || b < 0) {
        return 0.0;
    }
    const std::size_t ta = static_cast<std::size_t>(a);
    const std::size_t tb = static_cast<std::size_t>(b);
    if (!field.valid_at(ta, col, row) || !field.valid_at(tb, col, row)) {
        return 0.0;
    }
    const std::uint8_t* pa = field.rgb_at(ta, col, row);
    const std::uint8_t* pb = field.rgb_at(tb, col, row);
    double sum = 0.0;
    for (int c = 0; c < 3; ++c) {
        sum += std::abs(static_cast<int>(pa[c]) - static_cast<int>(pb[c]));
    }
    return sum;
}

/// The seam penalty of one grid edge p--q between labels a and b: zero when the
/// labels agree or either is the empty label, else the mean per-channel grey
/// disagreement over the two ends (sum of channel differences / 255, in [0, 3])
/// scaled by seam_weight, plus the constant `smoothness` change penalty.
double seam_cost_at(const Field& field, int a, int b, int pcol, int prow, int qcol, int qrow,
                    const SeamOptions& options) {
    if (a == b || a < 0 || b < 0) {
        return 0.0;
    }
    const double disagreement =
        (colour_sum_at(field, a, b, pcol, prow) + colour_sum_at(field, a, b, qcol, qrow)) /
        kGreyRange;
    return options.seam_weight * disagreement + options.smoothness;
}

/// The linear-cell form of seam_cost_at(), for the callers that only have cell
/// indices (the alpha-expansion graph and the energy report).
double seam_cost(const Field& field, int a, int b, std::size_t p, std::size_t q,
                 const SeamOptions& options) {
    if (a == b || a < 0 || b < 0) {
        return 0.0;
    }
    const int pcol = static_cast<int>(p % static_cast<std::size_t>(field.width));
    const int prow = static_cast<int>(p / static_cast<std::size_t>(field.width));
    const int qcol = static_cast<int>(q % static_cast<std::size_t>(field.width));
    const int qrow = static_cast<int>(q / static_cast<std::size_t>(field.width));
    return seam_cost_at(field, a, b, pcol, prow, qcol, qrow, options);
}

/// True when tile `t` is marked deviant at (col, row) and is not the only view
/// there (the same rule data_cost_at applies).
bool is_avoided_at(const Field& field, std::size_t t, std::size_t cell, int col, int row) {
    return t < field.avoid.size() && field.avoid[t] != nullptr &&
           field.avoid[t][field.index(t, col, row)] != 0 && field.valid_count[cell] > 1;
}

/// The score-argmax label at one cell, or -1 when nothing covers it. A deviant
/// camera is skipped when another valid camera exists, so the filter also works
/// with seam refinement off (which takes this argmax directly).
int argmax_at(const Field& field, std::size_t cell) {
    const int col = static_cast<int>(cell % static_cast<std::size_t>(field.width));
    const int row = static_cast<int>(cell / static_cast<std::size_t>(field.width));
    for (int pass = 0; pass < 2; ++pass) {
        int best_label = -1;
        double best = 0.0;
        for (std::size_t t = 0; t < field.tiles(); ++t) {
            if (!field.valid_at(t, col, row)) {
                continue;
            }
            if (pass == 0 && is_avoided_at(field, t, cell, col, row)) {
                continue;
            }
            const double score = static_cast<double>(field.score_at(t, col, row));
            if (best_label < 0 || score > best) {
                best = score;
                best_label = static_cast<int>(t);
            }
        }
        if (best_label >= 0) {
            return best_label;
        }
    }
    return -1;
}

std::vector<std::int32_t> argmax_labels(const Field& field) {
    std::vector<std::int32_t> labels(field.cells(), -1);
    for (std::size_t cell = 0; cell < labels.size(); ++cell) {
        labels[cell] = argmax_at(field, cell);
    }
    return labels;
}

// -- push-relabel max-flow ---------------------------------------------------

/// FIFO push-relabel with periodic global relabeling (Cherkassky & Goldberg
/// 1997), which makes the flow fast enough for the 480x360 ortho grids.
/// Single-threaded, with a fixed edge order and queue, so the residual graph --
/// and hence the min cut -- is deterministic.
class MaxFlow {
public:
    explicit MaxFlow(int nodes)
        : head_(static_cast<std::size_t>(nodes), -1),
          height_(static_cast<std::size_t>(nodes), 0),
          current_(static_cast<std::size_t>(nodes), -1),
          excess_(static_cast<std::size_t>(nodes), 0.0),
          seen_(static_cast<std::size_t>(nodes), 0) {}

    void add_edge(int u, int v, double capacity) {
        if (capacity <= kFlowEps) {
            return;
        }
        const int a = static_cast<int>(to_.size());
        to_.push_back(v);
        cap_.push_back(capacity);
        next_.push_back(head_[static_cast<std::size_t>(u)]);
        head_[static_cast<std::size_t>(u)] = a;
        to_.push_back(u);
        cap_.push_back(0.0);
        next_.push_back(head_[static_cast<std::size_t>(v)]);
        head_[static_cast<std::size_t>(v)] = a + 1;
    }

    void run(int source, int sink) {
        source_ = source;
        sink_ = sink;
        nodes_ = static_cast<int>(head_.size());
        height_[static_cast<std::size_t>(source)] = nodes_;
        for (int e = head_[static_cast<std::size_t>(source)]; e != -1;
             e = next_[static_cast<std::size_t>(e)]) {
            const std::size_t es = static_cast<std::size_t>(e);
            const double capacity = cap_[es];
            if (capacity > kFlowEps) {
                cap_[es] = 0.0;
                cap_[es ^ 1] += capacity;
                excess_[static_cast<std::size_t>(to_[es])] += capacity;
            }
        }
        global_relabel();
        std::size_t head = 0;
        std::size_t discharges_since_relabel = 0;
        while (head < queue_.size()) {
            const int u = queue_[head++];
            seen_[static_cast<std::size_t>(u)] = 0;
            discharge(u);
            if (excess_[static_cast<std::size_t>(u)] > kFlowEps &&
                height_[static_cast<std::size_t>(u)] < 2 * nodes_) {
                enqueue(u);  // still pumpable; a capped node is done
            }
            // Periodic global relabeling bounds the relabel work on dense grids.
            if (++discharges_since_relabel > 4u * static_cast<std::size_t>(nodes_)) {
                global_relabel();
                head = 0;
                discharges_since_relabel = 0;
            }
            if (head > 8u * static_cast<std::size_t>(nodes_) && head * 2 > queue_.size()) {
                queue_.erase(queue_.begin(),
                             queue_.begin() + static_cast<std::ptrdiff_t>(head));
                head = 0;  // bound the queue's memory on pathological inputs
            }
        }
    }

    /// Residual reachability from `source`: the cut's source side, i.e. the
    /// cells that take the new label in an alpha-expansion move.
    std::vector<char> source_side(int source) const {
        std::vector<char> side(head_.size(), 0);
        std::vector<int> stack;
        stack.push_back(source);
        side[static_cast<std::size_t>(source)] = 1;
        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();
            for (int e = head_[static_cast<std::size_t>(u)]; e != -1;
                 e = next_[static_cast<std::size_t>(e)]) {
                const std::size_t es = static_cast<std::size_t>(e);
                const int v = to_[es];
                if (cap_[es] > kFlowEps && side[static_cast<std::size_t>(v)] == 0) {
                    side[static_cast<std::size_t>(v)] = 1;
                    stack.push_back(v);
                }
            }
        }
        return side;
    }

private:
    void enqueue(int v) {
        const std::size_t vs = static_cast<std::size_t>(v);
        if (seen_[vs] != 0) {
            return;
        }
        seen_[vs] = 1;
        queue_.push_back(v);
    }

    void discharge(int u) {
        const std::size_t us = static_cast<std::size_t>(u);
        // A node whose height reaches the top cannot reach the sink: its excess
        // is already on the source side of the cut, so it stays put. This caps
        // the relabel/push loop and guarantees termination.
        while (excess_[us] > kFlowEps && height_[us] < 2 * nodes_) {
            if (current_[us] == -1) {
                relabel(u);
                current_[us] = head_[us];
                continue;
            }
            const int e = current_[us];
            const std::size_t es = static_cast<std::size_t>(e);
            const int v = to_[es];
            if (cap_[es] > kFlowEps && height_[us] == height_[static_cast<std::size_t>(v)] + 1) {
                const double delta = std::min(excess_[us], cap_[es]);
                cap_[es] -= delta;
                cap_[es ^ 1] += delta;
                excess_[us] -= delta;
                excess_[static_cast<std::size_t>(v)] += delta;
                if (v != source_ && v != sink_) {
                    enqueue(v);
                }
            } else {
                current_[us] = next_[es];
            }
        }
    }

    void relabel(int u) {
        const std::size_t us = static_cast<std::size_t>(u);
        int best = 2 * nodes_;
        for (int e = head_[us]; e != -1; e = next_[static_cast<std::size_t>(e)]) {
            const std::size_t es = static_cast<std::size_t>(e);
            if (cap_[es] > kFlowEps) {
                best = std::min(best, height_[static_cast<std::size_t>(to_[es])] + 1);
            }
        }
        height_[us] = best;
    }

    /// Recompute exact heights by a backward BFS from the sink over the residual
    /// graph: height[v] is the residual distance to the sink, and nodes that
    /// cannot reach it get nodes_ + 1. Draining and requeuing the active set
    /// keeps FIFO behaviour with exact labels, which collapses the relabel work.
    void global_relabel() {
        std::fill(height_.begin(), height_.end(), nodes_ + 1);
        std::fill(current_.begin(), current_.end(), -1);
        height_[static_cast<std::size_t>(sink_)] = 0;
        std::vector<int> bfs;
        bfs.push_back(sink_);
        std::size_t head = 0;
        while (head < bfs.size()) {
            const int u = bfs[head++];
            const int next_height = height_[static_cast<std::size_t>(u)] + 1;
            for (int e = head_[static_cast<std::size_t>(u)]; e != -1;
                 e = next_[static_cast<std::size_t>(e)]) {
                const std::size_t es = static_cast<std::size_t>(e);
                // The reverse edge v -> u has residual capacity cap_[es ^ 1].
                if (cap_[es ^ 1] > kFlowEps &&
                    height_[static_cast<std::size_t>(to_[es])] == nodes_ + 1) {
                    height_[static_cast<std::size_t>(to_[es])] = next_height;
                    bfs.push_back(to_[es]);
                }
            }
        }
        height_[static_cast<std::size_t>(source_)] = nodes_;
        queue_.clear();
        for (int v = 0; v < nodes_; ++v) {
            const std::size_t vs = static_cast<std::size_t>(v);
            seen_[vs] = 0;
            if (v != source_ && v != sink_ && excess_[vs] > kFlowEps) {
                enqueue(v);
            }
        }
    }

    std::vector<int> head_;
    std::vector<int> to_;
    std::vector<int> next_;
    std::vector<double> cap_;
    std::vector<int> height_;
    std::vector<int> current_;
    std::vector<double> excess_;
    std::vector<char> seen_;
    std::vector<int> queue_;
    int nodes_ = 0;
    int source_ = -1;
    int sink_ = -1;
};

/// The objective of `labels` under `options`, evaluated on an existing Field.
/// The public energy() wraps this after building and de-ghosting the field.
double total_energy(const Field& field, const std::vector<std::int32_t>& labels,
                    const SeamOptions& options) {
    const int width = field.width;
    const int height = field.height;
    double total = 0.0;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const std::size_t cell =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(col);
            const int label = labels[cell];
            total += data_cost_at(field, label, cell, col, row, options);
            if (label < 0) {
                continue;  // no valid tile here: no seam contributes
            }
            // Count each unordered neighbour pair once (right and down).
            if (col + 1 < width && labels[cell + 1] != label) {
                total += seam_cost_at(field, label, labels[cell + 1], col, row, col + 1, row,
                                      options);
            }
            if (row + 1 < height &&
                labels[cell + static_cast<std::size_t>(width)] != label) {
                total += seam_cost_at(field, label,
                                      labels[cell + static_cast<std::size_t>(width)], col, row,
                                      col, row + 1, options);
            }
        }
    }
    return total;
}

/// A rectangular window of FIELD cells the full-resolution ghosting expansion
/// may relabel. Everything outside keeps the label the coarse pass gave it, so
/// the graph is bounded by the marked objects' neighbourhood instead of the
/// whole mosaic. A default Region is empty and never gated in.
struct Region {
    int x0 = 0;  ///< inclusive
    int y0 = 0;  ///< inclusive
    int x1 = 0;  ///< exclusive
    int y1 = 0;  ///< exclusive
    bool contains(int col, int row) const {
        return col >= x0 && col < x1 && row >= y0 && row < y1;
    }
};

/// One alpha-expansion move: the min-cut (Boykov, Veksler & Zabih 2001) for the
/// set of cells to relabel as `alpha`. Cells already carrying `alpha` and cells
/// outside `region` (nullptr = whole grid) are fixed on the keep side.
///
/// The seam term is a metric only where both tiles cover both ends of an edge.
/// A cropped tile can be invalid at one end, which makes E00 exceed
/// E(alpha,lq) + E(lp,alpha); every pair term is therefore truncated to stay
/// submodular (Rother, Kumar, Kolmogorov & Blake 2005, "Digital tapestry") and
/// the caller accepts a move only when the TRUE energy falls, so the sweep is a
/// monotone descent rather than an exact solve.
bool expand_alpha(const Field& field, std::vector<std::int32_t>& labels, int alpha,
                  const SeamOptions& options, const Region* region) {
    const int width = field.width;
    const int height = field.height;
    const std::size_t cells = field.cells();
    if (field.tile_width[static_cast<std::size_t>(alpha)] <= 0) {
        return false;  // the tile covers nothing
    }
    const std::size_t alpha_cells =
        static_cast<std::size_t>(field.tile_width[static_cast<std::size_t>(alpha)]) *
        static_cast<std::size_t>(field.tile_height[static_cast<std::size_t>(alpha)]);
    const bool alpha_covers_any =
        std::any_of(field.valid[static_cast<std::size_t>(alpha)],
                    field.valid[static_cast<std::size_t>(alpha)] +
                        static_cast<std::ptrdiff_t>(alpha_cells),
                    [](std::uint8_t v) { return v != 0; });
    if (!alpha_covers_any) {
        return false;
    }

    // Graph node per cell that is neither already `alpha` nor fixed by region;
    // source side = switch.
    std::vector<int> node(cells, -1);
    int count = 0;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const std::size_t cell =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(col);
            if (labels[cell] == alpha) {
                continue;
            }
            if (region != nullptr && !region->contains(col, row)) {
                continue;
            }
            node[cell] = count++;
        }
    }
    if (count == 0) {
        return false;
    }
    const int source = count;
    const int sink = count + 1;
    MaxFlow flow(count + 2);

    for (std::size_t cell = 0; cell < cells; ++cell) {
        const int p = node[cell];
        if (p < 0) {
            continue;
        }
        const double to_alpha =
            covers(field, alpha, cell) ? data_cost(field, alpha, cell, options) : kForbid;
        // A pinned cell may not switch: forbid the sink side outright, so the
        // pair decomposition below stays the standard one and the pin is a hard
        // constraint rather than a very expensive preference.
        const bool is_pinned = field.pinned != nullptr && field.pinned[cell] != 0;
        flow.add_edge(p, sink, is_pinned ? kForbid : to_alpha);     // p switches
        flow.add_edge(source, p, is_pinned ? 0.0 : data_cost(field, labels[cell], cell, options));
    }

    // Pairwise decomposition of E(x_p, x_q), where x = 1 means "switch to
    // alpha". With a = E(1,0) = e(alpha,lq), b = E(0,1) = e(lp,alpha) and
    // E(0,0) = e(lp,lq), E(1,1) = 0, the polynomial
    //
    //     E = (E00 - a) * y_p + a * y_q + (b + a - E00) * x_q * y_p,
    //     y = 1 - x ("keeps"),
    //
    // reproduces all four values. Where one tile is invalid at an end the
    // colour sum is 0, so E00 can exceed b + a and the crossing term would be
    // negative (and add_edge silently drops it). Truncating E00 to b + a
    // restores E00 + E11 <= E(1,0) + E(0,1) without changing the move.
    // The q -> p edge is cut when q switches and p keeps.
    auto add_fixed_unary = [&](int n, double keep_cost, double switch_cost) {
        if (keep_cost >= switch_cost) {
            flow.add_edge(source, n, keep_cost - switch_cost);  // cut when keeps
        } else {
            flow.add_edge(n, sink, switch_cost - keep_cost);  // cut when switches
        }
    };
    auto add_pair = [&](std::size_t pc, std::size_t qc) {
        const int np = node[pc];
        const int nq = node[qc];
        if (np < 0 && nq < 0) {
            return;  // both fixed: a constant, no cut
        }
        const int lp = labels[pc];
        const int lq = labels[qc];
        const double e10 = seam_cost(field, alpha, lq, pc, qc, options);
        const double e01 = seam_cost(field, lp, alpha, pc, qc, options);
        if (np >= 0 && nq >= 0) {
            double e00 = seam_cost(field, lp, lq, pc, qc, options);
            if (e00 > e01 + e10) {
                e00 = e01 + e10;  // non-submodular truncation
            }
            // p keeps: (E00 - e10) * y_p; a negative capacity is shifted onto
            // p -> t and the constant is dropped (it cannot change the argmin).
            if (e00 >= e10) {
                flow.add_edge(source, np, e00 - e10);
            } else {
                flow.add_edge(np, sink, e10 - e00);
            }
            flow.add_edge(source, nq, e10);          // q keeps
            flow.add_edge(nq, np, e01 + e10 - e00);  // q switches, p keeps
        } else if (np >= 0) {
            // q cannot switch (already alpha or outside the region): p keeps
            // at e00 and switches at e10.
            add_fixed_unary(np, seam_cost(field, lp, lq, pc, qc, options), e10);
        } else {
            add_fixed_unary(nq, seam_cost(field, lp, lq, pc, qc, options), e01);
        }
    };
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const std::size_t cell =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(col);
            if (col + 1 < width) {
                add_pair(cell, cell + 1);
            }
            if (row + 1 < height) {
                add_pair(cell, cell + static_cast<std::size_t>(width));
            }
        }
    }

    flow.run(source, sink);
    const std::vector<char> side = flow.source_side(source);
    std::vector<std::size_t> switched;
    std::vector<std::int32_t> previous;
    for (std::size_t cell = 0; cell < cells; ++cell) {
        const int p = node[cell];
        if (p >= 0 && side[static_cast<std::size_t>(p)] != 0) {
            switched.push_back(cell);
            previous.push_back(labels[cell]);
        }
    }
    if (switched.empty()) {
        return false;
    }
    // The truncated pair term makes the min cut optimal for the SURROGATE
    // energy, not necessarily for the true one. Accept the move only when the
    // true energy actually falls; a rejected move restores the previous labels,
    // so the sweep is monotone and can never oscillate.
    const double before = total_energy(field, labels, options);
    for (const std::size_t cell : switched) {
        labels[cell] = alpha;
    }
    if (total_energy(field, labels, options) < before) {
        return true;
    }
    for (std::size_t i = 0; i < switched.size(); ++i) {
        labels[switched[i]] = previous[i];
    }
    return false;
}

/// Sweep the expansion targets until a full cycle changes nothing. `alphas`
/// (nullptr = every tile) and `region` (nullptr = the whole grid) bound the
/// work of the full-resolution ghosting pass; the coarse levels pass both as
/// nullptr and keep the exact whole-level sweep.
void alpha_expansion(const Field& field, std::vector<std::int32_t>& labels,
                     const SeamOptions& options, int max_cycles,
                     const Region* region = nullptr,
                     const std::vector<int>* alphas = nullptr) {
    const int labels_count = static_cast<int>(field.tiles());
    const int count = alphas != nullptr ? static_cast<int>(alphas->size()) : labels_count;
    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        bool changed = false;
        for (int i = 0; i < count; ++i) {
            const int alpha = alphas != nullptr ? (*alphas)[static_cast<std::size_t>(i)] : i;
            changed = expand_alpha(field, labels, alpha, options, region) || changed;
        }
        if (!changed) {
            break;
        }
    }
}

/// One checkerboard sweep of iterated conditional modes, used as a cheap final
/// polish. The two parities are swept separately so no cell reads a
/// half-updated neighbour and the result is independent of the thread count.
void icm_sweep(const Field& field, std::vector<std::int32_t>& labels,
               const SeamOptions& options) {
    const int width = field.width;
    const int height = field.height;
    for (int phase = 0; phase < 2; ++phase) {
        const std::vector<std::size_t>& list = field.parity[phase];
        const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(list.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < count; ++i) {
            const size_t cell = list[static_cast<std::size_t>(i)];
            if (field.pinned != nullptr && field.pinned[cell] != 0) {
                continue;  // decided by an already-processed block
            }
            const int col = static_cast<int>(cell % static_cast<std::size_t>(width));
            const int row = static_cast<int>(cell / static_cast<std::size_t>(width));

            std::size_t neighbours[4];
            int neighbour_count = 0;
            if (col > 0) {
                neighbours[neighbour_count++] = cell - 1;
            }
            if (col + 1 < width) {
                neighbours[neighbour_count++] = cell + 1;
            }
            if (row > 0) {
                neighbours[neighbour_count++] = cell - static_cast<std::size_t>(width);
            }
            if (row + 1 < height) {
                neighbours[neighbour_count++] = cell + static_cast<std::size_t>(width);
            }

            const int current = labels[cell];
            const double current_data = data_cost_at(field, current, cell, col, row, options);
            double current_seam[4] = {0.0, 0.0, 0.0, 0.0};
            for (int n = 0; n < neighbour_count; ++n) {
                const int ncol =
                    static_cast<int>(neighbours[n] % static_cast<std::size_t>(width));
                const int nrow =
                    static_cast<int>(neighbours[n] / static_cast<std::size_t>(width));
                current_seam[n] = seam_cost_at(field, current, labels[neighbours[n]], col, row,
                                               ncol, nrow, options);
            }

            int best_label = current;
            double best_delta = 0.0;  // keep the current label on ties
            for (std::size_t t = 0; t < field.tiles(); ++t) {
                const int label = static_cast<int>(t);
                if (!field.valid_at(t, col, row)) {
                    continue;  // a label must cover the cell
                }
                double delta = data_cost_at(field, label, cell, col, row, options) - current_data;
                for (int n = 0; n < neighbour_count; ++n) {
                    const int ncol =
                        static_cast<int>(neighbours[n] % static_cast<std::size_t>(width));
                    const int nrow =
                        static_cast<int>(neighbours[n] / static_cast<std::size_t>(width));
                    delta += seam_cost_at(field, label, labels[neighbours[n]], col, row, ncol,
                                          nrow, options) -
                             current_seam[n];
                }
                if (delta < best_delta) {
                    best_delta = delta;
                    best_label = label;
                }
            }
            labels[cell] = best_label;
        }
    }
}

/// A block-mean copy of the tiles at `factor` x coarser resolution, owning its
/// storage. Like the fine level, each tile's coarse buffer covers only its own
/// crop box (scaled down), so a large flight does not pay C x "all coarse
/// cells" at every pyramid level.
struct Coarse {
    std::vector<std::vector<std::uint8_t>> rgb;
    std::vector<std::vector<float>> score;
    std::vector<std::vector<std::uint8_t>> valid;
    std::vector<std::vector<std::uint8_t>> avoid;
    Field field;
};

void build_coarse(Coarse& coarse, const std::vector<OrthoTile>& tiles, int width, int height,
                  int factor, const std::vector<std::vector<std::uint8_t>>* avoid) {
    const int cw = (width + factor - 1) / factor;
    const int ch = (height + factor - 1) / factor;
    const std::size_t tile_count = tiles.size();
    coarse.rgb.resize(tile_count);
    coarse.score.resize(tile_count);
    coarse.valid.resize(tile_count);
    coarse.avoid.resize(tile_count);

    coarse.field.width = cw;
    coarse.field.height = ch;
    coarse.field.score.resize(tile_count);
    coarse.field.rgb.resize(tile_count);
    coarse.field.valid.resize(tile_count);
    coarse.field.avoid.assign(tile_count, nullptr);
    coarse.field.x0.resize(tile_count);
    coarse.field.y0.resize(tile_count);
    coarse.field.tile_width.resize(tile_count);
    coarse.field.tile_height.resize(tile_count);
    coarse.field.stride.resize(tile_count);

    for (std::size_t t = 0; t < tile_count; ++t) {
        const OrthoTile& tile = tiles[t];
        if (tile.empty()) {
            coarse.field.x0[t] = 0;
            coarse.field.y0[t] = 0;
            coarse.field.tile_width[t] = 0;
            coarse.field.tile_height[t] = 0;
            coarse.field.stride[t] = 0;
            coarse.field.score[t] = nullptr;
            coarse.field.rgb[t] = nullptr;
            coarse.field.valid[t] = nullptr;
            continue;
        }
        // The tile's crop box, in coarse cells (a coarse cell covers the block of
        // fine cells it starts at, so floor on the origin and ceil on the far
        // edge keeps every fine cell the tile holds inside the coarse box).
        const int tx0 = tile.x0 / factor;
        const int ty0 = tile.y0 / factor;
        const int tx1 = (tile.x0 + tile.width + factor - 1) / factor;
        const int ty1 = (tile.y0 + tile.height + factor - 1) / factor;
        const int bw = std::max(0, tx1 - tx0);
        const int bh = std::max(0, ty1 - ty0);
        const std::size_t cells = static_cast<std::size_t>(bw) * static_cast<std::size_t>(bh);
        coarse.rgb[t].assign(cells * 3, 0);
        coarse.score[t].assign(cells, 0.0F);
        coarse.valid[t].assign(cells, 0);

        for (int brow = 0; brow < bh; ++brow) {
            for (int bcol = 0; bcol < bw; ++bcol) {
                const std::size_t cell =
                    static_cast<std::size_t>(brow) * static_cast<std::size_t>(bw) +
                    static_cast<std::size_t>(bcol);
                double srgb[3] = {0.0, 0.0, 0.0};
                double sscore = 0.0;
                int count = 0;
                for (int dr = 0; dr < factor; ++dr) {
                    const int row = (ty0 + brow) * factor + dr;
                    if (row >= height) {
                        break;
                    }
                    for (int dc = 0; dc < factor; ++dc) {
                        const int col = (tx0 + bcol) * factor + dc;
                        if (col >= width) {
                            break;
                        }
                        if (!tile.contains(col, row)) {
                            continue;
                        }
                        const std::size_t fine = tile.index(col, row);
                        if (tile.valid[fine] == 0) {
                            continue;
                        }
                        for (int c = 0; c < 3; ++c) {
                            srgb[c] += static_cast<double>(
                                tile.rgb[fine * 3 + static_cast<std::size_t>(c)]);
                        }
                        sscore += static_cast<double>(tile.score[fine]);
                        ++count;
                    }
                }
                if (count == 0) {
                    continue;
                }
                const double inv = 1.0 / static_cast<double>(count);
                for (int c = 0; c < 3; ++c) {
                    coarse.rgb[t][cell * 3 + static_cast<std::size_t>(c)] =
                        static_cast<std::uint8_t>(std::lround(std::clamp(srgb[c] * inv, 0.0, kGreyRange)));
                }
                coarse.score[t][cell] = static_cast<float>(sscore * inv);
                coarse.valid[t][cell] = 1;
            }
        }

        coarse.field.x0[t] = tx0;
        coarse.field.y0[t] = ty0;
        coarse.field.tile_width[t] = bw;
        coarse.field.tile_height[t] = bh;
        coarse.field.stride[t] = bw;  // the tile's own buffer width
        coarse.field.score[t] = coarse.score[t].data();
        coarse.field.rgb[t] = coarse.rgb[t].data();
        coarse.field.valid[t] = coarse.valid[t].data();
        // The deviant mask follows the same box: a coarse cell is deviant when
        // any of its fine cells is, which is the conservative direction (a
        // ghost is kept out, never hidden by downsampling).
        if (avoid != nullptr && avoid->size() > t && !(*avoid)[t].empty()) {
            coarse.avoid[t].assign(cells, 0);
            const std::vector<std::uint8_t>& fine_mask = (*avoid)[t];
            for (int row = 0; row < tile.height; ++row) {
                for (int col = 0; col < tile.width; ++col) {
                    const std::size_t fine = tile.index(col, row);
                    if (fine >= fine_mask.size() || fine_mask[fine] == 0) {
                        continue;
                    }
                    const int brow = (tile.y0 + row) / factor - ty0;
                    const int bcol = (tile.x0 + col) / factor - tx0;
                    if (brow >= 0 && brow < bh && bcol >= 0 && bcol < bw) {
                        coarse.avoid[t][static_cast<std::size_t>(brow) *
                                            static_cast<std::size_t>(bw) +
                                        static_cast<std::size_t>(bcol)] = 1;
                    }
                }
            }
            coarse.field.avoid[t] = coarse.avoid[t].data();
        }
    }
    finalize(coarse.field);
}

/// Upsample `source` labels (an octave coarser) onto `dest` by nearest block; a
/// fine cell keeps the source label only where that tile covers it, otherwise it
/// falls back to its own argmax.
std::vector<std::int32_t> seed_from(const std::vector<std::int32_t>& source, int source_width,
                                    const Field& dest, int ratio) {
    const int step = std::max(1, ratio);
    std::vector<std::int32_t> labels(dest.cells(), -1);
    for (int row = 0; row < dest.height; ++row) {
        for (int col = 0; col < dest.width; ++col) {
            const std::size_t cell =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(dest.width) +
                static_cast<std::size_t>(col);
            const std::size_t srow = static_cast<std::size_t>(row / step);
            const std::size_t scol = static_cast<std::size_t>(col / step);
            const int label =
                source[srow * static_cast<std::size_t>(source_width) + scol];
            if (covers(dest, label, cell)) {
                labels[cell] = label;
            } else {
                labels[cell] = argmax_at(dest, cell);
            }
        }
    }
    return labels;
}

/// Attach the optional per-tile deviant masks to a field. A mask whose size
/// does not match its tile is ignored rather than read out of bounds.
void attach_avoid(Field& field, const std::vector<OrthoTile>& tiles,
                  const SeamOptions& options) {
    if (options.avoid == nullptr || options.avoid->size() != tiles.size()) {
        return;
    }
    field.avoid.assign(tiles.size(), nullptr);
    for (std::size_t t = 0; t < tiles.size(); ++t) {
        const std::vector<std::uint8_t>& mask = (*options.avoid)[t];
        const std::size_t cells = static_cast<std::size_t>(tiles[t].width) *
                                  static_cast<std::size_t>(tiles[t].height);
        if (!mask.empty() && mask.size() == cells) {
            field.avoid[t] = mask.data();
        }
    }
}

/// A full-resolution Field viewing the tiles directly (no copy).
Field fine_field(const std::vector<OrthoTile>& tiles, int width, int height) {
    Field field;
    field.width = width;
    field.height = height;
    const std::size_t tile_count = tiles.size();
    field.score.resize(tile_count);
    field.rgb.resize(tile_count);
    field.valid.resize(tile_count);
    field.x0.resize(tile_count);
    field.y0.resize(tile_count);
    field.tile_width.resize(tile_count);
    field.tile_height.resize(tile_count);
    field.stride.resize(tile_count);
    for (std::size_t t = 0; t < tile_count; ++t) {
        const OrthoTile& tile = tiles[t];
        field.x0[t] = tile.x0;
        field.y0[t] = tile.y0;
        field.tile_width[t] = tile.width;
        field.tile_height[t] = tile.height;
        field.stride[t] = tile.width;
        field.score[t] = tile.score.data();
        field.rgb[t] = tile.rgb.data();
        field.valid[t] = tile.valid.data();
    }
    finalize(field);
    return field;
}

/// The active window for the full-resolution ghosting expansion: the bounding
/// box of every marked, valid mask cell grown by `margin`, plus every label
/// that covers at least one cell inside it (the labels a ghost region can be
/// moved TO). Returns false when no mask cell is set -- there is nothing to
/// remove, so the expansion is skipped entirely and an untouched ghosting run
/// costs exactly what a plain run does.
///
/// One window around all marks rather than one graph per component keeps the
/// code and the memory estimate simple; marks are normally a few small moving
/// objects. A scattered mask can widen the box, but never beyond the grid, and
/// the caller's cycle cap bounds the work.
bool ghost_expansion_window(const Field& field, const std::vector<OrthoTile>& tiles, int margin,
                            Region& region, std::vector<int>& alphas) {
    int min_col = field.width;
    int min_row = field.height;
    int max_col = -1;
    int max_row = -1;
    for (std::size_t t = 0; t < tiles.size(); ++t) {
        if (t >= field.avoid.size() || field.avoid[t] == nullptr) {
            continue;
        }
        const OrthoTile& tile = tiles[t];
        for (int row = tile.y0; row < tile.y0 + tile.height; ++row) {
            for (int col = tile.x0; col < tile.x0 + tile.width; ++col) {
                const std::size_t b = tile.index(col, row);
                if (field.avoid[t][b] == 0 || tile.valid[b] == 0) {
                    continue;
                }
                min_col = std::min(min_col, col);
                max_col = std::max(max_col, col);
                min_row = std::min(min_row, row);
                max_row = std::max(max_row, row);
            }
        }
    }
    if (max_col < min_col || max_row < min_row) {
        return false;  // no cell is marked
    }
    region.x0 = std::max(0, min_col - margin);
    region.y0 = std::max(0, min_row - margin);
    region.x1 = std::min(field.width, max_col + margin + 1);
    region.y1 = std::min(field.height, max_row + margin + 1);
    if (region.x1 <= region.x0 || region.y1 <= region.y0) {
        return false;
    }
    // Every camera holding at least one valid cell inside the window is a
    // candidate target; one that cannot improve any inside cell cannot help.
    for (std::size_t t = 0; t < tiles.size(); ++t) {
        const OrthoTile& tile = tiles[t];
        if (tile.empty()) {
            continue;
        }
        const int c0 = std::max(region.x0, tile.x0);
        const int r0 = std::max(region.y0, tile.y0);
        const int c1 = std::min(region.x1, tile.x0 + tile.width);
        const int r1 = std::min(region.y1, tile.y0 + tile.height);
        bool covers = false;
        for (int row = r0; row < r1 && !covers; ++row) {
            for (int col = c0; col < c1; ++col) {
                if (tile.valid[tile.index(col, row)] != 0) {
                    covers = true;
                    break;
                }
            }
        }
        if (covers) {
            alphas.push_back(static_cast<int>(t));
        }
    }
    return !alphas.empty();
}

}  // namespace

std::vector<std::int32_t> assign_seams(int width, int height,
                                       const std::vector<OrthoTile>& tiles,
                                       const SeamOptions& options,
                                       const std::vector<std::int32_t>* pinned_labels,
                                       const std::vector<std::uint8_t>* pinned) {
    const std::size_t cells =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<std::int32_t> labels(cells, -1);
    if (tiles.empty()) {
        return labels;
    }

    Field fine = fine_field(tiles, width, height);
    attach_avoid(fine, tiles, options);

    // A block-wise caller pins the margin cells already decided by an earlier
    // block. A pin only survives when the pinned tile actually covers the cell;
    // a pinned camera that this block's tile set does not hold (or that is
    // invalid there) is ignored, and the cell is optimised normally.
    std::vector<std::uint8_t> pin_mask;
    std::vector<std::int32_t> pin_values;
    const bool have_pins = pinned != nullptr && pinned_labels != nullptr &&
                           pinned->size() == cells && pinned_labels->size() == cells;
    if (have_pins) {
        pin_mask.assign(cells, 0);
        pin_values.assign(cells, -1);
        for (std::size_t cell = 0; cell < cells; ++cell) {
            const std::int32_t label = (*pinned_labels)[cell];
            if ((*pinned)[cell] == 0 || !covers(fine, label, cell)) {
                continue;
            }
            pin_mask[cell] = 1;
            pin_values[cell] = label;
        }
    }
    const bool any_pin = have_pins && std::any_of(pin_mask.begin(), pin_mask.end(),
                                                  [](std::uint8_t v) { return v != 0; });
    if (any_pin) {
        fine.pinned = pin_mask.data();
        fine.pinned_label = pin_values.data();
    }

    if (options.iterations <= 0) {
        labels = argmax_labels(fine);  // the plain per-cell argmax, no smoothing
        if (any_pin) {
            for (std::size_t cell = 0; cell < cells; ++cell) {
                if (pin_mask[cell] != 0) {
                    labels[cell] = pin_values[cell];
                }
            }
        }
        return labels;
    }

    // -- multi-scale: alpha-expand coarsest first, upsample, refine -----------
    std::vector<std::int32_t> previous;
    int previous_width = 0;
    const int levels = std::max(0, options.coarse_levels);
    for (int level = levels; level >= 1; --level) {
        Coarse coarse;
        build_coarse(coarse, tiles, width, height, 1 << level, options.avoid);
        std::vector<std::int32_t> level_labels =
            previous.empty() ? argmax_labels(coarse.field)
                             : seed_from(previous, previous_width, coarse.field, 2);
        alpha_expansion(coarse.field, level_labels, options, options.expansion_cycles);
        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            icm_sweep(coarse.field, level_labels, options);
        }
        previous = std::move(level_labels);
        previous_width = coarse.field.width;
    }

    labels = previous.empty() ? argmax_labels(fine)
                              : seed_from(previous, previous_width, fine, 2);
    // Pinned cells start from their decided label and never leave it: the ICM
    // sweep skips them (Field::pinned) and no coarse level can override them.
    if (any_pin) {
        for (std::size_t cell = 0; cell < cells; ++cell) {
            if (pin_mask[cell] != 0) {
                labels[cell] = pin_values[cell];
            }
        }
    }
    // Ghosting needs a full-resolution alpha-expansion pass. The avoid cost is
    // a per-cell unary term, but ICM can only flip a cell when its immediate
    // seam penalty falls below that saving; inside a large object every cell is
    // surrounded by object cells whose colour differs from the replacement, so
    // a single-cell move always looks expensive and the object would survive.
    // Alpha expansion flips the whole region at once, where the same seam term
    // is paid only on the object's perimeter.
    //
    // The pass is bounded twice so it cannot dominate the run: it is skipped
    // when nothing is marked, it builds a graph only over the marked cells'
    // bounding box grown by kGhostExpansionMargin (everything outside keeps the
    // coarse labelling), and it is capped at kGhostExpansionCycles cycles.
    if (options.avoid != nullptr) {
        constexpr int kGhostExpansionMargin = 32;  // cells of slack around a mark
        constexpr int kGhostExpansionCycles = 2;   // hard cap, independent of options
        Region region;
        std::vector<int> alphas;
        if (ghost_expansion_window(fine, tiles, kGhostExpansionMargin, region, alphas)) {
            const int cycles =
                std::min(std::max(1, options.expansion_cycles), kGhostExpansionCycles);
            alpha_expansion(fine, labels, options, cycles, &region, &alphas);
        }
    }
    // The coarse levels carry the global structure; from the upsampled labels
    // only the cheap ICM polish runs at full resolution.
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        icm_sweep(fine, labels, options);
    }
    return labels;
}

double energy(int width, int height, const std::vector<OrthoTile>& tiles,
              const std::vector<std::int32_t>& labels, const SeamOptions& options) {
    Field field = fine_field(tiles, width, height);
    attach_avoid(field, tiles, options);
    return total_energy(field, labels, options);
}

}  // namespace alphashape::ortho
