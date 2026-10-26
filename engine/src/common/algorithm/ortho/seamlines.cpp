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
    /// Tile crop boxes and buffer row strides, in FIELD cells (a coarse level
    /// scales them down with its own grid).
    std::vector<int> x0;
    std::vector<int> y0;
    std::vector<int> tile_width;
    std::vector<int> tile_height;
    std::vector<int> stride;
    std::vector<double> best;
    std::vector<std::uint8_t> any_valid;
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

/// True when `label` covers `cell`.
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
    return options.data_weight * std::max(0.0, shortfall);
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

/// The score-argmax label at one cell, or -1 when nothing covers it.
int argmax_at(const Field& field, std::size_t cell) {
    const int col = static_cast<int>(cell % static_cast<std::size_t>(field.width));
    const int row = static_cast<int>(cell / static_cast<std::size_t>(field.width));
    int best_label = -1;
    double best = 0.0;
    for (std::size_t t = 0; t < field.tiles(); ++t) {
        if (!field.valid_at(t, col, row)) {
            continue;
        }
        const double score = static_cast<double>(field.score_at(t, col, row));
        if (best_label < 0 || score > best) {
            best = score;
            best_label = static_cast<int>(t);
        }
    }
    return best_label;
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

/// One alpha-expansion move: the optimal set of cells to relabel as `alpha`,
/// found by a min-cut (Boykov, Veksler & Zabih 2001). The seam term is a metric
/// (a sum of absolute colour differences plus a constant), so every pairwise
/// term is submodular and the move is exact. Cells already carrying `alpha` are
/// fixed on the switch side.
bool expand_alpha(const Field& field, std::vector<std::int32_t>& labels, int alpha,
                  const SeamOptions& options) {
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

    // Graph node per cell that is not already `alpha`; source side = switch.
    std::vector<int> node(cells, -1);
    int count = 0;
    for (std::size_t cell = 0; cell < cells; ++cell) {
        if (labels[cell] != alpha) {
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
        flow.add_edge(p, sink, to_alpha);                          // p switches
        flow.add_edge(source, p, data_cost(field, labels[cell], cell, options));  // p keeps
    }

    // Exact pairwise decomposition of E(x_p, x_q), where x = 1 means "switch to
    // alpha". With a = E(1,0) = e(alpha,lq), b = E(0,1) = e(lp,alpha) and
    // E(0,0) = e(lp,lq), E(1,1) = 0, the polynomial
    //
    //     E = (E00 - a) * y_p + a * y_q + (b + a - E00) * x_q * y_p,
    //     y = 1 - x ("keeps"),
    //
    // reproduces all four values exactly. The metric property
    // E00 + E11 <= E(1,0) + E(0,1) makes b + a - E00 >= 0, so the move is
    // submodular. The q -> p edge is cut when q switches and p keeps.
    auto add_pair = [&](std::size_t pc, std::size_t qc) {
        const int np = node[pc];
        const int nq = node[qc];
        if (np < 0 && nq < 0) {
            return;  // both already alpha: no cut
        }
        const int lp = labels[pc];
        const int lq = labels[qc];
        const double e10 = seam_cost(field, alpha, lq, pc, qc, options);
        const double e01 = seam_cost(field, lp, alpha, pc, qc, options);
        if (np >= 0 && nq >= 0) {
            const double e00 = seam_cost(field, lp, lq, pc, qc, options);
            // p keeps: (E00 - e10) * y_p; a negative capacity is shifted onto
            // p -> t and the constant is dropped (it cannot change the argmin).
            if (e00 >= e10) {
                flow.add_edge(source, np, e00 - e10);
            } else {
                flow.add_edge(np, sink, e10 - e00);
            }
            flow.add_edge(source, nq, e10);                 // q keeps
            flow.add_edge(nq, np, e01 + e10 - e00);         // q switches, p keeps
        } else if (np >= 0) {
            flow.add_edge(source, np, e01);  // q is alpha already: p keeps costs e01
        } else {
            flow.add_edge(source, nq, e10);  // p is alpha already: q keeps costs e10
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
    bool changed = false;
    for (std::size_t cell = 0; cell < cells; ++cell) {
        const int p = node[cell];
        if (p >= 0 && side[static_cast<std::size_t>(p)] != 0) {
            labels[cell] = alpha;
            changed = true;
        }
    }
    return changed;
}

/// Sweep every label as the expansion target until a full cycle changes nothing.
void alpha_expansion(const Field& field, std::vector<std::int32_t>& labels,
                     const SeamOptions& options, int max_cycles) {
    const int labels_count = static_cast<int>(field.tiles());
    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        bool changed = false;
        for (int alpha = 0; alpha < labels_count; ++alpha) {
            changed = expand_alpha(field, labels, alpha, options) || changed;
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
            const std::size_t cell = list[static_cast<std::size_t>(i)];
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
    Field field;
};

void build_coarse(Coarse& coarse, const std::vector<OrthoTile>& tiles, int width, int height,
                  int factor) {
    const int cw = (width + factor - 1) / factor;
    const int ch = (height + factor - 1) / factor;
    const std::size_t tile_count = tiles.size();
    coarse.rgb.resize(tile_count);
    coarse.score.resize(tile_count);
    coarse.valid.resize(tile_count);

    coarse.field.width = cw;
    coarse.field.height = ch;
    coarse.field.score.resize(tile_count);
    coarse.field.rgb.resize(tile_count);
    coarse.field.valid.resize(tile_count);
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

}  // namespace

std::vector<std::int32_t> assign_seams(int width, int height,
                                       const std::vector<OrthoTile>& tiles,
                                       const SeamOptions& options) {
    const std::size_t cells =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    std::vector<std::int32_t> labels(cells, -1);
    if (tiles.empty()) {
        return labels;
    }

    const Field fine = fine_field(tiles, width, height);
    if (options.iterations <= 0) {
        return argmax_labels(fine);  // the plain per-cell argmax, no smoothing
    }

    // -- multi-scale: alpha-expand coarsest first, upsample, refine -----------
    std::vector<std::int32_t> previous;
    int previous_width = 0;
    const int levels = std::max(0, options.coarse_levels);
    for (int level = levels; level >= 1; --level) {
        Coarse coarse;
        build_coarse(coarse, tiles, width, height, 1 << level);
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
    // The coarse levels carry the global structure; from the upsampled labels
    // only the cheap ICM polish runs at full resolution.
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        icm_sweep(fine, labels, options);
    }
    return labels;
}

double energy(int width, int height, const std::vector<OrthoTile>& tiles,
              const std::vector<std::int32_t>& labels, const SeamOptions& options) {
    const Field field = fine_field(tiles, width, height);
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
            if (col + 1 < width) {
                const int q_label = labels[cell + 1];
                if (q_label != label) {
                    total += seam_cost_at(field, label, q_label, col, row, col + 1, row,
                                          options);
                }
            }
            if (row + 1 < height) {
                const int q_label = labels[cell + static_cast<std::size_t>(width)];
                if (q_label != label) {
                    total += seam_cost_at(field, label, q_label, col, row, col, row + 1,
                                          options);
                }
            }
        }
    }
    return total;
}

}  // namespace alphashape::ortho
