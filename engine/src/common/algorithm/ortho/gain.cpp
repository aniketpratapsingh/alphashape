#include "common/algorithm/ortho/gain.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>

namespace alphashape::ortho {
namespace {

/// Reciprocal of the stored-value scale: Image holds 0..255.
constexpr double kScale = 1.0 / 255.0;

/// The stride OverlapAccumulator packs pair keys with. Large enough for any
/// flight (2^21 cameras) and keeps the product inside int64.
constexpr std::int64_t kKeyStride = 1 << 21;

std::int64_t pair_key(int a, int b) {
    if (a > b) {
        std::swap(a, b);
    }
    return static_cast<std::int64_t>(a) * kKeyStride + static_cast<std::int64_t>(b);
}

}  // namespace

void OverlapAccumulator::add(int a, int b, const std::uint8_t rgb_a[3],
                             const std::uint8_t rgb_b[3]) {
    if (a < 0 || b < 0 || a == b) {
        return;
    }
    if (a > b) {
        std::swap(a, b);
        std::swap(rgb_a, rgb_b);
    }
    Sum& sum = sums_[pair_key(a, b)];
    sum.count += 1.0;
    for (int c = 0; c < 3; ++c) {
        sum.totals[static_cast<std::size_t>(c)] += static_cast<double>(rgb_a[c]);
        sum.totals[static_cast<std::size_t>(3 + c)] += static_cast<double>(rgb_b[c]);
    }
}

void OverlapAccumulator::merge(const OverlapAccumulator& other) {
    for (const auto& [key, sum] : other.sums_) {
        Sum& mine = sums_[key];
        mine.count += sum.count;
        for (std::size_t i = 0; i < mine.totals.size(); ++i) {
            mine.totals[i] += sum.totals[i];
        }
    }
}

std::vector<OverlapPair> OverlapAccumulator::finish() const {
    std::vector<OverlapPair> pairs;
    pairs.reserve(sums_.size());
    for (const auto& [key, sum] : sums_) {  // std::map iterates sorted: deterministic
        if (sum.count <= 0.0) {
            continue;
        }
        OverlapPair pair;
        pair.a = static_cast<int>(key / kKeyStride);
        pair.b = static_cast<int>(key % kKeyStride);
        pair.count = sum.count;
        for (int c = 0; c < 3; ++c) {
            pair.mean_a[static_cast<std::size_t>(c)] =
                sum.totals[static_cast<std::size_t>(c)] / sum.count * kScale;
            pair.mean_b[static_cast<std::size_t>(c)] =
                sum.totals[static_cast<std::size_t>(3 + c)] / sum.count * kScale;
        }
        pairs.push_back(pair);
    }
    return pairs;
}

std::vector<std::array<double, 3>> solve_gains(int camera_count,
                                               const std::vector<OverlapPair>& pairs,
                                               const GainOptions& options) {
    std::vector<std::array<double, 3>> gains(
        static_cast<std::size_t>(std::max(camera_count, 0)),
        std::array<double, 3>{1.0, 1.0, 1.0});
    if (camera_count <= 0 || pairs.empty()) {
        return gains;
    }

    const double inv_n2 = 1.0 / (options.sigma_n * options.sigma_n);
    const double prior = 1.0 / (options.sigma_g * options.sigma_g);
    const int n = camera_count;

    for (int channel = 0; channel < 3; ++channel) {
        // Build the symmetric system A g = b. A is sparse (a camera overlaps
        // only its neighbours), so it is assembled as a dense matrix here: a
        // flight of a few hundred cameras is a few hundred squared doubles,
        // negligible against the mosaic, and Cholesky on it is simpler and more
        // obviously correct than a sparse solver would be.
        std::vector<double> A(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
        std::vector<double> b(static_cast<std::size_t>(n), prior);
        const auto at = [n](std::vector<double>& m, int r, int c) -> double& {
            return m[static_cast<std::size_t>(r) * static_cast<std::size_t>(n) +
                     static_cast<std::size_t>(c)];
        };

        for (const OverlapPair& pair : pairs) {
            if (pair.a < 0 || pair.b < 0 || pair.a >= n || pair.b >= n) {
                continue;
            }
            const double count = pair.count;
            const double ia = pair.mean_a[static_cast<std::size_t>(channel)];
            const double ib = pair.mean_b[static_cast<std::size_t>(channel)];
            // Diagonal: camera a's own term, camera b's own term symmetric.
            at(A, pair.a, pair.a) += count * ia * ia * inv_n2;
            at(A, pair.b, pair.b) += count * ib * ib * inv_n2;
            // Off-diagonal: -count * ia * ib / sigma^2 (symmetric).
            const double off = -count * ia * ib * inv_n2;
            at(A, pair.a, pair.b) += off;
            at(A, pair.b, pair.a) += off;
        }
        for (int i = 0; i < n; ++i) {
            at(A, i, i) += prior;
        }

        // Cholesky: A = L L^T, solve L y = b then L^T x = y.
        std::vector<double> L(A.size(), 0.0);
        bool ok = true;
        for (int i = 0; i < n && ok; ++i) {
            for (int j = 0; j <= i; ++j) {
                double sum = at(A, i, j);
                for (int k = 0; k < j; ++k) {
                    sum -= at(L, i, k) * at(L, j, k);
                }
                if (i == j) {
                    if (sum <= 1e-15) {  // not positive definite: fall back to 1
                        ok = false;
                        break;
                    }
                    at(L, i, i) = std::sqrt(sum);
                } else {
                    at(L, i, j) = sum / at(L, j, j);
                }
            }
        }
        if (!ok) {
            continue;  // leave this channel's gains at 1
        }
        std::vector<double> y(static_cast<std::size_t>(n), 0.0);
        for (int i = 0; i < n; ++i) {
            double sum = b[static_cast<std::size_t>(i)];
            for (int k = 0; k < i; ++k) {
                sum -= at(L, i, k) * y[static_cast<std::size_t>(k)];
            }
            y[static_cast<std::size_t>(i)] = sum / at(L, i, i);
        }
        std::vector<double> x(static_cast<std::size_t>(n), 0.0);
        for (int i = n - 1; i >= 0; --i) {
            double sum = y[static_cast<std::size_t>(i)];
            for (int k = i + 1; k < n; ++k) {
                sum -= at(L, k, i) * x[static_cast<std::size_t>(k)];
            }
            x[static_cast<std::size_t>(i)] = sum / at(L, i, i);
        }
        for (int i = 0; i < n; ++i) {
            gains[static_cast<std::size_t>(i)][static_cast<std::size_t>(channel)] =
                std::clamp(x[static_cast<std::size_t>(i)], options.min_gain, options.max_gain);
        }

        // The absolute gain scale is unidentifiable from the data term alone —
        // multiplying every gain by a constant is the same solution — and the
        // prior is deliberately weak next to the (count-weighted) data term, so
        // on a flight whose photographs genuinely differ it lets the whole vector
        // drift dark or bright. Brown & Lowe fix the free scale by normalising the
        // gain vector: here, to a mean of 1, so the mosaic keeps the overall
        // brightness the photographs had instead of being uniformly darkened or
        // lightened. The RATIOS — which are what remove a seam — are unchanged.
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += gains[static_cast<std::size_t>(i)][static_cast<std::size_t>(channel)];
        }
        const double mean = sum / static_cast<double>(n);
        if (mean > 1e-9) {
            for (int i = 0; i < n; ++i) {
                double& gain = gains[static_cast<std::size_t>(i)][static_cast<std::size_t>(channel)];
                gain = std::clamp(gain / mean, options.min_gain, options.max_gain);
            }
        }
    }
    return gains;
}

void apply_gain(OrthoTile& tile, const std::array<double, 3>& gain) {
    if (tile.rgb.empty()) {
        return;
    }
    const std::size_t cells = tile.valid.size();
    for (std::size_t cell = 0; cell < cells; ++cell) {
        if (tile.valid[cell] == 0) {
            continue;
        }
        for (int c = 0; c < 3; ++c) {
            const double value =
                static_cast<double>(tile.rgb[cell * 3 + static_cast<std::size_t>(c)]) *
                gain[static_cast<std::size_t>(c)];
            tile.rgb[cell * 3 + static_cast<std::size_t>(c)] =
                static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0, 255.0)));
        }
    }
}

}  // namespace alphashape::ortho
