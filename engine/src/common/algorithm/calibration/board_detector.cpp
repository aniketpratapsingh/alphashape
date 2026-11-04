#include "common/algorithm/calibration/board_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <queue>
#include <set>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"

namespace alphashape {
namespace {

constexpr double kPi = std::numbers::pi_v<double>;
constexpr int kRingSamples = 32;

/// One verified-or-candidate X junction. Coordinates are in FloatImage's
/// integer-centred convention; the API adds half a pixel at the end.
struct Candidate {
    double x = 0.0;
    double y = 0.0;
    double response = 0.0;
    double dir0 = 0.0;  ///< Edge direction in [0, pi).
    double dir1 = 0.0;
};

double wrap_pi(double angle) {
    angle = std::fmod(angle, kPi);
    if (angle < 0.0) {
        angle += kPi;
    }
    return angle;
}

/// Angle between two unoriented lines, in [0, pi/2].
double angle_diff_pi(double a, double b) {
    const double d = wrap_pi(a - b);
    return d > 0.5 * kPi ? kPi - d : d;
}

/// Separable Gaussian with clamped edges. `sigma` is in pixels.
FloatImage gaussian_blur(const FloatImage& in, double sigma) {
    const int w = in.width();
    const int h = in.height();
    if (w <= 0 || h <= 0) {
        return FloatImage();
    }
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    std::vector<double> kernel(static_cast<std::size_t>(2 * radius + 1), 0.0);
    double sum = 0.0;
    for (int k = -radius; k <= radius; ++k) {
        const double v = std::exp(-0.5 * static_cast<double>(k * k) / (sigma * sigma));
        kernel[static_cast<std::size_t>(k + radius)] = v;
        sum += v;
    }
    for (double& v : kernel) {
        v /= sum;
    }
    FloatImage tmp(w, h, 0.0F);
    FloatImage out(w, h, 0.0F);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double acc = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                acc += kernel[static_cast<std::size_t>(k + radius)] *
                       static_cast<double>(in.at_clamped(x + k, y));
            }
            tmp.at(x, y) = static_cast<float>(acc);
        }
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double acc = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                acc += kernel[static_cast<std::size_t>(k + radius)] *
                       static_cast<double>(tmp.at_clamped(x, y + k));
            }
            out.at(x, y) = static_cast<float>(acc);
        }
    }
    return out;
}

/// Response r = -det(H), clamped at 0. At an X junction the Hessian has
/// eigenvalues of opposite sign, so the determinant is negative and r > 0.
FloatImage hessian_response(const FloatImage& smooth) {
    const int w = smooth.width();
    const int h = smooth.height();
    FloatImage out(w, h, 0.0F);
    if (w < 3 || h < 3) {
        return out;
    }
    for (int y = 1; y + 1 < h; ++y) {
        for (int x = 1; x + 1 < w; ++x) {
            const double c = static_cast<double>(smooth.at(x, y));
            const double ixx = static_cast<double>(smooth.at(x + 1, y)) - 2.0 * c +
                               static_cast<double>(smooth.at(x - 1, y));
            const double iyy = static_cast<double>(smooth.at(x, y + 1)) - 2.0 * c +
                               static_cast<double>(smooth.at(x, y - 1));
            const double ixy = 0.25 * (static_cast<double>(smooth.at(x + 1, y + 1)) -
                                       static_cast<double>(smooth.at(x + 1, y - 1)) -
                                       static_cast<double>(smooth.at(x - 1, y + 1)) +
                                       static_cast<double>(smooth.at(x - 1, y - 1)));
            const double det = ixx * iyy - ixy * ixy;
            if (det < 0.0) {
                out.at(x, y) = static_cast<float>(-det);
            }
        }
    }
    return out;
}

/// Local maxima of the response. A discrete peak can sit exactly between two
/// pixel centres (a half-pixel symmetry), where the two samples have equal
/// response. Requiring a strictly greater value than every neighbour would
/// then reject both, so equal-valued neighbours are accepted as a plateau as
/// long as at least one neighbour is lower (which excludes flat regions).
std::vector<Candidate> local_maxima(const FloatImage& response) {
    const int w = response.width();
    const int h = response.height();
    std::vector<Candidate> out;
    for (int y = 1; y + 1 < h; ++y) {
        for (int x = 1; x + 1 < w; ++x) {
            const float v = response.at(x, y);
            if (!(v > 0.0F)) {
                continue;
            }
            bool peak = true;
            bool has_lower = false;
            for (int dy = -1; dy <= 1 && peak; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    const float neighbour = response.at(x + dx, y + dy);
                    if (neighbour > v) {
                        peak = false;
                        break;
                    }
                    if (neighbour < v) {
                        has_lower = true;
                    }
                }
            }
            if (peak && has_lower) {
                Candidate candidate;
                candidate.x = static_cast<double>(x);
                candidate.y = static_cast<double>(y);
                candidate.response = static_cast<double>(v);
                out.push_back(candidate);
            }
        }
    }
    return out;
}

/// Greedy non-maximum suppression, stronger response first, on a cell grid.
std::vector<Candidate> suppress_candidates(const std::vector<Candidate>& candidates,
                                           double min_distance) {
    std::vector<Candidate> accepted;
    if (candidates.empty() || !(min_distance > 0.0)) {
        return accepted;
    }
    const double cell = std::max(1.0, min_distance);
    int max_x = 0;
    int max_y = 0;
    for (const Candidate& candidate : candidates) {
        max_x = std::max(max_x, static_cast<int>(std::floor(candidate.x / cell)) + 1);
        max_y = std::max(max_y, static_cast<int>(std::floor(candidate.y / cell)) + 1);
    }
    const int nx = max_x + 2;
    const int ny = max_y + 2;
    std::vector<std::vector<int>> buckets(static_cast<std::size_t>(nx * ny));
    const double min_distance2 = min_distance * min_distance;
    for (const Candidate& candidate : candidates) {
        int cx = 0;
        int cy = 0;
        if (candidate.x >= 0.0 && candidate.y >= 0.0) {
            cx = static_cast<int>(std::floor(candidate.x / cell));
            cy = static_cast<int>(std::floor(candidate.y / cell));
        }
        bool suppressed = false;
        for (int gy = std::max(0, cy - 1); gy <= std::min(ny - 1, cy + 1) && !suppressed; ++gy) {
            for (int gx = std::max(0, cx - 1); gx <= std::min(nx - 1, cx + 1) && !suppressed; ++gx) {
                for (int index : buckets[static_cast<std::size_t>(gy * nx + gx)]) {
                    const Candidate& other = accepted[static_cast<std::size_t>(index)];
                    const double dx = other.x - candidate.x;
                    const double dy = other.y - candidate.y;
                    if (dx * dx + dy * dy < min_distance2) {
                        suppressed = true;
                        break;
                    }
                }
            }
        }
        if (!suppressed) {
            buckets[static_cast<std::size_t>(cy * nx + cx)].push_back(
                static_cast<int>(accepted.size()));
            accepted.push_back(candidate);
        }
    }
    return accepted;
}

/// Adaptive response cut at `fraction` of the 99th percentile, then NMS.
std::vector<Candidate> select_candidates(const FloatImage& response, double fraction,
                                         double min_distance) {
    std::vector<Candidate> candidates = local_maxima(response);
    if (candidates.size() < 2) {
        return candidates;
    }
    std::vector<double> values;
    values.reserve(candidates.size());
    for (const Candidate& candidate : candidates) {
        values.push_back(candidate.response);
    }
    const std::size_t index =
        static_cast<std::size_t>(std::floor(0.99 * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index),
                     values.end());
    const double p99 = values[index];
    if (!(p99 > 0.0)) {
        return {};
    }
    const double cut = fraction * p99;
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                    [cut](const Candidate& candidate) {
                                        return candidate.response < cut;
                                    }),
                     candidates.end());
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.response > b.response;
              });
    return suppress_candidates(candidates, min_distance);
}

/// Circular mean of two directions that are known to be close modulo pi.
double circular_mean_pair(double a, double b) {
    double d = b - a;
    if (d > 0.5 * kPi) {
        d -= kPi;
    } else if (d < -0.5 * kPi) {
        d += kPi;
    }
    return wrap_pi(a + 0.5 * d);
}

/// Verify an X junction by sampling a ring of radius `radius`. A chessboard
/// corner crosses the mean intensity exactly four times around the ring, with
/// roughly 90 degrees between crossings. Returns the two edge directions
/// (modulo pi) on success.
std::optional<std::pair<double, double>> ring_directions(const FloatImage& smooth,
                                                         double cx, double cy,
                                                         double radius) {
    std::vector<double> values(static_cast<std::size_t>(kRingSamples), 0.0);
    double mean = 0.0;
    for (int i = 0; i < kRingSamples; ++i) {
        const double angle = 2.0 * kPi * static_cast<double>(i) /
                             static_cast<double>(kRingSamples);
        values[static_cast<std::size_t>(i)] =
            static_cast<double>(smooth.sample(static_cast<float>(cx + radius * std::cos(angle)),
                                              static_cast<float>(cy + radius * std::sin(angle))));
        mean += values[static_cast<std::size_t>(i)];
    }
    mean /= static_cast<double>(kRingSamples);
    std::vector<int> changes;
    for (int i = 0; i < kRingSamples; ++i) {
        const bool here = values[static_cast<std::size_t>(i)] >= mean;
        const bool next = values[static_cast<std::size_t>((i + 1) % kRingSamples)] >= mean;
        if (here != next) {
            changes.push_back(i);
        }
    }
    if (changes.size() != 4) {
        return std::nullopt;
    }
    const int quarter = kRingSamples / 4;
    for (std::size_t k = 0; k < changes.size(); ++k) {
        const int gap = (changes[(k + 1) % changes.size()] - changes[k] + kRingSamples) %
                        kRingSamples;
        if (gap < quarter / 2 || gap > quarter + quarter / 2) {
            return std::nullopt;  // crossings not roughly 90 degrees apart
        }
    }
    std::vector<double> angles(static_cast<std::size_t>(4), 0.0);
    for (std::size_t k = 0; k < changes.size(); ++k) {
        const int index = changes[k];
        const double here =
            values[static_cast<std::size_t>(index)] - mean;
        const double next =
            values[static_cast<std::size_t>((index + 1) % kRingSamples)] - mean;
        // Linear interpolation of the mean crossing between the two samples:
        // 32 samples alone would quantise the edge direction to 11.25 degrees.
        double fraction = 0.5;
        const double span = here - next;
        if (std::abs(span) > 1e-12) {
            fraction = here / span;
        }
        fraction = std::clamp(fraction, 0.0, 1.0);
        angles[k] = wrap_pi((static_cast<double>(index) + fraction) * 2.0 * kPi /
                            static_cast<double>(kRingSamples));
    }
    const double a01 = angle_diff_pi(angles[0], angles[1]);
    const double a23 = angle_diff_pi(angles[2], angles[3]);
    const double a02 = angle_diff_pi(angles[0], angles[2]);
    const double a13 = angle_diff_pi(angles[1], angles[3]);
    const double a03 = angle_diff_pi(angles[0], angles[3]);
    const double a12 = angle_diff_pi(angles[1], angles[2]);
    const double cost_a = a01 + a23;
    const double cost_b = a02 + a13;
    const double cost_c = a03 + a12;
    if (cost_a <= cost_b && cost_a <= cost_c) {
        return std::make_pair(circular_mean_pair(angles[0], angles[1]),
                              circular_mean_pair(angles[2], angles[3]));
    }
    if (cost_b <= cost_c) {
        return std::make_pair(circular_mean_pair(angles[0], angles[2]),
                              circular_mean_pair(angles[1], angles[3]));
    }
    return std::make_pair(circular_mean_pair(angles[0], angles[3]),
                          circular_mean_pair(angles[1], angles[2]));
}

/// True when both corners carry the same pair of edge directions, unordered.
bool edge_match(double a0, double a1, double b0, double b1, double tolerance) {
    const bool direct = angle_diff_pi(a0, b0) <= tolerance &&
                        angle_diff_pi(a1, b1) <= tolerance;
    const bool swapped = angle_diff_pi(a0, b1) <= tolerance &&
                         angle_diff_pi(a1, b0) <= tolerance;
    return direct || swapped;
}

/// Saddle-point refinement (Lucchese & Mitra 2001). Fits a quadratic to a
/// window around `point` and moves to where its gradient vanishes. Returns
/// nullopt when the surface is not a saddle or the step is implausibly large.
std::optional<Eigen::Vector2d> refine_saddle(const FloatImage& smooth,
                                             const Eigen::Vector2d& start,
                                             int radius) {
    Eigen::Vector2d point = start;
    for (int iteration = 0; iteration < 3; ++iteration) {
        Eigen::Matrix<double, 6, 6> normal = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> rhs = Eigen::Matrix<double, 6, 1>::Zero();
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                const double x = static_cast<double>(dx);
                const double y = static_cast<double>(dy);
                Eigen::Matrix<double, 6, 1> basis;
                basis << x * x, x * y, y * y, x, y, 1.0;
                const double value = static_cast<double>(smooth.sample(
                    static_cast<float>(point.x()) + static_cast<float>(x),
                    static_cast<float>(point.y()) + static_cast<float>(y)));
                normal.noalias() += basis * basis.transpose();
                rhs.noalias() += basis * value;
            }
        }
        const Eigen::Matrix<double, 6, 1> coefficient = normal.ldlt().solve(rhs);
        if (!coefficient.allFinite()) {
            return std::nullopt;
        }
        const double a = coefficient[0];
        const double b = coefficient[1];
        const double c = coefficient[2];
        const double d = coefficient[3];
        const double e = coefficient[4];
        const double det = 4.0 * a * c - b * b;
        if (!(det < 0.0)) {
            return std::nullopt;  // not a saddle
        }
        const double shift_x = ((2.0 * c) * (-d) - b * (-e)) / det;
        const double shift_y = ((2.0 * a) * (-e) - b * (-d)) / det;
        if (!std::isfinite(shift_x) || !std::isfinite(shift_y) ||
            std::hypot(shift_x, shift_y) > 1.5) {
            return std::nullopt;
        }
        point.x() += shift_x;
        point.y() += shift_y;
    }
    return point;
}

/// Ring-test the candidates and refine their positions to subpixel accuracy.
std::vector<Candidate> verify_candidates(const FloatImage& smooth,
                                         const std::vector<Candidate>& raw,
                                         double ring_radius, int refine_radius) {
    std::vector<Candidate> verified;
    verified.reserve(raw.size());
    for (const Candidate& candidate : raw) {
        const std::optional<std::pair<double, double>> directions =
            ring_directions(smooth, candidate.x, candidate.y, ring_radius);
        if (!directions) {
            continue;
        }
        Candidate corner = candidate;
        corner.dir0 = directions->first;
        corner.dir1 = directions->second;
        const std::optional<Eigen::Vector2d> refined =
            refine_saddle(smooth, Eigen::Vector2d(candidate.x, candidate.y), refine_radius);
        if (refined) {
            corner.x = refined->x();
            corner.y = refined->y();
        }
        verified.push_back(corner);
    }
    return verified;
}

struct LatticeLabel {
    int col = 0;
    int row = 0;
};

struct GrownLattice {
    std::vector<std::optional<LatticeLabel>> labels;
};

/// Nearest verified corner lying roughly along `direction` (mod pi) from the
/// seed. On success `step` is the measured offset.
bool find_seed_step(const std::vector<Candidate>& corners, int seed, double direction,
                    double tolerance, Eigen::Vector2d& step) {
    double best_distance = std::numeric_limits<double>::infinity();
    int best_index = -1;
    for (int i = 0; i < static_cast<int>(corners.size()); ++i) {
        if (i == seed) {
            continue;
        }
        const double dx = corners[static_cast<std::size_t>(i)].x - corners[static_cast<std::size_t>(seed)].x;
        const double dy = corners[static_cast<std::size_t>(i)].y - corners[static_cast<std::size_t>(seed)].y;
        const double distance = std::hypot(dx, dy);
        if (distance < 1e-6) {
            continue;
        }
        if (angle_diff_pi(std::atan2(dy, dx), direction) <= tolerance &&
            distance < best_distance) {
            best_distance = distance;
            best_index = i;
        }
    }
    if (best_index < 0) {
        return false;
    }
    step = Eigen::Vector2d(corners[static_cast<std::size_t>(best_index)].x -
                               corners[static_cast<std::size_t>(seed)].x,
                           corners[static_cast<std::size_t>(best_index)].y -
                               corners[static_cast<std::size_t>(seed)].y);
    return true;
}

/// Greedy, geometry-gated breadth-first lattice growth.
std::optional<GrownLattice> grow_lattice(const std::vector<Candidate>& corners, int seed,
                                         double angle_tolerance, double spacing_tolerance) {
    const std::size_t n = corners.size();
    GrownLattice lattice;
    lattice.labels.resize(n);
    std::vector<Eigen::Vector2d> e1(n, Eigen::Vector2d::Zero());
    std::vector<Eigen::Vector2d> e2(n, Eigen::Vector2d::Zero());
    lattice.labels[static_cast<std::size_t>(seed)] = LatticeLabel{0, 0};
    std::set<std::pair<int, int>> occupied;
    occupied.insert(std::make_pair(0, 0));
    Eigen::Vector2d first;
    Eigen::Vector2d second;
    if (!find_seed_step(corners, seed, corners[static_cast<std::size_t>(seed)].dir0,
                        angle_tolerance, first)) {
        return std::nullopt;
    }
    if (!find_seed_step(corners, seed, corners[static_cast<std::size_t>(seed)].dir1,
                        angle_tolerance, second)) {
        return std::nullopt;
    }
    if (!(first.norm() > 1e-6) || !(second.norm() > 1e-6)) {
        return std::nullopt;
    }
    // Name the more horizontal edge "col" and orient it rightwards, the other
    // "row" and orient it downwards. The label origin stays arbitrary for a
    // partial view, but its axes then follow the image, so lattice differences
    // are comparable across views.
    const bool first_horizontal = std::abs(first.x()) >= std::abs(second.x());
    Eigen::Vector2d seed_e1 = first_horizontal ? first : second;
    Eigen::Vector2d seed_e2 = first_horizontal ? second : first;
    if (seed_e1.x() < 0.0) {
        seed_e1 = -seed_e1;
    }
    if (seed_e2.y() < 0.0) {
        seed_e2 = -seed_e2;
    }
    e1[static_cast<std::size_t>(seed)] = seed_e1;
    e2[static_cast<std::size_t>(seed)] = seed_e2;
    std::queue<int> queue;
    queue.push(seed);
    while (!queue.empty()) {
        const int current = queue.front();
        queue.pop();
        const Eigen::Vector2d position(corners[static_cast<std::size_t>(current)].x,
                                       corners[static_cast<std::size_t>(current)].y);
        const LatticeLabel label = *lattice.labels[static_cast<std::size_t>(current)];
        for (int axis = 0; axis < 2; ++axis) {
            const Eigen::Vector2d step =
                axis == 0 ? e1[static_cast<std::size_t>(current)]
                          : e2[static_cast<std::size_t>(current)];
            const double length = step.norm();
            if (!(length > 1e-6)) {
                continue;
            }
            for (int sign = -1; sign <= 1; sign += 2) {
                const Eigen::Vector2d predicted =
                    position + static_cast<double>(sign) * step;
                int best_index = -1;
                double best_distance = 0.3 * length;
                for (int i = 0; i < static_cast<int>(n); ++i) {
                    if (i == current) {
                        continue;
                    }
                    const double distance =
                        std::hypot(corners[static_cast<std::size_t>(i)].x - predicted.x(),
                                   corners[static_cast<std::size_t>(i)].y - predicted.y());
                    if (distance < best_distance) {
                        best_distance = distance;
                        best_index = i;
                    }
                }
                if (best_index < 0) {
                    continue;
                }
                const Eigen::Vector2d measured(
                    corners[static_cast<std::size_t>(best_index)].x - position.x(),
                    corners[static_cast<std::size_t>(best_index)].y - position.y());
                const double spacing = measured.norm();
                if (std::abs(spacing - length) > spacing_tolerance * length) {
                    continue;
                }
                if (!edge_match(corners[static_cast<std::size_t>(current)].dir0,
                                corners[static_cast<std::size_t>(current)].dir1,
                                corners[static_cast<std::size_t>(best_index)].dir0,
                                corners[static_cast<std::size_t>(best_index)].dir1,
                                angle_tolerance)) {
                    continue;
                }
                const LatticeLabel wanted{label.col + (axis == 0 ? sign : 0),
                                          label.row + (axis == 1 ? sign : 0)};
                const std::pair<int, int> wanted_key{wanted.col, wanted.row};
                if (occupied.count(wanted_key) != 0) {
                    continue;  // a label is never assigned twice
                }
                if (lattice.labels[static_cast<std::size_t>(best_index)]) {
                    const LatticeLabel existing =
                        *lattice.labels[static_cast<std::size_t>(best_index)];
                    if (existing.col != wanted.col || existing.row != wanted.row) {
                        log(LogLevel::Debug)
                            << "board_detector: lattice conflict at (" << wanted.col
                            << "," << wanted.row << "); keeping (" << existing.col
                            << "," << existing.row << ")";
                    }
                    continue;
                }
                lattice.labels[static_cast<std::size_t>(best_index)] = wanted;
                occupied.insert(wanted_key);
                if (axis == 0) {
                    e1[static_cast<std::size_t>(best_index)] =
                        static_cast<double>(sign) * measured;
                    e2[static_cast<std::size_t>(best_index)] = e2[static_cast<std::size_t>(current)];
                } else {
                    e2[static_cast<std::size_t>(best_index)] =
                        static_cast<double>(sign) * measured;
                    e1[static_cast<std::size_t>(best_index)] = e1[static_cast<std::size_t>(current)];
                }
                queue.push(best_index);
            }
        }
    }
    return lattice;
}

struct LabelledCorner {
    int candidate_index = 0;
    int col = 0;
    int row = 0;
};

/// Labelled corner pixels keyed by (col, row).
using LabelGrid = std::map<std::pair<int, int>, Eigen::Vector2d>;

/// Mean pixel offset from a labelled corner to its col+1 (axis 0) or row+1
/// (axis 1) neighbour, averaged over every pair present.
Eigen::Vector2d mean_lattice_step(const LabelGrid& grid, int axis) {
    Eigen::Vector2d sum = Eigen::Vector2d::Zero();
    int count = 0;
    for (const auto& entry : grid) {
        const std::pair<int, int> next =
            axis == 0 ? std::make_pair(entry.first.first + 1, entry.first.second)
                      : std::make_pair(entry.first.first, entry.first.second + 1);
        const auto found = grid.find(next);
        if (found == grid.end()) {
            continue;
        }
        sum += found->second - entry.second;
        ++count;
    }
    if (count == 0) {
        return Eigen::Vector2d::Zero();
    }
    return sum / static_cast<double>(count);
}

/// Sample the blurred image at `point` when it lies inside the pixel extent.
std::optional<double> sample_inside(const FloatImage& image,
                                    const Eigen::Vector2d& point) {
    const double last_x = static_cast<double>(image.width() - 1);
    const double last_y = static_cast<double>(image.height() - 1);
    if (point.x() < 0.0 || point.y() < 0.0 || point.x() > last_x ||
        point.y() > last_y) {
        return std::nullopt;
    }
    return static_cast<double>(
        image.sample(static_cast<float>(point.x()), static_cast<float>(point.y())));
}

struct OrientationVote {
    bool decided = false;
    bool keep = false;
};

/// Decide whether the origin corner is the one whose outside-diagonal square
/// is dark. Returns undecided when neither diagonal pair is fully inside the
/// image. Only meaningful when the board is colour-asymmetric under 180 deg.
OrientationVote vote_origin_dark(const LabelGrid& grid, const FloatImage& smooth,
                                 int cols, int rows) {
    const Eigen::Vector2d origin = grid.at(std::make_pair(0, 0));
    const Eigen::Vector2d last = grid.at(std::make_pair(cols - 1, rows - 1));
    const Eigen::Vector2d u0 = grid.at(std::make_pair(1, 0)) - origin;
    const Eigen::Vector2d w0 = grid.at(std::make_pair(0, 1)) - origin;
    const Eigen::Vector2d u1 = last - grid.at(std::make_pair(cols - 2, rows - 1));
    const Eigen::Vector2d w1 = last - grid.at(std::make_pair(cols - 1, rows - 2));
    const std::optional<double> outside_origin =
        sample_inside(smooth, origin - 0.5 * (u0 + w0));
    const std::optional<double> outside_last =
        sample_inside(smooth, last + 0.5 * (u1 + w1));
    if (outside_origin && outside_last) {
        return OrientationVote{true, *outside_origin <= *outside_last};
    }
    // The outside square is off-image: the inside diagonal square across the
    // corner carries the same colour, so compare those instead.
    const std::optional<double> inside_origin =
        sample_inside(smooth, origin + 0.5 * (u0 + w0));
    const std::optional<double> inside_last =
        sample_inside(smooth, last - 0.5 * (u1 + w1));
    if (inside_origin && inside_last) {
        return OrientationVote{true, *inside_origin <= *inside_last};
    }
    return OrientationVote{};
}

/// Orient a view's labels: make the label basis right-handed in image
/// coordinates for every view, then resolve the complete-board 180 degree
/// ambiguity. Returns false when a complete label lies outside the spec grid.
bool canonicalise_labels(std::vector<LabelledCorner>& labelled,
                         const std::vector<Candidate>& corners, const FloatImage& smooth,
                         const BoardSpec& spec, bool complete) {
    LabelGrid grid;
    const auto refill = [&]() {
        grid.clear();
        for (const LabelledCorner& corner : labelled) {
            const Candidate& source =
                corners[static_cast<std::size_t>(corner.candidate_index)];
            grid.emplace(std::make_pair(corner.col, corner.row),
                         Eigen::Vector2d(source.x, source.y));
        }
    };
    refill();
    int max_col = 0;
    for (const LabelledCorner& corner : labelled) {
        max_col = std::max(max_col, corner.col);
    }
    const Eigen::Vector2d u = mean_lattice_step(grid, 0);
    const Eigen::Vector2d w = mean_lattice_step(grid, 1);
    if (u.x() * w.y() - u.y() * w.x() < 0.0) {
        // Image y grows downwards, so a left-handed label basis is mirrored.
        for (LabelledCorner& corner : labelled) {
            corner.col = max_col - corner.col;
        }
        refill();
    }
    if (complete) {
        bool keep = false;
        bool decided = false;
        if ((spec.cols + spec.rows) % 2 != 0) {
            const OrientationVote vote =
                vote_origin_dark(grid, smooth, spec.cols, spec.rows);
            decided = vote.decided;
            keep = vote.keep;
        }
        if (!decided) {
            keep = grid.at(std::make_pair(0, 0)).squaredNorm() <=
                   grid.at(std::make_pair(spec.cols - 1, spec.rows - 1)).squaredNorm();
        }
        if (!keep) {
            for (LabelledCorner& corner : labelled) {
                corner.col = spec.cols - 1 - corner.col;
                corner.row = spec.rows - 1 - corner.row;
            }
        }
        for (const LabelledCorner& corner : labelled) {
            if (corner.col < 0 || corner.col >= spec.cols || corner.row < 0 ||
                corner.row >= spec.rows) {
                return false;
            }
        }
    }
    return true;
}

/// Shift, orient and assemble the final view. Returns nullopt when the grown
/// extent does not fit the spec in either orientation or there are too few
/// corners.
std::optional<BoardView> build_view(const std::vector<Candidate>& corners,
                                    const GrownLattice& lattice, const BoardSpec& spec,
                                    const BoardDetectOptions& options,
                                    const FloatImage& smooth) {
    std::vector<LabelledCorner> labelled;
    int min_col = std::numeric_limits<int>::max();
    int min_row = std::numeric_limits<int>::max();
    for (std::size_t i = 0; i < corners.size(); ++i) {
        if (!lattice.labels[i]) {
            continue;
        }
        const LatticeLabel label = *lattice.labels[i];
        labelled.push_back(
            LabelledCorner{static_cast<int>(i), label.col, label.row});
        min_col = std::min(min_col, label.col);
        min_row = std::min(min_row, label.row);
    }
    if (static_cast<int>(labelled.size()) < options.min_corners) {
        return std::nullopt;
    }
    int max_col = std::numeric_limits<int>::min();
    int max_row = std::numeric_limits<int>::min();
    for (LabelledCorner& corner : labelled) {
        corner.col -= min_col;
        corner.row -= min_row;
        max_col = std::max(max_col, corner.col);
        max_row = std::max(max_row, corner.row);
    }
    // Labels must be a bijection onto distinct grid coordinates. Growth never
    // reuses a label, so a duplicate means two candidates claimed the same
    // cell: keep the first and drop the duplicate corner (never the view).
    std::set<std::pair<int, int>> seen_labels;
    std::vector<LabelledCorner> unique;
    unique.reserve(labelled.size());
    for (const LabelledCorner& corner : labelled) {
        if (seen_labels.insert(std::make_pair(corner.col, corner.row)).second) {
            unique.push_back(corner);
        } else {
            log(LogLevel::Debug) << "board_detector: dropping duplicate label ("
                                 << corner.col << "," << corner.row << ")";
        }
    }
    labelled.swap(unique);
    if (static_cast<int>(labelled.size()) < options.min_corners) {
        return std::nullopt;
    }
    const int extent_cols = max_col + 1;
    const int extent_rows = max_row + 1;
    const bool fits = extent_cols <= spec.cols && extent_rows <= spec.rows;
    const bool transposed = extent_cols <= spec.rows && extent_rows <= spec.cols;
    if (!fits && !transposed) {
        return std::nullopt;
    }
    if (!fits) {
        for (LabelledCorner& corner : labelled) {
            std::swap(corner.col, corner.row);
        }
    }
    const int final_cols = fits ? extent_cols : extent_rows;
    const int final_rows = fits ? extent_rows : extent_cols;
    bool complete = static_cast<int>(labelled.size()) == spec.cols * spec.rows &&
                    final_cols == spec.cols && final_rows == spec.rows;
    if (!canonicalise_labels(labelled, corners, smooth, spec, complete)) {
        return std::nullopt;
    }
    for (const LabelledCorner& corner : labelled) {
        if (corner.col < 0 || corner.col >= final_cols || corner.row < 0 ||
            corner.row >= final_rows) {
            log(LogLevel::Error) << "board_detector: label (" << corner.col << ","
                                 << corner.row << ") outside the fitted view";
            return std::nullopt;
        }
    }
    BoardView view;
    view.complete = complete;
    view.square_size_m = spec.square_size_m;
    view.corners.reserve(labelled.size());
    for (const LabelledCorner& corner : labelled) {
        const Candidate& source =
            corners[static_cast<std::size_t>(corner.candidate_index)];
        BoardCorner result;
        result.pixel = Eigen::Vector2d(source.x + 0.5, source.y + 0.5);
        result.col = corner.col;
        result.row = corner.row;
        result.response = source.response;
        view.corners.push_back(result);
    }
    std::sort(view.corners.begin(), view.corners.end(),
              [](const BoardCorner& a, const BoardCorner& b) {
                  if (a.row != b.row) {
                      return a.row < b.row;
                  }
                  return a.col < b.col;
              });
    return view;
}

/// Strongest verified corner in the central half of the image, falling back to
/// the strongest overall. Returns -1 for an empty set.
int select_seed(const std::vector<Candidate>& corners, int width, int height) {
    if (corners.empty()) {
        return -1;
    }
    const double x0 = 0.25 * static_cast<double>(width);
    const double x1 = 0.75 * static_cast<double>(width);
    const double y0 = 0.25 * static_cast<double>(height);
    const double y1 = 0.75 * static_cast<double>(height);
    int best = -1;
    double best_response = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(corners.size()); ++i) {
        const Candidate& corner = corners[static_cast<std::size_t>(i)];
        if (corner.x >= x0 && corner.x <= x1 && corner.y >= y0 && corner.y <= y1 &&
            corner.response > best_response) {
            best_response = corner.response;
            best = i;
        }
    }
    if (best >= 0) {
        return best;
    }
    for (int i = 0; i < static_cast<int>(corners.size()); ++i) {
        if (corners[static_cast<std::size_t>(i)].response > best_response) {
            best_response = corners[static_cast<std::size_t>(i)].response;
            best = i;
        }
    }
    return best;
}

}  // namespace

std::vector<Eigen::Vector2d> BoardView::board_points() const {
    std::vector<Eigen::Vector2d> points;
    points.reserve(corners.size());
    for (const BoardCorner& corner : corners) {
        points.emplace_back(static_cast<double>(corner.col) * square_size_m,
                            static_cast<double>(corner.row) * square_size_m);
    }
    return points;
}

std::optional<BoardView> detect_board(const FloatImage& gray, const BoardSpec& spec,
                                      const BoardDetectOptions& options) {
    if (spec.cols < 2 || spec.rows < 2) {
        throw ValueError("BoardSpec needs at least 2x2 inner corners");
    }
    const auto non_positive = [](double value) { return !(value > 0.0); };
    if (non_positive(spec.square_size_m)) {
        throw ValueError("BoardSpec.square_size_m must be positive");
    }
    if (non_positive(options.sigma_px) || non_positive(options.min_distance_px) ||
        non_positive(options.response_fraction) || non_positive(options.ring_radius_px) ||
        non_positive(options.angle_tol_deg) || non_positive(options.spacing_tol) ||
        options.refine_radius <= 0 || options.min_corners <= 0) {
        throw ValueError("BoardDetectOptions values must be positive");
    }
    if (gray.empty() || gray.width() < 3 || gray.height() < 3) {
        return std::nullopt;
    }
    const FloatImage smooth = gaussian_blur(gray, options.sigma_px);
    const FloatImage response = hessian_response(smooth);
    const std::vector<Candidate> raw =
        select_candidates(response, options.response_fraction, options.min_distance_px);
    if (static_cast<int>(raw.size()) < options.min_corners) {
        return std::nullopt;
    }
    const double angle_tolerance = options.angle_tol_deg * kPi / 180.0;
    const std::vector<Candidate> verified =
        verify_candidates(smooth, raw, options.ring_radius_px, options.refine_radius);
    if (static_cast<int>(verified.size()) < options.min_corners) {
        return std::nullopt;
    }
    const int seed = select_seed(verified, gray.width(), gray.height());
    if (seed < 0) {
        return std::nullopt;
    }
    const std::optional<GrownLattice> lattice =
        grow_lattice(verified, seed, angle_tolerance, options.spacing_tol);
    if (!lattice) {
        return std::nullopt;
    }
    return build_view(verified, *lattice, spec, options, smooth);
}

}  // namespace alphashape
