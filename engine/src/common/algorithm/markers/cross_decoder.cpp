#include "common/algorithm/markers/cross_decoder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "common/algorithm/feature/dog_detector.hpp"

namespace alphashape {
namespace {

FloatImage invert_image(const FloatImage& gray) {
    FloatImage out(gray.width(), gray.height(), 0.0F);
    for (int y = 0; y < gray.height(); ++y) {
        for (int x = 0; x < gray.width(); ++x) {
            out.at(x, y) = 1.0F - gray.at(x, y);
        }
    }
    return out;
}

struct Hessian {
    double xx = 0.0;
    double yy = 0.0;
    double xy = 0.0;
};

Hessian hessian_at(const FloatImage& image, int x, int y) {
    Hessian h;
    const double centre = static_cast<double>(image.at(x, y));
    h.xx = static_cast<double>(image.at(x + 1, y)) - 2.0 * centre +
           static_cast<double>(image.at(x - 1, y));
    h.yy = static_cast<double>(image.at(x, y + 1)) - 2.0 * centre +
           static_cast<double>(image.at(x, y - 1));
    h.xy = 0.25 * (static_cast<double>(image.at(x + 1, y + 1)) -
                   static_cast<double>(image.at(x + 1, y - 1)) -
                   static_cast<double>(image.at(x - 1, y + 1)) +
                   static_cast<double>(image.at(x - 1, y - 1)));
    return h;
}

struct QuadraticFit {
    bool ok = false;
    Eigen::Vector2d gradient = Eigen::Vector2d::Zero();
    Eigen::Matrix2d hessian = Eigen::Matrix2d::Zero();
    double minimum = 0.0;
    double maximum = 0.0;
};

std::optional<QuadraticFit> fit_quadratic(const FloatImage& image,
                                          const Eigen::Vector2d& centre, int radius) {
    Eigen::Matrix<double, 6, 6> normal = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> rhs = Eigen::Matrix<double, 6, 1>::Zero();
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -minimum;
    const double sigma = static_cast<double>(radius) / 1.5;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const double fx = static_cast<double>(dx);
            const double fy = static_cast<double>(dy);
            const double value = static_cast<double>(
                image.sample(static_cast<float>(centre.x() + fx),
                             static_cast<float>(centre.y() + fy)));
            if (!std::isfinite(value)) {
                return std::nullopt;
            }
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            Eigen::Matrix<double, 6, 1> basis;
            basis << 1.0, fx, fy, fx * fx, fx * fy, fy * fy;
            const double weight =
                std::exp(-(fx * fx + fy * fy) / (2.0 * sigma * sigma));
            normal += weight * basis * basis.transpose();
            rhs += weight * basis * value;
        }
    }
    const Eigen::Matrix<double, 6, 1> coefficients = normal.ldlt().solve(rhs);
    if (!coefficients.allFinite()) {
        return std::nullopt;
    }
    QuadraticFit fit;
    fit.ok = true;
    fit.gradient = Eigen::Vector2d(coefficients[1], coefficients[2]);
    fit.hessian << 2.0 * coefficients[3], coefficients[4],
                   coefficients[4], 2.0 * coefficients[5];
    fit.minimum = minimum;
    fit.maximum = maximum;
    return fit;
}
struct RefinedSaddle {
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    Eigen::Matrix2d hessian = Eigen::Matrix2d::Zero();
    double minimum = 0.0;
    double maximum = 0.0;
};

std::optional<RefinedSaddle> refine_saddle(const FloatImage& image,
                                           const Eigen::Vector2d& seed) {
    Eigen::Vector2d position = seed;
    std::optional<QuadraticFit> fit = fit_quadratic(image, position, 3);
    if (!fit) {
        return std::nullopt;
    }
    for (int iteration = 0; iteration < 4; ++iteration) {
        const double determinant = fit->hessian.determinant();
        if (!(determinant < 0.0)) {
            return std::nullopt;
        }
        const Eigen::Vector2d step = -fit->hessian.inverse() * fit->gradient;
        if (!step.allFinite() || step.norm() > 3.0) {
            return std::nullopt;
        }
        position += step;
        fit = fit_quadratic(image, position, 3);
        if (!fit) {
            return std::nullopt;
        }
        if (step.norm() < 0.01) {
            break;
        }
    }
    if (!(fit->hessian.determinant() < 0.0)) {
        return std::nullopt;
    }
    RefinedSaddle out;
    out.position = position;
    out.hessian = fit->hessian;
    out.minimum = fit->minimum;
    out.maximum = fit->maximum;
    return out;
}
struct CrossCandidate {
    Eigen::Vector2d center = Eigen::Vector2d::Zero();  ///< Calibration convention.
    double radius = 0.0;
    double response = 0.0;
    CrossStyle style = CrossStyle::Saddle;
};

std::optional<CrossCandidate> evaluate_saddle(const FloatImage& image,
                                              const Eigen::Vector2d& seed,
                                              const CrossParams& params) {
    const auto refined = refine_saddle(image, seed);
    if (!refined) {
        return std::nullopt;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(refined->hessian);
    if (solver.info() != Eigen::Success) {
        return std::nullopt;
    }
    const double contrast = refined->maximum - refined->minimum;
    if (!(contrast > 0.08)) {
        return std::nullopt;
    }
    const Eigen::Vector2d axis_a = solver.eigenvectors().col(1).normalized();
    const Eigen::Vector2d axis_b = solver.eigenvectors().col(0).normalized();
    const Eigen::Vector2d directions[4] = {axis_a, -axis_a, axis_b, -axis_b};
    const double limit =
        std::min(64.0, 0.45 * static_cast<double>(
                                  std::min(image.width(), image.height())));
    std::array<double, 4> extent{};
    std::array<double, 4> plateau{};
    for (int i = 0; i < 4; ++i) {
        const Eigen::Vector2d q = directions[i];
        // 6 px clears the blur transition for every supported scale; the scan
        // then starts beyond it so the first departure is the far edge of the
        // quadrant rather than the end of that transition.
        const double reference = static_cast<double>(image.sample(
            static_cast<float>(refined->position.x() + 6.0 * q.x()),
            static_cast<float>(refined->position.y() + 6.0 * q.y())));
        plateau[static_cast<std::size_t>(i)] = reference;
        double distance = limit;
        for (double r = 7.0; r <= limit; r += 0.5) {
            const double value = static_cast<double>(image.sample(
                static_cast<float>(refined->position.x() + r * q.x()),
                static_cast<float>(refined->position.y() + r * q.y())));
            if (std::abs(value - reference) > 0.3 * contrast) {
                distance = r;
                break;
            }
        }
        extent[static_cast<std::size_t>(i)] = distance;
    }
    if (std::abs(plateau[0] - plateau[1]) > 0.25 * contrast ||
        std::abs(plateau[2] - plateau[3]) > 0.25 * contrast) {
        return std::nullopt;
    }
    if (std::abs(plateau[0] - plateau[2]) < 0.35 * contrast) {
        return std::nullopt;  // adjacent quadrants do not differ: not a checker
    }
    std::array<double, 4> sorted = extent;
    std::sort(sorted.begin(), sorted.end());
    if (!(sorted[0] > 0.0) || sorted[3] > 2.5 * sorted[0]) {
        return std::nullopt;
    }
    const double median = 0.5 * (sorted[1] + sorted[2]);
    const double radius = median / std::sqrt(2.0);
    const double floor = std::max(6.0, params.minimum_size);
    if (radius < floor) {
        return std::nullopt;
    }
    CrossCandidate out;
    out.center = refined->position + Eigen::Vector2d(0.5, 0.5);
    out.radius = radius;
    out.response = std::abs(refined->hessian.determinant());
    return out;
}
std::vector<CrossCandidate> find_saddles(const FloatImage& image,
                                          const CrossParams& params) {
    std::vector<CrossCandidate> out;
    const int width = image.width();
    const int height = image.height();
    if (width < 8 || height < 8) {
        return out;
    }
    FloatImage response(width, height, 0.0F);
    double peak = 0.0;
    for (int y = 1; y + 1 < height; ++y) {
        for (int x = 1; x + 1 < width; ++x) {
            const Hessian h = hessian_at(image, x, y);
            const double determinant = h.xx * h.yy - h.xy * h.xy;
            if (determinant < 0.0) {
                const float value = static_cast<float>(-determinant);
                response.at(x, y) = value;
                peak = std::max(peak, static_cast<double>(value));
            }
        }
    }
    const double threshold =
        params.min_response > 0.0 ? params.min_response : 0.02 * peak;
    if (!(threshold > 0.0)) {
        return out;
    }
    for (int y = 1; y + 1 < height; ++y) {
        for (int x = 1; x + 1 < width; ++x) {
            const double value = static_cast<double>(response.at(x, y));
            if (value < threshold) {
                continue;
            }
            bool is_maximum = true;
            for (int dy = -1; dy <= 1 && is_maximum; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    if (static_cast<double>(response.at_clamped(x + dx, y + dy)) >
                        value) {
                        is_maximum = false;
                        break;
                    }
                }
            }
            if (!is_maximum) {
                continue;
            }
            if (auto candidate = evaluate_saddle(image, Eigen::Vector2d(x, y), params)) {
                out.push_back(*candidate);
            }
        }
    }
    return out;
}
/// Reject saddles that belong to a lattice. An AprilTag cell corner, a
/// checkerboard cell corner and every other dense fiducial corner has at least
/// one sibling saddle within three target radii. A printed GCP cross is
/// isolated, so requiring zero such neighbours removes the whole lattice
/// rather than only the corner with four or more siblings.
///
/// Candidates closer than `minimum_dist` are collapsed first: a quantised peak
/// can have several adjacent local maxima, and those are one saddle, not a
/// lattice. Doing the suppression before the neighbourhood test keeps such a
/// plateau from rejecting itself.
std::vector<CrossCandidate> filter_saddles(std::vector<CrossCandidate> candidates,
                                            double minimum_dist) {
    std::vector<std::size_t> order(candidates.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return candidates[a].response > candidates[b].response;
    });
    std::vector<CrossCandidate> unique;
    for (const std::size_t index : order) {
        bool clear = true;
        for (const CrossCandidate& kept : unique) {
            if ((kept.center - candidates[index].center).norm() < minimum_dist) {
                clear = false;
                break;
            }
        }
        if (clear) {
            unique.push_back(candidates[index]);
        }
    }
    std::vector<char> keep(unique.size(), 1);
    for (std::size_t i = 0; i < unique.size(); ++i) {
        for (std::size_t j = 0; j < unique.size(); ++j) {
            if (i == j) {
                continue;
            }
            const double reach =
                3.0 * std::max(unique[i].radius, unique[j].radius);
            if ((unique[i].center - unique[j].center).norm() <= reach) {
                keep[i] = 0;
                break;
            }
        }
    }
    std::vector<CrossCandidate> out;
    for (std::size_t i = 0; i < unique.size(); ++i) {
        if (keep[i] != 0) {
            out.push_back(unique[i]);
        }
    }
    return out;
}
struct RidgePoint {
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    Eigen::Vector2d tangent = Eigen::Vector2d::UnitX();
    double strength = 0.0;
};

std::vector<RidgePoint> find_ridge_points(const FloatImage& image) {
    std::vector<RidgePoint> points;
    const int width = image.width();
    const int height = image.height();
    if (width < 8 || height < 8) {
        return points;
    }
    double peak = 0.0;
    for (int y = 1; y + 1 < height; ++y) {
        for (int x = 1; x + 1 < width; ++x) {
            const Hessian h = hessian_at(image, x, y);
            Eigen::Matrix2d matrix;
            matrix << h.xx, h.xy, h.xy, h.yy;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(matrix);
            if (solver.info() != Eigen::Success) {
                continue;
            }
            const double first = solver.eigenvalues()[0];
            const double second = solver.eigenvalues()[1];
            const bool first_is_major = std::abs(first) >= std::abs(second);
            const double major = first_is_major ? first : second;
            const double minor = first_is_major ? second : first;
            if (!(major > 0.0) || major < 3.0 * std::abs(minor)) {
                continue;  // not a dark ridge
            }
            const Eigen::Vector2d normal =
                solver.eigenvectors().col(first_is_major ? 0 : 1);
            const Eigen::Vector2d gradient(
                0.5 * (static_cast<double>(image.at(x + 1, y)) -
                       static_cast<double>(image.at(x - 1, y))),
                0.5 * (static_cast<double>(image.at(x, y + 1)) -
                       static_cast<double>(image.at(x, y - 1))));
            // Steger's subpixel step along the ridge normal: the centre line
            // is where the first directional derivative vanishes.
            const double step = -gradient.dot(normal) / major;
            if (!std::isfinite(step) || std::abs(step) > 0.5) {
                continue;  // an edge, not the centre line
            }
            RidgePoint point;
            point.position = Eigen::Vector2d(x + step * normal.x(),
                                             y + step * normal.y());
            point.tangent = Eigen::Vector2d(-normal.y(), normal.x());
            point.strength = major;
            peak = std::max(peak, major);
            points.push_back(point);
        }
    }
    if (!(peak > 0.0)) {
        return {};
    }
    const double threshold = 0.05 * peak;
    points.erase(std::remove_if(points.begin(), points.end(),
                                [threshold](const RidgePoint& point) {
                                    return point.strength < threshold;
                                }),
                 points.end());
    return points;
}
std::vector<std::vector<int>> cluster_ridge_points(
    const std::vector<RidgePoint>& points, double radius) {
    const int count = static_cast<int>(points.size());
    std::vector<int> parent(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        parent[static_cast<std::size_t>(i)] = i;
    }
    const auto find_root = [&parent](int index) {
        while (parent[static_cast<std::size_t>(index)] != index) {
            parent[static_cast<std::size_t>(index)] =
                parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(index)])];
            index = parent[static_cast<std::size_t>(index)];
        }
        return index;
    };
    double min_x = points[0].position.x();
    double min_y = points[0].position.y();
    double max_x = min_x;
    double max_y = min_y;
    for (const RidgePoint& point : points) {
        min_x = std::min(min_x, point.position.x());
        min_y = std::min(min_y, point.position.y());
        max_x = std::max(max_x, point.position.x());
        max_y = std::max(max_y, point.position.y());
    }
    const auto column = [&](double x) {
        return static_cast<int>(std::floor((x - min_x) / radius));
    };
    const auto row = [&](double y) {
        return static_cast<int>(std::floor((y - min_y) / radius));
    };
    const int columns = column(max_x) + 1;
    const int rows = row(max_y) + 1;
    std::vector<std::vector<int>> grid(
        static_cast<std::size_t>(columns * rows));
    for (int i = 0; i < count; ++i) {
        const int cx = column(points[static_cast<std::size_t>(i)].position.x());
        const int cy = row(points[static_cast<std::size_t>(i)].position.y());
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = cx + dx;
                const int ny = cy + dy;
                if (nx < 0 || ny < 0 || nx >= columns || ny >= rows) {
                    continue;
                }
                const auto& cell = grid[static_cast<std::size_t>(ny * columns + nx)];
                for (const int other : cell) {
                    if ((points[static_cast<std::size_t>(other)].position -
                         points[static_cast<std::size_t>(i)].position).norm() <= radius) {
                        const int root_a = find_root(i);
                        const int root_b = find_root(other);
                        if (root_a != root_b) {
                            parent[static_cast<std::size_t>(root_b)] = root_a;
                        }
                    }
                }
            }
        }
        grid[static_cast<std::size_t>(cy * columns + cx)].push_back(i);
    }
    std::vector<std::vector<int>> groups(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        groups[static_cast<std::size_t>(find_root(i))].push_back(i);
    }
    groups.erase(std::remove_if(groups.begin(), groups.end(),
                                [](const std::vector<int>& group) {
                                    return group.size() < 8U;
                                }),
                 groups.end());
    return groups;
}
std::vector<int> line_inliers(const std::vector<RidgePoint>& points,
                              const std::vector<int>& candidates,
                              const Eigen::Vector2d& origin,
                              const Eigen::Vector2d& normal, double threshold) {
    std::vector<int> inliers;
    for (const int index : candidates) {
        const Eigen::Vector2d delta =
            points[static_cast<std::size_t>(index)].position - origin;
        if (std::abs(delta.dot(normal)) <= threshold) {
            inliers.push_back(index);
        }
    }
    return inliers;
}

/// Pick the straight line supported by the most ridge points. Each point
/// proposes its own tangent as the line direction; the winner is the line with
/// the most points within `threshold` of it. This replaces an orientation
/// clustering, which is ambiguous exactly at a cross where the two arms meet.
std::vector<int> select_line(const std::vector<RidgePoint>& points,
                             const std::vector<int>& candidates,
                             double threshold) {
    std::vector<int> best;
    for (const int index : candidates) {
        const RidgePoint& point = points[static_cast<std::size_t>(index)];
        const Eigen::Vector2d normal(-point.tangent.y(), point.tangent.x());
        std::vector<int> inliers =
            line_inliers(points, candidates, point.position, normal, threshold);
        if (inliers.size() > best.size()) {
            best = std::move(inliers);
        }
    }
    return best;
}
struct LineFit {
    Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
    Eigen::Vector2d tangent = Eigen::Vector2d::UnitX();
    double rms = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

std::optional<LineFit> fit_ridge_line(const std::vector<RidgePoint>& points,
                                       const std::vector<int>& group) {
    if (group.size() < 4U) {
        return std::nullopt;
    }
    LineFit fit;
    for (const int index : group) {
        fit.centroid += points[static_cast<std::size_t>(index)].position;
    }
    fit.centroid /= static_cast<double>(group.size());
    Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
    for (const int index : group) {
        const Eigen::Vector2d delta =
            points[static_cast<std::size_t>(index)].position - fit.centroid;
        covariance += delta * delta.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(covariance);
    if (solver.info() != Eigen::Success) {
        return std::nullopt;
    }
    fit.tangent = solver.eigenvectors().col(1).normalized();
    double squared = 0.0;
    bool first = true;
    for (const int index : group) {
        const Eigen::Vector2d delta =
            points[static_cast<std::size_t>(index)].position - fit.centroid;
        const double along = delta.dot(fit.tangent);
        const double across = delta.dot(Eigen::Vector2d(-fit.tangent.y(), fit.tangent.x()));
        squared += across * across;
        if (first) {
            fit.minimum = along;
            fit.maximum = along;
            first = false;
        } else {
            fit.minimum = std::min(fit.minimum, along);
            fit.maximum = std::max(fit.maximum, along);
        }
    }
    fit.rms = std::sqrt(squared / static_cast<double>(group.size()));
    return fit;
}

std::optional<Eigen::Vector2d> intersect_lines(const LineFit& a, const LineFit& b) {
    const Eigen::Vector2d normal_a(-a.tangent.y(), a.tangent.x());
    const Eigen::Vector2d normal_b(-b.tangent.y(), b.tangent.x());
    Eigen::Matrix2d matrix;
    matrix << normal_a.x(), normal_a.y(), normal_b.x(), normal_b.y();
    if (std::abs(matrix.determinant()) < 1e-6) {
        return std::nullopt;
    }
    const Eigen::Vector2d rhs(normal_a.dot(a.centroid), normal_b.dot(b.centroid));
    return Eigen::Vector2d(matrix.inverse() * rhs);
}
std::vector<CrossCandidate> find_bar_crosses(const FloatImage& image,
                                              const CrossParams& params) {
    std::vector<CrossCandidate> out;
    const std::vector<RidgePoint> points = find_ridge_points(image);
    if (points.size() < 16U) {
        return out;
    }
    const double radius = std::max(12.0, 2.0 * params.minimum_dist);
    for (const std::vector<int>& group : cluster_ridge_points(points, radius)) {
        const std::vector<int> arm_a = select_line(points, group, 1.5);
        if (arm_a.size() < 8U || static_cast<double>(arm_a.size()) <
                                     0.3 * static_cast<double>(group.size())) {
            continue;
        }
        std::vector<int> remaining;
        for (const int index : group) {
            if (std::find(arm_a.begin(), arm_a.end(), index) == arm_a.end()) {
                remaining.push_back(index);
            }
        }
        if (remaining.size() < 8U) {
            continue;
        }
        const std::vector<int> arm_b = select_line(points, remaining, 1.5);
        if (arm_b.size() < 8U) {
            continue;
        }
        const auto line_a = fit_ridge_line(points, arm_a);
        const auto line_b = fit_ridge_line(points, arm_b);
        if (!line_a || !line_b || line_a->rms > 1.5 || line_b->rms > 1.5) {
            continue;
        }
        if (std::abs(line_a->tangent.dot(line_b->tangent)) > 0.258819) {
            continue;  // not 90 degrees +/- 15 degrees
        }
        const auto centre = intersect_lines(*line_a, *line_b);
        if (!centre) {
            continue;
        }
        const double length_a = line_a->maximum - line_a->minimum;
        const double length_b = line_b->maximum - line_b->minimum;
        if (length_a < 8.0 || length_b < 8.0) {
            continue;
        }
        const double ratio = length_a / length_b;
        if (ratio < 0.2 || ratio > 5.0) {
            continue;
        }
        const double along_a = (*centre - line_a->centroid).dot(line_a->tangent);
        const double along_b = (*centre - line_b->centroid).dot(line_b->tangent);
        // The intersection must be well inside both arms, not at an endpoint.
        // Without this, two adjacent sides of a square outline (for example an
        // AprilTag's black border) meet at a corner and look like a cross.
        const double inset_a = 0.2 * length_a;
        const double inset_b = 0.2 * length_b;
        if (along_a < line_a->minimum + inset_a ||
            along_a > line_a->maximum - inset_a ||
            along_b < line_b->minimum + inset_b ||
            along_b > line_b->maximum - inset_b) {
            continue;
        }
        CrossCandidate candidate;
        candidate.center = *centre + Eigen::Vector2d(0.5, 0.5);
        candidate.radius = 0.5 * std::min(length_a, length_b);
        if (candidate.radius < std::max(6.0, params.minimum_size)) {
            continue;
        }
        // A bar cross has dark pixels along both arms and one uniform
        // background in all four quadrants. A checkerboard crossing has the
        // same two ridge lines but alternating quadrants, so its four
        // diagonal samples span nearly the full contrast.
        const Eigen::Vector2d diagonal_a =
            (line_a->tangent + line_b->tangent).normalized();
        const Eigen::Vector2d diagonal_b =
            (line_a->tangent - line_b->tangent).normalized();
        const double probe = 0.5 * candidate.radius;
        std::array<double, 4> quadrants{};
        for (int i = 0; i < 4; ++i) {
            const Eigen::Vector2d direction =
                (i & 1) == 0 ? diagonal_a : diagonal_b;
            const double sign = (i & 2) == 0 ? 1.0 : -1.0;
            quadrants[static_cast<std::size_t>(i)] = static_cast<double>(
                image.sample(static_cast<float>(centre->x() + sign * probe * direction.x()),
                             static_cast<float>(centre->y() + sign * probe * direction.y())));
        }
        const auto [low, high] =
            std::minmax_element(quadrants.begin(), quadrants.end());
        const double background =
            0.25 * (quadrants[0] + quadrants[1] + quadrants[2] + quadrants[3]);
        const double arm_a_value = static_cast<double>(image.sample(
            static_cast<float>(centre->x() + probe * line_a->tangent.x()),
            static_cast<float>(centre->y() + probe * line_a->tangent.y())));
        const double arm_b_value = static_cast<double>(image.sample(
            static_cast<float>(centre->x() + probe * line_b->tangent.x()),
            static_cast<float>(centre->y() + probe * line_b->tangent.y())));
        const double contrast =
            std::max(std::abs(background - arm_a_value),
                     std::abs(background - arm_b_value));
        if (!(contrast > 0.05) || *high - *low > 0.35 * contrast) {
            continue;
        }
        candidate.response = static_cast<double>(group.size());
        candidate.style = CrossStyle::Bar;
        out.push_back(candidate);
    }
    return out;
}
}  // namespace

std::vector<CrossDetection> detect_crosses_styled(const FloatImage& gray,
                                                  const CrossParams& params) {
    std::vector<CrossDetection> out;
    if (gray.width() < 9 || gray.height() < 9) {
        return out;
    }
    const FloatImage inverted = params.inverted ? invert_image(gray) : FloatImage();
    const FloatImage& image = params.inverted ? inverted : gray;
    const FloatImage smoothed = gaussian_blur(image, 1.5F);
    std::vector<CrossCandidate> candidates =
        filter_saddles(find_saddles(smoothed, params), params.minimum_dist);
    // A printed bar is flat-topped, so the ridge centre line only becomes a
    // curvature extremum once the bar is blurred by roughly half its width.
    const FloatImage ridge_image = gaussian_blur(image, 3.0F);
    for (const CrossCandidate& bar : find_bar_crosses(ridge_image, params)) {
        CrossCandidate* duplicate = nullptr;
        for (CrossCandidate& kept : candidates) {
            if ((kept.center - bar.center).norm() < params.minimum_dist) {
                duplicate = &kept;
                break;
            }
        }
        if (duplicate == nullptr) {
            candidates.push_back(bar);
        } else {
            // Two fitted ridge lines are direct evidence of a bar target; keep
            // the bar style so full-resolution refinement re-fits the arms.
            duplicate->style = CrossStyle::Bar;
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const CrossCandidate& a, const CrossCandidate& b) {
                  if (a.center.y() != b.center.y()) {
                      return a.center.y() < b.center.y();
                  }
                  return a.center.x() < b.center.x();
              });
    for (const CrossCandidate& candidate : candidates) {
        CodedTarget target;
        target.code = -1;
        target.center = candidate.center;
        target.radius_px = candidate.radius;
        // GCC 13 reports a false "maybe uninitialized" on the four Eigen
        // vectors when this object is copied into the vector; filling them
        // makes the write explicit and silences it (the member is `{}`).
        target.corners.fill(Eigen::Vector2d::Zero());
        CrossDetection detection;
        detection.target = target;
        detection.style = candidate.style;
        out.push_back(detection);
    }
    return out;
}

std::vector<CodedTarget> detect_crosses(const FloatImage& gray,
                                        const CrossParams& params) {
    std::vector<CodedTarget> out;
    for (const CrossDetection& detection : detect_crosses_styled(gray, params)) {
        out.push_back(detection.target);
    }
    return out;
}

std::optional<Eigen::Vector2d> refine_cross_center(const FloatImage& gray,
                                                    const Eigen::Vector2d& approximate,
                                                    double radius_hint) {
    const Eigen::Vector2d point = approximate - Eigen::Vector2d(0.5, 0.5);
    const int half =
        static_cast<int>(std::ceil(std::max(8.0, 2.0 * radius_hint + 4.0)));
    const int x0 = std::max(0, static_cast<int>(std::floor(point.x())) - half);
    const int y0 = std::max(0, static_cast<int>(std::floor(point.y())) - half);
    const int x1 = std::min(gray.width() - 1, x0 + 2 * half);
    const int y1 = std::min(gray.height() - 1, y0 + 2 * half);
    if (x1 <= x0 || y1 <= y0) {
        return std::nullopt;
    }
    FloatImage crop(x1 - x0 + 1, y1 - y0 + 1, 0.0F);
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            crop.at(x - x0, y - y0) = gray.at(x, y);
        }
    }
    const FloatImage smoothed = gaussian_blur(crop, 1.5F);
    const auto refined = refine_saddle(
        smoothed, Eigen::Vector2d(point.x() - static_cast<double>(x0),
                                  point.y() - static_cast<double>(y0)));
    if (!refined) {
        return std::nullopt;
    }
    return refined->position + Eigen::Vector2d(static_cast<double>(x0) + 0.5,
                                               static_cast<double>(y0) + 0.5);
}

std::optional<Eigen::Vector2d> refine_bar_cross_center(
    const FloatImage& gray, const Eigen::Vector2d& approximate,
    double radius_hint) {
    const Eigen::Vector2d point = approximate - Eigen::Vector2d(0.5, 0.5);
    const int half =
        static_cast<int>(std::ceil(std::max(12.0, 2.0 * radius_hint + 8.0)));
    const int x0 = std::max(0, static_cast<int>(std::floor(point.x())) - half);
    const int y0 = std::max(0, static_cast<int>(std::floor(point.y())) - half);
    const int x1 = std::min(gray.width() - 1, x0 + 2 * half);
    const int y1 = std::min(gray.height() - 1, y0 + 2 * half);
    if (x1 <= x0 || y1 <= y0) {
        return std::nullopt;
    }
    FloatImage crop(x1 - x0 + 1, y1 - y0 + 1, 0.0F);
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            crop.at(x - x0, y - y0) = gray.at(x, y);
        }
    }
    // A flat-topped bar only becomes a curvature extremum once it is blurred
    // by roughly half its width. The bar width scales with the target, so the
    // full-resolution pass derives its scale from the detected radius instead
    // of reusing the global pass's narrow-bar 3 px.
    const double sigma = std::clamp(0.2 * radius_hint, 2.5, 12.0);
    const FloatImage smoothed = gaussian_blur(crop, static_cast<float>(sigma));
    const Eigen::Vector2d seed(point.x() - static_cast<double>(x0),
                               point.y() - static_cast<double>(y0));
    const std::vector<RidgePoint> all_points = find_ridge_points(smoothed);
    const double reach = 1.6 * std::max(radius_hint, 8.0) + 10.0;
    std::vector<int> points;
    for (std::size_t i = 0; i < all_points.size(); ++i) {
        if ((all_points[i].position - seed).norm() <= reach) {
            points.push_back(static_cast<int>(i));
        }
    }
    if (points.size() < 16U) {
        return std::nullopt;
    }
    const std::vector<int> arm_a = select_line(all_points, points, 1.5);
    if (arm_a.size() < 8U) {
        return std::nullopt;
    }
    std::vector<int> remaining;
    for (const int index : points) {
        if (std::find(arm_a.begin(), arm_a.end(), index) == arm_a.end()) {
            remaining.push_back(index);
        }
    }
    const std::vector<int> arm_b = select_line(all_points, remaining, 1.5);
    if (arm_b.size() < 8U) {
        return std::nullopt;
    }
    const auto line_a = fit_ridge_line(all_points, arm_a);
    const auto line_b = fit_ridge_line(all_points, arm_b);
    if (!line_a || !line_b || line_a->rms > 1.5 || line_b->rms > 1.5) {
        return std::nullopt;
    }
    if (std::abs(line_a->tangent.dot(line_b->tangent)) > 0.258819) {
        return std::nullopt;
    }
    const auto first_centre = intersect_lines(*line_a, *line_b);
    if (!first_centre) {
        return std::nullopt;
    }
    // Refit both lines without the central region, where ridge points from one
    // arm contaminate the other arm's fit.
    const double exclusion = std::max(6.0, 0.25 * radius_hint);
    const auto outside = [&](const std::vector<int>& group,
                             const LineFit& line) {
        std::vector<int> kept;
        for (const int index : group) {
            const double along =
                (all_points[static_cast<std::size_t>(index)].position -
                 *first_centre)
                    .dot(line.tangent);
            if (std::abs(along) >= exclusion) {
                kept.push_back(index);
            }
        }
        return kept;
    };
    const std::vector<int> clean_a = outside(arm_a, *line_a);
    const std::vector<int> clean_b = outside(arm_b, *line_b);
    const auto refined_a = fit_ridge_line(all_points, clean_a);
    const auto refined_b = fit_ridge_line(all_points, clean_b);
    if (!refined_a || !refined_b || refined_a->rms > 1.5 ||
        refined_b->rms > 1.5) {
        return std::nullopt;
    }
    const auto centre = intersect_lines(*refined_a, *refined_b);
    if (!centre) {
        return std::nullopt;
    }
    const double length_a = refined_a->maximum - refined_a->minimum;
    const double length_b = refined_b->maximum - refined_b->minimum;
    if (length_a < 8.0 || length_b < 8.0) {
        return std::nullopt;
    }
    const double ratio = length_a / length_b;
    if (ratio < 0.2 || ratio > 5.0) {
        return std::nullopt;
    }
    const double along_a = (*centre - refined_a->centroid).dot(refined_a->tangent);
    const double along_b = (*centre - refined_b->centroid).dot(refined_b->tangent);
    if (along_a < refined_a->minimum + 0.2 * length_a ||
        along_a > refined_a->maximum - 0.2 * length_a ||
        along_b < refined_b->minimum + 0.2 * length_b ||
        along_b > refined_b->maximum - 0.2 * length_b) {
        return std::nullopt;
    }
    return *centre + Eigen::Vector2d(static_cast<double>(x0) + 0.5,
                                     static_cast<double>(y0) + 0.5);
}

}  // namespace alphashape
