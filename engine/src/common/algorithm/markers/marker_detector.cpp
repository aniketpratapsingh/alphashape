#include "common/algorithm/markers/marker_detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "common/algorithm/feature/dog_detector.hpp"

namespace alphashape {
namespace {

constexpr double kPi = std::numbers::pi_v<double>;
constexpr double kTwoPi = 2.0 * kPi;
constexpr int kRadialRays = 64;
constexpr double kEdgeStep = 0.5;
constexpr double kMinEdgeGradient = 0.005;

struct WeightedPoint {
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    double weight = 1.0;
};

double median_of(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    return values[mid];
}

/// Fold an ellipse axis angle into [0, pi), because the major and minor axis
/// directions are the same line.
double wrap_axis_angle(double angle) {
    angle = std::fmod(angle, kPi);
    if (angle < 0.0) {
        angle += kPi;
    }
    return angle;
}

FloatImage gradient_magnitude(const FloatImage& gray) {
    const int w = gray.width();
    const int h = gray.height();
    FloatImage out(w, h, 0.0F);
    const int last_row = h - 1;
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 1; y < last_row; ++y) {
        for (int x = 1; x + 1 < w; ++x) {
            const float gx = 0.5F * (gray.at(x + 1, y) - gray.at(x - 1, y));
            const float gy = 0.5F * (gray.at(x, y + 1) - gray.at(x, y - 1));
            out.at(x, y) = std::sqrt(gx * gx + gy * gy);
        }
    }
    return out;
}

/// Weighted direct least-squares ellipse fit (Fitzgibbon, Pilu & Fisher 1999,
/// numerically stable Halir/Flusser split form). Points are normalised first so
/// the design matrix stays well conditioned; the geometry is transformed back
/// afterwards. A non-positive weight means "ignored".
std::optional<Ellipse> fit_ellipse_points(const std::vector<WeightedPoint>& samples) {
    if (samples.size() < 5) {
        return std::nullopt;
    }
    double weight_sum = 0.0;
    double mean_x = 0.0;
    double mean_y = 0.0;
    for (const WeightedPoint& sample : samples) {
        if (!sample.point.allFinite() || !(sample.weight > 0.0)) continue;
        weight_sum += sample.weight;
        mean_x += sample.weight * sample.point.x();
        mean_y += sample.weight * sample.point.y();
    }
    if (!(weight_sum > 0.0)) {
        return std::nullopt;
    }
    mean_x /= weight_sum;
    mean_y /= weight_sum;
    double spread = 0.0;
    for (const WeightedPoint& sample : samples) {
        if (!sample.point.allFinite() || !(sample.weight > 0.0)) continue;
        const double dx = sample.point.x() - mean_x;
        const double dy = sample.point.y() - mean_y;
        spread += sample.weight * (dx * dx + dy * dy);
    }
    spread /= weight_sum;
    const double scale = std::sqrt(spread);
    if (!(scale > 1e-12)) {
        return std::nullopt;
    }

    Eigen::Matrix<double, 6, 6> scatter = Eigen::Matrix<double, 6, 6>::Zero();
    for (const WeightedPoint& sample : samples) {
        if (!sample.point.allFinite() || !(sample.weight > 0.0)) continue;
        const double x = (sample.point.x() - mean_x) / scale;
        const double y = (sample.point.y() - mean_y) / scale;
        Eigen::Matrix<double, 6, 1> row;
        row << x * x, x * y, y * y, x, y, 1.0;
        scatter.noalias() += (sample.weight / weight_sum) * (row * row.transpose());
    }

    const Eigen::Matrix3d block1 = scatter.topLeftCorner<3, 3>();
    const Eigen::Matrix3d block2 = scatter.topRightCorner<3, 3>();
    const Eigen::Matrix3d block3 = scatter.bottomRightCorner<3, 3>();
    const Eigen::FullPivLU<Eigen::Matrix3d> lu(block3);
    if (!lu.isInvertible()) {
        return std::nullopt;
    }
    const Eigen::Matrix3d reduced = block1 - block2 * lu.solve(block2.transpose());
    Eigen::Matrix3d constraint_inverse;
    constraint_inverse << 0.0, 0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0, 0.0;
    const Eigen::Matrix3d pencil = constraint_inverse * reduced;
    Eigen::EigenSolver<Eigen::Matrix3d> solver(pencil, true);
    if (solver.info() != Eigen::Success) {
        return std::nullopt;
    }
    const auto eigenvalues = solver.eigenvalues();
    const auto eigenvectors = solver.eigenvectors();
    Eigen::Vector3d quad = Eigen::Vector3d::Zero();
    double best_constraint = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < 3; ++i) {
        if (std::abs(eigenvalues(i).imag()) > 1e-6) continue;
        Eigen::Vector3d candidate = eigenvectors.col(i).real();
        if (!candidate.allFinite()) continue;
        candidate.normalize();
        const double constraint = 4.0 * candidate(0) * candidate(2) - candidate(1) * candidate(1);
        if (constraint > best_constraint) {
            best_constraint = constraint;
            quad = candidate;
        }
    }
    if (!(best_constraint > 0.0)) {
        return std::nullopt;
    }
    const Eigen::Vector3d tail = -lu.solve(block2.transpose() * quad);
    const double qa = quad(0);
    const double qb = quad(1);
    const double qc = quad(2);
    const double qd = tail(0);
    const double qe = tail(1);
    const double qf = tail(2);
    const double discriminant = qb * qb - 4.0 * qa * qc;
    if (!(discriminant < 0.0) || !std::isfinite(discriminant)) {
        return std::nullopt;
    }
    const double centre_x = (2.0 * qc * qd - qb * qe) / discriminant;
    const double centre_y = (2.0 * qa * qe - qb * qd) / discriminant;
    const double constant = qf - (qa * centre_x * centre_x + qb * centre_x * centre_y +
                                  qc * centre_y * centre_y);
    Eigen::Matrix2d quadratic;
    quadratic << qa, 0.5 * qb, 0.5 * qb, qc;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(quadratic);
    if (eigensolver.info() != Eigen::Success) {
        return std::nullopt;
    }
    const double lambda0 = eigensolver.eigenvalues()(0);
    const double lambda1 = eigensolver.eigenvalues()(1);
    if (!(lambda0 * lambda1 > 0.0)) {
        return std::nullopt;
    }
    const double ratio0 = -constant / lambda0;
    const double ratio1 = -constant / lambda1;
    if (!(ratio0 > 0.0) || !(ratio1 > 0.0)) {
        return std::nullopt;
    }

    const double semi0 = std::sqrt(ratio0);
    const double semi1 = std::sqrt(ratio1);
    const bool first_is_major = semi0 >= semi1;
    const double semi_major = first_is_major ? semi0 : semi1;
    const double semi_minor = first_is_major ? semi1 : semi0;
    const Eigen::Vector2d axis = eigensolver.eigenvectors().col(first_is_major ? 0 : 1);
    Ellipse ellipse;
    ellipse.center = Eigen::Vector2d(mean_x + scale * centre_x, mean_y + scale * centre_y);
    ellipse.a = scale * semi_major;
    ellipse.b = scale * semi_minor;
    ellipse.angle_rad = wrap_axis_angle(std::atan2(axis.y(), axis.x()));
    if (!ellipse.center.allFinite() || !std::isfinite(ellipse.a) || !std::isfinite(ellipse.b) ||
        !(ellipse.a > 0.0) || !(ellipse.b > 0.0)) {
        return std::nullopt;
    }
    return ellipse;
}

double ellipse_residual(const Ellipse& ellipse, const Eigen::Vector2d& point) {
    const double c = std::cos(ellipse.angle_rad);
    const double s = std::sin(ellipse.angle_rad);
    const double dx = point.x() - ellipse.center.x();
    const double dy = point.y() - ellipse.center.y();
    const double u = dx * c + dy * s;
    const double v = -dx * s + dy * c;
    const double radius = std::sqrt(u * u + v * v);
    const double rho = std::sqrt((u * u) / (ellipse.a * ellipse.a) + (v * v) / (ellipse.b * ellipse.b));
    return std::abs(rho - 1.0) * radius;
}

/// Fit, then drop samples more than three robust deviations off the first fit
/// and fit once more. A sample that lies on the wrong edge (a neighbouring
/// target, texture) is removed this way instead of dragging the ellipse with it.
std::optional<Ellipse> robust_fit(std::vector<WeightedPoint> samples) {
    const std::optional<Ellipse> first = fit_ellipse_points(samples);
    if (!first) {
        return std::nullopt;
    }
    std::vector<double> residuals;
    residuals.reserve(samples.size());
    for (const WeightedPoint& sample : samples) {
        if (!sample.point.allFinite() || !(sample.weight > 0.0)) continue;
        residuals.push_back(ellipse_residual(*first, sample.point));
    }
    if (residuals.size() < 5) {
        return first;
    }
    const double centre = median_of(residuals);
    std::vector<double> deviations;
    deviations.reserve(residuals.size());
    for (double value : residuals) {
        deviations.push_back(std::abs(value - centre));
    }
    const double mad = median_of(std::move(deviations));
    const double tolerance = centre + 3.0 * std::max(1.4826 * mad, 1e-3);
    std::vector<WeightedPoint> kept;
    kept.reserve(samples.size());
    std::size_t index = 0;
    for (const WeightedPoint& sample : samples) {
        if (!sample.point.allFinite() || !(sample.weight > 0.0)) continue;
        if (residuals[index] <= tolerance) kept.push_back(sample);
        ++index;
    }
    if (kept.size() < 5 || kept.size() == samples.size()) {
        return first;
    }
    const std::optional<Ellipse> second = fit_ellipse_points(kept);
    return second ? second : first;
}

/// Sample gradient-magnitude maxima along rays from `centre`. Each accepted
/// maximum gets a parabolic subpixel correction, so the returned points sit on
/// the target boundary to a fraction of a pixel.
std::vector<WeightedPoint> radial_edge_samples(const FloatImage& gradient,
                                               const Eigen::Vector2d& centre,
                                               double radius) {
    std::vector<WeightedPoint> samples;
    if (!(radius > 0.0) || gradient.empty()) {
        return samples;
    }
    const double outer = std::max(8.0, 2.5 * radius);
    const int steps = std::max(8, static_cast<int>((outer - kEdgeStep) / kEdgeStep));
    std::vector<double> values(static_cast<std::size_t>(steps + 1), 0.0);
    samples.reserve(static_cast<std::size_t>(kRadialRays));
    for (int ray = 0; ray < kRadialRays; ++ray) {
        const double angle =
            kTwoPi * (static_cast<double>(ray) + 0.5) / static_cast<double>(kRadialRays);
        const double dx = std::cos(angle);
        const double dy = std::sin(angle);
        for (int i = 0; i <= steps; ++i) {
            const double t = kEdgeStep + static_cast<double>(i) * kEdgeStep;
            values[static_cast<std::size_t>(i)] = static_cast<double>(gradient.sample(
                static_cast<float>(centre.x() + dx * t),
                static_cast<float>(centre.y() + dy * t)));
        }
        double best = -1.0;
        int best_index = -1;
        for (int i = 1; i < steps; ++i) {
            const double value = values[static_cast<std::size_t>(i)];
            if (value >= values[static_cast<std::size_t>(i - 1)] &&
                value > values[static_cast<std::size_t>(i + 1)] && value > best) {
                best = value;
                best_index = i;
            }
        }
        if (best_index <= 0 || best_index >= steps || best < kMinEdgeGradient) {
            continue;
        }
        const double left = values[static_cast<std::size_t>(best_index - 1)];
        const double mid = values[static_cast<std::size_t>(best_index)];
        const double right = values[static_cast<std::size_t>(best_index + 1)];
        const double denominator = left - 2.0 * mid + right;
        double shift = 0.0;
        if (denominator < -1e-12) {
            shift = 0.5 * (left - right) / denominator;
        }
        shift = std::clamp(shift, -0.5, 0.5);
        const double t = kEdgeStep + (static_cast<double>(best_index) + shift) * kEdgeStep;
        WeightedPoint sample;
        sample.point = Eigen::Vector2d(centre.x() + dx * t, centre.y() + dy * t);
        sample.weight = best;
        samples.push_back(sample);
    }
    return samples;
}

std::optional<Ellipse> fit_boundary(const FloatImage& gradient, const Eigen::Vector2d& centre,
                                    double radius) {
    return robust_fit(radial_edge_samples(gradient, centre, radius));
}

/// Scale-normalised Hessian determinant. The derivatives are of the image
/// already smoothed at `sigma`; multiplying both second derivatives by sigma^2
/// makes the response scale invariant (Steger 1998). The determinant is
/// positive for both intensity minima and maxima, so polarity is selected from
/// the trace: a dark-on-light target is a minimum (positive-definite Hessian),
/// a light-on-dark target is a maximum (negative-definite Hessian). The wrong
/// polarity becomes a negative response and is rejected by the threshold.
FloatImage hessian_response(const FloatImage& blurred, double sigma, bool dark_on_light) {
    const int w = blurred.width();
    const int h = blurred.height();
    FloatImage out(w, h, 0.0F);
    const float scale = static_cast<float>(sigma * sigma);
    const int last_row = h - 1;
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 1; y < last_row; ++y) {
        for (int x = 1; x + 1 < w; ++x) {
            const float value = blurred.at(x, y);
            const float dxx = blurred.at(x + 1, y) + blurred.at(x - 1, y) - 2.0F * value;
            const float dyy = blurred.at(x, y + 1) + blurred.at(x, y - 1) - 2.0F * value;
            const float dxy = 0.25F * (blurred.at(x + 1, y + 1) - blurred.at(x - 1, y + 1) -
                                       blurred.at(x + 1, y - 1) + blurred.at(x - 1, y - 1));
            const float nxx = scale * dxx;
            const float nyy = scale * dyy;
            const float nxy = scale * dxy;
            const float determinant = nxx * nyy - nxy * nxy;
            const float trace = nxx + nyy;
            const bool matches_polarity = dark_on_light ? trace > 0.0F : trace < 0.0F;
            out.at(x, y) = (matches_polarity ? 1.0F : -1.0F) * determinant;
        }
    }
    return out;
}

struct Seed {
    int x = 0;
    int y = 0;
    float response = 0.0F;
    double radius = 0.0;
};

/// Strict 3x3x3 maximum across the two neighbouring scales.
bool is_scale_maximum(const FloatImage& current, const FloatImage* previous,
                      const FloatImage* next, int x, int y, float value) {
    const FloatImage* levels[3] = {previous, &current, next};
    for (const FloatImage* level : levels) {
        if (level == nullptr) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (level == &current && dx == 0 && dy == 0) continue;
                if (level->at(x + dx, y + dy) >= value) return false;
            }
        }
    }
    return true;
}

void scan_scale(const FloatImage& current, const FloatImage* previous, const FloatImage* next,
                float threshold, double radius, std::vector<Seed>& seeds) {
    const int w = current.width();
    const int h = current.height();
    for (int y = 1; y + 1 < h; ++y) {
        for (int x = 1; x + 1 < w; ++x) {
            const float value = current.at(x, y);
            if (!(value >= threshold)) continue;
            if (!is_scale_maximum(current, previous, next, x, y, value)) continue;
            seeds.push_back({x, y, value, radius});
        }
    }
}

double circle_intersection_area(double radius_a, double radius_b, double distance) {
    if (distance >= radius_a + radius_b) {
        return 0.0;
    }
    if (distance <= std::abs(radius_a - radius_b)) {
        const double small = std::min(radius_a, radius_b);
        return kPi * small * small;
    }
    const double term_a = std::clamp(
        (distance * distance + radius_a * radius_a - radius_b * radius_b) / (2.0 * distance * radius_a),
        -1.0, 1.0);
    const double term_b = std::clamp(
        (distance * distance + radius_b * radius_b - radius_a * radius_a) / (2.0 * distance * radius_b),
        -1.0, 1.0);
    const double lens = radius_a * radius_a * std::acos(term_a) +
                        radius_b * radius_b * std::acos(term_b);
    const double triangle = 0.5 * std::sqrt(std::max(
        0.0, (-distance + radius_a + radius_b) * (distance + radius_a - radius_b) *
                 (distance - radius_a + radius_b) * (distance + radius_a + radius_b)));
    return lens - triangle;
}

double circle_iou(const MarkerCandidate& a, const MarkerCandidate& b) {
    if (!(a.radius_px > 0.0) || !(b.radius_px > 0.0)) {
        return 0.0;
    }
    const double distance = (a.center - b.center).norm();
    const double intersection = circle_intersection_area(a.radius_px, b.radius_px, distance);
    const double union_area =
        kPi * (a.radius_px * a.radius_px + b.radius_px * b.radius_px) - intersection;
    return union_area > 0.0 ? intersection / union_area : 0.0;
}

bool by_strongest(const MarkerCandidate& a, const MarkerCandidate& b) {
    if (a.response != b.response) return a.response > b.response;
    if (a.center.y() != b.center.y()) return a.center.y() < b.center.y();
    return a.center.x() < b.center.x();
}

bool by_position(const MarkerCandidate& a, const MarkerCandidate& b) {
    if (a.center.y() != b.center.y()) return a.center.y() < b.center.y();
    if (a.center.x() != b.center.x()) return a.center.x() < b.center.x();
    return a.response > b.response;
}

/// Greedy non-maximum suppression: strongest first, keep a candidate only if it
/// overlaps no accepted one by more than half its bounding circle.
std::vector<MarkerCandidate> suppress_overlaps(std::vector<MarkerCandidate> candidates) {
    std::sort(candidates.begin(), candidates.end(), by_strongest);
    std::vector<MarkerCandidate> kept;
    kept.reserve(candidates.size());
    for (const MarkerCandidate& candidate : candidates) {
        bool overlaps = false;
        for (const MarkerCandidate& accepted : kept) {
            if (circle_iou(candidate, accepted) > 0.5) {
                overlaps = true;
                break;
            }
        }
        if (!overlaps) kept.push_back(candidate);
    }
    std::sort(kept.begin(), kept.end(), by_position);
    return kept;
}

std::vector<MarkerCandidate> detect_blobs_impl(const FloatImage& gray, const BlobParams& params) {
    std::vector<MarkerCandidate> candidates;
    if (gray.empty() || params.max_radius < params.min_radius || params.min_radius < 1) {
        return candidates;
    }
    std::vector<double> radii;
    const double growth = std::sqrt(2.0);
    for (double radius = static_cast<double>(params.min_radius);
         radius <= static_cast<double>(params.max_radius) + 1e-9; radius *= growth) {
        radii.push_back(radius);
    }
    if (radii.empty()) {
        radii.push_back(static_cast<double>(params.min_radius));
    }
    const int scale_count = static_cast<int>(radii.size());
    const FloatImage gradient = gradient_magnitude(gray);
    std::vector<Seed> seeds;

    // The scale-space response for three consecutive scales is kept: the 3x3x3
    // maximum test needs both neighbours. Blurring incrementally (each level by
    // the difference in sigma) costs less than filtering the original each time.
    std::array<FloatImage, 3> response;
    double sigma = radii[0] / growth;
    FloatImage blurred = gaussian_blur(gray, static_cast<float>(sigma));
    response[0] = hessian_response(blurred, sigma, params.dark_on_light);
    for (int index = 0; index < scale_count; ++index) {
        if (index + 1 < scale_count) {
            const double next_sigma = radii[static_cast<std::size_t>(index + 1)] / growth;
            const double increment =
                std::sqrt(std::max(0.0, next_sigma * next_sigma - sigma * sigma));
            blurred = gaussian_blur(blurred, static_cast<float>(increment));
            sigma = next_sigma;
            response[static_cast<std::size_t>((index + 1) % 3)] =
                hessian_response(blurred, sigma, params.dark_on_light);
        }
        const FloatImage& current = response[static_cast<std::size_t>(index % 3)];
        const FloatImage* previous =
            index > 0 ? &response[static_cast<std::size_t>((index + 2) % 3)] : nullptr;
        const FloatImage* next = index + 1 < scale_count
                                     ? &response[static_cast<std::size_t>((index + 1) % 3)]
                                     : nullptr;
        scan_scale(current, previous, next, params.threshold,
                   radii[static_cast<std::size_t>(index)], seeds);
    }

    for (const Seed& seed : seeds) {
        std::optional<Ellipse> fit = fit_boundary(
            gradient, Eigen::Vector2d(static_cast<double>(seed.x), static_cast<double>(seed.y)),
            seed.radius);
        if (!fit) {
            continue;
        }
        Eigen::Vector2d centre = fit->center;
        double radius = fit->a;
        std::optional<Ellipse> refined = fit;
        for (int iteration = 0; iteration < 2; ++iteration) {
            std::optional<Ellipse> next_fit = fit_boundary(gradient, centre, radius);
            if (!next_fit) break;
            const double move = (next_fit->center - centre).norm();
            refined = next_fit;
            centre = next_fit->center;
            radius = std::max(next_fit->a, 1.0);
            if (move < 1e-3) break;
        }
        const double ratio = refined->b / refined->a;
        if (!(refined->a >= 1.0) || !(ratio >= params.min_axis_ratio)) {
            continue;
        }
        if (refined->a > 4.0 * static_cast<double>(params.max_radius)) {
            continue;
        }
        MarkerCandidate candidate;
        candidate.center = refined->center + Eigen::Vector2d(0.5, 0.5);
        candidate.radius_px = refined->a;
        candidate.angle_rad = refined->angle_rad;
        candidate.axis_ratio = ratio;
        candidate.response = static_cast<double>(seed.response);
        candidates.push_back(std::move(candidate));
    }
    return suppress_overlaps(std::move(candidates));
}

std::vector<double> box_means(const FloatImage& gray, int radius) {
    const int w = gray.width();
    const int h = gray.height();
    const std::size_t stride = static_cast<std::size_t>(w) + 1;
    std::vector<double> integral((static_cast<std::size_t>(h) + 1) * stride, 0.0);
    for (int y = 0; y < h; ++y) {
        double row = 0.0;
        for (int x = 0; x < w; ++x) {
            row += static_cast<double>(gray.at(x, y));
            integral[static_cast<std::size_t>(y + 1) * stride + static_cast<std::size_t>(x + 1)] =
                integral[static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x + 1)] + row;
        }
    }
    std::vector<double> means(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0.0);
    for (int y = 0; y < h; ++y) {
        const int y0 = std::max(0, y - radius);
        const int y1 = std::min(h - 1, y + radius);
        for (int x = 0; x < w; ++x) {
            const int x0 = std::max(0, x - radius);
            const int x1 = std::min(w - 1, x + radius);
            const double sum =
                integral[static_cast<std::size_t>(y1 + 1) * stride + static_cast<std::size_t>(x1 + 1)] -
                integral[static_cast<std::size_t>(y0) * stride + static_cast<std::size_t>(x1 + 1)] -
                integral[static_cast<std::size_t>(y1 + 1) * stride + static_cast<std::size_t>(x0)] +
                integral[static_cast<std::size_t>(y0) * stride + static_cast<std::size_t>(x0)];
            const double area =
                static_cast<double>(y1 - y0 + 1) * static_cast<double>(x1 - x0 + 1);
            means[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                  static_cast<std::size_t>(x)] = sum / area;
        }
    }
    return means;
}

std::vector<std::uint8_t> threshold_mask(const FloatImage& gray,
                                         const std::vector<double>& means,
                                         float threshold, bool dark_on_light) {
    const int w = gray.width();
    const int h = gray.height();
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(w) * static_cast<std::size_t>(h),
                                   std::uint8_t{0});
    const std::vector<float>& pixels = gray.pixels();
    for (std::size_t index = 0; index < mask.size(); ++index) {
        const double difference = static_cast<double>(pixels[index]) - means[index];
        const bool foreground = dark_on_light ? difference < -static_cast<double>(threshold)
                                              : difference > static_cast<double>(threshold);
        mask[index] = foreground ? std::uint8_t{1} : std::uint8_t{0};
    }
    return mask;
}

struct Component {
    std::vector<int> pixels;
    int min_x = 0;
    int max_x = 0;
    int min_y = 0;
    int max_y = 0;
};

std::vector<Component> connected_components(const std::vector<std::uint8_t>& mask, int w, int h) {
    std::vector<std::uint8_t> visited(mask.size(), std::uint8_t{0});
    std::vector<Component> components;
    std::vector<int> stack;
    const std::size_t width = static_cast<std::size_t>(w);
    for (int start = 0; start < w * h; ++start) {
        const std::size_t seed = static_cast<std::size_t>(start);
        if (!mask[seed] || visited[seed] != 0) continue;
        Component component;
        component.min_x = w;
        component.min_y = h;
        component.max_x = -1;
        component.max_y = -1;
        stack.clear();
        stack.push_back(start);
        visited[seed] = std::uint8_t{1};
        while (!stack.empty()) {
            const int index = stack.back();
            stack.pop_back();
            const int x = index % w;
            const int y = index / w;
            component.pixels.push_back(index);
            component.min_x = std::min(component.min_x, x);
            component.max_x = std::max(component.max_x, x);
            component.min_y = std::min(component.min_y, y);
            component.max_y = std::max(component.max_y, y);
            const int nx[4] = {x - 1, x + 1, x, x};
            const int ny[4] = {y, y, y - 1, y + 1};
            for (int k = 0; k < 4; ++k) {
                if (nx[k] < 0 || ny[k] < 0 || nx[k] >= w || ny[k] >= h) continue;
                const std::size_t neighbour =
                    static_cast<std::size_t>(ny[k]) * width + static_cast<std::size_t>(nx[k]);
                if (mask[neighbour] != 0 && visited[neighbour] == 0) {
                    visited[neighbour] = std::uint8_t{1};
                    stack.push_back(ny[k] * w + nx[k]);
                }
            }
        }
        components.push_back(std::move(component));
    }
    return components;
}

struct Pixel {
    int x = 0;
    int y = 0;
};

/// Moore-neighbour boundary trace (clockwise in image coordinates). The outer
/// boundary of a filled or ring-shaped component is traced.
std::vector<Pixel> trace_contour(const std::vector<std::uint8_t>& mask, int w, int h,
                                 int start_index) {
    constexpr int kDx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    constexpr int kDy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    const std::size_t width = static_cast<std::size_t>(w);
    std::vector<Pixel> contour;
    const Pixel start{start_index % w, start_index / w};
    Pixel current = start;
    Pixel backtrack{start.x - 1, start.y};
    const long long limit = static_cast<long long>(w) * static_cast<long long>(h) * 8LL;
    for (long long guard = 0; guard < limit; ++guard) {
        contour.push_back(current);
        int back_direction = 4;
        for (int d = 0; d < 8; ++d) {
            if (current.x + kDx[d] == backtrack.x && current.y + kDy[d] == backtrack.y) {
                back_direction = d;
                break;
            }
        }
        int found = -1;
        for (int step = 1; step <= 8; ++step) {
            const int d = (back_direction + step) % 8;
            const int nx = current.x + kDx[d];
            const int ny = current.y + kDy[d];
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            if (mask[static_cast<std::size_t>(ny) * width + static_cast<std::size_t>(nx)] != 0) {
                found = d;
                break;
            }
        }
        if (found < 0) break;
        backtrack = Pixel{current.x + kDx[(found + 7) % 8], current.y + kDy[(found + 7) % 8]};
        current = Pixel{current.x + kDx[found], current.y + kDy[found]};
        if (current.x == start.x && current.y == start.y) break;
    }
    return contour;
}

double point_segment_distance(const Eigen::Vector2d& point, const Eigen::Vector2d& a,
                              const Eigen::Vector2d& b) {
    const Eigen::Vector2d delta = b - a;
    const double length2 = delta.squaredNorm();
    if (!(length2 > 1e-18)) {
        return (point - a).norm();
    }
    const double t = std::clamp((point - a).dot(delta) / length2, 0.0, 1.0);
    return (point - (a + t * delta)).norm();
}

/// Douglas-Peucker simplification of an open polyline (a closed contour is
/// passed with its first point repeated at the end).
std::vector<Eigen::Vector2d> simplify_polyline(const std::vector<Eigen::Vector2d>& points,
                                               double tolerance) {
    if (points.size() < 3) {
        return points;
    }
    std::vector<Eigen::Vector2d> out;
    std::vector<std::pair<std::size_t, std::size_t>> stack;
    stack.emplace_back(static_cast<std::size_t>(0), points.size() - 1);
    while (!stack.empty()) {
        const std::pair<std::size_t, std::size_t> segment = stack.back();
        stack.pop_back();
        const std::size_t first = segment.first;
        const std::size_t last = segment.second;
        double farthest = -1.0;
        std::size_t index = first;
        for (std::size_t i = first + 1; i < last; ++i) {
            const double distance = point_segment_distance(points[i], points[first], points[last]);
            if (distance > farthest) {
                farthest = distance;
                index = i;
            }
        }
        if (farthest > tolerance && last > first + 1) {
            stack.emplace_back(index, last);
            stack.emplace_back(first, index);
        } else {
            out.push_back(points[first]);
        }
    }
    return out;
}

struct Line {
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    Eigen::Vector2d direction = Eigen::Vector2d::UnitX();
};

/// Weighted total-least-squares line through the samples (principal axis of the
/// weighted covariance).
std::optional<Line> fit_line(const std::vector<WeightedPoint>& samples) {
    double weight_sum = 0.0;
    Eigen::Vector2d centre = Eigen::Vector2d::Zero();
    for (const WeightedPoint& sample : samples) {
        if (!(sample.weight > 0.0)) continue;
        weight_sum += sample.weight;
        centre += sample.weight * sample.point;
    }
    if (!(weight_sum > 0.0)) {
        return std::nullopt;
    }
    centre /= weight_sum;
    Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
    for (const WeightedPoint& sample : samples) {
        if (!(sample.weight > 0.0)) continue;
        const Eigen::Vector2d delta = sample.point - centre;
        covariance.noalias() += sample.weight * (delta * delta.transpose());
    }
    covariance /= weight_sum;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(covariance);
    if (solver.info() != Eigen::Success) {
        return std::nullopt;
    }
    const Eigen::Vector2d direction = solver.eigenvectors().col(1);
    if (!direction.allFinite() || !(direction.norm() > 1e-12)) {
        return std::nullopt;
    }
    Line line;
    line.point = centre;
    line.direction = direction.normalized();
    return line;
}

std::optional<Eigen::Vector2d> intersect(const Line& a, const Line& b) {
    const double denominator =
        a.direction.x() * b.direction.y() - a.direction.y() * b.direction.x();
    if (std::abs(denominator) < 1e-6) {
        return std::nullopt;
    }
    const Eigen::Vector2d delta = b.point - a.point;
    const double t = (delta.x() * b.direction.y() - delta.y() * b.direction.x()) / denominator;
    const Eigen::Vector2d point = a.point + t * a.direction;
    if (!point.allFinite()) {
        return std::nullopt;
    }
    return point;
}

/// Locate each side's edge points at the half-intensity crossing along the side
/// normal. A crossing of a linearly interpolated intensity profile is unbiased
/// to O(pixel^2), whereas the peak of a bilinearly interpolated per-pixel
/// gradient field can be biased by a fifth of a pixel. The edge points are
/// weighted by the gradient magnitude at the crossing and fitted with a
/// weighted line; adjacent intersections are the subpixel corners.
std::optional<std::array<Eigen::Vector2d, 4>> refine_quad(const FloatImage& gray,
                                                          const FloatImage& gradient,
                                                          const std::array<Eigen::Vector2d, 4>& corners,
                                                          double& response) {
    constexpr int kNormalSteps = 41;
    constexpr double kNormalStep = 0.25;
    constexpr double kNormalExtent = 5.0;
    std::array<Line, 4> lines;
    double response_sum = 0.0;
    int response_count = 0;
    for (int side = 0; side < 4; ++side) {
        const Eigen::Vector2d a = corners[static_cast<std::size_t>(side)];
        const Eigen::Vector2d b = corners[static_cast<std::size_t>((side + 1) % 4)];
        const Eigen::Vector2d delta = b - a;
        const double length = delta.norm();
        if (!(length > 4.0)) {
            return std::nullopt;
        }
        const Eigen::Vector2d direction = delta / length;
        const Eigen::Vector2d normal(-direction.y(), direction.x());
        const int count = std::clamp(static_cast<int>(length / 3.0), 8, 48);
        std::vector<WeightedPoint> points;
        points.reserve(static_cast<std::size_t>(count));
        for (int k = 0; k < count; ++k) {
            const double fraction =
                0.15 + 0.7 * (static_cast<double>(k) + 0.5) / static_cast<double>(count);
            const Eigen::Vector2d base = a + fraction * delta;
            std::array<double, kNormalSteps> values{};
            double minimum = std::numeric_limits<double>::infinity();
            double maximum = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < kNormalSteps; ++i) {
                const double t = -kNormalExtent + kNormalStep * static_cast<double>(i);
                const Eigen::Vector2d probe = base + t * normal;
                const double value = static_cast<double>(
                    gray.sample(static_cast<float>(probe.x()), static_cast<float>(probe.y())));
                values[static_cast<std::size_t>(i)] = value;
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }
            if (!(maximum - minimum > 2.0 * kMinEdgeGradient)) {
                continue;
            }
            const double midpoint = 0.5 * (minimum + maximum);
            int best_index = -1;
            double best_slope = 0.0;
            for (int i = 0; i + 1 < kNormalSteps; ++i) {
                const double lower = values[static_cast<std::size_t>(i)];
                const double upper = values[static_cast<std::size_t>(i + 1)];
                const double low = std::min(lower, upper);
                const double high = std::max(lower, upper);
                if (midpoint < low || midpoint > high) continue;
                const double difference = upper - lower;
                if (std::abs(difference) < 1e-12) continue;
                const double slope = std::abs(difference) / kNormalStep;
                if (slope > best_slope) {
                    best_slope = slope;
                    best_index = i;
                }
            }
            if (best_index < 0 || best_slope < kMinEdgeGradient) {
                continue;
            }
            const double crossing =
                -kNormalExtent + kNormalStep *
                    (static_cast<double>(best_index) +
                     (midpoint - values[static_cast<std::size_t>(best_index)]) /
                         (values[static_cast<std::size_t>(best_index + 1)] -
                          values[static_cast<std::size_t>(best_index)]));
            WeightedPoint sample;
            sample.point = base + crossing * normal;
            sample.weight = static_cast<double>(gradient.sample(
                static_cast<float>(sample.point.x()), static_cast<float>(sample.point.y())));
            if (!(sample.weight > 0.0)) {
                sample.weight = best_slope;
            }
            points.push_back(sample);
            response_sum += best_slope;
            ++response_count;
        }
        if (points.size() < 5) {
            return std::nullopt;
        }
        const std::optional<Line> line = fit_line(points);
        if (!line) {
            return std::nullopt;
        }
        lines[static_cast<std::size_t>(side)] = *line;
    }
    std::array<Eigen::Vector2d, 4> refined;
    for (int corner = 0; corner < 4; ++corner) {
        const std::optional<Eigen::Vector2d> point =
            intersect(lines[static_cast<std::size_t>((corner + 3) % 4)],
                      lines[static_cast<std::size_t>(corner)]);
        if (!point) {
            return std::nullopt;
        }
        refined[static_cast<std::size_t>(corner)] = *point;
    }
    response = response_count > 0 ? response_sum / static_cast<double>(response_count) : 0.0;
    return refined;
}

double signed_area(const std::array<Eigen::Vector2d, 4>& corners) {
    double sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        const Eigen::Vector2d& a = corners[static_cast<std::size_t>(i)];
        const Eigen::Vector2d& b = corners[static_cast<std::size_t>((i + 1) % 4)];
        sum += a.x() * b.y() - b.x() * a.y();
    }
    return 0.5 * sum;
}

bool is_convex(const std::array<Eigen::Vector2d, 4>& corners) {
    double sign = 0.0;
    for (int i = 0; i < 4; ++i) {
        const Eigen::Vector2d& a = corners[static_cast<std::size_t>(i)];
        const Eigen::Vector2d& b = corners[static_cast<std::size_t>((i + 1) % 4)];
        const Eigen::Vector2d& c = corners[static_cast<std::size_t>((i + 2) % 4)];
        const double cross =
            (b.x() - a.x()) * (c.y() - b.y()) - (b.y() - a.y()) * (c.x() - b.x());
        if (std::abs(cross) < 1e-9) return false;
        if (i == 0) {
            sign = cross;
        } else if (cross * sign <= 0.0) {
            return false;
        }
    }
    return true;
}

/// Put the corners in screen counter-clockwise order (negative shoelace area in
/// image coordinates, where Y points down), starting at the top-left corner.
void orient_corners(std::array<Eigen::Vector2d, 4>& corners) {
    if (signed_area(corners) > 0.0) {
        std::reverse(corners.begin(), corners.end());
    }
    std::size_t top_left = 0;
    for (std::size_t i = 1; i < corners.size(); ++i) {
        const double dy = corners[i].y() - corners[top_left].y();
        if (dy < -1e-9 || (std::abs(dy) <= 1e-9 && corners[i].x() < corners[top_left].x())) {
            top_left = i;
        }
    }
    std::rotate(corners.begin(), corners.begin() + static_cast<std::ptrdiff_t>(top_left),
                corners.end());
}

std::vector<MarkerCandidate> detect_quads_impl(const FloatImage& gray, const QuadParams& params) {
    std::vector<MarkerCandidate> candidates;
    if (gray.empty() || params.max_side < params.min_side || params.min_side < 2) {
        return candidates;
    }
    const int w = gray.width();
    const int h = gray.height();
    const int window = std::max(4, params.max_side / 4);
    const std::vector<double> means = box_means(gray, window);
    const std::vector<std::uint8_t> mask =
        threshold_mask(gray, means, params.threshold, params.dark_on_light);
    const std::vector<Component> components = connected_components(mask, w, h);
    const FloatImage gradient = gradient_magnitude(gray);
    const std::size_t area_limit =
        static_cast<std::size_t>(params.max_side) * static_cast<std::size_t>(params.max_side);
    for (const Component& component : components) {
        const int component_width = component.max_x - component.min_x + 1;
        const int component_height = component.max_y - component.min_y + 1;
        if (component_width < params.min_side || component_height < params.min_side) continue;
        if (component_width > params.max_side + params.max_side / 2 ||
            component_height > params.max_side + params.max_side / 2) {
            continue;
        }
        if (component.pixels.size() < 12 || component.pixels.size() > area_limit * 2U) continue;
        int start_index = component.pixels.front();
        int start_x = start_index % w;
        int start_y = start_index / w;
        for (int index : component.pixels) {
            const int x = index % w;
            const int y = index / w;
            if (y < start_y || (y == start_y && x < start_x)) {
                start_index = index;
                start_x = x;
                start_y = y;
            }
        }
        const std::vector<Pixel> contour = trace_contour(mask, w, h, start_index);
        if (contour.size() < 8) continue;
        std::vector<Eigen::Vector2d> points;
        points.reserve(contour.size() + 1);
        for (const Pixel& pixel : contour) {
            points.emplace_back(static_cast<double>(pixel.x), static_cast<double>(pixel.y));
        }
        const Eigen::Vector2d first_point = points.front();
        points.push_back(first_point);
        std::array<Eigen::Vector2d, 4> quad = {Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(),
                                               Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero()};
        bool found = false;
        for (const double tolerance :
             {0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0}) {
            const std::vector<Eigen::Vector2d> simplified = simplify_polyline(points, tolerance);
            if (simplified.size() == 4) {
                for (std::size_t i = 0; i < 4; ++i) quad[i] = simplified[i];
                found = true;
                break;
            }
            if (simplified.size() < 4) break;
        }
        if (!found || !is_convex(quad)) continue;

        double response = 0.0;
        const std::optional<std::array<Eigen::Vector2d, 4>> refined =
            refine_quad(gray, gradient, quad, response);
        if (!refined) continue;
        std::array<Eigen::Vector2d, 4> corners = *refined;
        if (!is_convex(corners)) continue;
        orient_corners(corners);
        for (Eigen::Vector2d& corner : corners) {
            corner += Eigen::Vector2d(0.5, 0.5);
        }
        double min_length = std::numeric_limits<double>::infinity();
        double max_length = 0.0;
        for (int i = 0; i < 4; ++i) {
            const Eigen::Vector2d delta =
                corners[static_cast<std::size_t>((i + 1) % 4)] - corners[static_cast<std::size_t>(i)];
            const double length = delta.norm();
            min_length = std::min(min_length, length);
            max_length = std::max(max_length, length);
        }
        if (!(min_length > 0.0) || max_length > 4.0 * min_length) continue;
        if (min_length < 0.5 * static_cast<double>(params.min_side)) continue;
        if (max_length > 1.6 * static_cast<double>(params.max_side)) continue;
        MarkerCandidate candidate;
        candidate.center = 0.25 * (corners[0] + corners[1] + corners[2] + corners[3]);
        const double diagonal0 = (corners[2] - corners[0]).norm();
        const double diagonal1 = (corners[3] - corners[1]).norm();
        candidate.radius_px = 0.25 * (diagonal0 + diagonal1);
        const Eigen::Vector2d top = corners[3] - corners[0];
        candidate.angle_rad = std::atan2(top.y(), top.x());
        candidate.axis_ratio = 1.0;
        candidate.response = response;
        candidate.corners.assign(corners.begin(), corners.end());
        candidates.push_back(std::move(candidate));
    }
    return suppress_overlaps(std::move(candidates));
}

}  // namespace

std::optional<Ellipse> fit_ellipse(const std::vector<Eigen::Vector2d>& points) {
    std::vector<WeightedPoint> samples;
    samples.reserve(points.size());
    for (const Eigen::Vector2d& point : points) {
        samples.push_back({point, 1.0});
    }
    return fit_ellipse_points(samples);
}

std::optional<Eigen::Vector2d> refine_ellipse_center(const FloatImage& gray,
                                                     const MarkerCandidate& seed,
                                                     bool dark_on_light) {
    // The boundary is located from the gradient magnitude, which is the same
    // whichever side of the edge is brighter, so the polarity does not enter
    // the subpixel search.
    static_cast<void>(dark_on_light);
    if (gray.empty() || !seed.center.allFinite()) {
        return std::nullopt;
    }
    const FloatImage gradient = gradient_magnitude(gray);
    Eigen::Vector2d centre = seed.center - Eigen::Vector2d(0.5, 0.5);
    double radius = seed.radius_px > 1.0 ? seed.radius_px : 8.0;
    std::optional<Ellipse> last;
    for (int iteration = 0; iteration < 3; ++iteration) {
        const std::optional<Ellipse> fit = fit_boundary(gradient, centre, radius);
        if (!fit) break;
        last = fit;
        const double move = (fit->center - centre).norm();
        centre = fit->center;
        radius = std::max(fit->a, 1.0);
        if (move < 1e-3) break;
    }
    if (!last || !centre.allFinite()) {
        return std::nullopt;
    }
    return centre + Eigen::Vector2d(0.5, 0.5);
}

std::vector<MarkerCandidate> detect_blobs(const FloatImage& gray, const BlobParams& params) {
    return detect_blobs_impl(gray, params);
}

std::vector<MarkerCandidate> detect_quads(const FloatImage& gray, const QuadParams& params) {
    return detect_quads_impl(gray, params);
}

}  // namespace alphashape
