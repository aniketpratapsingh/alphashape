#include "common/algorithm/markers/apriltag_decoder.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace alphashape {
namespace {

constexpr double kEdgeMinMagnitude = 0.01;  ///< Gradient floor; images are in [0,1].
constexpr double kMinContrast = 0.08;       ///< Light minus dark, at the centre.
constexpr double kDecisionMargin = 0.2;     ///< Fraction of contrast a cell must
                                            ///< clear the threshold by.
constexpr int kCornerSamples = 16;          ///< Samples along each refined side.
constexpr double kCellSampleHalf = 0.10;    ///< Cell half-extent sampled, in cells.
constexpr double kMinSidePx = 6.0;
constexpr int kMinSamplesPerCell = 5;       ///< Of the 3x3 cell samples.

/// Bilinear read at calibration coordinates (top-left pixel centre is 0.5).
bool sample_pixel(const FloatImage& image, double x, double y, double& value) {
    const double fx = x - 0.5;
    const double fy = y - 0.5;
    const double last_x = static_cast<double>(image.width() - 1);
    const double last_y = static_cast<double>(image.height() - 1);
    if (!(fx >= 0.0) || !(fy >= 0.0) || fx > last_x || fy > last_y) {
        return false;
    }
    value = static_cast<double>(
        image.sample(static_cast<float>(fx), static_cast<float>(fy)));
    return true;
}

/// Projective map from the tag frame to calibration image coordinates.
struct Homography {
    Eigen::Matrix3d matrix = Eigen::Matrix3d::Identity();

    bool project(double u, double v, Eigen::Vector2d& out) const {
        const Eigen::Vector3d p = matrix * Eigen::Vector3d(u, v, 1.0);
        if (!std::isfinite(p.z()) || std::abs(p.z()) < 1e-12) {
            return false;
        }
        out = Eigen::Vector2d(p.x() / p.z(), p.y() / p.z());
        return out.allFinite();
    }
};

struct NormalisedPoints {
    Eigen::Matrix3d transform = Eigen::Matrix3d::Identity();
    std::array<Eigen::Vector2d, 4> points{};
    bool valid = false;
};

NormalisedPoints normalise_points(const std::array<Eigen::Vector2d, 4>& points) {
    NormalisedPoints result;
    Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
    for (const Eigen::Vector2d& point : points) centroid += point;
    centroid *= 0.25;
    double distance = 0.0;
    for (const Eigen::Vector2d& point : points) {
        distance += (point - centroid).norm();
    }
    distance *= 0.25;
    if (!std::isfinite(distance) || !(distance > 1e-9)) {
        return result;
    }
    const double scale = std::sqrt(2.0) / distance;
    result.transform << scale, 0.0, -scale * centroid.x(),
                        0.0, scale, -scale * centroid.y(),
                        0.0, 0.0, 1.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        result.points[i] = (points[i] - centroid) * scale;
    }
    result.valid = true;
    return result;
}

/// Direct linear transform from four tag-frame to four image points,
/// normalised on both sides so the 8x8 system is well conditioned.
std::optional<Homography> fit_homography(
    const std::array<Eigen::Vector2d, 4>& tag,
    const std::array<Eigen::Vector2d, 4>& image) {
    const NormalisedPoints source = normalise_points(tag);
    const NormalisedPoints target = normalise_points(image);
    if (!source.valid || !target.valid) {
        return std::nullopt;
    }
    Eigen::Matrix<double, 8, 8> design = Eigen::Matrix<double, 8, 8>::Zero();
    Eigen::Matrix<double, 8, 1> rhs = Eigen::Matrix<double, 8, 1>::Zero();
    for (std::size_t i = 0; i < 4; ++i) {
        const double u = source.points[i].x();
        const double v = source.points[i].y();
        const double x = target.points[i].x();
        const double y = target.points[i].y();
        const Eigen::Index row = static_cast<Eigen::Index>(2 * i);
        design(row, 0) = u;
        design(row, 1) = v;
        design(row, 2) = 1.0;
        design(row, 6) = -u * x;
        design(row, 7) = -v * x;
        rhs(row) = x;
        design(row + 1, 3) = u;
        design(row + 1, 4) = v;
        design(row + 1, 5) = 1.0;
        design(row + 1, 6) = -u * y;
        design(row + 1, 7) = -v * y;
        rhs(row + 1) = y;
    }
    const Eigen::Matrix<double, 8, 1> h =
        design.fullPivLu().solve(rhs);
    if (!h.allFinite()) {
        return std::nullopt;
    }
    Eigen::Matrix3d normalised;
    normalised << h(0), h(1), h(2),
                  h(3), h(4), h(5),
                  h(6), h(7), 1.0;
    const Eigen::Matrix3d target_inverse = target.transform.inverse();
    Homography homography;
    homography.matrix = target_inverse * normalised * source.transform;
    if (!homography.matrix.allFinite() ||
        std::abs(homography.matrix(2, 2)) < 1e-12) {
        return std::nullopt;
    }
    homography.matrix /= homography.matrix(2, 2);
    return homography;
}

/// Mean of a 3x3 sub-grid inside one tag cell. `du`/`dv` are the tag-frame
/// half-extents of the sub-grid, so the same helper serves payload cells,
/// border cells and the bilinear reference samples.
bool sample_cell(const FloatImage& image, const Homography& homography,
                 double u, double v, double du, double dv, double& mean) {
    constexpr double offsets[3] = {-1.0, 0.0, 1.0};
    double sum = 0.0;
    int count = 0;
    for (double offset_v : offsets) {
        for (double offset_u : offsets) {
            Eigen::Vector2d point;
            if (!homography.project(u + offset_u * du, v + offset_v * dv, point)) {
                continue;
            }
            double value = 0.0;
            if (!sample_pixel(image, point.x(), point.y(), value)) {
                continue;
            }
            sum += value;
            ++count;
        }
    }
    if (count < kMinSamplesPerCell) {
        return false;
    }
    mean = sum / static_cast<double>(count);
    return true;
}

/// A fitted edge: a point on the line and a unit direction.
struct EdgeLine {
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    Eigen::Vector2d direction = Eigen::Vector2d(1.0, 0.0);
};

struct EdgeSample {
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    double weight = 0.0;
};

double median_of(std::vector<double> values);

/// Weighted total-least-squares line through the samples.
std::optional<EdgeLine> fit_weighted_tls(const std::vector<EdgeSample>& samples) {
    double weight_sum = 0.0;
    Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
    for (const EdgeSample& sample : samples) {
        weight_sum += sample.weight;
        centroid += sample.point * sample.weight;
    }
    if (!(weight_sum > 0.0)) {
        return std::nullopt;
    }
    centroid /= weight_sum;
    double sxx = 0.0;
    double sxy = 0.0;
    double syy = 0.0;
    for (const EdgeSample& sample : samples) {
        const Eigen::Vector2d delta = sample.point - centroid;
        sxx += sample.weight * delta.x() * delta.x();
        sxy += sample.weight * delta.x() * delta.y();
        syy += sample.weight * delta.y() * delta.y();
    }
    Eigen::Matrix2d covariance;
    covariance << sxx / weight_sum, sxy / weight_sum,
                  sxy / weight_sum, syy / weight_sum;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(covariance);
    if (solver.info() != Eigen::Success ||
        !(solver.eigenvalues()(1) > 1e-6)) {
        return std::nullopt;
    }
    EdgeLine line;
    line.point = centroid;
    line.direction = solver.eigenvectors().col(1).normalized();
    return line;
}

/// Fit the gradient-magnitude ridge along one side. Each of the samples along
/// the side searches a short distance across it; the peak positions are fitted
/// with a weighted total-least-squares line, the weight being the gradient
/// magnitude. Notches and foreign edges therefore contribute weakly.
enum class PeakChoice { Outermost, Strongest };

std::optional<EdgeLine> fit_edge(const GradientField& gradient,
                                 const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                                 const Eigen::Vector2d& center, double reach_factor,
                                 double reach_cap, PeakChoice choice) {
    const Eigen::Vector2d delta = b - a;
    const double length = delta.norm();
    if (!(length > 2.0 * kMinSidePx)) {
        return std::nullopt;
    }
    const Eigen::Vector2d direction = delta / length;
    Eigen::Vector2d normal(-direction.y(), direction.x());
    // Search outward from the initial edge. A narrow black border has a
    // parallel inner edge a few pixels inside; taking the outermost strong
    // peak keeps the fit on the tag's outer boundary (and on a notch-free
    // stretch of a reversed family's payload ring).
    if ((0.5 * (a + b) - center).dot(normal) < 0.0) {
        normal = -normal;
    }
    const double reach = std::clamp(reach_factor * length, 2.5, reach_cap);
    const int steps = static_cast<int>(std::ceil(reach / 0.5));
    const double step = reach / static_cast<double>(steps);
    std::vector<double> profile(static_cast<std::size_t>(2 * steps + 1), 0.0);
    std::vector<EdgeSample> samples;
    samples.reserve(static_cast<std::size_t>(kCornerSamples));
    for (int sample = 0; sample < kCornerSamples; ++sample) {
        const double alpha =
            (static_cast<double>(sample) + 0.5) / static_cast<double>(kCornerSamples);
        const Eigen::Vector2d base = a + delta * alpha;
        for (int index = 0; index <= 2 * steps; ++index) {
            const double offset = static_cast<double>(index - steps) * step;
            double magnitude = 0.0;
            if (!sample_pixel(gradient.magnitude, base.x() + normal.x() * offset,
                              base.y() + normal.y() * offset, magnitude)) {
                profile[static_cast<std::size_t>(index)] = 0.0;
                continue;
            }
            profile[static_cast<std::size_t>(index)] = magnitude;
        }
        double peak_magnitude = 0.0;
        for (int index = 0; index <= 2 * steps; ++index) {
            peak_magnitude =
                std::max(peak_magnitude, profile[static_cast<std::size_t>(index)]);
        }
        const double floor =
            std::max(kEdgeMinMagnitude, 0.45 * peak_magnitude);
        int best = -1;
        double best_magnitude = 0.0;
        for (int index = 1; index < 2 * steps; ++index) {
            const double value = profile[static_cast<std::size_t>(index)];
            if (value >= profile[static_cast<std::size_t>(index - 1)] &&
                value > profile[static_cast<std::size_t>(index + 1)] &&
                value >= floor) {
                if (choice == PeakChoice::Outermost) {
                    // The profile runs inward to outward; keeping the last
                    // qualifying maximum selects the outermost edge.
                    best = index;
                } else if (best < 0 || value > best_magnitude) {
                    best = index;
                    best_magnitude = value;
                }
            }
        }
        if (best <= 0 || best >= 2 * steps) {
            continue;
        }
        const double left = profile[static_cast<std::size_t>(best - 1)];
        const double middle = profile[static_cast<std::size_t>(best)];
        const double right = profile[static_cast<std::size_t>(best + 1)];
        const double denominator = left - 2.0 * middle + right;
        double shift = 0.0;
        if (denominator < -1e-12) {
            shift = 0.5 * (left - right) / denominator;
        }
        const double offset =
            (static_cast<double>(best - steps) + std::clamp(shift, -0.5, 0.5)) * step;
        const Eigen::Vector2d point = base + normal * offset;
        EdgeSample edge_sample;
        edge_sample.point = point;
        edge_sample.weight = middle;
        samples.push_back(edge_sample);
    }
    if (samples.size() < 4) {
        return std::nullopt;
    }
    const std::optional<EdgeLine> first = fit_weighted_tls(samples);
    if (!first) {
        return std::nullopt;
    }
    // A white payload cell touching a reference frame protrudes its edge by a
    // cell; a notch does the same to a silhouette. Those samples are outliers
    // of the straight run, so drop points far from the first line and refit.
    Eigen::Vector2d normal_to_line(-first->direction.y(), first->direction.x());
    std::vector<double> residuals;
    residuals.reserve(samples.size());
    for (const EdgeSample& sample : samples) {
        residuals.push_back(
            std::abs((sample.point - first->point).dot(normal_to_line)));
    }
    const double median = median_of(residuals);
    const double limit = std::max(0.75, 2.5 * median);
    std::vector<EdgeSample> kept;
    kept.reserve(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (residuals[i] <= limit) {
            kept.push_back(samples[i]);
        }
    }
    if (kept.size() >= 4 && kept.size() < samples.size()) {
        const std::optional<EdgeLine> second = fit_weighted_tls(kept);
        if (second) {
            return second;
        }
    }
    return first;
}

std::optional<Eigen::Vector2d> intersect_lines(const EdgeLine& first,
                                               const EdgeLine& second) {
    const double cross = first.direction.x() * second.direction.y() -
                         first.direction.y() * second.direction.x();
    if (std::abs(cross) < 1e-6) {
        return std::nullopt;
    }
    const Eigen::Vector2d delta = second.point - first.point;
    const double along =
        (delta.x() * second.direction.y() - delta.y() * second.direction.x()) / cross;
    return first.point + first.direction * along;
}

/// Replace each quad corner by the intersection of the lines fitted to the two
/// sides meeting there. A side that cannot be fitted keeps the original corner.
std::array<Eigen::Vector2d, 4> refine_quad_corners(
    const GradientField& gradient, const std::array<Eigen::Vector2d, 4>& quad,
    double reach_factor, double reach_cap, PeakChoice choice) {
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    for (const Eigen::Vector2d& corner : quad) {
        center += corner;
    }
    center *= 0.25;
    std::array<std::optional<EdgeLine>, 4> lines;
    for (int side = 0; side < 4; ++side) {
        lines[static_cast<std::size_t>(side)] = fit_edge(
            gradient, quad[static_cast<std::size_t>(side)],
            quad[static_cast<std::size_t>((side + 1) % 4)], center,
            reach_factor, reach_cap, choice);
    }
    std::array<Eigen::Vector2d, 4> refined = quad;
    for (int corner = 0; corner < 4; ++corner) {
        const std::optional<EdgeLine>& before =
            lines[static_cast<std::size_t>((corner + 3) % 4)];
        const std::optional<EdgeLine>& after = lines[static_cast<std::size_t>(corner)];
        if (!before || !after) {
            continue;
        }
        const std::optional<Eigen::Vector2d> intersection =
            intersect_lines(*before, *after);
        if (!intersection || !intersection->allFinite()) {
            continue;
        }
        // A thin frame can put the detector's initial quad on the inner edge
        // one cell in; the outward fit then legitimately moves the corner by
        // that cell (up to the search reach). Reject only wild intersections.
        if ((*intersection - quad[static_cast<std::size_t>(corner)]).norm() >
            reach_cap + 2.5) {
            continue;
        }
        refined[static_cast<std::size_t>(corner)] = *intersection;
    }
    return refined;
}

struct GraySample {
    double u = 0.0;
    double v = 0.0;
    double value = 0.0;
};

/// Bilinear intensity model `c0 + c1*u + c2*v + c3*u*v`. A plane or a
/// threshold read from the whole tag would be fooled by a lighting gradient;
/// fitting one model per class lets the threshold move with position.
struct BilinearModel {
    double c0 = 0.0;
    double c1 = 0.0;
    double c2 = 0.0;
    double c3 = 0.0;
    double mean_u = 0.0;
    double mean_v = 0.0;
    double scale_u = 1.0;
    double scale_v = 1.0;
    bool valid = false;

    double at(double u, double v) const {
        const double un = (u - mean_u) / scale_u;
        const double vn = (v - mean_v) / scale_v;
        return c0 + c1 * un + c2 * vn + c3 * un * vn;
    }
};

BilinearModel fit_bilinear(const std::vector<GraySample>& samples) {
    BilinearModel model;
    if (samples.size() < 6) {
        return model;
    }
    double mean_u = 0.0;
    double mean_v = 0.0;
    for (const GraySample& sample : samples) {
        mean_u += sample.u;
        mean_v += sample.v;
    }
    mean_u /= static_cast<double>(samples.size());
    mean_v /= static_cast<double>(samples.size());
    double spread_u = 0.0;
    double spread_v = 0.0;
    for (const GraySample& sample : samples) {
        spread_u += (sample.u - mean_u) * (sample.u - mean_u);
        spread_v += (sample.v - mean_v) * (sample.v - mean_v);
    }
    spread_u = std::sqrt(spread_u / static_cast<double>(samples.size()));
    spread_v = std::sqrt(spread_v / static_cast<double>(samples.size()));
    if (!(spread_u > 1e-6) || !(spread_v > 1e-6)) {
        return model;
    }
    Eigen::Matrix4d normal = Eigen::Matrix4d::Zero();
    Eigen::Vector4d rhs = Eigen::Vector4d::Zero();
    for (const GraySample& sample : samples) {
        const double un = (sample.u - mean_u) / spread_u;
        const double vn = (sample.v - mean_v) / spread_v;
        Eigen::Vector4d row;
        row << 1.0, un, vn, un * vn;
        normal += row * row.transpose();
        rhs += row * sample.value;
    }
    const double ridge = std::max(1e-9, 1e-9 * normal.diagonal().maxCoeff());
    normal.diagonal().array() += ridge;
    const Eigen::Vector4d coefficients = normal.ldlt().solve(rhs);
    if (!coefficients.allFinite()) {
        return model;
    }
    model.c0 = coefficients(0);
    model.c1 = coefficients(1);
    model.c2 = coefficients(2);
    model.c3 = coefficients(3);
    model.mean_u = mean_u;
    model.mean_v = mean_v;
    model.scale_u = spread_u;
    model.scale_v = spread_v;
    model.valid = true;
    return model;
}

double median_of(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
    return values[middle];
}

/// One refit after dropping samples far from the first model. The reversed
/// families' dark reference is the surrounding payload ring, which contains
/// white bits; those are the outliers this rejects.
BilinearModel robust_bilinear(const std::vector<GraySample>& samples) {
    const BilinearModel first = fit_bilinear(samples);
    if (!first.valid) {
        return first;
    }
    std::vector<double> residuals;
    residuals.reserve(samples.size());
    for (const GraySample& sample : samples) {
        residuals.push_back(std::abs(sample.value - first.at(sample.u, sample.v)));
    }
    const double limit = std::max(0.02, 4.0 * median_of(residuals));
    std::vector<GraySample> kept;
    kept.reserve(samples.size());
    for (const GraySample& sample : samples) {
        if (std::abs(sample.value - first.at(sample.u, sample.v)) <= limit) {
            kept.push_back(sample);
        }
    }
    if (kept.size() + 4 < samples.size() || kept.size() < 6) {
        return first;
    }
    const BilinearModel second = fit_bilinear(kept);
    return second.valid ? second : first;
}

double cell_to_tag(double cell, int width) {
    return 2.0 * cell / static_cast<double>(width) - 1.0;
}

struct HypothesisResult {
    TagMatch match;
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    std::array<Eigen::Vector2d, 4> corners{};
    double margin = 0.0;  ///< Smallest payload/border margin, in units of contrast.
};

/// Decode one quad under one scale hypothesis. `scale` is the tag-frame
/// coordinate of the candidate's corners: 1 for the border square, and
/// total_width/width_at_border when the candidate is a reversed family's outer
/// silhouette.
std::optional<HypothesisResult> decode_hypothesis(
    const FloatImage& gray, const std::array<Eigen::Vector2d, 4>& quad,
    const TagFamily& family, double scale, int allowed_hamming) {
    const int width = family.width_at_border;
    if (width < 3 || family.bits <= 0 || family.bits > 64 ||
        static_cast<int>(family.bit_xy.size()) < family.bits) {
        return std::nullopt;
    }
    const std::array<Eigen::Vector2d, 4> tag = {
        Eigen::Vector2d(-scale, scale), Eigen::Vector2d(scale, scale),
        Eigen::Vector2d(scale, -scale), Eigen::Vector2d(-scale, -scale)};
    const std::optional<Homography> homography = fit_homography(tag, quad);
    if (!homography) {
        return std::nullopt;
    }
    const double half = 2.0 * kCellSampleHalf / static_cast<double>(width);
    const bool border_is_light = family.reversed_border;
    std::vector<GraySample> light_samples;
    std::vector<GraySample> dark_samples;
    const auto add_reference = [&](double cell_x, double cell_y, bool is_border) {
        const double u = cell_to_tag(cell_x, width);
        const double v = cell_to_tag(cell_y, width);
        double value = 0.0;
        if (!sample_cell(gray, *homography, u, v, half, half, value)) {
            return;
        }
        GraySample sample;
        sample.u = u;
        sample.v = v;
        sample.value = value;
        const bool is_light = is_border ? border_is_light : !border_is_light;
        if (is_light) {
            light_samples.push_back(sample);
        } else {
            dark_samples.push_back(sample);
        }
    };
    for (int index = 0; index < width; ++index) {
        const double centre = static_cast<double>(index) + 0.5;
        const double last = static_cast<double>(width) - 0.5;
        // The ring cells of the reference frame, then the cells one step
        // outside it. The published reference pattern samples the same lines.
        add_reference(0.5, centre, true);
        add_reference(last, centre, true);
        add_reference(centre, 0.5, true);
        add_reference(centre, last, true);
        add_reference(-0.5, centre, false);
        add_reference(static_cast<double>(width) + 0.5, centre, false);
        add_reference(centre, -0.5, false);
        add_reference(centre, static_cast<double>(width) + 0.5, false);
    }
    const std::array<double, 4> quiet_x = {
        -0.5, static_cast<double>(width) + 0.5, -0.5,
        static_cast<double>(width) + 0.5};
    const std::array<double, 4> quiet_y = {
        -0.5, -0.5, static_cast<double>(width) + 0.5,
        static_cast<double>(width) + 0.5};
    for (std::size_t i = 0; i < 4; ++i) {
        add_reference(quiet_x[i], quiet_y[i], false);
    }
    const BilinearModel light = robust_bilinear(light_samples);
    const BilinearModel dark = robust_bilinear(dark_samples);
    if (!light.valid || !dark.valid) {
        return std::nullopt;
    }
    if (!(light.at(0.0, 0.0) - dark.at(0.0, 0.0) > kMinContrast)) {
        return std::nullopt;
    }
    const auto contrast_at = [&](double u, double v) {
        return light.at(u, v) - dark.at(u, v);
    };
    const auto threshold_at = [&](double u, double v) {
        return 0.5 * (light.at(u, v) + dark.at(u, v));
    };

    double worst_margin = std::numeric_limits<double>::infinity();
    const auto check_border_cell = [&](double cell_x, double cell_y) {
        const double u = cell_to_tag(cell_x, width);
        const double v = cell_to_tag(cell_y, width);
        double value = 0.0;
        if (!sample_cell(gray, *homography, u, v, half, half, value)) {
            return false;
        }
        const double contrast = contrast_at(u, v);
        if (!(contrast > kMinContrast)) {
            return false;
        }
        const double margin =
            border_is_light
                ? (value - threshold_at(u, v)) / contrast
                : (threshold_at(u, v) - value) / contrast;
        if (!(margin > kDecisionMargin)) {
            return false;
        }
        worst_margin = std::min(worst_margin, margin);
        return true;
    };
    for (int index = 0; index < width; ++index) {
        const double centre = static_cast<double>(index) + 0.5;
        const double last = static_cast<double>(width) - 0.5;
        if (!check_border_cell(0.5, centre) || !check_border_cell(last, centre) ||
            !check_border_cell(centre, 0.5) || !check_border_cell(centre, last)) {
            return std::nullopt;
        }
    }

    std::uint64_t word = 0;
    double worst_payload = std::numeric_limits<double>::infinity();
    for (int bit_index = 0; bit_index < family.bits; ++bit_index) {
        const std::pair<int, int>& cell =
            family.bit_xy[static_cast<std::size_t>(bit_index)];
        const double u =
            cell_to_tag(static_cast<double>(cell.first) + 0.5, width);
        const double v =
            cell_to_tag(static_cast<double>(cell.second) + 0.5, width);
        double value = 0.0;
        if (!sample_cell(gray, *homography, u, v, half, half, value)) {
            return std::nullopt;
        }
        const double contrast = contrast_at(u, v);
        if (!(contrast > kMinContrast)) {
            return std::nullopt;
        }
        const double threshold = threshold_at(u, v);
        const double offset = std::abs(value - threshold);
        if (offset < kDecisionMargin * contrast) {
            return std::nullopt;
        }
        if (value > threshold) {
            word |= std::uint64_t{1} << static_cast<unsigned>(bit_index);
        }
        worst_payload = std::min(worst_payload, offset / contrast);
    }

    // A solid shape reads as an almost constant word, which is farther from
    // every code than the weight bounds allow. This cheap gate stops a small
    // foreign quad before the table search; every genuine word within
    // `allowed_hamming` of a code satisfies it.
    const int ones = static_cast<int>(std::popcount(word));
    if (ones < min_code_weight(family) - allowed_hamming ||
        ones > max_code_weight(family) + allowed_hamming) {
        return std::nullopt;
    }
    const std::optional<TagMatch> match =
        match_tag_word(word, family, allowed_hamming);
    if (!match) {
        return std::nullopt;
    }
    Eigen::Vector2d center;
    if (!homography->project(0.0, 0.0, center)) {
        return std::nullopt;
    }
    HypothesisResult result;
    result.match = *match;
    result.center = center;
    for (int corner = 0; corner < 4; ++corner) {
        // `match->rotation` is the number of T steps that turn the sampled
        // word back into the code; the observed cyclic corner order is offset
        // by the same amount, so published corner i is observed corner i+k.
        result.corners[static_cast<std::size_t>(corner)] =
            quad[static_cast<std::size_t>((corner + match->rotation) % 4)];
    }
    result.margin = std::min(worst_margin, worst_payload);
    return result;
}

/// Layout-order word to the published integer (most significant bit first).
std::uint64_t layout_to_published(std::uint64_t word, int bits) {
    std::uint64_t out = 0;
    for (int i = 0; i < bits; ++i) {
        if (((word >> static_cast<unsigned>(i)) & 1u) != 0u) {
            out |= std::uint64_t{1} << static_cast<unsigned>(bits - 1 - i);
        }
    }
    return out;
}

}  // namespace

std::optional<TagMatch> match_tag_word(std::uint64_t word,
                                       const TagFamily& family,
                                       int max_hamming) {
    if (family.codes.empty() || family.bits <= 0 || family.bits > 64 ||
        max_hamming < 0) {
        return std::nullopt;
    }
    const int allowed = std::min(max_hamming, (family.min_hamming - 1) / 2);
    if (allowed < 0) {
        return std::nullopt;
    }
    TagMatch best;
    best.distance = family.bits + 1;
    std::uint64_t current = word;
    for (int rotation = 0; rotation < 4; ++rotation) {
        const std::uint64_t published =
            layout_to_published(current, family.bits);
        for (std::size_t index = 0; index < family.codes.size(); ++index) {
            const int distance = static_cast<int>(
                std::popcount(published ^ family.codes[index]));
            if (distance < best.distance) {
                best.id = static_cast<int>(index);
                best.distance = distance;
                best.rotation = rotation;
            }
        }
        current = rotate_tag_word(current, family);
    }
    if (best.id < 0 || best.distance > allowed) {
        return std::nullopt;
    }
    return best;
}

std::optional<CodedTarget> decode_tag(const FloatImage& gray,
                                      const MarkerCandidate& quad_candidate,
                                      const TagFamily& family,
                                      int max_hamming) {
    if (gray.empty() || family.codes.empty() || family.bits <= 0 ||
        family.bits > 64 || max_hamming < 0 ||
        family.width_at_border < 3) {
        return std::nullopt;
    }
    if (quad_candidate.corners.size() != 4) {
        return std::nullopt;
    }
    std::array<Eigen::Vector2d, 4> quad{};
    for (std::size_t i = 0; i < quad.size(); ++i) {
        quad[i] = quad_candidate.corners[i];
        if (!quad[i].allFinite()) {
            return std::nullopt;
        }
    }
    double min_side = std::numeric_limits<double>::infinity();
    double max_side = 0.0;
    for (std::size_t i = 0; i < quad.size(); ++i) {
        const double side = (quad[(i + 1) % 4] - quad[i]).norm();
        min_side = std::min(min_side, side);
        max_side = std::max(max_side, side);
    }
    if (!(min_side > kMinSidePx) || max_side > 10.0 * min_side) {
        return std::nullopt;
    }
    double shoelace = 0.0;
    for (std::size_t i = 0; i < quad.size(); ++i) {
        const Eigen::Vector2d& a = quad[i];
        const Eigen::Vector2d& b = quad[(i + 1) % 4];
        shoelace += a.x() * b.y() - b.x() * a.y();
    }
    if (!(std::abs(shoelace) > 0.5 * min_side * min_side)) {
        return std::nullopt;
    }

    const GradientField gradient = compute_gradient_field(gray);
    // Two search reaches: the short one stays on a wide border's outer edge,
    // while the long one recovers a thin reference frame when the detector's
    // initial quad sits on its inner edge (the 40 px and thin-frame cases).
    std::vector<std::array<Eigen::Vector2d, 4>> refined_quads;
    // The detector's own corners are already a subpixel line fit; keep them as
    // a fallback in case a short side or a notched silhouette makes the extra
    // refinement jump to a neighbouring edge.
    refined_quads.push_back(quad);
    refined_quads.push_back(
        refine_quad_corners(gradient, quad, 0.30, 6.0, PeakChoice::Outermost));
    refined_quads.push_back(
        refine_quad_corners(gradient, quad, 0.60, 9.0, PeakChoice::Outermost));
    // The strongest-peak search can move a side inward as well as outward, which
    // recovers a reference square whose detected edge was pushed out by a white
    // payload cell touching the frame.
    refined_quads.push_back(
        refine_quad_corners(gradient, quad, 0.60, 9.0, PeakChoice::Strongest));
    const int allowed = std::min(max_hamming, (family.min_hamming - 1) / 2);

    std::vector<double> scales{1.0};
    if (family.reversed_border && family.width_at_border > 0) {
        const double silhouette =
            static_cast<double>(family.total_width) /
            static_cast<double>(family.width_at_border);
        if (silhouette > 1.0 + 1e-9) {
            scales.push_back(silhouette);
        }
    }
    const auto evaluate = [&](const std::array<Eigen::Vector2d, 4>& candidate_quad,
                              std::vector<HypothesisResult>& results) {
        for (double scale : scales) {
            std::optional<HypothesisResult> result =
                decode_hypothesis(gray, candidate_quad, family, scale, allowed);
            if (result) {
                results.push_back(std::move(*result));
            }
        }
    };
    const auto choose = [](const std::vector<HypothesisResult>& results)
        -> std::optional<HypothesisResult> {
        if (results.empty()) {
            return std::nullopt;
        }
        int minimum = results.front().match.distance;
        for (const HypothesisResult& result : results) {
            minimum = std::min(minimum, result.match.distance);
        }
        std::optional<HypothesisResult> chosen;
        for (const HypothesisResult& result : results) {
            if (result.match.distance != minimum) {
                continue;
            }
            if (!chosen) {
                chosen = result;
                continue;
            }
            if (chosen->match.id != result.match.id) {
                return std::nullopt;  // equally close to two codes
            }
            if (result.margin > chosen->margin) {
                chosen = result;
            }
        }
        return chosen;
    };

    std::vector<HypothesisResult> results;
    for (const std::array<Eigen::Vector2d, 4>& refined : refined_quads) {
        evaluate(refined, results);
    }
    std::optional<HypothesisResult> best = choose(results);
    if (!best && family.reversed_border && family.total_width > 0) {
        // A reversed status tag's dark outer ring is only two cells wide. A
        // white payload cell on the perimeter can cut the extreme row or
        // column off the detected component, shrinking the candidate by one
        // cell on that side. The published white reference frame is always
        // intact, so grow each side by one cell in turn and retry; only the
        // side that was missing produces a valid tag.
        for (const std::array<Eigen::Vector2d, 4>& refined : refined_quads) {
            const double pair0 = 0.5 * ((refined[1] - refined[0]).norm() +
                                        (refined[2] - refined[3]).norm());
            const double pair1 = 0.5 * ((refined[2] - refined[1]).norm() +
                                        (refined[3] - refined[0]).norm());
            const double cell = std::max(pair0, pair1) /
                                static_cast<double>(family.total_width);
            Eigen::Vector2d center = Eigen::Vector2d::Zero();
            for (const Eigen::Vector2d& corner : refined) {
                center += corner;
            }
            center *= 0.25;
            for (int side = 0; side < 4; ++side) {
                const Eigen::Vector2d& a =
                    refined[static_cast<std::size_t>(side)];
                const Eigen::Vector2d& b =
                    refined[static_cast<std::size_t>((side + 1) % 4)];
                const Eigen::Vector2d direction = (b - a).normalized();
                Eigen::Vector2d normal(-direction.y(), direction.x());
                if ((0.5 * (a + b) - center).dot(normal) < 0.0) {
                    normal = -normal;
                }
                std::array<Eigen::Vector2d, 4> grown = refined;
                grown[static_cast<std::size_t>(side)] = a + normal * cell;
                grown[static_cast<std::size_t>((side + 1) % 4)] =
                    b + normal * cell;
                evaluate(grown, results);
            }
        }
        best = choose(results);
    }
    if (!best) {
        return std::nullopt;
    }

    CodedTarget target;
    target.code = best->match.id;
    target.center = best->center;
    target.bits = family.bits;
    target.hamming = best->match.distance;
    const double diagonal0 = (best->corners[2] - best->corners[0]).norm();
    const double diagonal1 = (best->corners[3] - best->corners[1]).norm();
    target.radius_px = 0.25 * (diagonal0 + diagonal1);
    target.corners = best->corners;
    target.has_corners = true;
    return target;
}

}  // namespace alphashape
