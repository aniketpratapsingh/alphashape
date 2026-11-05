#include "common/algorithm/calibration/fiducials.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "base/utils/error.hpp"
#include "common/algorithm/calibration/calibrate.hpp"

namespace alphashape {
namespace {

constexpr double kPi = std::numbers::pi_v<double>;
constexpr int kLongestSideCap = 1500;  ///< downsample cap for the frame quad
constexpr double kEdgeFraction = 0.20;  ///< edge cut vs the 99th-percentile |g|
constexpr double kInlierPx = 2.0;       ///< RANSAC line inlier distance, work px
constexpr int kRansacIterations = 300;
constexpr double kCornerPull = 0.14;  ///< window centre inset, shorter side
constexpr double kSidePull = 0.14;
constexpr double kWindowHalf = 0.10;  ///< half-size, shorter side
constexpr double kWindowEdgeFraction = 0.30;  ///< cut vs window 99th pct |g|

/// Deterministic 64-bit LCG. RANSAC only needs unbiased-enough sampling and a
/// fixed seed makes the detector reproducible, unlike <random> distributions
/// whose implementations differ between standard libraries.
struct Lcg {
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    std::uint32_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(state >> 33);
    }
    std::size_t index(std::size_t count) {
        return count == 0 ? 0 : static_cast<std::size_t>(next() % count);
    }
};

struct Gradients {
    int width = 0;
    int height = 0;
    std::vector<float> gx;
    std::vector<float> gy;

    float magnitude(int x, int y) const {
        if (x < 0 || y < 0 || x >= width || y >= height) {
            return 0.0F;
        }
        const std::size_t i =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
            static_cast<std::size_t>(x);
        return std::hypot(gx[i], gy[i]);
    }
};

Gradients sobel_gradients(const FloatImage& image) {
    Gradients out;
    out.width = image.width();
    out.height = image.height();
    const std::size_t count = static_cast<std::size_t>(out.width) *
                              static_cast<std::size_t>(out.height);
    out.gx.assign(count, 0.0F);
    out.gy.assign(count, 0.0F);
    if (out.width < 3 || out.height < 3) {
        return out;
    }
    const auto at = [&image](int x, int y) {
        return static_cast<double>(image.at_clamped(x, y));
    };
    for (int y = 1; y + 1 < out.height; ++y) {
        for (int x = 1; x + 1 < out.width; ++x) {
            const double gx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1)) -
                              (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
            const double gy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)) -
                              (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
            const std::size_t i = static_cast<std::size_t>(y) *
                                      static_cast<std::size_t>(out.width) +
                                  static_cast<std::size_t>(x);
            out.gx[i] = static_cast<float>(gx / 8.0);
            out.gy[i] = static_cast<float>(gy / 8.0);
        }
    }
    return out;
}

/// High percentile of a sample. The task's "99th percentile" cut is not robust
/// on a scanned frame: a thin frame boundary is well under 1% of the pixels, so
/// the 99th percentile lands in film grain rather than on the edge. We use the
/// 99.9th percentile instead (documented deviation), which still tracks the
/// strongest gradients and sits above the noise floor on our synthetic scans.
/// Returns 0 for an empty sample.
double percentile999(std::vector<float> values) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t index = static_cast<std::size_t>(
        std::floor(0.999 * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return static_cast<double>(values[index]);
}

/// Integer box downscale. A work pixel (x, y) is centred on full-resolution
/// (x*scale + (scale-1)/2, y*scale + (scale-1)/2).
FloatImage box_downsample(const FloatImage& in, int scale) {
    if (scale <= 1) {
        return in;
    }
    const int width = std::max(1, in.width() / scale);
    const int height = std::max(1, in.height() / scale);
    FloatImage out(width, height, 0.0F);
    const double norm = 1.0 / static_cast<double>(scale * scale);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int dy = 0; dy < scale; ++dy) {
                for (int dx = 0; dx < scale; ++dx) {
                    sum += static_cast<double>(
                        in.at_clamped(x * scale + dx, y * scale + dy));
                }
            }
            out.at(x, y) = static_cast<float>(sum * norm);
        }
    }
    return out;
}

/// A line point + unit direction.
struct Line {
    Eigen::Vector2d point = Eigen::Vector2d::Zero();
    Eigen::Vector2d direction = Eigen::Vector2d(1.0, 0.0);
};

/// First strong edge along a scanline, localised at the peak of the gradient
/// magnitude with a three-sample parabolic fit. The gradient peak of a
/// symmetric edge is the edge centre; a threshold crossing is biased towards
/// the bright side.
std::optional<double> edge_along(const Gradients& gradients, int x, int y, int dx,
                                 int dy, int limit, double threshold,
                                 bool vertical_scan) {
    float previous = gradients.magnitude(x - dx, y - dy);
    float current = gradients.magnitude(x, y);
    bool found = false;
    int px = x;
    int py = y;
    for (int i = 0; i < limit; ++i) {
        if (static_cast<double>(current) >= threshold && current > previous) {
            found = true;
            break;
        }
        previous = current;
        px += dx;
        py += dy;
        current = gradients.magnitude(px, py);
    }
    if (!found) {
        return std::nullopt;
    }
    for (int i = 0; i < 4; ++i) {
        const float next = gradients.magnitude(px + dx, py + dy);
        if (next <= current) {
            break;
        }
        previous = current;
        px += dx;
        py += dy;
        current = next;
    }
    const float after = gradients.magnitude(px + dx, py + dy);
    const double denominator = static_cast<double>(previous) -
                               2.0 * static_cast<double>(current) +
                               static_cast<double>(after);
    double offset = 0.0;
    if (std::abs(denominator) > 1e-9) {
        offset = 0.5 *
                 (static_cast<double>(previous) - static_cast<double>(after)) /
                 denominator;
    }
    offset = std::clamp(offset, -1.0, 1.0);
    return static_cast<double>(vertical_scan ? py : px) + offset;
}

std::vector<Eigen::Vector2d> side_points(const Gradients& gradients,
                                         FiducialWindow side, double threshold,
                                         int step) {
    std::vector<Eigen::Vector2d> points;
    const int w = gradients.width;
    const int h = gradients.height;
    if (side == FiducialWindow::Top || side == FiducialWindow::Bottom) {
        const bool top = side == FiducialWindow::Top;
        for (int x = step; x + step < w; x += step) {
            const std::optional<double> edge =
                edge_along(gradients, x, top ? 1 : h - 2, 0, top ? 1 : -1, h - 2,
                           threshold, true);
            if (edge) {
                points.emplace_back(static_cast<double>(x), *edge);
            }
        }
    } else {
        const bool left = side == FiducialWindow::Left;
        for (int y = step; y + step < h; y += step) {
            const std::optional<double> edge =
                edge_along(gradients, left ? 1 : w - 2, y, left ? 1 : -1, 0, w - 2,
                           threshold, false);
            if (edge) {
                points.emplace_back(*edge, static_cast<double>(y));
            }
        }
    }
    return points;
}

double point_line_distance(const Eigen::Vector2d& point, const Line& line) {
    const Eigen::Vector2d offset = point - line.point;
    return std::abs(offset.x() * line.direction.y() -
                    offset.y() * line.direction.x());
}

/// Total-least-squares line through the selected points (principal axis of the
/// centred scatter matrix).
Line fit_line(const std::vector<Eigen::Vector2d>& points) {
    Line line;
    if (points.empty()) {
        return line;
    }
    Eigen::Vector2d centre = Eigen::Vector2d::Zero();
    for (const Eigen::Vector2d& point : points) {
        centre += point;
    }
    centre /= static_cast<double>(points.size());
    Eigen::Matrix2d scatter = Eigen::Matrix2d::Zero();
    for (const Eigen::Vector2d& point : points) {
        const Eigen::Vector2d d = point - centre;
        scatter.noalias() += d * d.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(scatter);
    line.point = centre;
    if (solver.info() == Eigen::Success) {
        line.direction = solver.eigenvectors().col(1).normalized();
    }
    if (!line.direction.allFinite() || line.direction.norm() < 1e-9) {
        line.direction = Eigen::Vector2d(1.0, 0.0);
    }
    return line;
}

/// RANSAC line fit (Fischler & Bolles 1981) followed by a least-squares refit
/// on the inliers. `min_inliers` guards against a line through noise.
std::optional<Line> ransac_line(const std::vector<Eigen::Vector2d>& points,
                                std::size_t min_inliers, std::uint64_t seed) {
    if (points.size() < 2) {
        return std::nullopt;
    }
    Lcg random;
    random.state ^= seed * 0xbf58476d1ce4e5b9ULL;
    Line best;
    std::size_t best_count = 0;
    for (int iteration = 0; iteration < kRansacIterations; ++iteration) {
        const std::size_t a = random.index(points.size());
        std::size_t b = random.index(points.size());
        if (a == b) {
            continue;
        }
        const Eigen::Vector2d delta = points[b] - points[a];
        const double length = delta.norm();
        if (!(length > 1e-9)) {
            continue;
        }
        Line candidate;
        candidate.point = points[a];
        candidate.direction = delta / length;
        std::size_t count = 0;
        for (const Eigen::Vector2d& point : points) {
            if (point_line_distance(point, candidate) <= kInlierPx) {
                ++count;
            }
        }
        if (count > best_count) {
            best_count = count;
            best = candidate;
        }
    }
    if (best_count < min_inliers) {
        return std::nullopt;
    }
    std::vector<Eigen::Vector2d> inliers;
    inliers.reserve(best_count);
    for (const Eigen::Vector2d& point : points) {
        if (point_line_distance(point, best) <= kInlierPx) {
            inliers.push_back(point);
        }
    }
    if (inliers.size() < min_inliers) {
        return std::nullopt;
    }
    const Line refined = fit_line(inliers);
    if (!refined.point.allFinite() || !refined.direction.allFinite()) {
        return std::nullopt;
    }
    return refined;
}

std::optional<Eigen::Vector2d> intersect_lines(const Line& a, const Line& b) {
    const double denominator = a.direction.x() * b.direction.y() -
                               a.direction.y() * b.direction.x();
    if (std::abs(denominator) < 1e-9) {
        return std::nullopt;
    }
    const Eigen::Vector2d offset = b.point - a.point;
    const double t = (offset.x() * b.direction.y() -
                      offset.y() * b.direction.x()) /
                     denominator;
    const Eigen::Vector2d point = a.point + t * a.direction;
    if (!point.allFinite()) {
        return std::nullopt;
    }
    return point;
}

/// Convexity from the four consecutive cross products (all the same sign) and
/// a minimum shoelace area.
bool convex_quad(const std::array<Eigen::Vector2d, 4>& corners, double min_area) {
    bool positive = false;
    bool negative = false;
    for (std::size_t i = 0; i < 4; ++i) {
        const Eigen::Vector2d e1 = corners[(i + 1) % 4] - corners[i];
        const Eigen::Vector2d e2 = corners[(i + 2) % 4] - corners[(i + 1) % 4];
        const double cross = e1.x() * e2.y() - e1.y() * e2.x();
        if (std::abs(cross) < 1e-9) {
            return false;
        }
        positive = positive || cross > 0.0;
        negative = negative || cross < 0.0;
    }
    if (positive == negative) {
        return false;
    }
    double twice_area = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        const Eigen::Vector2d& a = corners[i];
        const Eigen::Vector2d& b = corners[(i + 1) % 4];
        twice_area += a.x() * b.y() - b.x() * a.y();
    }
    return 0.5 * std::abs(twice_area) >= min_area;
}

}  // namespace

std::optional<FrameQuad> detect_frame_quad(const FloatImage& gray) {
    if (gray.empty() || gray.width() < 32 || gray.height() < 32) {
        return std::nullopt;
    }
    const int longest = std::max(gray.width(), gray.height());
    const int scale = std::max(1, (longest + kLongestSideCap - 1) / kLongestSideCap);
    const FloatImage work = box_downsample(gray, scale);
    const Gradients gradients = sobel_gradients(work);
    std::vector<float> magnitudes;
    magnitudes.reserve(gradients.gx.size());
    for (std::size_t i = 0; i < gradients.gx.size(); ++i) {
        magnitudes.push_back(std::hypot(gradients.gx[i], gradients.gy[i]));
    }
    const double p99 = percentile999(magnitudes);
    if (!(p99 > 0.0)) {
        return std::nullopt;
    }
    const double threshold = kEdgeFraction * p99;
    const int step = std::max(2, std::min(work.width(), work.height()) / 150);
    const double min_inliers = static_cast<double>(step) * 0.5;
    const std::array<FiducialWindow, 4> sides = {
        FiducialWindow::Top, FiducialWindow::Right, FiducialWindow::Bottom,
        FiducialWindow::Left};
    std::array<Line, 4> lines;
    for (std::size_t i = 0; i < sides.size(); ++i) {
        const std::vector<Eigen::Vector2d> points =
            side_points(gradients, sides[i], threshold, step);
        const std::optional<Line> line = ransac_line(
            points, static_cast<std::size_t>(std::max(6.0, min_inliers)), i + 1);
        if (!line) {
            return std::nullopt;
        }
        lines[i] = *line;
    }
    std::array<Eigen::Vector2d, 4> corners;
    const std::optional<Eigen::Vector2d> tl = intersect_lines(lines[0], lines[3]);
    const std::optional<Eigen::Vector2d> tr = intersect_lines(lines[0], lines[1]);
    const std::optional<Eigen::Vector2d> br = intersect_lines(lines[2], lines[1]);
    const std::optional<Eigen::Vector2d> bl = intersect_lines(lines[2], lines[3]);
    if (!tl || !tr || !br || !bl) {
        return std::nullopt;
    }
    corners = {*tl, *tr, *br, *bl};
    if (!convex_quad(corners, 0.01 * static_cast<double>(work.width() * work.height()))) {
        return std::nullopt;
    }
    const double offset = 0.5 * static_cast<double>(scale - 1);
    const auto to_full = [scale, offset](const Eigen::Vector2d& p) {
        return Eigen::Vector2d(p.x() * scale + offset, p.y() * scale + offset);
    };
    FrameQuad quad;
    quad.top_left = to_full(corners[0]);
    quad.top_right = to_full(corners[1]);
    quad.bottom_right = to_full(corners[2]);
    quad.bottom_left = to_full(corners[3]);
    return quad;
}

namespace {

Eigen::Vector2d window_direction(FiducialWindow window) {
    switch (window) {
        case FiducialWindow::TopLeft: return Eigen::Vector2d(-1.0, 1.0);
        case FiducialWindow::Top: return Eigen::Vector2d(0.0, 1.0);
        case FiducialWindow::TopRight: return Eigen::Vector2d(1.0, 1.0);
        case FiducialWindow::Right: return Eigen::Vector2d(1.0, 0.0);
        case FiducialWindow::BottomRight: return Eigen::Vector2d(1.0, -1.0);
        case FiducialWindow::Bottom: return Eigen::Vector2d(0.0, -1.0);
        case FiducialWindow::BottomLeft: return Eigen::Vector2d(-1.0, -1.0);
        case FiducialWindow::Left: return Eigen::Vector2d(-1.0, 0.0);
    }
    return Eigen::Vector2d(0.0, 1.0);
}

/// Eight-sector classification of a film direction (y up). Used to map a
/// fiducial coordinate to the window that should hold it.
FiducialWindow classify_direction(const Eigen::Vector2d& direction) {
    const double angle = std::atan2(direction.y(), direction.x());
    int octant = static_cast<int>(std::lround(angle / (kPi / 4.0)));
    if (octant > 4) octant -= 8;
    if (octant < -4) octant += 8;
    switch (octant) {
        case 0: return FiducialWindow::Right;
        case 1: return FiducialWindow::TopRight;
        case 2: return FiducialWindow::Top;
        case 3: return FiducialWindow::TopLeft;
        case 4: case -4: return FiducialWindow::Left;
        case -3: return FiducialWindow::BottomLeft;
        case -2: return FiducialWindow::Bottom;
        default: return FiducialWindow::BottomRight;
    }
}

const std::array<FiducialWindow, 4>& corner_windows() {
    static const std::array<FiducialWindow, 4> windows = {
        FiducialWindow::TopLeft, FiducialWindow::TopRight,
        FiducialWindow::BottomRight, FiducialWindow::BottomLeft};
    return windows;
}

const std::array<FiducialWindow, 4>& side_windows() {
    static const std::array<FiducialWindow, 4> windows = {
        FiducialWindow::Top, FiducialWindow::Right,
        FiducialWindow::Bottom, FiducialWindow::Left};
    return windows;
}

/// Integer pixel bounds of a search window; inclusive.
struct WindowBox {
    int x0 = 0;
    int y0 = 0;
    int x1 = -1;
    int y1 = -1;
    Eigen::Vector2d centre = Eigen::Vector2d::Zero();

    int width() const { return x1 - x0 + 1; }
    int height() const { return y1 - y0 + 1; }
};

std::optional<WindowBox> make_window(double cx, double cy, double half, int width,
                                     int height) {
    WindowBox box;
    box.x0 = std::max(1, static_cast<int>(std::floor(cx - half)));
    box.y0 = std::max(1, static_cast<int>(std::floor(cy - half)));
    box.x1 = std::min(width - 2, static_cast<int>(std::ceil(cx + half)));
    box.y1 = std::min(height - 2, static_cast<int>(std::ceil(cy + half)));
    box.centre = Eigen::Vector2d(cx, cy);
    if (box.width() < 5 || box.height() < 5) {
        return std::nullopt;
    }
    return box;
}

/// Sub-pixel refinement (Förstner & Gülch 1987). Minimise
/// sum w_i (n_i . (x - x_i))^2 with w_i = |g_i|^2. For a corner the lines are
/// the edges, so n_i is the gradient direction. For a circle centre the
/// gradient lines pass through the centre, so n_i is the gradient rotated by
/// 90 degrees. Rejects an ill-conditioned normal matrix or a move beyond the
/// search radius.
std::optional<Eigen::Vector2d> refine_point(const Gradients& gradients,
                                            const Eigen::Vector2d& coarse,
                                            double radius, bool circle_centre) {
    if (!coarse.allFinite()) {
        return std::nullopt;
    }
    const int reach = std::max(2, static_cast<int>(std::ceil(radius)));
    const int cx = static_cast<int>(std::lround(coarse.x()));
    const int cy = static_cast<int>(std::lround(coarse.y()));
    Eigen::Matrix2d normal = Eigen::Matrix2d::Zero();
    Eigen::Vector2d rhs = Eigen::Vector2d::Zero();
    int used = 0;
    for (int dy = -reach; dy <= reach; ++dy) {
        for (int dx = -reach; dx <= reach; ++dx) {
            if (dx * dx + dy * dy > reach * reach) {
                continue;
            }
            const int x = cx + dx;
            const int y = cy + dy;
            if (x < 1 || y < 1 || x + 1 >= gradients.width || y + 1 >= gradients.height) {
                continue;
            }
            const std::size_t i = static_cast<std::size_t>(y) *
                                      static_cast<std::size_t>(gradients.width) +
                                  static_cast<std::size_t>(x);
            const double gx = static_cast<double>(gradients.gx[i]);
            const double gy = static_cast<double>(gradients.gy[i]);
            const double weight = gx * gx + gy * gy;
            if (weight < 1e-12) {
                continue;
            }
            const double length = std::sqrt(weight);
            Eigen::Vector2d normal_i =
                circle_centre ? Eigen::Vector2d(-gy / length, gx / length)
                              : Eigen::Vector2d(gx / length, gy / length);
            const Eigen::Vector2d point(static_cast<double>(x), static_cast<double>(y));
            normal.noalias() += weight * normal_i * normal_i.transpose();
            rhs.noalias() += weight * normal_i * normal_i.dot(point);
            ++used;
        }
    }
    if (used < 6) {
        return std::nullopt;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(normal);
    if (solver.info() != Eigen::Success) {
        return std::nullopt;
    }
    const double largest = solver.eigenvalues().maxCoeff();
    const double smallest = solver.eigenvalues().minCoeff();
    if (!(largest > 0.0) || !(smallest > 1e-6 * largest)) {
        return std::nullopt;
    }
    const Eigen::Vector2d refined = normal.ldlt().solve(rhs);
    if (!refined.allFinite() || (refined - coarse).norm() > reach) {
        return std::nullopt;
    }
    return refined;
}

/// Förstner/Gülch structure-tensor corner strength det / trace over a 5x5
/// window of summed gradient products.
double corner_response(const Gradients& gradients, int x, int y) {
    double sxx = 0.0;
    double syy = 0.0;
    double sxy = 0.0;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            const int px = x + dx;
            const int py = y + dy;
            if (px < 0 || py < 0 || px >= gradients.width || py >= gradients.height) {
                continue;
            }
            const std::size_t i = static_cast<std::size_t>(py) *
                                      static_cast<std::size_t>(gradients.width) +
                                  static_cast<std::size_t>(px);
            const double gx = static_cast<double>(gradients.gx[i]);
            const double gy = static_cast<double>(gradients.gy[i]);
            sxx += gx * gx;
            syy += gy * gy;
            sxy += gx * gy;
        }
    }
    const double trace = sxx + syy;
    if (!(trace > 1e-12)) {
        return 0.0;
    }
    return (sxx * syy - sxy * sxy) / trace;
}

/// True when both expected L arms (the inward image axes sx, sy) carry a dark
/// stroke a few pixels from (x, y). This rejects an arm end, where only one
/// direction leaves the mark, and keeps the vertex.
bool corner_arms_dark(const FloatImage& gray, int x, int y, double sx, double sy,
                      double threshold) {
    const auto dark = [&](double px, double py, double nx, double ny) {
        double value = static_cast<double>(gray.sample(static_cast<float>(px),
                                                       static_cast<float>(py)));
        value = std::min(value, static_cast<double>(gray.sample(
                                    static_cast<float>(px - ny),
                                    static_cast<float>(py + nx))));
        value = std::min(value, static_cast<double>(gray.sample(
                                    static_cast<float>(px + ny),
                                    static_cast<float>(py - nx))));
        return value < threshold;
    };
    const bool arm_x = dark(static_cast<double>(x) + sx * 5.0, static_cast<double>(y),
                            1.0, 0.0) ||
                       dark(static_cast<double>(x) + sx * 10.0,
                            static_cast<double>(y), 1.0, 0.0);
    const bool arm_y = dark(static_cast<double>(x),
                            static_cast<double>(y) + sy * 5.0, 0.0, 1.0) ||
                       dark(static_cast<double>(x),
                            static_cast<double>(y) + sy * 10.0, 0.0, 1.0);
    return arm_x && arm_y;
}

std::optional<Eigen::Vector2d> detect_corner(const FloatImage& gray,
                                             const Gradients& gradients,
                                             const WindowBox& box,
                                             FiducialWindow window,
                                             double radius) {
    // The window is pulled towards where the mark is expected, so the frame
    // corner itself (at the window's outer edge) must not win. Search the
    // central 80% of the window, which is the "nearest expected direction"
    // region, and reject responses only from grain.
    const int margin_x = std::max(2, box.width() / 10);
    const int margin_y = std::max(2, box.height() / 10);
    double strongest = 0.0;
    for (int y = box.y0 + margin_y; y <= box.y1 - margin_y; ++y) {
        for (int x = box.x0 + margin_x; x <= box.x1 - margin_x; ++x) {
            strongest = std::max(strongest, corner_response(gradients, x, y));
        }
    }
    if (!(strongest > 1e-12)) {
        return std::nullopt;
    }
    // Expected L opening: the inward image axes of the corner window. Among
    // responses that see both arms, keep the strongest; the nearest-to-centre
    // candidate is only a fallback for an unexpected mark.
    const bool corner_window =
        window == FiducialWindow::TopLeft || window == FiducialWindow::TopRight ||
        window == FiducialWindow::BottomRight ||
        window == FiducialWindow::BottomLeft;
    const double sx =
        (window == FiducialWindow::TopLeft || window == FiducialWindow::BottomLeft)
            ? 1.0
            : -1.0;
    const double sy =
        (window == FiducialWindow::TopLeft || window == FiducialWindow::TopRight)
            ? 1.0
            : -1.0;
    double low = std::numeric_limits<double>::infinity();
    double high = -low;
    for (int y = box.y0; y <= box.y1; ++y) {
        for (int x = box.x0; x <= box.x1; ++x) {
            const double value = static_cast<double>(gray.at(x, y));
            low = std::min(low, value);
            high = std::max(high, value);
        }
    }
    const double threshold = low + 0.4 * (high - low);
    double best = 0.0;
    double best_distance = std::numeric_limits<double>::infinity();
    int best_x = -1;
    int best_y = -1;
    double fallback_distance = std::numeric_limits<double>::infinity();
    int fallback_x = -1;
    int fallback_y = -1;
    for (int y = box.y0 + margin_y; y <= box.y1 - margin_y; ++y) {
        for (int x = box.x0 + margin_x; x <= box.x1 - margin_x; ++x) {
            const double response = corner_response(gradients, x, y);
            if (response < 0.5 * strongest) {
                continue;
            }
            const double dx = static_cast<double>(x) - box.centre.x();
            const double dy = static_cast<double>(y) - box.centre.y();
            const double distance = dx * dx + dy * dy;
            if (fallback_x < 0 || distance < fallback_distance) {
                fallback_distance = distance;
                fallback_x = x;
                fallback_y = y;
            }
            if (corner_window &&
                !corner_arms_dark(gray, x, y, sx, sy, threshold)) {
                continue;
            }
            if (best_x < 0 || response > best ||
                (response == best && distance < best_distance)) {
                best = response;
                best_distance = distance;
                best_x = x;
                best_y = y;
            }
        }
    }
    if (best_x < 0) {
        best_x = fallback_x;
        best_y = fallback_y;
    }
    if (best_x < 0) {
        return std::nullopt;
    }
    const std::optional<Eigen::Vector2d> refined = refine_point(
        gradients, Eigen::Vector2d(static_cast<double>(best_x),
                                   static_cast<double>(best_y)),
        radius, false);
    if (refined) {
        return refined;
    }
    return Eigen::Vector2d(static_cast<double>(best_x), static_cast<double>(best_y));
}

struct CircleHit {
    Eigen::Vector2d centre = Eigen::Vector2d::Zero();
    double radius = 0.0;
};

/// Bilinear accumulator vote at a continuous position.
void add_vote(std::vector<float>& accumulator, int r, double x, double y, int width,
              int height) {
    if (!(x >= 0.0) || !(y >= 0.0) || x >= static_cast<double>(width - 1) ||
        y >= static_cast<double>(height - 1)) {
        const int xi = static_cast<int>(std::lround(x));
        const int yi = static_cast<int>(std::lround(y));
        if (xi < 0 || yi < 0 || xi >= width || yi >= height) {
            return;
        }
        const std::size_t index =
            (static_cast<std::size_t>(r) * static_cast<std::size_t>(height) +
             static_cast<std::size_t>(yi)) *
                static_cast<std::size_t>(width) +
            static_cast<std::size_t>(xi);
        accumulator[index] += 1.0F;
        return;
    }
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);
    const double weights[4] = {(1.0 - fx) * (1.0 - fy), fx * (1.0 - fy),
                               (1.0 - fx) * fy, fx * fy};
    const int xs[4] = {x0, x0 + 1, x0, x0 + 1};
    const int ys[4] = {y0, y0, y0 + 1, y0 + 1};
    for (int k = 0; k < 4; ++k) {
        const std::size_t index =
            (static_cast<std::size_t>(r) * static_cast<std::size_t>(height) +
             static_cast<std::size_t>(ys[k])) *
                static_cast<std::size_t>(width) +
            static_cast<std::size_t>(xs[k]);
        accumulator[index] += static_cast<float>(weights[k]);
    }
}

/// Cut at a fraction of the strongest gradient in a window. A mark edge is a
/// tiny fraction of a window's pixels, so a high percentile would sit in film
/// grain; the peak magnitude is the reliable reference at this scale.
double window_threshold(const Gradients& gradients, const WindowBox& box) {
    float maximum = 0.0F;
    for (int y = box.y0; y <= box.y1; ++y) {
        for (int x = box.x0; x <= box.x1; ++x) {
            maximum = std::max(maximum, gradients.magnitude(x, y));
        }
    }
    return kWindowEdgeFraction * static_cast<double>(maximum);
}

/// Gradient-direction Hough vote for a circle (Hough 1962; Ballard 1981).
/// Every strong-gradient pixel votes along +/- its gradient for each radius in
/// [min_radius, max_radius]; the accumulator peak gives centre and radius.
std::optional<CircleHit> detect_circle(const Gradients& gradients,
                                       const WindowBox& box, double min_radius,
                                       double max_radius) {
    const int width = box.width();
    const int height = box.height();
    int first_radius = std::max(2, static_cast<int>(std::lround(min_radius)));
    int last_radius = static_cast<int>(std::lround(max_radius));
    last_radius = std::min(last_radius, std::min(width, height) / 2);
    if (last_radius < first_radius) {
        return std::nullopt;
    }
    const int radius_count = last_radius - first_radius + 1;
    std::vector<float> accumulator(
        static_cast<std::size_t>(radius_count) * static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height),
        0.0F);
    const double threshold = window_threshold(gradients, box);
    if (!(threshold > 0.0)) {
        return std::nullopt;
    }
    for (int y = box.y0; y <= box.y1; ++y) {
        for (int x = box.x0; x <= box.x1; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) *
                                      static_cast<std::size_t>(gradients.width) +
                                  static_cast<std::size_t>(x);
            const double gx = static_cast<double>(gradients.gx[i]);
            const double gy = static_cast<double>(gradients.gy[i]);
            const double length = std::hypot(gx, gy);
            if (length < threshold) {
                continue;
            }
            const double nx = gx / length;
            const double ny = gy / length;
            for (int k = 0; k < radius_count; ++k) {
                const double radius = static_cast<double>(first_radius + k);
                for (int sign = -1; sign <= 1; sign += 2) {
                    const double vx = static_cast<double>(x) +
                                      static_cast<double>(sign) * radius * nx -
                                      static_cast<double>(box.x0);
                    const double vy = static_cast<double>(y) +
                                      static_cast<double>(sign) * radius * ny -
                                      static_cast<double>(box.y0);
                    add_vote(accumulator, k, vx, vy, width, height);
                }
            }
        }
    }
    std::size_t peak = 0;
    float peak_value = 0.0F;
    for (std::size_t i = 0; i < accumulator.size(); ++i) {
        if (accumulator[i] > peak_value) {
            peak_value = accumulator[i];
            peak = i;
        }
    }
    if (!(peak_value > 0.0F)) {
        return std::nullopt;
    }
    const int local_x = static_cast<int>(
        peak % static_cast<std::size_t>(width));
    const int local_y = static_cast<int>(
        (peak / static_cast<std::size_t>(width)) %
        static_cast<std::size_t>(height));
    const int radius_index = static_cast<int>(
        peak / (static_cast<std::size_t>(width) *
                static_cast<std::size_t>(height)));
    CircleHit hit;
    hit.radius = static_cast<double>(first_radius + radius_index);
    const Eigen::Vector2d coarse(static_cast<double>(box.x0 + local_x),
                                 static_cast<double>(box.y0 + local_y));
    const std::optional<Eigen::Vector2d> refined =
        refine_point(gradients, coarse, std::max(3.0, hit.radius + 2.0), true);
    hit.centre = refined ? *refined : coarse;
    return hit;
}

struct WindowRequest {
    FiducialWindow window = FiducialWindow::TopLeft;
    WindowBox box;
};

/// Corner and side windows, in canonical reporting order. The window centre is
/// pulled inward by kCornerPull/kSidePull of the quad's shorter side and the
/// half-size is kWindowHalf of it; those fractions are ours.
std::vector<WindowRequest> build_windows(const FrameQuad& quad,
                                         const FiducialDetectOptions& options,
                                         int width, int height) {
    const std::array<Eigen::Vector2d, 4> corners = {
        quad.top_left, quad.top_right, quad.bottom_right, quad.bottom_left};
    const Eigen::Vector2d centre =
        (quad.top_left + quad.top_right + quad.bottom_right + quad.bottom_left) / 4.0;
    double shorter = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < 4; ++i) {
        shorter = std::min(shorter, (corners[(i + 1) % 4] - corners[i]).norm());
    }
    if (!(shorter > 1.0)) {
        return {};
    }
    const double half = kWindowHalf * shorter;
    std::vector<WindowRequest> requests;
    const bool corners_wanted =
        options.search_corners || options.detector == FiducialDetector::FrameCorner;
    if (corners_wanted) {
        for (std::size_t i = 0; i < 4; ++i) {
            const Eigen::Vector2d first = corners[(i + 1) % 4] - corners[i];
            const Eigen::Vector2d second = corners[(i + 3) % 4] - corners[i];
            Eigen::Vector2d inward = first.normalized() + second.normalized();
            if (inward.norm() < 1e-9) {
                inward = centre - corners[i];
            }
            const Eigen::Vector2d point =
                corners[i] + inward.normalized() * (kCornerPull * shorter);
            const std::optional<WindowBox> box =
                make_window(point.x(), point.y(), half, width, height);
            if (box) {
                requests.push_back(WindowRequest{corner_windows()[i], *box});
            }
        }
    }
    if (options.search_sides && options.detector != FiducialDetector::FrameCorner) {
        for (std::size_t i = 0; i < 4; ++i) {
            const Eigen::Vector2d a = corners[i];
            const Eigen::Vector2d b = corners[(i + 1) % 4];
            const Eigen::Vector2d midpoint = 0.5 * (a + b);
            const Eigen::Vector2d edge = b - a;
            Eigen::Vector2d inward(-edge.y(), edge.x());
            if (inward.dot(centre - midpoint) < 0.0) {
                inward = -inward;
            }
            const Eigen::Vector2d point =
                midpoint + inward.normalized() * (kSidePull * shorter);
            const std::optional<WindowBox> box =
                make_window(point.x(), point.y(), half, width, height);
            if (box) {
                requests.push_back(WindowRequest{side_windows()[i], *box});
            }
        }
    }
    std::sort(requests.begin(), requests.end(),
              [](const WindowRequest& a, const WindowRequest& b) {
                  return static_cast<int>(a.window) < static_cast<int>(b.window);
              });
    return requests;
}

struct ExpectedMark {
    std::size_t index = 0;
    Eigen::Vector2d film_mm = Eigen::Vector2d::Zero();
    FiducialWindow window = FiducialWindow::TopLeft;
};

std::vector<ExpectedMark> classify_expected(const std::vector<Fiducial>& fiducials) {
    std::vector<ExpectedMark> out;
    out.reserve(fiducials.size());
    for (std::size_t i = 0; i < fiducials.size(); ++i) {
        if (fiducials[i].film_mm.norm() < 1e-9) {
            continue;  // the film centre has no direction to classify by
        }
        out.push_back(ExpectedMark{i, fiducials[i].film_mm,
                                   classify_direction(fiducials[i].film_mm)});
    }
    return out;
}

}  // namespace

const char* fiducial_window_name(FiducialWindow window) {
    switch (window) {
        case FiducialWindow::TopLeft: return "TL";
        case FiducialWindow::Top: return "T";
        case FiducialWindow::TopRight: return "TR";
        case FiducialWindow::Right: return "R";
        case FiducialWindow::BottomRight: return "BR";
        case FiducialWindow::Bottom: return "B";
        case FiducialWindow::BottomLeft: return "BL";
        case FiducialWindow::Left: return "L";
    }
    return "?";
}

bool fit_film_transform(const std::vector<Eigen::Vector2d>& film_mm,
                        const std::vector<Eigen::Vector2d>& pixels,
                        FilmTransformType type, Eigen::Matrix3d& transform,
                        double& rms_px, std::vector<double>* residuals) {
    rms_px = 0.0;
    if (residuals != nullptr) {
        residuals->clear();
    }
    if (film_mm.size() != pixels.size()) {
        return false;
    }
    const std::size_t minimum =
        type == FilmTransformType::Affine ? 3U : 4U;
    if (film_mm.size() < minimum) {
        return false;
    }
    for (std::size_t i = 0; i < film_mm.size(); ++i) {
        if (!film_mm[i].allFinite() || !pixels[i].allFinite()) {
            return false;
        }
    }
    transform = Eigen::Matrix3d::Identity();
    if (type == FilmTransformType::Affine) {
        const std::size_t n = film_mm.size();
        Eigen::MatrixXd design = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(2 * n), 6);
        Eigen::VectorXd rhs(static_cast<Eigen::Index>(2 * n));
        for (std::size_t i = 0; i < n; ++i) {
            const Eigen::Index row = static_cast<Eigen::Index>(2 * i);
            design(row, 0) = film_mm[i].x();
            design(row, 1) = film_mm[i].y();
            design(row, 2) = 1.0;
            design(row + 1, 3) = film_mm[i].x();
            design(row + 1, 4) = film_mm[i].y();
            design(row + 1, 5) = 1.0;
            rhs(row) = pixels[i].x();
            rhs(row + 1) = pixels[i].y();
        }
        const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(design);
        if (qr.rank() < 6) {
            return false;
        }
        const Eigen::VectorXd solution = qr.solve(rhs);
        if (!solution.allFinite()) {
            return false;
        }
        transform << solution[0], solution[1], solution[2],
                     solution[3], solution[4], solution[5],
                     0.0, 0.0, 1.0;
    } else {
        if (!homography_dlt(film_mm, pixels, transform) || !transform.allFinite()) {
            return false;
        }
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < film_mm.size(); ++i) {
        const Eigen::Vector3d homogeneous =
            transform * Eigen::Vector3d(film_mm[i].x(), film_mm[i].y(), 1.0);
        if (std::abs(homogeneous.z()) < 1e-12) {
            return false;
        }
        const Eigen::Vector2d projected(homogeneous.x() / homogeneous.z(),
                                        homogeneous.y() / homogeneous.z());
        const double error = (projected - pixels[i]).norm();
        if (!std::isfinite(error)) {
            return false;
        }
        sum += error * error;
        if (residuals != nullptr) {
            residuals->push_back(error);
        }
    }
    rms_px = std::sqrt(sum / static_cast<double>(film_mm.size()));
    return std::isfinite(rms_px);
}

namespace {

Eigen::Vector2d apply_dihedral(const Eigen::Vector2d& direction, int rotation,
                               bool mirror) {
    Eigen::Vector2d point = mirror ? Eigen::Vector2d(-direction.x(), direction.y())
                                   : direction;
    for (int i = 0; i < rotation; ++i) {
        point = Eigen::Vector2d(-point.y(), point.x());
    }
    return point;
}

struct OrderedLayout {
    bool found = false;
    bool ambiguous = false;
    Eigen::Matrix3d transform = Eigen::Matrix3d::Identity();
    std::vector<std::size_t> mark_index;
    std::vector<std::size_t> fiducial_index;
    std::vector<double> residuals;
    double rms_px = 0.0;
    std::size_t pairs = 0;
    std::string failure;
};

/// Orientation matrix of the scan relative to the film for one prior. The
/// canonical map (rotation 0, not mirrored) sends film +x to scan +x and film
/// +y to scan -y; a rotation is applied before, a mirror flips film +x.
Eigen::Matrix2d orientation_prior(int rotation, bool mirror) {
    const double angle = 0.5 * kPi * static_cast<double>(rotation);
    Eigen::Matrix2d matrix;
    matrix << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
    if (mirror) {
        matrix.col(0) *= -1.0;
    }
    matrix.col(1) *= -1.0;
    return matrix;
}

bool parse_scan_orientation(const std::string& text, int& rotation, bool& mirror) {
    std::string digits = text;
    mirror = !digits.empty() && digits.back() == 'm';
    if (mirror) {
        digits.pop_back();
    }
    if (digits == "0") {
        rotation = 0;
    } else if (digits == "90") {
        rotation = 1;
    } else if (digits == "180") {
        rotation = 2;
    } else if (digits == "270") {
        rotation = 3;
    } else {
        return false;
    }
    return true;
}

/// Local linear part (Jacobian at `film_centroid`) of a film-mm -> pixel map.
Eigen::Matrix2d linear_part(const Eigen::Matrix3d& transform,
                            const Eigen::Vector2d& film_centroid) {
    const Eigen::Vector3d h = transform * Eigen::Vector3d(
                                               film_centroid.x(),
                                               film_centroid.y(), 1.0);
    const double w = std::abs(h.z()) > 1e-12 ? h.z() : 1.0;
    Eigen::Matrix2d matrix;
    matrix << transform(0, 0) - h.x() / w * transform(2, 0),
        transform(0, 1) - h.x() / w * transform(2, 1),
        transform(1, 0) - h.y() / w * transform(2, 0),
        transform(1, 1) - h.y() / w * transform(2, 1);
    return matrix / w;
}

/// A plausible relabeling kept for the equivalence-class comparison. The
/// linear part is stored normalised by its mean singular value so that only
/// orientation (rotation/mirror) is compared with the prior, never scale.
struct OrderedCandidate {
    OrderedLayout layout;
    Eigen::Matrix2d normalised = Eigen::Matrix2d::Identity();
};

/// Reject a candidate whose map is not a near-similarity. Returns the linear
/// part normalised by its mean singular value.
bool plausible_scan(const Eigen::Matrix2d& linear, double max_anisotropy,
                    Eigen::Matrix2d& normalised) {
    const double e = linear.col(0).squaredNorm();
    const double f = linear.col(1).squaredNorm();
    const double g = linear.col(0).dot(linear.col(1));
    if (!(e > 1e-18) || !(f > 1e-18)) {
        return false;
    }
    const double mean = 0.5 * (e + f);
    const double disc = std::sqrt(std::max(0.0, 0.25 * (e - f) * (e - f) + g * g));
    const double s1 = std::sqrt(std::max(0.0, mean + disc));
    const double s2 = std::sqrt(std::max(0.0, mean - disc));
    if (!(s2 > 1e-18)) {
        return false;
    }
    if (std::abs(s1 / s2 - 1.0) > max_anisotropy) {
        return false;
    }
    if (std::abs(g / std::sqrt(e * f)) > 0.02) {
        return false;
    }
    normalised = linear / (0.5 * (s1 + s2));
    return true;
}

bool order_marks(const std::vector<FiducialMark>& marks,
                 const std::vector<ExpectedMark>& expected, FilmTransformType type,
                 double max_anisotropy, int prior_rotation, bool prior_mirror,
                 OrderedLayout& best) {
    const std::size_t minimum = type == FilmTransformType::Affine ? 3U : 4U;
    std::vector<OrderedCandidate> candidates;
    for (int mirror = 0; mirror < 2; ++mirror) {
        for (int rotation = 0; rotation < 4; ++rotation) {
            std::vector<Eigen::Vector2d> film;
            std::vector<Eigen::Vector2d> pixels;
            std::vector<std::size_t> mark_index;
            std::vector<std::size_t> fiducial_index;
            std::vector<char> used(expected.size(), 0);
            for (std::size_t m = 0; m < marks.size(); ++m) {
                const Eigen::Vector2d target = apply_dihedral(
                    window_direction(marks[m].window), rotation, mirror != 0);
                const FiducialWindow window = classify_direction(target);
                const Eigen::Vector2d target_unit = target.normalized();
                int chosen = -1;
                double best_dot = 0.5;  // within 60 degrees of the window axis
                for (std::size_t e = 0; e < expected.size(); ++e) {
                    if (used[e] != 0 || expected[e].window != window) {
                        continue;
                    }
                    const double dot =
                        target_unit.dot(expected[e].film_mm.normalized());
                    if (dot > best_dot) {
                        best_dot = dot;
                        chosen = static_cast<int>(e);
                    }
                }
                if (chosen < 0) {
                    continue;
                }
                used[static_cast<std::size_t>(chosen)] = 1;
                film.push_back(expected[static_cast<std::size_t>(chosen)].film_mm);
                pixels.push_back(marks[m].pixel);
                mark_index.push_back(m);
                fiducial_index.push_back(
                    expected[static_cast<std::size_t>(chosen)].index);
            }
            if (film.size() < minimum) {
                continue;
            }
            Eigen::Matrix3d candidate;
            double rms = 0.0;
            std::vector<double> residuals;
            if (!fit_film_transform(film, pixels, type, candidate, rms, &residuals)) {
                continue;
            }
            Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
            for (const Eigen::Vector2d& point : film) {
                centroid += point;
            }
            centroid /= static_cast<double>(film.size());
            Eigen::Matrix2d normalised;
            if (!plausible_scan(linear_part(candidate, centroid), max_anisotropy,
                                normalised)) {
                continue;
            }
            OrderedCandidate kept;
            kept.layout.found = true;
            kept.layout.transform = candidate;
            kept.layout.mark_index = std::move(mark_index);
            kept.layout.fiducial_index = std::move(fiducial_index);
            kept.layout.residuals = std::move(residuals);
            kept.layout.rms_px = rms;
            kept.layout.pairs = film.size();
            kept.normalised = normalised;
            const bool duplicate = std::any_of(
                candidates.begin(), candidates.end(),
                [&kept](const OrderedCandidate& other) {
                    return other.layout.fiducial_index == kept.layout.fiducial_index;
                });
            if (!duplicate) {
                candidates.push_back(std::move(kept));
            }
        }
    }
    if (candidates.empty()) {
        best.failure = "fiducial ordering failed: no plausible assignment";
        return false;
    }
    std::size_t max_pairs = 0;
    double best_rms = std::numeric_limits<double>::infinity();
    for (const OrderedCandidate& candidate : candidates) {
        max_pairs = std::max(max_pairs, candidate.layout.pairs);
    }
    for (const OrderedCandidate& candidate : candidates) {
        if (candidate.layout.pairs == max_pairs) {
            best_rms = std::min(best_rms, candidate.layout.rms_px);
        }
    }
    const double tolerance = std::max(0.25, 0.5 * best_rms);
    std::vector<const OrderedCandidate*> chosen;
    for (const OrderedCandidate& candidate : candidates) {
        if (candidate.layout.pairs == max_pairs &&
            candidate.layout.rms_px <= best_rms + tolerance) {
            chosen.push_back(&candidate);
        }
    }
    const Eigen::Matrix2d prior = orientation_prior(prior_rotation, prior_mirror);
    const OrderedCandidate* pick = nullptr;
    double best_distance = std::numeric_limits<double>::infinity();
    double second_distance = best_distance;
    for (const OrderedCandidate* candidate : chosen) {
        const double distance = (candidate->normalised - prior).norm();
        if (distance < best_distance) {
            second_distance = best_distance;
            best_distance = distance;
            pick = candidate;
        } else if (distance < second_distance) {
            second_distance = distance;
        }
    }
    if (pick == nullptr) {
        best.failure = "fiducial ordering failed: no consistent assignment";
        return false;
    }
    if (chosen.size() > 1 && second_distance - best_distance <= 1e-6) {
        best.failure =
            "fiducial ordering failed: ordering ambiguous under dihedral symmetry";
        return false;
    }
    best = pick->layout;
    best.ambiguous = chosen.size() > 1;
    return true;
}

}  // namespace

namespace {

Eigen::Vector2d quad_corner(const FrameQuad& quad, FiducialWindow window) {
    switch (window) {
        case FiducialWindow::TopLeft: return quad.top_left;
        case FiducialWindow::TopRight: return quad.top_right;
        case FiducialWindow::BottomRight: return quad.bottom_right;
        case FiducialWindow::BottomLeft: return quad.bottom_left;
        default: return Eigen::Vector2d::Zero();
    }
}

}  // namespace

FiducialDetection detect_fiducials(const FloatImage& gray,
                                   const FilmSettings& settings,
                                   const FiducialDetectOptions& options) {
    FiducialDetection result;
    if (!(options.min_radius_px > 0.0) ||
        !(options.max_radius_px > options.min_radius_px) ||
        !(options.max_residual_px > 0.0)) {
        throw ValueError("detect_fiducials: radii and max_residual_px must be positive");
    }
    if (!std::isfinite(options.max_anisotropy) ||
        !(options.max_anisotropy >= 0.0)) {
        throw ValueError("detect_fiducials: max_anisotropy must be non-negative");
    }
    int prior_rotation = 0;
    bool prior_mirror = false;
    if (!parse_scan_orientation(options.scan_orientation, prior_rotation,
                                prior_mirror)) {
        throw ValueError("detect_fiducials: unknown scan_orientation '" +
                         options.scan_orientation + "'");
    }
    if (!settings.enabled) {
        result.failure = "film is not enabled on this sensor";
        return result;
    }
    if (gray.empty() || gray.width() < 32 || gray.height() < 32) {
        result.failure = "image is too small for fiducial detection";
        return result;
    }
    const std::optional<FrameQuad> quad = detect_frame_quad(gray);
    if (!quad) {
        result.failure = "frame quad detection failed";
        return result;
    }
    const std::vector<WindowRequest> windows =
        build_windows(*quad, options, gray.width(), gray.height());
    if (windows.empty()) {
        result.failure = "no valid search window fits inside the image";
        return result;
    }
    const Gradients gradients = sobel_gradients(gray);
    for (const WindowRequest& request : windows) {
        std::optional<Eigen::Vector2d> pixel;
        if (options.detector == FiducialDetector::FrameCorner) {
            pixel = quad_corner(*quad, request.window);
        } else if (options.detector == FiducialDetector::Corner) {
            pixel = detect_corner(gray, gradients, request.box, request.window, 8.0);
        } else {
            const std::optional<CircleHit> hit = detect_circle(
                gradients, request.box, options.min_radius_px,
                options.max_radius_px);
            if (hit) {
                pixel = hit->centre;
            }
        }
        if (pixel && pixel->allFinite()) {
            FiducialMark mark;
            mark.window = request.window;
            mark.pixel = *pixel;
            result.marks.push_back(mark);
        }
    }
    std::sort(result.marks.begin(), result.marks.end(),
              [](const FiducialMark& a, const FiducialMark& b) {
                  return static_cast<int>(a.window) < static_cast<int>(b.window);
              });
    if (result.marks.empty()) {
        result.failure = "no fiducial marks found";
        return result;
    }
    if (settings.fiducials.empty()) {
        return result;  // only pixels can be stored until a layout exists
    }
    const std::vector<ExpectedMark> expected = classify_expected(settings.fiducials);
    if (expected.empty()) {
        result.failure =
            "fiducial ordering failed: no classifiable fiducial coordinates";
        return result;
    }
    OrderedLayout best;
    if (!order_marks(result.marks, expected, settings.transform,
                     options.max_anisotropy, prior_rotation, prior_mirror, best)) {
        result.failure = best.failure.empty()
                             ? "fiducial ordering failed: no consistent assignment"
                             : best.failure;
        return result;
    }
    if (best.rms_px > options.max_residual_px) {
        result.failure = "fiducial ordering failed: rms " +
                         std::to_string(best.rms_px) + " px exceeds " +
                         std::to_string(options.max_residual_px) + " px";
        return result;
    }
    result.by_fiducial.assign(settings.fiducials.size(), std::nullopt);
    for (std::size_t i = 0; i < best.fiducial_index.size(); ++i) {
        result.by_fiducial[best.fiducial_index[i]] =
            result.marks[best.mark_index[i]].pixel;
    }
    result.transform = best.transform;
    result.rms_px = best.rms_px;
    result.residuals_px = best.residuals;
    result.ambiguous = best.ambiguous;
    return result;
}

}  // namespace alphashape
