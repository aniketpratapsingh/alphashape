#include "make_film_frame.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>

#include <Eigen/Dense>

#include "base/utils/portable_rng.hpp"

namespace alphashape::test {
namespace {

constexpr double kPi = std::numbers::pi_v<double>;
constexpr int kSupersample = 4;

/// Distance from (x, y) to the segment (ax, ay)-(bx, by) within `radius`.
bool near_segment(double x, double y, double ax, double ay, double bx, double by,
                  double radius) {
    const double dx = bx - ax;
    const double dy = by - ay;
    const double length2 = dx * dx + dy * dy;
    double t = 0.0;
    if (length2 > 0.0) {
        t = ((x - ax) * dx + (y - ay) * dy) / length2;
    }
    t = std::clamp(t, 0.0, 1.0);
    const double px = ax + t * dx;
    const double py = ay + t * dy;
    return (x - px) * (x - px) + (y - py) * (y - py) <= radius * radius;
}

/// True when a film-coordinate sample falls on a drawn mark. L marks open
/// towards the frame centre.
bool inside_mark(const Eigen::Vector2d& film, const Eigen::Vector2d& centre,
                 const FilmFrameSpec& spec) {
    if (spec.mark_type == FiducialDetector::Circle) {
        return (film - centre).squaredNorm() <=
               spec.mark_radius_mm * spec.mark_radius_mm;
    }
    if (spec.mark_type != FiducialDetector::Corner) {
        return false;
    }
    const double ex = centre.x() > 0.0 ? -1.0 : 1.0;
    const double ey = centre.y() > 0.0 ? -1.0 : 1.0;
    const bool horizontal =
        near_segment(film.x(), film.y(), centre.x(), centre.y(),
                     centre.x() + ex * spec.mark_arm_mm, centre.y(),
                     spec.mark_radius_mm);
    const bool vertical =
        near_segment(film.x(), film.y(), centre.x(), centre.y(), centre.x(),
                     centre.y() + ey * spec.mark_arm_mm, spec.mark_radius_mm);
    return horizontal || vertical;
}

}  // namespace

FilmFrame render_film_frame(const FilmFrameSpec& spec) {
    FilmFrame out;
    const double angle = spec.rotation_deg * kPi / 180.0;
    Eigen::Matrix2d rotation;
    rotation << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
    Eigen::Matrix2d mirror = Eigen::Matrix2d::Identity();
    if (spec.mirror) {
        mirror(0, 0) = -1.0;
    }
    Eigen::Matrix2d yflip = Eigen::Matrix2d::Identity();
    yflip(1, 1) = -1.0;
    const Eigen::Matrix2d linear =
        spec.scale_px_per_mm * rotation * mirror * yflip;
    const double cx = 0.5 * static_cast<double>(spec.width - 1);
    const double cy = 0.5 * static_cast<double>(spec.height - 1);
    out.transform << linear(0, 0), linear(0, 1), cx,
                     linear(1, 0), linear(1, 1), cy,
                     0.0, 0.0, 1.0;
    const Eigen::Matrix3d inverse = out.transform.inverse();
    out.image = FloatImage(spec.width, spec.height, 0.0F);
    const double step = 1.0 / static_cast<double>(kSupersample);
    for (int y = 0; y < spec.height; ++y) {
        for (int x = 0; x < spec.width; ++x) {
            double sum = 0.0;
            for (int sy = 0; sy < kSupersample; ++sy) {
                for (int sx = 0; sx < kSupersample; ++sx) {
                    const double px = static_cast<double>(x) +
                                      (static_cast<double>(sx) + 0.5) * step - 0.5;
                    const double py = static_cast<double>(y) +
                                      (static_cast<double>(sy) + 0.5) * step - 0.5;
                    const Eigen::Vector3d film =
                        inverse * Eigen::Vector3d(px, py, 1.0);
                    const double fx = film.x() / film.z();
                    const double fy = film.y() / film.z();
                    double value = spec.background;
                    if (std::abs(fx) <= spec.frame_half_mm &&
                        std::abs(fy) <= spec.frame_half_mm) {
                        value = spec.frame;
                        for (const Eigen::Vector2d& mark : spec.marks_mm) {
                            if (inside_mark(Eigen::Vector2d(fx, fy), mark, spec)) {
                                value = spec.mark;
                                break;
                            }
                        }
                    }
                    sum += value;
                }
            }
            out.image.at(x, y) = static_cast<float>(
                sum / static_cast<double>(kSupersample * kSupersample));
        }
    }
    std::mt19937 engine(spec.seed);
    for (int y = 0; y < spec.height; ++y) {
        for (int x = 0; x < spec.width; ++x) {
            const double u1 = rng::uniform_real(engine, 1e-9, 1.0);
            const double u2 = rng::uniform_real(engine, 0.0, 1.0);
            const double gaussian =
                std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
            out.image.at(x, y) += static_cast<float>(spec.noise * gaussian);
        }
    }
    const double half = spec.frame_half_mm;
    const auto project = [&out](const Eigen::Vector2d& mm) {
        const Eigen::Vector3d h =
            out.transform * Eigen::Vector3d(mm.x(), mm.y(), 1.0);
        return Eigen::Vector2d(h.x() / h.z(), h.y() / h.z());
    };
    out.frame_corners = {project(Eigen::Vector2d(-half, half)),
                         project(Eigen::Vector2d(half, half)),
                         project(Eigen::Vector2d(half, -half)),
                         project(Eigen::Vector2d(-half, -half))};
    out.mark_pixels.reserve(spec.marks_mm.size());
    for (const Eigen::Vector2d& mark : spec.marks_mm) {
        out.mark_pixels.push_back(project(mark));
    }
    return out;
}

}  // namespace alphashape::test
