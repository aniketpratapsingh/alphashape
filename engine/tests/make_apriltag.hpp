#pragma once

// Synthetic AprilTag renderer for the WP-13.7 tests.
//
// The printed tag is the published family image: a total_width x total_width
// grid whose cells are filled exactly as the published table describes,
// including the white reference frame of the reversed-border families and the
// white margin of the plain families. Rendering maps the grid through a
// rotation, a mild projective term and a scale, supersamples 2x2, then the
// test blurs, adds noise and quantises to 8 bits.
//
// Coordinates use the calibration convention (the centre of the top-left pixel
// is (0.5, 0.5), matching MarkerCandidate and CodedTarget). `center` is the
// tag's projected centre in those coordinates.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "common/algorithm/markers/tag_families.hpp"
#include "image/image.hpp"

namespace alphashape::test {

struct AprilTagTruth {
    double cx = 0.0;              ///< Projected tag centre, calibration convention.
    double cy = 0.0;
    double tag_px = 80.0;         ///< Width of the whole total_width grid.
    double rotation = 0.0;        ///< Radians, positive is clockwise on screen.
    double perspective_u = 0.0;   ///< Denominator terms per grid cell.
    double perspective_v = 0.0;
    int id = 0;
    AprilTagFamily family = AprilTagFamily::Tag36h11;
    /// Optional explicit payload word (published bit order). When `use_word`
    /// is set the family's code table is not consulted, which is how the
    /// foreign-word rejection tests render a tag that is not in the family.
    std::uint64_t word = 0;
    bool use_word = false;
};

/// Grid (cell) coordinates to image calibration coordinates and back.
struct TagGridTransform {
    Eigen::Matrix3d to_image = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d to_grid = Eigen::Matrix3d::Identity();
    int total_width = 0;

    bool project(double gx, double gy, Eigen::Vector2d& out) const {
        const Eigen::Vector3d p = to_image * Eigen::Vector3d(gx, gy, 1.0);
        if (!std::isfinite(p.z()) || std::abs(p.z()) < 1e-12) {
            return false;
        }
        out = Eigen::Vector2d(p.x() / p.z(), p.y() / p.z());
        return out.allFinite();
    }

    bool inverse(double x, double y, Eigen::Vector2d& out) const {
        const Eigen::Vector3d p = to_grid * Eigen::Vector3d(x, y, 1.0);
        if (!std::isfinite(p.z()) || std::abs(p.z()) < 1e-12) {
            return false;
        }
        out = Eigen::Vector2d(p.x() / p.z(), p.y() / p.z());
        return out.allFinite();
    }
};

inline TagGridTransform make_tag_transform(const AprilTagTruth& truth) {
    const TagFamily& family = tag_family(truth.family);
    TagGridTransform transform;
    transform.total_width = family.total_width;
    const double cells = static_cast<double>(family.total_width);
    const double cell = truth.tag_px / cells;
    const double c = std::cos(truth.rotation);
    const double s = std::sin(truth.rotation);
    Eigen::Matrix3d scale_rotate;
    scale_rotate << cell * c, -cell * s, 0.0,
                    cell * s, cell * c, 0.0,
                    truth.perspective_u, truth.perspective_v, 1.0;
    Eigen::Matrix3d centre_shift;
    centre_shift << 1.0, 0.0, -0.5 * cells,
                    0.0, 1.0, -0.5 * cells,
                    0.0, 0.0, 1.0;
    Eigen::Matrix3d place;
    place << 1.0, 0.0, truth.cx,
             0.0, 1.0, truth.cy,
             0.0, 0.0, 1.0;
    transform.to_image = place * scale_rotate * centre_shift;
    transform.to_grid = transform.to_image.inverse();
    return transform;
}

/// The four published corners in image coordinates, ordered corner 0 (the
/// bottom-left of an upright tag), 1 (bottom-right), 2 (top-right),
/// 3 (top-left). `silhouette` selects the whole total_width grid instead of
/// the reference `width_at_border` square.
inline std::array<Eigen::Vector2d, 4> april_tag_corners(
    const AprilTagTruth& truth, bool silhouette) {
    const TagFamily& family = tag_family(truth.family);
    const int total = family.total_width;
    const int border_start = (total - family.width_at_border) / 2;
    const double lo =
        silhouette ? 0.0 : static_cast<double>(border_start);
    const double hi = silhouette
                          ? static_cast<double>(total)
                          : static_cast<double>(border_start +
                                                family.width_at_border);
    const TagGridTransform transform = make_tag_transform(truth);
    const std::array<Eigen::Vector2d, 4> grid = {
        Eigen::Vector2d(lo, hi), Eigen::Vector2d(hi, hi),
        Eigen::Vector2d(hi, lo), Eigen::Vector2d(lo, lo)};
    std::array<Eigen::Vector2d, 4> corners{};
    for (std::size_t i = 0; i < corners.size(); ++i) {
        transform.project(grid[i].x(), grid[i].y(), corners[i]);
    }
    return corners;
}

/// True when grid cell (gx, gy) (both in [0, total_width)) is white for the
/// published code `code` (payload bits in published order).
inline bool tag_cell_is_white(const TagFamily& family, std::uint64_t code,
                              int gx, int gy) {
    const int border_start = (family.total_width - family.width_at_border) / 2;
    const int x = gx - border_start;
    const int y = gy - border_start;
    for (int bit = 0; bit < family.bits; ++bit) {
        if (family.bit_xy[static_cast<std::size_t>(bit)].first == x &&
            family.bit_xy[static_cast<std::size_t>(bit)].second == y) {
            return ((code >> static_cast<unsigned>(family.bits - 1 - bit)) & 1u) != 0u;
        }
    }
    if (x == 0 || y == 0 || x == family.width_at_border - 1 ||
        y == family.width_at_border - 1) {
        return family.reversed_border;
    }
    return !family.reversed_border;
}

/// Canvas large enough for a rotated, mildly perspective-warped tag of
/// `tag_px` plus a quiet zone on every side.
inline int april_tag_canvas_size(double tag_px) {
    return static_cast<int>(std::ceil(1.70 * tag_px)) + 16;
}

struct AprilTagRender {
    FloatImage image;
    std::array<Eigen::Vector2d, 4> border{};
    std::array<Eigen::Vector2d, 4> silhouette{};
};

/// Render one tag into a width x height canvas. White cells and the background
/// use `background`; black cells use `foreground`.
inline AprilTagRender render_april_tag(int width, int height,
                                       const AprilTagTruth& truth,
                                       double background = 0.95,
                                       double foreground = 0.05,
                                       int supersample = 2) {
    const TagFamily& family = tag_family(truth.family);
    const int tw = family.total_width;
    const TagGridTransform transform = make_tag_transform(truth);
    std::uint64_t code = truth.word;
    if (!truth.use_word) {
        const std::size_t index = static_cast<std::size_t>(truth.id);
        code = index < family.codes.size() ? family.codes[index] : 0;
    }
    std::vector<std::uint8_t> cells(static_cast<std::size_t>(tw * tw), 0);
    for (int gy = 0; gy < tw; ++gy) {
        for (int gx = 0; gx < tw; ++gx) {
            cells[static_cast<std::size_t>(gy * tw + gx)] =
                static_cast<std::uint8_t>(
                    tag_cell_is_white(family, code, gx, gy) ? 1 : 0);
        }
    }
    FloatImage image(width, height, static_cast<float>(background));
    const int samples = std::max(1, supersample);
    const double total = static_cast<double>(samples) * static_cast<double>(samples);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int sy = 0; sy < samples; ++sy) {
                for (int sx = 0; sx < samples; ++sx) {
                    const double px =
                        static_cast<double>(x) +
                        (static_cast<double>(sx) + 0.5) / static_cast<double>(samples);
                    const double py =
                        static_cast<double>(y) +
                        (static_cast<double>(sy) + 0.5) / static_cast<double>(samples);
                    Eigen::Vector2d grid;
                    if (!transform.inverse(px, py, grid)) {
                        sum += background;
                        continue;
                    }
                    const int gx = static_cast<int>(std::floor(grid.x()));
                    const int gy = static_cast<int>(std::floor(grid.y()));
                    if (gx < 0 || gy < 0 || gx >= tw || gy >= tw) {
                        sum += background;
                        continue;
                    }
                    const bool white =
                        cells[static_cast<std::size_t>(gy * tw + gx)] != 0U;
                    sum += white ? background : foreground;
                }
            }
            image.at(x, y) = static_cast<float>(sum / total);
        }
    }
    const int border_start = (tw - family.width_at_border) / 2;
    const double last_border =
        static_cast<double>(border_start + family.width_at_border);
    const double last = static_cast<double>(tw);
    AprilTagRender render;
    const double first_border = static_cast<double>(border_start);
    const std::array<Eigen::Vector2d, 4> border_grid = {
        Eigen::Vector2d(first_border, last_border),
        Eigen::Vector2d(last_border, last_border),
        Eigen::Vector2d(last_border, first_border),
        Eigen::Vector2d(first_border, first_border)};
    const std::array<Eigen::Vector2d, 4> silhouette_grid = {
        Eigen::Vector2d(0.0, last), Eigen::Vector2d(last, last),
        Eigen::Vector2d(last, 0.0), Eigen::Vector2d(0.0, 0.0)};
    for (std::size_t i = 0; i < 4; ++i) {
        transform.project(border_grid[i].x(), border_grid[i].y(), render.border[i]);
        transform.project(silhouette_grid[i].x(), silhouette_grid[i].y(),
                          render.silhouette[i]);
    }
    render.image = std::move(image);
    return render;
}

/// Round to the 8-bit levels a camera would deliver.
inline void quantise_to_8bit(FloatImage& image) {
    for (float& value : image.pixels()) {
        const double clamped =
            std::clamp(static_cast<double>(value), 0.0, 1.0);
        const int level = static_cast<int>(std::lround(clamped * 255.0));
        value = static_cast<float>(static_cast<double>(level) / 255.0);
    }
}

}  // namespace alphashape::test
