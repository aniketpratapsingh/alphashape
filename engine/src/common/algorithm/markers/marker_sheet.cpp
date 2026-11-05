#include "common/algorithm/markers/marker_sheet.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <initializer_list>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "base/io/pdf.hpp"
#include "base/utils/error.hpp"
#include "common/algorithm/markers/tag_families.hpp"
#include "common/algorithm/markers/target_patterns.hpp"

namespace alphashape {
namespace {

constexpr double kMmToPt = 72.0 / 25.4;
constexpr double kPi = std::numbers::pi_v<double>;
constexpr double kPrintableMarginMm = 10.0;
constexpr int kSupersample = 4;

struct PaperSize {
    double width_mm = 0.0;
    double height_mm = 0.0;
};

PaperSize paper_size(const std::string& name) {
    if (name == "A4") {
        return PaperSize{210.0, 297.0};
    }
    if (name == "Letter") {
        return PaperSize{215.9, 279.4};
    }
    throw ValueError("marker sheet: unknown paper '" + name + "'");
}

/// One black region in millimetre coordinates with the origin at the sheet's
/// top-left and y growing downwards (the raster orientation). The PDF writer
/// mirrors y when it emits the same shape, so PDF and raster cannot drift.
enum class ShapeKind { Rect, Polygon, Circle, Ring, Sector };

struct Shape {
    ShapeKind kind = ShapeKind::Rect;
    double x = 0.0;   ///< Rect left edge, circle/sector centre x.
    double y = 0.0;   ///< Rect top edge, circle/sector centre y.
    double w = 0.0;   ///< Rect width.
    double h = 0.0;   ///< Rect height.
    double r_inner = 0.0;
    double r_outer = 0.0;
    double a0 = 0.0;  ///< Sector start angle, radians, atan2 orientation.
    double a1 = 0.0;  ///< Sector end angle, a1 >= a0.
    std::array<std::array<double, 2>, 8> points{};
    int point_count = 0;
    double gray = 0.0;
};

void add_rect(std::vector<Shape>& shapes, double x, double y, double w,
              double h) {
    Shape shape;
    shape.kind = ShapeKind::Rect;
    shape.x = x;
    shape.y = y;
    shape.w = w;
    shape.h = h;
    shapes.push_back(shape);
}

void add_circle(std::vector<Shape>& shapes, double x, double y, double r) {
    Shape shape;
    shape.kind = ShapeKind::Circle;
    shape.x = x;
    shape.y = y;
    shape.r_outer = r;
    shapes.push_back(shape);
}

void add_ring(std::vector<Shape>& shapes, double x, double y, double r_inner,
              double r_outer) {
    Shape shape;
    shape.kind = ShapeKind::Ring;
    shape.x = x;
    shape.y = y;
    shape.r_inner = r_inner;
    shape.r_outer = r_outer;
    shapes.push_back(shape);
}

void add_sector(std::vector<Shape>& shapes, double x, double y, double r_inner,
                double r_outer, double a0, double a1) {
    Shape shape;
    shape.kind = ShapeKind::Sector;
    shape.x = x;
    shape.y = y;
    shape.r_inner = r_inner;
    shape.r_outer = r_outer;
    shape.a0 = a0;
    shape.a1 = a1;
    shapes.push_back(shape);
}

bool shape_contains(const Shape& shape, double x, double y) {
    const double dx = x - shape.x;
    const double dy = y - shape.y;
    const double distance = std::hypot(dx, dy);
    switch (shape.kind) {
        case ShapeKind::Rect:
            return x >= shape.x && x <= shape.x + shape.w && y >= shape.y &&
                   y <= shape.y + shape.h;
        case ShapeKind::Circle:
            return distance <= shape.r_outer;
        case ShapeKind::Ring:
            return distance >= shape.r_inner && distance <= shape.r_outer;
        case ShapeKind::Sector: {
            if (distance < shape.r_inner || distance > shape.r_outer) {
                return false;
            }
            const double span = shape.a1 - shape.a0;
            if (span >= 2.0 * kPi) {
                return true;
            }
            double relative = std::atan2(dy, dx) - shape.a0;
            relative = std::fmod(relative, 2.0 * kPi);
            if (relative < 0.0) {
                relative += 2.0 * kPi;
            }
            return relative <= span;
        }
        default:
            break;
    }
    bool inside = false;
    for (int i = 0, j = shape.point_count - 1; i < shape.point_count; j = i++) {
        const double xi = shape.points[static_cast<std::size_t>(i)][0];
        const double yi = shape.points[static_cast<std::size_t>(i)][1];
        const double xj = shape.points[static_cast<std::size_t>(j)][0];
        const double yj = shape.points[static_cast<std::size_t>(j)][1];
        if (((yi > y) != (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
}

/// The printed circular pattern of target_patterns.hpp. Sector 0 starts at
/// twelve o'clock and advances clockwise on the printed sheet, which in the
/// top-down millimetre frame is atan2 increasing from -pi/2.
void add_circular_target(std::vector<Shape>& shapes, double cx, double cy,
                         double size_mm, int code, int bits) {
    const double outer = 0.5 * size_mm;
    const std::vector<std::uint32_t>& table = circular_codes(bits);
    const std::uint32_t word = table[static_cast<std::size_t>(code / 2)];
    if ((code % 2) != 0) {
        add_circle(shapes, cx, cy, 0.25 * outer);
    }
    add_ring(shapes, cx, cy, 0.9 * outer, outer);
    const double sector = 2.0 * kPi / static_cast<double>(bits);
    for (int i = 0; i < bits; ++i) {
        const int shift = bits - 1 - i;
        if (((word >> static_cast<unsigned>(shift)) & 1U) == 0U) {
            continue;
        }
        const double a0 = -0.5 * kPi + static_cast<double>(i) * sector;
        add_sector(shapes, cx, cy, 0.5 * outer, 0.85 * outer, a0,
                   a0 + sector);
    }
}

/// Published cell colour of one AprilTag grid cell; the same rule the test
/// renderer uses, ported from the published family description.
bool tag_cell_is_white(const TagFamily& family, std::uint64_t code, int gx,
                       int gy) {
    const int border_start =
        (family.total_width - family.width_at_border) / 2;
    const int x = gx - border_start;
    const int y = gy - border_start;
    for (int bit = 0; bit < family.bits; ++bit) {
        if (family.bit_xy[static_cast<std::size_t>(bit)].first == x &&
            family.bit_xy[static_cast<std::size_t>(bit)].second == y) {
            const int shift = family.bits - 1 - bit;
            return ((code >> static_cast<unsigned>(shift)) & 1U) != 0U;
        }
    }
    const bool inside = x >= 0 && y >= 0 && x < family.width_at_border &&
                        y < family.width_at_border;
    if (inside) {
        if (x == 0 || y == 0 || x == family.width_at_border - 1 ||
            y == family.width_at_border - 1) {
            return family.reversed_border;
        }
        return false;
    }
    return !family.reversed_border;
}

void add_april_target(std::vector<Shape>& shapes, double cx, double cy,
                      double size_mm, int code) {
    const TagFamily& family = tag_family(AprilTagFamily::Tag36h11);
    const int total = family.total_width;
    const double cell = size_mm / static_cast<double>(total);
    const double x0 = cx - 0.5 * size_mm;
    const double y0 = cy - 0.5 * size_mm;
    const std::uint64_t word = family.codes[static_cast<std::size_t>(code)];
    for (int gy = 0; gy < total; ++gy) {
        for (int gx = 0; gx < total; ++gx) {
            if (tag_cell_is_white(family, word, gx, gy)) {
                continue;
            }
            add_rect(shapes, x0 + static_cast<double>(gx) * cell,
                     y0 + static_cast<double>(gy) * cell, cell, cell);
        }
    }
}

/// Checker (bowtie) cross: the dark quadrants are the ones whose two
/// half-axis signs differ, matching cross_decoder.hpp and the test renderer.
void add_cross_target(std::vector<Shape>& shapes, double cx, double cy,
                      double size_mm) {
    const double half = 0.5 * size_mm;
    add_rect(shapes, cx, cy - half, half, half);  // right-above
    add_rect(shapes, cx - half, cy, half, half);  // left-below
}

struct TextItem {
    double x_mm = 0.0;             ///< Left edge of the baseline.
    double baseline_y_mm = 0.0;    ///< Baseline distance from the top edge.
    double size_pt = 6.0;
    std::string text;
};

struct PageContent {
    std::vector<Shape> shapes;
    std::vector<TextItem> texts;
};

struct PageSetup {
    SheetOptions options;
    PaperSize paper;
    int cols = 1;
    int rows = 1;
    double cell_width_mm = 0.0;
    double cell_height_mm = 0.0;
    double grid_x_mm = kPrintableMarginMm;
    double grid_y_mm = kPrintableMarginMm;
    int capacity = 1;
    int total_codes = 0;
    int pages = 1;
    int bits = 0;   ///< Circular ring width, or 0.
    bool rotated = false;  ///< Chessboard rotated 90 degrees to fit.

    bool coded() const {
        return options.target != SheetTarget::Cross &&
               options.target != SheetTarget::Chessboard;
    }
    bool chessboard() const {
        return options.target == SheetTarget::Chessboard;
    }
};

int circular_bits(SheetTarget target) {
    switch (target) {
        case SheetTarget::Circular12:
            return 12;
        case SheetTarget::Circular14:
            return 14;
        case SheetTarget::Circular16:
            return 16;
        case SheetTarget::Circular20:
            return 20;
        default:
            return 0;
    }
}

PageSetup make_setup(const SheetOptions& options) {
    PageSetup setup;
    setup.options = options;
    if (!(options.target_size_mm > 0.0)) {
        throw ValueError("marker sheet: target size must be positive");
    }
    if (!(options.quiet_zone_mm >= 0.0)) {
        throw ValueError("marker sheet: quiet zone must not be negative");
    }
    setup.paper = paper_size(options.paper);
    const double printable_w = setup.paper.width_mm - 2.0 * kPrintableMarginMm;
    const double printable_h =
        setup.paper.height_mm - 2.0 * kPrintableMarginMm;
    if (setup.chessboard()) {
        if (options.board_cols < 3 || options.board_rows < 3) {
            throw ValueError(
                "marker sheet: a chessboard needs at least 3x3 squares "
                "(2x2 inner corners)");
        }
        if (!(options.square_mm > 0.0)) {
            throw ValueError("marker sheet: square size must be positive");
        }
        const double board_w =
            static_cast<double>(options.board_cols) * options.square_mm;
        const double board_h =
            static_cast<double>(options.board_rows) * options.square_mm;
        double visual_w = board_w;
        double visual_h = board_h;
        if (visual_w > printable_w + 1e-9 ||
            visual_h > printable_h + 1e-9) {
            if (board_h <= printable_w + 1e-9 &&
                board_w <= printable_h + 1e-9) {
                setup.rotated = true;
                visual_w = board_h;
                visual_h = board_w;
            } else {
                throw ValueError(
                    "marker sheet: chessboard does not fit the selected "
                    "paper");
            }
        }
        setup.cols = options.board_cols;
        setup.rows = options.board_rows;
        setup.cell_width_mm = visual_w;
        setup.cell_height_mm = visual_h;
        setup.grid_x_mm = kPrintableMarginMm + 0.5 * (printable_w - visual_w);
        setup.grid_y_mm = kPrintableMarginMm + 0.5 * (printable_h - visual_h);
        setup.capacity = 1;
        setup.total_codes = 0;
        setup.pages = 1;
        return setup;
    }
    if (options.count < 0) {
        throw ValueError("marker sheet: count must not be negative");
    }
    setup.bits = circular_bits(options.target);
    setup.cell_width_mm =
        options.target_size_mm + 2.0 * options.quiet_zone_mm;
    setup.cell_height_mm = setup.cell_width_mm;
    setup.cols = std::max(
        0, static_cast<int>(std::floor(printable_w / setup.cell_width_mm)));
    setup.rows = std::max(
        0, static_cast<int>(std::floor(printable_h / setup.cell_height_mm)));
    if (setup.cols < 1 || setup.rows < 1) {
        throw ValueError(
            "marker sheet: target and quiet zone do not fit the selected "
            "paper");
    }
    setup.capacity = setup.cols * setup.rows;
    const double grid_w = static_cast<double>(setup.cols) * setup.cell_width_mm;
    const double grid_h = static_cast<double>(setup.rows) * setup.cell_height_mm;
    setup.grid_x_mm = kPrintableMarginMm + 0.5 * (printable_w - grid_w);
    setup.grid_y_mm = kPrintableMarginMm + 0.5 * (printable_h - grid_h);
    const int requested =
        options.count == 0 ? setup.capacity : options.count;
    if (options.count == 0 && requested < 1) {
        throw ValueError("marker sheet: no cell fits the selected paper");
    }
    if (setup.coded()) {
        const int available = sheet_target_code_count(options.target);
        if (options.first_code < 0 ||
            options.first_code + requested > available) {
            throw ValueError(
                "marker sheet: code range lies outside the target family");
        }
    }
    setup.total_codes = requested;
    setup.pages = (requested + setup.capacity - 1) / setup.capacity;
    return setup;
}

/// Small L-shaped registration marks at the four corners of the target grid.
void add_corner_marks(std::vector<Shape>& shapes, double x, double y, double w,
                      double h) {
    const double length = std::min(4.0, 0.2 * std::min(w, h));
    const double thickness = 0.3;
    // Top-left.
    add_rect(shapes, x, y, length, thickness);
    add_rect(shapes, x, y, thickness, length);
    // Top-right.
    add_rect(shapes, x + w - length, y, length, thickness);
    add_rect(shapes, x + w - thickness, y, thickness, length);
    // Bottom-left.
    add_rect(shapes, x, y + h - thickness, length, thickness);
    add_rect(shapes, x, y + h - length, thickness, length);
    // Bottom-right.
    add_rect(shapes, x + w - length, y + h - thickness, length, thickness);
    add_rect(shapes, x + w - thickness, y + h - length, thickness, length);
}

std::string millimetres(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%.1f", value);
    return std::string(buffer);
}

TextItem footer_item(const PageSetup& setup) {
    std::string text = setup.chessboard()
                           ? "chessboard " + std::to_string(setup.cols) + "x" +
                                 std::to_string(setup.rows) + " squares, " +
                                 millimetres(setup.options.square_mm) +
                                 " mm squares"
                           : std::string(sheet_target_name(setup.options.target)) +
                                 " " + millimetres(setup.options.target_size_mm) +
                                 " mm";
    text += " - print at 100% / actual size";
    TextItem item;
    item.x_mm = kPrintableMarginMm;
    item.baseline_y_mm = setup.paper.height_mm - 4.0;
    item.size_pt = 7.0;
    item.text = text;
    return item;
}

void add_chessboard(const PageSetup& setup, PageContent& content) {
    const double square = setup.options.square_mm;
    const int cols = setup.options.board_cols;
    const int rows = setup.options.board_rows;
    const double board_h = static_cast<double>(rows) * square;
    for (int iy = 0; iy < rows; ++iy) {
        for (int ix = 0; ix < cols; ++ix) {
            if (((ix + iy) % 2) != 0) {
                continue;
            }
            const double x = static_cast<double>(ix) * square;
            const double y = static_cast<double>(iy) * square;
            if (setup.rotated) {
                // Quarter turn clockwise: (x, y) maps to (H - y - square, x).
                add_rect(content.shapes,
                         setup.grid_x_mm + board_h - y - square,
                         setup.grid_y_mm + x, square, square);
            } else {
                add_rect(content.shapes, setup.grid_x_mm + x,
                         setup.grid_y_mm + y, square, square);
            }
        }
    }
}

PageContent build_page(const PageSetup& setup, int page) {
    PageContent content;
    if (setup.chessboard()) {
        add_chessboard(setup, content);
        if (setup.options.corner_marks) {
            add_corner_marks(content.shapes, setup.grid_x_mm,
                             setup.grid_y_mm, setup.cell_width_mm,
                             setup.cell_height_mm);
        }
        if (setup.options.labels) {
            content.texts.push_back(footer_item(setup));
        }
        return content;
    }
    const int start = page * setup.capacity;
    const int stop = std::min(setup.total_codes, start + setup.capacity);
    for (int index = start; index < stop; ++index) {
        const int local = index - start;
        const int row = local / setup.cols;
        const int col = local % setup.cols;
        const double cx = setup.grid_x_mm +
                          (static_cast<double>(col) + 0.5) *
                              setup.cell_width_mm;
        const double cy = setup.grid_y_mm +
                          (static_cast<double>(row) + 0.5) *
                              setup.cell_height_mm;
        const int code = setup.options.first_code + index;
        if (setup.bits != 0) {
            add_circular_target(content.shapes, cx, cy,
                                setup.options.target_size_mm, code,
                                setup.bits);
        } else if (setup.options.target == SheetTarget::AprilTag36h11) {
            add_april_target(content.shapes, cx, cy,
                             setup.options.target_size_mm, code);
        } else {
            add_cross_target(content.shapes, cx, cy,
                             setup.options.target_size_mm);
        }
        if (setup.options.labels && setup.coded()) {
            TextItem label;
            label.x_mm = cx - 2.0;
            label.baseline_y_mm =
                cy + 0.5 * setup.options.target_size_mm +
                0.6 * setup.options.quiet_zone_mm;
            label.size_pt = 6.0;
            label.text = std::to_string(code);
            content.texts.push_back(label);
        }
    }
    if (setup.options.corner_marks) {
        add_corner_marks(content.shapes, setup.grid_x_mm, setup.grid_y_mm,
                         static_cast<double>(setup.cols) *
                             setup.cell_width_mm,
                         static_cast<double>(setup.rows) *
                             setup.cell_height_mm);
    }
    if (setup.options.labels) {
        content.texts.push_back(footer_item(setup));
    }
    return content;
}

/// Emit one page of shapes and text through the PDF vector primitives. The
/// millimetre frame is top-down, so y is mirrored into PDF user space.
void emit_shapes(PdfDocument& pdf, const PageContent& content,
                 const PaperSize& paper) {
    const double page_h = paper.height_mm;
    auto px = [](double value) { return value * kMmToPt; };
    auto py = [page_h](double value) { return (page_h - value) * kMmToPt; };
    for (const Shape& shape : content.shapes) {
        switch (shape.kind) {
            case ShapeKind::Rect:
                pdf.fill_rect(px(shape.x), py(shape.y + shape.h),
                              px(shape.w), px(shape.h), shape.gray);
                break;
            case ShapeKind::Circle:
                pdf.fill_circle(px(shape.x), py(shape.y), px(shape.r_outer),
                                shape.gray);
                break;
            case ShapeKind::Ring:
                pdf.fill_ring(px(shape.x), py(shape.y), px(shape.r_outer),
                              px(shape.r_inner), shape.gray);
                break;
            case ShapeKind::Sector:
                pdf.fill_annular_sector(px(shape.x), py(shape.y),
                                        px(shape.r_inner), px(shape.r_outer),
                                        -shape.a1, -shape.a0, shape.gray);
                break;
            case ShapeKind::Polygon: {
                std::vector<std::array<double, 2>> points;
                points.reserve(static_cast<std::size_t>(shape.point_count));
                for (int i = 0; i < shape.point_count; ++i) {
                    points.push_back({px(shape.points[static_cast<std::size_t>(i)][0]),
                                      py(shape.points[static_cast<std::size_t>(i)][1])});
                }
                pdf.fill_polygon(points, shape.gray);
                break;
            }
        }
    }
    for (const TextItem& item : content.texts) {
        pdf.text(px(item.x_mm), py(item.baseline_y_mm), item.size_pt,
                 item.text);
    }
}

void shape_bounds(const Shape& shape, double& min_x, double& min_y,
                  double& max_x, double& max_y) {
    switch (shape.kind) {
        case ShapeKind::Rect:
            min_x = shape.x;
            min_y = shape.y;
            max_x = shape.x + shape.w;
            max_y = shape.y + shape.h;
            return;
        case ShapeKind::Circle:
        case ShapeKind::Ring:
        case ShapeKind::Sector:
            min_x = shape.x - shape.r_outer;
            min_y = shape.y - shape.r_outer;
            max_x = shape.x + shape.r_outer;
            max_y = shape.y + shape.r_outer;
            return;
        case ShapeKind::Polygon:
            break;
    }
    min_x = shape.points[0][0];
    min_y = shape.points[0][1];
    max_x = min_x;
    max_y = min_y;
    for (int i = 1; i < shape.point_count; ++i) {
        min_x = std::min(min_x, shape.points[static_cast<std::size_t>(i)][0]);
        min_y = std::min(min_y, shape.points[static_cast<std::size_t>(i)][1]);
        max_x = std::max(max_x, shape.points[static_cast<std::size_t>(i)][0]);
        max_y = std::max(max_y, shape.points[static_cast<std::size_t>(i)][1]);
    }
}

/// Rasterise the shape list with a per-pixel 4x4 sample mask. Each sample bit
/// is independent, so overlapping shapes union correctly, and coverage maps
/// linearly to grey: 16/16 is black, 0/16 is page white.
Image raster_page(const PageContent& content, const PaperSize& paper,
                  double dpi) {
    const double mm_per_px = 25.4 / dpi;
    const int width = std::max(
        1, static_cast<int>(std::lround(paper.width_mm / mm_per_px)));
    const int height = std::max(
        1, static_cast<int>(std::lround(paper.height_mm / mm_per_px)));
    std::vector<std::uint16_t> mask(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0U);
    const double step = 1.0 / static_cast<double>(kSupersample);
    for (const Shape& shape : content.shapes) {
        double min_x = 0.0;
        double min_y = 0.0;
        double max_x = 0.0;
        double max_y = 0.0;
        shape_bounds(shape, min_x, min_y, max_x, max_y);
        const int x0 = std::max(
            0, static_cast<int>(std::floor(min_x / mm_per_px)));
        const int y0 = std::max(
            0, static_cast<int>(std::floor(min_y / mm_per_px)));
        const int x1 = std::min(
            width - 1, static_cast<int>(std::ceil(max_x / mm_per_px)));
        const int y1 = std::min(
            height - 1, static_cast<int>(std::ceil(max_y / mm_per_px)));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                std::uint16_t bits = 0U;
                for (int sy = 0; sy < kSupersample; ++sy) {
                    for (int sx = 0; sx < kSupersample; ++sx) {
                        const double sample_x =
                            (static_cast<double>(x) +
                             (static_cast<double>(sx) + 0.5) * step) *
                            mm_per_px;
                        const double sample_y =
                            (static_cast<double>(y) +
                             (static_cast<double>(sy) + 0.5) * step) *
                            mm_per_px;
                        if (shape_contains(shape, sample_x, sample_y)) {
                            bits = static_cast<std::uint16_t>(
                                bits | (static_cast<std::uint16_t>(1U)
                                        << (sy * kSupersample + sx)));
                        }
                    }
                }
                const std::size_t index =
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(width) +
                    static_cast<std::size_t>(x);
                mask[index] = static_cast<std::uint16_t>(mask[index] | bits);
            }
        }
    }
    Image image(width, height, 1);
    const double total =
        static_cast<double>(kSupersample * kSupersample);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::uint16_t bits = mask[static_cast<std::size_t>(y) *
                                                static_cast<std::size_t>(width) +
                                            static_cast<std::size_t>(x)];
            int count = 0;
            for (int bit = 0; bit < kSupersample * kSupersample; ++bit) {
                if ((bits & (static_cast<std::uint16_t>(1U) << bit)) != 0U) {
                    ++count;
                }
            }
            const double value =
                255.0 * (1.0 - static_cast<double>(count) / total);
            image.at(x, y, 0) = static_cast<std::uint8_t>(
                std::lround(std::clamp(value, 0.0, 255.0)));
        }
    }
    return image;
}

}  // namespace

const char* sheet_target_name(SheetTarget target) {
    switch (target) {
        case SheetTarget::Circular12:
            return "circular12";
        case SheetTarget::Circular14:
            return "circular14";
        case SheetTarget::Circular16:
            return "circular16";
        case SheetTarget::Circular20:
            return "circular20";
        case SheetTarget::Cross:
            return "cross";
        case SheetTarget::AprilTag36h11:
            return "apriltag36h11";
        case SheetTarget::Chessboard:
            return "chessboard";
    }
    return "unknown";
}

std::optional<SheetTarget> parse_sheet_target(std::string_view name) {
    for (SheetTarget target :
         {SheetTarget::Circular12, SheetTarget::Circular14,
          SheetTarget::Circular16, SheetTarget::Circular20, SheetTarget::Cross,
          SheetTarget::AprilTag36h11, SheetTarget::Chessboard}) {
        if (name == sheet_target_name(target)) {
            return target;
        }
    }
    return std::nullopt;
}

int sheet_target_code_count(SheetTarget target) {
    if (target == SheetTarget::Chessboard) {
        throw ValueError("marker sheet: a chessboard has no code space");
    }
    const int bits = circular_bits(target);
    if (bits != 0) {
        return 2 * static_cast<int>(circular_codes(bits).size());
    }
    if (target == SheetTarget::AprilTag36h11) {
        return static_cast<int>(
            tag_family(AprilTagFamily::Tag36h11).codes.size());
    }
    return 0;
}

std::vector<std::uint8_t> marker_sheet_pdf(const SheetOptions& options) {
    const PageSetup setup = make_setup(options);
    PdfDocument pdf;
    for (int page = 0; page < setup.pages; ++page) {
        pdf.add_page(setup.paper.width_mm * kMmToPt,
                     setup.paper.height_mm * kMmToPt);
        emit_shapes(pdf, build_page(setup, page), setup.paper);
    }
    return pdf.bytes();
}

Image marker_sheet_raster(const SheetOptions& options, double dpi, int page) {
    if (!(dpi > 0.0) || !std::isfinite(dpi)) {
        throw ValueError("marker sheet: dpi must be positive and finite");
    }
    if (page < 0) {
        throw ValueError("marker sheet: page index must not be negative");
    }
    const PageSetup setup = make_setup(options);
    if (page >= setup.pages) {
        throw ValueError("marker sheet: page index is outside the sheet");
    }
    return raster_page(build_page(setup, page), setup.paper, dpi);
}

}  // namespace alphashape
