// WP-13.17: printable marker and calibration sheets.
//
// The sheet renderer is checked against the independent decoders: a rasterised
// page must decode back to exactly the printed code ids, a circular target's
// outer diameter must measure 25 mm at 300 dpi, and a chessboard sheet must
// produce the full inner-corner lattice at the printed spacing.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "base/utils/error.hpp"
#include "common/algorithm/calibration/board_detector.hpp"
#include "common/algorithm/markers/apriltag_decoder.hpp"
#include "common/algorithm/markers/circular_decoder.hpp"
#include "common/algorithm/markers/marker_detector.hpp"
#include "common/algorithm/markers/marker_sheet.hpp"
#include "common/algorithm/markers/tag_families.hpp"
#include "common/algorithm/markers/target_patterns.hpp"
#include "image/image.hpp"

namespace alphashape {
namespace {

std::string pdf_text(const std::vector<std::uint8_t>& bytes) {
    return std::string(reinterpret_cast<const char*>(bytes.data()),
                       bytes.size());
}

double mediabox_number(const std::string& pdf, int index) {
    const std::string key = "/MediaBox [0 0 ";
    const std::size_t start = pdf.find(key);
    if (start == std::string::npos) {
        return -1.0;
    }
    std::size_t position = start + key.size();
    for (int i = 0; i < index; ++i) {
        position = pdf.find(' ', position) + 1U;
    }
    return std::stod(pdf.substr(position));
}

Image crop(const Image& source, int x0, int y0, int width, int height) {
    Image out(width, height, source.channels());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int sx = std::clamp(x0 + x, 0, source.width() - 1);
            const int sy = std::clamp(y0 + y, 0, source.height() - 1);
            for (int c = 0; c < source.channels(); ++c) {
                out.at(x, y, c) = source.at(sx, sy, c);
            }
        }
    }
    return out;
}

FloatImage to_float_image(const Image& image) {
    FloatImage out(image.width(), image.height(), 0.0F);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            out.at(x, y) = static_cast<float>(image.at(x, y, 0)) / 255.0F;
        }
    }
    return out;
}

Image raster(const SheetOptions& options, double dpi) {
    return marker_sheet_raster(options, dpi, 0);
}

SheetOptions coded(SheetTarget target, int count) {
    SheetOptions options;
    options.target = target;
    options.count = count;
    options.corner_marks = false;  // keep decoders looking only at targets
    return options;
}

TEST(MarkerSheet, PdfPageSizeExact) {
    SheetOptions a4;
    a4.target = SheetTarget::Circular12;
    a4.count = 1;
    const std::string a4_pdf = pdf_text(marker_sheet_pdf(a4));
    EXPECT_NEAR(mediabox_number(a4_pdf, 0), 595.276, 0.01);
    EXPECT_NEAR(mediabox_number(a4_pdf, 1), 841.890, 0.01);

    SheetOptions letter;
    letter.target = SheetTarget::Circular12;
    letter.count = 1;
    letter.paper = "Letter";
    const std::string letter_pdf = pdf_text(marker_sheet_pdf(letter));
    EXPECT_NEAR(mediabox_number(letter_pdf, 0), 612.0, 0.01);
    EXPECT_NEAR(mediabox_number(letter_pdf, 1), 792.0, 0.01);
}

TEST(MarkerSheet, PdfIsWellFormed) {
    const std::vector<std::uint8_t> bytes =
        marker_sheet_pdf(coded(SheetTarget::Circular16, 4));
    ASSERT_GE(bytes.size(), 8U);
    const std::string text = pdf_text(bytes);
    EXPECT_EQ(text.compare(0, 8, "%PDF-1.4"), 0);
    EXPECT_EQ(text.compare(text.size() - 6U, 6U, "%%EOF\n"), 0);
    std::size_t position = 0;
    int streams = 0;
    while ((position = text.find("stream\n", position)) != std::string::npos) {
        const std::size_t data_start = position + 7U;
        const std::size_t dictionary = text.rfind("<<", position);
        ASSERT_NE(dictionary, std::string::npos);
        const bool filtered = text.find("/Filter", dictionary) < position;
        const std::size_t end =
            filtered ? text.find("\nendstream", data_start)
                     : text.find("endstream", data_start);
        ASSERT_NE(end, std::string::npos) << "stream " << streams;
        const std::size_t payload = end - data_start;
        const std::size_t length_key = text.rfind("/Length ", position);
        ASSERT_NE(length_key, std::string::npos);
        ASSERT_GT(length_key, dictionary) << "stream " << streams;
        const std::size_t digits = length_key + 8U;
        const std::size_t digits_end =
            text.find_first_not_of("0123456789", digits);
        ASSERT_LT(digits_end, position);
        const std::size_t declared = static_cast<std::size_t>(
            std::stoull(text.substr(digits, digits_end - digits)));
        EXPECT_EQ(declared, payload) << "stream " << streams;
        ++streams;
        position = filtered ? end + 10U : end + 9U;
    }
    EXPECT_GT(streams, 0);
}

TEST(MarkerSheet, RenderDecodesBack) {
    // 200 dpi keeps the whole detection under the five-second budget; the
    // printed targets are still ~98 px in radius at this scale.
    const double dpi = 200.0;
    const int page_width = static_cast<int>(std::lround(210.0 / 25.4 * dpi));
    const int page_height = static_cast<int>(std::lround(100.0 / 25.4 * dpi));
    for (SheetTarget target :
         {SheetTarget::Circular12, SheetTarget::Circular14,
          SheetTarget::Circular16, SheetTarget::Circular20}) {
        const Image sheet = raster(coded(target, 4), dpi);
        const FloatImage gray =
            to_float_image(crop(sheet, 0, 0, page_width, page_height));
        BlobParams params;
        params.min_radius = 2;
        params.max_radius = 120;
        params.threshold = 0.015F;
        const std::vector<MarkerCandidate> blobs = detect_blobs(gray, params);
        const int bits = target == SheetTarget::Circular12   ? 12
                         : target == SheetTarget::Circular14 ? 14
                         : target == SheetTarget::Circular16 ? 16
                                                             : 20;
        std::set<int> decoded;
        int foreign = 0;
        for (const MarkerCandidate& blob : blobs) {
            const std::optional<CodedTarget> result =
                decode_circular(gray, blob, bits, 0);
            if (!result.has_value()) {
                continue;
            }
            if (result->code < 1 || result->code > 4) {
                ++foreign;
            }
            decoded.insert(result->code);
        }
        EXPECT_EQ(foreign, 0) << sheet_target_name(target);
        for (int code = 1; code <= 4; ++code) {
            EXPECT_TRUE(decoded.count(code) != 0U)
                << sheet_target_name(target) << " code " << code;
        }
    }

    const Image sheet = raster(coded(SheetTarget::AprilTag36h11, 4), dpi);
    const FloatImage gray =
        to_float_image(crop(sheet, 0, 0, page_width, page_height));
    QuadParams params;
    params.min_side = 8;
    params.max_side = 512;
    const std::vector<CodedTarget> tags =
        decode_tags(gray, AprilTagFamily::Tag36h11, -1, params);
    std::set<int> ids;
    for (const CodedTarget& tag : tags) {
        ids.insert(tag.code);
    }
    for (int code = 1; code <= 4; ++code) {
        EXPECT_TRUE(ids.count(code) != 0U) << "apriltag code " << code;
    }
    for (int code : ids) {
        EXPECT_GE(code, 1);
        EXPECT_LE(code, 4);
    }
}

TEST(MarkerSheet, TargetSizeIs25mm) {
    SheetOptions options = coded(SheetTarget::Circular12, 1);
    options.corner_marks = false;
    const double dpi = 300.0;
    const Image sheet = raster(options, dpi);
    int best_row = 0;
    int best_extent = -1;
    for (int y = 0; y < sheet.height(); ++y) {
        int first = -1;
        int last = -1;
        for (int x = 0; x < sheet.width(); ++x) {
            if (sheet.at(x, y, 0) < 128) {
                if (first < 0) {
                    first = x;
                }
                last = x;
            }
        }
        if (first >= 0 && last - first > best_extent) {
            best_extent = last - first;
            best_row = y;
        }
    }
    ASSERT_GE(best_extent, 0);
    const double threshold = 127.5;
    auto edge = [&](int y, bool left) -> double {
        const int step = left ? 1 : -1;
        int x = left ? 1 : sheet.width() - 2;
        const int stop = left ? sheet.width() : -1;
        while (x != stop) {
            const double current = static_cast<double>(sheet.at(x, y, 0));
            const double previous =
                static_cast<double>(sheet.at(x - step, y, 0));
            if (current <= threshold && previous > threshold) {
                const double fraction =
                    (previous - threshold) / (previous - current);
                return static_cast<double>(x - step) + step * fraction;
            }
            x += step;
        }
        return left ? 0.0 : static_cast<double>(sheet.width());
    };
    const double left = edge(best_row, true);
    const double right = edge(best_row, false);
    const double measured_mm = (right - left) * 25.4 / dpi;
    std::printf("[TargetSize] row %d diameter %.4f mm\n", best_row,
                measured_mm);
    EXPECT_NEAR(measured_mm, 25.0, 0.1);
}

TEST(MarkerSheet, ChessboardSheetMatchesSpec) {
    SheetOptions options;
    options.target = SheetTarget::Chessboard;
    options.board_cols = 9;
    options.board_rows = 6;
    options.square_mm = 20.0;
    options.corner_marks = false;
    const double dpi = 200.0;
    const Image sheet = raster(options, dpi);
    const int width = sheet.width();
    const int height = std::min(
        sheet.height(),
        static_cast<int>(std::lround(220.0 / 25.4 * dpi)));
    const FloatImage gray = to_float_image(crop(sheet, 0, 0, width, height));
    BoardSpec spec;
    spec.cols = options.board_cols - 1;
    spec.rows = options.board_rows - 1;
    spec.square_size_m = options.square_mm / 1000.0;
    const std::optional<BoardView> view = detect_board(gray, spec);
    ASSERT_TRUE(view.has_value());
    EXPECT_TRUE(view->complete);
    EXPECT_EQ(view->corners.size(),
              static_cast<std::size_t>(spec.cols * spec.rows));
    auto corner = [&](int col, int row) -> const BoardCorner* {
        for (const BoardCorner& c : view->corners) {
            if (c.col == col && c.row == row) {
                return &c;
            }
        }
        return nullptr;
    };
    double sum = 0.0;
    int count = 0;
    const double px_per_mm = dpi / 25.4;
    for (int row = 0; row < spec.rows; ++row) {
        for (int col = 0; col < spec.cols; ++col) {
            const BoardCorner* here = corner(col, row);
            ASSERT_NE(here, nullptr) << "corner " << col << "," << row;
            if (col + 1 < spec.cols) {
                const BoardCorner* next = corner(col + 1, row);
                ASSERT_NE(next, nullptr);
                sum += (next->pixel - here->pixel).norm() / px_per_mm;
                ++count;
            }
            if (row + 1 < spec.rows) {
                const BoardCorner* next = corner(col, row + 1);
                ASSERT_NE(next, nullptr);
                sum += (next->pixel - here->pixel).norm() / px_per_mm;
                ++count;
            }
        }
    }
    ASSERT_GT(count, 0);
    const double spacing = sum / static_cast<double>(count);
    std::printf("[Chessboard] corners %zu mean spacing %.4f mm\n",
                view->corners.size(), spacing);
    EXPECT_NEAR(spacing, 20.0, 0.05);
}

TEST(MarkerSheet, RejectsCodeRangeOutsideFamily) {
    SheetOptions options;
    options.target = SheetTarget::Circular12;
    options.first_code = 0;
    options.count = 100000;
    EXPECT_THROW(marker_sheet_pdf(options), ValueError);
}

TEST(MarkerSheet, ChessboardTooLargeForPaperThrows) {
    SheetOptions options;
    options.target = SheetTarget::Chessboard;
    options.board_cols = 9;
    options.board_rows = 6;
    options.square_mm = 100.0;
    EXPECT_THROW(marker_sheet_pdf(options), ValueError);
}

}  // namespace
}  // namespace alphashape
