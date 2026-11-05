// WP-13.15b: PDF writer and calibration report tests.
//
// The PDF checks parse the writer's own xref table and confirm every in-use
// offset lands on an object header, which is what a reader relies on. The
// report checks feed a hand-built result (no solve) and inspect the output.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/io/pdf.hpp"
#include "base/utils/error.hpp"
#include "common/algorithm/calibration/calibration_report.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

CalibrationReportInput sample_input() {
    CalibrationReportInput input;
    input.sensor_label = "Main";
    input.result.calibration =
        Calibration::initial(CameraType::Frame, 640, 480, 500.0);
    input.result.calibration.k1 = -0.1;
    input.result.rms_px = 0.25;
    input.result.max_px = 0.9;
    input.result.views_used = 4;
    input.result.sigma.fill(0.0);
    input.result.sigma[0] = 1.5;  // f is free, so it has a sigma
    ViewError view;
    view.camera_id = 7;
    view.corners = 54;
    view.rms_px = 0.2;
    view.max_px = 0.6;
    input.result.views.push_back(view);
    input.view_labels = {"a<b&c"};
    return input;
}

}  // namespace
namespace {

std::vector<std::uint8_t> sample_pdf() {
    PdfDocument pdf;
    pdf.add_page(595.0, 842.0);
    pdf.text(40.0, 800.0, 12.0, "Hello (world) \\ test", true);
    pdf.line(40.0, 790.0, 200.0, 790.0, 0.5);
    const std::vector<std::uint8_t> rgb{255, 0, 0, 0, 255, 0};
    pdf.image_rgb(40.0, 600.0, 100.0, 50.0, 2, 1, rgb);
    return pdf.bytes();
}

}  // namespace

TEST(Pdf, WritesValidHeader) {
    const std::vector<std::uint8_t> bytes = sample_pdf();
    const std::string text(bytes.begin(), bytes.end());
    ASSERT_GE(text.size(), 8U);
    EXPECT_EQ(text.compare(0, 8, "%PDF-1.4"), 0);
    EXPECT_EQ(text.compare(text.size() - 6U, 6U, "%%EOF\n"), 0);

    const std::size_t start = text.rfind("startxref\n");
    ASSERT_NE(start, std::string::npos);
    const std::size_t number_start = start + 10U;
    const std::size_t number_end = text.find('\n', number_start);
    const std::size_t xref_offset = static_cast<std::size_t>(
        std::stoull(text.substr(number_start, number_end - number_start)));
    ASSERT_LT(xref_offset, text.size());
    EXPECT_EQ(text.compare(xref_offset, 4U, "xref"), 0);

    std::size_t position = text.find('\n', xref_offset) + 1U;
    const std::size_t header_end = text.find('\n', position);
    const std::string header = text.substr(position, header_end - position);
    const std::size_t space = header.find(' ');
    ASSERT_NE(space, std::string::npos);
    const int count = std::stoi(header.substr(space + 1U));
    position = header_end + 1U;
    EXPECT_EQ(text.compare(position, 20U, "0000000000 65535 f \n"), 0);
    position += 20U;
    for (int i = 1; i < count; ++i) {
        ASSERT_LE(position + 20U, text.size()) << "object " << i;
        const std::string entry = text.substr(position, 20U);
        const std::size_t offset =
            static_cast<std::size_t>(std::stoull(entry.substr(0, 10U)));
        const std::string expected = std::to_string(i) + " 0 obj";
        EXPECT_EQ(text.compare(offset, expected.size(), expected), 0)
            << "xref offset for object " << i;
        position += 20U;
    }
}

TEST(Pdf, TextBeforePageThrows) {
    PdfDocument pdf;
    EXPECT_THROW(pdf.text(10.0, 10.0, 10.0, "x"), Error);
}

TEST(CalibrationReport, HtmlContainsCoefficientsAndEscapesLabels) {
    const std::string html = calibration_report_html({sample_input()});
    EXPECT_NE(html.find("a&lt;b&amp;c"), std::string::npos);
    EXPECT_NE(html.find("k1"), std::string::npos);
    EXPECT_NE(html.find("data:image/png;base64,"), std::string::npos);
    EXPECT_NE(html.find("RMS"), std::string::npos);
    std::printf("[calibration report html] %zu bytes\n", html.size());
}

TEST(CalibrationReport, PdfIsWellFormed) {
    const std::vector<std::uint8_t> bytes =
        calibration_report_pdf({sample_input()});
    ASSERT_GE(bytes.size(), 8U);
    const std::string head(bytes.begin(), bytes.begin() + 8);
    EXPECT_EQ(head, "%PDF-1.4");
    std::printf("[calibration report pdf] %zu bytes\n", bytes.size());
}

}  // namespace alphashape
