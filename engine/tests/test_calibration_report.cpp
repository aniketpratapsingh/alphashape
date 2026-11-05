// WP-13.15b: PDF writer and calibration report tests.
//
// The PDF checks parse the writer's own xref table and confirm every in-use
// offset lands on an object header, which is what a reader relies on. The
// report checks feed a hand-built result (no solve) and inspect the output.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/io/pdf.hpp"
#include "base/utils/error.hpp"
#include "common/algorithm/calibration/calibration_report.hpp"
#include "image/io/image_io.hpp"
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
    const std::string text(reinterpret_cast<const char*>(bytes.data()),
                           bytes.size());
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

TEST(Pdf, StreamLengthsMatch) {
    const std::vector<std::uint8_t> bytes = sample_pdf();
    const std::string text(reinterpret_cast<const char*>(bytes.data()),
                           bytes.size());
    std::size_t position = 0;
    int streams = 0;
    while ((position = text.find("stream\n", position)) != std::string::npos) {
        const std::size_t data_start = position + 7U;
        const std::size_t dictionary = text.rfind("<<", position);
        ASSERT_NE(dictionary, std::string::npos);
        // An image stream is filtered and the writer separates its data from
        // the keyword with a newline; unfiltered content streams end with
        // their own newline and are followed by "endstream" directly.
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

TEST(Pdf, TextBeforePageThrows) {
    PdfDocument pdf;
    EXPECT_THROW(pdf.text(10.0, 10.0, 10.0, "x"), Error);
}

namespace {

const char* const kBase64Alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// The base64 payload of the first PNG data URI in the HTML, or empty.
std::string data_uri_payload(const std::string& html) {
    const std::string marker = "data:image/png;base64,";
    const std::size_t start = html.find(marker);
    if (start == std::string::npos) {
        return {};
    }
    const std::size_t begin = start + marker.size();
    const std::size_t end = html.find('"', begin);
    if (end == std::string::npos) {
        return {};
    }
    return html.substr(begin, end - begin);
}

/// Reference decoder: enough to prove the payload is valid base64 of the PNG.
std::vector<std::uint8_t> decode_base64(const std::string& text) {
    std::vector<std::uint8_t> out;
    std::uint32_t accumulator = 0;
    int bits = 0;
    for (char ch : text) {
        if (ch == '=') {
            break;
        }
        const char* found = std::strchr(kBase64Alphabet, ch);
        if (found == nullptr) {
            return {};
        }
        accumulator =
            (accumulator << 6U) |
            static_cast<std::uint32_t>(found - kBase64Alphabet);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(
                static_cast<std::uint8_t>((accumulator >> bits) & 0xFFU));
        }
    }
    return out;
}

}  // namespace

TEST(CalibrationReport, Base64Padding) {
    // The encoder is file-local, so exercise it through the report's data URI.
    // The report's plot size is fixed at 640x480, so a 1x1 / 2x1 image cannot
    // be fed through it; the equivalent check is made on the payload the
    // report does emit: it must be a multiple of four, carry exactly the '='
    // padding its input length requires, and decode back to that same PNG.
    const CalibrationReportInput input = sample_input();
    const std::string html = calibration_report_html({input});
    const std::string payload = data_uri_payload(html);
    ASSERT_FALSE(payload.empty());
    ASSERT_EQ(payload.size() % 4U, 0U);

    const std::vector<std::uint8_t> png = encode_png(
        render_distortion_plot(input.result.calibration, 640, 480));
    const std::size_t padding = (3U - png.size() % 3U) % 3U;
    ASSERT_GE(payload.size(), padding);
    EXPECT_EQ(payload.compare(payload.size() - padding, padding,
                              std::string(padding, '=')),
              0)
        << "padding " << padding;
    EXPECT_EQ(decode_base64(payload), png);
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
    const std::string head(reinterpret_cast<const char*>(bytes.data()), 8U);
    EXPECT_EQ(head, "%PDF-1.4");
    std::printf("[calibration report pdf] %zu bytes\n", bytes.size());
}

}  // namespace alphashape
