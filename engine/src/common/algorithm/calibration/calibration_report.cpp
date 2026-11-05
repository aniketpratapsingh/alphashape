#include "common/algorithm/calibration/calibration_report.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "base/io/pdf.hpp"
#include "image/io/image_io.hpp"

namespace alphashape {
namespace {

constexpr int kPlotColumns = 17;
constexpr int kPlotRows = 13;
constexpr double kArrowFraction = 0.08;

/// snprintf follows LC_NUMERIC, so these values use the process decimal
/// separator. The applications keep LC_NUMERIC at "C", which makes it '.';
/// they are not locale-independent.
std::string format_number(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%.6g", value);
    return std::string(buffer);
}

std::string format_fixed2(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%.2f", value);
    return std::string(buffer);
}

std::string escape_html(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (char ch : text) {
        switch (ch) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            case '\'': out += "&#39;"; break;
            default: out.push_back(ch); break;
        }
    }
    return out;
}

const char kBase64Alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64(const std::vector<std::uint8_t>& data) {
    std::string out;
    out.reserve((data.size() + 2U) / 3U * 4U);
    std::size_t i = 0;
    while (i + 2U < data.size()) {
        const std::uint32_t chunk =
            (static_cast<std::uint32_t>(data[i]) << 16U) |
            (static_cast<std::uint32_t>(data[i + 1U]) << 8U) |
            static_cast<std::uint32_t>(data[i + 2U]);
        out.push_back(kBase64Alphabet[(chunk >> 18U) & 0x3FU]);
        out.push_back(kBase64Alphabet[(chunk >> 12U) & 0x3FU]);
        out.push_back(kBase64Alphabet[(chunk >> 6U) & 0x3FU]);
        out.push_back(kBase64Alphabet[chunk & 0x3FU]);
        i += 3U;
    }
    if (i < data.size()) {
        std::uint32_t chunk = static_cast<std::uint32_t>(data[i]) << 16U;
        const bool two = i + 1U < data.size();
        if (two) {
            chunk |= static_cast<std::uint32_t>(data[i + 1U]) << 8U;
        }
        out.push_back(kBase64Alphabet[(chunk >> 18U) & 0x3FU]);
        out.push_back(kBase64Alphabet[(chunk >> 12U) & 0x3FU]);
        out.push_back(two ? kBase64Alphabet[(chunk >> 6U) & 0x3FU] : '=');
        out.push_back('=');
    }
    return out;
}

bool listed(const std::vector<std::string>& names, const std::string& name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}

/// A term is pruned when the solver zeroed it, and fixed when it was held from
/// the start. Both have a zero sigma; only the mark distinguishes them.
bool is_pruned(const CalibrationResult& result, std::size_t index) {
    return listed(result.pruned, std::string(kCalibrationParamNames[index]));
}

bool is_fixed(const CalibrationResult& result, std::size_t index) {
    return !is_pruned(result, index) && result.sigma[index] == 0.0;
}

void draw_line(Image& image, int x0, int y0, int x1, int y1) {
    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;
    while (true) {
        if (x0 >= 0 && x0 < image.width() && y0 >= 0 && y0 < image.height()) {
            image.at(x0, y0, 0) = static_cast<std::uint8_t>(40);
            image.at(x0, y0, 1) = static_cast<std::uint8_t>(40);
            image.at(x0, y0, 2) = static_cast<std::uint8_t>(40);
        }
        if (x0 == x1 && y0 == y1) {
            break;
        }
        const int doubled = 2 * error;
        if (doubled >= dy) {
            error += dy;
            x0 += sx;
        }
        if (doubled <= dx) {
            error += dx;
            y0 += sy;
        }
    }
}

double coefficient_value(const Calibration& calibration, std::size_t index) {
    switch (index) {
        case 0: return calibration.f;
        case 1: return calibration.cx;
        case 2: return calibration.cy;
        case 3: return calibration.b1;
        case 4: return calibration.b2;
        case 5: return calibration.k1;
        case 6: return calibration.k2;
        case 7: return calibration.k3;
        case 8: return calibration.k4;
        case 9: return calibration.p1;
        case 10: return calibration.p2;
        case 11: return calibration.p3;
        case 12: return calibration.p4;
        default: return 0.0;
    }
}

std::vector<std::size_t> free_terms(const CalibrationResult& result) {
    std::vector<std::size_t> free;
    for (std::size_t i = 0; i < static_cast<std::size_t>(kCalibrationParamCount);
         ++i) {
        if (!is_pruned(result, i) && !is_fixed(result, i)) {
            free.push_back(i);
        }
    }
    return free;
}

std::string term_status(const CalibrationResult& result, std::size_t index) {
    if (is_pruned(result, index)) {
        return "pruned";
    }
    if (is_fixed(result, index)) {
        return "fixed";
    }
    return "free";
}

void append_coefficients(std::string& html, const CalibrationResult& result) {
    html += "<h3>Coefficients</h3>\n<table>\n";
    html += "<tr><th>Name</th><th>Value</th><th>1&sigma;</th>";
    html += "<th>Status</th></tr>\n";
    for (std::size_t i = 0; i < static_cast<std::size_t>(kCalibrationParamCount);
         ++i) {
        const std::string name(kCalibrationParamNames[i]);
        const std::string status = term_status(result, i);
        html += "<tr><td>" + escape_html(name) + "</td><td>" +
                format_number(coefficient_value(result.calibration, i)) +
                "</td><td>";
        html += status == "free" ? format_number(result.sigma[i]) : "&mdash;";
        html += "</td><td>" + status + "</td></tr>\n";
    }
    html += "</table>\n";
}

void append_correlation(std::string& html, const CalibrationResult& result) {
    const std::vector<std::size_t> free = free_terms(result);
    html += "<h3>Correlation (free terms)</h3>\n";
    if (free.empty()) {
        html += "<p>No free terms.</p>\n";
        return;
    }
    const Eigen::MatrixXd& correlation = result.correlation;
    const Eigen::Index count = static_cast<Eigen::Index>(kCalibrationParamCount);
    if (correlation.rows() < count || correlation.cols() < count) {
        html += "<p>Not available.</p>\n";
        return;
    }
    html += "<table>\n<tr><th></th>";
    for (std::size_t i : free) {
        html += "<th>" + escape_html(std::string(kCalibrationParamNames[i])) +
                "</th>";
    }
    html += "</tr>\n";
    for (std::size_t row : free) {
        html += "<tr><th>" + escape_html(std::string(kCalibrationParamNames[row])) +
                "</th>";
        for (std::size_t col : free) {
            html += "<td>" +
                    format_fixed2(correlation(
                        static_cast<Eigen::Index>(row),
                        static_cast<Eigen::Index>(col))) +
                    "</td>";
        }
        html += "</tr>\n";
    }
    html += "</table>\n";
}

void append_views(std::string& html, const CalibrationReportInput& input) {
    html += "<h3>Views</h3>\n<table>\n";
    html += "<tr><th>Camera</th><th>Corners</th><th>RMS px</th>";
    html += "<th>Max px</th></tr>\n";
    for (std::size_t i = 0; i < input.result.views.size(); ++i) {
        const ViewError& view = input.result.views[i];
        const std::string label =
            i < input.view_labels.size()
                ? input.view_labels[i]
                : std::to_string(view.camera_id);
        html += "<tr><td>" + escape_html(label) + "</td><td>" +
                std::to_string(view.corners) + "</td><td>" +
                format_number(view.rms_px) + "</td><td>" +
                format_number(view.max_px) + "</td></tr>\n";
    }
    html += "</table>\n";
}

}  // namespace
Image render_distortion_plot(const Calibration& calibration, int width_px,
                             int height_px) {
    if (width_px <= 0 || height_px <= 0) {
        throw ValueError("render_distortion_plot: plot size must be positive");
    }
    if (calibration.width <= 0 || calibration.height <= 0 ||
        !(calibration.f > 0.0)) {
        throw ValueError("render_distortion_plot: invalid calibration geometry");
    }
    Image image(width_px, height_px, 3);
    for (int y = 0; y < height_px; ++y) {
        for (int x = 0; x < width_px; ++x) {
            image.at(x, y, 0) = static_cast<std::uint8_t>(255);
            image.at(x, y, 1) = static_cast<std::uint8_t>(255);
            image.at(x, y, 2) = static_cast<std::uint8_t>(255);
        }
    }
    const double sensor_w = static_cast<double>(calibration.width);
    const double sensor_h = static_cast<double>(calibration.height);
    const double cx = sensor_w / 2.0 + calibration.cx;
    const double cy = sensor_h / 2.0 + calibration.cy;
    struct Sample {
        double ideal_u = 0.0;
        double ideal_v = 0.0;
        double source_u = 0.0;
        double source_v = 0.0;
    };
    std::vector<Sample> samples;
    double max_shift = 0.0;
    for (int row = 0; row < kPlotRows; ++row) {
        const double v = (static_cast<double>(row) + 0.5) * sensor_h / kPlotRows;
        for (int col = 0; col < kPlotColumns; ++col) {
            const double u =
                (static_cast<double>(col) + 0.5) * sensor_w / kPlotColumns;
            const double x = (u - cx) / calibration.f;
            const double y = (v - cy) / calibration.f;
            try {
                const Eigen::Vector2d source =
                    calibration.project(Eigen::Vector3d(x, y, 1.0));
                if (!std::isfinite(source.x()) || !std::isfinite(source.y())) {
                    continue;
                }
                Sample sample;
                sample.ideal_u = u;
                sample.ideal_v = v;
                sample.source_u = source.x();
                sample.source_v = source.y();
                samples.push_back(sample);
                const double du =
                    (source.x() - u) * static_cast<double>(width_px) / sensor_w;
                const double dv = (source.y() - v) *
                                  static_cast<double>(height_px) / sensor_h;
                max_shift = std::max(max_shift, std::hypot(du, dv));
            } catch (const Error&) {
                // Undefined projection: nothing sensible to draw for it.
            }
        }
    }
    if (samples.empty() || !(max_shift > 0.0)) {
        return image;
    }
    const double magnification =
        kArrowFraction * static_cast<double>(width_px) / max_shift;
    for (const Sample& sample : samples) {
        const double ideal_x =
            sample.ideal_u * static_cast<double>(width_px) / sensor_w;
        const double ideal_y =
            sample.ideal_v * static_cast<double>(height_px) / sensor_h;
        const double end_x =
            ideal_x + (sample.source_u - sample.ideal_u) *
                          static_cast<double>(width_px) / sensor_w * magnification;
        const double end_y =
            ideal_y + (sample.source_v - sample.ideal_v) *
                          static_cast<double>(height_px) / sensor_h * magnification;
        draw_line(image, static_cast<int>(std::lround(ideal_x)),
                  static_cast<int>(std::lround(ideal_y)),
                  static_cast<int>(std::lround(end_x)),
                  static_cast<int>(std::lround(end_y)));
    }
    return image;
}

std::string calibration_report_html(
    const std::vector<CalibrationReportInput>& sensors) {
    std::string html;
    html += "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n";
    html += "<meta charset=\"utf-8\">\n";
    html += "<title>Camera calibration report</title>\n<style>\n";
    html += "body{font-family:sans-serif;margin:2em;color:#222}\n";
    html += "table{border-collapse:collapse;margin:0.5em 0 1.5em}\n";
    html += "th,td{border:1px solid #999;padding:2px 8px;text-align:right}\n";
    html += "th{background:#eee}\n.skip{color:#a00}\nimg{max-width:100%}\n";
    html += "</style>\n</head>\n<body>\n<h1>Camera calibration report</h1>\n";
    for (const CalibrationReportInput& input : sensors) {
        html += "<h2>" + escape_html(input.sensor_label) + "</h2>\n";
        if (!input.skipped.empty()) {
            html += "<p class=\"skip\">Skipped: " +
                    escape_html(input.skipped) + "</p>\n";
            continue;
        }
        const CalibrationResult& result = input.result;
        html += "<p>Views used: " + std::to_string(result.views_used) +
                " &middot; RMS: " + format_number(result.rms_px) +
                " px &middot; Max: " + format_number(result.max_px) +
                " px</p>\n";
        append_coefficients(html, result);
        append_correlation(html, result);
        append_views(html, input);
        const Image plot = render_distortion_plot(result.calibration, 640, 480);
        const std::vector<std::uint8_t> png = encode_png(plot);
        html += "<h3>Distortion</h3>\n";
        html += "<img alt=\"distortion\" src=\"data:image/png;base64,";
        html += base64(png);
        html += "\">\n";
    }
    html += "</body>\n</html>\n";
    return html;
}

std::vector<std::uint8_t> calibration_report_pdf(
    const std::vector<CalibrationReportInput>& sensors) {
    PdfDocument pdf;
    double cursor = 802.0;
    auto ensure = [&pdf, &cursor](double needed) {
        if (cursor - needed < 40.0) {
            pdf.add_page(595.0, 842.0);
            cursor = 802.0;
        }
    };
    auto put = [&pdf, &cursor, &ensure](const std::string& text, double size,
                                        bool bold) {
        ensure(size + 2.0);
        pdf.text(40.0, cursor - size, size, text, bold);
        cursor -= size + 4.0;
    };
    pdf.add_page(595.0, 842.0);
    put("Camera calibration report", 16.0, true);
    for (const CalibrationReportInput& input : sensors) {
        put(input.sensor_label, 13.0, true);
        if (!input.skipped.empty()) {
            put("Skipped: " + input.skipped, 10.0, false);
            continue;
        }
        const CalibrationResult& result = input.result;
        put("Views used: " + std::to_string(result.views_used) + "   RMS: " +
                format_number(result.rms_px) + " px   Max: " +
                format_number(result.max_px) + " px",
            10.0, false);
        put("Coefficients", 11.0, true);
        for (std::size_t i = 0;
             i < static_cast<std::size_t>(kCalibrationParamCount); ++i) {
            const std::string status = term_status(result, i);
            const std::string sigma =
                status == "free" ? format_number(result.sigma[i]) : "-";
            put(std::string(kCalibrationParamNames[i]) + " = " +
                    format_number(coefficient_value(result.calibration, i)) +
                    "   1 sigma " + sigma + "   " + status,
                10.0, false);
        }
        put("Correlation (free terms)", 11.0, true);
        const std::vector<std::size_t> free = free_terms(result);
        const Eigen::Index count =
            static_cast<Eigen::Index>(kCalibrationParamCount);
        if (free.empty()) {
            put("(no free terms)", 10.0, false);
        } else if (result.correlation.rows() < count ||
                   result.correlation.cols() < count) {
            put("(not available)", 10.0, false);
        } else {
            for (std::size_t row : free) {
                std::string text =
                    std::string(kCalibrationParamNames[row]) + ":";
                for (std::size_t col : free) {
                    text += " " + format_fixed2(result.correlation(
                                      static_cast<Eigen::Index>(row),
                                      static_cast<Eigen::Index>(col)));
                }
                put(text, 10.0, false);
            }
        }
        put("Views", 11.0, true);
        for (std::size_t i = 0; i < result.views.size(); ++i) {
            const ViewError& view = result.views[i];
            const std::string label =
                i < input.view_labels.size()
                    ? input.view_labels[i]
                    : std::to_string(view.camera_id);
            put(label + "   corners " + std::to_string(view.corners) +
                    "   RMS " + format_number(view.rms_px) + "   Max " +
                    format_number(view.max_px),
                10.0, false);
        }
        const Image plot = render_distortion_plot(result.calibration, 480, 360);
        const double image_height = 386.0;
        ensure(image_height + 14.0);
        std::vector<std::uint8_t> rgb(plot.data(),
                                      plot.data() + plot.size_bytes());
        pdf.image_rgb(40.0, cursor - image_height, 515.0, image_height,
                      plot.width(), plot.height(), rgb);
        cursor -= image_height + 14.0;
    }
    return pdf.bytes();
}

}  // namespace alphashape
