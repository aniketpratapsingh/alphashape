#include "base/io/pdf.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numbers>
#include <string>
#include <utility>

#include <zlib.h>

#include "base/utils/error.hpp"

namespace alphashape {
namespace {

/// Locale-independent fixed four-decimal token. A PDF real requires '.', never
/// the LC_NUMERIC separator, so the conversion stays in integer arithmetic
/// rather than the C library's floating-point formatting.
std::string fixed(double value) {
    const double scaled = value * 10000.0;
    if (!std::isfinite(scaled) || std::abs(scaled) > 9.0e15) {
        throw ValueError("PdfDocument: number is out of range");
    }
    const bool negative = std::signbit(value);
    std::int64_t units = static_cast<std::int64_t>(std::llround(scaled));
    if (units < 0) {
        units = -units;
    }
    const std::uint64_t magnitude = static_cast<std::uint64_t>(units);
    const std::uint64_t whole = magnitude / 10000U;
    const std::uint64_t fraction = magnitude % 10000U;
    char whole_text[24];
    char* whole_end =
        std::to_chars(whole_text, whole_text + sizeof whole_text, whole).ptr;
    char fraction_text[8];
    char* fraction_end =
        std::to_chars(fraction_text, fraction_text + sizeof fraction_text,
                      fraction)
            .ptr;
    std::string out;
    if (negative) {
        out.push_back('-');
    }
    out.append(whole_text, whole_end);
    out.push_back('.');
    const std::size_t length =
        static_cast<std::size_t>(fraction_end - fraction_text);
    out.append(4U - length, '0');
    out.append(fraction_text, fraction_end);
    return out;
}

/// Text sizes, stroke widths, and image matrices also accept the same
/// fixed-point token; a four-decimal PDF real is valid in every context.
std::string number(double value) {
    return fixed(value);
}

/// A grey level outside [0, 1] is a programming error, not a printable value.
void require_gray(double gray) {
    if (!(gray >= 0.0) || !(gray <= 1.0)) {
        throw ValueError("PdfDocument: grey level must be in [0, 1]");
    }
}

/// Append one cubic Bezier for the circular arc t0..t1 of radius `r`. The
/// caller guarantees |t1 - t0| <= pi/2, where the standard tangent control
/// point construction is accurate to a fraction of a thousandth.
void append_arc(std::string& out, double cx, double cy, double r, double t0,
                double t1) {
    const double k = (4.0 / 3.0) * std::tan(0.25 * (t1 - t0));
    const double x0 = cx + r * std::cos(t0);
    const double y0 = cy + r * std::sin(t0);
    const double x1 = cx + r * std::cos(t1);
    const double y1 = cy + r * std::sin(t1);
    const double c0x = x0 - k * r * std::sin(t0);
    const double c0y = y0 + k * r * std::cos(t0);
    const double c1x = x1 + k * r * std::sin(t1);
    const double c1y = y1 - k * r * std::cos(t1);
    out += fixed(c0x) + " " + fixed(c0y) + " " + fixed(c1x) + " " +
           fixed(c1y) + " " + fixed(x1) + " " + fixed(y1) + " c\n";
}

/// Append a closed circular subpath, four quadrants of at most 90 degrees.
void append_circle(std::string& out, double cx, double cy, double r) {
    out += fixed(cx + r) + " " + fixed(cy) + " m\n";
    const double quarter = 0.5 * std::numbers::pi_v<double>;
    for (int i = 0; i < 4; ++i) {
        const double t0 = static_cast<double>(i) * quarter;
        append_arc(out, cx, cy, r, t0, t0 + quarter);
    }
}

/// Append an arc of arbitrary span, split so no Bezier exceeds 90 degrees.
void append_arc_split(std::string& out, double cx, double cy, double r,
                      double a0, double a1) {
    const double quarter = 0.5 * std::numbers::pi_v<double>;
    const double span = a1 - a0;
    const int pieces =
        std::max(1, static_cast<int>(std::ceil(std::abs(span) / quarter)));
    const double step = span / static_cast<double>(pieces);
    double t = a0;
    for (int i = 0; i < pieces; ++i) {
        append_arc(out, cx, cy, r, t, t + step);
        t += step;
    }
}

/// PDF literal-string escaping: delimiters and the backslash are escaped;
/// anything non-ASCII or a control character becomes '?'.
std::string escape_text(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (char raw : text) {
        const unsigned char ch = static_cast<unsigned char>(raw);
        if (ch >= 0x20U && ch < 0x7FU) {
            if (ch == '(' || ch == ')' || ch == '\\') {
                out.push_back('\\');
            }
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('?');
        }
    }
    return out;
}

/// Raw zlib stream for the /FlateDecode image data.
std::vector<std::uint8_t> deflate(const std::vector<std::uint8_t>& input) {
    uLongf size = compressBound(static_cast<uLong>(input.size()));
    std::vector<std::uint8_t> out(static_cast<std::size_t>(size));
    const int status =
        compress2(out.data(), &size, input.data(),
                  static_cast<uLong>(input.size()), Z_BEST_SPEED);
    if (status != Z_OK) {
        throw Error("PdfDocument: zlib compression failed");
    }
    out.resize(static_cast<std::size_t>(size));
    return out;
}

/// Serialise the object bodies into a complete file with a byte-offset xref.
/// Every xref line is exactly 20 bytes, as the format requires.
std::vector<std::uint8_t> serialise(const std::vector<std::string>& objects) {
    std::vector<std::uint8_t> out;
    auto append = [&out](const std::string& text) {
        out.reserve(out.size() + text.size());
        for (char raw : text) {
            out.push_back(static_cast<std::uint8_t>(raw));
        }
    };
    append("%PDF-1.4\n");
    append("%\xE2\xE3\xCF\xD3\n");
    std::vector<std::size_t> offsets;
    offsets.reserve(objects.size());
    for (std::size_t i = 0; i < objects.size(); ++i) {
        offsets.push_back(out.size());
        append(std::to_string(i + 1) + " 0 obj\n");
        append(objects[i]);
        append("\nendobj\n");
    }
    const std::size_t xref = out.size();
    append("xref\n0 " + std::to_string(objects.size() + 1) + "\n");
    append("0000000000 65535 f \n");
    for (std::size_t offset : offsets) {
        char entry[24];
        std::snprintf(entry, sizeof entry, "%010llu 00000 n \n",
                      static_cast<unsigned long long>(offset));
        append(entry);
    }
    append("trailer\n<< /Size " + std::to_string(objects.size() + 1) +
           " /Root 1 0 R >>\nstartxref\n" + std::to_string(xref) +
           "\n%%EOF\n");
    return out;
}

}  // namespace

void PdfDocument::add_page(double width_pt, double height_pt) {
    if (!(width_pt > 0.0) || !(height_pt > 0.0)) {
        throw ValueError("PdfDocument: page size must be positive");
    }
    Page page;
    page.width = width_pt;
    page.height = height_pt;
    pages_.push_back(std::move(page));
}

void PdfDocument::text(double x, double y, double size_pt,
                       const std::string& ascii, bool bold) {
    if (pages_.empty()) {
        throw Error("PdfDocument: text() needs a page first");
    }
    const char* font = bold ? "F2" : "F1";
    pages_.back().content += "BT /";
    pages_.back().content += font;
    pages_.back().content += " " + number(size_pt) + " Tf ";
    pages_.back().content += number(x) + " " + number(y) + " Td (";
    pages_.back().content += escape_text(ascii) + ") Tj ET\n";
}

void PdfDocument::line(double x0, double y0, double x1, double y1,
                       double width_pt) {
    if (pages_.empty()) {
        throw Error("PdfDocument: line() needs a page first");
    }
    std::string& content = pages_.back().content;
    content += number(width_pt) + " w\n";
    content += number(x0) + " " + number(y0) + " m\n";
    content += number(x1) + " " + number(y1) + " l\nS\n";
}

void PdfDocument::fill_rect(double x, double y, double w, double h,
                            double gray) {
    if (pages_.empty()) {
        throw Error("PdfDocument: fill_rect() needs a page first");
    }
    require_gray(gray);
    std::string& content = pages_.back().content;
    content += fixed(gray) + " g\n";
    content += fixed(x) + " " + fixed(y) + " " + fixed(w) + " " +
               fixed(h) + " re\nf\n";
}

void PdfDocument::fill_polygon(
    const std::vector<std::array<double, 2>>& points, double gray) {
    if (pages_.empty()) {
        throw Error("PdfDocument: fill_polygon() needs a page first");
    }
    if (points.size() < 3U) {
        throw ValueError("PdfDocument: a polygon needs three points");
    }
    require_gray(gray);
    std::string& content = pages_.back().content;
    content += fixed(gray) + " g\n";
    content += fixed(points[0][0]) + " " + fixed(points[0][1]) + " m\n";
    for (std::size_t i = 1; i < points.size(); ++i) {
        content += fixed(points[i][0]) + " " + fixed(points[i][1]) + " l\n";
    }
    content += "f\n";
}

void PdfDocument::fill_circle(double cx, double cy, double r, double gray) {
    if (pages_.empty()) {
        throw Error("PdfDocument: fill_circle() needs a page first");
    }
    if (!(r > 0.0)) {
        throw ValueError("PdfDocument: circle radius must be positive");
    }
    require_gray(gray);
    std::string& content = pages_.back().content;
    content += fixed(gray) + " g\n";
    append_circle(content, cx, cy, r);
    content += "f\n";
}

void PdfDocument::fill_ring(double cx, double cy, double r_outer,
                            double r_inner, double gray) {
    if (pages_.empty()) {
        throw Error("PdfDocument: fill_ring() needs a page first");
    }
    if (!(r_outer > 0.0) || !(r_inner >= 0.0) || !(r_inner < r_outer)) {
        throw ValueError("PdfDocument: ring radii must satisfy 0 <= inner < outer");
    }
    require_gray(gray);
    std::string& content = pages_.back().content;
    content += fixed(gray) + " g\n";
    append_circle(content, cx, cy, r_outer);
    append_circle(content, cx, cy, r_inner);
    content += "f*\n";  // even-odd: the inner circle is a hole
}

void PdfDocument::fill_annular_sector(double cx, double cy, double r_inner,
                                      double r_outer, double a0, double a1,
                                      double gray) {
    if (pages_.empty()) {
        throw Error("PdfDocument: fill_annular_sector() needs a page first");
    }
    if (!(r_outer > 0.0) || !(r_inner >= 0.0) || !(r_inner < r_outer)) {
        throw ValueError(
            "PdfDocument: sector radii must satisfy 0 <= inner < outer");
    }
    if (!(a1 >= a0)) {
        throw ValueError("PdfDocument: sector angle span must be non-negative");
    }
    require_gray(gray);
    std::string& content = pages_.back().content;
    content += fixed(gray) + " g\n";
    const double sx = cx + r_inner * std::cos(a0);
    const double sy = cy + r_inner * std::sin(a0);
    content += fixed(sx) + " " + fixed(sy) + " m\n";
    content += fixed(cx + r_outer * std::cos(a0)) + " " +
               fixed(cy + r_outer * std::sin(a0)) + " l\n";
    append_arc_split(content, cx, cy, r_outer, a0, a1);
    content += fixed(cx + r_inner * std::cos(a1)) + " " +
               fixed(cy + r_inner * std::sin(a1)) + " l\n";
    append_arc_split(content, cx, cy, r_inner, a1, a0);
    content += "f\n";
}

void PdfDocument::image_rgb(double x, double y, double w, double h, int px_w,
                            int px_h, const std::vector<std::uint8_t>& rgb) {
    if (pages_.empty()) {
        throw Error("PdfDocument: image_rgb() needs a page first");
    }
    if (px_w <= 0 || px_h <= 0) {
        throw ValueError("PdfDocument: image size must be positive");
    }
    const std::size_t expected = static_cast<std::size_t>(px_w) *
                                 static_cast<std::size_t>(px_h) * 3U;
    if (rgb.size() != expected) {
        throw ValueError("PdfDocument: image data does not match its size");
    }
    Page& page = pages_.back();
    const std::size_t index = page.images.size();
    Image image;
    image.width = px_w;
    image.height = px_h;
    image.rgb = rgb;
    page.images.push_back(std::move(image));
    page.content += "q\n";
    page.content += number(w) + " 0 0 " + number(h) + " " + number(x) + " " +
                    number(y) + " cm\n";
    page.content += "/Im" + std::to_string(index) + " Do\nQ\n";
}

std::vector<std::uint8_t> PdfDocument::bytes() const {
    std::vector<std::string> objects;
    objects.push_back("<< /Type /Catalog /Pages 2 0 R >>");
    objects.push_back("");  // the pages node, filled once numbering is known
    objects.push_back(
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
    objects.push_back(
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>");

    std::string kids;
    int next = 5;
    for (const Page& page : pages_) {
        const int page_number = next++;
        const int content_number = next++;
        std::vector<int> image_numbers;
        for (std::size_t i = 0; i < page.images.size(); ++i) {
            image_numbers.push_back(next++);
        }
        kids += std::to_string(page_number) + " 0 R ";
        std::string xobjects;
        for (std::size_t i = 0; i < image_numbers.size(); ++i) {
            xobjects += "/Im" + std::to_string(i) + " " +
                        std::to_string(image_numbers[i]) + " 0 R ";
        }
        std::string dictionary =
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 " +
            number(page.width) + " " + number(page.height) +
            "] /Resources << /Font << /F1 3 0 R /F2 4 0 R >>";
        if (!xobjects.empty()) {
            dictionary += " /XObject << " + xobjects + ">>";
        }
        dictionary +=
            " >> /Contents " + std::to_string(content_number) + " 0 R >>";
        objects.push_back(std::move(dictionary));
        objects.push_back("<< /Length " +
                          std::to_string(page.content.size()) +
                          " >>\nstream\n" + page.content + "endstream");
        for (const Image& image : page.images) {
            const std::vector<std::uint8_t> data = deflate(image.rgb);
            std::string body =
                "<< /Type /XObject /Subtype /Image /Width " +
                std::to_string(image.width) + " /Height " +
                std::to_string(image.height) +
                " /ColorSpace /DeviceRGB /BitsPerComponent 8"
                " /Filter /FlateDecode /Length " +
                std::to_string(data.size()) + " >>\nstream\n";
            body.append(reinterpret_cast<const char*>(data.data()),
                        data.size());
            body += "\nendstream";
            objects.push_back(std::move(body));
        }
    }
    objects[1] = "<< /Type /Pages /Kids [" + kids + "] /Count " +
                 std::to_string(pages_.size()) + " >>";
    return serialise(objects);
}

void PdfDocument::save(const std::string& path) const {
    const std::vector<std::uint8_t> data = bytes();
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw Error("PdfDocument: cannot open '" + path + "'");
    }
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
    file.close();
    if (!file) {
        throw Error("PdfDocument: writing '" + path + "' failed");
    }
}

}  // namespace alphashape
