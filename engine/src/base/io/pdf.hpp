#pragma once

// A minimal PDF 1.4 writer.
//
// Covers what printed sheets and calibration reports need: pages of text in
// the standard Type1 faces (Helvetica and Helvetica-Bold, not embedded),
// straight lines, and DeviceRGB images compressed with Flate. Content streams
// stay uncompressed so a failure is diagnosable in a text editor.
//
// Coordinates are PDF user space: points, origin bottom-left, y upwards.
// `text` takes the baseline position, not the top of the glyphs.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace alphashape {

class PdfDocument {
public:
    /// Append a blank page of the given size in points (A4 is 595 x 842).
    /// Throws ValueError for a non-positive size.
    void add_page(double width_pt, double height_pt);

    /// Draw one line of ASCII text. Callers handle layout. Throws Error when
    /// no page exists yet. Non-ASCII bytes and control characters become '?';
    /// the PDF string delimiters are escaped.
    void text(double x, double y, double size_pt, const std::string& ascii,
              bool bold = false);

    /// Draw a straight line of `width_pt` points. Throws Error without a page.
    void line(double x0, double y0, double x1, double y1, double width_pt);

    /// Filled vector primitives, all in points with the PDF native origin at
    /// the bottom-left. `gray` is a non-stroking grey level in [0, 1] (0 is
    /// black, 1 is white); values outside throw ValueError. Coordinates are
    /// written with four decimal places, relying on LC_NUMERIC being "C" --
    /// the same caveat as `text()` and `line()`. Every primitive throws Error
    /// when no page exists yet.
    void fill_rect(double x, double y, double w, double h, double gray);

    /// Fill a simple convex or concave polygon. Fewer than three points throw
    /// ValueError. The path closes implicitly with the `f` operator.
    void fill_polygon(const std::vector<std::array<double, 2>>& points,
                      double gray);

    /// Fill a circle with four cubic Bezier quadrants.
    void fill_circle(double cx, double cy, double r, double gray);

    /// Fill an annulus (outer radius `r_outer`, inner radius `r_inner`) using
    /// the even-odd rule, so the hole is real and not a white overpaint.
    /// Throws ValueError when 0 <= r_inner < r_outer does not hold.
    void fill_ring(double cx, double cy, double r_outer, double r_inner,
                   double gray);

    /// Fill the sector of an annulus between radii `r_inner` and `r_outer` and
    /// angles `a0`..`a1` radians (counter-clockwise in PDF coordinates; the
    /// caller mirrors the y axis for image-oriented patterns). Arc spans are
    /// split into pieces of at most 90 degrees, each a cubic Bezier. Throws
    /// ValueError for a negative inner radius, a non-positive outer radius, a
    /// negative angular span, or radii out of order.
    void fill_annular_sector(double cx, double cy, double r_inner,
                             double r_outer, double a0, double a1,
                             double gray);

    /// Draw an 8-bit RGB image into the rectangle (x, y, w, h), with `px_w` x
    /// `px_h` samples. `rgb` must hold px_w * px_h * 3 bytes. Throws Error when
    /// no page exists, and ValueError for a size/id mismatch.
    void image_rgb(double x, double y, double w, double h, int px_w, int px_h,
                   const std::vector<std::uint8_t>& rgb);

    /// Serialise the document. The result starts with "%PDF-1.4" and ends with
    /// "%%EOF\n". An empty document with no pages is still well formed.
    std::vector<std::uint8_t> bytes() const;

    /// Write `bytes()` to `path`. Throws Error when the file cannot be
    /// written or a short write occurs.
    void save(const std::string& path) const;

private:
    struct Image {
        int width = 0;
        int height = 0;
        std::vector<std::uint8_t> rgb;
    };
    struct Page {
        double width = 0.0;
        double height = 0.0;
        std::string content;
        std::vector<Image> images;
    };

    std::vector<Page> pages_;
};

}  // namespace alphashape
