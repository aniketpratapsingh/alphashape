#pragma once

/// @file marker_sheet.hpp
/// @brief Printable sheets of coded markers and calibration boards (WP-13.17).
///
/// The sheets reproduce, at an exact physical size, the same target geometry
/// our decoders read (target_patterns.hpp, tag_families.hpp, cross_decoder.hpp,
/// board_detector.hpp). A sheet printed at 100% should therefore decode with
/// `decode_circular`, `decode_tags`, `detect_crosses` or `detect_board`
/// without any rescaling.
///
/// Paper sizes are the ISO A4 and US Letter rectangles. Layout is centred
/// inside a printable margin of at least 10 mm; the footer baseline also sits
/// inside that bottom margin, at least 5 mm from the paper edge, so ordinary
/// printers do not clip it. Coordinates in the PDF are
/// PDF user-space points (origin bottom-left, y up); the raster is a top-down
/// grayscale image. The geometry is shared: the raster evaluates the same
/// millimetre-space shapes analytically with 4x4 supersampling per pixel.
///
/// Text (code labels and the footer) is vector text in the PDF only; a raster
/// has no font rasteriser, so it omits the labels while keeping every target,
/// quiet zone and corner mark.
///
/// Circular codes print the documented centre dot for parity 1 and a white
/// centre for parity 0. The WP-13.5 blob detector seeds a circular candidate
/// from that dark centre dot, so a decode-back test uses parity-1 ids; the
/// printed geometry of every id is complete and a caller that already has a
/// candidate decodes both parities.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "image/image.hpp"

namespace alphashape {

/// Target families for which a sheet can be printed. Circular targets are
/// offered only at the ring widths with a table in target_patterns.hpp, and
/// the square target only for the AprilTag family `decode_tags` supports as a
/// quad (36h11).
enum class SheetTarget {
    Circular12,
    Circular14,
    Circular16,
    Circular20,
    Cross,
    AprilTag36h11,
    Chessboard,
};

/// Sheet layout. `board_cols`/`board_rows` count chessboard SQUARES, not inner
/// corners; a board detector configured for this sheet takes
/// `board_cols - 1` by `board_rows - 1` inner corners. A board is rotated 90
/// degrees when that is the only orientation that fits the paper. For the
/// coded and cross families `count == 0` fills the first page, trimmed to the
/// codes the family still has from `first_code`; a positive count spans as
/// many pages as needed. A chessboard is always a single page.
struct SheetOptions {
    SheetTarget target = SheetTarget::Circular12;
    std::string paper = "A4";  ///< "A4" (210x297 mm) or "Letter" (8.5x11 in).
    /// Circular outer diameter / cross side. For an AprilTag this is the
    /// outer extent of the black border; the white margin cell lies outside it
    /// inside the quiet zone.
    double target_size_mm = 25.0;
    double quiet_zone_mm = 5.0;     ///< White border around each target.
    int first_code = 1;             ///< First printed code id.
    int count = 0;                  ///< 0 = fill the first page.
    bool corner_marks = true;       ///< Cut/registration marks at grid corners.
    int board_cols = 9;             ///< Chessboard squares across.
    int board_rows = 6;             ///< Chessboard squares down.
    double square_mm = 25.0;        ///< Chessboard square edge.
    bool labels = true;             ///< Code label and footer text (PDF only).
};

/// Lower-case CLI/Python name, e.g. "circular12" or "apriltag36h11".
const char* sheet_target_name(SheetTarget target);

/// Parse a name produced by `sheet_target_name`. Nullopt for anything else.
std::optional<SheetTarget> parse_sheet_target(std::string_view name);

/// Number of distinct codes the target family can print. Cross returns 0
/// (uncoded). Throws ValueError for Chessboard, which has no code space.
int sheet_target_code_count(SheetTarget target);

/// Number of pages the options produce. Throws ValueError for the same
/// invalid layouts as marker_sheet_pdf.
int marker_sheet_page_count(const SheetOptions& options);

/// Serialise a complete PDF. Throws ValueError for an invalid layout (paper
/// name, size, or a code range outside the family) and Error for I/O shapes.
std::vector<std::uint8_t> marker_sheet_pdf(const SheetOptions& options);

/// Render one page as an 8-bit single-channel image. `dpi` is the print
/// resolution; the page raster is `round(page_mm / 25.4 * dpi)` pixels.
/// Throws ValueError for a non-positive dpi or an out-of-range page index.
Image marker_sheet_raster(const SheetOptions& options, double dpi,
                          int page = 0);

}  // namespace alphashape
