#pragma once

// EXIF metadata reader: a self-contained TIFF/EXIF byte parser.
//
// There is deliberately no libexif here. The parser is written from the
// published EXIF 2.32 / TIFF 6.0 tag and IFD layout so that we control exactly
// what happens with hostile input, and so that the camera/gps fields the
// pipeline needs are extracted without dragging in a dependency.
//
// ---------------------------------------------------------------------------
// Hostile input contract
// ---------------------------------------------------------------------------
// Every input is treated as hostile. The parser bounds-checks every offset and
// count, rejects `count * type_size` that overflows, refuses to follow an IFD
// that points back to itself (or to any IFD already visited), caps the value
// size it will materialise, and never reads outside the block it was given.
//
//   * Malformed EXIF data never throws: whatever was parsed before the problem
//     is returned, and the rest of the fields stay nullopt.
//   * A file that cannot be opened at all *does* throw Error — that is an I/O
//     failure, not malformed metadata.
//
// Both byte orders are handled. `read_exif` accepts a JPEG (APP1 "Exif\0\0"),
// a TIFF (IFD0 + ExifIFD + GPSIFD) and, when present, a PNG eXIf chunk.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace alphashape {

/// A WGS-84 position. Latitude is north-positive and longitude east-positive:
/// a GPSLatitudeRef of "S" and a GPSLongitudeRef of "W" negate the value.
struct GpsPosition {
    double latitude_deg = 0;
    double longitude_deg = 0;
    /// Present only when the GPS IFD carried an altitude. Negative when
    /// GPSAltitudeRef indicates "below sea level" (ref == 1).
    std::optional<double> altitude_m;
};

/// The EXIF fields the pipeline consumes, plus the raw focal-plane numbers the
/// sensor-grouping stage needs. Every field is nullopt when absent or when the
/// value could not be parsed.
struct ExifData {
    std::optional<std::string> make;               // 0x010F
    std::optional<std::string> model;              // 0x0110
    std::optional<std::string> body_serial;        // 0xA431
    std::optional<std::string> lens_model;         // 0xA434
    std::optional<std::string> datetime_original;  // 0x9003
    std::optional<double> focal_length_mm;             // 0x920A RATIONAL
    std::optional<double> focal_length_35mm;           // 0xA405 SHORT, widened to double
    std::optional<double> focal_plane_x_resolution;    // 0xA20E RATIONAL
    std::optional<double> focal_plane_y_resolution;    // 0xA20F RATIONAL
    std::optional<int> focal_plane_resolution_unit;    // 0xA210 SHORT
    std::optional<int> pixel_x_dimension;              // 0xA002 SHORT or LONG
    std::optional<int> pixel_y_dimension;              // 0xA003 SHORT or LONG
    std::optional<int> orientation;                    // 0x0112, 1..8
    /// Present only when both latitude and longitude were parsed.
    std::optional<GpsPosition> gps;

    /// Pixel pitch in micrometres, derived from FocalPlaneXResolution and
    /// FocalPlaneResolutionUnit. Nullopt when either is missing or the
    /// resolution is zero/negative.
    ///
    /// Units: 2 => 25400 um per inch, 3 => 10000 um per cm, 4 => 1000 um per mm,
    /// 5 => 1 um per um. The resolution is pixels per unit, so the pitch is
    /// unit_um / resolution.
    std::optional<double> pixel_size_um() const;
};

/// Reads the metadata of an image file.
///
///  * JPEG: walks the marker segments, takes the first APP1 segment whose
///    payload begins "Exif\0\0", and stops at SOS. Non-Exif APP1 segments
///    (XMP, "http://ns.adobe.com/xap/1.0/") are skipped.
///  * TIFF: parses IFD0 in place, then follows 0x8769 and 0x8825.
///  * PNG: parses the eXIf chunk when one is present.
///
/// Any other container, or a file with no metadata, yields a default ExifData.
/// Throws Error only when the file itself cannot be opened.
ExifData read_exif(const std::string& path);

/// Parses a raw TIFF-structured block: the block must begin with a TIFF header
/// ("II*\0" or "MM\0*"). This is what `read_exif` hands to the parser once it
/// has located the embedded block, and it is the entry point the tests drive
/// with hand-built data. Never throws.
ExifData parse_exif_tiff_block(const std::uint8_t* data, std::size_t size);

}  // namespace alphashape
