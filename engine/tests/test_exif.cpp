// Tests for the EXIF reader.
//
// The bulk of the coverage drives parse_exif_tiff_block with hand-built TIFF
// blocks, so every tag layout, both byte orders and every hostile shape is
// constructed in-process rather than depending on a camera's output. A few
// tests then read the committed JPEG/TIFF/PNG fixtures that carry real EXIF.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "image/io/exif.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

#ifdef ALPHASHAPE_DATA_DIR
const fs::path kDataDir = ALPHASHAPE_DATA_DIR;
#else
// tests/test_exif.cpp -> the repository root is two levels up.
const fs::path kDataDir = fs::path(__FILE__).parent_path().parent_path() / "data";
#endif

std::string fixture(const std::string& name) { return (kDataDir / name).string(); }

// ---------------------------------------------------------------------------
// Byte-order-aware primitives
// ---------------------------------------------------------------------------

void put16(std::vector<std::uint8_t>& out, std::size_t at, std::uint16_t v, bool little) {
    if (little) {
        out[at] = static_cast<std::uint8_t>(v & 0xFF);
        out[at + 1] = static_cast<std::uint8_t>((v >> 8) & 0xFF);
    } else {
        out[at] = static_cast<std::uint8_t>((v >> 8) & 0xFF);
        out[at + 1] = static_cast<std::uint8_t>(v & 0xFF);
    }
}

void put32(std::vector<std::uint8_t>& out, std::size_t at, std::uint32_t v, bool little) {
    if (little) {
        for (int i = 0; i < 4; ++i) {
            out[at + static_cast<std::size_t>(i)] =
                static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF);
        }
    } else {
        for (int i = 0; i < 4; ++i) {
            out[at + static_cast<std::size_t>(i)] =
                static_cast<std::uint8_t>((v >> (8 * (3 - i))) & 0xFF);
        }
    }
}

// Field type codes.
constexpr std::uint16_t kAscii = 2;
constexpr std::uint16_t kShort = 3;
constexpr std::uint16_t kLong = 4;
constexpr std::uint16_t kRational = 5;
constexpr std::uint16_t kSRational = 10;

/// A minimal TIFF block assembler. Fields store their value bytes already in
/// the target byte order, so at build time only the layout (inline vs pooled,
/// and the offsets of sub-IFD pointers) has to be resolved.
class TiffBuilder {
public:
    explicit TiffBuilder(bool little) : little_(little) {}

    int add_ifd() {
        ifds_.emplace_back();
        return static_cast<int>(ifds_.size()) - 1;
    }

    void ascii(int ifd, std::uint16_t tag, const std::string& text) {
        Field field;
        field.tag = tag;
        field.type = kAscii;
        // A proper EXIF string is NUL-terminated, so the count includes it.
        field.count = static_cast<std::uint32_t>(text.size()) + 1;
        for (char c : text) {
            field.bytes.push_back(static_cast<std::uint8_t>(c));
        }
        field.bytes.push_back(0);
        ifds_.at(static_cast<std::size_t>(ifd)).push_back(std::move(field));
    }

    void short_value(int ifd, std::uint16_t tag, std::uint16_t value) {
        Field field;
        field.tag = tag;
        field.type = kShort;
        field.count = 1;
        field.bytes.resize(2);
        put16(field.bytes, 0, value, little_);
        ifds_.at(static_cast<std::size_t>(ifd)).push_back(std::move(field));
    }

    void long_value(int ifd, std::uint16_t tag, std::uint32_t value) {
        Field field;
        field.tag = tag;
        field.type = kLong;
        field.count = 1;
        field.bytes.resize(4);
        put32(field.bytes, 0, value, little_);
        ifds_.at(static_cast<std::size_t>(ifd)).push_back(std::move(field));
    }

    void rational(int ifd, std::uint16_t tag, std::uint32_t numerator, std::uint32_t denominator) {
        Field field;
        field.tag = tag;
        field.type = kRational;
        field.count = 1;
        field.bytes.resize(8);
        put32(field.bytes, 0, numerator, little_);
        put32(field.bytes, 4, denominator, little_);
        ifds_.at(static_cast<std::size_t>(ifd)).push_back(std::move(field));
    }

    void srational(int ifd, std::uint16_t tag, std::int32_t numerator, std::int32_t denominator) {
        rational(ifd, tag, static_cast<std::uint32_t>(numerator),
                 static_cast<std::uint32_t>(denominator));
        ifds_.at(static_cast<std::size_t>(ifd)).back().type = kSRational;
    }

    /// Three RATIONALs, the shape a GPS coordinate takes.
    void coordinate(int ifd, std::uint16_t tag, std::uint32_t degrees, std::uint32_t minutes,
                    std::uint32_t seconds) {
        Field field;
        field.tag = tag;
        field.type = kRational;
        field.count = 3;
        field.bytes.resize(24);
        const std::uint32_t values[3] = {degrees, minutes, seconds};
        for (int i = 0; i < 3; ++i) {
            put32(field.bytes, static_cast<std::size_t>(i) * 8, values[i], little_);
            put32(field.bytes, static_cast<std::size_t>(i) * 8 + 4, 1, little_);
        }
        ifds_.at(static_cast<std::size_t>(ifd)).push_back(std::move(field));
    }

    /// A LONG field whose value is another IFD's byte offset, resolved at build
    /// time. Passing the same IFD index back produces the self-reference the
    /// loop test needs.
    void pointer(int ifd, std::uint16_t tag, int target) {
        Field field;
        field.tag = tag;
        field.type = kLong;
        field.count = 1;
        field.target = target;
        ifds_.at(static_cast<std::size_t>(ifd)).push_back(std::move(field));
    }

    std::vector<std::uint8_t> build() const {
        const std::size_t n = ifds_.size();
        std::vector<std::uint32_t> ifd_off(n, 0);
        std::uint32_t off = 8;
        for (std::size_t i = 0; i < n; ++i) {
            ifd_off[i] = off;
            off += 2 + 12 * static_cast<std::uint32_t>(ifds_[i].size()) + 4;
        }
        const std::uint32_t pool_off = off;

        std::vector<std::uint8_t> out(pool_off, 0);
        out[0] = little_ ? 'I' : 'M';
        out[1] = little_ ? 'I' : 'M';
        put16(out, 2, 0x002A, little_);
        put32(out, 4, 8, little_);

        std::vector<std::uint8_t> pool;
        std::uint32_t pool_cursor = pool_off;
        std::size_t cursor = 8;
        for (std::size_t i = 0; i < n; ++i) {
            put16(out, cursor, static_cast<std::uint16_t>(ifds_[i].size()), little_);
            cursor += 2;
            for (const Field& f : ifds_[i]) {
                put16(out, cursor, f.tag, little_);
                put16(out, cursor + 2, f.type, little_);
                put32(out, cursor + 4, f.count, little_);

                std::vector<std::uint8_t> value = f.bytes;
                if (f.target >= 0) {
                    value.resize(4);
                    put32(value, 0, ifd_off[static_cast<std::size_t>(f.target)], little_);
                }
                if (value.size() <= 4) {
                    for (std::size_t k = 0; k < value.size(); ++k) {
                        out[cursor + 8 + k] = value[k];
                    }
                } else {
                    put32(out, cursor + 8, pool_cursor, little_);
                    pool.insert(pool.end(), value.begin(), value.end());
                    pool_cursor += static_cast<std::uint32_t>(value.size());
                }
                cursor += 12;
            }
            put32(out, cursor, 0, little_);  // no next IFD
            cursor += 4;
        }
        out.insert(out.end(), pool.begin(), pool.end());
        return out;
    }

private:
    struct Field {
        std::uint16_t tag = 0;
        std::uint16_t type = 0;
        std::uint32_t count = 0;
        std::vector<std::uint8_t> bytes;
        int target = -1;
    };

    bool little_ = true;
    std::vector<std::vector<Field>> ifds_;
};

ExifData parse(const std::vector<std::uint8_t>& block) {
    return parse_exif_tiff_block(block.data(), block.size());
}

/// Builds the same content in both byte orders and returns the two results, so
/// a test can assert the reader is order-agnostic.
std::pair<ExifData, ExifData> parse_both(const std::vector<std::uint8_t>& le,
                                         const std::vector<std::uint8_t>& be) {
    return {parse_exif_tiff_block(le.data(), le.size()),
            parse_exif_tiff_block(be.data(), be.size())};
}

// ---------------------------------------------------------------------------
// Basic fields, both byte orders
// ---------------------------------------------------------------------------

/// The canonical block: Make/Model in IFD0, everything else in the Exif IFD.
std::vector<std::uint8_t> build_camera_block(bool little) {
    TiffBuilder builder(little);
    const int ifd0 = builder.add_ifd();
    const int exif = builder.add_ifd();

    builder.ascii(ifd0, 0x010F, "TestMake");   // Make
    builder.ascii(ifd0, 0x0110, "TestModel");  // Model
    builder.short_value(ifd0, 0x0112, 6);      // Orientation (rotate 90 CW)
    builder.pointer(ifd0, 0x8769, exif);       // ExifIFDPointer

    builder.ascii(exif, 0x9003, "2023:03:14 15:09:26");  // DateTimeOriginal
    builder.rational(exif, 0x920A, 50, 1);               // FocalLength = 50 mm
    builder.short_value(exif, 0xA405, 75);               // FocalLengthIn35mmFilm = 75
    builder.rational(exif, 0xA20E, 4000, 1);             // FocalPlaneXResolution
    builder.rational(exif, 0xA20F, 4000, 1);             // FocalPlaneYResolution
    builder.short_value(exif, 0xA210, 2);                // FocalPlaneResolutionUnit = inch
    builder.short_value(exif, 0xA002, 6000);             // PixelXDimension as SHORT
    builder.long_value(exif, 0xA003, 4000);              // PixelYDimension as LONG
    builder.ascii(exif, 0xA431, "SN-42");                // BodySerialNumber
    builder.ascii(exif, 0xA434, "TestLens 50mm");        // LensModel
    return builder.build();
}

TEST(Exif, ParsesCameraFieldsLittleEndian) {
    const ExifData exif = parse(build_camera_block(true));
    ASSERT_TRUE(exif.make.has_value());
    EXPECT_EQ(*exif.make, "TestMake");
    ASSERT_TRUE(exif.model.has_value());
    EXPECT_EQ(*exif.model, "TestModel");
    ASSERT_TRUE(exif.datetime_original.has_value());
    EXPECT_EQ(*exif.datetime_original, "2023:03:14 15:09:26");
    ASSERT_TRUE(exif.orientation.has_value());
    EXPECT_EQ(*exif.orientation, 6);
    ASSERT_TRUE(exif.body_serial.has_value());
    EXPECT_EQ(*exif.body_serial, "SN-42");
    ASSERT_TRUE(exif.lens_model.has_value());
    EXPECT_EQ(*exif.lens_model, "TestLens 50mm");
}

TEST(Exif, ParsesCameraFieldsBigEndian) {
    const ExifData exif = parse(build_camera_block(false));
    ASSERT_TRUE(exif.make.has_value());
    EXPECT_EQ(*exif.make, "TestMake");
    ASSERT_TRUE(exif.model.has_value());
    EXPECT_EQ(*exif.model, "TestModel");
    ASSERT_TRUE(exif.orientation.has_value());
    EXPECT_EQ(*exif.orientation, 6);
    ASSERT_TRUE(exif.datetime_original.has_value());
    EXPECT_EQ(*exif.datetime_original, "2023:03:14 15:09:26");
}

TEST(Exif, ReadsFocalRationalsInBothOrders) {
    const auto [le, be] = parse_both(build_camera_block(true), build_camera_block(false));
    for (const ExifData* exif : {&le, &be}) {
        ASSERT_TRUE(exif->focal_length_mm.has_value());
        EXPECT_DOUBLE_EQ(*exif->focal_length_mm, 50.0);
        ASSERT_TRUE(exif->focal_length_35mm.has_value());
        EXPECT_DOUBLE_EQ(*exif->focal_length_35mm, 75.0);
        ASSERT_TRUE(exif->focal_plane_x_resolution.has_value());
        EXPECT_DOUBLE_EQ(*exif->focal_plane_x_resolution, 4000.0);
        ASSERT_TRUE(exif->focal_plane_y_resolution.has_value());
        EXPECT_DOUBLE_EQ(*exif->focal_plane_y_resolution, 4000.0);
        ASSERT_TRUE(exif->focal_plane_resolution_unit.has_value());
        EXPECT_EQ(*exif->focal_plane_resolution_unit, 2);
    }
}

TEST(Exif, ReadsPixelDimensionsAsShortAndAsLong) {
    const ExifData exif = parse(build_camera_block(true));
    ASSERT_TRUE(exif.pixel_x_dimension.has_value());
    EXPECT_EQ(*exif.pixel_x_dimension, 6000);  // stored as SHORT
    ASSERT_TRUE(exif.pixel_y_dimension.has_value());
    EXPECT_EQ(*exif.pixel_y_dimension, 4000);  // stored as LONG
}

TEST(Exif, FractionalFocalLengthIsExact) {
    // 17/5 mm must not be truncated to an integer.
    TiffBuilder builder(true);
    const int ifd0 = builder.add_ifd();
    const int exif = builder.add_ifd();
    builder.pointer(ifd0, 0x8769, exif);
    builder.rational(exif, 0x920A, 17, 5);
    const ExifData parsed = parse(builder.build());
    ASSERT_TRUE(parsed.focal_length_mm.has_value());
    EXPECT_DOUBLE_EQ(*parsed.focal_length_mm, 3.4);
}

TEST(Exif, TrimsTrailingNulsAndSpaces) {
    // The builder appends the NUL; the reader must also strip trailing spaces.
    TiffBuilder builder(true);
    const int ifd0 = builder.add_ifd();
    builder.ascii(ifd0, 0x010F, "Padded  ");
    const ExifData exif = parse(builder.build());
    ASSERT_TRUE(exif.make.has_value());
    EXPECT_EQ(*exif.make, "Padded");
}

TEST(Exif, OrientationOutOfRangeIsIgnored) {
    TiffBuilder builder(true);
    const int ifd0 = builder.add_ifd();
    builder.short_value(ifd0, 0x0112, 9);  // valid range is 1..8
    const ExifData exif = parse(builder.build());
    EXPECT_FALSE(exif.orientation.has_value());
}

TEST(Exif, HeaderOnlyBlockYieldsNothing) {
    const std::vector<std::uint8_t> block = {'I', 'I', 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00};
    const ExifData exif = parse(block);
    EXPECT_FALSE(exif.make.has_value());
    EXPECT_FALSE(exif.orientation.has_value());
}

TEST(Exif, NotATiffBlockYieldsNothing) {
    const std::vector<std::uint8_t> block = {'X', 'X', 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00};
    const ExifData exif = parse(block);
    EXPECT_FALSE(exif.make.has_value());
}

TEST(Exif, EmptyBlockYieldsNothing) {
    const ExifData exif = parse_exif_tiff_block(nullptr, 0);
    EXPECT_FALSE(exif.make.has_value());
}

// ---------------------------------------------------------------------------
// GPS
// ---------------------------------------------------------------------------

TEST(Exif, ParsesNorthernEasternGps) {
    TiffBuilder builder(true);
    const int ifd0 = builder.add_ifd();
    const int gps = builder.add_ifd();
    builder.pointer(ifd0, 0x8825, gps);
    builder.ascii(gps, 0x0001, "N");
    builder.coordinate(gps, 0x0002, 43, 30, 0);
    builder.ascii(gps, 0x0003, "E");
    builder.coordinate(gps, 0x0004, 79, 20, 0);
    const ExifData exif = parse(builder.build());

    ASSERT_TRUE(exif.gps.has_value());
    EXPECT_NEAR(exif.gps->latitude_deg, 43.5, 1e-9);
    EXPECT_NEAR(exif.gps->longitude_deg, 79.0 + 20.0 / 60.0, 1e-9);
    EXPECT_FALSE(exif.gps->altitude_m.has_value());
}

TEST(Exif, SouthernWesternGpsIsNegative) {
    TiffBuilder builder(true);
    const int ifd0 = builder.add_ifd();
    const int gps = builder.add_ifd();
    builder.pointer(ifd0, 0x8825, gps);
    builder.ascii(gps, 0x0001, "S");
    builder.coordinate(gps, 0x0002, 33, 52, 0);
    builder.ascii(gps, 0x0003, "W");
    builder.coordinate(gps, 0x0004, 151, 12, 0);
    const ExifData exif = parse(builder.build());

    ASSERT_TRUE(exif.gps.has_value());
    EXPECT_LT(exif.gps->latitude_deg, 0.0);
    EXPECT_LT(exif.gps->longitude_deg, 0.0);
    EXPECT_NEAR(exif.gps->latitude_deg, -(33.0 + 52.0 / 60.0), 1e-9);
    EXPECT_NEAR(exif.gps->longitude_deg, -(151.0 + 12.0 / 60.0), 1e-9);
}

TEST(Exif, GpsIsParsedInBothByteOrders) {
    auto make = [](bool little) {
        TiffBuilder builder(little);
        const int ifd0 = builder.add_ifd();
        const int gps = builder.add_ifd();
        builder.pointer(ifd0, 0x8825, gps);
        builder.ascii(gps, 0x0001, "S");
        builder.coordinate(gps, 0x0002, 12, 30, 0);
        builder.ascii(gps, 0x0003, "W");
        builder.coordinate(gps, 0x0004, 45, 15, 0);
        return builder.build();
    };
    const auto [le, be] = parse_both(make(true), make(false));
    for (const ExifData* exif : {&le, &be}) {
        ASSERT_TRUE(exif->gps.has_value());
        EXPECT_NEAR(exif->gps->latitude_deg, -12.5, 1e-9);
        EXPECT_NEAR(exif->gps->longitude_deg, -45.25, 1e-9);
    }
}

TEST(Exif, AltitudeRefOneIsBelowSeaLevel) {
    TiffBuilder builder(true);
    const int ifd0 = builder.add_ifd();
    const int gps = builder.add_ifd();
    builder.pointer(ifd0, 0x8825, gps);
    builder.ascii(gps, 0x0001, "N");
    builder.coordinate(gps, 0x0002, 1, 0, 0);
    builder.ascii(gps, 0x0003, "E");
    builder.coordinate(gps, 0x0004, 1, 0, 0);
    builder.short_value(gps, 0x0005, 1);    // altitude ref: below sea level
    builder.rational(gps, 0x0006, 350, 1);  // altitude magnitude
    const ExifData exif = parse(builder.build());

    ASSERT_TRUE(exif.gps.has_value());
    ASSERT_TRUE(exif.gps->altitude_m.has_value());
    EXPECT_DOUBLE_EQ(*exif.gps->altitude_m, -350.0);
}

TEST(Exif, AltitudeRefZeroIsAboveSeaLevel) {
    TiffBuilder builder(true);
    const int ifd0 = builder.add_ifd();
    const int gps = builder.add_ifd();
    builder.pointer(ifd0, 0x8825, gps);
    builder.ascii(gps, 0x0001, "N");
    builder.coordinate(gps, 0x0002, 1, 0, 0);
    builder.ascii(gps, 0x0003, "E");
    builder.coordinate(gps, 0x0004, 1, 0, 0);
    builder.short_value(gps, 0x0005, 0);    // above sea level
    builder.rational(gps, 0x0006, 120, 1);
    const ExifData exif = parse(builder.build());

    ASSERT_TRUE(exif.gps.has_value());
    ASSERT_TRUE(exif.gps->altitude_m.has_value());
    EXPECT_DOUBLE_EQ(*exif.gps->altitude_m, 120.0);
}

TEST(Exif, GpsNeedsBothLatitudeAndLongitude) {
    // Latitude with no longitude must not produce a position.
    TiffBuilder builder(true);
    const int ifd0 = builder.add_ifd();
    const int gps = builder.add_ifd();
    builder.pointer(ifd0, 0x8825, gps);
    builder.ascii(gps, 0x0001, "N");
    builder.coordinate(gps, 0x0002, 43, 30, 0);
    const ExifData exif = parse(builder.build());
    EXPECT_FALSE(exif.gps.has_value());
}

// ---------------------------------------------------------------------------
// pixel_size_um
// ---------------------------------------------------------------------------

/// Builds a block that carries only the two tags pixel_size_um needs.
ExifData pixel_size_block(std::uint32_t resolution, bool have_unit, std::uint16_t unit) {
    TiffBuilder builder(true);
    const int ifd0 = builder.add_ifd();
    const int exif = builder.add_ifd();
    builder.pointer(ifd0, 0x8769, exif);
    builder.rational(exif, 0xA20E, resolution, 1);
    if (have_unit) {
        builder.short_value(exif, 0xA210, unit);
    }
    return parse(builder.build());
}

TEST(Exif, PixelSizeUnits) {
    // unit_um / resolution, for each of the four units EXIF defines.
    ASSERT_TRUE(pixel_size_block(25400, true, 2).pixel_size_um().has_value());
    EXPECT_NEAR(*pixel_size_block(25400, true, 2).pixel_size_um(), 1.0, 1e-9);
    ASSERT_TRUE(pixel_size_block(10000, true, 3).pixel_size_um().has_value());
    EXPECT_NEAR(*pixel_size_block(10000, true, 3).pixel_size_um(), 1.0, 1e-9);
    ASSERT_TRUE(pixel_size_block(1000, true, 4).pixel_size_um().has_value());
    EXPECT_NEAR(*pixel_size_block(1000, true, 4).pixel_size_um(), 1.0, 1e-9);
    ASSERT_TRUE(pixel_size_block(1, true, 5).pixel_size_um().has_value());
    EXPECT_NEAR(*pixel_size_block(1, true, 5).pixel_size_um(), 1.0, 1e-9);
    // A realistic cm-unit sensor: 3000 px/cm gives ~3.33 um.
    ASSERT_TRUE(pixel_size_block(3000, true, 3).pixel_size_um().has_value());
    EXPECT_NEAR(*pixel_size_block(3000, true, 3).pixel_size_um(), 10000.0 / 3000.0, 1e-9);
}

TEST(Exif, PixelSizeNeedsUsableInputs) {
    EXPECT_FALSE(pixel_size_block(3000, false, 0).pixel_size_um().has_value());  // no unit
    EXPECT_FALSE(pixel_size_block(0, true, 3).pixel_size_um().has_value());      // zero res
    EXPECT_FALSE(pixel_size_block(3000, true, 1).pixel_size_um().has_value());   // unit 1 = none
    EXPECT_FALSE(pixel_size_block(3000, true, 99).pixel_size_um().has_value());  // unknown unit
}

// ---------------------------------------------------------------------------
// Hostile input: every one must return partial data, never crash
// ---------------------------------------------------------------------------

TEST(Exif, IfdOffsetBeyondEndIsIgnored) {
    // IFD0 is claimed at offset 0x1000 in an 8-byte block.
    std::vector<std::uint8_t> block = {'I', 'I', 0x2A, 0x00, 0x00, 0x10, 0x00, 0x00};
    const ExifData exif = parse(block);
    EXPECT_FALSE(exif.make.has_value());
    EXPECT_FALSE(exif.focal_length_mm.has_value());
}

TEST(Exif, IfdPointingToItselfTerminates) {
    // A single IFD whose ExifIFDPointer points back at its own offset. Without
    // loop protection this would recurse forever.
    TiffBuilder builder(true);
    const int ifd0 = builder.add_ifd();
    builder.ascii(ifd0, 0x010F, "Loopy");
    builder.pointer(ifd0, 0x8769, ifd0);  // points at itself
    const ExifData exif = parse(builder.build());
    ASSERT_TRUE(exif.make.has_value());
    EXPECT_EQ(*exif.make, "Loopy");
    EXPECT_FALSE(exif.focal_length_mm.has_value());
}

TEST(Exif, TwoPointersToTheSameIfdAreNotFollowedTwice) {
    TiffBuilder builder(true);
    const int ifd0 = builder.add_ifd();
    const int sub = builder.add_ifd();
    builder.pointer(ifd0, 0x8769, sub);
    builder.pointer(ifd0, 0x8825, sub);  // same offset via the other pointer
    builder.ascii(sub, 0x9003, "2024:01:01 00:00:00");
    const ExifData exif = parse(builder.build());
    ASSERT_TRUE(exif.datetime_original.has_value());
    EXPECT_EQ(*exif.datetime_original, "2024:01:01 00:00:00");
}

TEST(Exif, HugeIfdCountIsRejected) {
    // count = 0xFFFFFFFF would claim 51 GB of entries.
    std::vector<std::uint8_t> block = {'I', 'I', 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00};
    block.push_back(0xFF);
    block.push_back(0xFF);
    block.push_back(0xFF);
    block.push_back(0xFF);
    const ExifData exif = parse(block);
    EXPECT_FALSE(exif.make.has_value());
}

TEST(Exif, IfdCountLargerThanTheBlockIsRejected) {
    // count = 100, but the block holds no entries at all.
    std::vector<std::uint8_t> block = {'I', 'I', 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00};
    block.push_back(100);
    block.push_back(0);
    const ExifData exif = parse(block);
    EXPECT_FALSE(exif.make.has_value());
}

TEST(Exif, CountTimesTypeSizeBeyondTheCapIsRejected) {
    // One RATIONAL-typed entry claiming 0xFFFFFFFF elements: 8 * that overflows
    // any sane buffer, so the field must be ignored rather than sized.
    std::vector<std::uint8_t> block = {'I', 'I', 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00};

    // Append n zero bytes and return the offset they start at.
    const auto reserve = [&block](std::size_t n) {
        const std::size_t at = block.size();
        block.resize(at + n, 0);
        return at;
    };

    // IFD: count = 1
    const std::size_t ifd = reserve(2);
    put16(block, ifd, 1, true);
    // entry: tag 0x010F (Make), type 5 (RATIONAL), count 0xFFFFFFFF, value 0
    const std::size_t entry = reserve(12);
    put16(block, entry, 0x010F, true);
    put16(block, entry + 2, 5, true);
    put32(block, entry + 4, 0xFFFFFFFFU, true);
    put32(block, entry + 8, 0, true);
    // next-IFD offset
    const std::size_t next = reserve(4);
    put32(block, next, 0, true);

    const ExifData exif = parse(block);
    EXPECT_FALSE(exif.make.has_value());
}

TEST(Exif, ZeroDenominatorIsIgnoredButOtherFieldsSurvive) {
    TiffBuilder builder(true);
    const int ifd0 = builder.add_ifd();
    const int exif = builder.add_ifd();
    builder.ascii(ifd0, 0x010F, "ZeroDen");
    builder.pointer(ifd0, 0x8769, exif);
    builder.rational(exif, 0x920A, 35, 0);  // focal length with denominator 0
    const ExifData exif_data = parse(builder.build());
    // The bad rational is dropped...
    EXPECT_FALSE(exif_data.focal_length_mm.has_value());
    // ...but the walk continued and the earlier field is intact.
    ASSERT_TRUE(exif_data.make.has_value());
    EXPECT_EQ(*exif_data.make, "ZeroDen");
}

TEST(Exif, InlineShortValueIsReadFromTheEntryItself) {
    // A SHORT with count 1 is 2 bytes, so it is stored in the entry's value
    // slot. Reading it from the wrong place is a classic parser bug.
    TiffBuilder builder(true);
    const int ifd0 = builder.add_ifd();
    const int exif = builder.add_ifd();
    builder.pointer(ifd0, 0x8769, exif);
    builder.short_value(exif, 0xA210, 4);  // millimetres, inline
    const ExifData exif_data = parse(builder.build());
    ASSERT_TRUE(exif_data.focal_plane_resolution_unit.has_value());
    EXPECT_EQ(*exif_data.focal_plane_resolution_unit, 4);
}

TEST(Exif, TruncatedEntryTableStopsCleanly) {
    // An IFD whose count claims more entries than the block holds.
    TiffBuilder builder(true);
    const int ifd0 = builder.add_ifd();
    builder.ascii(ifd0, 0x010F, "Trunc");
    std::vector<std::uint8_t> block = builder.build();
    // Rewrite the count to a value whose entries run past the block.
    put16(block, 8, 200, true);
    const ExifData exif = parse(block);
    EXPECT_FALSE(exif.make.has_value());
}

TEST(Exif, GpsCoordinateWithZeroDenominatorIsDropped) {
    TiffBuilder builder(true);
    const int ifd0 = builder.add_ifd();
    const int gps = builder.add_ifd();
    builder.pointer(ifd0, 0x8825, gps);
    builder.ascii(gps, 0x0001, "N");
    // Degrees rational with a zero denominator: the whole coordinate is refused.
    builder.rational(gps, 0x0002, 43, 0);
    builder.ascii(gps, 0x0003, "E");
    builder.coordinate(gps, 0x0004, 79, 20, 0);
    const ExifData exif = parse(builder.build());
    EXPECT_FALSE(exif.gps.has_value());
}

// ---------------------------------------------------------------------------
// Real fixtures
// ---------------------------------------------------------------------------

TEST(Exif, ReadsJpegExif) {
    const ExifData exif = read_exif(fixture("cam_a1.jpg"));
    ASSERT_TRUE(exif.make.has_value());
    EXPECT_EQ(*exif.make, "AlphaOptics");
    ASSERT_TRUE(exif.model.has_value());
    EXPECT_EQ(*exif.model, "AO-100");
    ASSERT_TRUE(exif.focal_length_mm.has_value());
    EXPECT_DOUBLE_EQ(*exif.focal_length_mm, 35.0);
    ASSERT_TRUE(exif.focal_length_35mm.has_value());
    EXPECT_DOUBLE_EQ(*exif.focal_length_35mm, 35.0);
    ASSERT_TRUE(exif.focal_plane_x_resolution.has_value());
    EXPECT_DOUBLE_EQ(*exif.focal_plane_x_resolution, 3000.0);
    ASSERT_TRUE(exif.focal_plane_resolution_unit.has_value());
    EXPECT_EQ(*exif.focal_plane_resolution_unit, 3);  // centimetres
    ASSERT_TRUE(exif.orientation.has_value());
    EXPECT_EQ(*exif.orientation, 1);
    ASSERT_TRUE(exif.body_serial.has_value());
    EXPECT_EQ(*exif.body_serial, "AO100-0001");
    ASSERT_TRUE(exif.pixel_x_dimension.has_value());
    EXPECT_EQ(*exif.pixel_x_dimension, 200);
    ASSERT_TRUE(exif.pixel_y_dimension.has_value());
    EXPECT_EQ(*exif.pixel_y_dimension, 150);
    ASSERT_TRUE(exif.datetime_original.has_value());
    EXPECT_EQ(*exif.datetime_original, "2024:06:15 09:30:00");
}

TEST(Exif, ReadsJpegGpsNorthernEastern) {
    const ExifData exif = read_exif(fixture("cam_a1.jpg"));
    ASSERT_TRUE(exif.gps.has_value());
    EXPECT_NEAR(exif.gps->latitude_deg, 43.5, 1e-6);
    EXPECT_NEAR(exif.gps->longitude_deg, 79.0 + 20.0 / 60.0, 1e-6);
    EXPECT_FALSE(exif.gps->altitude_m.has_value());
}

TEST(Exif, ReadsJpegGpsSouthernWesternBelowSeaLevel) {
    const ExifData exif = read_exif(fixture("cam_a2.jpg"));
    ASSERT_TRUE(exif.gps.has_value());
    EXPECT_NEAR(exif.gps->latitude_deg, -(33.0 + 52.0 / 60.0), 1e-6);
    EXPECT_NEAR(exif.gps->longitude_deg, -(151.0 + 12.0 / 60.0), 1e-6);
    ASSERT_TRUE(exif.gps->altitude_m.has_value());
    EXPECT_DOUBLE_EQ(*exif.gps->altitude_m, -350.0);
}

TEST(Exif, CameraAFixturesShareSensorMetadata) {
    // A later grouping stage keys on this: cam_a1 and cam_a2 are the same
    // camera, cam_b1 is not.
    const ExifData a1 = read_exif(fixture("cam_a1.jpg"));
    const ExifData a2 = read_exif(fixture("cam_a2.jpg"));
    const ExifData b1 = read_exif(fixture("cam_b1.jpg"));

    ASSERT_TRUE(a1.make.has_value() && a2.make.has_value() && b1.make.has_value());
    EXPECT_EQ(*a1.make, *a2.make);
    ASSERT_TRUE(a1.model.has_value() && a2.model.has_value() && b1.model.has_value());
    EXPECT_EQ(*a1.model, *a2.model);
    EXPECT_NE(*a1.model, *b1.model);

    ASSERT_TRUE(a1.focal_length_mm.has_value() && a2.focal_length_mm.has_value());
    EXPECT_DOUBLE_EQ(*a1.focal_length_mm, *a2.focal_length_mm);
    ASSERT_TRUE(a1.focal_plane_x_resolution.has_value() && a2.focal_plane_x_resolution.has_value());
    EXPECT_DOUBLE_EQ(*a1.focal_plane_x_resolution, *a2.focal_plane_x_resolution);
    ASSERT_TRUE(a1.focal_plane_resolution_unit.has_value() &&
                a2.focal_plane_resolution_unit.has_value());
    EXPECT_EQ(*a1.focal_plane_resolution_unit, *a2.focal_plane_resolution_unit);

    // cam_b1 deliberately has no focal-plane resolution and no GPS.
    EXPECT_FALSE(b1.focal_plane_x_resolution.has_value());
    EXPECT_FALSE(b1.gps.has_value());
    // ...and it does carry a 35 mm-equivalent focal length.
    ASSERT_TRUE(b1.focal_length_35mm.has_value());
    EXPECT_DOUBLE_EQ(*b1.focal_length_35mm, 75.0);
}

TEST(Exif, ReadsTiffExif) {
    const ExifData exif = read_exif(fixture("exif_grey.tiff"));
    ASSERT_TRUE(exif.make.has_value());
    EXPECT_EQ(*exif.make, "AlphaOptics");
    ASSERT_TRUE(exif.model.has_value());
    EXPECT_EQ(*exif.model, "AO-T100");
    ASSERT_TRUE(exif.focal_length_mm.has_value());
    EXPECT_DOUBLE_EQ(*exif.focal_length_mm, 24.0);
    ASSERT_TRUE(exif.focal_length_35mm.has_value());
    EXPECT_DOUBLE_EQ(*exif.focal_length_35mm, 36.0);
    ASSERT_TRUE(exif.body_serial.has_value());
    EXPECT_EQ(*exif.body_serial, "AOT100-0001");
    ASSERT_TRUE(exif.pixel_size_um().has_value());
    EXPECT_NEAR(*exif.pixel_size_um(), 10000.0 / 2500.0, 1e-9);
}

TEST(Exif, ReadsPngExifChunk) {
    const ExifData exif = read_exif(fixture("exif_grey.png"));
    ASSERT_TRUE(exif.make.has_value());
    EXPECT_EQ(*exif.make, "AlphaOptics");
    ASSERT_TRUE(exif.model.has_value());
    EXPECT_EQ(*exif.model, "AO-T100");
    ASSERT_TRUE(exif.focal_length_mm.has_value());
    EXPECT_DOUBLE_EQ(*exif.focal_length_mm, 24.0);
}

TEST(Exif, PlainImageWithNoExifYieldsEmpty) {
    // A PNG with no eXIf chunk, and a PNM, both read cleanly and report
    // nothing rather than throwing.
    EXPECT_FALSE(read_exif(fixture("rgb.png")).make.has_value());
    EXPECT_FALSE(read_exif(fixture("rgb8.tiff")).make.has_value());
    EXPECT_FALSE(read_exif(fixture("fixture_a.pgm")).make.has_value());
}

TEST(Exif, MissingFileThrows) {
    EXPECT_THROW(read_exif(fixture("no_such_file.jpg")), Error);
}

TEST(Exif, GarbageJpegApp1DoesNotThrow) {
    // An APP1 segment that claims to be Exif but holds junk must be parsed
    // defensively: partial data, no exception, no crash.
    std::vector<std::uint8_t> bytes = {0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x10};
    const char* header = "Exif\0\0";
    for (int i = 0; i < 6; ++i) {
        bytes.push_back(static_cast<std::uint8_t>(header[i]));
    }
    // A TIFF header whose IFD offset is nonsense, then padding to the declared
    // segment length.
    const std::uint8_t tail[4] = {'I', 'I', 0x2A, 0x00};
    for (std::uint8_t b : tail) {
        bytes.push_back(b);
    }
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("alphashape_exif_garbage_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)) +
         ".jpg");
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }
    const ExifData exif = read_exif(path.string());
    EXPECT_FALSE(exif.make.has_value());
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(Exif, SkipsXmpApp1AndFindsTheExifOne) {
    // XMP is also APP1; the reader must skip it and find the later Exif segment.
    std::vector<std::uint8_t> bytes = {0xFF, 0xD8};

    // APP1 XMP segment: "http://ns.adobe.com/xap/1.0/\0" + one byte payload.
    const std::string xmp_id = "http://ns.adobe.com/xap/1.0/";
    const std::vector<std::uint8_t> xmp_payload = [&] {
        std::vector<std::uint8_t> p(xmp_id.begin(), xmp_id.end());
        p.push_back(0);
        p.push_back('x');
        return p;
    }();
    const std::uint16_t xmp_len = static_cast<std::uint16_t>(xmp_payload.size() + 2);
    bytes.push_back(0xFF);
    bytes.push_back(0xE1);
    bytes.push_back(static_cast<std::uint8_t>((xmp_len >> 8) & 0xFF));
    bytes.push_back(static_cast<std::uint8_t>(xmp_len & 0xFF));
    bytes.insert(bytes.end(), xmp_payload.begin(), xmp_payload.end());

    // A real Exif APP1 built by the assembler.
    TiffBuilder builder(true);
    const int ifd0 = builder.add_ifd();
    builder.ascii(ifd0, 0x010F, "AfterXmp");
    const std::vector<std::uint8_t> tiff = builder.build();
    const std::uint16_t exif_len = static_cast<std::uint16_t>(tiff.size() + 6 + 2);
    bytes.push_back(0xFF);
    bytes.push_back(0xE1);
    bytes.push_back(static_cast<std::uint8_t>((exif_len >> 8) & 0xFF));
    bytes.push_back(static_cast<std::uint8_t>(exif_len & 0xFF));
    const char* exif_header = "Exif\0\0";
    for (int i = 0; i < 6; ++i) {
        bytes.push_back(static_cast<std::uint8_t>(exif_header[i]));
    }
    bytes.insert(bytes.end(), tiff.begin(), tiff.end());

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("alphashape_exif_xmp_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".jpg");
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }
    const ExifData exif = read_exif(path.string());
    ASSERT_TRUE(exif.make.has_value());
    EXPECT_EQ(*exif.make, "AfterXmp");
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

}  // namespace
}  // namespace alphashape
