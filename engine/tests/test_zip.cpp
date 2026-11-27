// Zip64 (PKWARE APPNOTE 4.5.3) in the project container's zip layer.
//
// A project whose depth maps push an entry or the archive past 4 GiB could not
// be saved at all: the writer refused any size, offset or entry count that did
// not fit the classic 32/16-bit fields. These tests cover the Zip64 structures
// the writer now emits when a field overflows, that a small archive is still
// byte-for-byte the classic layout, and that the reader walks both. None of them
// allocates a 4 GiB buffer: the writer's `force_zip64` option exercises the
// Zip64 paths on a small archive, and the entry-count path is reached with
// 70000 empty entries.

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/io/zip.hpp"
#include "base/utils/error.hpp"

namespace alphashape {
namespace {

constexpr std::uint32_t kLocalHeaderSig = 0x04034b50;
constexpr std::uint32_t kCentralHeaderSig = 0x02014b50;
constexpr std::uint32_t kEndOfCentralSig = 0x06054b50;
constexpr std::uint32_t kZip64EndOfCentralSig = 0x06064b50;
constexpr std::uint32_t kZip64LocatorSig = 0x07064b50;

std::uint32_t read32(const std::vector<std::uint8_t>& bytes, std::size_t at) {
    std::uint32_t v = 0;
    for (unsigned i = 0; i < 4; ++i) {
        v |= static_cast<std::uint32_t>(bytes[at + i]) << (8U * i);
    }
    return v;
}

std::uint16_t read16(const std::vector<std::uint8_t>& bytes, std::size_t at) {
    return static_cast<std::uint16_t>(static_cast<unsigned>(bytes[at]) |
                                      (static_cast<unsigned>(bytes[at + 1]) << 8U));
}

std::uint64_t read64(const std::vector<std::uint8_t>& bytes, std::size_t at) {
    std::uint64_t v = 0;
    for (unsigned i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(bytes[at + i]) << (8U * i);
    }
    return v;
}

/// Offset of the first occurrence of a 4-byte signature, or npos.
std::size_t find_sig(const std::vector<std::uint8_t>& bytes, std::uint32_t sig) {
    for (std::size_t i = 0; i + 4 <= bytes.size(); ++i) {
        if (read32(bytes, i) == sig) {
            return i;
        }
    }
    return std::string::npos;
}

/// Offset of the classic end-of-central-directory record (the last one).
std::size_t find_eocd(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() < 4) {
        return std::string::npos;
    }
    for (std::size_t i = bytes.size() - 4 + 1; i-- > 0;) {
        if (read32(bytes, i) == kEndOfCentralSig) {
            return i;
        }
    }
    return std::string::npos;
}

// -- classic-layout golden bytes -------------------------------------------
//
// A hand-built classic archive for one stored entry, independent of the writer
// under test. If the writer ever emits a Zip64 structure for an archive that
// fits the classic fields, the comparison fails.

void put16(std::vector<std::uint8_t>& out, std::uint16_t v) {
    out.push_back(static_cast<std::uint8_t>(v & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((v >> 8U) & 0xFFU));
}
void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((v >> shift) & 0xFFU));
    }
}
void put64(std::vector<std::uint8_t>& out, std::uint64_t v) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((v >> shift) & 0xFFU));
    }
}

std::vector<std::uint8_t> classic_archive(const std::string& name,
                                          const std::vector<std::uint8_t>& data,
                                          std::uint32_t crc) {
    constexpr std::uint16_t kVersion = 20;
    constexpr std::uint16_t kDosDate = 0x21;
    std::vector<std::uint8_t> out;
    const auto name_len = static_cast<std::uint16_t>(name.size());
    const auto size = static_cast<std::uint32_t>(data.size());

    put32(out, kLocalHeaderSig);
    put16(out, kVersion);  // version needed
    put16(out, 0);         // flags
    put16(out, 0);         // method: store
    put16(out, 0);         // time
    put16(out, kDosDate);  // date
    put32(out, crc);
    put32(out, size);
    put32(out, size);
    put16(out, name_len);
    put16(out, 0);  // extra
    out.insert(out.end(), name.begin(), name.end());
    out.insert(out.end(), data.begin(), data.end());

    std::vector<std::uint8_t> central;
    put32(central, kCentralHeaderSig);
    put16(central, kVersion);  // made by
    put16(central, kVersion);  // needed
    put16(central, 0);
    put16(central, 0);         // method
    put16(central, 0);         // time
    put16(central, kDosDate);  // date
    put32(central, crc);
    put32(central, size);
    put32(central, size);
    put16(central, name_len);
    put16(central, 0);  // extra
    put16(central, 0);  // comment
    put16(central, 0);  // disk
    put16(central, 0);  // internal attrs
    put32(central, 0);  // external attrs
    put32(central, 0);  // local offset
    central.insert(central.end(), name.begin(), name.end());

    const auto central_offset = static_cast<std::uint32_t>(out.size());
    out.insert(out.end(), central.begin(), central.end());

    put32(out, kEndOfCentralSig);
    put16(out, 0);
    put16(out, 0);
    put16(out, 1);  // entries on this disk
    put16(out, 1);  // entries total
    put32(out, static_cast<std::uint32_t>(central.size()));
    put32(out, central_offset);
    put16(out, 0);  // comment length
    return out;
}

// -- tests ------------------------------------------------------------------

/// A small archive that fits the classic fields is byte-for-byte the classic
/// layout: no Zip64 end records, version needed 20, no 0x0001 extra field.
TEST(Zip64, SmallArchiveStaysClassic) {
    ZipArchive archive;
    archive.add("a", std::string("hello"));
    const std::vector<std::uint8_t> bytes = archive.serialise();

    // CRC32("hello") = 0x3610a686; the entry is stored because deflate cannot
    // shrink five bytes.
    const std::vector<std::uint8_t> expected =
        classic_archive("a", {'h', 'e', 'l', 'l', 'o'}, 0x3610a686U);
    EXPECT_EQ(bytes, expected);

    EXPECT_EQ(find_sig(bytes, kZip64EndOfCentralSig), std::string::npos);
    EXPECT_EQ(find_sig(bytes, kZip64LocatorSig), std::string::npos);

    const std::size_t central = find_sig(bytes, kCentralHeaderSig);
    ASSERT_NE(central, std::string::npos);
    EXPECT_EQ(read16(bytes, central + 6), 20U);  // version needed
    EXPECT_EQ(read16(bytes, central + 30), 0U);  // extra field length

    const std::size_t eocd = find_eocd(bytes);
    ASSERT_NE(eocd, std::string::npos);
    EXPECT_EQ(read16(bytes, eocd + 10), 1U);  // entries, not the 0xFFFF sentinel
}

/// Forced Zip64 round-trips: several entries, one stored and one deflated, read
/// back with identical bytes.
TEST(Zip64, ForcedZip64RoundTrips) {
    ZipArchive archive;
    archive.add("stored.bin", std::vector<std::uint8_t>{1, 2, 3, 4, 5});
    archive.add("deflated.txt", std::string(4096, 'z'));  // compresses well
    archive.add("empty", std::vector<std::uint8_t>{});

    ZipWriteOptions options;
    options.force_zip64 = true;
    const std::vector<std::uint8_t> bytes = archive.serialise(options);

    const ZipArchive parsed = ZipArchive::parse(bytes);
    ASSERT_EQ(parsed.entries().size(), 3U);
    EXPECT_EQ(parsed.get("stored.bin"), (std::vector<std::uint8_t>{1, 2, 3, 4, 5}));
    EXPECT_EQ(parsed.get_text("deflated.txt"), std::string(4096, 'z'));
    EXPECT_TRUE(parsed.get("empty").empty());
}

/// A forced-Zip64 archive carries the Zip64 end records, the classic EOCD
/// sentinels and a 0x0001 extra field on every entry, so an independent reader
/// (Python's zipfile, Info-ZIP) is directed to the 64-bit values.
TEST(Zip64, ForcedZip64HasTheExpectedStructure) {
    ZipArchive archive;
    archive.add("a", std::string("hello"));
    archive.add("b", std::string(2048, 'q'));

    ZipWriteOptions options;
    options.force_zip64 = true;
    const std::vector<std::uint8_t> bytes = archive.serialise(options);

    // Both Zip64 end records are present, in order: the EOCD record then the
    // locator, immediately before the classic EOCD.
    const std::size_t zip64_eocd = find_sig(bytes, kZip64EndOfCentralSig);
    const std::size_t locator = find_sig(bytes, kZip64LocatorSig);
    const std::size_t eocd = find_eocd(bytes);
    ASSERT_NE(zip64_eocd, std::string::npos);
    ASSERT_NE(locator, std::string::npos);
    ASSERT_NE(eocd, std::string::npos);
    EXPECT_LT(zip64_eocd, locator);
    EXPECT_LT(locator, eocd);
    EXPECT_EQ(locator + 20, eocd);  // the locator sits right before the EOCD

    // The Zip64 EOCD record: size of the remainder is 44, and it holds the real
    // entry count and central-directory offset.
    EXPECT_EQ(read64(bytes, zip64_eocd + 4), 44U);
    EXPECT_EQ(read64(bytes, zip64_eocd + 32), 2U);  // entries total
    // The locator points back at the Zip64 EOCD record.
    EXPECT_EQ(read64(bytes, locator + 8), static_cast<std::uint64_t>(zip64_eocd));

    // The classic EOCD carries the sentinels.
    EXPECT_EQ(read16(bytes, eocd + 10), 0xFFFFU);
    EXPECT_EQ(read32(bytes, eocd + 12), 0xFFFFFFFFU);
    EXPECT_EQ(read32(bytes, eocd + 16), 0xFFFFFFFFU);

    // Every central-directory entry has version needed 45 and a 0x0001 extra
    // field whose length covers both sizes and the local offset (24 bytes).
    std::size_t central = find_sig(bytes, kCentralHeaderSig);
    ASSERT_NE(central, std::string::npos);
    for (int i = 0; i < 2; ++i) {
        ASSERT_EQ(read32(bytes, central), kCentralHeaderSig);
        EXPECT_EQ(read16(bytes, central + 6), 45U);  // version needed
        const std::size_t name_len = read16(bytes, central + 28);
        const std::size_t extra_len = read16(bytes, central + 30);
        ASSERT_GE(extra_len, 4U);
        const std::size_t extra = central + 46 + name_len;
        EXPECT_EQ(read16(bytes, extra), 0x0001U);  // Zip64 extra header ID
        EXPECT_EQ(read16(bytes, extra + 2), 24U);  // two sizes + offset
        central = extra + extra_len;
    }
}

/// Build a valid Zip64 archive of `count` empty stored entries directly, so the
/// >65535-entry test is O(n) rather than the O(n^2) of ZipArchive::add's
/// duplicate scan. The classic EOCD carries the sentinels and the Zip64 end
/// records hold the real count and central-directory offset.
std::vector<std::uint8_t> build_many_entry_archive(int count) {
    std::vector<std::uint8_t> out;
    std::vector<std::uint8_t> central;
    for (int i = 0; i < count; ++i) {
        const std::string name = "e" + std::to_string(i);
        const auto name_len = static_cast<std::uint16_t>(name.size());
        const auto offset = static_cast<std::uint32_t>(out.size());

        put32(out, kLocalHeaderSig);
        put16(out, 20);  // version needed
        put16(out, 0);   // flags
        put16(out, 0);   // method: store
        put16(out, 0);   // time
        put16(out, 0x21);  // date
        put32(out, 0);   // crc of empty
        put32(out, 0);   // compressed size
        put32(out, 0);   // uncompressed size
        put16(out, name_len);
        put16(out, 0);  // extra
        out.insert(out.end(), name.begin(), name.end());

        put32(central, kCentralHeaderSig);
        put16(central, 20);  // made by
        put16(central, 20);  // needed
        put16(central, 0);
        put16(central, 0);   // method
        put16(central, 0);   // time
        put16(central, 0x21);  // date
        put32(central, 0);   // crc
        put32(central, 0);   // compressed size
        put32(central, 0);   // uncompressed size
        put16(central, name_len);
        put16(central, 0);  // extra
        put16(central, 0);  // comment
        put16(central, 0);  // disk
        put16(central, 0);  // internal attrs
        put32(central, 0);  // external attrs
        put32(central, offset);
        central.insert(central.end(), name.begin(), name.end());
    }

    const auto central_offset = static_cast<std::uint64_t>(out.size());
    out.insert(out.end(), central.begin(), central.end());
    const auto central_size = static_cast<std::uint64_t>(central.size());
    const auto total = static_cast<std::uint64_t>(count);

    const std::uint64_t zip64_eocd_offset = out.size();
    put32(out, kZip64EndOfCentralSig);
    put64(out, 44U);
    put16(out, 45);  // version made by
    put16(out, 45);  // version needed
    put32(out, 0);   // this disk
    put32(out, 0);   // disk with the central directory
    put64(out, total);
    put64(out, total);
    put64(out, central_size);
    put64(out, central_offset);

    put32(out, kZip64LocatorSig);
    put32(out, 0);
    put64(out, zip64_eocd_offset);
    put32(out, 1);

    put32(out, kEndOfCentralSig);
    put16(out, 0);
    put16(out, 0);
    put16(out, 0xFFFFU);
    put16(out, 0xFFFFU);
    put32(out, 0xFFFFFFFFU);
    put32(out, 0xFFFFFFFFU);
    put16(out, 0);
    return out;
}

/// More than 65535 entries overflow the classic count field. The reader walks
/// the Zip64 end records, and re-serialising the parsed archive (the writer's
/// own count-overflow path) round-trips. 70000 empty stored entries are a few
/// megabytes and never approach 4 GiB.
TEST(Zip64, MoreThan65535EntriesRoundTrips) {
    constexpr int kCount = 70000;
    const std::vector<std::uint8_t> bytes = build_many_entry_archive(kCount);

    const ZipArchive parsed = ZipArchive::parse(bytes);
    ASSERT_EQ(parsed.entries().size(), static_cast<std::size_t>(kCount));
    EXPECT_TRUE(parsed.contains("e0"));
    EXPECT_TRUE(parsed.contains("e69999"));
    EXPECT_TRUE(parsed.get("e69999").empty());

    // Re-serialise: the writer sees the count overflow and emits the Zip64 end
    // records itself, and the result parses back to the same entries.
    const std::vector<std::uint8_t> resaved = parsed.serialise();
    EXPECT_NE(find_sig(resaved, kZip64EndOfCentralSig), std::string::npos);
    const std::size_t eocd = find_eocd(resaved);
    ASSERT_NE(eocd, std::string::npos);
    EXPECT_EQ(read16(resaved, eocd + 10), 0xFFFFU);

    const ZipArchive again = ZipArchive::parse(resaved);
    ASSERT_EQ(again.entries().size(), static_cast<std::size_t>(kCount));
    EXPECT_TRUE(again.contains("e69999"));
}

/// A Zip64 archive whose central-directory extra field is truncated is rejected
/// rather than read past its end.
TEST(Zip64, TruncatedZip64ExtraIsRejected) {
    ZipArchive archive;
    archive.add("a", std::string("hello"));
    ZipWriteOptions options;
    options.force_zip64 = true;
    std::vector<std::uint8_t> bytes = archive.serialise(options);

    // Shrink the central entry's Zip64 extra length from 24 to 8, so it no
    // longer holds both sizes and the offset.
    const std::size_t central = find_sig(bytes, kCentralHeaderSig);
    ASSERT_NE(central, std::string::npos);
    const std::size_t name_len = read16(bytes, central + 28);
    const std::size_t extra = central + 46 + name_len;
    ASSERT_EQ(read16(bytes, extra), 0x0001U);
    bytes[extra + 2] = 8;
    bytes[extra + 3] = 0;
    EXPECT_THROW(ZipArchive::parse(bytes), ProjectIOError);
}

}  // namespace
}  // namespace alphashape
