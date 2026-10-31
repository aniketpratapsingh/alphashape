// WP-12.10: GeoTIFF GeoKeys carrying a full CRS.
//
// The writer stores an authority-coded CRS as its EPSG code plus a citation
// name, and a custom (uncoded) CRS as 32767 plus its full WKT2 definition in
// GeoAsciiParams. The reader reverses that priority: EPSG key, then WKT from
// the citation, then the local frame. These tests write with the real writer,
// and additionally build GDAL-convention files by hand with libtiff so the
// reader is checked against bytes it did not produce.
#include "model/raster/geotiff.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <tiffio.h>

#include "base/io/geotiff_tags.hpp"
#include "base/platform/temp.hpp"
#include "model/crs/coordinate_system.hpp"

namespace alphashape {
namespace {

using crs::CoordinateSystem;

std::string temp_path(const std::string& name) {
    static int counter = 0;
    return platform::unique_temp_path(
        "alphashape_geotiff_crs_test_" + std::to_string(++counter) + "_", "_" + name);
}

GeoRaster<std::uint8_t> make_raster(int w, int h) {
    GeoRaster<std::uint8_t> r;
    r.grid.width = w;
    r.grid.height = h;
    r.grid.origin_x = 500000.0;
    r.grid.origin_y = 4000000.0;
    r.grid.pixel_size_x = 1.0;
    r.grid.pixel_size_y = 1.0;
    r.channels = 1;
    r.data.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 128);
    return r;
}

/// A Transverse Mercator with a deliberately unusual central meridian: no EPSG
/// code exists for it, so it exercises the user-defined WKT path.
CoordinateSystem custom_crs() {
    return CoordinateSystem::from_proj(
        "+proj=tmerc +lat_0=0 +lon_0=12.3456 +k=0.9999 +x_0=654321 +y_0=0 "
        "+datum=WGS84 +units=m");
}

/// The two GeoTIFF directories read back verbatim.
struct RawGeoKeys {
    std::vector<std::uint16_t> keys;
    std::string ascii;
};

RawGeoKeys read_raw_geo_keys(const std::string& path) {
    RawGeoKeys out;
    install_geotiff_tags();
    TIFF* tif = TIFFOpen(path.c_str(), "r");
    if (tif == nullptr) {
        return out;
    }
    std::uint16_t count = 0;
    std::uint16_t* keys = nullptr;
    if (TIFFGetField(tif, kTagGeoKeyDirectory, &count, &keys) != 0 && keys != nullptr) {
        out.keys.assign(keys, keys + count);
    }
    char* ascii = nullptr;
    if (TIFFGetField(tif, kTagGeoAsciiParams, &ascii) != 0 && ascii != nullptr) {
        out.ascii = ascii;
    }
    TIFFClose(tif);
    return out;
}

/// Writes a valid stripped 8-bit TIFF carrying exactly the given GeoKeys, the
/// way GDAL writes them, so the reader is tested on bytes it did not produce.
void write_raw_geotiff(const std::string& path, const std::vector<std::uint16_t>& keys,
                       const std::string& ascii) {
    install_geotiff_tags();
    TIFF* tif = TIFFOpen(path.c_str(), "w");
    ASSERT_NE(tif, nullptr) << path;
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, 16u);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, 16u);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    const double scale[3] = {1.0, 1.0, 0.0};
    TIFFSetField(tif, kTagModelPixelScale, 3, scale);
    const double tiepoint[6] = {0.0, 0.0, 0.0, 500000.0, 4000000.0, 0.0};
    TIFFSetField(tif, kTagModelTiepoint, 6, tiepoint);
    TIFFSetField(tif, kTagGeoKeyDirectory, static_cast<std::uint32_t>(keys.size()),
                 keys.data());
    if (!ascii.empty()) {
        TIFFSetField(tif, kTagGeoAsciiParams, ascii.c_str());
    }
    std::vector<std::uint8_t> row(16, 0);
    for (std::uint32_t y = 0; y < 16; ++y) {
        TIFFWriteScanline(tif, row.data(), y, 0);
    }
    TIFFClose(tif);
}

/// Builds a GDAL-convention directory, one entry at a time.
struct KeyBuilder {
    std::vector<std::uint16_t> keys{1, 1, 0, 0};  // NumberOfKeys patched below
    std::string ascii;

    void inline_key(std::uint16_t id, std::uint16_t value) {
        keys.insert(keys.end(), {id, 0, 1, value});
        ++keys[3];
    }

    /// Appends `text` to the pool and points a key at it, count including '|'.
    void ascii_key(std::uint16_t id, const std::string& text) {
        const auto offset = static_cast<std::uint16_t>(ascii.size());
        const auto count = static_cast<std::uint16_t>(text.size() + 1);
        ascii.append(text);
        ascii.push_back('|');
        keys.insert(keys.end(), {id, static_cast<std::uint16_t>(kTagGeoAsciiParams),
                                 count, offset});
        ++keys[3];
    }
};

}  // namespace
}  // namespace alphashape

namespace alphashape {
namespace {

TEST(Crs, GeoTiffWktGeoKeys) {
    const CoordinateSystem custom = custom_crs();
    ASSERT_EQ(custom.epsg(), 0) << custom.to_wkt();
    const std::string path = temp_path("custom_wkt.tif");
    GeoTiffOptions options;
    options.crs = custom;
    write_geotiff(path, make_raster(16, 16), options);

    const GeoTiffInfo info = read_geotiff_info(path);
    EXPECT_EQ(info.epsg, 0);
    ASSERT_TRUE(info.crs.has_value());
    EXPECT_TRUE(info.crs->is_equivalent_to(custom));
    EXPECT_FALSE(info.crs->is_local());

    // The directory must ascend by KeyID, and the citation must point into
    // GeoAsciiParams at the full WKT.
    const RawGeoKeys raw = read_raw_geo_keys(path);
    ASSERT_GE(raw.keys.size(), 4U);
    const std::uint16_t entries = raw.keys[3];
    ASSERT_EQ(raw.keys.size(), 4U + 4U * entries);
    std::uint16_t previous = 0;
    bool saw_type = false;
    bool saw_citation = false;
    for (std::uint16_t i = 0; i < entries; ++i) {
        const std::size_t base = 4 + static_cast<std::size_t>(i) * 4;
        EXPECT_GT(raw.keys[base], previous) << "GeoKeys must ascend by KeyID";
        previous = raw.keys[base];
        if (raw.keys[base] == 3072) {
            saw_type = true;
            EXPECT_EQ(raw.keys[base + 3], 32767);
        }
        if (raw.keys[base] == 3073) {
            saw_citation = true;
            EXPECT_EQ(raw.keys[base + 1], kTagGeoAsciiParams);
            const std::string wkt =
                raw.ascii.substr(raw.keys[base + 3], raw.keys[base + 2]);
            EXPECT_NE(wkt.find("PROJCRS["), std::string::npos);
        }
    }
    EXPECT_TRUE(saw_type);
    EXPECT_TRUE(saw_citation);
    std::remove(path.c_str());
}

/// Every curated built-in CRS survives a write/read with the same EPSG code.
TEST(Crs, GeoTiffEpsgKeys) {
    for (const crs::BuiltinCrs& builtin : crs::builtin_crs_list()) {
        const CoordinateSystem cs = CoordinateSystem::from_epsg(builtin.epsg);
        GeoTiffOptions options;
        options.crs = cs;
        const std::string path =
            temp_path("epsg_" + std::to_string(builtin.epsg) + ".tif");
        write_geotiff(path, make_raster(8, 8), options);

        const GeoTiffInfo info = read_geotiff_info(path);
        EXPECT_EQ(info.epsg, builtin.epsg) << builtin.name;
        ASSERT_TRUE(info.crs.has_value()) << builtin.name;
        EXPECT_EQ(info.crs->epsg(), builtin.epsg) << builtin.name;
        std::remove(path.c_str());
    }
}

/// Files written by GDAL: an EPSG key plus a name citation, and a user-defined
/// key 32767 whose citation is the WKT2 text. Neither is produced by our writer.
TEST(Crs, ReadsGdalGeoTiff) {
    // (a) EPSG key + name citation, with the name not first in the pool.
    KeyBuilder epsg_file;
    epsg_file.inline_key(1024, 1);  // projected
    epsg_file.inline_key(1025, 1);  // PixelIsArea
    epsg_file.inline_key(3072, 32633);
    epsg_file.ascii = "unused|";
    epsg_file.ascii_key(3073, "WGS 84 / UTM zone 33N");
    const std::string epsg_path = temp_path("gdal_epsg.tif");
    write_raw_geotiff(epsg_path, epsg_file.keys, epsg_file.ascii);

    const GeoTiffInfo epsg_info = read_geotiff_info(epsg_path);
    EXPECT_EQ(epsg_info.epsg, 32633);
    ASSERT_TRUE(epsg_info.crs.has_value());
    EXPECT_EQ(epsg_info.crs->epsg(), 32633);
    std::remove(epsg_path.c_str());

    // (b) user-defined model and type, with the full WKT in the citation.
    const CoordinateSystem custom = custom_crs();
    KeyBuilder wkt_file;
    wkt_file.inline_key(1024, 1);
    wkt_file.inline_key(1025, 1);
    wkt_file.inline_key(3072, 32767);
    wkt_file.ascii_key(3073, custom.to_wkt());
    const std::string wkt_path = temp_path("gdal_wkt.tif");
    write_raw_geotiff(wkt_path, wkt_file.keys, wkt_file.ascii);

    const GeoTiffInfo wkt_info = read_geotiff_info(wkt_path);
    EXPECT_EQ(wkt_info.epsg, 0);
    ASSERT_TRUE(wkt_info.crs.has_value());
    EXPECT_TRUE(wkt_info.crs->is_equivalent_to(custom));
    std::remove(wkt_path.c_str());
}

/// No CRS means model type 32767 and no EPSG/type key, and reads back local.
TEST(Crs, GeoTiffNoCrsWritesUserDefined) {
    const std::string path = temp_path("no_crs.tif");
    write_geotiff(path, make_raster(16, 16), GeoTiffOptions{});

    const RawGeoKeys raw = read_raw_geo_keys(path);
    ASSERT_GE(raw.keys.size(), 4U);
    const std::uint16_t entries = raw.keys[3];
    bool saw_model = false;
    bool saw_crs_key = false;
    for (std::uint16_t i = 0; i < entries; ++i) {
        const std::size_t base = 4 + static_cast<std::size_t>(i) * 4;
        switch (raw.keys[base]) {
            case 1024:
                saw_model = true;
                EXPECT_EQ(raw.keys[base + 3], 32767);
                break;
            case 2048:
            case 3072:
                saw_crs_key = true;
                break;
            default:
                break;
        }
    }
    EXPECT_TRUE(saw_model);
    EXPECT_FALSE(saw_crs_key);
    EXPECT_TRUE(raw.ascii.empty());

    const GeoTiffInfo info = read_geotiff_info(path);
    EXPECT_EQ(info.epsg, 0);
    ASSERT_TRUE(info.crs.has_value());
    EXPECT_TRUE(info.crs->is_local());
    std::remove(path.c_str());
}

}  // namespace
}  // namespace alphashape
