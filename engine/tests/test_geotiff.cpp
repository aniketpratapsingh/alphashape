// Tests for the georeferenced raster grid (geodesy-adjacent, header-only) and
// the GeoTIFF writer/reader. The GeoTIFF tags cannot be inspected with libtiff
// directly from here (TIFF is a PRIVATE link dependency of alphashape_model),
// so tag correctness is asserted through read_geotiff_info.
#include "model/raster/geotiff.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "model/raster/geo_raster.hpp"

namespace alphashape {
namespace {

RasterGrid make_grid(int w, int h) {
    RasterGrid g;
    g.width = w;
    g.height = h;
    g.origin_x = 500000.0;
    g.origin_y = 5300000.0;
    g.pixel_size_x = 0.5;
    g.pixel_size_y = 0.25;
    return g;
}

std::string temp_path(const std::string& name) {
    static int counter = 0;
    return platform::unique_temp_path(
        "alphashape_m9_geotiff_test_" + std::to_string(++counter) + "_", "_" + name);
}

/// Reads a whole small file as text.
std::string slurp(const std::string& path) {
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// ---------------------------------------------------------------------------
// GeoRaster / RasterGrid
// ---------------------------------------------------------------------------

TEST(GeoRaster, PixelCenterCornerConvention) {
    const RasterGrid g = make_grid(4, 3);
    const Eigen::Vector2d c00 = g.pixel_center(0, 0);
    EXPECT_DOUBLE_EQ(c00.x(), 500000.0 + 0.25);
    EXPECT_DOUBLE_EQ(c00.y(), 5300000.0 - 0.125);
    const Eigen::Vector2d c23 = g.pixel_center(2, 1);
    EXPECT_DOUBLE_EQ(c23.x(), 500000.0 + 2.5 * 0.5);
    EXPECT_DOUBLE_EQ(c23.y(), 5300000.0 - 1.5 * 0.25);
    EXPECT_TRUE(c00.y() < g.origin_y) << "row 0 must sit below the top edge (north-up)";
}

TEST(GeoRaster, ToPixelIsInverseOfPixelCenter) {
    const RasterGrid g = make_grid(300, 173);
    for (int r = 0; r < g.height; r += 37) {
        for (int c = 0; c < g.width; c += 41) {
            const Eigen::Vector2d world = g.pixel_center(c, r);
            const Eigen::Vector2d pixel = g.to_pixel(world.x(), world.y());
            EXPECT_NEAR(pixel.x(), c + 0.5, 1e-9);
            EXPECT_NEAR(pixel.y(), r + 0.5, 1e-9);
        }
    }
}

TEST(GeoRaster, ContainsPixelBounds) {
    const RasterGrid g = make_grid(4, 3);
    EXPECT_TRUE(g.contains_pixel(0, 0));
    EXPECT_TRUE(g.contains_pixel(3, 2));
    EXPECT_FALSE(g.contains_pixel(-1, 0));
    EXPECT_FALSE(g.contains_pixel(0, -1));
    EXPECT_FALSE(g.contains_pixel(4, 0));
    EXPECT_FALSE(g.contains_pixel(0, 3));
}

TEST(GeoRaster, AtIndexesChannelInterleaved) {
    GeoRaster<std::uint8_t> r;
    r.grid = make_grid(3, 2);
    r.channels = 3;
    r.data.assign(3 * 2 * 3, 0);
    r.at(1, 1, 2) = 200;
    EXPECT_EQ(r.data[(1 * 3 + 1) * 3 + 2], 200);
    EXPECT_FALSE(r.empty());
    EXPECT_TRUE(GeoRaster<float>{}.empty());
}

// ---------------------------------------------------------------------------
// Float GeoTIFF (DEM)
// ---------------------------------------------------------------------------

GeoRaster<float> make_float_raster(int w, int h) {
    GeoRaster<float> r;
    r.grid = make_grid(w, h);
    r.channels = 1;
    r.data.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            r.at(x, y) = static_cast<float>(x) * 0.5F + static_cast<float>(y) * 100.0F - 7.5F;
        }
    }
    return r;
}

TEST(GeoTiff, FloatTiledRoundTripNonMultipleSize) {
    const GeoRaster<float> raster = make_float_raster(300, 173);  // neither is a multiple of 256
    const std::string path = temp_path("float_tiled.tif");
    GeoTiffOptions options;
    options.epsg = 32633;
    options.nodata = -32767.0;
    write_geotiff(path, raster, options);

    const GeoTiffInfo info = read_geotiff_info(path);
    EXPECT_EQ(info.grid, raster.grid);
    EXPECT_EQ(info.channels, 1);
    EXPECT_EQ(info.bits_per_sample, 32);
    EXPECT_TRUE(info.is_float);
    EXPECT_EQ(info.epsg, 32633);
    EXPECT_TRUE(info.tiled);
    ASSERT_TRUE(info.nodata.has_value());
    EXPECT_DOUBLE_EQ(*info.nodata, -32767.0);

    const GeoRaster<float> back = read_geotiff_float(path);
    EXPECT_EQ(back.grid, raster.grid);
    EXPECT_EQ(back.channels, 1);
    EXPECT_EQ(back.data, raster.data);  // exact: float32 bytes survive deflate
    std::remove(path.c_str());
}

TEST(GeoTiff, FloatStrippedRoundTrip) {
    const GeoRaster<float> raster = make_float_raster(64, 48);
    const std::string path = temp_path("float_strip.tif");
    GeoTiffOptions options;
    options.tiled = false;
    write_geotiff(path, raster, options);

    const GeoTiffInfo info = read_geotiff_info(path);
    EXPECT_FALSE(info.tiled);
    EXPECT_EQ(info.epsg, 0);
    EXPECT_FALSE(info.nodata.has_value());

    const GeoRaster<float> back = read_geotiff_float(path);
    EXPECT_EQ(back.data, raster.data);
    std::remove(path.c_str());
}

TEST(GeoTiff, FloatGeographic4326ReportsGeographic) {
    GeoRaster<float> raster = make_float_raster(20, 16);
    raster.grid.origin_x = 11.0;
    raster.grid.origin_y = 48.5;
    raster.grid.pixel_size_x = 1.0 / 3600.0;
    raster.grid.pixel_size_y = 1.0 / 3600.0;
    const std::string path = temp_path("float_geographic.tif");
    GeoTiffOptions options;
    options.epsg = 4326;
    write_geotiff(path, raster, options);

    const GeoTiffInfo info = read_geotiff_info(path);
    EXPECT_EQ(info.epsg, 4326);
    const GeoRaster<float> back = read_geotiff_float(path);
    EXPECT_EQ(back.grid, raster.grid);
    std::remove(path.c_str());
}

TEST(GeoTiff, FloatEpsgZeroIsLocal) {
    const GeoRaster<float> raster = make_float_raster(17, 9);
    const std::string path = temp_path("float_local.tif");
    write_geotiff(path, raster, GeoTiffOptions{});
    const GeoTiffInfo info = read_geotiff_info(path);
    EXPECT_EQ(info.epsg, 0);
    std::remove(path.c_str());
}

// Regression: libtiff keeps custom field definitions per handle, so registering
// the GeoTIFF tags once only covered the first file opened in the process.
TEST(GeoTiff, GeoKeysSurviveSeveralFilesInOneProcess) {
    const int codes[] = {32633, 4326, 32756};
    for (int code : codes) {
        const std::string path = temp_path("multi_" + std::to_string(code) + ".tif");
        GeoTiffOptions options;
        options.epsg = code;
        write_geotiff(path, make_float_raster(8, 8), options);
        EXPECT_EQ(read_geotiff_info(path).epsg, code);
        EXPECT_EQ(read_geotiff_float(path).grid, make_grid(8, 8));
        std::remove(path.c_str());
    }
}

TEST(GeoTiff, FloatTinyImageSmallerThanTile) {
    const GeoRaster<float> raster = make_float_raster(3, 2);
    const std::string path = temp_path("float_tiny.tif");
    write_geotiff(path, raster, GeoTiffOptions{});
    const GeoRaster<float> back = read_geotiff_float(path);
    EXPECT_EQ(back.grid, raster.grid);
    EXPECT_EQ(back.data, raster.data);
    std::remove(path.c_str());
}

TEST(GeoTiff, FloatUncompressedRoundTrip) {
    const GeoRaster<float> raster = make_float_raster(40, 40);
    const std::string path = temp_path("float_raw.tif");
    GeoTiffOptions options;
    options.deflate = false;
    write_geotiff(path, raster, options);
    const GeoRaster<float> back = read_geotiff_float(path);
    EXPECT_EQ(back.data, raster.data);
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Elevation values with special bit patterns survive byte-identically.
// ---------------------------------------------------------------------------

TEST(GeoTiff, FloatPreservesExactBits) {
    GeoRaster<float> raster;
    raster.grid = make_grid(5, 1);
    raster.channels = 1;
    const float values[5] = {0.0F, -0.0F, 1.40129846e-45F, 3.40282347e+38F, -12345.6789F};
    raster.data.assign(values, values + 5);
    const std::string path = temp_path("float_bits.tif");
    write_geotiff(path, raster, GeoTiffOptions{});
    const GeoRaster<float> back = read_geotiff_float(path);
    for (std::size_t i = 0; i < back.data.size(); ++i) {
        EXPECT_EQ(back.data[i], raster.data[i]) << "sample " << i;
    }
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Invalid inputs
// ---------------------------------------------------------------------------

TEST(GeoTiff, FloatRejectsWrongChannelCount) {
    GeoRaster<float> raster = make_float_raster(4, 4);
    raster.channels = 3;
    raster.data.resize(4 * 4 * 3, 0.0F);
    EXPECT_THROW(write_geotiff(temp_path("bad_ch.tif"), raster, GeoTiffOptions{}), ValueError);
}

TEST(GeoTiff, FloatRejectsDataSizeMismatch) {
    GeoRaster<float> raster = make_float_raster(4, 4);
    raster.data.pop_back();
    EXPECT_THROW(write_geotiff(temp_path("bad_size.tif"), raster, GeoTiffOptions{}), ValueError);
}

TEST(GeoTiff, RejectsZeroDimensions) {
    GeoRaster<float> raster = make_float_raster(4, 4);
    raster.grid.width = 0;
    EXPECT_THROW(write_geotiff(temp_path("bad_dim.tif"), raster, GeoTiffOptions{}), ValueError);
}

TEST(GeoTiff, RejectsNonPositivePixelSize) {
    GeoRaster<float> raster = make_float_raster(4, 4);
    raster.grid.pixel_size_y = 0.0;
    EXPECT_THROW(write_geotiff(temp_path("bad_ps.tif"), raster, GeoTiffOptions{}), ValueError);
}

TEST(GeoTiff, RejectsBadTileSize) {
    GeoRaster<float> raster = make_float_raster(4, 4);
    GeoTiffOptions options;
    options.tile_size = 100;  // not a multiple of 16
    EXPECT_THROW(write_geotiff(temp_path("bad_tile.tif"), raster, options), ValueError);
}

TEST(GeoTiff, ReadMissingFileThrows) {
    EXPECT_THROW(read_geotiff_info(platform::unique_temp_path("alphashape_m9_absent_", ".tif")),
                 ProjectIOError);
}

TEST(GeoTiff, ReadNonTiffThrows) {
    const std::string path = temp_path("not_a_tiff.bin");
    {
        std::ofstream out(path, std::ios::binary);
        out << "this is definitely not a TIFF file, not even close";
    }
    EXPECT_THROW(read_geotiff_info(path), ProjectIOError);
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// 8-bit GeoTIFF (imagery)
// ---------------------------------------------------------------------------

GeoRaster<std::uint8_t> make_u8_raster(int w, int h, int channels) {
    GeoRaster<std::uint8_t> r;
    r.grid = make_grid(w, h);
    r.channels = channels;
    r.data.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) *
                  static_cast<std::size_t>(channels));
    for (std::size_t i = 0; i < r.data.size(); ++i) {
        r.data[i] = static_cast<std::uint8_t>((i * 7 + 13) & 0xFF);
    }
    return r;
}

TEST(GeoTiff, U8GreyRoundTripNonMultipleSize) {
    const GeoRaster<std::uint8_t> raster = make_u8_raster(300, 173, 1);
    const std::string path = temp_path("u8_grey.tif");
    GeoTiffOptions options;
    options.epsg = 32633;
    write_geotiff(path, raster, options);

    const GeoTiffInfo info = read_geotiff_info(path);
    EXPECT_EQ(info.grid, raster.grid);
    EXPECT_EQ(info.channels, 1);
    EXPECT_EQ(info.bits_per_sample, 8);
    EXPECT_FALSE(info.is_float);
    EXPECT_TRUE(info.tiled);

    const GeoRaster<std::uint8_t> back = read_geotiff_u8(path);
    EXPECT_EQ(back.grid, raster.grid);
    EXPECT_EQ(back.channels, 1);
    EXPECT_EQ(back.data, raster.data);
    std::remove(path.c_str());
}

TEST(GeoTiff, U8RgbRoundTrip) {
    const GeoRaster<std::uint8_t> raster = make_u8_raster(97, 65, 3);
    const std::string path = temp_path("u8_rgb.tif");
    write_geotiff(path, raster, GeoTiffOptions{});
    const GeoTiffInfo info = read_geotiff_info(path);
    EXPECT_EQ(info.channels, 3);
    const GeoRaster<std::uint8_t> back = read_geotiff_u8(path);
    EXPECT_EQ(back.data, raster.data);
    std::remove(path.c_str());
}

TEST(GeoTiff, U8RgbaRoundTrip) {
    const GeoRaster<std::uint8_t> raster = make_u8_raster(33, 27, 4);
    const std::string path = temp_path("u8_rgba.tif");
    GeoTiffOptions options;
    options.tiled = false;  // exercise the stripped path with 4 channels
    write_geotiff(path, raster, options);
    const GeoTiffInfo info = read_geotiff_info(path);
    EXPECT_EQ(info.channels, 4);
    EXPECT_FALSE(info.tiled);
    const GeoRaster<std::uint8_t> back = read_geotiff_u8(path);
    EXPECT_EQ(back.data, raster.data);
    std::remove(path.c_str());
}

TEST(GeoTiff, U8RejectsFloatRead) {
    const GeoRaster<std::uint8_t> raster = make_u8_raster(16, 16, 1);
    const std::string path = temp_path("u8_as_float.tif");
    write_geotiff(path, raster, GeoTiffOptions{});
    EXPECT_THROW(read_geotiff_float(path), ProjectIOError);
    std::remove(path.c_str());
}

TEST(GeoTiff, FloatRejectsU8Read) {
    const GeoRaster<float> raster = make_float_raster(16, 16);
    const std::string path = temp_path("float_as_u8.tif");
    write_geotiff(path, raster, GeoTiffOptions{});
    EXPECT_THROW(read_geotiff_u8(path), ProjectIOError);
    std::remove(path.c_str());
}

TEST(GeoTiff, U8RejectsBadChannelCount) {
    GeoRaster<std::uint8_t> raster = make_u8_raster(4, 4, 2);
    EXPECT_THROW(write_geotiff(temp_path("u8_bad_ch.tif"), raster, GeoTiffOptions{}), ValueError);
}

TEST(GeoTiff, U8RejectsDataSizeMismatch) {
    GeoRaster<std::uint8_t> raster = make_u8_raster(4, 4, 3);
    raster.data.pop_back();
    EXPECT_THROW(write_geotiff(temp_path("u8_bad_size.tif"), raster, GeoTiffOptions{}), ValueError);
}

// ---------------------------------------------------------------------------
// Corrupt file with a TIFF signature but truncated content
// ---------------------------------------------------------------------------

TEST(GeoTiff, ReadTruncatedTiffThrows) {
    const std::string path = temp_path("truncated.tif");
    {
        std::ofstream out(path, std::ios::binary);
        const char header[8] = {'I', 'I', 42, 0, 8, 0, 0, 0};  // little-endian, IFD at 8 (past EOF)
        out.write(header, sizeof(header));
    }
    EXPECT_THROW(read_geotiff_info(path), ProjectIOError);
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// World file
// ---------------------------------------------------------------------------

TEST(GeoTiff, WorldFileContents) {
    const RasterGrid g = make_grid(10, 10);
    const std::string path = temp_path("raster.tfw");
    write_world_file(path, g);
    // psx, 0, 0, -psy, x of top-left PIXEL CENTRE, y of top-left pixel centre.
    const std::string expected = "0.5\n0\n0\n-0.25\n500000.25\n5299999.875\n";
    EXPECT_EQ(slurp(path), expected);
    std::remove(path.c_str());
}

}  // namespace
}  // namespace alphashape
