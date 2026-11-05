// WP-13.15b: STMap generation and TIFF export tests.
//
// The map is a backward resampling map: sampling it at an output pixel gives
// the source pixel, normalised with a bottom-left origin. The tests invert that
// round trip through the camera model and read the writer's tags back.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>
#include <tiffio.h>

#include "base/platform/temp.hpp"
#include "common/algorithm/calibration/stmap.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

constexpr double kWidth = 640.0;
constexpr double kHeight = 480.0;

Calibration distorted_calibration() {
    Calibration calibration =
        Calibration::initial(CameraType::Frame, 640, 480, 500.0);
    calibration.cx = 3.0;
    calibration.cy = -2.0;
    calibration.k1 = -0.12;
    calibration.k2 = 0.03;
    calibration.p1 = 1e-3;
    calibration.p2 = -5e-4;
    return calibration;
}

Eigen::Vector2d source_of(const STMap& map, int col, int row) {
    const std::size_t index =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(map.width) +
        static_cast<std::size_t>(col);
    return Eigen::Vector2d(
        static_cast<double>(map.s[index]) * kWidth,
        (1.0 - static_cast<double>(map.t[index])) * kHeight);
}

}  // namespace

TEST(Stmap, UndistortsGrid) {
    const Calibration calibration = distorted_calibration();
    const STMap map = compute_stmap(calibration);
    ASSERT_EQ(map.width, 640);
    ASSERT_EQ(map.height, 480);
    const double principal_x = kWidth / 2.0 + calibration.cx;
    const double principal_y = kHeight / 2.0 + calibration.cy;
    for (int j = 0; j < 7; ++j) {
        const int row = (j * 479) / 6;
        for (int i = 0; i < 9; ++i) {
            const int col = (i * 639) / 8;
            const Eigen::Vector2d source = source_of(map, col, row);
            ASSERT_TRUE(std::isfinite(source.x()));
            ASSERT_TRUE(std::isfinite(source.y()));
            const Eigen::Vector3d ray = calibration.unproject(source);
            const double u = principal_x + calibration.f * ray.x() / ray.z();
            const double v = principal_y + calibration.f * ray.y() / ray.z();
            EXPECT_NEAR(u, static_cast<double>(col) + 0.5, 0.05)
                << "pixel " << col << "," << row;
            EXPECT_NEAR(v, static_cast<double>(row) + 0.5, 0.05)
                << "pixel " << col << "," << row;
        }
    }
}

TEST(Stmap, IdentityForZeroDistortion) {
    const Calibration calibration =
        Calibration::initial(CameraType::Frame, 640, 480, 500.0);
    const STMap map = compute_stmap(calibration);
    ASSERT_EQ(map.width, 640);
    ASSERT_EQ(map.height, 480);
    for (int row = 0; row < 480; row += 53) {
        for (int col = 0; col < 640; col += 71) {
            const Eigen::Vector2d source = source_of(map, col, row);
            EXPECT_NEAR(source.x(), static_cast<double>(col) + 0.5, 1e-4)
                << "pixel " << col << "," << row;
            EXPECT_NEAR(source.y(), static_cast<double>(row) + 0.5, 1e-4)
                << "pixel " << col << "," << row;
        }
    }
}

TEST(Stmap, IgnoresAffinityAndSkewInIdealPinhole) {
    // Powers of two keep every map value exactly representable in float, so
    // the 1e-9 bound checks the model rather than storage rounding.
    Calibration calibration =
        Calibration::initial(CameraType::Frame, 512, 512, 512.0);
    calibration.b1 = 256.0;
    calibration.b2 = 192.0;
    const STMap map = compute_stmap(calibration);
    ASSERT_EQ(map.width, 512);
    ASSERT_EQ(map.height, 512);
    double largest_shift = 0.0;
    for (int row = 0; row < 512; row += 7) {
        for (int col = 0; col < 512; col += 5) {
            const double u_out = static_cast<double>(col) + 0.5;
            const double v_out = static_cast<double>(row) + 0.5;
            // The ray uses the focal length alone; affinity and skew must
            // enter only through the projection, i.e. the source coordinate.
            const double x = (u_out - 256.0) / 512.0;
            const double y = (v_out - 256.0) / 512.0;
            const Eigen::Vector2d expected =
                calibration.project(Eigen::Vector3d(x, y, 1.0));
            const std::size_t index =
                static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(map.width) +
                static_cast<std::size_t>(col);
            const double source_u = static_cast<double>(map.s[index]) * 512.0;
            const double source_v =
                (1.0 - static_cast<double>(map.t[index])) * 512.0;
            EXPECT_NEAR(source_u, expected.x(), 1e-9)
                << "pixel " << col << "," << row;
            EXPECT_NEAR(source_v, expected.y(), 1e-9)
                << "pixel " << col << "," << row;
            largest_shift = std::max(
                largest_shift,
                std::hypot(expected.x() - u_out, expected.y() - v_out));
        }
    }
    // Non-zero b1/b2 must actually move a source coordinate; without this the
    // test would also pass for a model that silently dropped them.
    EXPECT_GT(largest_shift, 1.0);
}

TEST(Stmap, ResampledSizeKeepsNormalisedCoordinates) {
    const Calibration calibration =
        Calibration::initial(CameraType::Frame, 640, 480, 500.0);
    const STMap map = compute_stmap(calibration, 320, 240);
    ASSERT_EQ(map.width, 320);
    ASSERT_EQ(map.height, 240);
    ASSERT_FALSE(map.s.empty());
    EXPECT_NEAR(static_cast<double>(map.s[0]), 1.0 / 640.0, 1e-6);
    std::printf("[stmap resampled] s(0,0) %.9f expected %.9f\n",
                static_cast<double>(map.s[0]), 1.0 / 640.0);
}

TEST(Stmap, WritesTwoChannelFloatTiff) {
    STMap map;
    map.width = 4;
    map.height = 3;
    map.s.resize(12U);
    map.t.resize(12U);
    for (int i = 0; i < 12; ++i) {
        const std::size_t index = static_cast<std::size_t>(i);
        map.s[index] = static_cast<float>(i) * 0.125F;
        map.t[index] = -0.5F + static_cast<float>(i) * 0.25F;
    }
    const std::string path =
        platform::unique_temp_path("alphashape_stmap_", ".tif");
    write_stmap_tiff(path, map);

    TIFF* tif = TIFFOpen(path.c_str(), "r");
    ASSERT_NE(tif, nullptr);
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint16_t samples = 0;
    std::uint16_t bits = 0;
    std::uint16_t format = 0;
    ASSERT_EQ(TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width), 1);
    ASSERT_EQ(TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height), 1);
    ASSERT_EQ(TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples), 1);
    ASSERT_EQ(TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits), 1);
    ASSERT_EQ(TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &format), 1);
    std::vector<float> row(static_cast<std::size_t>(map.width) * 2U);
    ASSERT_GE(TIFFReadScanline(tif, row.data(), 0, 0), 0);
    TIFFClose(tif);
    std::error_code error;
    std::filesystem::remove(path, error);

    std::printf("[stmap tiff] %ux%u samples %u bits %u format %u "
                "s0 %.6f t0 %.6f\n",
                width, height, static_cast<unsigned int>(samples),
                static_cast<unsigned int>(bits),
                static_cast<unsigned int>(format),
                static_cast<double>(row[0]), static_cast<double>(row[1]));
    EXPECT_EQ(width, 4U);
    EXPECT_EQ(height, 3U);
    EXPECT_EQ(samples, 2U);
    EXPECT_EQ(bits, 32U);
    EXPECT_EQ(format, SAMPLEFORMAT_IEEEFP);
    EXPECT_EQ(row[0], map.s[0]);
    EXPECT_EQ(row[1], map.t[0]);
}

}  // namespace alphashape
