// Box-downsample operator: geometry, averages, rounding, and validation.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "image/image.hpp"
#include "image/operators/downsample.hpp"

namespace alphashape {
namespace {

// -- FloatImage --------------------------------------------------------------

TEST(Downsample, FloatGeometryUsesIntegerDivision) {
    // 10x7 with factor 4: output is 2x1, dropping the trailing strip.
    const FloatImage src(10, 7, 1.0F);
    const FloatImage out = downsample_box(src, 4);
    EXPECT_EQ(out.width(), 2);   // 10 / 4
    EXPECT_EQ(out.height(), 1);  // 7 / 4
}

TEST(Downsample, FloatAveragesEachBlock) {
    FloatImage src(4, 2, 0.0F);
    // Two 2x2 blocks across (x=0..1 and x=2..3) at each row.
    src.at(0, 0) = 1.0F;
    src.at(1, 0) = 3.0F;
    src.at(0, 1) = 5.0F;
    src.at(1, 1) = 7.0F;  // block 0 mean = 4
    src.at(2, 0) = 2.0F;
    src.at(3, 0) = 4.0F;
    src.at(2, 1) = 6.0F;
    src.at(3, 1) = 8.0F;  // block 1 mean = 5

    const FloatImage out = downsample_box(src, 2);
    ASSERT_EQ(out.width(), 2);
    ASSERT_EQ(out.height(), 1);
    EXPECT_FLOAT_EQ(out.at(0, 0), 4.0F);
    EXPECT_FLOAT_EQ(out.at(1, 0), 5.0F);
}

TEST(Downsample, FloatFactorOneIsACopy) {
    FloatImage src(3, 2, 0.0F);
    src.at(2, 1) = 42.0F;
    const FloatImage out = downsample_box(src, 1);
    EXPECT_EQ(out.width(), 3);
    EXPECT_EQ(out.height(), 2);
    EXPECT_FLOAT_EQ(out.at(2, 1), 42.0F);
    // A copy: writing the output must not touch the source.
    FloatImage copy = out;
    copy.at(0, 0) = 7.0F;
    EXPECT_FLOAT_EQ(src.at(0, 0), 0.0F);
}

TEST(Downsample, FloatNonPowerOfTwoThrows) {
    const FloatImage src(8, 8, 0.0F);
    EXPECT_THROW(downsample_box(src, 3), ValueError);
    EXPECT_THROW(downsample_box(src, 0), ValueError);
    EXPECT_THROW(downsample_box(src, -4), ValueError);
}

// -- Image -------------------------------------------------------------------

TEST(Downsample, ImagePerChannelRoundsToNearest) {
    Image src(2, 2, 3);
    // Channel 0 block mean = (0 + 0 + 1 + 2)/4 = 0.75 -> 1 (rounded).
    // Channel 1 block mean = (0 + 1 + 1 + 1)/4 = 0.75 -> 1 (rounded).
    // Channel 2 block mean = (0 + 0 + 0 + 0)/4 = 0 -> 0.
    src.at(0, 0, 0) = 0; src.at(1, 0, 0) = 0;
    src.at(0, 1, 0) = 1; src.at(1, 1, 0) = 2;
    src.at(0, 0, 1) = 0; src.at(1, 0, 1) = 1;
    src.at(0, 1, 1) = 1; src.at(1, 1, 1) = 1;
    src.at(0, 0, 2) = 0; src.at(1, 0, 2) = 0;
    src.at(0, 1, 2) = 0; src.at(1, 1, 2) = 0;

    const Image out = downsample_box(src, 2);
    ASSERT_EQ(out.width(), 1);
    ASSERT_EQ(out.height(), 1);
    ASSERT_EQ(out.channels(), 3);
    EXPECT_EQ(out.at(0, 0, 0), 1);
    EXPECT_EQ(out.at(0, 0, 1), 1);
    EXPECT_EQ(out.at(0, 0, 2), 0);
}

TEST(Downsample, ImageFactorOneIsACopy) {
    Image src(2, 2, 1);
    src.at(1, 1, 0) = 200;
    const Image out = downsample_box(src, 1);
    EXPECT_EQ(out.width(), 2);
    EXPECT_EQ(out.at(1, 1, 0), 200);
}

TEST(Downsample, ImageGeometryAndValidation) {
    const Image src(9, 5, 3);
    const Image out = downsample_box(src, 2);
    EXPECT_EQ(out.width(), 4);   // 9 / 2
    EXPECT_EQ(out.height(), 2);  // 5 / 2
    EXPECT_THROW(downsample_box(src, 6), ValueError);
}

}  // namespace
}  // namespace alphashape
