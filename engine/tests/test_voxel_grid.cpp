// Voxel grid indexing, geometry and make_grid.

#include <cstddef>

#include <gtest/gtest.h>

#include "base/algorithm/voxel_grid.hpp"
#include "base/utils/error.hpp"

namespace alphashape {
namespace {

TEST(GridSpec, IndexIsXFastest) {
    GridSpec s;
    s.origin = Eigen::Vector3d(1.0, 2.0, 3.0);
    s.voxel = 0.5;
    s.nx = 4;
    s.ny = 3;
    s.nz = 2;

    EXPECT_EQ(s.size(), 24U);
    EXPECT_EQ(s.index(0, 0, 0), 0U);
    EXPECT_EQ(s.index(1, 0, 0), 1U);
    EXPECT_EQ(s.index(0, 1, 0), 4U);
    EXPECT_EQ(s.index(0, 0, 1), 12U);
    EXPECT_EQ(s.index(3, 2, 1), 23U);
}

TEST(GridSpec, CentreIsCornerPlusHalf) {
    GridSpec s;
    s.origin = Eigen::Vector3d(10.0, 20.0, 30.0);
    s.voxel = 2.0;
    s.nx = 5;
    s.ny = 5;
    s.nz = 5;

    // Voxel (0,0,0) is centred half a voxel inside the origin corner.
    EXPECT_EQ(s.centre(0, 0, 0), Eigen::Vector3d(11.0, 21.0, 31.0));
    EXPECT_EQ(s.centre(2, 3, 4), Eigen::Vector3d(10.0 + 2.0 * 2.5, 20.0 + 2.0 * 3.5,
                                                 30.0 + 2.0 * 4.5));
}

TEST(GridSpec, ContainsRejectsOutOfRange) {
    GridSpec s;
    s.nx = 2;
    s.ny = 3;
    s.nz = 4;
    EXPECT_TRUE(s.contains(0, 0, 0));
    EXPECT_TRUE(s.contains(1, 2, 3));
    EXPECT_FALSE(s.contains(-1, 0, 0));
    EXPECT_FALSE(s.contains(2, 0, 0));
    EXPECT_FALSE(s.contains(0, 3, 0));
    EXPECT_FALSE(s.contains(0, 0, 4));
}

TEST(VoxelGridTest, ConstructAndAccess) {
    GridSpec s;
    s.origin = Eigen::Vector3d::Zero();
    s.voxel = 1.0;
    s.nx = 2;
    s.ny = 2;
    s.nz = 2;
    VoxelGrid<float> grid(s, -1.0F);
    ASSERT_EQ(grid.data.size(), 8U);
    EXPECT_FLOAT_EQ(grid.at(0, 0, 0), -1.0F);

    grid.at(1, 0, 0) = 5.0F;
    EXPECT_FLOAT_EQ(grid.data[grid.spec.index(1, 0, 0)], 5.0F);
    EXPECT_FLOAT_EQ(grid.at(1, 0, 0), 5.0F);
}

TEST(MakeGridTest, LongestAxisGetsMaxResolution) {
    // x is the longest axis (extent 10); voxel = 10/100 = 0.1.
    const GridSpec s = make_grid(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(10.0, 5.0, 2.0), 100);
    EXPECT_EQ(s.nx, 100);
    EXPECT_DOUBLE_EQ(s.voxel, 0.1);
    EXPECT_EQ(s.ny, 50);
    EXPECT_EQ(s.nz, 20);
    EXPECT_EQ(s.origin, Eigen::Vector3d(0, 0, 0));
}

TEST(MakeGridTest, ThinAxisClampsToOne) {
    // A flat box: y and z are far below one voxel, so each clamps to 1.
    const GridSpec s = make_grid(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(8.0, 1e-6, 1e-6), 8);
    EXPECT_EQ(s.nx, 8);
    EXPECT_EQ(s.ny, 1);
    EXPECT_EQ(s.nz, 1);
    EXPECT_DOUBLE_EQ(s.voxel, 1.0);
}

TEST(MakeGridTest, NonZeroOriginIsPreserved) {
    const GridSpec s = make_grid(Eigen::Vector3d(-1.0, 2.0, 3.0), Eigen::Vector3d(3.0, 6.0, 11.0), 8);
    EXPECT_EQ(s.origin, Eigen::Vector3d(-1.0, 2.0, 3.0));
    // z is the longest axis (extent 8), so voxel = 1 and its resolution is
    // exactly 8; the two extent-4 axes each get ceil(4/1) = 4 voxels.
    EXPECT_EQ(s.nz, 8);
    EXPECT_EQ(s.nx, 4);
    EXPECT_EQ(s.ny, 4);
    EXPECT_DOUBLE_EQ(s.voxel, 1.0);
}

TEST(MakeGridTest, ThrowsForLowResolution) {
    EXPECT_THROW(make_grid(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 1, 1), 1), ValueError);
}

TEST(MakeGridTest, ThrowsForEmptyOrInvertedBox) {
    EXPECT_THROW(make_grid(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 1, 1), 8), ValueError);
    EXPECT_THROW(make_grid(Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 1), 8), ValueError);
}

}  // namespace
}  // namespace alphashape
