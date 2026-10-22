// test_rasterize.cpp -- z-buffer rasterisation (milestone 8 texturing).
//
// Everything here is analytic: the mesh is a set of planar squares, so the
// expected pixel coverage follows from intersecting each pixel's unit bearing
// with the plane and the expected depth is that bearing's parameter. That keeps
// the tests independent of the rasteriser's own interpolation.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "model/camera/calibration.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/rasterize.hpp"

namespace alphashape {
namespace {

constexpr int kW = 64;
constexpr int kH = 64;

Calibration pinhole(double f = 64.0, double k1 = 0.0) {
    Calibration c = Calibration::initial(CameraType::Frame, kW, kH, f);
    c.k1 = k1;
    return c;
}

/// Two triangles making a fronto-parallel square centred on the optical axis in
/// the plane z = z0, wound so the geometric normal points back at the camera.
Mesh axis_square(double half, double z0) {
    Mesh m;
    m.vertices = {Eigen::Vector3d(-half, -half, z0), Eigen::Vector3d(half, -half, z0),
                  Eigen::Vector3d(half, half, z0), Eigen::Vector3d(-half, half, z0)};
    m.faces = {Eigen::Vector3i(0, 2, 1), Eigen::Vector3i(0, 3, 2)};
    return m;
}

/// Append `src` (shifted on z) to `dst`, returning the index its vertices got.
/// Keeps the face ids contiguous so a caller can name them.
void append_square(Mesh& dst, const Mesh& src) {
    const int base = static_cast<int>(dst.vertices.size());
    for (const Eigen::Vector3d& v : src.vertices) {
        dst.vertices.push_back(v);
    }
    for (const Eigen::Vector3i& f : src.faces) {
        dst.faces.push_back(Eigen::Vector3i(f[0] + base, f[1] + base, f[2] + base));
    }
}

/// Depth of the plane z = z0 along the unit bearing through pixel centre (i, j).
double plane_depth(const Calibration& c, int i, int j, double z0) {
    const Eigen::Vector3d bearing = c.unproject(Eigen::Vector2d(i + 0.5, j + 0.5));
    return z0 / bearing.z();
}

/// Number of pixels whose stored face id is in `ids`.
std::size_t count_faces(const DepthBuffer& b, std::initializer_list<int> ids) {
    std::size_t n = 0;
    for (std::int32_t f : b.face) {
        for (int id : ids) {
            if (f == id) {
                ++n;
            }
        }
    }
    return n;
}

/// Analytic coverage: pixels whose bearing hits the interior of the square.
std::size_t expected_square_pixels(const Calibration& c, double half, double z0) {
    std::size_t n = 0;
    for (int j = 0; j < kH; ++j) {
        for (int i = 0; i < kW; ++i) {
            const Eigen::Vector3d p = c.unproject(Eigen::Vector2d(i + 0.5, j + 0.5));
            const double t = z0 / p.z();
            const Eigen::Vector3d hit = t * p;
            if (hit.x() > -half && hit.x() < half && hit.y() > -half && hit.y() < half) {
                ++n;
            }
        }
    }
    return n;
}

const Eigen::Matrix3d kIdentity = Eigen::Matrix3d::Identity();
const Eigen::Vector3d kZero = Eigen::Vector3d::Zero();

}  // namespace

// A fronto-parallel square covers exactly the pixels whose centres fall inside
// its projected rectangle, and every covered pixel carries the analytic
// distance along its bearing.
TEST(Rasterize, FrontoParallelSquareCoversExpectedRect) {
    const Calibration cal = pinhole();
    const Mesh m = axis_square(2.0, 5.0);
    const DepthBuffer b = render_depth(m, cal, kIdentity, kZero);

    ASSERT_EQ(b.width, kW);
    ASSERT_EQ(b.height, kH);
    ASSERT_EQ(b.depth.size(), static_cast<std::size_t>(kW * kH));
    ASSERT_EQ(b.face.size(), static_cast<std::size_t>(kW * kH));

    const std::size_t expected = expected_square_pixels(cal, 2.0, 5.0);
    EXPECT_GT(expected, 0u);
    EXPECT_EQ(count_faces(b, {0, 1}), expected);

    int checked = 0;
    for (int j = 8; j < kH - 8; ++j) {
        for (int i = 8; i < kW - 8; ++i) {
            const std::size_t at = static_cast<std::size_t>(j) * kW + static_cast<std::size_t>(i);
            ASSERT_GE(b.face[at], 0) << "hole at (" << i << ", " << j << ")";
            const double want = plane_depth(cal, i, j, 5.0);
            EXPECT_NEAR(static_cast<double>(b.depth[at]), want, 1e-4 * want)
                << "at (" << i << ", " << j << ")";
            ++checked;
        }
    }
    EXPECT_GT(checked, 0);
}

// Two triangles sharing the diagonal of the square: no gap and no double
// coverage along it, i.e. the per-face counts sum to the full square.
TEST(Rasterize, SharedEdgeIsCoveredExactlyOnce) {
    const Calibration cal = pinhole();
    const Mesh m = axis_square(2.0, 5.0);
    const DepthBuffer b = render_depth(m, cal, kIdentity, kZero);

    const std::size_t total = expected_square_pixels(cal, 2.0, 5.0);
    EXPECT_EQ(count_faces(b, {0}) + count_faces(b, {1}), total);

    // Every pixel is owned by exactly one face or empty -- nothing else.
    for (std::int32_t f : b.face) {
        ASSERT_TRUE(f == 0 || f == 1 || f == -1);
    }
    // Both triangles got real area (the split is not degenerate).
    EXPECT_GT(count_faces(b, {0}), 0u);
    EXPECT_GT(count_faces(b, {1}), 0u);
}

// A nearer square owns every pixel it covers; the farther one shows through
// only where the near square does not reach.
TEST(Rasterize, NearSquareOccludesFarSquare) {
    const Calibration cal = pinhole();
    Mesh m = axis_square(1.0, 3.0);       // faces 0, 1: u in [10.7, 53.3]
    append_square(m, axis_square(3.0, 6.0));  // faces 2, 3: u in [0, 64]
    ASSERT_EQ(m.faces.size(), 4u);

    const DepthBuffer b = render_depth(m, cal, kIdentity, kZero);

    // Centre: both squares project there, the near one wins.
    const std::size_t centre = static_cast<std::size_t>(32) * kW + 32;
    EXPECT_TRUE(b.face[centre] == 0 || b.face[centre] == 1);
    const double near_want = plane_depth(cal, 32, 32, 3.0);
    EXPECT_NEAR(static_cast<double>(b.depth[centre]), near_want, 1e-4 * near_want);

    // Column 2 is outside the near square but inside the far one.
    const std::size_t far_only = static_cast<std::size_t>(32) * kW + 2;
    EXPECT_TRUE(b.face[far_only] == 2 || b.face[far_only] == 3)
        << "face " << b.face[far_only] << " at the far-only pixel";
    const double far_want = plane_depth(cal, 2, 32, 6.0);
    EXPECT_NEAR(static_cast<double>(b.depth[far_only]), far_want, 1e-4 * far_want);
}

// Geometry entirely behind the camera is refused, not clipped: the buffer stays
// empty and an empty pixel is +inf / -1.
TEST(Rasterize, FacesBehindTheCameraAreIgnored) {
    const Calibration cal = pinhole();
    const Mesh m = axis_square(2.0, 5.0);
    const Eigen::Vector3d t(0.0, 0.0, -6.0);  // square lands at camera z = -1
    const DepthBuffer b = render_depth(m, cal, kIdentity, t);

    for (int j = 0; j < kH; ++j) {
        for (int i = 0; i < kW; ++i) {
            const std::size_t at = static_cast<std::size_t>(j) * kW + static_cast<std::size_t>(i);
            ASSERT_EQ(b.face[at], -1) << "at (" << i << ", " << j << ")";
            ASSERT_TRUE(std::isinf(b.depth[at])) << "at (" << i << ", " << j << ")";
        }
    }
}

// Radial distortion changes where the square projects and where each bearing
// lands, but the stored depth is still the analytic ray distance to the plane.
// A screen-space rasteriser approximates the curved distorted edges as straight
// chords, so the tolerance widens with distance from the principal point; the
// spec's claim is the image centre, which is float-limited.
TEST(Rasterize, DistortedCalibrationStillMatchesAnalyticDepth) {
    const Calibration cal = pinhole(64.0, -0.1);
    const Mesh m = axis_square(2.0, 5.0);
    const DepthBuffer b = render_depth(m, cal, kIdentity, kZero);

    const std::size_t expected = expected_square_pixels(cal, 2.0, 5.0);
    EXPECT_EQ(count_faces(b, {0, 1}), expected);

    // The exact centre pixel: analytic to 1e-4 relative.
    const std::size_t centre = static_cast<std::size_t>(32) * kW + 32;
    ASSERT_GE(b.face[centre], 0);
    const double want_centre = plane_depth(cal, 32, 32, 5.0);
    EXPECT_NEAR(static_cast<double>(b.depth[centre]), want_centre, 1e-4 * want_centre);

    // A central window meets a tolerance that grows as O(k1 * (x/z)^2).
    int checked = 0;
    for (int j = 28; j <= 36; ++j) {
        for (int i = 28; i <= 36; ++i) {
            const std::size_t at = static_cast<std::size_t>(j) * kW + static_cast<std::size_t>(i);
            ASSERT_GE(b.face[at], 0) << "at (" << i << ", " << j << ")";
            const double want = plane_depth(cal, i, j, 5.0);
            EXPECT_NEAR(static_cast<double>(b.depth[at]), want, 5e-4 * want)
                << "at (" << i << ", " << j << ")";
            ++checked;
        }
    }
    EXPECT_GT(checked, 0);
}

// An empty mesh produces a correctly sized, entirely empty buffer.
TEST(Rasterize, EmptyMeshLeavesBufferEmpty) {
    const Calibration cal = pinhole();
    const Mesh m;
    const DepthBuffer b = render_depth(m, cal, kIdentity, kZero);
    EXPECT_EQ(b.width, kW);
    EXPECT_EQ(b.depth.size(), static_cast<std::size_t>(kW * kH));
    for (std::int32_t f : b.face) {
        EXPECT_EQ(f, -1);
    }
}

}  // namespace alphashape
