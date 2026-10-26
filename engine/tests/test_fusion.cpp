// Depth-map fusion: analytic plane scenes, merging, normals, colour, filters,
// determinism, and input validation.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "base/utils/error.hpp"
#include "common/algorithm/stereo/fusion.hpp"
#include "common/core/chunk.hpp"
#include "image/image.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace stereo {
namespace {

using Eigen::Matrix3d;
using Eigen::Vector3d;

// A world plane: points X satisfy normal . X == offset, `normal` unit.
struct Plane {
    Vector3d normal = Vector3d(0.0, 0.0, 1.0);
    double offset = 5.0;
};

Matrix3d rot(double ax, double ay, double az) {
    return (Eigen::AngleAxisd(az, Vector3d::UnitZ()) *
            Eigen::AngleAxisd(ay, Vector3d::UnitY()) *
            Eigen::AngleAxisd(ax, Vector3d::UnitX()))
        .toRotationMatrix();
}

// Analytic depth of the plane as seen from a pinhole pose. Depth is the
// distance along the pixel's unit bearing (X_cam = depth * ray).
DepthMap make_plane_map(const Calibration& cal, const Matrix3d& R, const Vector3d& t,
                        const Plane& plane, int camera_id) {
    DepthMap map;
    map.camera_id = camera_id;
    map.width = cal.width;
    map.height = cal.height;
    map.downscale = 2;
    const std::size_t n =
        static_cast<std::size_t>(map.width) * static_cast<std::size_t>(map.height);
    map.depths.assign(n, -1.0F);
    map.normals.assign(n * 3, 0.0F);

    const Vector3d n_cam = R * plane.normal;
    const double s_num = plane.offset + n_cam.dot(t);
    for (int j = 0; j < map.height; ++j) {
        for (int i = 0; i < map.width; ++i) {
            const std::size_t idx =
                static_cast<std::size_t>(j) * static_cast<std::size_t>(map.width) +
                static_cast<std::size_t>(i);
            const Vector3d ray =
                cal.unproject(Eigen::Vector2d(static_cast<double>(i) + 0.5,
                                              static_cast<double>(j) + 0.5));
            const double denom = n_cam.dot(ray);
            if (denom <= 1e-9) {
                continue;
            }
            const double s = s_num / denom;
            if (!(s > 0.0)) {
                continue;
            }
            map.depths[idx] = static_cast<float>(s);
            // Store the camera-frame normal, oriented to face the camera.
            const Vector3d nn = denom > 0.0 ? -n_cam : n_cam;
            map.normals[idx * 3 + 0] = static_cast<float>(nn.x());
            map.normals[idx * 3 + 1] = static_cast<float>(nn.y());
            map.normals[idx * 3 + 2] = static_cast<float>(nn.z());
        }
    }
    return map;
}

Calibration map_calibration() {
    return Calibration::initial(CameraType::Frame, 64, 48, 60.0);
}

Image filled_image(int w, int h, int channels, const std::uint8_t rgb[3]) {
    Image img(w, h, channels);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < channels; ++c) {
                img.at(x, y, c) = (channels == 1) ? rgb[0] : rgb[c];
            }
        }
    }
    return img;
}

// -- tests ------------------------------------------------------------------

TEST(Fusion, EmptyInputsGiveEmptyCloud) {
    const PointCloud cloud = fuse_depth_maps({}, FusionOptions{});
    EXPECT_TRUE(cloud.points.empty());
    EXPECT_TRUE(cloud.normals.empty());
    EXPECT_TRUE(cloud.colours.empty());
    EXPECT_TRUE(cloud.confidence.empty());
}

TEST(Fusion, AllInvalidDepthsGiveEmptyCloud) {
    DepthMap map = make_plane_map(map_calibration(), Matrix3d::Identity(), Vector3d::Zero(),
                                  Plane{}, 0);
    std::fill(map.depths.begin(), map.depths.end(), -1.0F);
    FusionInput in;
    in.map = &map;
    in.calibration = map_calibration();
    const PointCloud cloud = fuse_depth_maps({in}, FusionOptions{});
    EXPECT_TRUE(cloud.points.empty());
}

TEST(Fusion, PlanePointsAndMerging) {
    // Three identical poses: every back-projected point coincides, so the
    // three maps must collapse onto one another under voxel merging.
    const Calibration cal = map_calibration();
    const std::size_t per_map = static_cast<std::size_t>(cal.width) * static_cast<std::size_t>(cal.height);
    std::vector<DepthMap> maps;
    maps.reserve(3);
    std::vector<FusionInput> inputs;
    for (int k = 0; k < 3; ++k) {
        maps.push_back(make_plane_map(cal, Matrix3d::Identity(), Vector3d::Zero(), Plane{}, k));
        FusionInput in;
        in.map = &maps.back();
        in.calibration = cal;
        inputs.push_back(in);
    }
    const PointCloud cloud = fuse_depth_maps(inputs, FusionOptions{});
    std::cout << "[fusion] identical-pose cloud size = " << cloud.points.size()
              << " of " << per_map * 3 << " valid pixels" << std::endl;

    EXPECT_GT(cloud.points.size(), 0U);
    EXPECT_LT(cloud.points.size(), per_map * 3);
    EXPECT_LE(cloud.points.size(), per_map);

    // Every fused point lies on the plane z = 5.
    double max_dist = 0.0;
    for (const Vector3d& p : cloud.points) {
        max_dist = std::max(max_dist, std::abs(p.z() - 5.0));
    }
    EXPECT_LT(max_dist, 1e-6) << "max plane distance = " << max_dist;
}

TEST(Fusion, RotatedPosesStayOnPlane) {
    const Calibration cal = map_calibration();
    const Plane plane;
    const Matrix3d R[3] = {Matrix3d::Identity(), rot(0.0, 0.15, 0.0), rot(-0.1, 0.05, 0.0)};
    const Vector3d t[3] = {Vector3d(0, 0, 0), Vector3d(0.2, 0.05, -0.1), Vector3d(-0.15, 0.1, -0.05)};
    std::vector<DepthMap> maps;
    maps.reserve(3);
    std::vector<FusionInput> inputs;
    for (int k = 0; k < 3; ++k) {
        maps.push_back(make_plane_map(cal, R[k], t[k], plane, k));
        FusionInput in;
        in.map = &maps.back();
        in.calibration = cal;
        in.R = R[k];
        in.t = t[k];
        inputs.push_back(in);
    }
    const PointCloud cloud = fuse_depth_maps(inputs, FusionOptions{});
    ASSERT_GT(cloud.points.size(), 0U);
    double max_dist = 0.0;
    for (const Vector3d& p : cloud.points) {
        max_dist = std::max(max_dist, std::abs(plane.normal.dot(p) - plane.offset));
    }
    EXPECT_LT(max_dist, 1e-6) << "max plane distance = " << max_dist;
}

TEST(Fusion, PointNormalsMatchPlaneNormal) {
    const Calibration cal = map_calibration();
    const Plane plane;
    std::vector<DepthMap> maps;
    maps.reserve(2);
    std::vector<FusionInput> inputs;
    const Matrix3d R[2] = {Matrix3d::Identity(), rot(0.0, 0.2, 0.0)};
    const Vector3d t[2] = {Vector3d::Zero(), Vector3d(0.1, 0.0, -0.05)};
    for (int k = 0; k < 2; ++k) {
        maps.push_back(make_plane_map(cal, R[k], t[k], plane, k));
        FusionInput in;
        in.map = &maps.back();
        in.calibration = cal;
        in.R = R[k];
        in.t = t[k];
        inputs.push_back(in);
    }
    const PointCloud cloud = fuse_depth_maps(inputs, FusionOptions{});
    ASSERT_EQ(cloud.normals.size(), cloud.points.size());
    ASSERT_GT(cloud.normals.size(), 0U);
    const double cos_limit = std::cos(1.0 * 3.14159265358979323846 / 180.0);
    double worst = 1.0;
    for (const Eigen::Vector3f& n : cloud.normals) {
        // World normals face the camera, i.e. ~(0,0,-1) for a camera looking
        // along +Z; the plane normal direction is irrelevant up to sign.
        const double dot =
            std::abs(n.cast<double>().normalized().dot(plane.normal));
        worst = std::min(worst, dot);
    }
    std::cout << "[fusion] worst normal cos = " << worst << std::endl;
    EXPECT_GT(worst, cos_limit);
}

TEST(Fusion, ColoursFromConstantRgbImage) {
    const Calibration cal = map_calibration();
    const std::uint8_t rgb[3] = {10, 200, 30};
    const Image img = filled_image(cal.width * 2, cal.height * 2, 3, rgb);
    const DepthMap map = make_plane_map(cal, Matrix3d::Identity(), Vector3d::Zero(), Plane{}, 0);
    FusionInput in;
    in.map = &map;
    in.calibration = cal;
    in.image = &img;
    const PointCloud cloud = fuse_depth_maps({in}, FusionOptions{});
    ASSERT_GT(cloud.points.size(), 0U);
    ASSERT_EQ(cloud.colours.size(), cloud.points.size() * 3);
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        EXPECT_EQ(cloud.colours[i * 3 + 0], 10);
        EXPECT_EQ(cloud.colours[i * 3 + 1], 200);
        EXPECT_EQ(cloud.colours[i * 3 + 2], 30);
    }
}

TEST(Fusion, GreyImageReplicatesToRgb) {
    const Calibration cal = map_calibration();
    const std::uint8_t grey[3] = {77, 77, 77};
    const Image img = filled_image(cal.width * 2, cal.height * 2, 1, grey);
    const DepthMap map = make_plane_map(cal, Matrix3d::Identity(), Vector3d::Zero(), Plane{}, 0);
    FusionInput in;
    in.map = &map;
    in.calibration = cal;
    in.image = &img;
    const PointCloud cloud = fuse_depth_maps({in}, FusionOptions{});
    ASSERT_EQ(cloud.colours.size(), cloud.points.size() * 3);
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        EXPECT_EQ(cloud.colours[i * 3 + 0], 77);
        EXPECT_EQ(cloud.colours[i * 3 + 1], 77);
        EXPECT_EQ(cloud.colours[i * 3 + 2], 77);
    }
}

TEST(Fusion, ColoursOffGivesNoColours) {
    const Calibration cal = map_calibration();
    const std::uint8_t rgb[3] = {10, 200, 30};
    const Image img = filled_image(cal.width * 2, cal.height * 2, 3, rgb);
    const DepthMap map = make_plane_map(cal, Matrix3d::Identity(), Vector3d::Zero(), Plane{}, 0);
    FusionInput in;
    in.map = &map;
    in.calibration = cal;
    in.image = &img;
    FusionOptions options;
    options.colours = false;
    const PointCloud cloud = fuse_depth_maps({in}, options);
    EXPECT_GT(cloud.points.size(), 0U);
    EXPECT_TRUE(cloud.colours.empty());
}

TEST(Fusion, ColoursRequireAnImageOnEveryInput) {
    // Two coincident maps: one carries an image, the other does not. Every voxel
    // is fed by both, so a coloured output would leave the image-less input's
    // contribution colourless. Colours must therefore be suppressed entirely
    // rather than emit black (or half-black) voxels.
    const Calibration cal = map_calibration();
    const std::uint8_t rgb[3] = {10, 200, 30};
    const Image img = filled_image(cal.width * 2, cal.height * 2, 3, rgb);
    const DepthMap map = make_plane_map(cal, Matrix3d::Identity(), Vector3d::Zero(), Plane{}, 0);

    FusionInput with_image;
    with_image.map = &map;
    with_image.calibration = cal;
    with_image.image = &img;

    FusionInput without_image;
    without_image.map = &map;
    without_image.calibration = cal;

    const PointCloud cloud = fuse_depth_maps({with_image, without_image}, FusionOptions{});
    EXPECT_GT(cloud.points.size(), 0U);
    EXPECT_TRUE(cloud.colours.empty())
        << "colours emitted although one input had no image";

    // With an image on every input the colours come back.
    const PointCloud both =
        fuse_depth_maps({with_image, with_image}, FusionOptions{});
    ASSERT_GT(both.points.size(), 0U);
    EXPECT_EQ(both.colours.size(), both.points.size() * 3);
}

TEST(Fusion, MinConfidenceFiltersLowConfidencePixels) {
    const Calibration cal = map_calibration();
    DepthMap map = make_plane_map(cal, Matrix3d::Identity(), Vector3d::Zero(), Plane{}, 0);
    const std::size_t n = map.depths.size();
    map.confidence.assign(n, 0.5F);
    // Mark a handful of pixels as low confidence.
    int low = 0;
    for (std::size_t i = 0; i < n && low < 200; ++i) {
        if (map.depths[i] > 0.0F) {
            map.confidence[i] = 0.1F;
            ++low;
        }
    }
    FusionInput in;
    in.map = &map;
    in.calibration = cal;

    FusionOptions keep;
    keep.min_confidence = 0.0F;
    const std::size_t all = fuse_depth_maps({in}, keep).points.size();

    FusionOptions drop;
    drop.min_confidence = 0.2F;
    const PointCloud filtered = fuse_depth_maps({in}, drop);
    std::cout << "[fusion] confidence 0.0 -> " << all << " points, 0.2 -> "
              << filtered.points.size() << " points" << std::endl;
    EXPECT_GT(all, 0U);
    EXPECT_LT(filtered.points.size(), all);
    // Confidence is summed per voxel; with 0.5 per kept sample it is positive.
    for (float c : filtered.confidence) {
        EXPECT_GT(c, 0.0F);
    }
}

TEST(Fusion, UnfilteredMapIsNotDroppedByMinConfidence) {
    // A map with no confidence buffer was never cross-view filtered: there is
    // no measurement to compare against min_confidence. Treating the absence of
    // a measurement as confidence 0 would silently discard every point of an
    // expensive map, so an unfiltered map passes the filter untouched. The
    // caller (BuildPointCloud) warns that the filter could not judge it.
    const Calibration cal = map_calibration();
    const DepthMap map = make_plane_map(cal, Matrix3d::Identity(), Vector3d::Zero(), Plane{}, 0);
    ASSERT_TRUE(map.confidence.empty());

    FusionInput in;
    in.map = &map;
    in.calibration = cal;

    FusionOptions options;
    options.min_confidence = 2.0F;  // a threshold no real confidence would pass
    const PointCloud cloud = fuse_depth_maps({in}, options);
    EXPECT_GT(cloud.points.size(), 0U)
        << "an unfiltered map must not be dropped by min_confidence";

    // It is the same cloud as with the filter off: the threshold did not touch
    // it, rather than letting a filtered subset through.
    FusionOptions off;
    off.min_confidence = 0.0F;
    EXPECT_EQ(cloud.points.size(), fuse_depth_maps({in}, off).points.size());
}

TEST(Fusion, MinConfidenceStillFiltersAFilteredMap) {
    // The control for the case above: a map that DOES carry confidence is
    // filtered exactly as before.
    const Calibration cal = map_calibration();
    DepthMap map = make_plane_map(cal, Matrix3d::Identity(), Vector3d::Zero(), Plane{}, 0);
    map.confidence.assign(map.depths.size(), 0.5F);

    FusionInput in;
    in.map = &map;
    in.calibration = cal;

    FusionOptions options;
    options.min_confidence = 2.0F;
    EXPECT_TRUE(fuse_depth_maps({in}, options).points.empty())
        << "a filtered map must still respect min_confidence";
}

TEST(Fusion, DoublingVoxelSizeReducesPointCount) {
    const Calibration cal = map_calibration();
    const Matrix3d R = rot(0.05, 0.1, 0.0);
    const DepthMap map = make_plane_map(cal, R, Vector3d(0.02, 0.0, -0.03), Plane{}, 0);
    FusionInput in;
    in.map = &map;
    in.calibration = cal;
    in.R = R;
    in.t = Vector3d(0.02, 0.0, -0.03);

    // Point spacing on the plane is ~depth/f = 5/60 = 0.083 world units, so
    // the voxel sizes must straddle that scale for merging to differ.
    FusionOptions coarse;
    coarse.voxel_size = 0.25;
    FusionOptions fine;
    fine.voxel_size = 0.1;
    const std::size_t n_fine = fuse_depth_maps({in}, fine).points.size();
    const std::size_t n_coarse = fuse_depth_maps({in}, coarse).points.size();
    std::cout << "[fusion] voxel 0.01 -> " << n_fine << " points, 0.02 -> " << n_coarse
              << " points" << std::endl;
    EXPECT_GT(n_fine, 0U);
    EXPECT_LT(n_coarse, n_fine);
}

TEST(Fusion, DeterministicAcrossThreadCounts) {
    const Calibration cal = map_calibration();
    std::vector<DepthMap> maps;
    maps.reserve(3);
    std::vector<FusionInput> inputs;
    for (int k = 0; k < 3; ++k) {
        maps.push_back(make_plane_map(cal, Matrix3d::Identity(), Vector3d::Zero(), Plane{}, k));
        FusionInput in;
        in.map = &maps.back();
        in.calibration = cal;
        inputs.push_back(in);
    }
    FusionOptions options;
    options.voxel_size = 0.005;
    PointCloud a;
    PointCloud b;
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    a = fuse_depth_maps(inputs, options);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    b = fuse_depth_maps(inputs, options);
    EXPECT_EQ(a.points.size(), b.points.size());
    ASSERT_GT(a.points.size(), 0U);
    for (std::size_t i = 0; i < a.points.size(); ++i) {
        EXPECT_EQ(a.points[i], b.points[i]);
    }
    EXPECT_EQ(a.colours, b.colours);
    EXPECT_EQ(a.confidence, b.confidence);
    EXPECT_EQ(a.normals, b.normals);
}

TEST(Fusion, InvalidInputsThrow) {
    const Calibration cal = map_calibration();
    const DepthMap good = make_plane_map(cal, Matrix3d::Identity(), Vector3d::Zero(), Plane{}, 0);

    FusionOptions options;

    // Null map.
    FusionInput null_in;
    null_in.calibration = cal;
    EXPECT_THROW(fuse_depth_maps({null_in}, options), ValueError);

    // Calibration size does not match.
    FusionInput bad_cal;
    bad_cal.map = &good;
    bad_cal.calibration = Calibration::initial(CameraType::Frame, 32, 24, 30.0);
    EXPECT_THROW(fuse_depth_maps({bad_cal}, options), ValueError);

    // Image smaller than the map's full resolution.
    const std::uint8_t rgb[3] = {1, 2, 3};
    Image tiny(10, 10, 3);
    tiny.at(0, 0, 0) = rgb[0];
    FusionInput bad_img;
    bad_img.map = &good;
    bad_img.calibration = cal;
    bad_img.image = &tiny;
    EXPECT_THROW(fuse_depth_maps({bad_img}, options), ValueError);
}

}  // namespace
}  // namespace stereo
}  // namespace alphashape