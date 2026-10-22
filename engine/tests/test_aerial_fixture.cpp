// test_aerial_fixture.cpp -- milestone-9 aerial scene sanity tests.
//
// These prove the fixture itself (data/make_aerial_fixture.py + the products it
// writes) is self-consistent: the truth poses project the scene the way the
// images were rendered, the DEM matches the analytic height function, the
// chunk built from the truth poses is georeferenced, and ground pixels are
// photo-consistent across views. The DEM/ortho *algorithms* are tested
// elsewhere; this file only guards the ground truth they will be measured
// against.

#include "aerial_fixture.hpp"
#include "base/utils/portable_rng.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/utils/error.hpp"

namespace alphashape {
namespace {

using aerial_fixture::TruthGrid;
using aerial_fixture::TruthPose;

// -- 1. products exist, 12 poses, DEM matches the analytic height ------------

TEST(AerialFixture, ProductsExistAndDemMatchesTruthHeight) {
    const std::vector<std::string> files = aerial_fixture::image_files();
    ASSERT_EQ(files.size(), 12U);
    for (const std::string& file : files) {
        EXPECT_TRUE(std::filesystem::exists(file)) << file;
    }

    const std::vector<TruthPose> poses = aerial_fixture::load_truth();
    ASSERT_EQ(poses.size(), 12U);
    for (std::size_t i = 0; i < poses.size(); ++i) {
        // The stored centre must be consistent with R and t: centre = -R^T t.
        EXPECT_LT((poses[i].centre - poses[i].centre_from_pose()).norm(), 1e-6)
            << "pose " << i;
        // Nadir grid: ~60 m altitude, small deterministic tilt.
        EXPECT_GT(poses[i].centre.z(), 55.0);
        EXPECT_LT(poses[i].centre.z(), 65.0);
        EXPECT_NEAR(poses[i].focal, aerial_fixture::kFocal, 1e-9);
    }

    const TruthGrid grid = aerial_fixture::load_truth_dem();
    EXPECT_EQ(grid.width, 480);
    EXPECT_EQ(grid.height, 360);
    EXPECT_NEAR(grid.cell, 0.25, 1e-12);
    ASSERT_EQ(grid.dem.size(), 480U * 360U);

    // Every cell centre away from the building footprint must equal the analytic
    // terrain height to 1e-4 m. Cells inside the box are excluded: the DEM holds
    // the top surface (roof), which is not the terrain. The margin also skips
    // cells straddling a wall.
    const double margin = 0.75;
    std::size_t checked = 0;
    for (int row = 0; row < grid.height; ++row) {
        for (int col = 0; col < grid.width; ++col) {
            const Eigen::Vector2d c = grid.cell_centre(col, row);
            const bool in_box =
                c.x() >= aerial_fixture::kBuildX0 - margin &&
                c.x() <= aerial_fixture::kBuildX1 + margin &&
                c.y() >= aerial_fixture::kBuildY0 - margin &&
                c.y() <= aerial_fixture::kBuildY1 + margin;
            if (in_box) {
                continue;
            }
            const double truth = aerial_fixture::truth_height(c.x(), c.y());
            EXPECT_NEAR(static_cast<double>(grid.at(col, row)), truth, 1e-4)
                << "cell (" << col << ", " << row << ") at (" << c.x() << ", " << c.y() << ")";
            ++checked;
        }
    }
    EXPECT_GT(checked, 400U * 300U);  // the vast majority of cells are terrain
}

// -- 2. projection consistency + roof visibility -----------------------------

// Project an ENU point through a truth pose into full-frame pixels (u, v).
Eigen::Vector3d project_pixel(const TruthPose& pose, const Eigen::Vector3d& world,
                              double& u, double& v) {
    const Eigen::Vector3d cam = pose.R * world + pose.t;
    u = aerial_fixture::kImageWidth / 2.0 + pose.focal * cam.x() / cam.z();
    v = aerial_fixture::kImageHeight / 2.0 + pose.focal * cam.y() / cam.z();
    return cam;
}

TEST(AerialFixture, AreaCornersProjectConsistentlyAndRoofIsVisible) {
    const std::vector<TruthPose> poses = aerial_fixture::load_truth();
    ASSERT_EQ(poses.size(), 12U);

    // Projection consistency: a point in front of a camera, mapped to pixels
    // and back to its view ray, must lie on that ray to machine precision. A
    // wrong image x/y or z sign would break this for some pose.
    const Eigen::Vector3d probes[5] = {
        {aerial_fixture::kAreaX0, aerial_fixture::kAreaY0, 0.0},
        {aerial_fixture::kAreaX1, aerial_fixture::kAreaY0, 0.0},
        {aerial_fixture::kAreaX0, aerial_fixture::kAreaY1, 0.0},
        {aerial_fixture::kAreaX1, aerial_fixture::kAreaY1, 0.0},
        {0.5 * (aerial_fixture::kBuildX0 + aerial_fixture::kBuildX1),
         0.5 * (aerial_fixture::kBuildY0 + aerial_fixture::kBuildY1), aerial_fixture::kBuildTop}};
    for (const TruthPose& pose : poses) {
        const Eigen::Vector3d centre = pose.centre;
        for (const Eigen::Vector3d& world : probes) {
            double u = 0.0, v = 0.0;
            const Eigen::Vector3d cam = project_pixel(pose, world, u, v);
            if (cam.z() <= 0.0) {
                continue;  // behind this camera; no pixel
            }
            // The pixel's view ray in world space.
            const Eigen::Vector3d ray =
                pose.R.transpose() * Eigen::Vector3d((u - aerial_fixture::kImageWidth / 2.0) /
                                                         pose.focal,
                                                     (v - aerial_fixture::kImageHeight / 2.0) /
                                                         pose.focal,
                                                     1.0);
            const Eigen::Vector3d to_point = world - centre;
            const double range = to_point.norm();
            const Eigen::Vector3d on_ray = ray.normalized() * range;
            EXPECT_LT((to_point - on_ray).norm(), 1e-9)
                << "pose " << pose.name << " point (" << world.transpose() << ")";
        }
    }

    // The nadir grid (480x360 frame, 500 px focal, 60 m AGL) covers the interior
    // of the area: every point on a ±5 m lattice well inside the boundary is in
    // at least one footprint. (The extreme y = ±45 boundary corners fall just
    // outside the ~20.9 m half-footprint of the short axis -- and at 60 m the
    // 480x360 frame is taller than wide in world terms -- so the *interior* is
    // the meaningful coverage claim.)
    for (double x = -55.0; x <= 55.0 + 1e-6; x += 5.0) {
        for (double y = -40.0; y <= 40.0 + 1e-6; y += 5.0) {
            const Eigen::Vector3d world{x, y, aerial_fixture::truth_height(x, y)};
            bool seen = false;
            for (const TruthPose& pose : poses) {
                double u = 0.0, v = 0.0;
                const Eigen::Vector3d cam = project_pixel(pose, world, u, v);
                if (cam.z() > 0.0 && u >= 0.0 && u < aerial_fixture::kImageWidth && v >= 0.0 &&
                    v < aerial_fixture::kImageHeight) {
                    seen = true;
                    break;
                }
            }
            EXPECT_TRUE(seen) << "interior point (" << x << ", " << y
                              << ") not in any camera footprint";
        }
    }

    // Roof centre: at least 4 of the 12 cameras must contain it in frame with
    // the point in front of the sensor.
    const Eigen::Vector3d roof_centre{0.5 * (aerial_fixture::kBuildX0 + aerial_fixture::kBuildX1),
                                      0.5 * (aerial_fixture::kBuildY0 + aerial_fixture::kBuildY1),
                                      aerial_fixture::kBuildTop};
    int visible = 0;
    for (const TruthPose& pose : poses) {
        double u = 0.0, v = 0.0;
        const Eigen::Vector3d cam = project_pixel(pose, roof_centre, u, v);
        if (cam.z() > 0.0 && u >= 0.0 && u < aerial_fixture::kImageWidth && v >= 0.0 &&
            v < aerial_fixture::kImageHeight) {
            ++visible;
        }
    }
    EXPECT_GE(visible, 4) << "roof centre visible in only " << visible << " cameras";
}

// -- 3. the truth chunk document is georeferenced ----------------------------

TEST(AerialFixture, TruthChunkDocumentIsGeoreferenced) {
    Document document = aerial_fixture::make_truth_chunk_document();
    const Chunk& chunk = *document.active_chunk();

    ASSERT_EQ(chunk.cameras().size(), 12U);
    EXPECT_EQ(chunk.sensors().size(), 1U);

    for (const Camera& camera : chunk.cameras()) {
        EXPECT_TRUE(camera.enabled);
        // AddPhotos loaded the EXIF GPS into the camera reference.
        ASSERT_TRUE(camera.reference.location.has_value()) << camera.label;
        const Eigen::Vector3d& gps = *camera.reference.location;
        EXPECT_GT(gps.x(), -180.0);  // longitude
        EXPECT_LT(gps.x(), 180.0);
        EXPECT_GT(gps.y(), -90.0);   // latitude
        EXPECT_LT(gps.y(), 90.0);
        EXPECT_TRUE(std::isfinite(gps.z()));
    }

    // The chunk's local frame is a genuine ENU placement: the origin (0,0,0)
    // maps onto the ECEF of the truth geodetic origin, and it is not identity.
    const geo::Geodetic origin{9.2, 48.7, 300.0};
    const Eigen::Vector3d mapped = (chunk.transform() * Eigen::Vector4d(0, 0, 0, 1)).head<3>();
    const Eigen::Vector3d expected = geo::geodetic_to_ecef(origin);
    EXPECT_LT((mapped - expected).norm(), 1e-6);
    EXPECT_GT((chunk.transform() - Transform::Identity()).cwiseAbs().maxCoeff(), 1.0);
    EXPECT_EQ(chunk.crs().epsg, 4326);
}

// -- 4. photo-consistency of ground pixels across views ----------------------

TEST(AerialFixture, GroundPixelsArePhotoConsistentAcrossViews) {
    const std::vector<TruthPose> poses = aerial_fixture::load_truth();
    ASSERT_EQ(poses.size(), 12U);

    std::vector<Image> images;
    images.reserve(poses.size());
    for (const TruthPose& pose : poses) {
        images.push_back(read_image(aerial_fixture::fixture(pose.name)));
        ASSERT_EQ(images.back().width(), aerial_fixture::kImageWidth);
        ASSERT_EQ(images.back().height(), aerial_fixture::kImageHeight);
    }

    // 200 random ground points away from the building (whose walls occlude and
    // flip which surface a ray sees). Terrain height is analytic and on-frame.
    std::mt19937 rng(20261019U);
    // Portable draws: std::uniform_real_distribution is implementation-defined
    // across libc++ and libstdc++ (TESTING.md, "Numeric consistency").
    const auto dx = [](std::mt19937& g) { return alphashape::rng::uniform_real(g, -58.0, 58.0); };
    const auto dy = [](std::mt19937& g) { return alphashape::rng::uniform_real(g, -43.0, 43.0); };

    double sum_difference = 0.0;
    std::size_t samples = 0;
    std::size_t attempts = 0;
    while (samples < 200 && attempts < 20000) {
        ++attempts;
        const double x = dx(rng);
        const double y = dy(rng);
        // Skip the building footprint with a margin: wall pixels are not
        // photo-consistent ground.
        if (x >= aerial_fixture::kBuildX0 - 2.0 && x <= aerial_fixture::kBuildX1 + 2.0 &&
            y >= aerial_fixture::kBuildY0 - 2.0 && y <= aerial_fixture::kBuildY1 + 2.0) {
            continue;
        }
        const Eigen::Vector3d world{x, y, aerial_fixture::truth_height(x, y)};

        // Greyscale of the point as seen by the first two cameras that contain
        // it, and their difference.
        std::vector<double> greys;
        for (std::size_t i = 0; i < poses.size() && greys.size() < 2; ++i) {
            double u = 0.0, v = 0.0;
            const Eigen::Vector3d cam = project_pixel(poses[i], world, u, v);
            if (cam.z() <= 0.0) {
                continue;
            }
            const int px = static_cast<int>(std::lround(u));
            const int py = static_cast<int>(std::lround(v));
            if (px < 0 || px >= aerial_fixture::kImageWidth || py < 0 ||
                py >= aerial_fixture::kImageHeight) {
                continue;
            }
            const Image& image = images[i];
            const double grey = 0.2126 * image.at(px, py, 0) + 0.7152 * image.at(px, py, 1) +
                                0.0722 * image.at(px, py, 2);
            greys.push_back(grey);
        }
        if (greys.size() < 2) {
            continue;
        }
        sum_difference += std::abs(greys[0] - greys[1]);
        ++samples;
    }

    ASSERT_GE(samples, 150U) << "too few sampled points landed in two views";
    const double mean_difference = sum_difference / static_cast<double>(samples);
    EXPECT_LT(mean_difference, 25.0) << "mean grey difference " << mean_difference
                                     << " over " << samples << " points";
}

}  // namespace
}  // namespace alphashape
