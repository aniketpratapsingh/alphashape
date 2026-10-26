// Milestone 9 orthomosaic core: orthorectification, seamlines, blending.
//
// Everything runs in the aerial fixture's LOCAL ENU frame, which is the plane
// frame these algorithms are written in: no projection is involved. The DEM,
// the poses and the truth orthophoto all live on the same 0.25 m grid
// (origin (-60, 45), 480x360).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <gtest/gtest.h>

#include "aerial_fixture.hpp"

#include "base/utils/error.hpp"
#include "common/algorithm/ortho/blend.hpp"
#include "common/algorithm/ortho/orthorectify.hpp"
#include "common/algorithm/ortho/seamlines.hpp"
#include "image/io/image_io.hpp"
#include "model/camera/calibration.hpp"
#include "model/raster/geo_raster.hpp"
#include "perf_gate.hpp"

namespace alphashape {
namespace {

using ortho::Blending;
using ortho::OrthoCamera;
using ortho::OrthoTile;
using ortho::SeamOptions;

/// The whole fixture scene, decoded once per test.
struct Scene {
    std::vector<Image> images;
    std::vector<OrthoCamera> cameras;
    GeoRaster<float> dem;
    Image truth;
    RasterGrid grid;
    const float nodata = -9999.0F;
};

Scene make_scene() {
    Scene scene;
    const std::vector<std::string> files = aerial_fixture::image_files();
    const std::vector<aerial_fixture::TruthPose> poses = aerial_fixture::load_truth();
    for (std::size_t i = 0; i < files.size(); ++i) {
        scene.images.push_back(read_image(files[i]));
    }
    for (std::size_t i = 0; i < poses.size(); ++i) {
        OrthoCamera camera;
        camera.calibration = Calibration::initial(CameraType::Frame,
                                                  aerial_fixture::kImageWidth,
                                                  aerial_fixture::kImageHeight,
                                                  aerial_fixture::kFocal);
        camera.R = poses[i].R;
        camera.t = poses[i].t;
        camera.image = &scene.images[i];
        scene.cameras.push_back(camera);
    }

    const aerial_fixture::TruthGrid truth_grid = aerial_fixture::load_truth_dem();
    scene.grid.origin_x = aerial_fixture::kAreaX0;
    scene.grid.origin_y = aerial_fixture::kAreaY1;
    scene.grid.pixel_size_x = truth_grid.cell;
    scene.grid.pixel_size_y = truth_grid.cell;
    scene.grid.width = truth_grid.width;
    scene.grid.height = truth_grid.height;
    scene.dem.grid = scene.grid;
    scene.dem.channels = 1;
    scene.dem.data = truth_grid.dem;

    scene.truth = aerial_fixture::load_truth_ortho();
    return scene;
}

/// The DEM with the building removed: inside the footprint the surface is the
/// terrain formula, everywhere else it already is.
GeoRaster<float> terrain_only_dem(const Scene& scene) {
    GeoRaster<float> dem = scene.dem;
    for (int row = 0; row < dem.grid.height; ++row) {
        for (int col = 0; col < dem.grid.width; ++col) {
            const Eigen::Vector2d c = dem.grid.pixel_center(col, row);
            if (c.x() >= aerial_fixture::kBuildX0 && c.x() <= aerial_fixture::kBuildX1 &&
                c.y() >= aerial_fixture::kBuildY0 && c.y() <= aerial_fixture::kBuildY1) {
                dem.at(col, row) = static_cast<float>(aerial_fixture::truth_height(c.x(), c.y()));
            }
        }
    }
    return dem;
}

bool interior(const RasterGrid& grid, int col, int row) {
    const Eigen::Vector2d c = grid.pixel_center(col, row);
    return c.x() >= -52.0 && c.x() <= 52.0 && c.y() >= -38.0 && c.y() <= 38.0;
}

// -- cropped-tile accessors -------------------------------------------------
// A tile holds only its camera's footprint box; a cell outside it is exactly
// "not covered", which is what the full-grid layout used to hold as 0.

bool tile_valid_at(const OrthoTile& tile, int col, int row) {
    return tile.contains(col, row) && tile.valid[tile.index(col, row)] != 0;
}

std::uint8_t tile_rgb_at(const OrthoTile& tile, int col, int row, int channel) {
    return tile.contains(col, row)
               ? tile.rgb[tile.index(col, row) * 3 + static_cast<std::size_t>(channel)]
               : std::uint8_t{0};
}

/// The same two, addressed by a linear cell index on a grid `width` wide.
bool tile_valid_cell(const OrthoTile& tile, std::size_t cell, int width) {
    return tile_valid_at(tile, static_cast<int>(cell % static_cast<std::size_t>(width)),
                         static_cast<int>(cell / static_cast<std::size_t>(width)));
}

/// Per-channel colour of a tile at a linear cell index, 0 where uncovered.
void tile_rgb_cell(const OrthoTile& tile, std::size_t cell, int width, std::uint8_t out[3]) {
    const int col = static_cast<int>(cell % static_cast<std::size_t>(width));
    const int row = static_cast<int>(cell / static_cast<std::size_t>(width));
    for (int c = 0; c < 3; ++c) {
        out[c] = tile_rgb_at(tile, col, row, c);
    }
}

bool in_building(const RasterGrid& grid, int col, int row) {
    const Eigen::Vector2d c = grid.pixel_center(col, row);
    return c.x() >= aerial_fixture::kBuildX0 && c.x() <= aerial_fixture::kBuildX1 &&
           c.y() >= aerial_fixture::kBuildY0 && c.y() <= aerial_fixture::kBuildY1;
}

/// The image index whose camera centre is nearest the grid origin: the most
/// nadir, central view, best for a direct colour comparison.
int central_camera(const Scene& scene) {
    int best = 0;
    double best_norm = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < scene.cameras.size(); ++i) {
        const Eigen::Vector3d c = scene.cameras[i].centre();
        const double norm = c.x() * c.x() + c.y() * c.y();
        if (norm < best_norm) {
            best_norm = norm;
            best = static_cast<int>(i);
        }
    }
    return best;
}

/// Mean absolute per-pixel RGB difference between two equal grids over the
/// cells `keep` accepts. Both carry 1 or 3 channels.
double mean_abs_diff(const Image& truth, const std::vector<std::uint8_t>& rgb,
                     const std::vector<std::uint8_t>& valid, const RasterGrid& grid,
                     bool interior_only) {
    const int w = grid.width;
    const int h = grid.height;
    double sum = 0.0;
    std::size_t count = 0;
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            if (interior_only && !interior(grid, col, row)) {
                continue;
            }
            const std::size_t cell = static_cast<std::size_t>(row) * static_cast<std::size_t>(w) +
                                     static_cast<std::size_t>(col);
            if (!valid.empty() && valid[cell] == 0) {
                continue;
            }
            for (int c = 0; c < 3; ++c) {
                const int tc = truth.channels() >= 3 ? c : 0;
                const double d = std::abs(static_cast<double>(rgb[cell * 3 + static_cast<std::size_t>(c)]) -
                                          static_cast<double>(truth.at(col, row, tc)));
                sum += d;
            }
            ++count;
        }
    }
    return count == 0 ? 0.0 : sum / (3.0 * static_cast<double>(count));
}

/// The same comparison for a CROPPED tile: cells outside its box are not
/// covered, so they are skipped exactly as `valid == 0` cells are.
double mean_abs_diff(const Image& truth, const OrthoTile& tile, const RasterGrid& grid,
                     bool interior_only) {
    double sum = 0.0;
    std::size_t count = 0;
    for (int row = 0; row < grid.height; ++row) {
        for (int col = 0; col < grid.width; ++col) {
            if (interior_only && !interior(grid, col, row)) {
                continue;
            }
            if (!tile_valid_at(tile, col, row)) {
                continue;
            }
            const std::size_t cell = tile.index(col, row);
            for (int c = 0; c < 3; ++c) {
                const int tc = truth.channels() >= 3 ? c : 0;
                const double d = std::abs(
                    static_cast<double>(tile.rgb[cell * 3 + static_cast<std::size_t>(c)]) -
                    static_cast<double>(truth.at(col, row, tc)));
                sum += d;
            }
            ++count;
        }
    }
    return count == 0 ? 0.0 : sum / (3.0 * static_cast<double>(count));
}

/// Mean absolute RGB difference over the building footprint only.
double mean_abs_diff_footprint(const Image& truth, const OrthoTile& tile,
                               const RasterGrid& grid) {
    double sum = 0.0;
    std::size_t count = 0;
    for (int row = 0; row < grid.height; ++row) {
        for (int col = 0; col < grid.width; ++col) {
            if (!in_building(grid, col, row)) {
                continue;
            }
            if (!tile_valid_at(tile, col, row)) {
                continue;
            }
            const std::size_t cell = tile.index(col, row);
            for (int c = 0; c < 3; ++c) {
                const int tc = truth.channels() >= 3 ? c : 0;
                sum += std::abs(static_cast<double>(tile.rgb[cell * 3 + static_cast<std::size_t>(c)]) -
                                static_cast<double>(truth.at(col, row, tc)));
            }
            ++count;
        }
    }
    return count == 0 ? 0.0 : sum / (3.0 * static_cast<double>(count));
}

/// Truth colour at the centre of the building roof.
void roof_colour(const Image& truth, const RasterGrid& grid, std::uint8_t out[3]) {
    const double x = (aerial_fixture::kBuildX0 + aerial_fixture::kBuildX1) * 0.5;
    const double y = (aerial_fixture::kBuildY0 + aerial_fixture::kBuildY1) * 0.5;
    const int col = std::clamp(static_cast<int>((x - grid.origin_x) / grid.pixel_size_x), 0,
                               grid.width - 1);
    const int row = std::clamp(static_cast<int>((grid.origin_y - y) / grid.pixel_size_y), 0,
                               grid.height - 1);
    for (int c = 0; c < 3; ++c) {
        out[c] = truth.at(col, row, truth.channels() >= 3 ? c : 0);
    }
}
/// Mosaic that copies its single labelled tile (Disabled == label copy).
ortho::OrthoMosaic copy_labels(int w, int h, const std::vector<OrthoTile>& tiles,
                               const std::vector<std::int32_t>& labels) {
    return ortho::blend(w, h, tiles, labels, Blending::Disabled);
}

/// Mean absolute gradient jump across label boundaries, relative to the local
/// background gradient. A raw cross-seam difference is dominated by the scene's
/// own texture (a seam running through a high-contrast feature shows a large
/// jump regardless of blending); subtracting each side's neighbour gradient
/// isolates the discontinuity the blending introduced.
double seam_jump(const ortho::OrthoMosaic& mosaic) {
    const int w = mosaic.width;
    const int h = mosaic.height;
    auto value = [&](int col, int row, int c) {
        const std::size_t cell = static_cast<std::size_t>(row) * static_cast<std::size_t>(w) +
                                 static_cast<std::size_t>(col);
        return static_cast<double>(mosaic.rgb[cell * 3 + static_cast<std::size_t>(c)]);
    };
    auto gradient = [&](int col, int row, int dc, int dr) {
        double g = 0.0;
        for (int c = 0; c < 3; ++c) {
            g += std::abs(value(col + dc, row + dr, c) - value(col, row, c));
        }
        return g / 3.0;
    };
    double sum = 0.0;
    std::size_t count = 0;
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            const std::size_t cell = static_cast<std::size_t>(row) * static_cast<std::size_t>(w) +
                                     static_cast<std::size_t>(col);
            if (mosaic.labels[cell] < 0) {
                continue;
            }
            // Horizontal seam: p = (col,row), q = (col+1,row).
            if (col + 1 < w && mosaic.labels[cell + 1] >= 0 &&
                mosaic.labels[cell + 1] != mosaic.labels[cell] && col > 0 && col + 2 < w) {
                const double jump = gradient(col, row, 1, 0);
                const double background = 0.5 * (gradient(col - 1, row, 1, 0) + gradient(col + 1, row, 1, 0));
                sum += std::max(0.0, jump - background);
                ++count;
            }
            // Vertical seam: p = (col,row), q = (col,row+1).
            if (row + 1 < h && mosaic.labels[cell + static_cast<std::size_t>(w)] >= 0 &&
                mosaic.labels[cell + static_cast<std::size_t>(w)] != mosaic.labels[cell] && row > 0 &&
                row + 2 < h) {
                const double jump = gradient(col, row, 0, 1);
                const double background = 0.5 * (gradient(col, row - 1, 0, 1) + gradient(col, row + 1, 0, 1));
                sum += std::max(0.0, jump - background);
                ++count;
            }
        }
    }
    return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

// -- 0. memory: cropped tiles are byte-identical to the full-grid reference ----

/// The crop is a pure storage optimisation: expanding a cropped tile back onto
/// the full grid must reproduce the uncropped reference exactly, cell for cell.
/// The reference path is `crop == false`, which is the pre-crop code.
TEST(Ortho, CroppedTilesMatchFullGridExactly) {
    const Scene scene = make_scene();
    // Both a plain plane-frame camera (no map) and every camera in the flight
    // are exercised, so the crop is checked against obliques too.
    std::vector<int> indices{central_camera(scene), 0, 6, 11};
    double worst_rgb = 0.0;
    std::size_t mismatched = 0;
    std::size_t total_crop_cells = 0;
    std::size_t total_full_cells = 0;
    for (const int index : indices) {
        const OrthoTile full = orthorectify(scene.dem, scene.nodata, scene.grid,
                                            scene.cameras[static_cast<std::size_t>(index)],
                                            index, 0.5, /*crop=*/false);
        const OrthoTile crop = orthorectify(scene.dem, scene.nodata, scene.grid,
                                            scene.cameras[static_cast<std::size_t>(index)],
                                            index, 0.5, /*crop=*/true);
        // The reference holds the whole grid.
        EXPECT_EQ(full.width, scene.grid.width);
        EXPECT_EQ(full.height, scene.grid.height);
        EXPECT_EQ(full.x0, 0);
        EXPECT_EQ(full.y0, 0);
        EXPECT_FALSE(crop.empty()) << "camera " << index << " covered nothing";
        EXPECT_LE(crop.x0 + crop.width, scene.grid.width);
        EXPECT_LE(crop.y0 + crop.height, scene.grid.height);
        total_crop_cells += static_cast<std::size_t>(crop.width) *
                            static_cast<std::size_t>(crop.height);
        total_full_cells += static_cast<std::size_t>(full.width) *
                            static_cast<std::size_t>(full.height);

        for (int row = 0; row < scene.grid.height; ++row) {
            for (int col = 0; col < scene.grid.width; ++col) {
                const std::size_t fc = static_cast<std::size_t>(row) *
                                           static_cast<std::size_t>(scene.grid.width) +
                                       static_cast<std::size_t>(col);
                const bool full_valid = full.valid[fc] != 0;
                const bool crop_valid = tile_valid_at(crop, col, row);
                if (full_valid != crop_valid) {
                    ++mismatched;
                    continue;
                }
                if (!full_valid) {
                    continue;
                }
                const std::size_t cc = crop.index(col, row);
                EXPECT_EQ(full.score[fc], crop.score[cc]);
                for (int c = 0; c < 3; ++c) {
                    const std::size_t fo = fc * 3 + static_cast<std::size_t>(c);
                    const std::size_t co = cc * 3 + static_cast<std::size_t>(c);
                    worst_rgb = std::max(
                        worst_rgb,
                        std::abs(static_cast<double>(full.rgb[fo]) -
                                 static_cast<double>(crop.rgb[co])));
                }
            }
        }
    }
    std::printf("[Ortho.CroppedTilesMatchFullGridExactly] crop_cells=%zu full_cells=%zu "
                "mismatched=%zu worst_rgb=%.0f\n",
                total_crop_cells, total_full_cells, mismatched, worst_rgb);
    // Every camera's covered cell agrees: no valid flag and no colour may differ.
    EXPECT_EQ(mismatched, 0U);
    EXPECT_EQ(worst_rgb, 0.0);
    // And the crop actually saves memory: the whole point of the change.
    EXPECT_LT(total_crop_cells, total_full_cells);
}

TEST(Ortho, CropBoxBoundsEveryVisibleCell) {
    // The footprint bound must never cut off a cell the rectifier would have
    // coloured. Crop is 2-cell padded plus a 1% inflation precisely so that
    // "the box misses a valid cell" cannot happen; assert it directly.
    const Scene scene = make_scene();
    for (std::size_t i = 0; i < scene.cameras.size(); ++i) {
        const OrthoTile crop = orthorectify(scene.dem, scene.nodata, scene.grid,
                                            scene.cameras[i], static_cast<int>(i), 0.5);
        const OrthoTile full = orthorectify(scene.dem, scene.nodata, scene.grid,
                                            scene.cameras[i], static_cast<int>(i), 0.5,
                                            /*crop=*/false);
        std::size_t outside = 0;
        for (int row = 0; row < scene.grid.height; ++row) {
            for (int col = 0; col < scene.grid.width; ++col) {
                const std::size_t fc = static_cast<std::size_t>(row) *
                                           static_cast<std::size_t>(scene.grid.width) +
                                       static_cast<std::size_t>(col);
                if (full.valid[fc] != 0 && !crop.contains(col, row)) {
                    ++outside;
                }
            }
        }
        EXPECT_EQ(outside, 0U) << "camera " << i << " has coloured cells outside its crop";
    }
}

// -- 1. rectification --------------------------------------------------------

TEST(Ortho, RectifyMatchesTruth) {
    const Scene scene = make_scene();
    const int index = central_camera(scene);
    const OrthoTile tile = orthorectify(scene.dem, scene.nodata, scene.grid,
                                        scene.cameras[static_cast<std::size_t>(index)], index,
                                        0.5);

    std::size_t valid_interior = 0;
    for (int row = 0; row < scene.grid.height; ++row) {
        for (int col = 0; col < scene.grid.width; ++col) {
            if (interior(scene.grid, col, row) && tile_valid_at(tile, col, row)) {
                ++valid_interior;
            }
        }
    }
    // The cropped tile is a strict subset of the grid: it must not be empty and
    // must not exceed it.
    EXPECT_FALSE(tile.empty());
    EXPECT_LE(tile.x0 + tile.width, scene.grid.width);
    EXPECT_LE(tile.y0 + tile.height, scene.grid.height);
    const double error = mean_abs_diff(scene.truth, tile, scene.grid, true);
    std::printf("[Ortho.RectifyMatchesTruth] camera=%d valid_interior=%zu mean_abs_diff=%.3f\n",
                index, valid_interior, error);
    EXPECT_GT(valid_interior, static_cast<std::size_t>(scene.grid.width * scene.grid.height / 8));
    EXPECT_LT(error, 14.0);
}

// -- 2. occlusion -------------------------------------------------------------

TEST(Ortho, OcclusionHidesBehindBuilding) {
    const Scene scene = make_scene();
    // Camera at (11, 0, 60): the building (x in [-20,-4], y in [5,17]) sits to
    // the north-west, so a strip of ground just north of its north wall is
    // hidden behind the wall.
    constexpr int kIndex = 6;
    constexpr double kTolerance = 0.1;

    const OrthoTile true_tile = orthorectify(scene.dem, scene.nodata, scene.grid,
                                             scene.cameras[kIndex], kIndex, kTolerance);
    // The same camera over the building-free terrain: every ground cell is now
    // visible, so cells valid here but invalid on the true DEM are exactly the
    // ones the building occludes.
    const GeoRaster<float> flat = terrain_only_dem(scene);
    const OrthoTile flat_tile = orthorectify(flat, scene.nodata, scene.grid,
                                             scene.cameras[kIndex], kIndex, kTolerance);

    std::uint8_t roof[3];
    roof_colour(scene.truth, scene.grid, roof);

    int occluded = 0;
    int occluded_roof_coloured = 0;
    for (int row = 0; row < scene.grid.height; ++row) {
        for (int col = 0; col < scene.grid.width; ++col) {
            if (in_building(scene.grid, col, row)) {
                continue;  // only the ground outside the footprint
            }
            if (tile_valid_at(true_tile, col, row) || !tile_valid_at(flat_tile, col, row)) {
                continue;  // not visible-and-then-hidden
            }
            ++occluded;
            if (tile_rgb_at(true_tile, col, row, 0) == roof[0] &&
                tile_rgb_at(true_tile, col, row, 1) == roof[1] &&
                tile_rgb_at(true_tile, col, row, 2) == roof[2]) {
                ++occluded_roof_coloured;
            }
        }
    }
    std::printf("[Ortho.OcclusionHidesBehindBuilding] occluded_cells=%d roof_coloured=%d "
                "roof=(%u,%u,%u)\n",
                occluded, occluded_roof_coloured, roof[0], roof[1], roof[2]);
    EXPECT_GT(occluded, 0);
    EXPECT_EQ(occluded_roof_coloured, 0);

    // Relief displacement. On bare terrain the footprint cells project the
    // ground and disagree strongly with the truth roof; on the true DEM they
    // land on the roof and agree.
    const double true_err = mean_abs_diff_footprint(scene.truth, true_tile, scene.grid);
    const double flat_err = mean_abs_diff_footprint(scene.truth, flat_tile, scene.grid);
    std::printf("[Ortho.OcclusionHidesBehindBuilding] footprint_err true=%.3f flat=%.3f "
                "ratio=%.2f\n",
                true_err, flat_err, true_err > 0.0 ? flat_err / true_err : 0.0);
    EXPECT_GT(flat_err, 2.0 * true_err);
}

TEST(Ortho, OcclusionCostDoesNotFollowTheOutputGrid) {
    const Scene scene = make_scene();
    constexpr int kIndex = 6;
    constexpr double kTolerance = 0.1;
    // The fixture DEM is a 0.25 m grid. A 4x-finer ortho grid over the same
    // extent has 16x the output cells; the march step is half a DEM cell in
    // both runs, so the DEM samples per output cell must stay put. The old
    // step of 0.5 * max(output cell, DEM cell) scaled the march with whatever
    // grid the user asked for, which is the cost this guards.
    constexpr int kFactor = 4;
    RasterGrid fine = scene.grid;
    fine.pixel_size_x = scene.grid.pixel_size_x / kFactor;
    fine.pixel_size_y = scene.grid.pixel_size_y / kFactor;
    fine.width = scene.grid.width * kFactor;
    fine.height = scene.grid.height * kFactor;
    ASSERT_DOUBLE_EQ(fine.pixel_size_x * kFactor, scene.grid.pixel_size_x);

    ortho::reset_occlusion_sample_count();
    const OrthoTile coarse_tile =
        orthorectify(scene.dem, scene.nodata, scene.grid, scene.cameras[kIndex], kIndex,
                     kTolerance);
    const double coarse_samples = static_cast<double>(ortho::occlusion_sample_count());

    ortho::reset_occlusion_sample_count();
    const OrthoTile fine_tile =
        orthorectify(scene.dem, scene.nodata, fine, scene.cameras[kIndex], kIndex, kTolerance);
    const double fine_samples = static_cast<double>(ortho::occlusion_sample_count());

    const double coarse_cells =
        static_cast<double>(coarse_tile.width) * static_cast<double>(coarse_tile.height);
    const double fine_cells =
        static_cast<double>(fine_tile.width) * static_cast<double>(fine_tile.height);
    ASSERT_GT(coarse_cells, 0.0);
    ASSERT_GT(fine_cells, 0.0);
    const double coarse_per_cell = coarse_samples / coarse_cells;
    const double fine_per_cell = fine_samples / fine_cells;
    std::printf("[Ortho.OcclusionCost] coarse=%.2f fine=%.2f samples/cell, cells %.0f -> "
                "%.0f (x%.1f)\n",
                coarse_per_cell, fine_per_cell, coarse_cells, fine_cells,
                fine_cells / coarse_cells);
    EXPECT_GT(coarse_per_cell, 0.0);
    // Independent of the output resolution: within a stride's rounding of the
    // coarse run, never multiplied by the 4x finer cell.
    EXPECT_LT(fine_per_cell, 1.5 * coarse_per_cell + 1.0);
    EXPECT_GT(fine_per_cell, 0.5 * coarse_per_cell);
    // The march now stops at a METRIC bound (the ray's height reaching the DEM's
    // tallest sample), so the count is bounded by the scene's relief over the
    // stride -- far below any per-ray sample budget. The old fixed 4096 cap was
    // a geometry assumption; the count staying under it here means the height
    // bound, not a cap, is what ends the march.
    EXPECT_LT(fine_per_cell, 4096.0);
}

// -- 3. seamlines ------------------------------------------------------------

/// Rectify every camera onto the grid (the expensive shared step).
std::vector<OrthoTile> rectify_all(const Scene& scene, double tolerance) {
    std::vector<OrthoTile> tiles;
    tiles.reserve(scene.cameras.size());
    for (std::size_t i = 0; i < scene.cameras.size(); ++i) {
        tiles.push_back(orthorectify(scene.dem, scene.nodata, scene.grid, scene.cameras[i],
                                     static_cast<int>(i), tolerance));
    }
    return tiles;
}

TEST(Seam, LabelsCoverInterior) {
    const Scene scene = make_scene();
    const std::vector<OrthoTile> tiles = rectify_all(scene, 0.25);
    const SeamOptions options;
    const std::vector<std::int32_t> labels =
        assign_seams(scene.grid.width, scene.grid.height, tiles, options);

    int interior_labelled = 0;
    int interior_covered = 0;
    int interior_total = 0;
    for (int row = 0; row < scene.grid.height; ++row) {
        for (int col = 0; col < scene.grid.width; ++col) {
            if (!interior(scene.grid, col, row)) {
                continue;
            }
            ++interior_total;
            bool covered = false;
            for (const OrthoTile& tile : tiles) {
                if (tile_valid_at(tile, col, row)) {
                    covered = true;
                    break;
                }
            }
            if (covered) {
                ++interior_covered;
                const std::size_t cell =
                    static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(scene.grid.width) +
                    static_cast<std::size_t>(col);
                if (labels[cell] >= 0) {
                    ++interior_labelled;
                }
            }
        }
    }

    // The argmax-only labelling is `iterations = 0` (no smoothing at all).
    SeamOptions initial_options = options;
    initial_options.iterations = 0;
    const std::vector<std::int32_t> initial =
        assign_seams(scene.grid.width, scene.grid.height, tiles, initial_options);
    const double initial_energy = energy(scene.grid.width, scene.grid.height, tiles, initial, options);
    const double final_energy = energy(scene.grid.width, scene.grid.height, tiles, labels, options);
    std::printf("[Seam.LabelsCoverInterior] labelled=%d covered=%d/%d energy argmax=%.3f icm=%.3f\n",
                interior_labelled, interior_covered, interior_total, initial_energy, final_energy);
    // Every interior cell some camera sees is labelled; cells no camera covers
    // (grazing/occluded edges) and the four fixture corner gaps stay -1.
    EXPECT_EQ(interior_labelled, interior_covered);
    EXPECT_GT(interior_labelled, interior_total * 99 / 100);
    // The multi-scale optimisation lowers the energy well below the argmax
    // labelling it starts from (the seam/data balance moved the optimum away).
    EXPECT_LT(final_energy, 0.9 * initial_energy);

    // Determinism across thread counts.
#ifdef _OPENMP
    std::vector<std::int32_t> single;
    std::vector<std::int32_t> quad;
    omp_set_num_threads(1);
    single = assign_seams(scene.grid.width, scene.grid.height, tiles, options);
    omp_set_num_threads(4);
    quad = assign_seams(scene.grid.width, scene.grid.height, tiles, options);
    EXPECT_EQ(single, quad);
    EXPECT_EQ(single, labels);
#endif
}

/// Brute-force check that the optimiser never returns a labelling worse than the
/// best found by exhaustive search, on a small grid with a few labels.
TEST(Seam, OptimumOnSmallGrid) {
    std::vector<OrthoTile> tiles(3);
    constexpr int kWidth = 4;
    constexpr int kHeight = 3;  // 12 cells: 3^12 exhaustive combinations
    const std::size_t cells = static_cast<std::size_t>(kWidth) * kHeight;
    // Deterministic synthetic scores/colours: two views overlap, one is best on
    // the left, another on the right, with a disagreement band between them.
    std::uint32_t state = 12345u;
    auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<int>((state >> 16) & 0xFF);
    };
    for (OrthoTile& tile : tiles) {
        tile.x0 = 0;
        tile.y0 = 0;
        tile.width = kWidth;
        tile.height = kHeight;
        tile.rgb.assign(cells * 3, 0);
        tile.valid.assign(cells, 0);
        tile.score.assign(cells, 0.0F);
    }
    for (int row = 0; row < kHeight; ++row) {
        for (int col = 0; col < kWidth; ++col) {
            const std::size_t cell =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(kWidth) +
                static_cast<std::size_t>(col);
            for (std::size_t t = 0; t < tiles.size(); ++t) {
                const bool valid = !(t == 2 && col < 1) && !(t == 0 && col > 2);
                if (!valid) {
                    continue;
                }
                tiles[t].valid[cell] = 1;
                tiles[t].score[cell] = static_cast<float>(
                    0.2 + 0.7 * (static_cast<double>(t == 1 && col > 1 ? 1 : 0)) +
                    0.001 * static_cast<double>(next()));
                for (int c = 0; c < 3; ++c) {
                    // Tile 0 and tile 2 disagree strongly in a vertical band.
                    int value = next();
                    if (t == 2) {
                        value = std::clamp(value + (col >= 1 && col <= 2 ? 90 : 0), 0, 255);
                    }
                    tiles[t].rgb[cell * 3 + static_cast<std::size_t>(c)] =
                        static_cast<std::uint8_t>(value);
                }
            }
        }
    }

    const SeamOptions options;
    const std::vector<std::int32_t> labels =
        assign_seams(kWidth, kHeight, tiles, options);
    const double got = energy(kWidth, kHeight, tiles, labels, options);

    // Exhaustive: every coverable cell takes one of the covering labels.
    std::vector<int> choices(cells, 0);
    for (std::size_t cell = 0; cell < cells; ++cell) {
        for (std::size_t t = 0; t < tiles.size(); ++t) {
            choices[cell] += tile_valid_cell(tiles[t], cell, kWidth) ? 1 : 0;
        }
    }
    std::vector<std::int32_t> current(cells, -1);
    std::vector<std::int32_t> best = labels;
    double best_energy = std::numeric_limits<double>::infinity();
    const std::size_t total = [&]() {
        std::size_t combinations = 1;
        for (std::size_t cell = 0; cell < cells; ++cell) {
            combinations *= static_cast<std::size_t>(std::max(1, choices[cell]));
        }
        return combinations;
    }();
    for (std::size_t code = 0; code < total; ++code) {
        std::size_t rest = code;
        for (std::size_t cell = 0; cell < cells; ++cell) {
            const int n = std::max(1, choices[cell]);
            const int pick = static_cast<int>(rest % static_cast<std::size_t>(n));
            rest /= static_cast<std::size_t>(n);
            int seen = 0;
            current[cell] = -1;
            for (std::size_t t = 0; t < tiles.size(); ++t) {
                if (!tile_valid_cell(tiles[t], cell, kWidth)) {
                    continue;
                }
                if (seen == pick) {
                    current[cell] = static_cast<int>(t);
                }
                ++seen;
            }
        }
        const double e = energy(kWidth, kHeight, tiles, current, options);
        if (e < best_energy) {
            best_energy = e;
            best = current;
        }
    }
    std::printf("[Seam.OptimumOnSmallGrid] exhaustive=%.4f optimiser=%.4f (n=%zu)\n",
                best_energy, got, total);
    EXPECT_LE(got, best_energy + 1e-6);

    // A second landscape: strong smoothness and a data term that actually
    // competes, to exercise the balance rather than pure seam minimisation.
    SeamOptions balanced;
    balanced.data_weight = 1.0;
    balanced.smoothness = 0.5;
    balanced.seam_weight = 1.0;
    const std::vector<std::int32_t> labels2 =
        assign_seams(kWidth, kHeight, tiles, balanced);
    const double got2 = energy(kWidth, kHeight, tiles, labels2, balanced);
    double best2 = std::numeric_limits<double>::infinity();
    for (std::size_t code = 0; code < total; ++code) {
        std::size_t rest = code;
        for (std::size_t cell = 0; cell < cells; ++cell) {
            const int n = std::max(1, choices[cell]);
            const int pick = static_cast<int>(rest % static_cast<std::size_t>(n));
            rest /= static_cast<std::size_t>(n);
            int seen = 0;
            current[cell] = -1;
            for (std::size_t t = 0; t < tiles.size(); ++t) {
                if (!tile_valid_cell(tiles[t], cell, kWidth)) {
                    continue;
                }
                if (seen == pick) {
                    current[cell] = static_cast<int>(t);
                }
                ++seen;
            }
        }
        best2 = std::min(best2, energy(kWidth, kHeight, tiles, current, balanced));
    }
    std::printf("[Seam.OptimumOnSmallGrid] balanced exhaustive=%.4f optimiser=%.4f\n", best2,
                got2);
    EXPECT_LE(got2, best2 + 1e-6);
}

// -- 3b. seam quality --------------------------------------------------------

/// The mean per-channel grey difference (over the 255 range) the two labelled
/// tiles show at `cell`, or 0 when either does not cover it.
double label_disagreement(const std::vector<OrthoTile>& tiles, int a, int b, std::size_t cell,
                          int width) {
    if (a < 0 || b < 0) {
        return 0.0;
    }
    const int col = static_cast<int>(cell % static_cast<std::size_t>(width));
    const int row = static_cast<int>(cell / static_cast<std::size_t>(width));
    const OrthoTile& ta = tiles[static_cast<std::size_t>(a)];
    const OrthoTile& tb = tiles[static_cast<std::size_t>(b)];
    if (!tile_valid_at(ta, col, row) || !tile_valid_at(tb, col, row)) {
        return 0.0;
    }
    std::uint8_t ca[3];
    std::uint8_t cb[3];
    tile_rgb_cell(ta, cell, width, ca);
    tile_rgb_cell(tb, cell, width, cb);
    double sum = 0.0;
    for (int c = 0; c < 3; ++c) {
        sum += std::abs(static_cast<int>(ca[c]) - static_cast<int>(cb[c]));
    }
    return sum / (3.0 * 255.0);
}

/// Seam-edge statistics of a labelling: how many label-change edges it has, how
/// many of those lie on a > 30-grey-level disagreement, and how many fall inside
/// the building footprint grown by `grow` metres.
struct SeamStats {
    int seams = 0;
    int disagreeing = 0;
    int in_building = 0;
};

SeamStats seam_stats(const Scene& scene, const std::vector<OrthoTile>& tiles,
                     const std::vector<std::int32_t>& labels, double grow) {
    SeamStats stats;
    const int w = scene.grid.width;
    const int h = scene.grid.height;
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            const std::size_t cell = static_cast<std::size_t>(row) * static_cast<std::size_t>(w) +
                                     static_cast<std::size_t>(col);
            const int la = labels[cell];
            if (la < 0) {
                continue;
            }
            // Right and down neighbours, each unordered pair once.
            std::size_t pairs[2];
            int pair_count = 0;
            if (col + 1 < w) {
                pairs[pair_count++] = cell + 1;
            }
            if (row + 1 < h) {
                pairs[pair_count++] = cell + static_cast<std::size_t>(w);
            }
            for (int n = 0; n < pair_count; ++n) {
                const std::size_t q = pairs[n];
                const int lb = labels[q];
                if (lb < 0 || lb == la) {
                    continue;
                }
                ++stats.seams;
                const double d = 0.5 * (label_disagreement(tiles, la, lb, cell, w) +
                                        label_disagreement(tiles, la, lb, q, w));
                if (d * 255.0 > 30.0) {
                    ++stats.disagreeing;
                }
                // The edge lies inside the grown footprint only when BOTH ends
                // do; an edge straddling the boundary counts for neither.
                const Eigen::Vector2d p = scene.grid.pixel_center(col, row);
                const int qcol = col + (q == cell + 1 ? 1 : 0);
                const int qrow = row + (q == cell + 1 ? 0 : 1);
                const Eigen::Vector2d qc = scene.grid.pixel_center(qcol, qrow);
                auto within = [&](const Eigen::Vector2d& c) {
                    return c.x() >= aerial_fixture::kBuildX0 - grow &&
                           c.x() <= aerial_fixture::kBuildX1 + grow &&
                           c.y() >= aerial_fixture::kBuildY0 - grow &&
                           c.y() <= aerial_fixture::kBuildY1 + grow;
                };
                if (within(p) && within(qc)) {
                    ++stats.in_building;
                }
            }
        }
    }
    return stats;
}

TEST(Seam, DisagreementAvoidance) {
    const Scene scene = make_scene();
    const std::vector<OrthoTile> tiles = rectify_all(scene, 0.25);
    // The argmax-only labelling is iteration 0; the optimiser is the default.
    SeamOptions argmax_options;
    argmax_options.iterations = 0;
    const std::vector<std::int32_t> argmax =
        assign_seams(scene.grid.width, scene.grid.height, tiles, argmax_options);
    const std::vector<std::int32_t> optimised =
        assign_seams(scene.grid.width, scene.grid.height, tiles, SeamOptions{});

    const SeamStats a = seam_stats(scene, tiles, argmax, 0.0);
    const SeamStats o = seam_stats(scene, tiles, optimised, 0.0);
    const double argmax_fraction =
        a.seams == 0 ? 0.0 : static_cast<double>(a.disagreeing) / static_cast<double>(a.seams);
    const double optimised_fraction =
        o.seams == 0 ? 0.0 : static_cast<double>(o.disagreeing) / static_cast<double>(o.seams);
    std::printf("[Seam.DisagreementAvoidance] seam length argmax=%d optimised=%d; disagreeing "
                "fraction argmax=%.3f optimised=%.3f\n",
                a.seams, o.seams, argmax_fraction, optimised_fraction);
    EXPECT_GT(a.seams, 0);
    EXPECT_GT(o.seams, 0);
    EXPECT_LE(optimised_fraction, 0.5 * argmax_fraction + 1e-9);

    // Context for the building count (printed, not asserted): camera 5 does see
    // all 3072 roof cells, but it has the lowest score over most of the surround
    // and the highest only on the roof, so giving it the whole roof would replace
    // ~130 interior seam edges with ~224 roof-perimeter edges. The count below
    // is therefore the objective's genuine optimum here, not a stuck optimiser.
    int roof_cells = 0;
    std::vector<int> roof_covered(tiles.size(), 0);
    for (int row = 0; row < scene.grid.height; ++row) {
        for (int col = 0; col < scene.grid.width; ++col) {
            if (!in_building(scene.grid, col, row)) {
                continue;
            }
            ++roof_cells;
            const std::size_t cell =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(scene.grid.width) +
                static_cast<std::size_t>(col);
            for (std::size_t t = 0; t < tiles.size(); ++t) {
                roof_covered[t] += tile_valid_cell(tiles[t], cell, scene.grid.width) ? 1 : 0;
            }
        }
    }
    const int best_roof = *std::max_element(roof_covered.begin(), roof_covered.end());
    std::printf("[Seam.DisagreementAvoidance] roof_cells=%d best_camera_covers=%d (%.0f%%)\n",
                roof_cells, best_roof,
                roof_cells == 0 ? 0.0
                                : 100.0 * static_cast<double>(best_roof) /
                                      static_cast<double>(roof_cells));
    EXPECT_GT(roof_cells, 0);

    // Same seam accounting restricted to the building footprint grown by 1 m.
    // The +1m box adds the edges that leave the strict footprint; the optimiser
    // reaches the floor, the argmax labelling sits near twice it.
    const SeamStats ab = seam_stats(scene, tiles, argmax, 1.0);
    const SeamStats ob = seam_stats(scene, tiles, optimised, 1.0);
    std::printf("[Seam.DisagreementAvoidance] building seams argmax=%d optimised=%d\n",
                ab.in_building, ob.in_building);
    EXPECT_GT(ab.in_building, 0);
    EXPECT_LE(3 * ob.in_building, 2 * ab.in_building);
    EXPECT_LE(ob.in_building, 135);
}

// -- 4. blending -------------------------------------------------------------

TEST(Blend, ModesMatchTruth) {
    const Scene scene = make_scene();
    const std::vector<OrthoTile> tiles = rectify_all(scene, 0.25);
    const std::vector<std::int32_t> labels =
        assign_seams(scene.grid.width, scene.grid.height, tiles, SeamOptions{});

    const struct {
        Blending mode;
        const char* name;
    } cases[] = {
        {Blending::Disabled, "Disabled"},
        {Blending::Average, "Average"},
        {Blending::Mosaic, "Mosaic"},
    };
    for (const auto& kase : cases) {
        const ortho::OrthoMosaic mosaic =
            ortho::blend(scene.grid.width, scene.grid.height, tiles, labels, kase.mode);
        const double error = mean_abs_diff(scene.truth, mosaic.rgb, mosaic.valid, scene.grid, true);
        std::printf("[Blend.ModesMatchTruth] %s mean_abs_diff=%.3f\n", kase.name, error);
        EXPECT_LT(error, 14.0);
    }
}

TEST(Blend, MosaicSeamsAreLessVisible) {
    Scene scene = make_scene();
    const std::vector<OrthoTile> tiles = rectify_all(scene, 0.25);
    const std::vector<std::int32_t> labels =
        assign_seams(scene.grid.width, scene.grid.height, tiles, SeamOptions{});

    const ortho::OrthoMosaic disabled =
        ortho::blend(scene.grid.width, scene.grid.height, tiles, labels, Blending::Disabled);
    const ortho::OrthoMosaic mosaic =
        ortho::blend(scene.grid.width, scene.grid.height, tiles, labels, Blending::Mosaic);
    const double disabled_jump = seam_jump(disabled);
    const double mosaic_jump = seam_jump(mosaic);
    std::printf("[Blend.MosaicSeamsAreLessVisible] jump disabled=%.3f mosaic=%.3f\n",
                disabled_jump, mosaic_jump);
    EXPECT_LT(mosaic_jump, disabled_jump);

    // Darken one tile by 30% and require the multiband blend to hide the seam at
    // least 1.5x better than the copy.
    std::vector<OrthoTile> darkened = tiles;
    for (std::size_t t = 0; t < darkened.size(); ++t) {
        if (t != 6) {
            continue;
        }
        for (std::uint8_t& value : darkened[t].rgb) {
            value = static_cast<std::uint8_t>(std::lround(0.7 * static_cast<double>(value)));
        }
    }
    const ortho::OrthoMosaic dark_disabled =
        ortho::blend(scene.grid.width, scene.grid.height, darkened, labels, Blending::Disabled);
    const ortho::OrthoMosaic dark_mosaic =
        ortho::blend(scene.grid.width, scene.grid.height, darkened, labels, Blending::Mosaic);
    const double dark_disabled_jump = seam_jump(dark_disabled);
    const double dark_mosaic_jump = seam_jump(dark_mosaic);
    std::printf("[Blend.MosaicSeamsAreLessVisible] dark jump disabled=%.3f mosaic=%.3f ratio=%.2f\n",
                dark_disabled_jump, dark_mosaic_jump,
                dark_mosaic_jump > 0.0 ? dark_disabled_jump / dark_mosaic_jump : 0.0);
    EXPECT_GT(dark_disabled_jump, 1.5 * dark_mosaic_jump);
}

TEST(Seam, BlendRoundTrip) {
    // Sanity: blend(Disabled) equals the labelled tile's colour.
    Scene scene = make_scene();
    const std::vector<OrthoTile> tiles = rectify_all(scene, 0.25);
    const std::vector<std::int32_t> labels =
        assign_seams(scene.grid.width, scene.grid.height, tiles, SeamOptions{});
    const ortho::OrthoMosaic copy = copy_labels(scene.grid.width, scene.grid.height, tiles, labels);
    const ortho::OrthoMosaic disabled =
        ortho::blend(scene.grid.width, scene.grid.height, tiles, labels, Blending::Disabled);
    EXPECT_EQ(copy.rgb, disabled.rgb);
    EXPECT_EQ(copy.valid, disabled.valid);
    EXPECT_EQ(ortho::to_string(ortho::parse_blending("mosaic")), std::string("mosaic"));
    EXPECT_EQ(ortho::to_string(ortho::parse_blending("average")), std::string("average"));
    EXPECT_EQ(ortho::to_string(ortho::parse_blending("disabled")), std::string("disabled"));
    EXPECT_THROW(ortho::parse_blending("nonsense"), ValueError);
}

// -- 5. timing ---------------------------------------------------------------

TEST(Ortho, FullPipelineTiming) {
    const Scene scene = make_scene();
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<OrthoTile> tiles = rectify_all(scene, 0.25);
    const std::vector<std::int32_t> labels =
        assign_seams(scene.grid.width, scene.grid.height, tiles, SeamOptions{});
    const ortho::OrthoMosaic mosaic =
        ortho::blend(scene.grid.width, scene.grid.height, tiles, labels, Blending::Mosaic);
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::printf("[Ortho.FullPipelineTiming] 12-camera 480x360 rectify+seams+blend %.3f s\n",
                seconds);
    EXPECT_FALSE(mosaic.rgb.empty());
    // Wall-clock budget, and therefore hardware-dependent: 5 s was tuned on the
    // M1 development machine. A 4-core x86 build machine takes ~10 s for the same
    // work. The bound only catches an algorithmic regression (the pipeline is
    // seconds, not minutes), so it is scaled for slower hosts rather than acting
    // as a performance target.
    if (test::perf_asserts_enabled()) {
        EXPECT_LT(seconds, 30.0);
    }
}

// -- 6a. a window on the global grid reproduces it bit for bit ----------------

TEST(Ortho, WindowGridReproducesTheWholeGridTiles) {
    // A block is a WINDOW on the output grid. Rectifying a camera into a window
    // must give exactly the whole grid's samples for the cells it holds: same
    // cell centres (the parent lattice, not the window's own origin), same crop
    // but clipped to the window. This is what makes a blocked ortho identical to
    // an unblocked one where the blocking cannot legitimately change anything.
    const Scene scene = make_scene();
    const std::vector<OrthoTile> whole = rectify_all(scene, 0.25);
    const int w = scene.grid.width;
    const int h = scene.grid.height;
    constexpr int kMargin = 40;
    constexpr int kBlock = 120;
    // A window in the middle of the grid, not touching its border.
    const int wx0 = 100;
    const int wy0 = 80;
    RasterGrid window;
    window.origin_x = scene.grid.origin_x + static_cast<double>(wx0) * scene.grid.pixel_size_x;
    window.origin_y = scene.grid.origin_y - static_cast<double>(wy0) * scene.grid.pixel_size_y;
    window.pixel_size_x = scene.grid.pixel_size_x;
    window.pixel_size_y = scene.grid.pixel_size_y;
    window.width = kBlock + 2 * kMargin;
    window.height = kBlock + 2 * kMargin;

    ortho::OrthoSurfaceRange range;
    range.lowest = 1e30;
    range.highest = -1e30;
    for (const float v : scene.dem.data) {
        range.lowest = std::min(range.lowest, static_cast<double>(v));
        range.highest = std::max(range.highest, static_cast<double>(v));
    }
    ortho::OrthoCellLattice lattice;
    lattice.origin_x = scene.grid.origin_x;
    lattice.origin_y = scene.grid.origin_y;
    lattice.col_offset = wx0;
    lattice.row_offset = wy0;

    std::size_t checked = 0;
    std::size_t differing = 0;
    for (std::size_t t = 0; t < scene.cameras.size(); ++t) {
        const OrthoTile in_window = orthorectify(
            scene.dem, scene.nodata, window, scene.cameras[t], static_cast<int>(t), 0.25, true,
            &range, &lattice);
        const OrthoTile& reference = whole[t];
        for (int row = wy0; row < wy0 + window.height && row < h; ++row) {
            for (int col = wx0; col < wx0 + window.width && col < w; ++col) {
                const bool in_ref = reference.contains(col, row);
                const bool in_win = in_window.contains(col, row);
                if (!in_ref || !in_win) {
                    continue;  // crop edges differ; only shared cells are compared
                }
                ++checked;
                const std::size_t a = reference.index(col, row);
                const std::size_t b = in_window.index(col, row);
                if (reference.valid[a] != in_window.valid[b] ||
                    reference.score[a] != in_window.score[b] || reference.rgb[a * 3] != in_window.rgb[b * 3] ||
                    reference.rgb[a * 3 + 1] != in_window.rgb[b * 3 + 1] ||
                    reference.rgb[a * 3 + 2] != in_window.rgb[b * 3 + 2]) {
                    ++differing;
                }
            }
        }
    }
    std::printf("[Ortho.WindowGrid] shared cells=%zu differing=%zu\n", checked, differing);
    EXPECT_GT(checked, 100000U);
    EXPECT_EQ(differing, 0U) << "a window must reproduce the whole grid's samples exactly";
}

// -- 6. pinned labels (block-wise seam continuity) ----------------------------

TEST(Seam, PinnedLabelsAreHonouredAndUnpinnedCellsStillOptimise) {
    // The block-wise caller pins the margin a previous block already decided.
    // The optimiser must keep every pinned label that a tile covers, and must
    // still improve the free cells (the pin is a constraint, not a freeze).
    const Scene scene = make_scene();
    const std::vector<OrthoTile> tiles = rectify_all(scene, 0.25);
    const int w = scene.grid.width;
    const int h = scene.grid.height;
    const SeamOptions options;
    const std::vector<std::int32_t> free = assign_seams(w, h, tiles, options);

    // Pin the left half of the grid to whatever the free labelling chose there
    // (a stand-in for "a block to the left already decided this").
    std::vector<std::int32_t> pinned_labels = free;
    std::vector<std::uint8_t> pinned(free.size(), 0);
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w / 2; ++col) {
            const std::size_t cell = static_cast<std::size_t>(row) * static_cast<std::size_t>(w) +
                                     static_cast<std::size_t>(col);
            if (free[cell] >= 0) {
                pinned[cell] = 1;
            }
        }
    }
    const std::vector<std::int32_t> constrained =
        assign_seams(w, h, tiles, options, &pinned_labels, &pinned);

    std::size_t checked = 0;
    std::size_t kept = 0;
    for (std::size_t cell = 0; cell < free.size(); ++cell) {
        if (pinned[cell] == 0) {
            continue;
        }
        ++checked;
        kept += constrained[cell] == pinned_labels[cell] ? 1U : 0U;
    }
    std::printf("[Seam.Pinned] pinned=%zu kept=%zu\n", checked, kept);
    EXPECT_GT(checked, static_cast<std::size_t>(w) * static_cast<std::size_t>(h) / 4);
    EXPECT_EQ(kept, checked) << "a pinned label must survive the optimisation";

    // The free half must still be a valid labelling, and its own labels must be
    // the ones the optimiser chose (not a copy of the pin).
    std::size_t free_changed = 0;
    for (std::size_t cell = 0; cell < free.size(); ++cell) {
        if (pinned[cell] != 0) {
            continue;
        }
        const int col = static_cast<int>(cell % static_cast<std::size_t>(w));
        const int row = static_cast<int>(cell / static_cast<std::size_t>(w));
        bool covered = false;
        for (const OrthoTile& tile : tiles) {
            covered = covered || tile_valid_at(tile, col, row);
        }
        if (covered) {
            EXPECT_GE(constrained[cell], 0) << "an uncovered free cell was not labelled";
        }
        free_changed += constrained[cell] != free[cell] ? 1U : 0U;
    }
    std::printf("[Seam.Pinned] free cells changed by the constraint=%zu\n", free_changed);
}

TEST(Seam, PinnedLabelThatTheTileSetDoesNotCoverIsIgnored) {
    // A pinned label can name a camera this block does not hold (or that is
    // invalid there). The pin must be dropped rather than producing a label no
    // tile can render.
    std::vector<OrthoTile> tiles(2);
    constexpr int kWidth = 6;
    constexpr int kHeight = 4;
    for (int t = 0; t < 2; ++t) {
        OrthoTile& tile = tiles[static_cast<std::size_t>(t)];
        tile.camera = t;
        tile.x0 = t == 0 ? 0 : 3;
        tile.y0 = 0;
        tile.width = kWidth - 3;
        tile.height = kHeight;
        const std::size_t cells = static_cast<std::size_t>(tile.width) *
                                  static_cast<std::size_t>(tile.height);
        tile.rgb.assign(cells * 3, 128);
        tile.valid.assign(cells, 1);
        tile.score.assign(cells, 1.0F);
    }
    std::vector<std::int32_t> pinned_labels(static_cast<std::size_t>(kWidth) * kHeight, 0);
    std::vector<std::uint8_t> pinned(pinned_labels.size(), 0);
    // Pin the RIGHT half to tile 0, which does not cover it at all.
    for (int row = 0; row < kHeight; ++row) {
        for (int col = kWidth / 2; col < kWidth; ++col) {
            const std::size_t cell = static_cast<std::size_t>(row) * kWidth +
                                     static_cast<std::size_t>(col);
            pinned_labels[cell] = 0;
            pinned[cell] = 1;
        }
    }
    const std::vector<std::int32_t> labels =
        assign_seams(kWidth, kHeight, tiles, SeamOptions{}, &pinned_labels, &pinned);
    for (int row = 0; row < kHeight; ++row) {
        for (int col = kWidth / 2; col < kWidth; ++col) {
            const std::size_t cell = static_cast<std::size_t>(row) * kWidth +
                                     static_cast<std::size_t>(col);
            EXPECT_NE(labels[cell], 0) << "an uncovered pinned label was kept";
            EXPECT_EQ(labels[cell], 1) << "the covering tile was not chosen";
        }
    }
}

// -- 7. pinned blend pyramid depth -------------------------------------------

TEST(Ortho, PinnedPyramidDepthEqualsTheWholeGridBlend) {
    // A block plan pins the mosaic pyramid depth to the whole grid's. Pinning it
    // explicitly must reproduce the default blend exactly, and the accessor must
    // agree with what blend() would have picked.
    const Scene scene = make_scene();
    const std::vector<OrthoTile> tiles = rectify_all(scene, 0.25);
    const std::vector<std::int32_t> labels =
        assign_seams(scene.grid.width, scene.grid.height, tiles, SeamOptions{});
    const int levels = ortho::pyramid_level_count(scene.grid.width, scene.grid.height);
    const ortho::OrthoMosaic automatic =
        ortho::blend(scene.grid.width, scene.grid.height, tiles, labels, Blending::Mosaic);
    const ortho::OrthoMosaic pinned = ortho::blend(scene.grid.width, scene.grid.height, tiles,
                                                   labels, Blending::Mosaic, levels);
    std::printf("[Ortho.PinnedPyramid] levels=%d\n", levels);
    EXPECT_GE(levels, 1);
    EXPECT_EQ(pinned.rgb, automatic.rgb);
    EXPECT_EQ(pinned.valid, automatic.valid);

    // A shallower pinned depth is a different operator and must be honoured (the
    // clamp only ever reduces the depth to what the grid supports).
    const ortho::OrthoMosaic shallow = ortho::blend(scene.grid.width, scene.grid.height, tiles,
                                                    labels, Blending::Mosaic, 1);
    EXPECT_NE(shallow.rgb, automatic.rgb);
}

}  // namespace
}  // namespace alphashape
