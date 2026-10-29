// test_build_dem.cpp -- tests for the BuildDem / ExportDem tasks (milestone 9).
//
// Exact-input tests run on an unreferenced chunk in the local frame:
//   * the point-cloud source against the analytic truth_height surface,
//   * the model source rasterising an exact tilted plane,
//   * the interpolation modes on a point cloud with known holes,
//   * the automatic-resolution rule.
//
// Pipeline test on the georeferenced aerial fixture: truth poses -> depth maps
// -> BuildDem, measured against truth_height through the UTM plane frame.

#include "aerial_fixture.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "common/algorithm/elevation/build_dem_task.hpp"
#include "common/algorithm/elevation/dem_builder.hpp"
#include "common/algorithm/elevation/export_dem_task.hpp"
#include "common/algorithm/elevation/surface_frame.hpp"
#include "common/core/document.hpp"
#include "common/tasks/registry.hpp"
#include "model/camera/calibration.hpp"
#include "model/geo/geodesy.hpp"
#include "model/raster/geotiff.hpp"

namespace alphashape {
namespace {

using elevation::DemOptions;
using elevation::HoleFill;

/// AddPhotos-free chunk: one camera-less, sensor-less chunk in the local frame.
Document make_local_chunk_document() {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    return document;
}

Params dem_params(const std::string& source = "point_cloud",
                  const std::string& interpolation = "enabled", double resolution = 0.0,
                  std::int64_t projection = 0, const std::string& fill_method = "natural_neighbour",
                  const std::string& smoothing = "tgv") {
    Params params = make_params(builtin_registry().require("BuildDem"));
    params.set("source_data", ParamValue{source});
    params.set("interpolation", ParamValue{interpolation});
    params.set("resolution", ParamValue{resolution});
    params.set("projection", ParamValue{projection});
    params.set("fill_method", ParamValue{fill_method});
    params.set("smoothing", ParamValue{smoothing});
    return params;
}

void run_build_dem(Document& document, Params& params) {
    run_task(builtin_registry(), document, "BuildDem", params);
}

/// A point cloud sampled on a 0.1 m lattice from the analytic truth_height
/// surface over the requested rectangle (z stored in the point's z component;
/// the chunk-local frame is the plane frame because the chunk is unreferenced).
void fill_truth_point_cloud(Chunk& chunk, double x0, double x1, double y0, double y1,
                            double spacing) {
    PointCloud& cloud = chunk.point_cloud();
    cloud.points.clear();
    for (double x = x0; x <= x1 + 1e-9; x += spacing) {
        for (double y = y0; y <= y1 + 1e-9; y += spacing) {
            cloud.points.push_back(
                Eigen::Vector3d(x, y, aerial_fixture::truth_height(x, y)));
        }
    }
}

/// Linear-interpolated percentile of an ascending-sorted vector.
double density_percentile(const std::vector<double>& sorted, double fraction) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double idx = fraction * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double f = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - f) + sorted[hi] * f;
}

/// Interpolated truth height at a grid cell centre.
double truth_at_cell(const RasterGrid& grid, int col, int row) {
    const Eigen::Vector2d c = grid.pixel_center(col, row);
    return aerial_fixture::truth_height(c.x(), c.y());
}

/// True when a cell holds the DEM's nodata sentinel (the builder writes
/// `nodata`, not NaN, for unmeasured cells).
bool is_nodata(const Elevation& dem, int col, int row) {
    return dem.raster.at(col, row) == dem.nodata;
}

/// True when a cell centre sits within `margin` of a building/ramp edge, where
/// the analytic height is discontinuous and the median rule cannot be exact.
bool near_scene_edge(double x, double y, double margin) {
    const double bx0 = -20.0, bx1 = -4.0, by0 = 5.0, by1 = 17.0;
    const double rx0 = 0.0, rx1 = 20.0, ry0 = 15.0, ry1 = 30.0;
    const bool near_box = x >= bx0 - margin && x <= bx1 + margin && y >= by0 - margin &&
                          y <= by1 + margin;
    const bool near_ramp = x >= rx0 - margin && x <= rx1 + margin && y >= ry0 - margin &&
                           y <= ry1 + margin;
    return near_box || near_ramp;
}

// ---------------------------------------------------------------------------
// 1. Point cloud against the analytic truth surface (local frame)
// ---------------------------------------------------------------------------

TEST(Dem, PointCloudMatchesTruthHeight) {
    Document document = make_local_chunk_document();
    Chunk& chunk = *document.active_chunk();
    fill_truth_point_cloud(chunk, -50.0, 50.0, -40.0, 40.0, 0.1);

    Params params = dem_params("point_cloud", "enabled", 0.25);
    // The exact-input test measures the interpolation + TGV surface against
    // truth, so request no smoothing: TGV's job is robustness on noisy stereo,
    // not exactness on an already-exact point cloud.
    params.set("smoothing", ParamValue{std::string("none")});
    run_build_dem(document, params);

    const Elevation& dem = chunk.elevation();
    ASSERT_FALSE(dem.empty());
    EXPECT_EQ(dem.source, "point_cloud");
    EXPECT_NEAR(dem.raster.grid.pixel_size_x, 0.25, 1e-9);

    std::vector<double> errors;
    for (int row = 0; row < dem.raster.grid.height; ++row) {
        for (int col = 0; col < dem.raster.grid.width; ++col) {
            const Eigen::Vector2d c = dem.raster.grid.pixel_center(col, row);
            if (near_scene_edge(c.x(), c.y(), 1.0)) {
                continue;
            }
            if (is_nodata(dem, col, row)) {
                continue;
            }
            errors.push_back(std::abs(static_cast<double>(dem.raster.at(col, row)) -
                                      truth_at_cell(dem.raster.grid, col, row)));
        }
    }
    std::sort(errors.begin(), errors.end());
    const double median = density_percentile(errors, 0.50);
    const double p99 = density_percentile(errors, 0.99);
    std::printf("[Dem.PointCloudMatchesTruthHeight] cells=%zu median=%.5f p99=%.5f\n",
                errors.size(), median, p99);
    EXPECT_GT(errors.size(), 100000U);
    EXPECT_LT(median, 0.01);
    EXPECT_LT(p99, 0.05);
}

// ---------------------------------------------------------------------------
// 2. Model source rasterises an exact tilted plane
// ---------------------------------------------------------------------------

/// A square [-half, half]^2 carrying z = 0.1 x + 0.05 y + 3, split into two
/// triangles.
Mesh tilted_plane_mesh(double half) {
    Mesh mesh;
    const double z0 = 3.0;
    mesh.vertices = {
        Eigen::Vector3d(-half, -half, z0 + 0.1 * -half + 0.05 * -half),
        Eigen::Vector3d(half, -half, z0 + 0.1 * half + 0.05 * -half),
        Eigen::Vector3d(half, half, z0 + 0.1 * half + 0.05 * half),
        Eigen::Vector3d(-half, half, z0 + 0.1 * -half + 0.05 * half)};
    mesh.faces = {Eigen::Vector3i(0, 1, 2), Eigen::Vector3i(0, 2, 3)};
    return mesh;
}

TEST(Dem, ModelRasterisesTiltedPlaneExactly) {
    Document document = make_local_chunk_document();
    Chunk& chunk = *document.active_chunk();
    chunk.model() = tilted_plane_mesh(5.0);

    Params params = dem_params("model", "disabled", 0.317, 0, "nearest", "none");
    run_build_dem(document, params);

    const Elevation& dem = chunk.elevation();
    ASSERT_FALSE(dem.empty());
    EXPECT_EQ(dem.source, "model");

    std::size_t covered = 0;
    double worst = 0.0;
    for (int row = 0; row < dem.raster.grid.height; ++row) {
        for (int col = 0; col < dem.raster.grid.width; ++col) {
            if (is_nodata(dem, col, row)) {
                continue;
            }
            const double value = static_cast<double>(dem.raster.at(col, row));
            const Eigen::Vector2d c = dem.raster.grid.pixel_center(col, row);
            const double expected = 0.1 * c.x() + 0.05 * c.y() + 3.0;
            worst = std::max(worst, std::abs(value - expected));
            ++covered;
        }
    }
    std::printf("[Dem.ModelRasterisesTiltedPlaneExactly] covered=%zu worst=%.2e\n", covered,
                worst);
    EXPECT_GT(covered, 0U);
    EXPECT_LT(worst, 1e-4);
}

// ---------------------------------------------------------------------------
// 3. Interpolation modes
// ---------------------------------------------------------------------------

TEST(Dem, InterpolationModesFillAsDocumented) {
    // A plane point cloud on the cell-centre lattice (0.5 m spacing, offset half
    // a cell so every cell gets exactly one sample). Two holes are punched: an
    // interior disc at (-10, 0) (surrounded by samples, so "interior"), and a
    // full-height band over 0 < x < 12 (which touches the raster's north and
    // south borders through the invalid rows, so it is NOT interior).
    const auto make_holed_cloud = [](Chunk& chunk) {
        PointCloud& cloud = chunk.point_cloud();
        cloud.points.clear();
        for (int ix = -40; ix < 40; ++ix) {
            for (int iy = -40; iy < 40; ++iy) {
                const double x = 0.5 * static_cast<double>(ix) + 0.25;
                const double y = 0.5 * static_cast<double>(iy) + 0.25;
                const double dx = x + 10.0, dy = y;
                if (std::sqrt(dx * dx + dy * dy) < 4.0) {
                    continue;  // interior disc
                }
                if (x > 0.0 && x < 12.0) {
                    continue;  // full-height band: a border-connected hole
                }
                cloud.points.push_back(Eigen::Vector3d(x, y, 10.0 + 0.01 * x));
            }
        }
    };

    struct Counts {
        std::size_t disc_nodata = 0;
        std::size_t strip_nodata = 0;
    };
    // The interior disc is centred (5, 5) with radius 4; the border strip is the
    // east third (x > 10). A cell is "disc" when its centre is inside the disc
    // and "strip" when it lies beyond x = 14 (well inside the removed strip).
    const auto count = [](const Elevation& dem) {
        Counts counts;
        for (int row = 0; row < dem.raster.grid.height; ++row) {
            for (int col = 0; col < dem.raster.grid.width; ++col) {
                if (!is_nodata(dem, col, row)) {
                    continue;
                }
                const Eigen::Vector2d c = dem.raster.grid.pixel_center(col, row);
                const double dx = c.x() + 10.0, dy = c.y();
                if (std::sqrt(dx * dx + dy * dy) < 3.5) {
                    ++counts.disc_nodata;
                }
                if (c.x() > 1.0 && c.x() < 11.0) {
                    ++counts.strip_nodata;
                }
            }
        }
        return counts;
    };

    Document disabled_doc = make_local_chunk_document();
    make_holed_cloud(*disabled_doc.active_chunk());
    Params p1 = dem_params("point_cloud", "disabled", 0.5, 0, "nearest", "none");
    run_build_dem(disabled_doc, p1);
    const Counts disabled = count(disabled_doc.active_chunk()->elevation());

    Document enabled_doc = make_local_chunk_document();
    make_holed_cloud(*enabled_doc.active_chunk());
    Params p2 = dem_params("point_cloud", "enabled", 0.5, 0, "nearest", "none");
    run_build_dem(enabled_doc, p2);
    const Counts enabled = count(enabled_doc.active_chunk()->elevation());

    Document extrapolated_doc = make_local_chunk_document();
    make_holed_cloud(*extrapolated_doc.active_chunk());
    Params p3 = dem_params("point_cloud", "extrapolated", 0.5, 0, "nearest", "none");
    run_build_dem(extrapolated_doc, p3);
    const Counts extrapolated = count(extrapolated_doc.active_chunk()->elevation());

    std::printf("[Dem.InterpolationModesFillAsDocumented] disc d/e/x=%zu/%zu/%zu "
                "strip d/e/x=%zu/%zu/%zu\n",
                disabled.disc_nodata, enabled.disc_nodata, extrapolated.disc_nodata,
                disabled.strip_nodata, enabled.strip_nodata, extrapolated.strip_nodata);

    // disabled: both holes remain nodata.
    EXPECT_GT(disabled.disc_nodata, 0U);
    EXPECT_GT(disabled.strip_nodata, 0U);
    // enabled: this chunk has NO cameras, so there is no footprint region and the
    // builder falls back to the interior-only rule: the disc is filled, the
    // border strip is not. (With cameras, "enabled" would fill the strip too --
    // see Dem.EnabledFillsHolesInsideTheCameraFootprint.)
    EXPECT_EQ(enabled.disc_nodata, 0U);
    EXPECT_GT(enabled.strip_nodata, 0U);
    // extrapolated: both are filled.
    EXPECT_EQ(extrapolated.disc_nodata, 0U);
    EXPECT_EQ(extrapolated.strip_nodata, 0U);
}

// ---------------------------------------------------------------------------
// 4. Automatic resolution
// ---------------------------------------------------------------------------

TEST(Dem, AutoResolutionIsPositiveAndNearSampleSpacing) {
    Document document = make_local_chunk_document();
    Chunk& chunk = *document.active_chunk();
    fill_truth_point_cloud(chunk, -10.0, 10.0, -10.0, 10.0, 0.5);

    Params params = dem_params("point_cloud", "enabled", 0.0, 0, "nearest", "none");
    run_build_dem(document, params);

    const double resolution = chunk.elevation().raster.grid.pixel_size_x;
    std::printf("[Dem.AutoResolutionIsPositiveAndNearSampleSpacing] resolution=%.4f\n",
                resolution);
    EXPECT_GT(resolution, 0.0);
    // sqrt(area / N) over a 0.5 m lattice is 0.5, so the automatic value must be
    // within a factor of two of the sample spacing.
    EXPECT_GT(resolution, 0.25);
    EXPECT_LT(resolution, 1.0);
}

// ---------------------------------------------------------------------------
// 4b. Extent robustness: a few far outliers cannot explode the grid
// ---------------------------------------------------------------------------
//
// The owner's 8 GB laptop hit "bad allocation" on a 36-photo block at the OLD
// 0.25 m ortho default, when the ortho had only ~0.2 M cells. The grid follows
// the DEM, the DEM's extent followed raw sample min/max, and a handful of far
// depth-map samples turned a 100 m block into a kilometres-wide grid. These
// tests pin the robust bound: outliers must not change the extent materially,
// while an outlier-free scene must keep the exact grid it always had.

TEST(Dem, FarOutliersDoNotExplodeTheExtent) {
    Document document = make_local_chunk_document();
    Chunk& chunk = *document.active_chunk();
    fill_truth_point_cloud(chunk, -50.0, 50.0, -40.0, 40.0, 0.25);
    const std::size_t clean_count = chunk.point_cloud().points.size();

    // Baseline: the exact extent of the clean lattice.
    Params baseline = dem_params("point_cloud", "enabled", 0.25, 0, "nearest", "none");
    run_build_dem(document, baseline);
    const RasterGrid clean_grid = chunk.elevation().raster.grid;
    const double clean_x0 = clean_grid.origin_x;
    const double clean_y0 = clean_grid.origin_y;

    // Now inject 20 far points: 10 "at infinity" along a grazing ray and 10
    // spurious close-range mismatches. 20 of ~128000 samples is 0.016%, inside
    // the 0.5% discarded tail on each side.
    PointCloud& cloud = chunk.point_cloud();
    for (int i = 0; i < 10; ++i) {
        const double angle = 2.0 * 3.14159265358979 * static_cast<double>(i) / 10.0;
        cloud.points.push_back(Eigen::Vector3d(
            5000.0 * std::cos(angle), 5000.0 * std::sin(angle),
            aerial_fixture::truth_height(0.0, 0.0)));
    }
    for (int i = 0; i < 10; ++i) {
        cloud.points.push_back(Eigen::Vector3d(
            -3000.0 + 7.0 * static_cast<double>(i), -2500.0, 0.0));
    }
    ASSERT_GT(cloud.points.size(), clean_count);

    Params dirty = dem_params("point_cloud", "enabled", 0.25, 0, "nearest", "none");
    run_build_dem(document, dirty);
    const RasterGrid dirty_grid = chunk.elevation().raster.grid;

    const double clean_east =
        clean_grid.origin_x + static_cast<double>(clean_grid.width) * clean_grid.pixel_size_x;
    const double clean_south =
        clean_grid.origin_y - static_cast<double>(clean_grid.height) * clean_grid.pixel_size_y;
    const double dirty_east =
        dirty_grid.origin_x + static_cast<double>(dirty_grid.width) * dirty_grid.pixel_size_x;
    const double dirty_south =
        dirty_grid.origin_y - static_cast<double>(dirty_grid.height) * dirty_grid.pixel_size_y;
    std::printf("[Dem.FarOutliersDoNotExplodeTheExtent] clean=%dx%d dirty=%dx%d "
                "dx=[%.2f,%.2f] dy=[%.2f,%.2f]\n",
                clean_grid.width, clean_grid.height, dirty_grid.width, dirty_grid.height,
                dirty_grid.origin_x - clean_x0, dirty_east - clean_east,
                dirty_grid.origin_y - clean_y0, dirty_south - clean_south);

    // The outliers sit kilometres away; without the robust bound the grid would
    // be thousands of cells wider. A 6 m tolerance allows the 3% core margin
    // (3 m on a 100 m block) plus the origin snapping to the 0.25 m cell.
    constexpr double kExtentTolerance = 6.0;
    EXPECT_LT(std::abs(dirty_grid.origin_x - clean_x0), kExtentTolerance);
    EXPECT_LT(std::abs(dirty_east - clean_east), kExtentTolerance);
    EXPECT_LT(std::abs(dirty_grid.origin_y - clean_y0), kExtentTolerance);
    EXPECT_LT(std::abs(dirty_south - clean_south), kExtentTolerance);
    // And the cell count is the same order, not 100x.
    const double clean_cells =
        static_cast<double>(clean_grid.width) * static_cast<double>(clean_grid.height);
    const double dirty_cells =
        static_cast<double>(dirty_grid.width) * static_cast<double>(dirty_grid.height);
    EXPECT_LT(dirty_cells, 1.5 * clean_cells);
}

TEST(Dem, CleanCloudKeepsTheExactExtent) {
    // The robust bound can only shrink the exact box, never grow it: an
    // outlier-free cloud must produce byte-comparable grid metadata to the
    // pre-robust code (origin exactly the outward-snapped minimum).
    Document document = make_local_chunk_document();
    Chunk& chunk = *document.active_chunk();
    fill_truth_point_cloud(chunk, -10.0, 10.0, -10.0, 10.0, 0.5);
    Params params = dem_params("point_cloud", "enabled", 0.5, 0, "nearest", "none");
    run_build_dem(document, params);

    const RasterGrid& grid = chunk.elevation().raster.grid;
    // origin_x = floor(-10 / 0.5) * 0.5 = -10, origin_y = ceil(10/0.5)*0.5 = 10,
    // and 20 m / 0.5 m = 40 cells each way.
    EXPECT_NEAR(grid.origin_x, -10.0, 1e-9);
    EXPECT_NEAR(grid.origin_y, 10.0, 1e-9);
    EXPECT_EQ(grid.width, 40);
    EXPECT_EQ(grid.height, 40);
}

// ---------------------------------------------------------------------------
// 4c. "enabled" fills what the cameras see, not only enclosed holes
// ---------------------------------------------------------------------------
//
// The aerial block's DEM is ~1/3 nodata, and most of the holes inside the flown
// area touch the nodata RIM outside the cameras' cover. Filling only holes
// enclosed by data ("interior_only") therefore fills almost nothing. The rule is
// now the union of the cameras' ground footprints: inside it, a hole is filled
// even when it reaches the raster border; outside it, nothing is invented.

namespace {

/// A nadir pinhole camera at `height` metres over local (x, y) using sensor 0.
void add_nadir_camera(Chunk& chunk, double x, double y, double height) {
    Camera camera;
    camera.id = static_cast<std::int32_t>(chunk.cameras().size());
    camera.sensor_id = 0;
    camera.enabled = true;
    // World -> camera: look straight down (camera +z = world -z), optical
    // centre (x, y, height). R is its own inverse, so t = -R * centre.
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    R(0, 0) = 1.0;
    R(1, 1) = -1.0;
    R(2, 2) = -1.0;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3, 3>(0, 0) = R;
    pose.block<3, 1>(0, 3) = -R * Eigen::Vector3d(x, y, height);
    camera.pose = pose;
    chunk.cameras().push_back(camera);
}

/// A point cloud over [-60, 60]^2 on the 0.5 m cell lattice (so every 0.5 m DEM
/// cell holds exactly one sample), with every sample removed for `hole(x)` true.
/// The removed columns form a full-height invalid band that touches the raster's
/// north and south borders -- the "border-connected but inside the survey" case
/// the footprint region exists for.
void fill_cloud_with_band(Chunk& chunk, const std::function<bool(double)>& hole) {
    PointCloud& cloud = chunk.point_cloud();
    cloud.points.clear();
    for (int ix = -120; ix <= 120; ++ix) {
        for (int iy = -120; iy <= 120; ++iy) {
            const double x = 0.5 * static_cast<double>(ix) + 0.25;
            const double y = 0.5 * static_cast<double>(iy) + 0.25;
            if (hole(x)) {
                continue;
            }
            cloud.points.push_back(Eigen::Vector3d(x, y, 10.0 + 0.01 * x));
        }
    }
}

/// Nodata cells of `dem` whose centre satisfies `predicate`, out of the
/// matching cells in total.
std::pair<std::size_t, std::size_t> count_nodata(
    const Elevation& dem, const std::function<bool(double, double)>& predicate) {
    std::size_t nodata = 0;
    std::size_t total = 0;
    for (int row = 0; row < dem.raster.grid.height; ++row) {
        for (int col = 0; col < dem.raster.grid.width; ++col) {
            const Eigen::Vector2d c = dem.raster.grid.pixel_center(col, row);
            if (!predicate(c.x(), c.y())) {
                continue;
            }
            ++total;
            if (is_nodata(dem, col, row)) {
                ++nodata;
            }
        }
    }
    return {nodata, total};
}

}  // namespace

TEST(Dem, EnabledFillsHolesInsideTheCameraFootprint) {
    Document document = make_local_chunk_document();
    Chunk& chunk = *document.active_chunk();
    Sensor sensor;
    sensor.type = CameraType::Frame;
    sensor.calibration = Calibration::initial(CameraType::Frame, 2000, 2000, 2000.0);
    chunk.sensors().push_back(sensor);

    // Band of holes at 0 < x < 12, reaching the raster border top and bottom.
    fill_cloud_with_band(chunk, [](double x) { return x > 0.0 && x < 12.0; });
    // At 100 m with f = 2000 px the half-footprint is 50 m: the band is inside.
    add_nadir_camera(chunk, 0.0, 0.0, 100.0);

    Params params = dem_params("point_cloud", "enabled", 0.5, 0, "nearest", "none");
    run_build_dem(document, params);

    const auto band = count_nodata(chunk.elevation(), [](double x, double y) {
        return x > 1.0 && x < 11.0 && y > -40.0 && y < 40.0;
    });
    std::printf("[Dem.EnabledFillsHolesInsideTheCameraFootprint] band nodata=%zu/%zu\n",
                band.first, band.second);
    EXPECT_GT(band.second, 0U);
    // Inside the footprint the band is a hole the survey covers: filled, even
    // though it is border-connected.
    EXPECT_EQ(band.first, 0U);
}

TEST(Dem, EnabledKeepsTheConcaveGapAndTheOutsideNodata) {
    // Two cameras 60 m apart, each with a ~18 m half-footprint on the ground (90 m
    // of relief below a 100 m camera at f = 5000 px over 2000 px): the union is
    // [-48, -12] + [12, 48] with a 24 m gap no camera sees, though it lies inside
    // the convex hull of the two footprints.
    Document document = make_local_chunk_document();
    Chunk& chunk = *document.active_chunk();
    Sensor sensor;
    sensor.type = CameraType::Frame;
    sensor.calibration = Calibration::initial(CameraType::Frame, 2000, 2000, 5000.0);
    chunk.sensors().push_back(sensor);

    // One band of holes across the middle, spanning both footprints, the gap and
    // 5 m beyond the cameras' cover on each side, so "outside every footprint"
    // is exercised on hole cells too.
    fill_cloud_with_band(chunk, [](double x) { return x > -55.0 && x < 55.0; });
    add_nadir_camera(chunk, -30.0, 0.0, 100.0);
    add_nadir_camera(chunk, 30.0, 0.0, 100.0);

    Params params = dem_params("point_cloud", "enabled", 0.5, 0, "nearest", "none");
    run_build_dem(document, params);

    const Elevation& dem = chunk.elevation();
    // Inside a footprint's interior in BOTH axes: the footprints are 36 m squares
    // centred at x = -30 and x = 30, so y has to stay within (-15, 15).
    const auto west = count_nodata(dem, [](double x, double y) {
        return x > -40.0 && x < -15.0 && y > -15.0 && y < 15.0;
    });
    const auto gap = count_nodata(dem, [](double x, double y) {
        return x > -5.0 && x < 5.0 && y > -15.0 && y < 15.0;
    });
    const auto east = count_nodata(dem, [](double x, double y) {
        return x > 15.0 && x < 40.0 && y > -15.0 && y < 15.0;
    });
    const auto outside = count_nodata(dem, [](double x, double y) {
        return x > 50.5 && x < 54.5 && y > -15.0 && y < 15.0;
    });
    std::printf("[Dem.EnabledKeepsTheConcaveGapAndTheOutsideNodata] "
                "west=%zu/%zu gap=%zu/%zu east=%zu/%zu outside=%zu/%zu\n",
                west.first, west.second, gap.first, gap.second, east.first, east.second,
                outside.first, outside.second);

    EXPECT_GT(west.second, 0U);
    EXPECT_GT(gap.second, 0U);
    EXPECT_GT(east.second, 0U);
    EXPECT_GT(outside.second, 0U);
    // Under a camera: filled.
    EXPECT_EQ(west.first, 0U);
    EXPECT_EQ(east.first, 0U);
    // The gap is seen by neither camera: it stays nodata. A convex hull of the
    // two footprints would have filled it.
    EXPECT_EQ(gap.first, gap.second);
    // Beyond the last camera's cover: nodata, both because it is a hole and
    // because no footprint claims it.
    EXPECT_EQ(outside.first, outside.second);
}

TEST(Dem, MaxFillDistanceLeavesTheDeepHoleCentreNodata) {
    // A hole 30 cells wide (cell centres x = -7.25 .. 7.25 at 0.5 m), running the
    // full height, on a point cloud with a camera covering the block.
    // max_fill_distance = 2.5 m is 5 cells: the rim (1-5 cells from the data)
    // fills, the centre (6+ cells from it) stays nodata. Unlimited (0) fills
    // all of it.
    Document unlimited_doc = make_local_chunk_document();
    Document capped_doc = make_local_chunk_document();
    for (Document* doc : {&unlimited_doc, &capped_doc}) {
        Chunk& chunk = *doc->active_chunk();
        Sensor sensor;
        sensor.type = CameraType::Frame;
        sensor.calibration = Calibration::initial(CameraType::Frame, 2000, 2000, 2000.0);
        chunk.sensors().push_back(sensor);
        fill_cloud_with_band(chunk, [](double x) { return x > -7.5 && x < 7.5; });
        // ~50 m half-footprint at 100 m, f = 2000 px: the band is well inside.
        add_nadir_camera(chunk, 0.0, 0.0, 100.0);
    }

    Params unlimited = dem_params("point_cloud", "enabled", 0.5, 0, "nearest", "none");
    unlimited.set("max_fill_distance", ParamValue{0.0});
    run_build_dem(unlimited_doc, unlimited);

    Params capped = dem_params("point_cloud", "enabled", 0.5, 0, "nearest", "none");
    capped.set("max_fill_distance", ParamValue{2.5});  // 5 cells at 0.5 m
    run_build_dem(capped_doc, capped);

    // Band centre: the nearest cell centres here are 6 m (12 cells) from the
    // data columns at x = -7.75 and x = 7.75, so they are well beyond the cap.
    // The middle two cells are 7.5 m (15 cells) out either way.
    const auto centre_capped = count_nodata(capped_doc.active_chunk()->elevation(),
                                            [](double x, double y) {
        return std::abs(x) < 2.0 && std::abs(y) < 40.0;
    });
    const auto centre_unlimited = count_nodata(unlimited_doc.active_chunk()->elevation(),
                                               [](double x, double y) {
        return std::abs(x) < 2.0 && std::abs(y) < 40.0;
    });
    // Band rim: x in (-7, -5.5) holds the cell centres at -6.75, -6.25 and
    // -5.75, i.e. 2, 3 and 4 cells from the data column at x = -7.75 -- inside
    // the 5-cell cap, so they must fill.
    const auto rim_capped = count_nodata(capped_doc.active_chunk()->elevation(),
                                         [](double x, double y) {
        return x > -7.0 && x < -5.5 && std::abs(y) < 40.0;
    });
    std::printf("[Dem.MaxFillDistanceLeavesTheDeepHoleCentreNodata] "
                "capped centre=%zu/%zu rim=%zu/%zu unlimited centre=%zu/%zu\n",
                centre_capped.first, centre_capped.second, rim_capped.first, rim_capped.second,
                centre_unlimited.first, centre_unlimited.second);

    EXPECT_GT(centre_capped.second, 0U);
    EXPECT_GT(rim_capped.second, 0U);
    // The cap leaves the centre unfilled ...
    EXPECT_EQ(centre_capped.first, centre_capped.second);
    // ... and fills the rim.
    EXPECT_EQ(rim_capped.first, 0U);
    // Without the cap the whole hole fills.
    EXPECT_GT(centre_unlimited.second, 0U);
    EXPECT_EQ(centre_unlimited.first, 0U);
}

// ---------------------------------------------------------------------------
// 5. Errors
// ---------------------------------------------------------------------------

TEST(BuildDemErrors, UnknownSourceDataThrows) {
    Document document = make_local_chunk_document();
    Chunk& chunk = *document.active_chunk();
    fill_truth_point_cloud(chunk, -5.0, 5.0, -5.0, 5.0, 0.5);
    Params params = dem_params("bogus");
    EXPECT_THROW(run_build_dem(document, params), ValueError);
    EXPECT_TRUE(chunk.elevation().empty());
}

TEST(BuildDemErrors, EmptyChunkThrows) {
    Document document = make_local_chunk_document();
    Params params = dem_params("point_cloud");
    EXPECT_THROW(run_build_dem(document, params), ValueError);
}

TEST(BuildDemErrors, UnknownInterpolationThrows) {
    Document document = make_local_chunk_document();
    fill_truth_point_cloud(*document.active_chunk(), -5.0, 5.0, -5.0, 5.0, 0.5);
    Params params = dem_params("point_cloud", "bogus");
    EXPECT_THROW(run_build_dem(document, params), ValueError);
}

TEST(BuildDemErrors, EmptyModelSourceThrows) {
    Document document = make_local_chunk_document();
    Params params = dem_params("model");
    EXPECT_THROW(run_build_dem(document, params), ValueError);
}

TEST(BuildDemErrors, NegativeMaxFillDistanceThrows) {
    Document document = make_local_chunk_document();
    fill_truth_point_cloud(*document.active_chunk(), -5.0, 5.0, -5.0, 5.0, 0.5);
    Params params = dem_params("point_cloud", "enabled", 0.5, 0, "nearest", "none");
    params.set("max_fill_distance", ParamValue{-1.0});
    EXPECT_THROW(run_build_dem(document, params), ValueError);
}

// ---------------------------------------------------------------------------
// 6. Pipeline on the georeferenced aerial fixture
// ---------------------------------------------------------------------------

/// DEM accuracy over the interior, measured through the plane frame back into
/// chunk-local coordinates. Excludes a margin around the building and ramp.
struct DemAccuracy {
    std::size_t valid = 0;
    std::size_t interior = 0;
    double coverage = 0.0;
    double median = 0.0, p90 = 0.0;
    double roof_error = 0.0;
    double edge_x = std::numeric_limits<double>::quiet_NaN();
};

DemAccuracy measure_dem(const Chunk& chunk, const elevation::SurfaceFrame& frame) {
    const Elevation& dem = chunk.elevation();
    const RasterGrid& grid = dem.raster.grid;
    std::vector<double> errors;
    double roof_sum = 0.0;
    std::size_t roof_count = 0;
    // Samples (local x, height) whose local y is near 11, for the wall crossing.
    std::vector<std::pair<double, double>> row_at_y11;
    DemAccuracy result;
    for (int row = 0; row < grid.height; ++row) {
        for (int col = 0; col < grid.width; ++col) {
            const Eigen::Vector2d c = grid.pixel_center(col, row);
            const double value = static_cast<double>(dem.raster.at(col, row));
            const bool nodata = value == static_cast<double>(dem.nodata);
            // Measure in the LOCAL frame: the plane frame is UTM, so the truth
            // surface and the interior box are chunk-local. Local x/y of a DEM
            // cell are recovered via to_local at the cell's own height.
            const Eigen::Vector3d probe =
                nodata ? frame.to_local(Eigen::Vector3d(c.x(), c.y(), 0.0))
                       : frame.to_local(Eigen::Vector3d(c.x(), c.y(), value));
            const double lx = probe.x(), ly = probe.y();
            if (lx < -50.0 || lx > 50.0 || ly < -38.0 || ly > 38.0) {
                continue;
            }
            ++result.interior;
            if (nodata) {
                continue;
            }
            ++result.valid;
            const double height = probe.z();  // the DEM height in chunk-local z
            const double truth = aerial_fixture::truth_height(lx, ly);
            const bool in_box =
                lx >= -20.0 && lx <= -4.0 && ly >= 5.0 && ly <= 17.0;
            if (in_box) {
                // Inside the footprint the top surface is the flat roof.
                roof_sum += std::abs(height - aerial_fixture::kBuildTop);
                ++roof_count;
            } else if (!near_scene_edge(lx, ly, 1.5)) {
                errors.push_back(std::abs(height - truth));
            }
            if (std::abs(ly - 11.0) <= 0.3 && lx >= -24.0 && lx <= -16.0) {
                row_at_y11.emplace_back(lx, height);
            }
        }
    }
    result.coverage =
        result.interior == 0
            ? 0.0
            : static_cast<double>(result.valid) / static_cast<double>(result.interior);
    std::sort(errors.begin(), errors.end());
    result.median = density_percentile(errors, 0.50);
    result.p90 = density_percentile(errors, 0.90);
    result.roof_error = roof_count == 0 ? 0.0 : roof_sum / static_cast<double>(roof_count);

    // The wall crossing: half way between the surrounding terrain and the roof.
    std::sort(row_at_y11.begin(), row_at_y11.end());
    const double half_height =
        0.5 * (aerial_fixture::truth_height(-12.0, 11.0) + aerial_fixture::kBuildTop);
    for (std::size_t i = 1; i < row_at_y11.size(); ++i) {
        const bool below0 = row_at_y11[i - 1].second < half_height;
        const bool below1 = row_at_y11[i].second < half_height;
        if (below0 == below1) {
            continue;
        }
        const double h0 = row_at_y11[i - 1].second, h1 = row_at_y11[i].second;
        const double x0 = row_at_y11[i - 1].first, x1 = row_at_y11[i].first;
        result.edge_x = x0 + (half_height - h0) * (x1 - x0) / (h1 - h0);
        break;
    }
    return result;
}

/// Give the aerial chunk synthetic tie points: a lattice of terrain points with
/// an observation in every camera that sees them. BuildDepthMaps needs these
/// both to accept the chunk (it requires >= 1 triangulated track) and to derive
/// per-camera depth ranges and neighbourhoods. The images themselves are never
/// matched, so the points only need to be plausible.
void add_synthetic_tie_points(Chunk& chunk) {
    const double kTieStep = 5.0;  // terrain lattice spacing, metres
    std::vector<Track>& tracks = chunk.tie_points().tracks;
    tracks.clear();
    const auto push_point = [&](const Eigen::Vector3d& world) {
        Track track;
        for (const Camera& camera : chunk.cameras()) {
            const Eigen::Vector3d cam =
                camera.pose.block<3, 3>(0, 0) * world + camera.pose.block<3, 1>(0, 3);
            if (!(cam.z() > 1.0)) {
                continue;
            }
            const double u =
                aerial_fixture::kImageWidth / 2.0 + aerial_fixture::kFocal * cam.x() / cam.z();
            const double v =
                aerial_fixture::kImageHeight / 2.0 + aerial_fixture::kFocal * cam.y() / cam.z();
            if (u < 0.0 || v < 0.0 || u >= aerial_fixture::kImageWidth ||
                v >= aerial_fixture::kImageHeight) {
                continue;
            }
            Observation observation;
            observation.camera_id = camera.id;
            observation.keypoint_index = 0;
            track.observations.push_back(observation);
        }
        if (track.observations.size() < 2) {
            return;
        }
        std::sort(track.observations.begin(), track.observations.end(),
                  [](const Observation& a, const Observation& b) {
                      return a.camera_id < b.camera_id;
                  });
        track.point = world;
        tracks.push_back(std::move(track));
    };

    // Terrain lattice.
    for (double x = -55.0; x <= 55.0 + 1e-6; x += kTieStep) {
        for (double y = -40.0; y <= 40.0 + 1e-6; y += kTieStep) {
            push_point(Eigen::Vector3d(x, y, aerial_fixture::truth_height(x, y)));
        }
    }
    // Building roof: without points on the roof (z = +10 m over the terrain) the
    // per-camera depth range derived from the tracks would clip the roof away.
    for (double x = -19.0; x <= -5.0 + 1e-6; x += 2.0) {
        for (double y = 6.0; y <= 16.0 + 1e-6; y += 2.0) {
            push_point(Eigen::Vector3d(x, y, aerial_fixture::kBuildTop));
        }
    }
    // Ramp crest, so the raised ramp is inside every camera's depth range too.
    for (double x = 1.0; x <= 19.0 + 1e-6; x += 2.0) {
        for (double y = 16.0; y <= 29.0 + 1e-6; y += 2.0) {
            push_point(Eigen::Vector3d(x, y, aerial_fixture::truth_height(x, y)));
        }
    }
}

// Stage-by-stage diagnostic kept for the aerial dense-coverage investigation.
// DISABLED_ so it is not part of the normal run; enable it (or pass
// --gtest_also_run_disabled_tests) to re-measure. It documents why the pipeline
// test asserts coverage >= 0.95 and why it builds the DEM with interpolation
// "extrapolated".
TEST(BuildDemPipeline, DISABLED_DepthMapDiagnostic) {
    // Accuracy of the cells ONLY `extrapolated` fills (nodata under `enabled`):
    // is the extra coverage honest, or invented? Compare the two DEM builds
    // cell by cell against truth. Also reports the stereo stage breakdown
    // (raw PatchMatch valid -> DEM coverage), which is the number the coverage
    // assertion rests on.
    Document document = aerial_fixture::make_truth_chunk_document();
    Chunk& chunk = *document.active_chunk();
    add_synthetic_tie_points(chunk);

    Params depth = make_params(builtin_registry().require("BuildDepthMaps"));
    depth.set("downscale", ParamValue{std::int64_t{1}});
    depth.set("filter_mode", ParamValue{std::string("mild")});
    run_task(builtin_registry(), document, "BuildDepthMaps", depth);

    const elevation::SurfaceFrame frame = elevation::SurfaceFrame::for_chunk(chunk, 0);

    Params en = dem_params("depth_maps", "enabled", 0.25, 0, "natural_neighbour", "tgv");
    run_build_dem(document, en);
    const Elevation enabled_dem = chunk.elevation();

    Params ex = dem_params("depth_maps", "extrapolated", 0.25, 0, "natural_neighbour", "tgv");
    run_build_dem(document, ex);
    const Elevation& extrap_dem = chunk.elevation();

    const RasterGrid& grid = extrap_dem.raster.grid;
    std::vector<double> extrap_err;
    std::vector<double> kept_err;
    for (int row = 0; row < grid.height; ++row) {
        for (int col = 0; col < grid.width; ++col) {
            const Eigen::Vector2d c = grid.pixel_center(col, row);
            const double v = static_cast<double>(extrap_dem.raster.at(col, row));
            if (v == static_cast<double>(extrap_dem.nodata)) continue;
            const Eigen::Vector3d probe = frame.to_local(Eigen::Vector3d(c.x(), c.y(), v));
            const double lx = probe.x(), ly = probe.y();
            if (lx < -50.0 || lx > 50.0 || ly < -38.0 || ly > 38.0) continue;
            const double err = std::abs(probe.z() - aerial_fixture::truth_height(lx, ly));
            const double ev = static_cast<double>(enabled_dem.raster.at(col, row));
            if (ev == static_cast<double>(enabled_dem.nodata)) {
                extrap_err.push_back(err);  // filled only by extrapolated
            } else {
                kept_err.push_back(err);  // observed in both
            }
        }
    }
    const auto pct = [](std::vector<double>& v, double q) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[static_cast<std::size_t>(q * static_cast<double>(v.size() - 1))];
    };
    std::printf("[diag] extrap-only cells n=%zu median=%.4f p90=%.4f\n", extrap_err.size(),
                pct(extrap_err, 0.5), pct(extrap_err, 0.9));
    std::printf("[diag] observed cells n=%zu median=%.4f p90=%.4f\n", kept_err.size(),
                pct(kept_err, 0.5), pct(kept_err, 0.9));
    SUCCEED();
}

TEST(BuildDemPipeline, AerialDepthMapsToDem) {
    using clock = std::chrono::steady_clock;
    const auto seconds = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
    };

    Document document = aerial_fixture::make_truth_chunk_document();
    Chunk& chunk = *document.active_chunk();
    add_synthetic_tie_points(chunk);

    // Full-resolution depth maps at downscale 1: the 0.24 m ground sample
    // distance lands several samples per 0.25 m cell, which is what holds the
    // DEM's median error at ~1 cm. Downscale 2 is faster but blurs the surface
    // enough that TGV's tail no longer beats the unsmoothed tail, so the honest
    // accuracy comparison needs the full-resolution maps.
    //
    // The DEM is built with interpolation "extrapolated": the interior box this
    // test measures (local x in [-50, 50], y in [-38, 38]) includes a ~12% outer
    // band that only ONE camera sees (the 22-24 m nadir grid at 60 m has a ~27 m
    // x / ~20 m y half-footprint, so the outermost columns/rows fall outside the
    // second camera). No stereo matcher can observe those cells, so "enabled"
    // has to interpolate them rather than measure them. "extrapolated" extends
    // the observed surface outward from the grid's edge, which is what a
    // complete DEM product is for; on this smooth scene those cells are accurate
    // to ~2.5 cm median / ~11 cm p90 against truth (measured by the
    // DISABLED_DepthMapDiagnostic test), so the added coverage is honest.
    const auto t0 = clock::now();
    Params depth = make_params(builtin_registry().require("BuildDepthMaps"));
    depth.set("downscale", ParamValue{std::int64_t{1}});
    depth.set("filter_mode", ParamValue{std::string("mild")});
    run_task(builtin_registry(), document, "BuildDepthMaps", depth);
    const auto t1 = clock::now();
    ASSERT_FALSE(chunk.depth_maps().empty());
    std::printf("[BuildDemPipeline] BuildDepthMaps %zu maps in %.1f s\n", chunk.depth_maps().size(),
                seconds(t0, t1));

    Params dem = dem_params("depth_maps", "extrapolated", 0.25, 0, "natural_neighbour", "tgv");
    const auto t2 = clock::now();
    run_build_dem(document, dem);
    const auto t3 = clock::now();
    std::printf("[BuildDemPipeline] BuildDem (tgv) %.1f s, grid %dx%d epsg %d\n",
                seconds(t2, t3), chunk.elevation().raster.grid.width,
                chunk.elevation().raster.grid.height, chunk.elevation().epsg());
    ASSERT_FALSE(chunk.elevation().empty());
    EXPECT_EQ(chunk.elevation().epsg(), 32632);  // UTM zone 32N

    const elevation::SurfaceFrame frame = elevation::SurfaceFrame::for_chunk(chunk, 0);
    ASSERT_FALSE(frame.is_local());
    EXPECT_EQ(frame.epsg(), 32632);  // UTM zone 32N

    // -- TGV smoothing on --------------------------------------------------
    const DemAccuracy tgv = measure_dem(chunk, frame);
    const auto t4 = clock::now();

    // -- no smoothing, for the TGV-vs-none comparison ----------------------
    Params flat = dem_params("depth_maps", "extrapolated", 0.25, 0, "natural_neighbour", "none");
    run_build_dem(document, flat);
    const DemAccuracy none = measure_dem(chunk, frame);
    const auto t5 = clock::now();

    std::printf("[BuildDemPipeline] tgv   valid=%zu/%zu cov=%.3f median=%.4f p90=%.4f "
                "roof=%.4f edge=%.3f (build %.1f s)\n",
                tgv.valid, tgv.interior, tgv.coverage, tgv.median, tgv.p90, tgv.roof_error,
                tgv.edge_x, seconds(t3, t4));
    std::printf("[BuildDemPipeline] none  valid=%zu/%zu cov=%.3f median=%.4f p90=%.4f "
                "roof=%.4f edge=%.3f (build %.1f s)\n",
                none.valid, none.interior, none.coverage, none.median, none.p90, none.roof_error,
                none.edge_x, seconds(t4, t5));

    // Coverage: with the multi-view cost fixed (patch_match: a neighbour that
    // cannot see the window is no longer a phantom no-match vote) the matcher
    // accepts 74-95% per camera, up from 26-54%, and the DEM covers >= 0.95 of
    // the interior. The residual is the single-view frame border described
    // above; cells with >= 2 viewing cameras reach 0.97-0.99 coverage.
    EXPECT_GT(tgv.interior, 100000U);
    EXPECT_GE(tgv.coverage, 0.95);
    // Accuracy of the observed cells against truth_height.
    EXPECT_LT(tgv.median, 0.10);
    EXPECT_LT(tgv.p90, 0.30);
    // TGV must not worsen the median or the tail.
    EXPECT_LE(tgv.p90, none.p90 + 1e-6);
    EXPECT_LE(tgv.median, none.median + 1e-6);
    // Building edge within 0.75 m of the truth wall at x = -20.
    if (std::isfinite(tgv.edge_x)) {
        EXPECT_LT(std::abs(tgv.edge_x - -20.0), 0.75);
    } else {
        ADD_FAILURE() << "no building-edge crossing found in the DEM";
    }
    // Roof mean within 0.15 m of the truth roof height.
    EXPECT_LT(tgv.roof_error, 0.15);
}

// 7. ExportDem
// ---------------------------------------------------------------------------

TEST(ExportDem, WritesGeoTiffWithChunkEpsg) {
    Document document = make_local_chunk_document();
    Chunk& chunk = *document.active_chunk();
    chunk.model() = tilted_plane_mesh(4.0);
    Params build = dem_params("model", "disabled", 0.25, 0, "nearest", "none");
    build.set("projection", ParamValue{std::int64_t{32632}});
    chunk.transform() = Transform::Identity();
    chunk.crs().epsg = 4326;
    // Give the chunk a real georeference so epsg 32632 is valid for its origin.
    chunk.transform() = geo::enu_to_ecef(geo::Geodetic{9.2, 48.7, 300.0});
    run_build_dem(document, build);
    ASSERT_FALSE(chunk.elevation().empty());
    EXPECT_EQ(chunk.elevation().epsg(), 32632);

    const std::string dir = ::testing::TempDir() + "/alphashape_export_dem";
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/dem.tif";

    Params params = make_params(builtin_registry().require("ExportDem"));
    params.set("path", ParamValue{path});
    params.set("world_file", ParamValue{true});
    run_task(builtin_registry(), document, "ExportDem", params);

    const GeoTiffInfo info = read_geotiff_info(path);
    EXPECT_EQ(info.epsg, 32632);
    EXPECT_TRUE(info.is_float);
    EXPECT_EQ(info.grid.width, chunk.elevation().raster.grid.width);
    EXPECT_EQ(info.grid.height, chunk.elevation().raster.grid.height);
    ASSERT_TRUE(info.nodata.has_value());
    EXPECT_NEAR(*info.nodata, -32767.0, 1e-6);
    EXPECT_TRUE(std::filesystem::exists(dir + "/dem.tfw"));

    // The written raster round-trips: same grid, same heights.
    const GeoRaster<float> back = read_geotiff_float(path);
    EXPECT_EQ(back.grid, chunk.elevation().raster.grid);
    EXPECT_EQ(back.data.size(), chunk.elevation().raster.data.size());
    std::filesystem::remove(path);
    std::filesystem::remove(dir + "/dem.tfw");
}

TEST(ExportDem, WorldFileTextWhenEnabled) {
    Document document = make_local_chunk_document();
    Chunk& chunk = *document.active_chunk();
    fill_truth_point_cloud(chunk, -10.0, 10.0, -10.0, 10.0, 0.5);
    Params build = dem_params("point_cloud", "enabled", 0.5, 0, "nearest", "none");
    run_build_dem(document, build);

    const std::string dir = ::testing::TempDir() + "/alphashape_export_dem_wf";
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/local.tif";
    Params params = make_params(builtin_registry().require("ExportDem"));
    params.set("path", ParamValue{path});
    params.set("world_file", ParamValue{true});
    run_task(builtin_registry(), document, "ExportDem", params);

    const std::string tfw = dir + "/local.tfw";
    ASSERT_TRUE(std::filesystem::exists(tfw));
    std::ifstream in(tfw);
    ASSERT_TRUE(in.good());
    const RasterGrid& grid = chunk.elevation().raster.grid;
    double line1 = 0.0, line4 = 0.0, line5 = 0.0, line6 = 0.0;
    in >> line1;
    double skip = 0.0;
    in >> skip;  // D
    in >> skip;  // B
    in >> line4;  // E
    in >> line5;  // C = x of the top-left pixel centre
    in >> line6;  // F = y of the top-left pixel centre
    EXPECT_NEAR(line1, grid.pixel_size_x, 1e-9);
    EXPECT_NEAR(line4, -grid.pixel_size_y, 1e-9);
    const Eigen::Vector2d centre = grid.pixel_center(0, 0);
    EXPECT_NEAR(line5, centre.x(), 1e-9);
    EXPECT_NEAR(line6, centre.y(), 1e-9);
    // Close before removing: POSIX unlinks an open file, Windows refuses with
    // "the file is being used by another process".
    in.close();
    std::filesystem::remove(path);
    std::filesystem::remove(tfw);
}

TEST(ExportDem, EmptyElevationThrows) {
    Document document = make_local_chunk_document();
    Params params = make_params(builtin_registry().require("ExportDem"));
    params.set("path", ParamValue{platform::unique_temp_path("alphashape_unused_dem_", ".tif")});
    EXPECT_THROW(run_task(builtin_registry(), document, "ExportDem", params), Error);
}

// ---------------------------------------------------------------------------
// 8. Automatic resolution on a Geographic (EPSG:4326) grid
// ---------------------------------------------------------------------------
//
// The depth-map GSD the automatic rule measures is a GROUND distance in
// metres, but an EPSG:4326 grid's pixel size is in degrees. The rule must
// convert with the frame's metres-per-degree at the surface's own latitude: a
// value left in metres would make the DEM ~111000x too coarse per degree, and a
// value divided by the wrong constant would be just as wrong.

TEST(DemAutoResolution, GeographicDepthMapCellIsInDegrees) {
    Document document = aerial_fixture::make_truth_chunk_document();
    Chunk& chunk = *document.active_chunk();
    add_synthetic_tie_points(chunk);

    Params depth = make_params(builtin_registry().require("BuildDepthMaps"));
    depth.set("downscale", ParamValue{std::int64_t{1}});
    depth.set("filter_mode", ParamValue{std::string("mild")});
    run_task(builtin_registry(), document, "BuildDepthMaps", depth);
    ASSERT_FALSE(chunk.depth_maps().empty());

    // Ask for the automatic cell in the chunk's own geographic CRS: the DEM is
    // written in degrees.
    Params dem = dem_params("depth_maps", "enabled", 0.0, 4326, "nearest", "none");
    run_build_dem(document, dem);
    const Elevation& elevation = chunk.elevation();
    ASSERT_FALSE(elevation.empty());
    ASSERT_EQ(elevation.epsg(), 4326);

    const double cell_deg = elevation.raster.grid.pixel_size_x;
    const elevation::SurfaceFrame frame = elevation::SurfaceFrame::for_chunk(chunk, 4326);
    const Eigen::Vector2d metres_per_degree = frame.metres_per_plane_unit(
        elevation.raster.grid.pixel_center(0, elevation.raster.grid.height / 2).y());
    const double cell_metres =
        cell_deg * std::max(metres_per_degree.x(), metres_per_degree.y());
    std::printf("[DemAutoResolution.Geographic] cell=%.9f deg = %.4f m (m/deg %.1f, %.1f)\n",
                cell_deg, cell_metres, metres_per_degree.x(), metres_per_degree.y());

    // The 12-camera nadir block at 60 m with f = 500 covers ~0.12 m per pixel.
    // The converted cell must be a plausible ground distance of that order:
    // never a metre value reinterpreted as degrees (~111 km per pixel), and
    // never a degree value left unconverted (~1e-7 m per pixel).
    EXPECT_GT(cell_deg, 0.0);
    EXPECT_LT(cell_deg, 1e-4) << "a degree cell for this block is ~1e-6 deg";
    EXPECT_GT(cell_metres, 0.01);
    EXPECT_LT(cell_metres, 5.0);
}

// 8. CRS migration (M12 §4.9, WP-12.8)
// ---------------------------------------------------------------------------

TEST(BuildDem, InvalidEpsgRejected) {
    // An EPSG code PROJ does not know is refused at the task boundary, before
    // any work: the DEM build names the same CRS the grid would be written in.
    Document document = make_local_chunk_document();
    Chunk& chunk = *document.active_chunk();
    fill_truth_point_cloud(chunk, -10.0, 10.0, -10.0, 10.0, 1.0);
    Params params = dem_params("point_cloud", "enabled", 1.0, 999999, "nearest", "none");
    EXPECT_THROW(run_build_dem(document, params), ValueError);
    EXPECT_TRUE(chunk.elevation().empty());

    // The same code through the `crs` string form.
    Params text = dem_params("point_cloud", "enabled", 1.0, 0, "nearest", "none");
    text.set("crs", ParamValue{std::string("EPSG:999999")});
    EXPECT_THROW(run_build_dem(document, text), ValueError);
    EXPECT_TRUE(chunk.elevation().empty());

    // A negative projection is rejected too (the parameter carries a minimum,
    // but a direct struct/ParamValue write is still guarded).
    Params negative = dem_params("point_cloud", "enabled", 1.0, -1, "nearest", "none");
    EXPECT_THROW(run_build_dem(document, negative), ValueError);
}

TEST(BuildDem, ProjectionAndCrsMustAgree) {
    // `projection` (Int EPSG) and `crs` (String) are two spellings of one
    // request. Both set to the same CRS is fine; disagreeing values are a
    // ValueError naming the conflict, before any grid is allocated.
    Document document = aerial_fixture::make_truth_chunk_document();
    Chunk& chunk = *document.active_chunk();
    fill_truth_point_cloud(chunk, -10.0, 10.0, -10.0, 10.0, 1.0);

    Params agree = dem_params("point_cloud", "enabled", 1.0, 32632, "nearest", "none");
    agree.set("crs", ParamValue{std::string("EPSG:32632")});
    run_build_dem(document, agree);
    ASSERT_FALSE(chunk.elevation().empty());
    EXPECT_EQ(chunk.elevation().epsg(), 32632);
    // A plain EPSG request records the code and no WKT beyond the canonical one
    // the builder writes; doc.xml then prefers the code on the way back in.
    EXPECT_EQ(chunk.elevation().crs.epsg, 32632);

    Params disagree = dem_params("point_cloud", "enabled", 1.0, 32632, "nearest", "none");
    disagree.set("crs", ParamValue{std::string("EPSG:32633")});
    EXPECT_THROW(run_build_dem(document, disagree), ValueError);
}

TEST(BuildDem, MatchesTruthInEtrsUtm) {
    // The spec's MatchesTruthInEtrsLaea: a DEM built directly in a CRS outside
    // the retired three-CRS set, with the interior accuracy unchanged because
    // the plane frame is exact and the build does no resampling.
    //
    // The CRS is EPSG:25832 (ETRS89 / UTM zone 32N), not a WGS 84 zone: the
    // datum pair is real (EPSG:1149, a null shift stated at 1 m) and PROJ pairs
    // 4979 with it directly. EPSG:3035 is deliberately NOT used here: PROJ's
    // strict candidate listing for 4979 -> 3035 exposes only a ballpark
    // geographic offset, which CrsTransform refuses by design (an unpinned
    // transform never silently accepts a null datum shift). That refusal is
    // asserted below so the boundary is documented rather than surprising.
    Document document = aerial_fixture::make_truth_chunk_document();
    Chunk& chunk = *document.active_chunk();
    fill_truth_point_cloud(chunk, -55.0, 55.0, -42.0, 42.0, 0.5);

    Params params = dem_params("point_cloud", "enabled", 0.5, 0, "natural_neighbour", "none");
    params.set("crs", ParamValue{std::string("EPSG:25832")});
    run_build_dem(document, params);

    const Elevation& dem = chunk.elevation();
    ASSERT_FALSE(dem.empty());
    ASSERT_EQ(dem.epsg(), 25832);

    const elevation::SurfaceFrame frame = elevation::SurfaceFrame::for_chunk(chunk, 25832);
    ASSERT_EQ(frame.epsg(), 25832);

    // RMSE of the observed cells against truth, inside the same interior box the
    // UTM pipeline test uses. The point cloud samples the analytic surface on
    // 0.5 m, so the median rule is well inside the 0.03 m bound.
    std::vector<double> errors;
    const RasterGrid& grid = dem.raster.grid;
    for (int row = 0; row < grid.height; ++row) {
        for (int col = 0; col < grid.width; ++col) {
            const double value = static_cast<double>(dem.raster.at(col, row));
            if (value == static_cast<double>(dem.nodata)) {
                continue;
            }
            const Eigen::Vector2d c = grid.pixel_center(col, row);
            const Eigen::Vector3d local =
                frame.to_local(Eigen::Vector3d(c.x(), c.y(), value));
            const double lx = local.x(), ly = local.y();
            if (lx < -50.0 || lx > 50.0 || ly < -38.0 || ly > 38.0) {
                continue;
            }
            if (near_scene_edge(lx, ly, 1.5)) {
                continue;
            }
            errors.push_back(local.z() - aerial_fixture::truth_height(lx, ly));
        }
    }
    ASSERT_GT(errors.size(), 10000U);
    double sum_sq = 0.0;
    for (const double e : errors) {
        sum_sq += e * e;
    }
    const double rmse = std::sqrt(sum_sq / static_cast<double>(errors.size()));
    std::printf("[BuildDem.EtrsUtm] n=%zu RMSE=%.5f m\n", errors.size(), rmse);
    EXPECT_LT(rmse, 0.03);

    // EPSG:3035 (ETRS89 / LAEA Europe) is expressible through the CRS engine,
    // but its only candidate from WGS 84 geographic 3D is a ballpark geographic
    // offset, so the build refuses it rather than silently shifting the grid.
    Params laea = dem_params("point_cloud", "enabled", 0.5, 0, "natural_neighbour", "none");
    laea.set("crs", ParamValue{std::string("EPSG:3035")});
    bool refused = false;
    try {
        run_build_dem(document, laea);
    } catch (const Error& error) {
        refused = true;
        EXPECT_NE(std::string(error.what()).find("allow_ballpark"), std::string::npos)
            << error.what();
    }
    EXPECT_TRUE(refused) << "4979 -> 3035 has only a ballpark candidate";
}

}  // namespace
}  // namespace alphashape
