// test_build_orthomosaic.cpp -- tests for the BuildOrthomosaic /
// ExportOrthomosaic tasks (milestone 9).
//
// One expensive setup is shared by every test: the georeferenced aerial
// fixture, full-resolution depth maps, then BuildDem. The orthomosaic tests
// measure the produced colour against the fixture's truth orthophoto through
// the UTM plane frame -> chunk local -> truth grid path.

#include "aerial_fixture.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/algorithm/elevation/surface_frame.hpp"
#include "common/algorithm/ortho/build_orthomosaic_task.hpp"
#include "common/algorithm/ortho/export_orthomosaic_task.hpp"
#include "common/core/document.hpp"
#include "common/io/project_io.hpp"
#include "common/tasks/registry.hpp"
#include "model/raster/geotiff.hpp"

namespace alphashape {
namespace {

using clock_type = std::chrono::steady_clock;

double seconds_since(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

/// Synthetic tie points, the same construction test_build_dem uses: a terrain
/// lattice plus roof and ramp-crest points with an observation in every camera
/// that sees them. BuildDepthMaps needs a triangulated track both to accept the
/// chunk and to derive per-camera depth ranges; the images are never matched.
void add_synthetic_tie_points(Chunk& chunk) {
    constexpr double kTieStep = 5.0;
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
    for (double x = -55.0; x <= 55.0 + 1e-6; x += kTieStep) {
        for (double y = -40.0; y <= 40.0 + 1e-6; y += kTieStep) {
            push_point(Eigen::Vector3d(x, y, aerial_fixture::truth_height(x, y)));
        }
    }
    for (double x = -19.0; x <= -5.0 + 1e-6; x += 2.0) {
        for (double y = 6.0; y <= 16.0 + 1e-6; y += 2.0) {
            push_point(Eigen::Vector3d(x, y, aerial_fixture::kBuildTop));
        }
    }
    for (double x = 1.0; x <= 19.0 + 1e-6; x += 2.0) {
        for (double y = 16.0; y <= 29.0 + 1e-6; y += 2.0) {
            push_point(Eigen::Vector3d(x, y, aerial_fixture::truth_height(x, y)));
        }
    }
}

/// Everything the measurement needs, fixed across tests.
struct OrthoSetup {
    Document document;
    // SurfaceFrame has no public default constructor, so it is filled once the
    // chunk (and its DEM) exist.
    std::optional<elevation::SurfaceFrame> frame;
    Image truth_ortho;
    aerial_fixture::TruthGrid truth_grid;
};

OrthoSetup& setup() {
    static std::unique_ptr<OrthoSetup> cached;
    if (cached) {
        return *cached;
    }
    auto built = std::make_unique<OrthoSetup>();
    built->document = aerial_fixture::make_truth_chunk_document();
    Chunk& chunk = *built->document.active_chunk();
    add_synthetic_tie_points(chunk);

    Params depth = make_params(builtin_registry().require("BuildDepthMaps"));
    depth.set("downscale", ParamValue{std::int64_t{1}});
    depth.set("filter_mode", ParamValue{std::string("mild")});
    run_task(builtin_registry(), built->document, "BuildDepthMaps", depth);

    Params dem = make_params(builtin_registry().require("BuildDem"));
    dem.set("source_data", ParamValue{std::string("depth_maps")});
    dem.set("interpolation", ParamValue{std::string("extrapolated")});
    dem.set("resolution", ParamValue{0.25});
    dem.set("fill_method", ParamValue{std::string("natural_neighbour")});
    dem.set("smoothing", ParamValue{std::string("tgv")});
    run_task(builtin_registry(), built->document, "BuildDem", dem);

    built->frame = elevation::SurfaceFrame::for_chunk(chunk, 0);
    built->truth_ortho = aerial_fixture::load_truth_ortho();
    built->truth_grid = aerial_fixture::load_truth_dem();
    cached = std::move(built);
    return *cached;
}

/// Params for BuildOrthomosaic with the requested overrides.
Params ortho_params(const std::string& blending = "mosaic",
                    const std::string& surface = "elevation", double resolution = 0.0,
                    bool fill_holes = true, bool refine_seamlines = true,
                    std::int64_t projection = 0) {
    Params params = make_params(builtin_registry().require("BuildOrthomosaic"));
    params.set("surface_data", ParamValue{surface});
    params.set("blending_mode", ParamValue{blending});
    params.set("fill_holes", ParamValue{fill_holes});
    params.set("refine_seamlines", ParamValue{refine_seamlines});
    params.set("resolution", ParamValue{resolution});
    params.set("projection", ParamValue{projection});
    return params;
}

/// Bilinear, nodata-aware DEM height (the orthorectifier's sampling rule).
double dem_height(const Elevation& dem, double x, double y) {
    const RasterGrid& grid = dem.raster.grid;
    const Eigen::Vector2d c = grid.to_pixel(x, y);
    const double cf = c.x() - 0.5;
    const double rf = c.y() - 0.5;
    const int c0 = static_cast<int>(std::floor(cf));
    const int r0 = static_cast<int>(std::floor(rf));
    const double fx = cf - static_cast<double>(c0);
    const double fy = rf - static_cast<double>(r0);
    double sum = 0.0;
    double weight = 0.0;
    for (int dr = 0; dr <= 1; ++dr) {
        for (int dc = 0; dc <= 1; ++dc) {
            const int col = c0 + dc;
            const int row = r0 + dr;
            if (!grid.contains_pixel(col, row)) {
                continue;
            }
            const float h = dem.raster.at(col, row);
            if (h == dem.nodata || std::isnan(h)) {
                continue;
            }
            const double w = (dc == 0 ? 1.0 - fx : fx) * (dr == 0 ? 1.0 - fy : fy);
            sum += w * static_cast<double>(h);
            weight += w;
        }
    }
    return weight <= 0.0 ? static_cast<double>(dem.nodata) : sum / weight;
}

/// The same interior window test_build_dem and test_ortho use: local x in
/// [-52, 52], y in [-38, 38].
bool interior_cell(const elevation::SurfaceFrame& frame, const Elevation& dem,
                   const Orthomosaic& ortho, int col, int row) {
    const Eigen::Vector2d c = ortho.raster.grid.pixel_center(col, row);
    const double h = dem_height(dem, c.x(), c.y());
    if (h == static_cast<double>(dem.nodata) || !std::isfinite(h)) {
        return false;
    }
    const Eigen::Vector3d local = frame.to_local(Eigen::Vector3d(c.x(), c.y(), h));
    return local.x() >= -52.0 && local.x() <= 52.0 && local.y() >= -38.0 &&
           local.y() <= 38.0;
}

/// Truth-grid cell nearest a chunk-local (x, y) -- the truth ortho is a texture
/// on the truth DEM grid, so the nearest cell is the sampling rule.
void truth_cell(const aerial_fixture::TruthGrid& grid, double x, double y, int& col,
                int& row) {
    col = static_cast<int>(std::floor((x - grid.origin_x) / grid.cell));
    row = static_cast<int>(std::floor((grid.origin_y - y) / grid.cell));
    col = std::clamp(col, 0, grid.width - 1);
    row = std::clamp(row, 0, grid.height - 1);
}

struct ColourError {
    std::size_t compared = 0;   ///< alpha-valid interior cells
    std::size_t interior = 0;   ///< interior cells with a DEM height
    double coverage = 0.0;
    double mean = 0.0;
    double footprint = 0.0;     ///< mean error inside the roof box shrunk by 1 m
    std::size_t footprint_cells = 0;
    double elsewhere = 0.0;     ///< mean error outside the unshrunk footprint
};

/// Mean absolute RGB difference of the ortho against the truth orthophoto.
ColourError measure_colour(OrthoSetup& s, const Orthomosaic& ortho) {
    const Elevation& dem = s.document.active_chunk()->elevation();
    const RasterGrid& grid = ortho.raster.grid;
    ColourError result;
    double sum = 0.0;
    double foot_sum = 0.0;
    double other_sum = 0.0;
    std::size_t other_cells = 0;
    for (int row = 0; row < grid.height; ++row) {
        for (int col = 0; col < grid.width; ++col) {
            if (!interior_cell(*s.frame, dem, ortho, col, row)) {
                continue;
            }
            ++result.interior;
            if (ortho.raster.at(col, row, 3) != 255) {
                continue;
            }
            ++result.compared;
            const Eigen::Vector2d c = grid.pixel_center(col, row);
            const double h = dem_height(dem, c.x(), c.y());
            const Eigen::Vector3d local = s.frame->to_local(Eigen::Vector3d(c.x(), c.y(), h));
            int tcol = 0, trow = 0;
            truth_cell(s.truth_grid, local.x(), local.y(), tcol, trow);
            double diff = 0.0;
            for (int ch = 0; ch < 3; ++ch) {
                const int tc = s.truth_ortho.channels() >= 3 ? ch : 0;
                diff += std::abs(static_cast<double>(ortho.raster.at(col, row, ch)) -
                                 static_cast<double>(s.truth_ortho.at(tcol, trow, tc)));
            }
            diff /= 3.0;
            sum += diff;
            // The roof box shrunk by 1 m: inside it the top surface is one flat
            // roof, so a true ortho must show no relief displacement there.
            const bool in_shrunk =
                local.x() >= aerial_fixture::kBuildX0 + 1.0 &&
                local.x() <= aerial_fixture::kBuildX1 - 1.0 &&
                local.y() >= aerial_fixture::kBuildY0 + 1.0 &&
                local.y() <= aerial_fixture::kBuildY1 - 1.0;
            const bool in_full = local.x() >= aerial_fixture::kBuildX0 &&
                                 local.x() <= aerial_fixture::kBuildX1 &&
                                 local.y() >= aerial_fixture::kBuildY0 &&
                                 local.y() <= aerial_fixture::kBuildY1;
            if (in_shrunk) {
                foot_sum += diff;
                ++result.footprint_cells;
            } else if (!in_full) {
                other_sum += diff;
                ++other_cells;
            }
        }
    }
    if (result.compared > 0) {
        result.mean = sum / static_cast<double>(result.compared);
    }
    if (result.interior > 0) {
        result.coverage = static_cast<double>(result.compared) /
                          static_cast<double>(result.interior);
    }
    if (result.footprint_cells > 0) {
        result.footprint = foot_sum / static_cast<double>(result.footprint_cells);
    }
    if (other_cells > 0) {
        result.elsewhere = other_sum / static_cast<double>(other_cells);
    }
    return result;
}

/// The envelope of the ortho against the DEM grid: the ortho must cover the DEM
/// extent (its origin may be snapped outward by < one ortho cell).
void expect_grid_covers_dem(const RasterGrid& ortho, const RasterGrid& dem,
                            double ortho_resolution) {
    EXPECT_LE(ortho.origin_x, dem.origin_x + 1e-9);
    EXPECT_GE(ortho.origin_y, dem.origin_y - 1e-9);
    const double ortho_east = ortho.origin_x + static_cast<double>(ortho.width) * ortho.pixel_size_x;
    const double dem_east = dem.origin_x + static_cast<double>(dem.width) * dem.pixel_size_x;
    EXPECT_GE(ortho_east, dem_east - 1e-9);
    const double ortho_south = ortho.origin_y - static_cast<double>(ortho.height) * ortho.pixel_size_y;
    const double dem_south = dem.origin_y - static_cast<double>(dem.height) * dem.pixel_size_y;
    EXPECT_LE(ortho_south, dem_south + 1e-9);
    EXPECT_LT(ortho.origin_x - dem.origin_x, ortho_resolution + 1e-9);
    EXPECT_LT(ortho.origin_y - dem.origin_y, ortho_resolution + 1e-9);
}

// ---------------------------------------------------------------------------
// 1. The orthomosaic against the truth orthophoto, all three blending modes
// ---------------------------------------------------------------------------

TEST(BuildOrthomosaic, MatchesTruthOrtho) {
    OrthoSetup& s = setup();
    Chunk& chunk = *s.document.active_chunk();
    const RasterGrid dem_grid = chunk.elevation().raster.grid;

    struct Row {
        const char* mode;
        ColourError error;
        double seconds;
    };
    std::vector<Row> rows;
    for (const char* mode : {"mosaic", "average", "disabled"}) {
        const auto start = clock_type::now();
        Params params = ortho_params(mode);
        run_task(builtin_registry(), s.document, "BuildOrthomosaic", params);
        rows.push_back(Row{mode, measure_colour(s, chunk.orthomosaic()),
                           seconds_since(start)});
    }

    const Orthomosaic& ortho = chunk.orthomosaic();
    ASSERT_FALSE(ortho.empty());
    ASSERT_EQ(ortho.raster.channels, 4);
    ASSERT_EQ(ortho.epsg, 32632);
    ASSERT_EQ(ortho.surface, "elevation");
    ASSERT_EQ(ortho.labels.size(),
              static_cast<std::size_t>(ortho.raster.grid.width) *
                  static_cast<std::size_t>(ortho.raster.grid.height));
    const double resolution = ortho.raster.grid.pixel_size_x;
    std::printf("[BuildOrthomosaic.MatchesTruthOrtho] auto resolution=%.6f (DEM %.3f)\n",
                resolution, dem_grid.pixel_size_x);
    EXPECT_GT(resolution, 0.0);
    EXPECT_GE(resolution, 0.25 * dem_grid.pixel_size_x - 1e-9);
    expect_grid_covers_dem(ortho.raster.grid, dem_grid, resolution);

    // Every label is a chunk camera id or -1.
    for (const std::int32_t label : ortho.labels) {
        if (label < 0) {
            continue;
        }
        bool found = false;
        for (const Camera& camera : chunk.cameras()) {
            found = found || camera.id == label;
        }
        ASSERT_TRUE(found) << "label " << label << " is not a chunk camera id";
    }

    for (const Row& row : rows) {
        std::printf("[BuildOrthomosaic.MatchesTruthOrtho] %-8s mean=%.3f "
                    "footprint=%.3f elsewhere=%.3f count=%zu coverage=%.4f interior=%zu "
                    "footcells=%zu time=%.1f s\n",
                    row.mode, row.error.mean, row.error.footprint, row.error.elsewhere,
                    row.error.compared, row.error.coverage, row.error.interior,
                    row.error.footprint_cells, row.seconds);
        EXPECT_LT(row.error.mean, 15.0) << "blending mode " << row.mode;
        EXPECT_GE(row.error.coverage, 0.95) << "blending mode " << row.mode;
        EXPECT_GT(row.error.footprint_cells, 100U);
        // No relief displacement: the roof must not be materially worse than
        // the rest of the scene.
        EXPECT_LT(row.error.footprint, 2.0 * row.error.elsewhere + 1e-9)
            << "blending mode " << row.mode;
    }

    // Every labelled cell is opaque. With fill_holes (default) the only opaque
    // cells without a label are filled interior holes: a small fraction.
    std::size_t covered = 0;
    std::size_t labelled = 0;
    std::size_t labelled_transparent = 0;
    for (std::size_t i = 0; i < ortho.labels.size(); ++i) {
        const bool opaque = ortho.raster.data[i * 4 + 3] == 255;
        covered += opaque ? 1U : 0U;
        if (ortho.labels[i] >= 0) {
            ++labelled;
            labelled_transparent += opaque ? 0U : 1U;
        }
    }
    std::printf("[BuildOrthomosaic.MatchesTruthOrtho] covered=%zu labelled=%zu filled=%zu\n",
                covered, labelled, covered - labelled);
    EXPECT_GT(covered, 100000U);
    EXPECT_EQ(labelled_transparent, 0U);
    EXPECT_GE(covered, labelled);
    EXPECT_LT(covered - labelled, covered / 100);
}

// ---------------------------------------------------------------------------
// 2. The model surface (a DSM rasterised from the mesh)
// ---------------------------------------------------------------------------

TEST(BuildOrthomosaic, ModelSurface) {
    OrthoSetup& s = setup();
    Chunk& chunk = *s.document.active_chunk();

    Params model = make_params(builtin_registry().require("BuildModel"));
    const auto t0 = clock_type::now();
    run_task(builtin_registry(), s.document, "BuildModel", model);
    ASSERT_FALSE(chunk.model().empty());
    const double model_seconds = seconds_since(t0);

    // An explicit 0.25 m cell (the DEM's own) keeps this second full-scene
    // ortho cheap; the point of the test is the model-derived surface, not the
    // automatic resolution rule (covered by MatchesTruthOrtho).
    const auto t1 = clock_type::now();
    Params params = ortho_params("mosaic", "model", 0.25);
    run_task(builtin_registry(), s.document, "BuildOrthomosaic", params);
    const double ortho_seconds = seconds_since(t1);

    const Orthomosaic& ortho = chunk.orthomosaic();
    ASSERT_FALSE(ortho.empty());
    EXPECT_EQ(ortho.surface, "model");
    EXPECT_EQ(ortho.raster.channels, 4);
    const ColourError error = measure_colour(s, ortho);
    std::printf("[BuildOrthomosaic.ModelSurface] mean=%.3f coverage=%.4f count=%zu "
                "grid=%dx%d model=%.1f s ortho=%.1f s\n",
                error.mean, error.coverage, error.compared, ortho.raster.grid.width,
                ortho.raster.grid.height, model_seconds, ortho_seconds);
    EXPECT_LT(error.mean, 18.0);
    EXPECT_GE(error.coverage, 0.90);
    EXPECT_GT(error.compared, 10000U);
}

// ---------------------------------------------------------------------------
// 3. plane_to_local: exact and interpolated paths
// ---------------------------------------------------------------------------

TEST(BuildOrthomosaic, PlaneToLocalIsExact) {
    OrthoSetup& s = setup();
    const Chunk& chunk = *s.document.active_chunk();
    const Elevation& dem = chunk.elevation();
    ASSERT_TRUE(s.frame.has_value());
    ASSERT_FALSE(s.frame->is_local());

    // The map is built on the same grid the task uses. Build it here on the DEM
    // grid itself: the node lattice then coincides with the DEM cells, and the
    // cell-centre queries interpolate between the four nodes around a cell.
    const std::function<Eigen::Vector3d(const Eigen::Vector3d&)> map =
        make_plane_to_local(*s.frame, dem.raster.grid, dem.raster, dem.nodata);
    ASSERT_TRUE(static_cast<bool>(map));

    // (a) Fidelity against the exact surface frame: every interior cell centre,
    // at the DEM's own height, must map to within 1 mm of frame.to_local.
    const RasterGrid& grid = dem.raster.grid;
    double worst = 0.0;
    std::size_t checked = 0;
    double worst_near_node = 0.0;
    std::size_t node_checks = 0;
    for (int row = 0; row < grid.height; ++row) {
        for (int col = 0; col < grid.width; ++col) {
            const Eigen::Vector2d c = grid.pixel_center(col, row);
            const double h = dem_height(dem, c.x(), c.y());
            if (h == static_cast<double>(dem.nodata) || !std::isfinite(h)) {
                continue;
            }
            const Eigen::Vector3d exact =
                s.frame->to_local(Eigen::Vector3d(c.x(), c.y(), h));
            const Eigen::Vector3d mapped =
                map(Eigen::Vector3d(c.x(), c.y(), h));
            worst = std::max(worst, (mapped - exact).norm());
            ++checked;
        }
    }
    // (b) A point 1 m ABOVE the surface must also map within 1 mm: the height
    // carries exactly, only the horizontal interpolation is approximate.
    for (int row = 0; row < grid.height; row += 7) {
        for (int col = 0; col < grid.width; col += 7) {
            const Eigen::Vector2d c = grid.pixel_center(col, row);
            const double h = dem_height(dem, c.x(), c.y());
            if (h == static_cast<double>(dem.nodata) || !std::isfinite(h)) {
                continue;
            }
            const Eigen::Vector3d p(c.x(), c.y(), h + 1.0);
            worst_near_node =
                std::max(worst_near_node, (map(p) - s.frame->to_local(p)).norm());
            ++node_checks;
        }
    }
    std::printf("[BuildOrthomosaic.PlaneToLocalIsExact] cells=%zu worst=%.3e m "
                "raised=%zu worst=%.3e m\n",
                checked, worst, node_checks, worst_near_node);
    EXPECT_GT(checked, 100000U);
    EXPECT_LT(worst, 1e-3);
    EXPECT_LT(worst_near_node, 1e-3);

    // (c) 10 m off the top of the raster still maps sanely (clamped, not NaN).
    const Eigen::Vector3d outside = map(Eigen::Vector3d(grid.origin_x + 1.0,
                                                       grid.origin_y + 10.0, 2.0));
    EXPECT_TRUE(outside.allFinite());
}

// ---------------------------------------------------------------------------
// 4. Errors
// ---------------------------------------------------------------------------

TEST(BuildOrthomosaic, Errors) {
    OrthoSetup& s = setup();
    Chunk& chunk = *s.document.active_chunk();

    // A chunk with no DEM refuses "elevation" but not "model".
    Document empty = aerial_fixture::make_truth_chunk_document();
    Chunk& empty_chunk = *empty.active_chunk();
    Params no_dem = ortho_params();
    EXPECT_THROW(run_task(builtin_registry(), empty, "BuildOrthomosaic", no_dem),
                 ValueError);
    EXPECT_TRUE(empty_chunk.orthomosaic().empty());

    Params model_no_mesh = ortho_params("mosaic", "model");
    EXPECT_THROW(run_task(builtin_registry(), empty, "BuildOrthomosaic", model_no_mesh),
                 ValueError);

    // A projection that differs from the DEM's is refused with the documented
    // message: the task does not reproject.
    Params reproject = ortho_params("mosaic", "elevation", 0.0, true, true,
                                    std::int64_t{32633});
    bool threw = false;
    try {
        run_task(builtin_registry(), s.document, "BuildOrthomosaic", reproject);
    } catch (const ValueError& error) {
        threw = true;
        EXPECT_NE(std::string(error.what()).find("reproject the DEM first"),
                  std::string::npos)
            << error.what();
    }
    EXPECT_TRUE(threw);

    Params bad_blend = ortho_params("bogus");
    EXPECT_THROW(run_task(builtin_registry(), s.document, "BuildOrthomosaic", bad_blend),
                 ValueError);

    Params bad_surface = ortho_params("mosaic", "bogus");
    EXPECT_THROW(run_task(builtin_registry(), s.document, "BuildOrthomosaic", bad_surface),
                 ValueError);

    Params negative = ortho_params("mosaic", "elevation", -1.0);
    EXPECT_THROW(run_task(builtin_registry(), s.document, "BuildOrthomosaic", negative),
                 ValueError);

    // The failures above must not have replaced a working orthomosaic. Build a
    // cheap one (1 m cells) and re-check it after one more failing call.
    Params coarse = ortho_params("mosaic", "elevation", 1.0);
    run_task(builtin_registry(), s.document, "BuildOrthomosaic", coarse);
    ASSERT_FALSE(chunk.orthomosaic().empty());
    const Orthomosaic before = chunk.orthomosaic();
    Params still_bad = ortho_params("bogus");
    EXPECT_THROW(run_task(builtin_registry(), s.document, "BuildOrthomosaic", still_bad),
                 ValueError);
    EXPECT_TRUE(chunk.orthomosaic() == before);
    EXPECT_EQ(chunk.orthomosaic().raster.grid.pixel_size_x, 1.0);
}

// ---------------------------------------------------------------------------
// 5. Export
// ---------------------------------------------------------------------------

TEST(ExportOrthomosaic, WritesGeoTiff) {
    OrthoSetup& s = setup();
    Chunk& chunk = *s.document.active_chunk();

    // A small explicit resolution keeps this test's ortho fast; the grid still
    // covers the DEM and carries the DEM's EPSG.
    Params build = ortho_params("mosaic", "elevation", 1.0);
    run_task(builtin_registry(), s.document, "BuildOrthomosaic", build);
    ASSERT_FALSE(chunk.orthomosaic().empty());
    const RasterGrid grid = chunk.orthomosaic().raster.grid;

    const std::string dir = ::testing::TempDir() + "/alphashape_export_ortho";
    std::filesystem::create_directories(dir);
    const std::string rgba_path = dir + "/ortho_rgba.tif";
    Params params = make_params(builtin_registry().require("ExportOrthomosaic"));
    params.set("path", ParamValue{rgba_path});
    params.set("world_file", ParamValue{true});
    run_task(builtin_registry(), s.document, "ExportOrthomosaic", params);

    const GeoTiffInfo info = read_geotiff_info(rgba_path);
    EXPECT_EQ(info.epsg, 32632);
    EXPECT_FALSE(info.is_float);
    EXPECT_EQ(info.bits_per_sample, 8);
    EXPECT_EQ(info.channels, 4);
    EXPECT_EQ(info.grid, grid);

    const GeoRaster<std::uint8_t> back = read_geotiff_u8(rgba_path);
    EXPECT_EQ(back.grid, grid);
    EXPECT_EQ(back.channels, 4);
    ASSERT_EQ(back.data.size(), chunk.orthomosaic().raster.data.size());
    std::size_t mismatched = 0;
    for (std::size_t i = 0; i < back.data.size(); ++i) {
        mismatched += back.data[i] != chunk.orthomosaic().raster.data[i] ? 1U : 0U;
    }
    EXPECT_EQ(mismatched, 0U);

    // World file: same convention as ExportDem (top-left pixel centre).
    const std::string tfw = dir + "/ortho_rgba.tfw";
    ASSERT_TRUE(std::filesystem::exists(tfw));
    std::ifstream in(tfw);
    ASSERT_TRUE(in.good());
    double a = 0.0, d = 0.0, b = 0.0, e = 0.0, c = 0.0, f = 0.0;
    in >> a >> d >> b >> e >> c >> f;
    EXPECT_NEAR(a, grid.pixel_size_x, 1e-9);
    EXPECT_NEAR(e, -grid.pixel_size_y, 1e-9);
    const Eigen::Vector2d centre = grid.pixel_center(0, 0);
    EXPECT_NEAR(c, centre.x(), 1e-9);
    EXPECT_NEAR(f, centre.y(), 1e-9);
    // Close before the later remove(): Windows refuses to unlink a file that is
    // still open, POSIX quietly succeeds.
    in.close();

    // alpha = false: three channels, same grid.
    const std::string rgb_path = dir + "/ortho_rgb.tif";
    params.set("path", ParamValue{rgb_path});
    params.set("alpha", ParamValue{false});
    params.set("world_file", ParamValue{false});
    run_task(builtin_registry(), s.document, "ExportOrthomosaic", params);
    const GeoTiffInfo rgb_info = read_geotiff_info(rgb_path);
    EXPECT_EQ(rgb_info.channels, 3);
    EXPECT_EQ(rgb_info.grid, grid);
    EXPECT_EQ(rgb_info.epsg, 32632);
    const GeoRaster<std::uint8_t> rgb = read_geotiff_u8(rgb_path);
    EXPECT_EQ(rgb.grid, grid);
    EXPECT_EQ(rgb.channels, 3);

    // No orthomosaic: a hard error.
    Document empty = aerial_fixture::make_truth_chunk_document();
    Params missing = make_params(builtin_registry().require("ExportOrthomosaic"));
    missing.set("path", ParamValue{dir + "/unused.tif"});
    EXPECT_THROW(run_task(builtin_registry(), empty, "ExportOrthomosaic", missing), Error);

    // Path is required.
    Params no_path = make_params(builtin_registry().require("ExportOrthomosaic"));
    EXPECT_THROW(run_task(builtin_registry(), s.document, "ExportOrthomosaic", no_path),
                 ValueError);

    std::filesystem::remove(rgba_path);
    std::filesystem::remove(tfw);
    std::filesystem::remove(rgb_path);
}

// ---------------------------------------------------------------------------
// 6. The orthomosaic and the DEM it was built from persist exactly
// ---------------------------------------------------------------------------

/// Saving a chunk with both products writes them as GeoTIFF side files, and the
/// reloaded chunk compares equal to the original container for container.
TEST(ExportOrthomosaic, ProductsSurviveSaveAndLoad) {
    Document document = aerial_fixture::make_truth_chunk_document();
    Chunk& chunk = *document.active_chunk();
    add_synthetic_tie_points(chunk);

    Params depth = make_params(builtin_registry().require("BuildDepthMaps"));
    depth.set("downscale", ParamValue{std::int64_t{4}});
    run_task(builtin_registry(), document, "BuildDepthMaps", depth);

    Params dem = make_params(builtin_registry().require("BuildDem"));
    dem.set("source_data", ParamValue{std::string("depth_maps")});
    dem.set("interpolation", ParamValue{std::string("extrapolated")});
    dem.set("resolution", ParamValue{1.0});
    dem.set("smoothing", ParamValue{std::string("none")});
    run_task(builtin_registry(), document, "BuildDem", dem);

    Params ortho = ortho_params("average", "elevation", 1.0);
    run_task(builtin_registry(), document, "BuildOrthomosaic", ortho);
    ASSERT_FALSE(chunk.elevation().empty());
    ASSERT_FALSE(chunk.orthomosaic().empty());

    // Move the two products onto a bare chunk. The synthetic tie points above
    // are only what the dense tasks need; they reference no keypoint sets, so a
    // project carrying them would be rejected for a reason unrelated to the
    // products under test here.
    Document project_document;
    Chunk& project_chunk = project_document.workspace().add_chunk("products");
    project_document.set_active_chunk_id(project_chunk.id());
    project_chunk.elevation() = chunk.elevation();
    project_chunk.orthomosaic() = chunk.orthomosaic();

    const std::string dir = ::testing::TempDir() + "/alphashape_ortho_save";
    std::filesystem::create_directories(dir);
    const std::string project = dir + "/ortho.psx";
    ProjectIO::save(project_document, project);
    ASSERT_TRUE(std::filesystem::exists(project));

    std::shared_ptr<Document> loaded = ProjectIO::load(project);
    const Chunk& got = *loaded->workspace().chunks().front();
    EXPECT_TRUE(got == project_chunk);
    EXPECT_EQ(got.elevation().epsg, chunk.elevation().epsg);
    EXPECT_EQ(got.elevation().source, "depth_maps");
    EXPECT_EQ(got.orthomosaic().epsg, chunk.orthomosaic().epsg);
    EXPECT_EQ(got.orthomosaic().surface, "elevation");
    EXPECT_EQ(got.orthomosaic().labels, chunk.orthomosaic().labels);
}

}  // namespace
}  // namespace alphashape
