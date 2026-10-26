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
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "base/platform/env.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/elevation/surface_frame.hpp"
#include "common/algorithm/ortho/build_orthomosaic_task.hpp"
#include "common/algorithm/ortho/export_orthomosaic_task.hpp"
#include "common/core/document.hpp"
#include "common/io/project_io.hpp"
#include "common/tasks/registry.hpp"
#include "model/geo/geodesy.hpp"
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
    // The automatic cell is the measured ground GSD, floored only by the DEM/8
    // sanity check. For this fixture (0.25 m DEM, 60 m flight, f = 500) the
    // measured GSD is ~0.116 m, well above that floor and coarser than the old
    // quarter-DEM-cell clamp would have allowed.
    EXPECT_GE(resolution, 0.125 * dem_grid.pixel_size_x - 1e-9);
    EXPECT_NEAR(resolution, 0.116, 0.02);
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
// 5. Memory guard
// ---------------------------------------------------------------------------

namespace {

/// Set ALPHASHAPE_MEMORY_BUDGET_MB for the lifetime of the object, restoring the
/// previous value (or unsetting it) afterwards. The task reads the variable on
/// every run, so nothing has to be restarted.
class ScopedMemoryBudget {
public:
    explicit ScopedMemoryBudget(const std::string& megabytes) {
        previous_ = platform::get_env("ALPHASHAPE_MEMORY_BUDGET_MB");
        had_previous_ = !previous_.empty();
        platform::set_env("ALPHASHAPE_MEMORY_BUDGET_MB", megabytes);
    }
    ~ScopedMemoryBudget() {
        platform::set_env("ALPHASHAPE_MEMORY_BUDGET_MB",
                          had_previous_ ? previous_ : std::string());
    }
    ScopedMemoryBudget(const ScopedMemoryBudget&) = delete;
    ScopedMemoryBudget& operator=(const ScopedMemoryBudget&) = delete;

private:
    std::string previous_;
    bool had_previous_ = false;
};

/// Capture every log line emitted for the lifetime of the object, so a test can
/// assert on a warning that is otherwise only visible on the console. The sink
/// is process-wide, so the object must not outlive the test's own scope and no
/// other test may run concurrently in this process (gtest runs one at a time).
class ScopedLogCapture {
public:
    ScopedLogCapture() {
        previous_level_ = Log::level();
        if (!Log::enabled(LogLevel::Trace)) {
            Log::set_level(LogLevel::Trace);  // make sure Info/Warn are emitted
        }
        Log::set_sink([this](LogLevel level, const std::string& message) {
            lines_.push_back({level, message});
        });
    }
    ~ScopedLogCapture() {
        Log::set_sink({});
        Log::set_level(previous_level_);
    }
    ScopedLogCapture(const ScopedLogCapture&) = delete;
    ScopedLogCapture& operator=(const ScopedLogCapture&) = delete;

    /// Every captured line at `level`, joined by newline.
    std::string text(LogLevel level) const {
        std::string out;
        for (const auto& [lvl, message] : lines_) {
            if (lvl == level) {
                out += message;
                out += '\n';
            }
        }
        return out;
    }

private:
    std::vector<std::pair<LogLevel, std::string>> lines_;
    LogLevel previous_level_ = LogLevel::Info;
};

}  // namespace

TEST(BuildOrthomosaic, MemoryGuardRefusesAnImpossibleResolution) {
    OrthoSetup& s = setup();
    Chunk& chunk = *s.document.active_chunk();
    // The fixture is shared between tests, so an earlier case may already have
    // produced an orthomosaic. The contract is that a refused run leaves
    // whatever was there untouched, not that nothing exists.
    const Orthomosaic before = chunk.orthomosaic();

    // A 1 MB budget cannot hold any ortho of this scene: the guard must refuse
    // before allocating, naming the resolution, the machine and an actionable
    // coarser value.
    ScopedMemoryBudget budget("1");
    Params params = ortho_params("mosaic", "elevation", 0.05);
    bool threw = false;
    try {
        run_task(builtin_registry(), s.document, "BuildOrthomosaic", params);
    } catch (const ValueError& error) {
        threw = true;
        const std::string message = error.what();
        EXPECT_NE(message.find("BuildOrthomosaic needs about"), std::string::npos)
            << message;
        EXPECT_NE(message.find("this machine has"), std::string::npos) << message;
        EXPECT_NE(message.find("Set a coarser resolution"), std::string::npos) << message;
    }
    EXPECT_TRUE(threw) << "the guard did not refuse a 1 MB budget";
    EXPECT_TRUE(chunk.orthomosaic() == before);  // nothing was committed
}

TEST(BuildOrthomosaic, AbsurdResolutionIsRefusedBeforeItOverflows) {
    OrthoSetup& s = setup();
    Chunk& chunk = *s.document.active_chunk();
    const Orthomosaic before = chunk.orthomosaic();

    // 1e-6 m over the fixture's ~120 x 90 m DEM is ~1.1e16 cells: both the
    // per-side counts and the product overflow the grid's int dimensions, so
    // the grid builder must reject in double precision, before anything is
    // allocated. A generous memory budget must not let it through: the failure
    // is representational, not a shortage.
    ScopedMemoryBudget budget("65536");
    Params params = ortho_params("mosaic", "elevation", 1e-6);
    bool threw = false;
    try {
        run_task(builtin_registry(), s.document, "BuildOrthomosaic", params);
    } catch (const ValueError& error) {
        threw = true;
        const std::string message = error.what();
        EXPECT_NE(message.find("would need"), std::string::npos) << message;
        EXPECT_NE(message.find("choose a coarser resolution"), std::string::npos) << message;
        EXPECT_NE(message.find("cells (the limit is"), std::string::npos) << message;
    }
    EXPECT_TRUE(threw) << "a micrometre resolution was not refused";
    EXPECT_TRUE(chunk.orthomosaic() == before);  // nothing was committed
}

TEST(BuildOrthomosaic, MemoryGuardStillAllowsARealisticRun) {
    // With a generous budget the fixture's 1 m ortho must go through: the guard
    // is a floor, not a tax.
    ScopedMemoryBudget budget("4096");
    OrthoSetup& s = setup();
    Chunk& chunk = *s.document.active_chunk();
    Params params = ortho_params("mosaic", "elevation", 1.0);
    EXPECT_NO_THROW(run_task(builtin_registry(), s.document, "BuildOrthomosaic", params));
    EXPECT_FALSE(chunk.orthomosaic().empty());
}

TEST(BuildOrthomosaic, AutomaticResolutionIsCoarsenedToFitTheBudget) {
    // The key behaviour change: with the DEFAULT resolution (0) a run that does
    // not fit must be coarsened to the finest value that fits, complete, and say
    // so at Warning -- not refused.
    //
    // The fixture's measured GSD is about 0.12 m (60 m above ground, f = 500 px);
    // an 8 MB budget cannot hold that ortho (the 12 camera footprints alone are
    // ~26 MB), so the automatic run must land clearly coarser than the GSD while
    // still completing. No probe run is needed: the fixture's GSD is a scene
    // constant, not something this test has to measure.
    OrthoSetup& s = setup();
    Chunk& chunk = *s.document.active_chunk();
    const Orthomosaic before = chunk.orthomosaic();

    ScopedMemoryBudget budget("8");
    ScopedLogCapture capture;
    Params params = ortho_params();  // resolution 0 = automatic
    ASSERT_NO_THROW(run_task(builtin_registry(), s.document, "BuildOrthomosaic", params));

    const Orthomosaic& ortho = chunk.orthomosaic();
    ASSERT_FALSE(ortho.empty());
    EXPECT_FALSE(ortho == before) << "the coarsened run did not commit";
    const double coarsened_cell = ortho.raster.grid.pixel_size_x;
    std::printf("[BuildOrthomosaic.Coarsen] coarsened=%.4f\n", coarsened_cell);
    EXPECT_TRUE(std::isfinite(coarsened_cell));
    // The GSD default is ~0.12 m; the coarsened cell must be unmistakably above
    // it, otherwise the budget was not the constraint.
    EXPECT_GT(coarsened_cell, 0.25)
        << "the automatic run was not coarsened at all";

    // The warning names the GSD, the actual cell, the budget and how to override
    // it.
    const std::string warnings = capture.text(LogLevel::Warn);
    EXPECT_NE(warnings.find("ground sample distance is"), std::string::npos) << warnings;
    EXPECT_NE(warnings.find("building at"), std::string::npos) << warnings;
    EXPECT_NE(warnings.find("Set resolution explicitly to override"), std::string::npos)
        << warnings;
    EXPECT_NE(warnings.find("budget"), std::string::npos) << warnings;
}

TEST(BuildOrthomosaic, ExplicitResolutionIsStillRefusedWhenItCannotFit) {
    // An EXPLICIT cell is the user's decision: the task must not silently build
    // at another scale. It keeps the existing actionable refusal.
    OrthoSetup& s = setup();
    Chunk& chunk = *s.document.active_chunk();
    const Orthomosaic before = chunk.orthomosaic();

    ScopedMemoryBudget budget("8");
    Params params = ortho_params("mosaic", "elevation", 0.02);  // far finer than default
    bool threw = false;
    try {
        run_task(builtin_registry(), s.document, "BuildOrthomosaic", params);
    } catch (const ValueError& error) {
        threw = true;
        const std::string message = error.what();
        EXPECT_NE(message.find("BuildOrthomosaic needs about"), std::string::npos) << message;
        EXPECT_NE(message.find("at resolution 0.02"), std::string::npos) << message;
        EXPECT_NE(message.find("Set a coarser resolution"), std::string::npos) << message;
        // It must NOT have been silently coarsened into a different run.
        EXPECT_EQ(message.find("building at"), std::string::npos) << message;
    }
    EXPECT_TRUE(threw) << "an explicit resolution that cannot fit was not refused";
    EXPECT_TRUE(chunk.orthomosaic() == before);  // nothing was committed
}

// ---------------------------------------------------------------------------
// 6. Export
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

// ---------------------------------------------------------------------------
// 9. The automatic-resolution rule: true ground sample distance
// ---------------------------------------------------------------------------
//
// The default cell must follow the GSD a pixel actually covers on the ground,
// not "camera height / focal". The two agree for a nadir camera over level
// ground and differ by sec(incidence) for an oblique one, which is the case a
// circular flight (the reporter's dataset) exercises. These cases measure the
// shared helper against the analytic footprint, so a regression that reverts to
// the nadir formula fails here even if the whole-scene test still passes.

namespace {

/// A flat, infinite ground plane at height 0 with a square metre stride.
ortho::SurfaceHeight flat_ground() {
    return [](double, double) { return 0.0; };
}

/// One camera at `centre` looking along `direction`, with focal length
/// `focal_px`. `direction` is normalised by the rule itself.
ortho::FootprintCamera make_footprint_camera(const Eigen::Vector3d& centre,
                                             const Eigen::Vector3d& direction,
                                             double focal_px) {
    ortho::FootprintCamera camera;
    camera.centre = centre;
    camera.direction = direction;
    camera.focal_px = focal_px;
    return camera;
}

}  // namespace

TEST(OrthoAutoResolution, NadirMatchesHeightOverFocal) {
    // 50 m above level ground, f = 3000 px: one pixel covers 50/3000 m.
    const double expected = 50.0 / 3000.0;
    std::vector<ortho::FootprintCamera> cameras = {
        make_footprint_camera(Eigen::Vector3d(0, 0, 50), Eigen::Vector3d(0, 0, -1), 3000.0)};
    const double gsd = ortho::median_ground_sample_distance(cameras, flat_ground(), 1.0);
    std::printf("[OrthoAutoResolution.Nadir] gsd=%.6f expected=%.6f\n", gsd, expected);
    ASSERT_GT(gsd, 0.0);
    // The rule rounds to three significant figures; allow that plus the march's
    // one-stride landing error on a 50 m drop (stride 1 m here).
    EXPECT_NEAR(gsd, expected, 0.1 * expected);
}

TEST(OrthoAutoResolution, ObliqueIsTwiceTheNadirFootprintAt45Degrees) {
    // A 45-degree camera 50 m up: the principal ray lands 50 m out, the slant
    // range is 50*sqrt(2), and the footprint along the line of sight is that
    // range / focal / cos(45) = 2 * (50 / focal).
    const double nadir = 50.0 / 3000.0;
    const Eigen::Vector3d direction = Eigen::Vector3d(1.0, 0.0, -1.0).normalized();
    std::vector<ortho::FootprintCamera> cameras = {
        make_footprint_camera(Eigen::Vector3d(0, 0, 50), direction, 3000.0)};
    const double gsd = ortho::median_ground_sample_distance(cameras, flat_ground(), 1.0);
    std::printf("[OrthoAutoResolution.Oblique45] gsd=%.6f nadir=%.6f ratio=%.4f\n", gsd, nadir,
                gsd / nadir);
    ASSERT_GT(gsd, 0.0);
    EXPECT_NEAR(gsd, 2.0 * nadir, 0.1 * 2.0 * nadir);
}

TEST(OrthoAutoResolution, DefaultFollowsTheMeasuredGsdNotAQuarterDemCell) {
    // The default ortho cell must be the measured GROUND sample distance, not a
    // fraction of the DEM cell. A 1 m DEM with photographs that resolve ~0.4 m
    // used to come out at 0.25 m (a quarter of the DEM cell): four times the run
    // for detail the imagery does not have. Here the focal length is lowered to
    // 150 px so the analytic ground GSD is clearly above the new 1/8-cell sanity
    // floor, and the DEM is forced to a 1 m cell while the cameras are unchanged.
    Document document = aerial_fixture::make_truth_chunk_document();
    Chunk& chunk = *document.active_chunk();

    // Work in the fixture's own LOCAL ENU frame (identity transform, epsg 0), so
    // the synthetic DEM below is in the same metres as the poses and no UTM
    // projection is involved in the comparison.
    chunk.transform() = Transform::Identity();
    chunk.crs().epsg = 0;

    const double f_px = 150.0;
    for (Sensor& sensor : chunk.sensors()) {
        sensor.calibration =
            Calibration::initial(CameraType::Frame, aerial_fixture::kImageWidth,
                                 aerial_fixture::kImageHeight, f_px);
    }

    // A 1 m DEM over the fixture area, sampled from the analytic terrain. No
    // camera's principal ray lands on the building, so the terrain alone is the
    // surface the measurement sees.
    constexpr double kCell = 1.0;
    Elevation& elevation = chunk.elevation();
    elevation.epsg = 0;
    elevation.source = "depth_maps";
    elevation.nodata = -32767.0F;
    RasterGrid& grid = elevation.raster.grid;
    grid.origin_x = aerial_fixture::kAreaX0;
    grid.origin_y = aerial_fixture::kAreaY1;
    grid.pixel_size_x = kCell;
    grid.pixel_size_y = kCell;
    grid.width = static_cast<int>(
        std::ceil((aerial_fixture::kAreaX1 - aerial_fixture::kAreaX0) / kCell));
    grid.height = static_cast<int>(
        std::ceil((aerial_fixture::kAreaY1 - aerial_fixture::kAreaY0) / kCell));
    elevation.raster.channels = 1;
    elevation.raster.data.resize(static_cast<std::size_t>(grid.width) *
                                 static_cast<std::size_t>(grid.height));
    for (int row = 0; row < grid.height; ++row) {
        for (int col = 0; col < grid.width; ++col) {
            const Eigen::Vector2d c = grid.pixel_center(col, row);
            elevation.raster.at(col, row) =
                static_cast<float>(aerial_fixture::truth_height(c.x(), c.y()));
        }
    }

    // The analytic ground GSD: slant range to the surface / (f * cos incidence),
    // median over the cameras. This is the value the task's ray march estimates.
    const std::vector<aerial_fixture::TruthPose> poses = aerial_fixture::load_truth();
    std::vector<double> analytic;
    analytic.reserve(poses.size());
    for (const aerial_fixture::TruthPose& pose : poses) {
        const Eigen::Vector3d direction = pose.R.transpose() * Eigen::Vector3d(0, 0, 1);
        double s = 0.0;
        Eigen::Vector3d hit = pose.centre;
        for (; s < 500.0; s += 0.01) {
            hit = pose.centre + s * direction;
            if (hit.z() <= aerial_fixture::truth_height(hit.x(), hit.y())) {
                break;
            }
        }
        const double cos_incidence = std::abs(direction.z());
        analytic.push_back((hit - pose.centre).norm() / (f_px * cos_incidence));
    }
    std::sort(analytic.begin(), analytic.end());
    const double analytic_median = 0.5 * (analytic[analytic.size() / 2 - 1] +
                                          analytic[analytic.size() / 2]);

    Params params = ortho_params();  // resolution 0 = automatic
    run_task(builtin_registry(), document, "BuildOrthomosaic", params);

    const Orthomosaic& ortho = chunk.orthomosaic();
    ASSERT_FALSE(ortho.empty());
    const double cell = ortho.raster.grid.pixel_size_x;
    std::printf("[OrthoAutoResolution.Default] cell=%.4f analytic=%.4f ratio=%.3f "
                "(dem %.2f, quarter-cell would be %.3f)\n",
                cell, analytic_median, cell / analytic_median, kCell, 0.25 * kCell);
    EXPECT_NEAR(cell, analytic_median, 0.10 * analytic_median);
    // And explicitly not the old quarter-DEM-cell floor, which is 35% below.
    EXPECT_GT(cell, 1.2 * (0.25 * kCell));
}

TEST(OrthoAutoResolution, MedianOverTheFlightFootprint) {
    // A mixed nadir/45-degree flight: the median is the middle order statistic,
    // here the nadir value (2 cameras nadir at 50 m, 1 oblique at 100 m so its
    // footprint is also "2x", i.e. equal; use heights 50, 50 and 100 to make the
    // nadir footprint the median).
    const double nadir = 50.0 / 3000.0;
    std::vector<ortho::FootprintCamera> cameras = {
        make_footprint_camera(Eigen::Vector3d(0, 0, 50), Eigen::Vector3d(0, 0, -1), 3000.0),
        make_footprint_camera(Eigen::Vector3d(10, 0, 50), Eigen::Vector3d(0, 0, -1), 3000.0),
        make_footprint_camera(Eigen::Vector3d(20, 0, 100), Eigen::Vector3d(0.3, 0.0, -1.0),
                              3000.0)};
    const double gsd = ortho::median_ground_sample_distance(cameras, flat_ground(), 1.0);
    std::printf("[OrthoAutoResolution.Median] gsd=%.6f nadir=%.6f\n", gsd, nadir);
    EXPECT_NEAR(gsd, nadir, 0.1 * nadir);
}

TEST(OrthoAutoResolution, UnusableCamerasContributeNothing) {
    // A horizontal camera never reaches the ground; a camera with no focal has
    // no footprint. Neither may poison the median.
    std::vector<ortho::FootprintCamera> cameras = {
        make_footprint_camera(Eigen::Vector3d(0, 0, 50), Eigen::Vector3d(1, 0, 0), 3000.0),
        make_footprint_camera(Eigen::Vector3d(0, 0, 50), Eigen::Vector3d(0, 0, -1), 0.0)};
    EXPECT_DOUBLE_EQ(ortho::median_ground_sample_distance(cameras, flat_ground(), 1.0), 0.0);
    EXPECT_DOUBLE_EQ(ortho::median_ground_sample_distance({}, flat_ground(), 1.0), 0.0);
}

TEST(OrthoAutoResolution, GeographicGridIsInDegreesNotMetres) {
    // The metric rule is expressed in metres; an EPSG:4326 grid's pixel size is
    // in degrees. SurfaceFrame must therefore report the per-degree arc lengths
    // so a metre GSD can be converted. Checked on a georeferenced chunk placed
    // at 48.7 N (no photos needed: the frame only reads the transform).
    Document document;
    Chunk& chunk = document.workspace().add_chunk("geo");
    document.set_active_chunk_id(chunk.id());
    chunk.transform() = geo::enu_to_ecef(geo::Geodetic{9.2, 48.7, 300.0});
    chunk.crs().epsg = 4326;

    const elevation::SurfaceFrame geographic = elevation::SurfaceFrame::for_chunk(chunk, 4326);
    ASSERT_EQ(geographic.epsg(), 4326);
    const Eigen::Vector2d metres_per_degree = geographic.metres_per_plane_unit(48.7);
    // One degree of latitude is ~111.1 km; of longitude ~73.5 km at 48.7 N.
    EXPECT_NEAR(metres_per_degree.y(), 111132.0, 500.0);
    EXPECT_NEAR(metres_per_degree.x(), 73500.0, 800.0);

    // A 1 cm GSD becomes ~1 cm / metres-per-degree degrees: a number three
    // orders of magnitude below a degree, never 0.01 (a metre value used as a
    // degree cell) and never 1 (a bare metric fallback).
    //
    // The expectation is computed INDEPENDENTLY of the frame, from the WGS84
    // meridian arc and the parallel radius at 48.7 N, so it is not the same
    // expression the implementation evaluates:
    //
    //   meridian arc per degree  M(lat) * pi/180,
    //   parallel arc per degree  N(lat) * cos(lat) * pi/180,
    //   M = a(1-e^2)/w^(3/2), N = a/sqrt(w), w = 1 - e^2 sin^2(lat).
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kA = 6378137.0;
    constexpr double kF = 1.0 / 298.257223563;
    constexpr double kLatDeg = 48.7;
    const double e2 = 2.0 * kF - kF * kF;
    const double lat_rad = kLatDeg * kPi / 180.0;
    const double sin_lat = std::sin(lat_rad);
    const double w = 1.0 - e2 * sin_lat * sin_lat;
    const double meridian_per_deg = kA * (1.0 - e2) / (w * std::sqrt(w)) * kPi / 180.0;
    const double parallel_per_deg = kA / std::sqrt(w) * std::cos(lat_rad) * kPi / 180.0;
    // The frame's own arc lengths must agree with those to well inside a metre,
    // which is what makes the conversion below meaningful at all.
    EXPECT_NEAR(metres_per_degree.y(), meridian_per_deg, 1.0);
    EXPECT_NEAR(metres_per_degree.x(), parallel_per_deg, 1.0);

    // The 1 cm cell is 0.01 divided by the LARGER arc length (the meridian, y),
    // taken from the independent formula rather than the frame's accessor.
    const double cell_degrees = 0.01 / std::max(metres_per_degree.x(), metres_per_degree.y());
    const double expected_independent = 0.01 / meridian_per_deg;
    EXPECT_NEAR(cell_degrees, expected_independent, 0.01 * expected_independent);
    EXPECT_LT(cell_degrees, 1e-6);
}

TEST(OrthoAutoResolution, LocalAndUtmPlanesAreMetres) {
    // Local/UTM frames must report exactly 1 m per plane unit on both axes, the
    // identity the metric rule relies on.
    Document local;
    Chunk& chunk = local.workspace().add_chunk("local");
    local.set_active_chunk_id(chunk.id());
    const elevation::SurfaceFrame frame = elevation::SurfaceFrame::for_chunk(chunk, 0);
    ASSERT_TRUE(frame.is_local());
    EXPECT_DOUBLE_EQ(frame.metres_per_plane_unit(0.0).x(), 1.0);
    EXPECT_DOUBLE_EQ(frame.metres_per_plane_unit(1234.0).y(), 1.0);
}

}  // namespace
}  // namespace alphashape
