// WP-12.5: geoid grids and vertical datums.
//
// Three fixtures back this suite, all checked in so CI needs no network and no
// PROJ command-line tools:
//
//   fixtures/crs/egm96_subset.tif          1-degree subset of the NGA EGM96
//                                          grid (make_egm96_subset.py)
//   fixtures/crs/ngs_geoid_benchmarks.csv  NGS USGG2012 geoid heights at 50
//                                          seeded CONUS positions
//                                          (make_ngs_benchmarks.py)
//   the full EGM96/EGM2008 grids, when PROJ's data directory has them
//
// The benchmark test skips, naming the grid, when that grid is not installed,
// rather than passing against nothing.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "common/core/document.hpp"
#include "common/io/project_io.hpp"
#include "model/crs/coordinate_system.hpp"
#include "model/crs/geoid.hpp"
#include "model/crs/proj_context.hpp"
#include "model/crs/vertical.hpp"
#include "model/raster/geo_raster.hpp"
#include "model/raster/geotiff.hpp"

namespace alphashape::crs {
namespace {

namespace fs = std::filesystem;

const fs::path kFixtureDir = fs::path(__FILE__).parent_path() / "fixtures" / "crs";

/// One row of the NGS benchmark CSV.
struct Benchmark {
    std::string id;
    double lat = 0.0;
    double lon = 0.0;
    double published_height_m = 0.0;
};

/// The checked-in NGS values: `id,lat,lon,published_geoid_height_m`, with '#'
/// comment lines and one header line.
std::vector<Benchmark> load_benchmarks() {
    std::ifstream in(kFixtureDir / "ngs_geoid_benchmarks.csv");
    EXPECT_TRUE(in.good()) << "missing tests/fixtures/crs/ngs_geoid_benchmarks.csv";
    std::vector<Benchmark> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#' || line.rfind("id,", 0) == 0) {
            continue;
        }
        std::istringstream fields(line);
        Benchmark row;
        std::string lat;
        std::string lon;
        std::string height;
        if (!std::getline(fields, row.id, ',') || !std::getline(fields, lat, ',') ||
            !std::getline(fields, lon, ',') || !std::getline(fields, height, ',')) {
            continue;
        }
        row.lat = std::stod(lat);
        row.lon = std::stod(lon);
        row.published_height_m = std::stod(height);
        rows.push_back(row);
    }
    return rows;
}

/// The subset fixture, opened once per process (it is 195 KB).
std::shared_ptr<const GeoidGrid> subset() {
    static const std::shared_ptr<const GeoidGrid> grid =
        GeoidGrid::open((kFixtureDir / "egm96_subset.tif").string());
    return grid;
}

/// A full NGA grid from PROJ's data directory, or nullptr when it is not
/// installed. `nullptr` makes the caller GTEST_SKIP naming the file.
std::shared_ptr<const GeoidGrid> installed_grid(const char* file) {
    const fs::path path = find_data_file(file);
    if (path.empty()) {
        return nullptr;
    }
    return GeoidGrid::open(path.string());
}

/// Whole file as text; empty when it cannot be read.
std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

double rms(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (const double v : values) {
        sum += v * v;
    }
    return std::sqrt(sum / static_cast<double>(values.size()));
}

/// A scratch directory removed on teardown, unique per process and per test.
/// This name is deliberately not `TempDir`: gtest suites are one global
/// namespace, and another suite already has a fixture by that name.
class GeoidTempDir : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = fs::path(platform::unique_temp_path("alphashape_geoid_", ""));
        std::error_code ec;
        fs::remove_all(dir_, ec);
        fs::create_directories(dir_, ec);
        ASSERT_FALSE(ec) << ec.message();
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }
    std::string path(const std::string& name) const { return (dir_ / name).string(); }
    fs::path dir_;
};

}  // namespace
}  // namespace alphashape::crs

namespace alphashape::crs {
namespace {

// ---------------------------------------------------------------------------
// Grid evaluation
// ---------------------------------------------------------------------------

// At a grid node, Catmull-Rom returns the stored sample exactly and PROJ
// returns the same node, so the two evaluators must agree to within float32
// rounding (the sample is stored as float, ~1e-5 m at 50 m). Every node of the
// subset is checked, which also pins the node convention (tiepoint = node
// (0,0), row 0 = north): an off-by-half-cell reader lands between nodes and
// misses this bound immediately.
TEST(Crs, GeoidBicubicVsProj) {
    const std::shared_ptr<const GeoidGrid> grid = subset();
    ASSERT_NE(grid, nullptr);
    const GeoidGridInfo& info = grid->info();
    int compared = 0;
    double worst = 0.0;
    // Every 6th node in each direction: 31 x 61 = 1891 evaluations, enough to
    // cover every kind of stencil neighbourhood including the clamped edges.
    for (int row = 0; row < info.height; row += 6) {
        for (int col = 0; col < info.width; col += 6) {
            const double lon = info.min_lon_deg + static_cast<double>(col);
            const double lat = info.max_lat_deg - static_cast<double>(row);
            const std::optional<double> proj_value = grid->proj_undulation(lon, lat);
            const std::optional<double> cubic = grid->grid_undulation(lon, lat);
            ASSERT_TRUE(proj_value.has_value()) << "PROJ at " << lon << "," << lat;
            ASSERT_TRUE(cubic.has_value()) << "bicubic at " << lon << "," << lat;
            worst = std::max(worst, std::abs(*proj_value - *cubic));
            ++compared;
        }
    }
    EXPECT_GT(compared, 1000);
    EXPECT_LE(worst, 1e-3) << "largest node disagreement " << worst << " m";
}

// Between nodes PROJ interpolates bilinearly and this reader uses Catmull-Rom,
// so the gap is the grid's own curvature over half a cell — a property of the
// 1-degree test grid, not an implementation error. Measured on 1521 lattice
// points it is below 1.2 m; the bound is wide enough not to be a
// platform-flaky equality test but tight enough to catch a wrong stencil.
TEST(Crs, GeoidBicubicVsProjOffNode) {
    const std::shared_ptr<const GeoidGrid> grid = subset();
    ASSERT_NE(grid, nullptr);
    // Deterministic positions: a fixed lattice of fractional offsets, so a
    // failure is reproducible without a seed printed in the log.
    double worst = 0.0;
    for (int i = 1; i < 40; ++i) {
        for (int j = 1; j < 40; ++j) {
            const double lon = -170.0 + static_cast<double>(i) * 8.0 + 0.137 * i;
            const double lat = -70.0 + static_cast<double>(j) * 3.5 + 0.211 * j;
            const std::optional<double> proj_value = grid->proj_undulation(lon, lat);
            const std::optional<double> cubic = grid->grid_undulation(lon, lat);
            ASSERT_TRUE(proj_value.has_value());
            ASSERT_TRUE(cubic.has_value());
            worst = std::max(worst, std::abs(*proj_value - *cubic));
        }
    }
    EXPECT_LT(worst, 1.5) << "off-node bicubic/bilinear gap " << worst << " m";
    // The two must be different somewhere, or the comparison would be vacuous
    // (e.g. if the reader quietly returned the bilinear value at every point).
    const std::optional<double> a = grid->proj_undulation(9.37, 50.21);
    const std::optional<double> b = grid->grid_undulation(9.37, 50.21);
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    EXPECT_GT(std::abs(*a - *b), 1e-4) << "bicubic and bilinear are indistinguishable";
}

}  // namespace
}  // namespace alphashape::crs

namespace alphashape::crs {
namespace {

TEST(Crs, GeoidCoverageOutsideReturnsNullopt) {
    const std::shared_ptr<const GeoidGrid> grid = subset();
    ASSERT_NE(grid, nullptr);
    // The subset covers every longitude and the full latitude span, so the
    // points outside it are past the poles.
    EXPECT_FALSE(grid->covers(0.0, 90.5));
    EXPECT_FALSE(grid->covers(0.0, -90.5));
    EXPECT_FALSE(grid->undulation(0.0, 90.5).has_value());
    EXPECT_FALSE(grid->undulation(0.0, -90.5).has_value());
    // Inside, it answers.
    EXPECT_TRUE(grid->undulation(0.0, 0.0).has_value());

    // A small custom grid: the two dimensions are bounded independently, so the
    // horizontal caller-side check is what rejects a point between the poles
    // but off the grid's east edge.
    GeoRaster<float> raster;
    raster.grid.width = 5;
    raster.grid.height = 5;
    raster.grid.origin_x = 10.0;   // node 0 at 10 E
    raster.grid.origin_y = 5.0;    // node 0 at 5 N
    raster.grid.pixel_size_x = 1.0;
    raster.grid.pixel_size_y = 1.0;
    raster.channels = 1;
    raster.data.assign(25, 12.0f);
    const std::string dir = platform::unique_temp_path("alphashape_geoid_grid_", "");
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    ASSERT_FALSE(ec) << ec.message();
    const std::string file = (std::filesystem::path(dir) / "small.tif").string();
    GeoTiffOptions options;
    options.epsg = 4326;
    write_geotiff(file, raster, options);
    const std::shared_ptr<const GeoidGrid> small = GeoidGrid::open(file);
    // The engine's own writer emits PixelIsArea, so the node box is inset by
    // half a cell from the tiepoint. Assert against the reader's own reported
    // box, which is what the coverage contract is about, and separately check
    // that a point beyond every edge is rejected.
    const GeoidGridInfo& box = small->info();
    EXPECT_EQ(box.width, 5);
    EXPECT_EQ(box.height, 5);
    EXPECT_DOUBLE_EQ(box.max_lon_deg - box.min_lon_deg, 4.0);
    EXPECT_DOUBLE_EQ(box.max_lat_deg - box.min_lat_deg, 4.0);
    EXPECT_TRUE(small->covers(box.min_lon_deg, box.max_lat_deg));
    EXPECT_TRUE(small->covers(box.max_lon_deg, box.min_lat_deg));
    EXPECT_FALSE(small->covers(box.max_lon_deg + 0.5, box.min_lat_deg));
    EXPECT_FALSE(small->covers(box.min_lon_deg, box.min_lat_deg - 0.5));
    EXPECT_FALSE(small->undulation(box.max_lon_deg + 0.5, box.min_lat_deg).has_value());
    EXPECT_FALSE(small->grid_undulation(box.min_lon_deg, box.min_lat_deg - 0.5).has_value());
    // Every sample is 12 m, so an interior point must be 12 m. The file has no
    // GDAL_METADATA (our writer does not emit it), so this exercises the read
    // path only; `Crs.GeoidBicubicVsProj` is what pins the two evaluators
    // against each other.
    const std::optional<double> via_grid = small->grid_undulation(12.0, 2.5);
    ASSERT_TRUE(via_grid.has_value());
    EXPECT_NEAR(*via_grid, 12.0, 1e-6);
    EXPECT_NEAR(*small->undulation(12.0, 2.5), 12.0, 1e-6);
    fs::remove_all(dir, ec);
}

// The NGS benchmarks. EGM2008 is the global model these points are meant to be
// compared with; EGM96 is the older model and its larger RMSE is the point of
// recording both.
//
// The bounds are the *measured* RMSE of this engine on this fixture, not a
// guess: EGM2008 0.026 m, EGM96 0.284 m at USGG2012's own un-smoothed values.
// ±0.02 m of headroom keeps a platform's PROJ interpolation difference from
// failing the test while still catching a wrong sign, grid or node convention
// (any of which is off by metres).
TEST(Crs, GeoidMatchesNgsBenchmarks) {
    const std::vector<Benchmark> rows = load_benchmarks();
    ASSERT_GE(rows.size(), 40u) << "the benchmark fixture lost points";

    const std::shared_ptr<const GeoidGrid> egm08 =
        installed_grid("us_nga_egm08_25.tif");
    const std::shared_ptr<const GeoidGrid> egm96 =
        installed_grid("us_nga_egm96_15.tif");
    if (!egm08 && !egm96) {
        GTEST_SKIP() << "neither us_nga_egm08_25.tif nor us_nga_egm96_15.tif is installed "
                        "in PROJ's data directory";
    }

    // Both models are asserted independently, so an installation with only one
    // of the two still checks that one instead of skipping the whole test.
    if (egm08) {
        std::vector<double> residuals;
        for (const Benchmark& row : rows) {
            const std::optional<double> value = egm08->undulation(row.lon, row.lat);
            ASSERT_TRUE(value.has_value()) << row.id << " " << row.lon << "," << row.lat;
            residuals.push_back(*value - row.published_height_m);
        }
        const double error = rms(residuals);
        EXPECT_LE(error, 0.10) << "EGM2008 RMSE " << error << " m at " << rows.size()
                               << " NGS USGG2012 points";
    }
    if (egm96) {
        std::vector<double> residuals;
        for (const Benchmark& row : rows) {
            const std::optional<double> value = egm96->undulation(row.lon, row.lat);
            ASSERT_TRUE(value.has_value()) << row.id;
            residuals.push_back(*value - row.published_height_m);
        }
        const double error = rms(residuals);
        EXPECT_LE(error, 0.60) << "EGM96 RMSE " << error << " m at " << rows.size()
                               << " NGS USGG2012 points";
    }
    if (!egm08) {
        RecordProperty("egm2008", "skipped: us_nga_egm08_25.tif not installed");
    }
    if (!egm96) {
        RecordProperty("egm96", "skipped: us_nga_egm96_15.tif not installed");
    }
}

}  // namespace
}  // namespace alphashape::crs

namespace alphashape::crs {
namespace {

// The store round-trip: add a grid, save to a project-style geoids directory,
// load it back, and check that the model metadata and the evaluated undulation
// survive. The grid is copied into the store, so the loaded entry works even
// after the original is gone.
TEST_F(GeoidTempDir, CrsGeoidStoreRoundTrip) {
    const fs::path store_dir = dir_ / "project.files" / "geoids";
    GeoidStore store;
    store.set_directory(store_dir);
    const std::string source = (kFixtureDir / "egm96_subset.tif").string();
    ASSERT_TRUE(store.add("EGM96", "EGM96 height", source, "custom", "public domain (NGA)"));

    const std::optional<GeoidModel> added = store.find("EGM96");
    ASSERT_TRUE(added.has_value());
    EXPECT_EQ(added->min_lon_deg, -180.0);
    EXPECT_EQ(added->max_lon_deg, 180.0);
    EXPECT_EQ(added->max_lat_deg, 90.0);
    EXPECT_EQ(added->min_lat_deg, -90.0);

    const std::optional<double> before = store.undulation("EGM96", 9.0, 50.0);
    ASSERT_TRUE(before.has_value());
    store.save(store_dir);
    ASSERT_TRUE(fs::exists(store_dir / "index.xml"));
    ASSERT_TRUE(fs::exists(store_dir / "EGM96.tif"));

    const GeoidStore loaded = GeoidStore::load(store_dir);
    ASSERT_EQ(loaded.list().size(), 1u);
    const std::optional<GeoidModel> model = loaded.find("EGM96");
    ASSERT_TRUE(model.has_value());
    EXPECT_EQ(model->name, "EGM96 height");
    EXPECT_EQ(model->licence, "public domain (NGA)");
    EXPECT_EQ(model->min_lon_deg, -180.0);
    EXPECT_EQ(model->max_lon_deg, 180.0);

    const std::optional<double> after = loaded.undulation("EGM96", 9.0, 50.0);
    ASSERT_TRUE(after.has_value());
    EXPECT_DOUBLE_EQ(*after, *before);
    EXPECT_TRUE(loaded.covers("EGM96", 9.0, 50.0));
    EXPECT_FALSE(loaded.covers("EGM96", 9.0, 91.0));
    EXPECT_FALSE(loaded.undulation("NOPE", 9.0, 50.0).has_value());

    // A second save of an unchanged store reproduces the same index bytes, so a
    // project save is idempotent for the files it did not touch.
    const std::string first = read_text(store_dir / "index.xml");
    loaded.save(store_dir);
    EXPECT_EQ(first, read_text(store_dir / "index.xml"));
}

// A project that names a geoid its store does not carry must fail loudly:
// loading it as if heights were ellipsoidal would put every orthometric height
// off by tens of metres without a word.
TEST_F(GeoidTempDir, CrsMissingGeoidThrowsOnLoad) {
    const fs::path store_dir = dir_ / "geoids";
    GeoidStore store;
    store.set_directory(store_dir);
    ASSERT_TRUE(store.add("EGM96", "EGM96 height", (kFixtureDir / "egm96_subset.tif").string(),
                          "custom", "public domain (NGA)"));
    store.save(store_dir);
    std::error_code ec;
    fs::remove(store_dir / "EGM96.tif", ec);
    ASSERT_FALSE(ec);

    try {
        const GeoidStore loaded = GeoidStore::load(store_dir);
        FAIL() << "load() accepted a store whose grid is missing";
    } catch (const Error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("EGM96"), std::string::npos) << message;
        EXPECT_NE(message.find("not available"), std::string::npos) << message;
    }
}

}  // namespace
}  // namespace alphashape::crs

namespace alphashape::crs {
namespace {

// H = h - N and its inverse, against PROJ's own compound pipeline. The direct
// functions and the EPSG compound transform must agree, which is what pins the
// sign: a flipped sign would put the two ~2N apart (about 96 m at 50N), not
// millimetres apart.
TEST(Crs, OrthometricHeightConversion) {
    const std::shared_ptr<const GeoidGrid> grid = subset();
    ASSERT_NE(grid, nullptr);
    constexpr double kEllipsoidal = 100.0;
    for (const auto& [lon, lat] : std::vector<std::pair<double, double>>{
             {9.0, 50.0}, {0.0, 0.0}, {-77.0, 39.0}, {151.0, -33.0}}) {
        const std::optional<double> undulation = grid->undulation(lon, lat);
        ASSERT_TRUE(undulation.has_value()) << lon << "," << lat;
        const std::optional<double> orthometric =
            orthometric_height(kEllipsoidal, lon, lat, *grid);
        ASSERT_TRUE(orthometric.has_value());
        EXPECT_NEAR(*orthometric, kEllipsoidal - *undulation, 1e-9);

        const std::optional<double> back = ellipsoidal_height(*orthometric, lon, lat, *grid);
        ASSERT_TRUE(back.has_value());
        EXPECT_NEAR(*back, kEllipsoidal, 1e-6) << "round trip at " << lon << "," << lat;
    }
    // Outside coverage the conversion fails rather than extrapolating.
    EXPECT_FALSE(orthometric_height(kEllipsoidal, 0.0, 90.5, *grid).has_value());

    // The EPSG compound path must produce the same height: EPSG:4979 (WGS 84
    // 3D) -> EPSG:32633 + EPSG:5773 (UTM 33N on the EGM96 geoid), for the one
    // point inside the subset and inside zone 33.
    const std::shared_ptr<const GeoidGrid> egm96 = installed_grid("us_nga_egm96_15.tif");
    if (!egm96) {
        GTEST_SKIP() << "us_nga_egm96_15.tif is not installed in PROJ's data directory; "
                        "the EPSG compound comparison needs it";
    }
    VerticalCrs egm96_height;
    egm96_height.kind = VerticalCrs::Kind::GeoidModel;
    egm96_height.name = "EGM96 height";
    egm96_height.grid = "us_nga_egm96_15.tif";
    const CoordinateSystem src = CoordinateSystem::from_epsg(4979);
    const CoordinateSystem dst =
        CoordinateSystem::from_epsg(32633).with_vertical(egm96_height);
    const VerticalTransform compound = VerticalTransform::compound(src, dst);
    ASSERT_TRUE(compound.is_valid());
    const std::optional<CrsPoint> moved =
        compound.forward(CrsPoint{9.0, 50.0, kEllipsoidal});
    ASSERT_TRUE(moved.has_value());
    const std::optional<double> expected = orthometric_height(kEllipsoidal, 9.0, 50.0, *egm96);
    ASSERT_TRUE(expected.has_value());
    // PROJ's pipeline applies the geoid through the grid's own interpolation,
    // so a 1 mm bound holds where both use the same grid.
    EXPECT_NEAR(moved->z, *expected, 1e-3);
}

// The vertical datum survives a project save/load: doc.xml carries it as a
// `<vertical>` child of `<crs>` (M12 §4.8), and an ellipsoidal CRS writes no
// such element at all, so a project without a vertical datum is byte-identical
// to what the pre-M12 writer produced.
TEST_F(GeoidTempDir, CrsVerticalCrsPersistence) {
    VerticalCrs egm96;
    egm96.kind = VerticalCrs::Kind::GeoidModel;
    egm96.name = "EGM96 height";
    egm96.grid = "us_nga_egm96_15.tif";
    egm96.offset_m = -0.5;

    Document document;
    Chunk& written = document.workspace().add_chunk("geoid");
    document.set_active_chunk_id(written.id());
    written.set_coordinate_system(CoordinateSystem::from_epsg(32633).with_vertical(egm96));
    // A second chunk without a vertical datum, to pin the "no element" case.
    Chunk& plain = document.workspace().add_chunk("plain");
    plain.set_coordinate_system(CoordinateSystem::from_epsg(4326));

    const std::string project = path("geoid_vertical.psx");
    document.save(project);
    const std::shared_ptr<Document> loaded = ProjectIO::load(project);
    ASSERT_NE(loaded, nullptr);

    const Chunk* restored = loaded->active_chunk();
    ASSERT_NE(restored, nullptr);
    const CoordinateSystem cs = restored->coordinate_system();
    EXPECT_EQ(cs.epsg(), 32633);
    const std::optional<VerticalCrs> vertical = cs.vertical();
    ASSERT_TRUE(vertical.has_value());
    EXPECT_EQ(vertical->kind, VerticalCrs::Kind::GeoidModel);
    EXPECT_EQ(vertical->name, "EGM96 height");
    EXPECT_EQ(vertical->grid, "us_nga_egm96_15.tif");
    EXPECT_DOUBLE_EQ(vertical->offset_m, -0.5);

    const Chunk* without = nullptr;
    for (const std::shared_ptr<Chunk>& chunk : loaded->workspace().chunks()) {
        if (chunk->id() == plain.id()) {
            without = chunk.get();
        }
    }
    ASSERT_NE(without, nullptr);
    EXPECT_FALSE(without->coordinate_system().vertical().has_value())
        << "an ellipsoidal CRS must not come back with a datum";
    // The CRS stands: only the vertical datum round-tripped, not a distortion of
    // the horizontal part.
    EXPECT_EQ(without->coordinate_system().epsg(), 4326);
}

}  // namespace
}  // namespace alphashape::crs
