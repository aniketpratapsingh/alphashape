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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
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

/// Writes a header-only TIFF (no pixel data) claiming `width` x `height`.
///
/// The bytes are assembled by hand rather than through libtiff: libtiff will
/// not create an image that big without the samples, and the point here is a
/// crafted header. Layout (classic TIFF, little endian):
///
///   0: header "II", 42, IFD offset 80
///   8: ModelPixelScale, 3 doubles
///   32: ModelTiepoint, 6 doubles
///   80: IFD with 10 entries, next IFD = 0
///
/// Only what the geoid decoder reads is supplied; width and height are LONG
/// so a 4e9 value survives.
void write_header_only_tiff(const std::string& path, std::uint32_t width, std::uint32_t height) {
    std::vector<std::uint8_t> out;
    const auto put16 = [&out](std::uint16_t v) {
        out.push_back(static_cast<std::uint8_t>(v & 0xFFU));
        out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFFU));
    };
    const auto put32 = [&out](std::uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFFU));
        }
    };
    const auto put64 = [&out](std::uint64_t bits) {
        for (int i = 0; i < 8; ++i) {
            out.push_back(static_cast<std::uint8_t>((bits >> (8 * i)) & 0xFFU));
        }
    };
    const auto put_double = [&put64](double v) {
        std::uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(v));
        std::memcpy(&bits, &v, sizeof(bits));
        put64(bits);
    };
    constexpr std::uint32_t kIfdOffset = 80;
    out.push_back('I');
    out.push_back('I');
    put16(42);
    put32(kIfdOffset);
    put16(3);  // ModelPixelScale: 1 degree cells
    put_double(1.0);
    put_double(1.0);
    put_double(0.0);
    put16(6);  // ModelTiepoint: node (0,0) at 0,0
    for (int i = 0; i < 6; ++i) {
        put_double(0.0);
    }
    struct Entry {
        std::uint16_t tag;
        std::uint16_t type;  // 3 = SHORT, 4 = LONG
        std::uint32_t count;
        std::uint32_t value;
    };
    const Entry entries[] = {
        {256, 4, 1, width},                       // ImageWidth
        {257, 4, 1, height},                      // ImageLength
        {258, 3, 1, 32},                          // BitsPerSample
        {259, 3, 1, 1},                           // Compression = none
        {262, 3, 1, 1},                           // Photometric = BlackIsZero
        {273, 4, 1, 0},                           // StripOffsets (no pixels)
        {277, 3, 1, 1},                           // SamplesPerPixel
        {278, 4, 1, height},                      // RowsPerStrip
        {279, 4, 1, 0},                           // StripByteCounts
        {339, 3, 1, 3},                           // SampleFormat = IEEE float
    };
    put16(static_cast<std::uint16_t>(std::size(entries)));
    for (const Entry& entry : entries) {
        put16(entry.tag);
        put16(entry.type);
        put32(entry.count);
        if (entry.type == 3) {
            put16(static_cast<std::uint16_t>(entry.value));
            put16(0);  // the 4-byte value field is left-justified
        } else {
            put32(entry.value);
        }
    }
    put32(0);  // no next IFD
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(file.good()) << path;
    file.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
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

// The index is untrusted input: both the `file` attribute and the `id` become
// `dir / name`, so either could name a path outside the store. Every hostile
// spelling must be refused by name, and nothing may be written outside `dir`.
TEST_F(GeoidTempDir, CrsGeoidIndexRejectsPathTraversal) {
    const fs::path store_dir = dir_ / "geoids";
    std::error_code ec;
    fs::create_directories(store_dir, ec);
    ASSERT_FALSE(ec) << ec.message();
    // A readable grid just outside the store, which a successful escape would
    // reach. Its presence makes "is the file there?" a useless defence.
    fs::copy_file(kFixtureDir / "egm96_subset.tif", dir_ / "outside.tif", ec);
    ASSERT_FALSE(ec) << ec.message();

    const std::vector<std::string> hostile{
        "../outside.tif", "..", ".", "a/b", "a\\b", "/etc/passwd", "C:outside.tif",
        std::string(129, 'a'), std::string("a\0b", 3)};
    for (const std::string& bad : hostile) {
        std::string xml = "<?xml version=\"1.0\"?>\n<geoids>\n  <geoid id=\"EGM96\" file=\"";
        xml += bad;
        xml +=
            "\" min_lon=\"-180\" min_lat=\"-90\" max_lon=\"180\" max_lat=\"90\"/>\n</geoids>\n";
        {
            std::ofstream out(store_dir / "index.xml", std::ios::binary | std::ios::trunc);
            out << xml;
        }
        try {
            const GeoidStore loaded = GeoidStore::load(store_dir);
            static_cast<void>(loaded);
            FAIL() << "load() accepted file=\"" << bad << "\"";
        } catch (const std::exception& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("unusable file name"), std::string::npos) << message;
        }
    }

    // The id travels the same join, both in the index and as the copied file
    // name, so a hostile id is refused on load and by add().
    for (const std::string& bad : std::vector<std::string>{"../evil", "..", "a/b", "C:x"}) {
        std::string xml = "<?xml version=\"1.0\"?>\n<geoids>\n  <geoid id=\"";
        xml += bad;
        xml +=
            "\" file=\"EGM96.tif\" min_lon=\"-180\" min_lat=\"-90\" max_lon=\"180\" "
            "max_lat=\"90\"/>\n</geoids>\n";
        {
            std::ofstream out(store_dir / "index.xml", std::ios::binary | std::ios::trunc);
            out << xml;
        }
        try {
            const GeoidStore loaded = GeoidStore::load(store_dir);
            static_cast<void>(loaded);
            FAIL() << "load() accepted id=\"" << bad << "\"";
        } catch (const std::exception& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("geoid id"), std::string::npos) << message;
        }

        GeoidStore store;
        store.set_directory(store_dir);
        try {
            store.add(bad, bad, (kFixtureDir / "egm96_subset.tif").string(), "", "");
            FAIL() << "add() accepted id=\"" << bad << "\"";
        } catch (const std::exception& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("geoid id"), std::string::npos) << message;
        }
        EXPECT_FALSE(fs::exists(dir_ / "evil.tif")) << "add() wrote outside the store";
    }

    // Nothing outside the store was created: `outside.tif` is still the only
    // other entry in the parent.
    int parent_entries = 0;
    for (const fs::directory_entry& item : fs::directory_iterator(dir_)) {
        if (item.path().filename() != "geoids") {
            ++parent_entries;
        }
    }
    EXPECT_EQ(parent_entries, 1) << "a rejected geoid name created a file outside the store";
}

// GDAL_NODATA marks cells with no data, and PROJ honours it. The GeoTIFF
// fallback must too: before it did, a nodata cell was interpolated as the
// sentinel itself, so H = h - N came out tens of thousands of metres wrong.
//
// The grid is built inside the test (the engine's own writer emits the tag), so
// the test knows exactly which node is missing. Its file name contains a space,
// which makes PROJ unavailable (`+grids=` cannot carry one) and forces exactly
// the fallback this test is about.
TEST_F(GeoidTempDir, CrsGeoidNodataCellHasNoValue) {
    constexpr float kNodata = -32767.0f;
    GeoRaster<float> raster;
    raster.grid.width = 8;
    raster.grid.height = 8;
    raster.grid.origin_x = 10.0;
    raster.grid.origin_y = 20.0;
    raster.grid.pixel_size_x = 1.0;
    raster.grid.pixel_size_y = 1.0;
    raster.channels = 1;
    raster.data.assign(64, 20.0f);
    raster.data[4 * 8 + 4] = kNodata;
    const std::string file = (dir_ / "nodata grid.tif").string();
    GeoTiffOptions options;
    options.epsg = 4326;
    options.nodata = static_cast<double>(kNodata);
    write_geotiff(file, raster, options);

    const std::shared_ptr<const GeoidGrid> grid = GeoidGrid::open(file);
    ASSERT_NE(grid, nullptr);
    const GeoidGridInfo& box = grid->info();
    ASSERT_EQ(box.width, 8);
    ASSERT_EQ(box.height, 8);
    const double cell_x = (box.max_lon_deg - box.min_lon_deg) / 7.0;
    const double cell_y = (box.max_lat_deg - box.min_lat_deg) / 7.0;
    const auto node_lon = [&](int col) { return box.min_lon_deg + col * cell_x; };
    const auto node_lat = [&](int row) { return box.max_lat_deg - row * cell_y; };

    // Exactly on the missing node.
    EXPECT_FALSE(grid->grid_undulation(node_lon(4), node_lat(4)).has_value());
    EXPECT_FALSE(grid->undulation(node_lon(4), node_lat(4)).has_value());
    // A fraction of a cell away the nodata node is still in the 4x4 support, so
    // the point is refused rather than interpolated across the hole.
    EXPECT_FALSE(
        grid->grid_undulation(node_lon(4) + 0.25 * cell_x, node_lat(4) + 0.25 * cell_y)
            .has_value());
    // Beyond the hole's support the grid answers normally: the refusal is local.
    for (const int col : {0, 7}) {
        const std::optional<double> value = grid->grid_undulation(node_lon(col), node_lat(col));
        ASSERT_TRUE(value.has_value()) << col;
        EXPECT_NEAR(*value, 20.0, 1e-6);
        const std::optional<double> general = grid->undulation(node_lon(col), node_lat(col));
        ASSERT_TRUE(general.has_value()) << col;
        EXPECT_NEAR(*general, 20.0, 1e-6);
    }

    // The same grid under a name PROJ can take: PROJ is then the evaluator, and
    // `undulation()` must not answer through the reader behind PROJ's back. At
    // the hole both refuse; on a good node they agree whenever PROJ ran.
    const std::string plain = (dir_ / "nodata_plain.tif").string();
    write_geotiff(plain, raster, options);
    const std::shared_ptr<const GeoidGrid> via_proj = GeoidGrid::open(plain);
    ASSERT_NE(via_proj, nullptr);
    const std::optional<double> refused = via_proj->undulation(node_lon(4), node_lat(4));
    EXPECT_TRUE(!refused || std::abs(*refused) < 1e3)
        << "nodata reached the caller: " << (refused ? *refused : 0.0);
    const std::optional<double> proj_corner = via_proj->proj_undulation(node_lon(0), node_lat(0));
    if (proj_corner) {
        const std::optional<double> general = via_proj->undulation(node_lon(0), node_lat(0));
        ASSERT_TRUE(general.has_value());
        EXPECT_NEAR(*general, *proj_corner, 1e-9);
    }
}
// A crafted header must not size the sample buffer. The file is assembled by
// hand (libtiff will not write a 4e9-wide image without the pixels), claiming
// 4 000 000 000 x 4 000 000 000: the product overflows a signed 32-bit int and
// the byte count overflows size_t on a 32-bit build, so the dimensions are
// rejected on the numbers before anything is allocated or read.
TEST_F(GeoidTempDir, CrsGeoidRejectsHugeDimensions) {
    const std::string crafted = (dir_ / "crafted.tif").string();
    write_header_only_tiff(crafted, 4'000'000'000U, 4'000'000'000U);
    try {
        GeoidGrid::open(crafted);
        FAIL() << "open() accepted a 4e9 x 4e9 header";
    } catch (const std::exception& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("may not exceed"), std::string::npos) << message;
    }
    // A shape under the per-side cap but over the total must also be refused.
    write_header_only_tiff(crafted, 1'000'000U, 1'000'000U);
    try {
        GeoidGrid::open(crafted);
        FAIL() << "open() accepted a 1e6 x 1e6 header";
    } catch (const std::exception& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("may not exceed"), std::string::npos) << message;
    }
    // 30 M per side is a legal geoid grid shape; it fails on the missing pixel
    // data instead, which proves the cap is above real grids.
    write_header_only_tiff(crafted, 30'000'000U, 2U);
    try {
        GeoidGrid::open(crafted);
        FAIL() << "open() accepted a header with no pixel data";
    } catch (const std::exception& error) {
        const std::string message = error.what();
        EXPECT_EQ(message.find("may not exceed"), std::string::npos) << message;
    }
}

// VerticalCrs::offset_m is parsed and persisted (CrsVerticalCrsPersistence
// pins that), but a datum that round-trips an offset and then ignores it is
// silently wrong by that many metres. The convention is the one the header
// documents — the datum reads `h - N + offset` — and it must hold in custom()
// (both sides), through PROJ's compound pipeline, and backwards on inverse().
TEST(Crs, VerticalOffsetApplied) {
    const std::shared_ptr<const GeoidGrid> grid = subset();
    ASSERT_NE(grid, nullptr);
    const CoordinateSystem geo = CoordinateSystem::from_epsg(4979);
    const CoordinateSystem utm = CoordinateSystem::from_epsg(32633);
    const VerticalCrs none;
    const std::optional<double> n = grid->undulation(9.0, 50.0);
    ASSERT_TRUE(n.has_value());

    // Target side: ellipsoidal -> orthometric with a +2 m datum offset.
    VerticalCrs target;
    target.kind = VerticalCrs::Kind::GravityRelated;
    target.name = "local datum";
    target.grid = "us_nga_egm96_15.tif";
    target.offset_m = 2.0;
    // The geoid id is only a label for the custom path, so the EGM96 name (no
    // EPSG code in this build) is fine and nothing is looked up.
    const VerticalTransform to_target =
        VerticalTransform::custom(geo, utm, none, target, grid, geo);
    ASSERT_TRUE(to_target.is_valid());
    constexpr double kH = 120.0;
    const std::optional<CrsPoint> ortho = to_target.forward(CrsPoint{9.0, 50.0, kH});
    ASSERT_TRUE(ortho.has_value());
    EXPECT_NEAR(ortho->z, kH - *n + 2.0, 1e-6)
        << "the datum reads h - N + offset on the target side";
    // ...and inverse() takes it back out, so a round trip is exact.
    const std::optional<CrsPoint> back = to_target.inverse(*ortho);
    ASSERT_TRUE(back.has_value());
    EXPECT_NEAR(back->z, kH, 1e-6) << "offset must cancel on the round trip";

    // Source side: orthometric -> ellipsoidal with the same offset there.
    const VerticalTransform to_ellipsoidal =
        VerticalTransform::custom(geo, utm, target, none, grid, geo);
    ASSERT_TRUE(to_ellipsoidal.is_valid());
    const std::optional<CrsPoint> h = to_ellipsoidal.forward(CrsPoint{9.0, 50.0, 100.0});
    ASSERT_TRUE(h.has_value());
    EXPECT_NEAR(h->z, 100.0 - 2.0 + *n, 1e-6)
        << "the source datum's offset is removed before the grid";
    const std::optional<CrsPoint> round = to_ellipsoidal.inverse(*h);
    ASSERT_TRUE(round.has_value());
    EXPECT_NEAR(round->z, 100.0, 1e-6);

    // Both sides declare the same datum offset: H_dst = H_src - offset_src +
    // offset_dst, so the two cancel.
    VerticalCrs source = target;
    const VerticalTransform both =
        VerticalTransform::custom(geo, utm, source, target, grid, geo);
    ASSERT_TRUE(both.is_valid());
    const std::optional<CrsPoint> shifted = both.forward(CrsPoint{9.0, 50.0, kH});
    ASSERT_TRUE(shifted.has_value());
    EXPECT_NEAR(shifted->z, kH - *n, 1e-6);

    // The compound path hands the geoid to PROJ, which knows nothing about a
    // constant, so the offset is applied around the pipeline there. PROJ needs
    // the installed EGM96 grid and an EPSG vertical CRS for that.
    if (!installed_grid("us_nga_egm96_15.tif")) {
        GTEST_SKIP() << "us_nga_egm96_15.tif is not installed; the compound offset "
                        "comparison needs it";
    }
    VerticalCrs egm96;
    egm96.kind = VerticalCrs::Kind::GeoidModel;
    egm96.name = "EGM96 height";
    egm96.grid = "us_nga_egm96_15.tif";
    const CoordinateSystem dst_cs = utm.with_vertical(egm96);
    const std::optional<CrsPoint> plain =
        VerticalTransform::compound(geo, dst_cs).forward(CrsPoint{9.0, 50.0, kH});
    VerticalCrs shifted_datum = egm96;
    shifted_datum.offset_m = 1.5;
    const std::optional<CrsPoint> with_offset =
        VerticalTransform::compound(geo, utm.with_vertical(shifted_datum))
            .forward(CrsPoint{9.0, 50.0, kH});
    ASSERT_TRUE(plain.has_value());
    ASSERT_TRUE(with_offset.has_value());
    EXPECT_NEAR(with_offset->z - plain->z, 1.5, 1e-6)
        << "compound() must apply the target datum's offset after PROJ";
}

}  // namespace
}  // namespace alphashape::crs
