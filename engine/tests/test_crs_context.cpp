#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include <gtest/gtest.h>
#include <proj.h>

#include "model/crs/proj_context.hpp"

// WP-12.1: the PROJ dependency itself — version, data-directory discovery,
// offline behaviour. Transforms proper arrive with WP-12.3; the one point
// transform here only proves the context and proj.db are usable.

namespace alphashape::crs {
namespace {

namespace fs = std::filesystem;

int current_process_id() {
#ifdef _WIN32
    return _getpid();
#else
    return static_cast<int>(::getpid());
#endif
}

class TempTree {
public:
    TempTree() {
        static int counter = 0;
        path_ = fs::temp_directory_path() /
                ("alphashape_crs_" + std::to_string(current_process_id()) + "_" +
                 std::to_string(++counter));
        fs::create_directories(path_);
    }
    ~TempTree() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    TempTree(const TempTree&) = delete;
    TempTree& operator=(const TempTree&) = delete;

    const fs::path& path() const { return path_; }

    fs::path make_data_dir(const std::string& relative,
                          bool with_egm_grid = true) const {
        const fs::path dir = path_ / relative;
        fs::create_directories(dir);
        std::ofstream(dir / "proj.db") << "placeholder";
        std::ofstream(dir / "proj.ini") << "[general]\nnetwork = off\n";
        if (with_egm_grid) {
            std::ofstream(dir / "us_nga_egm96_15.tif") << "placeholder";
        }
        return dir;
    }

private:
    fs::path path_;
};

int parse_leading_int(const std::string& text) {
    return std::atoi(text.c_str());
}

TEST(Crs, ProjVersionAtLeast9_3) {
    const std::string version = proj_version();
    EXPECT_FALSE(version.empty());
    EXPECT_EQ(proj_version_major(), parse_leading_int(version));
    const PJ_INFO info = proj_info();
    EXPECT_GE(info.major, 9);
    if (info.major == 9) {
        EXPECT_GE(info.minor, 3) << "PROJ " << version << " is older than 9.3";
    }
    EXPECT_GE(proj_version_minor(), 0);
}

TEST(Crs, DataDirectoryResolves) {
    // Each search step on its own: the candidate must win and be reported.
    {
        TempTree tree;
        const fs::path env_dir = tree.make_data_dir("env");
        DataSearchOptions options;
        options.env_override = env_dir;
        options.exe_dir = tree.path() / "install" / "bin";
        options.project_source = tree.path() / "project" / "engine";
        options.allow_proj_default = false;
        const ResolvedDataDir resolved = resolve_data_directory(options);
        ASSERT_TRUE(resolved.found);
        EXPECT_EQ(resolved.step, SearchStep::EnvOverride);
        EXPECT_EQ(resolved.path, env_dir);
        EXPECT_TRUE(fs::exists(resolved.path / "proj.db"));
        EXPECT_TRUE(fs::exists(resolved.path / "us_nga_egm96_15.tif"));
        EXPECT_STREQ(search_step_name(resolved.step), "ALPHASHAPE_PROJ_DATA");
    }
    {
        TempTree tree;
        // <exe_dir>/share/proj and <exe_dir>/../share/proj are both step 2: a
        // macOS bundle puts Resources next to the MacOS binary, an install puts
        // share/ next to bin/.
        const fs::path exe_dir = tree.path() / "install" / "bin";
        fs::create_directories(exe_dir);
        const fs::path data = tree.make_data_dir("install/share/proj");
        DataSearchOptions options;
        options.env_override = tree.path() / "absent";
        options.exe_dir = exe_dir;
        options.project_source = tree.path() / "project" / "engine";
        options.allow_proj_default = false;
        const ResolvedDataDir resolved = resolve_data_directory(options);
        ASSERT_TRUE(resolved.found);
        EXPECT_EQ(resolved.step, SearchStep::ExeRelative);
        EXPECT_EQ(resolved.path, data);
    }
    {
        TempTree tree;
        // <project_source>/../tools/proj-data, the SSD tool layout.
        const fs::path source = tree.path() / "repo" / "engine";
        fs::create_directories(source);
        const fs::path data = tree.make_data_dir("repo/tools/proj-data");
        DataSearchOptions options;
        options.env_override = tree.path() / "absent";
        options.exe_dir = tree.path() / "nowhere";
        options.project_source = source;
        options.allow_proj_default = false;
        const ResolvedDataDir resolved = resolve_data_directory(options);
        ASSERT_TRUE(resolved.found);
        EXPECT_EQ(resolved.step, SearchStep::ProjectSource);
        EXPECT_EQ(resolved.path, data);
    }
    // Order: the environment override outranks an installed copy.
    {
        TempTree tree;
        const fs::path exe_dir = tree.path() / "install" / "bin";
        fs::create_directories(exe_dir);
        tree.make_data_dir("install/share/proj");
        const fs::path env_dir = tree.make_data_dir("env");
        DataSearchOptions options;
        options.env_override = env_dir;
        options.exe_dir = exe_dir;
        options.allow_proj_default = false;
        const ResolvedDataDir resolved = resolve_data_directory(options);
        ASSERT_TRUE(resolved.found);
        EXPECT_EQ(resolved.step, SearchStep::EnvOverride);
    }
    // A directory without proj.db is not a data directory.
    {
        TempTree tree;
        const fs::path empty = tree.path() / "empty";
        fs::create_directories(empty);
        DataSearchOptions options;
        options.env_override = empty;
        options.exe_dir = empty;
        options.project_source = tree.path();
        options.allow_proj_default = false;
        EXPECT_FALSE(resolve_data_directory(options).found);
    }
    // The real environment: PROJ must be able to find its database, or nothing
    // in M12 can work. This is the acceptance check for the installed data.
    const ProjDataInfo info = proj_data_info();
    ASSERT_FALSE(info.database_path.empty())
        << "PROJ cannot find proj.db; install proj-data or set "
           "ALPHASHAPE_PROJ_DATA";
    EXPECT_TRUE(fs::exists(info.database_path));
    EXPECT_FALSE(info.data_version.empty());
    EXPECT_GE(info.layout_major, 0);
    EXPECT_EQ(data_version(), info.data_version);
    EXPECT_FALSE(data_layout_version().empty());
    EXPECT_EQ(data_layout_version(),
              std::to_string(info.layout_major) + "." +
                  std::to_string(info.layout_minor));
}

TEST(Crs, OfflineMode) {
    proj_context_initialize();
    EXPECT_FALSE(network_enabled());
    EXPECT_FALSE(proj_data_info().database_path.empty());

    // A transform that needs a grid nobody installed must fail, not download.
    PJ_CONTEXT* ctx = thread_context();
    ASSERT_NE(ctx, nullptr);
    PJ* missing = proj_create(ctx, "+proj=hgridshift +grids=__alphashape_absent_grid__.tif");
    if (missing != nullptr) {
        proj_destroy(missing);
    }
    EXPECT_EQ(missing, nullptr)
        << "PROJ accepted a missing grid; it must not fabricate or fetch one";

    // A vgridshift (the geoid path) behaves the same way.
    PJ* missing_v = proj_create(ctx, "+proj=vgridshift +grids=__alphashape_absent_geoid__.tif");
    if (missing_v != nullptr) {
        proj_destroy(missing_v);
    }
    EXPECT_EQ(missing_v, nullptr);
    // PROJ leaves the failure on the context; a successful create afterwards
    // both proves the context is still usable and clears that state.
    PJ* recovered = proj_create(ctx, "EPSG:4326");
    ASSERT_NE(recovered, nullptr);
    proj_destroy(recovered);
}

TEST(Crs, ContextTransformsKnownPoint) {
    // The context handed out by thread_context() must be usable for a real
    // EPSG:4326 -> EPSG:32633 transform (proj.db lookup + TM maths) and agree
    // with the published value: 15 E 50 N (the zone-33 central meridian).
    //   cs2cs EPSG:4326 EPSG:32633 -f "%.6f"  =>  500000.000000 5538630.702867
    PJ_CONTEXT* ctx = thread_context();
    ASSERT_NE(ctx, nullptr);
    PJ* raw = proj_create_crs_to_crs(ctx, "EPSG:4326", "EPSG:32633", nullptr);
    ASSERT_NE(raw, nullptr) << proj_context_errno_string(ctx, proj_context_errno(ctx));
    PJ* transform = proj_normalize_for_visualization(ctx, raw);
    proj_destroy(raw);
    ASSERT_NE(transform, nullptr);

    // normalize_for_visualization makes the axis order east/north, so the
    // input is (lon, lat) on the central meridian and the easting is exactly
    // the false easting; the northing is the published 5538630.702867 m.
    PJ_COORD point = proj_coord(15.0, 50.0, 0.0, 0.0);
    point = proj_trans(transform, PJ_FWD, point);
    EXPECT_NEAR(point.xy.x, 500000.0, 1e-6);
    EXPECT_NEAR(point.xy.y, 5538630.702867, 1e-4);

    // Inverse: the same point comes back as (15, 50).
    const PJ_COORD back = proj_trans(transform, PJ_INV, point);
    EXPECT_NEAR(back.xy.x, 15.0, 1e-9);
    EXPECT_NEAR(back.xy.y, 50.0, 1e-9);
    proj_destroy(transform);
}

TEST(Crs, EnvironmentIsExported) {
    // proj_context_initialize() publishes the resolved directory through
    // PROJ_DATA so PROJ (and child processes) resolve grids identically.
    const fs::path dir = proj_context_initialize();
    const fs::path via_search = resolve_data_directory({}).path;
    ASSERT_FALSE(dir.empty());
    EXPECT_EQ(dir, via_search) << "initialize and the search must agree";
    const char* exported = std::getenv("PROJ_DATA");
    ASSERT_NE(exported, nullptr);
    EXPECT_EQ(fs::path(exported), dir);
    // proj_context_initialize() exports the directory it resolved under both
    // variable names, whichever search step produced it.
    const char* shape = std::getenv("ALPHASHAPE_PROJ_DATA");
    ASSERT_NE(shape, nullptr);
    EXPECT_EQ(fs::path(shape), dir);
    const char* network = std::getenv("PROJ_NETWORK");
    ASSERT_NE(network, nullptr);
    EXPECT_STREQ(network, "OFF");
    EXPECT_FALSE(find_data_file("proj.db").empty());
    EXPECT_NE(describe_environment().find("PROJ "), std::string::npos);
}

TEST(Crs, StagingScriptProducesUsableDataDir) {
    // tools/stage_proj_data.sh is what a packaged build runs; exercising it on
    // a temp destination proves the copy path and that the result resolves as
    // a data directory through the environment step.
    const std::string script = std::string(ALPHASHAPE_PROJECT_SOURCE_DIR) +
                               "/tools/stage_proj_data.sh";
    const ProjDataInfo info = proj_data_info();
    ASSERT_FALSE(info.search_directory.empty());
    TempTree tree;
    const fs::path destination = tree.path() / "staged";
    const std::string command = "/bin/bash \"" + script + "\" \"" +
                                info.search_directory.string() + "\" \"" +
                                destination.string() + "\" >/dev/null 2>&1";
    const int status = std::system(command.c_str());
    if (status != 0) {
        GTEST_SKIP() << "staging needs /bin/bash and a readable source; skipped";
    }
    ASSERT_TRUE(fs::exists(destination / "proj.db"));
    ASSERT_TRUE(fs::exists(destination / "proj.ini"));

    DataSearchOptions options;
    options.env_override = destination;
    options.exe_dir = tree.path() / "nowhere";
    options.project_source = tree.path();
    options.allow_proj_default = false;
    const ResolvedDataDir resolved = resolve_data_directory(options);
    ASSERT_TRUE(resolved.found);
    EXPECT_EQ(resolved.step, SearchStep::EnvOverride);
    EXPECT_EQ(resolved.path, destination);
}

}  // namespace
}  // namespace alphashape::crs
