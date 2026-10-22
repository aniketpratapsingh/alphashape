#pragma once

/// @file proj_context.hpp
/// @brief PROJ dependency probe, data-directory discovery and thread contexts.
///
/// This header deliberately does not include <proj.h>: everything the rest of
/// the engine needs is expressed in terms of strings and paths, and the only
/// PROJ type that crosses the boundary is the opaque context handle declared
/// below. Translation units that actually call PROJ include <proj.h> themselves
/// (implementation, and later milestones' transform engine).
///
/// Data-directory resolution order (M12 §4.1):
///   1. ALPHASHAPE_PROJ_DATA, if set and valid.
///   2. <exe_dir>/share/proj, then <exe_dir>/../share/proj (packaged builds:
///      a bundle keeps data beside the binary, an install keeps share/ beside
///      bin/).
///   3. <project_source>/../tools/proj-data (the SSD tool layout).
///   4. PROJ's own search (proj_context_get_database_path()).
/// A candidate is valid when it contains proj.db.

#include <filesystem>
#include <string>
#include <vector>

// Opaque in C++, but the same tag PROJ's PJ_CONTEXT typedefs.
struct pj_ctx;

namespace alphashape::crs {

using ProjContext = pj_ctx*;

/// Version of the PROJ library linked into this binary, e.g. "9.9.0".
std::string proj_version();

/// proj_info().major / .minor of the linked library (0 if unknown).
int proj_version_major();
int proj_version_minor();

/// Facts read from the proj.db that PROJ will actually use.
struct ProjDataInfo {
    /// Value of the metadata key "PROJ.VERSION" (data package version).
    std::string data_version;
    /// "DATABASE.LAYOUT.VERSION.MAJOR"/"MINOR"; -1 when unavailable.
    int layout_major = -1;
    int layout_minor = -1;
    /// proj_context_get_database_path(), empty when no database was found.
    std::filesystem::path database_path;
    /// Directory PROJ resolves grids from; empty when unknown.
    std::filesystem::path search_directory;
};

ProjDataInfo proj_data_info();

/// proj.db metadata "PROJ.VERSION", e.g. "9.9.0". Empty when no database.
std::string data_version();

/// proj.db "DATABASE.LAYOUT.VERSION", e.g. "1.20". Empty when unavailable.
std::string data_layout_version();

/// Inputs to data-directory resolution. Empty fields mean "use the real one":
/// exe_dir defaults to the running executable's directory, project_source to
/// the configured source tree and env_override to ALPHASHAPE_PROJ_DATA.
struct DataSearchOptions {
    std::filesystem::path exe_dir;
    std::filesystem::path project_source;
    std::filesystem::path env_override;
    /// Skip PROJ's own search (used by tests to assert the earlier steps).
    bool allow_proj_default = true;
};

enum class SearchStep {
    EnvOverride = 0,
    ExeRelative = 1,
    ProjectSource = 2,
    ProjDefault = 3,
};

const char* search_step_name(SearchStep step);

struct ResolvedDataDir {
    std::filesystem::path path;
    SearchStep step = SearchStep::ProjDefault;
    bool found = false;
};

/// Apply the §4.1 search order. A candidate qualifies only when it contains
/// proj.db; the PROJ-default step asks PROJ for its configured directory.
ResolvedDataDir resolve_data_directory(const DataSearchOptions& options);

/// The resolved data directory for this process, initialising PROJ once.
std::filesystem::path data_directory();

/// Locate a data file (grid, proj.db, proj.ini) by name: the resolved data
/// directory first, then PROJ's default search directory. Empty when absent.
std::filesystem::path find_data_file(const std::string& name);

/// True when the context PROJ will use for transforms allows network access.
/// Always false after proj_context_initialize() (offline determinism, §4.1).
bool network_enabled();

/// Initialise PROJ once per process: resolve the data directory, publish it as
/// PROJ_DATA (and normalise an accepted ALPHASHAPE_PROJ_DATA), disable network
/// access and return the resolved directory. Empty when none was found.
/// Idempotent and thread-safe.
std::filesystem::path proj_context_initialize();

/// Thread-local PJ_CONTEXT, created on first use from the initialised state.
/// Callers must not cache it beyond their thread; PROJ documents one context
/// per thread for concurrent transforms.
ProjContext thread_context();

/// Multi-line summary (version, data version, database path, network state)
/// for startup logging.
std::string describe_environment();

/// describe_environment() written through the engine log at Info level.
void log_environment();

}  // namespace alphashape::crs
