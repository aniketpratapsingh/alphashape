#include "model/crs/proj_context.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include <proj.h>

#include "base/utils/log.hpp"

#ifdef __APPLE__
#include <dlfcn.h>
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace alphashape::crs {
namespace {

namespace fs = std::filesystem;

std::filesystem::path env_path(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? fs::path(value) : fs::path{};
}

bool has_database(const fs::path& dir) {
    if (dir.empty()) {
        return false;
    }
    std::error_code ec;
    return fs::is_regular_file(dir / "proj.db", ec);
}

std::filesystem::path executable_directory() {
#ifdef __APPLE__
    uint32_t size = 0;
    if (_NSGetExecutablePath(nullptr, &size) != -1 || size == 0) {
        return {};
    }
    std::string buffer(size, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        return {};
    }
    std::error_code ec;
    return fs::weakly_canonical(fs::path(buffer.c_str()), ec).parent_path();
#elif defined(__linux__)
    std::error_code ec;
    const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    return ec ? fs::path{} : exe.parent_path();
#else
    return {};
#endif
}

#ifdef __APPLE__
/// Directory of the Mach-O image this code is linked into (the executable, or the Python
/// extension module inside somebody else's interpreter).
std::filesystem::path image_directory() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&image_directory), &info) == 0 ||
        info.dli_fname == nullptr) {
        return {};
    }
    std::error_code ec;
    return fs::weakly_canonical(fs::path(info.dli_fname), ec).parent_path();
}
#endif

std::filesystem::path source_root() {
#ifdef ALPHASHAPE_PROJECT_SOURCE_DIR
    return fs::path(ALPHASHAPE_PROJECT_SOURCE_DIR);
#else
    return {};
#endif
}

/// Context used only to interrogate PROJ's configured defaults. A fresh
/// context (rather than the thread-local one) keeps this callable from inside
/// proj_context_initialize() without re-entering the once-flag.
struct TempContext {
    PJ_CONTEXT* ctx = proj_context_create();
    // Declaring the copy operations deletes the implicit default constructor.
    TempContext() = default;
    ~TempContext() {
        if (ctx != nullptr) {
            proj_context_destroy(ctx);
        }
    }
    TempContext(const TempContext&) = delete;
    TempContext& operator=(const TempContext&) = delete;
};

bool network_requested_by_env() {
    const char* value = std::getenv("PROJ_NETWORK");
    if (value == nullptr) {
        return false;
    }
    return std::strcmp(value, "ON") == 0 || std::strcmp(value, "on") == 0 ||
           std::strcmp(value, "1") == 0;
}

/// Write an environment variable for this process. `overwrite` is separate
/// from the caller's intent because tests set values before first use.
bool set_env(const char* name, const std::string& value,
             [[maybe_unused]] bool overwrite = true) {
#if defined(_WIN32)
    return _putenv_s(name, value.c_str()) == 0;
#else
    return ::setenv(name, value.c_str(), overwrite ? 1 : 0) == 0;
#endif
}

struct Environment {
    fs::path data_dir;
    bool network = false;
};

Environment& environment_state() {
    static Environment state;
    return state;
}

std::once_flag g_init_once;

int metadata_int(PJ_CONTEXT* ctx, const char* key, int fallback) {
    const char* value = proj_context_get_database_metadata(ctx, key);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

}  // namespace

const char* search_step_name(SearchStep step) {
    switch (step) {
        case SearchStep::EnvOverride:
            return "ALPHASHAPE_PROJ_DATA";
        case SearchStep::ExeRelative:
            return "exe/share/proj";
        case SearchStep::ProjectSource:
            return "tools/proj-data";
        case SearchStep::ProjDefault:
            return "proj-default";
    }
    return "unknown";
}

ResolvedDataDir resolve_data_directory(const DataSearchOptions& options) {
    DataSearchOptions opts = options;
    if (opts.exe_dir.empty()) {
        opts.exe_dir = executable_directory();
    }
    if (opts.project_source.empty()) {
        opts.project_source = source_root();
    }
    if (opts.env_override.empty()) {
        opts.env_override = env_path("ALPHASHAPE_PROJ_DATA");
    }

    ResolvedDataDir resolved;
    const auto accept = [&resolved](const fs::path& dir, SearchStep step) {
        if (!has_database(dir)) {
            return false;
        }
        resolved.path = dir;
        resolved.step = step;
        resolved.found = true;
        return true;
    };

    if (accept(opts.env_override, SearchStep::EnvOverride)) {
        return resolved;
    }
    // Packaged layout (§4.1 step 2): data beside the binary. A conventional
    // install puts the binary in bin/ and the data in the sibling share/, so
    // the parent is tried as part of the same step.
    if (accept(opts.exe_dir / "share" / "proj", SearchStep::ExeRelative) ||
        accept(opts.exe_dir.parent_path() / "share" / "proj", SearchStep::ExeRelative)) {
        return resolved;
    }
#ifdef __APPLE__
    // App bundle: Contents/MacOS/<exe> finds Contents/Resources/share/proj; the Python module in
    // Contents/Resources/python finds Contents/Resources/share/proj through its parent.
    {
        const fs::path image_dir = image_directory();
        if (!image_dir.empty() &&
            (accept(image_dir.parent_path() / "Resources" / "share" / "proj", SearchStep::ExeRelative) ||
             accept(image_dir / "share" / "proj", SearchStep::ExeRelative) ||
             accept(image_dir.parent_path() / "share" / "proj", SearchStep::ExeRelative))) {
            return resolved;
        }
    }
#endif
    if (!opts.project_source.empty()) {
        const fs::path tool_dir =
            opts.project_source.parent_path() / "tools" / "proj-data";
        if (accept(tool_dir, SearchStep::ProjectSource)) {
            return resolved;
        }
    }
    if (opts.allow_proj_default) {
        TempContext temp;
        const char* database = proj_context_get_database_path(temp.ctx);
        if (database != nullptr && *database != '\0') {
            const fs::path dir = fs::path(database).parent_path();
            if (accept(dir, SearchStep::ProjDefault)) {
                return resolved;
            }
        }
    }
    return resolved;
}

std::string proj_version() {
    const PJ_INFO info = proj_info();
    if (info.version != nullptr && *info.version != '\0') {
        return info.version;
    }
    if (info.release != nullptr && *info.release != '\0') {
        return info.release;
    }
    return "unknown";
}

int proj_version_major() { return proj_info().major; }

int proj_version_minor() { return proj_info().minor; }

ProjDataInfo proj_data_info() {
    // The answer is only meaningful once PROJ_DATA has been published by
    // proj_context_initialize(); without this the probe would report PROJ's
    // built-in default even when this library resolves a different directory.
    proj_context_initialize();
    ProjDataInfo info;
    TempContext temp;
    const char* database = proj_context_get_database_path(temp.ctx);
    if (database != nullptr && *database != '\0') {
        info.database_path = database;
        info.search_directory = fs::path(database).parent_path();
    }
    const char* version = proj_context_get_database_metadata(temp.ctx, "PROJ.VERSION");
    if (version != nullptr) {
        info.data_version = version;
    }
    info.layout_major = metadata_int(temp.ctx, "DATABASE.LAYOUT.VERSION.MAJOR", -1);
    info.layout_minor = metadata_int(temp.ctx, "DATABASE.LAYOUT.VERSION.MINOR", -1);
    if (info.search_directory.empty()) {
        info.search_directory = env_path("PROJ_DATA");
    }
    return info;
}

std::string data_version() { return proj_data_info().data_version; }

std::string data_layout_version() {
    const ProjDataInfo info = proj_data_info();
    if (info.layout_major < 0) {
        return {};
    }
    return std::to_string(info.layout_major) + "." + std::to_string(info.layout_minor);
}

std::filesystem::path find_data_file(const std::string& name) {
    if (name.empty()) {
        return {};
    }
    const std::vector<fs::path> candidates = {
        environment_state().data_dir / name,
        proj_data_info().search_directory / name,
    };
    std::error_code ec;
    for (const fs::path& candidate : candidates) {
        if (!candidate.empty() && fs::is_regular_file(candidate, ec)) {
            return candidate;
        }
    }
    return {};
}

namespace {

/// Every context this library hands out gets the same deterministic setup:
/// the resolved data directory first on the search path, grid caching off and
/// network access off.
void configure_context(PJ_CONTEXT* ctx) {
    if (ctx == nullptr) {
        return;
    }
    const fs::path& dir = environment_state().data_dir;
    if (!dir.empty()) {
        const std::string value = dir.string();
        const char* paths[1] = {value.c_str()};
        proj_context_set_search_paths(ctx, 1, paths);
    }
    // Offline determinism applies with or without a resolved directory: a
    // missing grid must be an error, never a download.
    proj_context_set_enable_network(ctx, 0);
    proj_grid_cache_set_enable(ctx, 0);
}

struct ThreadContext {
    PJ_CONTEXT* ctx = proj_context_create();

    ThreadContext() { configure_context(ctx); }

    ~ThreadContext() {
        if (ctx != nullptr) {
            proj_context_destroy(ctx);
        }
    }

    ThreadContext(const ThreadContext&) = delete;
    ThreadContext& operator=(const ThreadContext&) = delete;
};

ThreadContext& thread_state() {
    thread_local ThreadContext state;
    return state;
}

}  // namespace

std::filesystem::path proj_context_initialize() {
    std::call_once(g_init_once, [] {
        Environment& env = environment_state();

        const fs::path override_dir = env_path("ALPHASHAPE_PROJ_DATA");
        const ResolvedDataDir resolved = resolve_data_directory({});

        // Publish the resolved directory so PROJ itself (and any child
        // process) resolves grids identically.
        if (!resolved.found) {
            log(LogLevel::Warn)
                << "PROJ data directory not found; transformations needing grids "
                   "or proj.db will fail (set ALPHASHAPE_PROJ_DATA)";
        } else {
            env.data_dir = resolved.path;
            // Export the resolved directory under both names. PROJ reads
            // PROJ_DATA itself; ALPHASHAPE_PROJ_DATA is what our own search
            // consults at step 1 and what a child process inherits, so the
            // choice is stable across a process tree even when it was made by
            // the exe-relative or tools/proj-data step rather than by the
            // caller's environment.
            set_env("PROJ_DATA", resolved.path.string());
            set_env("ALPHASHAPE_PROJ_DATA", resolved.path.string());
            log(LogLevel::Info) << "PROJ data: " << resolved.path << " ("
                                << search_step_name(resolved.step) << ")";
            if (!override_dir.empty() && resolved.step != SearchStep::EnvOverride) {
                log(LogLevel::Warn)
                    << "ALPHASHAPE_PROJ_DATA=" << override_dir
                    << " has no proj.db; ignored";
            }
        }

        // Offline determinism: proj.ini ships network = off, but a PROJ_NETWORK
        // variable in the environment would override it, so force it off here.
        // Grids must be installed; a missing grid is an error, never a download.
        if (network_requested_by_env()) {
            log(LogLevel::Warn) << "PROJ_NETWORK requested but network access is "
                                   "disabled for offline determinism";
        }
        set_env("PROJ_NETWORK", "OFF");
        env.network = false;
    });
    return environment_state().data_dir;
}

bool network_enabled() {
    proj_context_initialize();
    PJ_CONTEXT* ctx = thread_context();
    return ctx != nullptr && proj_context_is_network_enabled(ctx) != 0;
}

std::filesystem::path data_directory() { return proj_context_initialize(); }

ProjContext thread_context() {
    proj_context_initialize();
    return thread_state().ctx;
}

std::string describe_environment() {
    proj_context_initialize();
    const ProjDataInfo info = proj_data_info();
    const Environment& env = environment_state();

    std::string text = "PROJ " + proj_version();
    if (!info.data_version.empty()) {
        text += " (data " + info.data_version;
        if (info.layout_major >= 0) {
            text += ", layout " + std::to_string(info.layout_major) + "." +
                    std::to_string(info.layout_minor);
        }
        text += ")";
    }
    text += "\n  data dir: ";
    text += env.data_dir.empty() ? std::string("(not resolved)")
                                 : env.data_dir.string();
    text += "\n  proj.db: ";
    text += info.database_path.empty() ? std::string("(missing)")
                                       : info.database_path.string();
    text += "\n  network: ";
    text += env.network ? "on" : "off (offline)";
    return text;
}

void log_environment() { log(LogLevel::Info) << describe_environment(); }

}  // namespace alphashape::crs
