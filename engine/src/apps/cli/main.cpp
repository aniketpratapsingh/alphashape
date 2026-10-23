// alphashape-cli: the command-line front end.
//
// This is one of two front ends onto the same task schema (the other is the
// Python module). It is deliberately thin: every parameter is bound by *name*
// against the task's declared schema via make_params()/parse_param_value(), and
// execution goes through run_task() exactly like the Python binder. That is the
// whole point of the architecture: the CLI has no private notion
// of what a task accepts, so it can never drift from the project file.
//
// Exit codes (chosen so scripts can branch without parsing text):
//   0   success
//   1   engine runtime error (any alphashape::Error / std::exception)
//   2   the requested task is not in the registry (UnsupportedTaskError)
//   64  bad usage (EX_USAGE from sysexits.h)

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "base/platform/console.hpp"
#include "base/utils/error.hpp"
#include "common/core/document.hpp"
#include "common/io/project_io.hpp"
#include "common/tasks/params.hpp"
#include "common/tasks/registry.hpp"
#include "gpu/gpu.hpp"
#include "model/crs/proj_context.hpp"

namespace {

constexpr const char* kUsage =
    "Usage:\n"
    "  alphashape-cli --list-tasks\n"
    "  alphashape-cli --gpu-info\n"
    "  alphashape-cli --proj-info\n"
    "  alphashape-cli --list-params <Task>\n"
    "  alphashape-cli --project <file.psx> --task <Task> [--set key=value ...]\n"
    "  alphashape-cli --new <file.psx> --add-images <img> [<img> ...]\n"
    "  alphashape-cli --help\n";

/// Thrown for malformed command lines; distinct from engine errors so main can
/// map it to exit 64 rather than 1.
struct UsageError {};

const char* type_name(alphashape::ParamType type) {
    switch (type) {
        case alphashape::ParamType::Bool: return "Bool";
        case alphashape::ParamType::Int: return "Int";
        case alphashape::ParamType::Float: return "Float";
        case alphashape::ParamType::String: return "String";
        case alphashape::ParamType::StringList: return "StringList";
    }
    return "?";
}

// Params::specs() returns a copy, so a caller must keep that vector alive for
// as long as it holds these pointers — hence passing it in by reference rather
// than referencing entry.schema.specs() (a temporary) inside the search.
const alphashape::ParamSpec* find_spec(const std::vector<alphashape::ParamSpec>& specs,
                                       const std::string& name) {
    for (const alphashape::ParamSpec& spec : specs) {
        if (spec.name == name) {
            return &spec;
        }
    }
    return nullptr;
}

/// Progress sink. On a TTY we overwrite one line with '\r'; piped or redirected
/// we emit a line per update, because '\r' progress is unreadable in a log.
alphashape::Progress make_progress(bool tty) {
    alphashape::Progress progress;
    progress.report = [tty](const std::string& stage, double fraction) {
        const int pct = static_cast<int>(fraction * 100.0 + 0.5);
        if (tty) {
            std::fprintf(stderr, "\r[%s] %3d%%", stage.c_str(), pct);
            std::fflush(stderr);
        } else {
            std::fprintf(stderr, "[%s] %d%%\n", stage.c_str(), pct);
        }
    };
    return progress;
}

/// The parsed command line. One flag per field; a flag is "has_*" when it takes
/// a value.
struct Options {
    bool help = false;
    bool list_tasks = false;
    bool gpu_info = false;
    bool proj_info = false;
    bool has_list_params = false;
    std::string list_params;
    bool has_project = false;
    std::string project;
    bool has_task = false;
    std::string task;
    bool has_new = false;
    std::string new_project;
    std::vector<std::string> set_values;
    std::vector<std::string> images;
};

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto take_value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw UsageError{};
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            opts.help = true;
        } else if (arg == "--list-tasks") {
            opts.list_tasks = true;
        } else if (arg == "--gpu-info") {
            opts.gpu_info = true;
        } else if (arg == "--proj-info") {
            opts.proj_info = true;
        } else if (arg == "--list-params") {
            opts.has_list_params = true;
            opts.list_params = take_value();
        } else if (arg == "--project") {
            opts.has_project = true;
            opts.project = take_value();
        } else if (arg == "--task") {
            opts.has_task = true;
            opts.task = take_value();
        } else if (arg == "--set") {
            opts.set_values.push_back(take_value());
        } else if (arg == "--new") {
            opts.has_new = true;
            opts.new_project = take_value();
        } else if (arg == "--add-images") {
            // Greedy: every following token that is not itself an option is an
            // image path, so the natural spelling "--add-images a b c" works.
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                opts.images.emplace_back(argv[++i]);
            }
        } else {
            throw UsageError{};
        }
    }
    return opts;
}

/// A relative path is made absolute by the AddPhotos task itself (against the
/// process's cwd, which is what the user typed), so the CLI passes the image
/// paths through untouched. That is deliberate: the CLI must not have a private
/// notion of how a path is stored, exactly as it has no private notion of what a
/// task accepts.

int run_list_tasks() {
    alphashape::TaskRegistry& registry = alphashape::builtin_registry();
    for (const std::string& name : registry.names()) {
        const alphashape::TaskEntry* entry = registry.find(name);
        const bool enabled = entry != nullptr && entry->enabled;
        std::printf("%s%s\n", name.c_str(), enabled ? "" : " (disabled)");
    }
    return 0;
}

/// Prints the Vulkan devices the GPU layer can see. Exit 0 when at least one device is listed, 1
/// otherwise (the reason is printed), so packaging smoke tests can prove the loader and driver
/// were found.
int run_gpu_info() {
    const std::vector<alphashape::gpu::DeviceInfo> devices = alphashape::gpu::enumerate_devices();
    if (devices.empty()) {
        std::printf("gpu: unavailable (%s)\n", alphashape::gpu::unavailable_reason().c_str());
        return 1;
    }
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const alphashape::gpu::DeviceInfo& device = devices[i];
        std::printf("gpu device %zu: %s (%s, vendor 0x%04x, %llu MiB)\n", i, device.name.c_str(),
                    device.software ? "software" : (device.integrated ? "integrated" : "discrete"),
                    device.vendor_id,
                    static_cast<unsigned long long>(device.memory_bytes >> 20U));
    }
    return 0;
}

/// Prints the PROJ library version and the proj.db the engine resolved. Exit 1 when no database
/// was found (coordinate transforms would fail).
int run_proj_info() {
    const alphashape::crs::ProjDataInfo info = alphashape::crs::proj_data_info();
    std::printf("proj %s\n", alphashape::crs::proj_version().c_str());
    std::printf("proj database: %s\n", info.database_path.string().c_str());
    return info.database_path.empty() ? 1 : 0;
}

int run_list_params(const std::string& task_name) {
    alphashape::TaskRegistry& registry = alphashape::builtin_registry();
    const alphashape::TaskEntry& entry = registry.require(task_name);  // throws if unknown
    for (const alphashape::ParamSpec& spec : entry.schema.specs()) {
        std::printf("%s\t%s\t%s\t%s\n", spec.name.c_str(), type_name(spec.type),
                    alphashape::format_param_value(spec.default_value).c_str(), spec.doc.c_str());
    }
    return 0;
}

int run_project(const std::string& project, const std::string& task_name,
                const std::vector<std::string>& set_values) {
    alphashape::TaskRegistry& registry = alphashape::builtin_registry();
    const alphashape::TaskEntry& entry = registry.require(task_name);  // throws if unknown

    auto document = alphashape::ProjectIO::load(project);
    alphashape::Params params = alphashape::make_params(entry);
    // Hold the schema copy: parse below borrows ParamSpec pointers from it.
    const std::vector<alphashape::ParamSpec> specs = entry.schema.specs();
    for (const std::string& assignment : set_values) {
        const std::size_t eq = assignment.find('=');
        if (eq == std::string::npos) {
            throw alphashape::ValueError("--set expects key=value, got '" + assignment + "'");
        }
        const std::string key = assignment.substr(0, eq);
        const std::string text = assignment.substr(eq + 1);
        const alphashape::ParamSpec* spec = find_spec(specs, key);
        if (spec == nullptr) {
            throw alphashape::InvalidArgumentError(key);
        }
        // Parse against the declared type: the same spelling the XML reader uses.
        params.set(key, alphashape::parse_param_value(spec->type, text));
    }

    const bool tty = alphashape::platform::is_tty(alphashape::platform::Stream::Stderr);
    alphashape::run_task(registry, *document, task_name, params, make_progress(tty));
    if (tty) {
        std::fprintf(stderr, "\n");
    }

    alphashape::ProjectIO::save(*document, project);
    return 0;
}

int run_new(const std::string& project, const std::vector<std::string>& images) {
    alphashape::Document document;
    document.set_path(project);
    alphashape::Workspace& workspace = document.workspace();
    alphashape::Chunk& chunk = workspace.add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());

    // Creating a document from images is just the AddPhotos task: the CLI has no
    // private code that builds cameras and sensors. A relative image path is
    // resolved by the task against the process's cwd (the project file does not
    // exist yet, so there is nothing to resolve against), which is what a user
    // typing `--new proj.psx --add-images a.jpg` expects.
    alphashape::TaskRegistry& registry = alphashape::builtin_registry();
    const alphashape::TaskEntry& entry = registry.require("AddPhotos");
    alphashape::Params params = alphashape::make_params(entry);
    alphashape::ParamValue filenames = images;
    params.set("filenames", std::move(filenames));

    const bool tty = alphashape::platform::is_tty(alphashape::platform::Stream::Stderr);
    alphashape::run_task(registry, document, "AddPhotos", params, make_progress(tty));
    if (tty) {
        std::fprintf(stderr, "\n");
    }

    alphashape::ProjectIO::save(document, project);
    return 0;
}

int dispatch(const Options& opts) {
    if (opts.help) {
        std::fputs(kUsage, stdout);
        return 0;
    }

    // Exactly one primary mode may be selected; mixing them is a usage error.
    const int modes = (opts.list_tasks ? 1 : 0) + (opts.gpu_info ? 1 : 0) + (opts.proj_info ? 1 : 0) + (opts.has_list_params ? 1 : 0) +
                      (opts.has_new ? 1 : 0) + (opts.has_project ? 1 : 0);
    if (modes > 1) {
        throw UsageError{};
    }
    if (opts.has_task && !opts.has_project) {
        throw UsageError{};  // --task only makes sense with --project
    }
    if (opts.has_project && !opts.has_task) {
        throw UsageError{};
    }

    if (opts.list_tasks) return run_list_tasks();
    if (opts.gpu_info) return run_gpu_info();
    if (opts.proj_info) return run_proj_info();
    if (opts.has_list_params) return run_list_params(opts.list_params);
    if (opts.has_new) return run_new(opts.new_project, opts.images);
    if (opts.has_project) return run_project(opts.project, opts.task, opts.set_values);
    throw UsageError{};  // no mode selected
}

}  // namespace

int main(int argc, char** argv) {
    // A redirected byte stream must not pass through CRLF translation, and a
    // Windows console needs the UTF-8 code page before the first message.
    alphashape::platform::set_binary_mode(stdout);
    alphashape::platform::set_binary_mode(stderr);
    alphashape::platform::setup_console_utf8();
    try {
        const Options opts = parse_args(argc, argv);
        return dispatch(opts);
    } catch (const UsageError&) {
        std::fputs(kUsage, stderr);
        return 64;
    } catch (const alphashape::UnsupportedTaskError& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    } catch (const alphashape::Error& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
