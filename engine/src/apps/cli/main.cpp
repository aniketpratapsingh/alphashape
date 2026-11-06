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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "base/platform/console.hpp"
#include "base/platform/env.hpp"
#include "base/utils/error.hpp"
#include "common/algorithm/markers/marker_sheet.hpp"
#include "common/core/document.hpp"
#include "common/io/project_io.hpp"
#include "common/tasks/params.hpp"
#include "common/tasks/registry.hpp"
#include "gpu/gpu.hpp"
#include "image/io/image_io.hpp"
#include "model/crs/proj_context.hpp"

namespace {

constexpr const char* kUsage =
    "Usage:\n"
    "  alphashape-cli --list-tasks\n"
    "  alphashape-cli --gpu-info\n"
    "  alphashape-cli --proj-info\n"
    "  alphashape-cli --list-params <Task>\n"
    "  alphashape-cli --project <file.psx> --task <Task> [--set key=value ...]\n"
    "      [--save-layout dir|psz] [--chunks <id,id,...>] [--python-path <dir>]\n"
    "  alphashape-cli --new <file.psx> --add-images <img> [<img> ...]\n"
    "      [--save-layout dir|psz] [--chunks <id,id,...>]\n"
    "  alphashape-cli --make-sheet <type> --out <file.pdf|file.png>\n"
    "      [--paper A4|Letter] [--size-mm 25] [--first-code N] [--count N]\n"
    "      [--dpi 300] [--cols 9 --rows 6 --square-mm 25]\n"
    "  alphashape-cli --help\n"
    "\n"
    "Sheet types: circular12 circular14 circular16 circular20 cross\n"
    "             apriltag36h11 chessboard\n";

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
    /// "dir" keeps the .psx + .files layout; "psz" writes one file; "" leaves
    /// the choice to the target's extension (.psz means the single-file form).
    std::string save_layout;
    /// Save only these chunk ids; empty means every chunk.
    std::vector<std::int32_t> chunk_ids;
    /// Directory prepended to RunScript's PYTHONPATH (the built module).
    std::string python_path;
    bool has_make_sheet = false;
    std::string make_sheet;
    bool has_out = false;
    std::string out;
    std::string paper = "A4";
    double size_mm = 25.0;
    int first_code = 1;
    int count = 0;
    double dpi = 300.0;
    int cols = 9;
    int rows = 6;
    double square_mm = 25.0;
};

int parse_int(const std::string& text) {
    try {
        std::size_t used = 0;
        const int value = std::stoi(text, &used);
        if (used == text.size()) {
            return value;
        }
    } catch (const std::exception&) {
    }
    throw UsageError{};
}

std::vector<std::int32_t> parse_chunk_ids(const std::string& text) {
    std::vector<std::int32_t> ids;
    std::size_t start = 0;
    for (;;) {
        const std::size_t comma = text.find(',', start);
        const std::string part =
            text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (part.empty()) {
            throw UsageError{};
        }
        ids.push_back(static_cast<std::int32_t>(parse_int(part)));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return ids;
}

/// The path a save writes for the requested layout. An unspecified layout is
/// inferred from the target: a ".psz" path is the single-file form, anything
/// else the directory form. An explicit layout must match the extension, and a
/// mismatch is a usage error rather than a silent write under another name.
std::string save_path_for(const std::string& project, const std::string& layout) {
    const std::filesystem::path path = alphashape::platform::path_from_utf8(project);
    const bool psz_target = path.extension() == ".psz";
    if (layout == "psz" && !psz_target) {
        throw UsageError{};
    }
    if (layout == "dir" && psz_target) {
        throw UsageError{};
    }
    return alphashape::platform::path_to_utf8(path);
}

double parse_double(const std::string& text) {
    try {
        std::size_t used = 0;
        const double value = std::stod(text, &used);
        if (used == text.size()) {
            return value;
        }
    } catch (const std::exception&) {
    }
    throw UsageError{};
}

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
        } else if (arg == "--save-layout") {
            opts.save_layout = take_value();
            if (opts.save_layout != "dir" && opts.save_layout != "psz") {
                throw UsageError{};
            }
        } else if (arg == "--chunks") {
            opts.chunk_ids = parse_chunk_ids(take_value());
        } else if (arg == "--python-path") {
            opts.python_path = take_value();
        } else if (arg == "--make-sheet") {
            opts.has_make_sheet = true;
            opts.make_sheet = take_value();
        } else if (arg == "--out") {
            opts.has_out = true;
            opts.out = take_value();
        } else if (arg == "--paper") {
            opts.paper = take_value();
        } else if (arg == "--size-mm") {
            opts.size_mm = parse_double(take_value());
        } else if (arg == "--first-code") {
            opts.first_code = parse_int(take_value());
        } else if (arg == "--count") {
            opts.count = parse_int(take_value());
        } else if (arg == "--dpi") {
            opts.dpi = parse_double(take_value());
        } else if (arg == "--cols") {
            opts.cols = parse_int(take_value());
        } else if (arg == "--rows") {
            opts.rows = parse_int(take_value());
        } else if (arg == "--square-mm") {
            opts.square_mm = parse_double(take_value());
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
    // printf takes bytes, so the path is converted to UTF-8 rather than
    // narrowed with the ANSI code page: an install under "C:\Users\Jörg\..."
    // would otherwise print mojibake and the smoke test's comparison against
    // the install directory would fail.
    std::printf("proj database: %s\n",
                alphashape::platform::path_to_utf8(info.database_path).c_str());
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
                const std::vector<std::string>& set_values,
                const std::string& save_layout,
                const std::vector<std::int32_t>& chunk_ids) {
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

    document->save(save_path_for(project, save_layout), chunk_ids);
    return 0;
}

int run_new(const std::string& project, const std::vector<std::string>& images,
            const std::string& save_layout,
            const std::vector<std::int32_t>& chunk_ids) {
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

    document.save(save_path_for(project, save_layout), chunk_ids);
    return 0;
}

void save_bytes(const std::string& path, const std::vector<std::uint8_t>& data) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw alphashape::Error("cannot open '" + path + "' for writing");
    }
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
    file.close();
    if (!file) {
        throw alphashape::Error("writing '" + path + "' failed");
    }
}

int run_make_sheet(const Options& opts) {
    if (!opts.has_out) {
        throw UsageError{};
    }
    const std::optional<alphashape::SheetTarget> target =
        alphashape::parse_sheet_target(opts.make_sheet);
    if (!target.has_value()) {
        throw alphashape::ValueError("unknown sheet type: " + opts.make_sheet);
    }
    alphashape::SheetOptions sheet;
    sheet.target = *target;
    sheet.paper = opts.paper;
    sheet.target_size_mm = opts.size_mm;
    sheet.first_code = opts.first_code;
    sheet.count = opts.count;
    sheet.board_cols = opts.cols;
    sheet.board_rows = opts.rows;
    sheet.square_mm = opts.square_mm;
    const std::string& path = opts.out;
    const bool png =
        path.size() >= 4U && path[path.size() - 4U] == '.' &&
        (path[path.size() - 3U] == 'p' || path[path.size() - 3U] == 'P') &&
        (path[path.size() - 2U] == 'n' || path[path.size() - 2U] == 'N') &&
        (path[path.size() - 1U] == 'g' || path[path.size() - 1U] == 'G');
    if (png) {
        const int pages = alphashape::marker_sheet_page_count(sheet);
        if (pages > 1) {
            throw alphashape::ValueError(
                "PNG output holds a single page, but this sheet needs " +
                std::to_string(pages) +
                " pages; write a PDF or reduce the count");
        }
        const alphashape::Image image =
            alphashape::marker_sheet_raster(sheet, opts.dpi, 0);
        alphashape::write_png(path, image);
    } else {
        save_bytes(path, alphashape::marker_sheet_pdf(sheet));
    }
    return 0;
}

int dispatch(const Options& opts) {
    if (opts.help) {
        std::fputs(kUsage, stdout);
        return 0;
    }

    // Exactly one primary mode may be selected; mixing them is a usage error.
    const int modes = (opts.list_tasks ? 1 : 0) + (opts.gpu_info ? 1 : 0) + (opts.proj_info ? 1 : 0) + (opts.has_list_params ? 1 : 0) +
                      (opts.has_new ? 1 : 0) + (opts.has_project ? 1 : 0) +
                      (opts.has_make_sheet ? 1 : 0);
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
    if (opts.has_new) {
        return run_new(opts.new_project, opts.images, opts.save_layout, opts.chunk_ids);
    }
    if (opts.has_make_sheet) return run_make_sheet(opts);
    if (opts.has_project) {
        return run_project(opts.project, opts.task, opts.set_values, opts.save_layout,
                           opts.chunk_ids);
    }
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
        if (!opts.python_path.empty()) {
            // RunScript reads this at run time; the CMake define is the default.
            (void)alphashape::platform::set_env("ALPHASHAPE_PYTHON_MODULE_DIR",
                                               opts.python_path);
        }
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
