#include "common/algorithm/jobs/run_script_task.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "base/platform/env.hpp"
#include "base/platform/process.hpp"
#include "base/utils/error.hpp"

#if !defined(ALPHASHAPE_PYTHON_MODULE_DIR)
#define ALPHASHAPE_PYTHON_MODULE_DIR ""
#endif

namespace alphashape {

namespace {

namespace fs = std::filesystem;

constexpr std::size_t kOutputLimit = 8U * 1024U;

/// Cuts `text` to at most `limit` bytes without splitting a UTF-8 character.
std::string truncate_utf8(const std::string& text, std::size_t limit) {
    if (text.size() <= limit) {
        return text;
    }
    std::size_t cut = limit;
    while (cut > 0 &&
           (static_cast<unsigned char>(text[cut]) & 0xC0U) == 0x80U) {
        --cut;
    }
    return text.substr(0, cut);
}

bool all_digits(const std::string& text) {
    return !text.empty() &&
           std::all_of(text.begin(), text.end(),
                       [](char c) { return c >= '0' && c <= '9'; });
}

/// The first free `RunScript/<n>/output` index on the chunk, starting at 1.
std::int64_t next_output_index(const Meta& meta) {
    std::int64_t next = 1;
    for (const auto& [key, value] : meta.items) {
        (void)value;
        constexpr const char* prefix = "RunScript/";
        constexpr const char* suffix = "/output";
        constexpr std::size_t prefix_size = 10;
        const std::size_t suffix_size = 7;
        if (key.size() <= prefix_size + suffix_size ||
            key.compare(0, prefix_size, prefix) != 0 ||
            key.compare(key.size() - suffix_size, suffix_size, suffix) != 0) {
            continue;
        }
        const std::string middle =
            key.substr(prefix_size, key.size() - prefix_size - suffix_size);
        if (!all_digits(middle)) {
            continue;
        }
        std::int64_t index = 0;
        for (const char digit : middle) {
            index = index * 10 + static_cast<std::int64_t>(digit - '0');
        }
        next = std::max(next, index + 1);
    }
    return next;
}

std::string module_directory() {
    const std::string configured = platform::get_env("ALPHASHAPE_PYTHON_MODULE_DIR");
    return configured.empty() ? std::string(ALPHASHAPE_PYTHON_MODULE_DIR) : configured;
}

/// The python command: python3 everywhere; on Windows fall back to python and
/// then the py launcher with -3.
std::vector<std::string> python_command() {
    const std::string python3 = platform::find_executable("python3");
    if (!python3.empty()) {
        return {python3};
    }
#if defined(_WIN32)
    const std::string python = platform::find_executable("python");
    if (!python.empty()) {
        return {python};
    }
    const std::string py = platform::find_executable("py");
    if (!py.empty()) {
        return {py, "-3"};
    }
#endif
    return {};
}

}  // namespace

void RunScriptTask::declare_params(Params& params) const {
    params.declare({"path", ParamType::String, ParamValue{},
                    "Path to the Python script to run."});
    params.declare({"args", ParamType::StringList, std::vector<std::string>{},
                    "Arguments appended after --project <path>."});
    ParamSpec timeout{"timeout_s", ParamType::Int, std::int64_t{600},
                      "Wall-clock limit; the child and its process tree are "
                      "killed when it expires."};
    timeout.minimum = std::int64_t{1};
    params.declare(timeout);
}

std::vector<Stage> RunScriptTask::stages() const {
    return {Stage{"run", 0}};
}

void RunScriptTask::apply(Document& document, Params& params,
                          const Progress& progress) {
    progress.update("run", 0.0);

    if (document.path().empty()) {
        throw Error(
            "RunScript: the document has no project path; save the project first");
    }
    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("RunScript needs an active chunk");
    }

    const std::string project = platform::path_to_utf8(
        fs::absolute(platform::path_from_utf8(document.path())).lexically_normal());
    const std::string script = params.get_string("path");
    const std::vector<std::string> extra_args = params.get_string_list("args");

    std::vector<std::string> argv = python_command();
    if (argv.empty()) {
        throw Error("RunScript: python3 not found");
    }
    argv.push_back(script);
    argv.push_back("--project");
    argv.push_back(project);
    for (const std::string& arg : extra_args) {
        argv.push_back(arg);
    }

    const std::int64_t seconds = params.get_int("timeout_s");
    if (seconds < 1) {
        throw ValueError("RunScript: timeout_s must be at least 1");
    }
    const std::int64_t max_ms = std::numeric_limits<int>::max();
    platform::ProcessOptions options;
    options.timeout_ms =
        seconds > max_ms / 1000
            ? static_cast<int>(max_ms)
            : static_cast<int>(seconds * 1000);

    options.env.emplace_back("ALPHASHAPE_PROJECT", project);
    const std::string module_dir = module_directory();
    const std::string existing = platform::get_env("PYTHONPATH");
#if defined(_WIN32)
    const char separator = ';';
#else
    const char separator = ':';
#endif
    std::string pythonpath = module_dir;
    if (!pythonpath.empty() && !existing.empty()) {
        pythonpath.push_back(separator);
    }
    pythonpath += existing;
    if (!pythonpath.empty()) {
        options.env.emplace_back("PYTHONPATH", pythonpath);
    }

    const platform::ProcessResult result =
        platform::run_process_capture(argv, options);
    if (!result.started) {
        throw Error("RunScript failed: " + result.output);
    }
    if (result.timed_out) {
        throw Error("RunScript timed out after " + std::to_string(seconds) + " s");
    }
    if (result.exit_code != 0) {
        throw Error("RunScript failed: exit " + std::to_string(result.exit_code));
    }

    const std::int64_t index = next_output_index(chunk->meta());
    const std::string key =
        "RunScript/" + std::to_string(index) + "/output";
    chunk->meta().set(key, truncate_utf8(result.output, kOutputLimit));
    progress.update("run", 1.0);
}

}  // namespace alphashape
