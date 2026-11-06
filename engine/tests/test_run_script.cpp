// WP-11.10: the RunScript task. Every test skips only when python3 is absent
// from PATH (the task itself reports "RunScript: python3 not found").

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/platform/env.hpp"
#include "base/platform/process.hpp"
#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "common/core/document.hpp"
#include "common/tasks/params.hpp"
#include "common/tasks/registry.hpp"
#include "perf_gate.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

class TempDir {
public:
    TempDir() : path_(platform::unique_temp_path("alphashape_runscript_", "")) {
        std::error_code ec;
        fs::create_directories(platform::path_from_utf8(path_), ec);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(platform::path_from_utf8(path_), ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    const std::string& path() const { return path_; }

private:
    std::string path_;
};

bool python3_available() {
    return !platform::find_executable("python3").empty();
}

std::string write_script(const TempDir& dir, const std::string& name,
                         const std::string& body) {
    const fs::path path = fs::path(platform::path_from_utf8(dir.path())) / name;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << body;
    out.close();
    return platform::path_to_utf8(path);
}

/// A document whose path is set (RunScript refuses an unsaved one) and that
/// has one active chunk to receive the captured output.
struct Project {
    TempDir dir;
    Document document;

    Project() {
        const fs::path path =
            fs::path(platform::path_from_utf8(dir.path())) / "project.psx";
        document.set_path(platform::path_to_utf8(path));
        Chunk& chunk = document.workspace().add_chunk("Chunk 1");
        document.set_active_chunk_id(chunk.id());
    }

    std::string absolute_path() const {
        return platform::path_to_utf8(
            fs::absolute(platform::path_from_utf8(document.path()))
                .lexically_normal());
    }
};

void run_script(Document& document, const std::string& script,
                const std::vector<std::string>& args,
                std::int64_t timeout_s) {
    TaskRegistry& registry = builtin_registry();
    const TaskEntry& entry = registry.require("RunScript");
    Params params = make_params(entry);
    params.set("path", ParamValue{script});
    params.set("args", ParamValue{args});
    params.set("timeout_s", ParamValue{timeout_s});
    run_task(registry, document, "RunScript", params, Progress{});
}

std::string script_output(const Document& document) {
    const Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        return std::string();
    }
    for (const auto& item : chunk->meta().items) {
        const std::string& key = item.first;
        if (key.rfind("RunScript/", 0) == 0 && key.size() > 17 &&
            key.compare(key.size() - 7, 7, "/output") == 0) {
            return item.second;
        }
    }
    return std::string();
}

TEST(RunScript, RunsAndCapturesOutput) {
    if (!python3_available()) {
        GTEST_SKIP() << "python3 not found on PATH";
    }
    Project project;
    const std::string script = write_script(
        project.dir, "args.py",
        "import os, sys\n"
        "print('ARGV:' + '|'.join(sys.argv[1:]))\n"
        "print('ENV:' + os.environ.get('ALPHASHAPE_PROJECT', ''))\n"
        "print('ERR:stderr', file=sys.stderr)\n");
    run_script(project.document, script, {"one", "two words"}, 60);

    const std::string output = script_output(project.document);
    const std::string absolute = project.absolute_path();
    EXPECT_NE(output.find("ARGV:--project|" + absolute + "|one|two words"),
              std::string::npos)
        << output;
    EXPECT_NE(output.find("ENV:" + absolute), std::string::npos) << output;
    EXPECT_NE(output.find("ERR:stderr"), std::string::npos) << output;
}

TEST(RunScript, NonZeroExitThrows) {
    if (!python3_available()) {
        GTEST_SKIP() << "python3 not found on PATH";
    }
    Project project;
    const std::string script =
        write_script(project.dir, "fail.py", "import sys\nsys.exit(3)\n");
    try {
        run_script(project.document, script, {}, 60);
        FAIL() << "RunScript should have thrown for exit 3";
    } catch (const Error& error) {
        EXPECT_NE(std::string(error.what()).find("RunScript failed: exit 3"),
                  std::string::npos)
            << error.what();
    }
}

TEST(RunScript, TimeoutKills) {
    if (!python3_available()) {
        GTEST_SKIP() << "python3 not found on PATH";
    }
    Project project;
    const std::string script =
        write_script(project.dir, "sleep.py", "import time\ntime.sleep(30)\n");
    const auto started = std::chrono::steady_clock::now();
    try {
        run_script(project.document, script, {}, 1);
        FAIL() << "RunScript should have timed out";
    } catch (const Error& error) {
        EXPECT_NE(std::string(error.what()).find("RunScript timed out after 1 s"),
                  std::string::npos)
            << error.what();
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - started)
                             .count();
    if (test::perf_asserts_enabled()) {
        EXPECT_LT(elapsed, 10000);
    }
}

TEST(RunScript, OutputTruncatedTo8KB) {
    if (!python3_available()) {
        GTEST_SKIP() << "python3 not found on PATH";
    }
    Project project;
    const std::string script = write_script(
        project.dir, "big.py",
        "import sys\nsys.stdout.write('x' * 20000)\nsys.stdout.flush()\n");
    run_script(project.document, script, {}, 60);
    const std::string output = script_output(project.document);
    EXPECT_EQ(output.size(), 8U * 1024U);
}

}  // namespace
}  // namespace alphashape
