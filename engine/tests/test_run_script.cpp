// WP-11.10: the RunScript task. Every test skips only when python3 is absent
// from PATH (the task itself reports "RunScript: python3 not found").

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
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

std::vector<std::string> script_output_keys(const Document& document) {
    std::vector<std::string> keys;
    const Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        return keys;
    }
    for (const auto& item : chunk->meta().items) {
        const std::string& key = item.first;
        if (key.rfind("RunScript/", 0) == 0 && key.size() > 17 &&
            key.compare(key.size() - 7, 7, "/output") == 0) {
            keys.push_back(key);
        }
    }
    return keys;
}

bool is_valid_utf8(const std::string& text) {
    std::size_t i = 0;
    while (i < text.size()) {
        const unsigned char lead = static_cast<unsigned char>(text[i]);
        std::size_t extra = 0;
        if (lead < 0x80U) {
            extra = 0;
        } else if ((lead & 0xE0U) == 0xC0U) {
            extra = 1;
        } else if ((lead & 0xF0U) == 0xE0U) {
            extra = 2;
        } else if ((lead & 0xF8U) == 0xF0U) {
            extra = 3;
        } else {
            return false;
        }
        if (i + extra >= text.size()) {
            return false;
        }
        for (std::size_t step = 1; step <= extra; ++step) {
            if ((static_cast<unsigned char>(text[i + step]) & 0xC0U) != 0x80U) {
                return false;
            }
        }
        i += extra + 1;
    }
    return true;
}

/// Waits up to ~5 s for `pid` to disappear.
bool wait_until_dead(long long pid) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        if (!platform::process_is_alive(pid)) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
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

TEST(RunScript, NonZeroExitCarriesOutputTail) {
    if (!python3_available()) {
        GTEST_SKIP() << "python3 not found on PATH";
    }
    Project project;
    const std::string script = write_script(
        project.dir, "boom.py",
        "import sys\n"
        "sys.stderr.write('boom-marker\\n')\n"
        "sys.stderr.flush()\n"
        "sys.exit(3)\n");
    try {
        run_script(project.document, script, {}, 60);
        FAIL() << "RunScript should have thrown for exit 3";
    } catch (const Error& error) {
        const std::string message = error.what();
        // The first line is the spec's message, unchanged; the captured output
        // follows because a failed task cannot store it on the document.
        EXPECT_EQ(message.rfind("RunScript failed: exit 3", 0), 0U) << message;
        EXPECT_NE(message.find("boom-marker"), std::string::npos) << message;
    }
}

TEST(RunScript, TimeoutKills) {
    if (!python3_available()) {
        GTEST_SKIP() << "python3 not found on PATH";
    }
    Project project;
    const fs::path pid_path = fs::path(platform::path_from_utf8(project.dir.path())) /
                              "pids.txt";
    const std::string pid_file = platform::path_to_utf8(pid_path);
    // Writes its own pid, waits for the job object / process group assignment,
    // then starts a grandchild and records that pid too. Both must die.
    const std::string script = write_script(
        project.dir, "sleep.py",
        "import os, subprocess, sys, time\n"
        "with open(sys.argv[-1], 'w') as handle:\n"
        "    handle.write('%d' % os.getpid())\n"
        "time.sleep(0.5)\n"
        "child = subprocess.Popen([sys.executable, '-c',\n"
        "                          'import time; time.sleep(30)'])\n"
        "with open(sys.argv[-1], 'a') as handle:\n"
        "    handle.write(' %d' % child.pid)\n"
        "time.sleep(30)\n");
    const auto started = std::chrono::steady_clock::now();
    try {
        run_script(project.document, script, {pid_file}, 5);
        FAIL() << "RunScript should have timed out";
    } catch (const Error& error) {
        EXPECT_NE(std::string(error.what()).find("RunScript timed out after 5 s"),
                  std::string::npos)
            << error.what();
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - started)
                             .count();
    if (test::perf_asserts_enabled()) {
        EXPECT_LT(elapsed, 10000);
    }

    long long own_pid = 0;
    long long grandchild_pid = 0;
    {
        std::ifstream in(pid_path);
        in >> own_pid >> grandchild_pid;
    }
    ASSERT_GT(own_pid, 0) << "the script never recorded its pid";
    ASSERT_GT(grandchild_pid, 0) << "the script never recorded its child's pid";
    EXPECT_TRUE(wait_until_dead(own_pid))
        << "pid " << own_pid << " survived the timeout";
    EXPECT_TRUE(wait_until_dead(grandchild_pid))
        << "grandchild pid " << grandchild_pid << " survived the timeout";
}

TEST(RunScript, OutputTruncatedTo8KB) {
    if (!python3_available()) {
        GTEST_SKIP() << "python3 not found on PATH";
    }
    Project project;
    // 8191 ASCII bytes, then a 3-byte code point (U+20AC) whose bytes straddle
    // the 8192-byte cut, then filler.
    const std::string script = write_script(
        project.dir, "big.py",
        "import sys\n"
        "sys.stdout.write('x' * 8191 + '\\u20ac' + 'y' * 100)\n"
        "sys.stdout.flush()\n");
    run_script(project.document, script, {}, 60);
    const std::string output = script_output(project.document);
    EXPECT_LE(output.size(), 8U * 1024U);
    EXPECT_TRUE(is_valid_utf8(output)) << "truncation split a UTF-8 character";
    // The cut backs off the straddling character, so exactly the 8191 ASCII
    // bytes survive rather than a partial euro sign.
    EXPECT_EQ(output.size(), 8191U);
}

TEST(RunScript, OutputIndexIncrements) {
    if (!python3_available()) {
        GTEST_SKIP() << "python3 not found on PATH";
    }
    Project project;
    const std::string script = write_script(
        project.dir, "twice.py", "print('run-marker')\n");
    run_script(project.document, script, {}, 60);
    run_script(project.document, script, {}, 60);
    const std::vector<std::string> keys = script_output_keys(project.document);
    ASSERT_EQ(keys.size(), 2U);
    EXPECT_NE(std::find(keys.begin(), keys.end(), "RunScript/1/output"), keys.end())
        << keys[0];
    EXPECT_NE(std::find(keys.begin(), keys.end(), "RunScript/2/output"), keys.end());
}

TEST(RunScript, PythonPathPrepended) {
    if (!python3_available()) {
        GTEST_SKIP() << "python3 not found on PATH";
    }
    Project project;
    const fs::path module_path =
        fs::path(platform::path_from_utf8(project.dir.path())) / "python_module";
    std::error_code ec;
    fs::create_directories(module_path, ec);
    const std::string module_dir = platform::path_to_utf8(module_path);
    const std::string previous_module =
        platform::get_env("ALPHASHAPE_PYTHON_MODULE_DIR");
    const std::string previous_path = platform::get_env("PYTHONPATH");
    const std::string existing = "alphashape-original-path";
    ASSERT_TRUE(platform::set_env("ALPHASHAPE_PYTHON_MODULE_DIR", module_dir));
    ASSERT_TRUE(platform::set_env("PYTHONPATH", existing));

    const std::string script = write_script(
        project.dir, "pathprobe.py",
        "import os, sys\n"
        "print('PYTHONPATH:' + os.environ.get('PYTHONPATH', ''))\n"
        "print('SYSPATH:' + '|'.join(sys.path))\n");
    std::string output;
    try {
        run_script(project.document, script, {}, 60);
        output = script_output(project.document);
    } catch (...) {
        (void)platform::set_env("ALPHASHAPE_PYTHON_MODULE_DIR", previous_module);
        (void)platform::set_env("PYTHONPATH", previous_path);
        throw;
    }
    (void)platform::set_env("ALPHASHAPE_PYTHON_MODULE_DIR", previous_module);
    (void)platform::set_env("PYTHONPATH", previous_path);

#if defined(_WIN32)
    const std::string expected = module_dir + ";" + existing;
#else
    const std::string expected = module_dir + ":" + existing;
#endif
    EXPECT_NE(output.find("PYTHONPATH:" + expected), std::string::npos) << output;
    EXPECT_NE(output.find("SYSPATH:"), std::string::npos) << output;
    EXPECT_NE(output.find(module_dir), std::string::npos) << output;
    EXPECT_NE(output.find(existing), std::string::npos) << output;
}

TEST(RunScript, UnsavedDocumentThrows) {
    // No python3 needed: the task refuses before spawning anything.
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    try {
        run_script(document, "unused.py", {}, 60);
        FAIL() << "RunScript should refuse an unsaved document";
    } catch (const Error& error) {
        EXPECT_NE(std::string(error.what()).find("no project path"),
                  std::string::npos)
            << error.what();
    }
}

}  // namespace
}  // namespace alphashape
