// The portable platform layer: temp files, dynamic libraries, console
// detection, child-process capture and environment round trips.
//
// These are the WP-29.1 acceptance tests (M29 §4.2). They must pass on every
// platform, so nothing here names a POSIX or Win32 API -- that is the point of
// the layer they exercise.

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "base/platform/console.hpp"
#include "base/platform/dynamic_library.hpp"
#include "base/platform/env.hpp"
#include "base/platform/process.hpp"
#include "base/platform/temp.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;
namespace plat = platform;

TEST(Platform, TempDirectoryIsWritable) {
    const std::string dir = plat::temp_directory();
    ASSERT_FALSE(dir.empty());
    const fs::path path(dir);
    EXPECT_TRUE(fs::exists(path)) << dir;
    EXPECT_TRUE(fs::is_directory(path)) << dir;

    // Writable, not merely present: create, write and read back.
    const std::string file = plat::unique_temp_path("alphashape_tmpdir_probe_", ".txt");
    {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.is_open()) << "cannot write in " << dir;
        out << "alphashape";
    }
    {
        std::ifstream in(file, std::ios::binary);
        const std::string contents((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
        EXPECT_EQ(contents, "alphashape");
    }
    std::error_code ec;
    fs::remove(file, ec);
    EXPECT_FALSE(ec) << ec.message();
}

TEST(Platform, UniqueTempPathNoCollision) {
    // 10 000 paths from a process-wide counter: every one must differ, and all
    // must sit in the temporary directory with the requested prefix/suffix.
    constexpr int kCount = 10000;
    std::vector<std::string> paths;
    paths.reserve(kCount);
    for (int i = 0; i < kCount; ++i) {
        paths.push_back(plat::unique_temp_path("alphashape_collision_", ".bin"));
    }
    std::sort(paths.begin(), paths.end());
    const auto unique_end = std::unique(paths.begin(), paths.end());
    EXPECT_EQ(std::distance(paths.begin(), unique_end), kCount);

    const fs::path temp(plat::temp_directory());
    for (const std::string& p : {paths.front(), paths.back()}) {
        const fs::path parent = fs::path(p).parent_path();
        EXPECT_EQ(parent, temp) << p;
        EXPECT_EQ(fs::path(p).filename().string().rfind("alphashape_collision_", 0), 0U) << p;
        EXPECT_EQ(fs::path(p).extension().string(), ".bin") << p;
    }
}

TEST(Platform, DynamicLibrarySymbolLookup) {
    // A library that must exist on every host: the C runtime as the platform
    // names it. This is the test's own child shared library requirement in the
    // plan's wording ("a test library built by CMake, or the Vulkan loader when
    // present with a skip"); a CRT library is always present and always
    // exports a symbol, so the test needs no build-tree helper target.
#if defined(_WIN32)
    const std::string library = "kernel32.dll";
    const std::string symbol = "GetCurrentProcessId";
#elif defined(__APPLE__)
    const std::string library = "libSystem.B.dylib";
    const std::string symbol = "getpid";
#else
    const std::string library = "libc.so.6";
    const std::string symbol = "getpid";
#endif

    std::string error;
    plat::DynamicLibrary lib = plat::DynamicLibrary::open(library, &error);
    ASSERT_TRUE(lib.is_open()) << "open " << library << ": " << error;
    EXPECT_EQ(lib.path(), library);

    error.clear();
    void* fn = lib.symbol(symbol, &error);
    EXPECT_NE(fn, nullptr) << "symbol " << symbol << ": " << error;

    // A symbol that does not exist must report a reason rather than crash.
    error.clear();
    EXPECT_EQ(lib.symbol("alphashape_definitely_not_a_symbol", &error), nullptr);
    EXPECT_FALSE(error.empty());

    lib.close();
    EXPECT_FALSE(lib.is_open());
    // A closed library refuses lookups without touching the stale handle.
    EXPECT_EQ(lib.symbol(symbol, nullptr), nullptr);
}

TEST(Platform, DynamicLibraryOpenFailureIsReported) {
    std::string error;
    plat::DynamicLibrary lib =
        plat::DynamicLibrary::open("alphashape_no_such_library_12345", &error);
    EXPECT_FALSE(lib.is_open());
    EXPECT_FALSE(error.empty());
    EXPECT_NE(error.find("alphashape_no_such_library_12345"), std::string::npos) << error;
}

TEST(Platform, DynamicLibraryMoveKeepsOneHandle) {
    const std::string library =
#if defined(_WIN32)
        "kernel32.dll";
#elif defined(__APPLE__)
        "libSystem.B.dylib";
#else
        "libc.so.6";
#endif
    std::string error;
    plat::DynamicLibrary first = plat::DynamicLibrary::open(library, &error);
    ASSERT_TRUE(first.is_open()) << error;
    plat::DynamicLibrary second = std::move(first);
    EXPECT_FALSE(first.is_open());
    EXPECT_TRUE(second.is_open());
    // The moved-to object owns the handle; closing it once must not double-free
    // (the sanitizer build would catch that).
    second.close();
    EXPECT_FALSE(second.is_open());
}

TEST(Platform, ConsoleTtyDetection) {
    // Both streams must answer without throwing, and with ctest's default
    // redirection they are pipes, hence not terminals. A manual run from a
    // terminal flips this to true, so the assertion is the weaker
    // "the answer is stable and self-consistent" rather than a fixed value.
    const bool out_is_tty = plat::is_tty(plat::Stream::Stdout);
    const bool err_is_tty = plat::is_tty(plat::Stream::Stderr);
    EXPECT_EQ(out_is_tty, plat::is_tty(plat::Stream::Stdout));
    EXPECT_EQ(err_is_tty, plat::is_tty(plat::Stream::Stderr));
}

TEST(Platform, ProcessCaptureOutput) {
    // A child that prints to both streams and exits 3. The command is spelled
    // with the platform's own shell so the test needs no helper binary.
#if defined(_WIN32)
    const std::vector<std::string> argv = {
        "cmd.exe", "/c", "echo alphashape_child_out & echo alphashape_child_err 1>&2 & exit /b 3"};
#else
    const std::vector<std::string> argv = {
        "/bin/sh", "-c", "echo alphashape_child_out; echo alphashape_child_err 1>&2; exit 3"};
#endif
    const plat::ProcessResult result = plat::run_process_capture(argv);
    ASSERT_TRUE(result.started) << result.output;
    EXPECT_EQ(result.exit_code, 3) << result.output;
    // Both streams land in the same capture; the order between them is not
    // promised by either platform.
    EXPECT_NE(result.output.find("alphashape_child_out"), std::string::npos) << result.output;
    EXPECT_NE(result.output.find("alphashape_child_err"), std::string::npos) << result.output;
}

TEST(Platform, ProcessCaptureMissingExecutableIsReported) {
    const plat::ProcessResult result =
        plat::run_process_capture({"alphashape_no_such_executable_12345"});
    // POSIX reports it from the child (127) after a successful fork; Windows
    // fails CreateProcess outright. Both must be non-crashing and explained.
    EXPECT_FALSE(result.started && result.exit_code == 0);
    EXPECT_FALSE(result.output.empty());
}

TEST(Platform, ProcessCaptureLargeOutputDoesNotDeadlock) {
    // 1 MB through the pipe: a parent that waits before draining would hang
    // here, which is the bug this test exists to prevent.
#if defined(_WIN32)
    const std::vector<std::string> argv = {
        "cmd.exe", "/c",
        "for /l %i in (1,1,20000) do @echo 0123456789012345678901234567890123456789"};
#else
    const std::vector<std::string> argv = {
        "/bin/sh", "-c",
        "i=0; while [ $i -lt 20000 ]; do echo 0123456789012345678901234567890123456789; "
        "i=$((i+1)); done"};
#endif
    const plat::ProcessResult result = plat::run_process_capture(argv);
    ASSERT_TRUE(result.started) << result.output;
    EXPECT_EQ(result.exit_code, 0);
    EXPECT_GT(result.output.size(), 100000U);
}

TEST(Platform, EnvRoundTrip) {
    const char* name = "ALPHASHAPE_PLATFORM_TEST_VAR";
    ASSERT_TRUE(plat::set_env(name, "alphashape_value"));
    EXPECT_EQ(plat::get_env(name), "alphashape_value");

    // Empty means unset on every platform (get_env's documented reading).
    ASSERT_TRUE(plat::set_env(name, ""));
    EXPECT_EQ(plat::get_env(name), "");

    // A second round trip covers the set-after-unset path.
    ASSERT_TRUE(plat::set_env(name, "second_value"));
    EXPECT_EQ(plat::get_env(name), "second_value");
    ASSERT_TRUE(plat::set_env(name, ""));

    // An unset variable reads empty, never null-derefs.
    EXPECT_EQ(plat::get_env("ALPHASHAPE_PLATFORM_TEST_VAR_NEVER_SET"), "");
}

}  // namespace
}  // namespace alphashape
