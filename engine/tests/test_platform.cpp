// The portable platform layer: temp files, dynamic libraries, console
// detection, child-process capture and environment round trips.
//
// These are the WP-29.1 acceptance tests (M29 §4.2). They must pass on every
// platform, so nothing here names a POSIX or Win32 API -- that is the point of
// the layer they exercise. The one exception is the Windows-only DLL search
// order test, which has to plant a file and ask the loader where a module came
// from: that behaviour is only observable through the Win32 API itself.

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>

#include <cwchar>  // _wcsnicmp
#endif

#include <gtest/gtest.h>

#include "base/platform/console.hpp"
#include "base/platform/dynamic_library.hpp"
#include "base/platform/env.hpp"
#include "base/platform/paths.hpp"
#include "base/platform/process.hpp"
#include "base/platform/temp.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;
namespace plat = platform;

/// A data address inside this test binary, used as a module anchor. Data
/// rather than a function so no function-pointer-to-void* conversion is
/// involved (that conversion is only conditionally supported).
const int kAnchorAddress = 0;

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

#if defined(_WIN32)
namespace {

/// Restores the process working directory when the test leaves scope: the
/// search-order test has to chdir, and a leaked cwd would silently change the
/// meaning of every later test in this binary.
class ScopedWorkingDirectory {
public:
    explicit ScopedWorkingDirectory(const std::filesystem::path& next) {
        std::error_code ec;
        previous_ = std::filesystem::current_path(ec);
        if (!ec) {
            std::filesystem::current_path(next, ec);
            changed_ = !ec;
        }
    }
    ~ScopedWorkingDirectory() { restore(); }
    ScopedWorkingDirectory(const ScopedWorkingDirectory&) = delete;
    ScopedWorkingDirectory& operator=(const ScopedWorkingDirectory&) = delete;

    bool changed() const { return changed_; }

    /// Puts the previous directory back. Called explicitly where the temp dir
    /// has to be removed inside the test: Windows cannot delete the process's
    /// current directory.
    void restore() {
        if (!changed_) {
            return;
        }
        std::error_code ec;
        std::filesystem::current_path(previous_, ec);
        changed_ = false;
    }

private:
    std::filesystem::path previous_;
    bool changed_ = false;
};

std::wstring system_directory() {
    std::wstring buffer(MAX_PATH, L'\0');
    for (int attempt = 0; attempt < 4; ++attempt) {
        const UINT length = ::GetSystemDirectoryW(buffer.data(), static_cast<UINT>(buffer.size()));
        if (length == 0) {
            return std::wstring();
        }
        if (length < static_cast<UINT>(buffer.size())) {
            buffer.resize(length);
            return buffer;
        }
        buffer.assign(static_cast<std::size_t>(length) + 1, L'\0');
    }
    return std::wstring();
}

/// The path of a loaded module, via GetModuleFileNameW with a growing buffer.
/// It truncates silently when the buffer is too small, so a full buffer means
/// "grow and ask again".
std::wstring module_file_name(HMODULE module) {
    std::wstring buffer(MAX_PATH, L'\0');
    for (int attempt = 0; attempt < 4; ++attempt) {
        const DWORD length =
            ::GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0) {
            return std::wstring();
        }
        if (length < static_cast<DWORD>(buffer.size()) - 1) {
            buffer.resize(length);
            return buffer;
        }
        buffer.assign(buffer.size() * 2, L'\0');
    }
    return std::wstring();
}

/// Case-insensitive "is `path` inside `directory`". Windows paths compare
/// case-insensitively, and these are ASCII system paths, so the CRT's ordinal
/// comparison in the C locale is enough and needs no extra API declaration.
bool path_is_under(const std::wstring& path, const std::wstring& directory) {
    if (path.size() < directory.size()) {
        return false;
    }
    if (::_wcsnicmp(path.c_str(), directory.c_str(), directory.size()) != 0) {
        return false;
    }
    if (path.size() == directory.size()) {
        return true;
    }
    const wchar_t next = path[directory.size()];
    return next == L'\\' || next == L'/';
}

}  // namespace

TEST(Platform, DynamicLibraryBareNameIgnoresAPlantedCopyInTheWorkingDirectory) {
    // The DLL-planting regression. Plain LoadLibraryW searches the application
    // directory, System32, the Windows directory, the current working
    // directory and PATH, in that order -- so a `vulkan-1.dll` beside the .exe,
    // or in whatever directory the process was started in, is loaded instead of
    // the graphics driver's loader. A bare name must therefore be loaded from
    // System32 only.
    //
    // The planted file is a *valid, loadable* DLL (a copy of a System32 DLL
    // under a new name), not a garbage file: garbage fails to load under the
    // old code too, so the test would pass without proving anything. With a
    // loadable copy in the cwd the two behaviours differ sharply:
    //
    //   * System32 has no vulkan-1.dll (a driverless machine, a bare runner):
    //     the open must FAIL. The old code returned a module -- the planted
    //     one.
    //   * System32 has one: the module that comes back must be the System32
    //     one, never the planted copy.
    const std::wstring system32 = system_directory();
    ASSERT_FALSE(system32.empty()) << "GetSystemDirectoryW failed";

    const std::string dirname = plat::unique_temp_path("alphashape_dllplant_", "");
    const fs::path dir = plat::path_from_utf8(dirname);
    ASSERT_TRUE(fs::create_directories(dir)) << dirname;

    std::error_code ec;
    const fs::path system_loader = fs::path(system32) / L"vulkan-1.dll";
    const bool system_has_loader = fs::exists(system_loader, ec);

    // Plant a copy of a System32 DLL under the hijacked name, and prove the
    // copy is loadable before relying on it: a file that cannot load would make
    // the negative assertion below pass against the vulnerable code too, and
    // the test would prove nothing. Several candidates are tried because a
    // renamed DLL can be refused for reasons of its own (an embedded
    // side-by-side manifest, a DllMain that needs its real name).
    //
    // The loadability probe uses a *different* file name in the same directory,
    // so nothing it maps can be mistaken for the planted module afterwards.
    const fs::path planted = dir / plat::path_from_utf8("vulkan-1.dll");
    bool planted_loads = false;
    int probe_index = 0;
    for (const wchar_t* source : {L"version.dll", L"winmm.dll", L"wtsapi32.dll", L"userenv.dll"}) {
        const fs::path candidate = fs::path(system32) / source;
        ec.clear();
        if (!fs::exists(candidate, ec)) {
            continue;
        }
        ++probe_index;
        const fs::path probe_copy = dir / plat::path_from_utf8(
                                              "alphashape_dllprobe_" + std::to_string(probe_index) +
                                              ".dll");
        ec.clear();
        fs::copy_file(candidate, probe_copy, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            continue;
        }
        std::string probe_error;
        {
            plat::DynamicLibrary probe =
                plat::DynamicLibrary::open(plat::path_to_utf8(probe_copy), &probe_error);
            if (probe.is_open()) {
                planted_loads = true;
            }
        }
        std::error_code remove_error_probe;
        fs::remove(probe_copy, remove_error_probe);
        if (!planted_loads) {
            continue;
        }
        ec.clear();
        fs::copy_file(candidate, planted, fs::copy_options::overwrite_existing, ec);
        planted_loads = !ec;
        if (planted_loads) {
            break;
        }
    }
    if (!planted_loads) {
        std::error_code cleanup_error;
        fs::remove_all(dir, cleanup_error);
        GTEST_SKIP() << "no System32 DLL could be planted as a loadable standalone image, so the "
                        "hijack this test guards against cannot be reproduced on this host";
    }

    ScopedWorkingDirectory cwd(dir);
    ASSERT_TRUE(cwd.changed()) << "could not make " << plat::path_to_utf8(dir) << " the cwd";

    {
        std::string error;
        plat::DynamicLibrary lib = plat::DynamicLibrary::open("vulkan-1.dll", &error);

        if (!system_has_loader) {
            // Nothing in System32 to find, so the only correct answer is a
            // clean, explained failure: the loadable `vulkan-1.dll` one
            // directory away must not be picked up.
            EXPECT_FALSE(lib.is_open())
                << "a bare name loaded the copy in the working directory ("
                << plat::path_to_utf8(planted)
                << "); bare names must resolve against System32 only";
            EXPECT_FALSE(error.empty());
            EXPECT_NE(error.find("vulkan-1.dll"), std::string::npos) << error;
            // Asked by full path, GetModuleHandleW answers only for a module
            // mapped from that exact path, so this catches the planted module
            // without being confused by anything else in the process.
            EXPECT_EQ(::GetModuleHandleW(planted.c_str()), nullptr)
                << "the planted DLL was mapped into this process";
        } else {
            ASSERT_TRUE(lib.is_open())
                << "System32 has vulkan-1.dll but it was not opened: " << error;
            // A module carries only its base name whether it came from System32
            // or from the cwd, so the path is what tells them apart: ask by the
            // shared base name, then read where that module actually lives.
            const HMODULE module = ::GetModuleHandleW(L"vulkan-1.dll");
            ASSERT_NE(module, nullptr) << "the opened module is not named vulkan-1.dll";
            const std::wstring loaded_path = module_file_name(module);
            ASSERT_FALSE(loaded_path.empty()) << "GetModuleFileNameW failed";
            EXPECT_TRUE(path_is_under(loaded_path, system32))
                << "loaded '" << plat::to_utf8(loaded_path) << "', expected it under '"
                << plat::to_utf8(system32) << "'";
            EXPECT_FALSE(path_is_under(loaded_path, dir.wstring()))
                << "the copy planted in the working directory was loaded: "
                << plat::to_utf8(loaded_path);
        }
    }

    // The working directory has to be restored before the temp dir can be
    // removed: Windows refuses to delete the process's current directory.
    cwd.restore();
    std::error_code remove_error;
    fs::remove_all(dir, remove_error);
    EXPECT_FALSE(remove_error) << remove_error.message();
}
#endif  // _WIN32

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

TEST(Platform, Utf8PathRoundTripsNonAscii) {
    // The property the installer depends on: a directory with characters
    // outside the ANSI code page must survive the string -> path -> string
    // round trip. std::filesystem::path's string constructor decodes in the
    // native narrow encoding, so a helper that went through it would return
    // mojibake here on Windows (and on a Linux box with a C locale).
    //
    // The comparison is made in two stages rather than against the input
    // bytes: a path re-spells '/' as '\' on Windows, so equality with the
    // original string would fail for a reason that has nothing to do with
    // encoding. What must hold is that the characters survive and that the
    // second round trip is a no-op.
    const std::string utf8 = "J\xc3\xb6rg/\xc3\xbc-umlaut/\xe6\x97\xa5\xe6\x9c\xac";
    const fs::path path = plat::path_from_utf8(utf8);
    const std::string round_trip = plat::path_to_utf8(path);
    EXPECT_NE(round_trip.find("J\xc3\xb6rg"), std::string::npos) << round_trip;
    EXPECT_NE(round_trip.find("\xc3\xbc-umlaut"), std::string::npos) << round_trip;
    EXPECT_NE(round_trip.find("\xe6\x97\xa5\xe6\x9c\xac"), std::string::npos) << round_trip;
    EXPECT_EQ(plat::path_to_utf8(plat::path_from_utf8(round_trip)), round_trip);
    // The component, not the spelling, is what the conversion must preserve.
    EXPECT_EQ(path.filename(), plat::path_from_utf8("\xe6\x97\xa5\xe6\x9c\xac"));

    // The same as a real directory: create, write and read back through the
    // converted path, so this is not merely a string comparison.
    const std::string dirname = plat::unique_temp_path("alphashape_utf8_\xc3\xb6_", "");
    const fs::path dir = plat::path_from_utf8(dirname);
    ASSERT_TRUE(fs::create_directories(dir));
    const fs::path file = dir / plat::path_from_utf8("\xe6\x97\xa5.txt");
    {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.is_open()) << plat::path_to_utf8(file);
        out << "alphashape";
    }
    EXPECT_TRUE(fs::is_regular_file(file)) << plat::path_to_utf8(file);
    std::error_code ec;
    fs::remove_all(dir, ec);
    EXPECT_FALSE(ec) << ec.message();
}

TEST(Platform, Utf8PathConversionsAgreeOnAscii) {
    // The ASCII case must be unchanged, so the helper can replace path(string)
    // everywhere without altering existing behaviour. Compared through the
    // conversion itself, because a path re-spells separators on Windows.
    for (const std::string& ascii : {std::string("plain"), std::string("plain/path"),
                                     std::string("usr/share/proj")}) {
        const fs::path converted = plat::path_from_utf8(ascii);
        EXPECT_EQ(plat::path_from_utf8(plat::path_to_utf8(converted)), converted);
        EXPECT_EQ(plat::path_to_utf8(plat::path_from_utf8(plat::path_to_utf8(converted))),
                  plat::path_to_utf8(converted));
    }
    // An empty string is an empty path, not an error.
    EXPECT_TRUE(plat::path_from_utf8(std::string()).empty());
    EXPECT_EQ(plat::path_to_utf8(fs::path{}), "");
}

TEST(Platform, Utf8PathOfATemporaryDirectorySurvivesNonAsciiParent) {
    // The smoke test installs under a non-ASCII directory; the engine must be
    // able to name a file below one. This is a direct check of the conversion
    // the PROJ data search performs on the executable's own path.
    const std::string dirname = plat::unique_temp_path("alphashape_\xe2\x82\xac_", "");
    const fs::path dir = plat::path_from_utf8(dirname);
    ASSERT_TRUE(fs::create_directories(dir));
    const std::string nested = plat::path_to_utf8(dir / "share" / "proj");
    EXPECT_NE(nested.find("\xe2\x82\xac"), std::string::npos) << nested;
    EXPECT_EQ(plat::path_to_utf8(plat::path_from_utf8(nested)), nested);
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST(Platform, ExecutablePathNamesThisBinary) {
    const std::string exe = plat::executable_path();
    ASSERT_FALSE(exe.empty()) << "the platform could not name the running image";
    const fs::path path(exe);
    EXPECT_TRUE(fs::exists(path)) << exe;
    EXPECT_TRUE(fs::is_regular_file(path)) << exe;
    // The test process is alphashape_tests(.exe): the stem must say so rather
    // than naming some unrelated loader path.
    EXPECT_NE(path.stem().string().find("alphashape"), std::string::npos) << exe;
}

TEST(Platform, ModulePathOfAnAddressInThisImage) {
    // An address inside this test binary: module_path must resolve it to the
    // image executable_path names. That is the property the PROJ data search
    // relies on when the Python extension module is the loaded image. A data
    // address is used rather than a function pointer because casting a function
    // pointer to void* is only conditionally supported.
    const std::string exe = plat::executable_path();
    ASSERT_FALSE(exe.empty());
    const std::string module = plat::module_path(&kAnchorAddress);    ASSERT_FALSE(module.empty()) << "no image resolved for an address in this binary";

    // Compare canonical paths: the loader may report one through a symlink
    // (Homebrew's bin/) and the other not.
    std::error_code ec_exe;
    std::error_code ec_module;
    const fs::path resolved_exe = fs::weakly_canonical(fs::path(exe), ec_exe);
    const fs::path resolved_module = fs::weakly_canonical(fs::path(module), ec_module);
    ASSERT_FALSE(ec_exe) << ec_exe.message();
    ASSERT_FALSE(ec_module) << ec_module.message();
    EXPECT_EQ(resolved_module.filename(), resolved_exe.filename());
    EXPECT_EQ(resolved_module.parent_path(), resolved_exe.parent_path());

    // A null anchor is a caller bug, not a crash.
    EXPECT_EQ(plat::module_path(nullptr), "");
}

}  // namespace
}  // namespace alphashape
