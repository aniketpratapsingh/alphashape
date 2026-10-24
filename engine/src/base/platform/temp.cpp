#include "base/platform/temp.hpp"

#include <atomic>
#include <filesystem>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "base/platform/env.hpp"

namespace alphashape::platform {

std::string temp_directory() {
#if defined(_WIN32)
    // GetTempPathW returns a trailing backslash and sizes in TCHARs. MAX_PATH
    // is not an upper bound when long paths are enabled, so grow on demand.
    wchar_t buffer[MAX_PATH + 1];
    const DWORD length = ::GetTempPathW(static_cast<DWORD>(std::size(buffer)), buffer);
    if (length == 0) {
        throw std::runtime_error("temp_directory: GetTempPathW failed");
    }
    std::wstring wide(buffer, length);
    while (!wide.empty() && (wide.back() == L'\\' || wide.back() == L'/')) {
        wide.pop_back();
    }
    const std::string utf8 = to_utf8(wide);
    if (utf8.empty()) {
        throw std::runtime_error("temp_directory: the temporary path is not valid UTF-8");
    }
    return utf8;
#else
    // TMPDIR first: macOS and most Linux shells set it to a per-user directory,
    // which is what std::filesystem::temp_directory_path also prefers. Tests
    // rely on that isolation, so the two must not disagree.
    const std::string from_env = get_env("TMPDIR");
    if (!from_env.empty()) {
        std::string trimmed = from_env;
        while (trimmed.size() > 1 && trimmed.back() == '/') {
            trimmed.pop_back();
        }
        return trimmed;
    }
    return "/tmp";
#endif
}

long long process_id() {
#if defined(_WIN32)
    return static_cast<long long>(::GetCurrentProcessId());
#else
    return static_cast<long long>(::getpid());
#endif
}

std::string unique_temp_path(const std::string& prefix, const std::string& suffix) {
    static std::atomic<unsigned long long> counter{0};
    const unsigned long long ticket = counter.fetch_add(1, std::memory_order_relaxed);
    std::string name = prefix;
    name += std::to_string(process_id());
    name += '_';
    name += std::to_string(ticket);
    name += suffix;
    // Both conversions go through the UTF-8 helpers: a temporary directory
    // under a user profile with non-ASCII characters would otherwise be
    // decoded and re-encoded with the ANSI code page on Windows.
    return path_to_utf8(path_from_utf8(temp_directory()) / path_from_utf8(name));
}

}  // namespace alphashape::platform
