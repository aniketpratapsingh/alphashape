#include "base/platform/memory.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "base/platform/env.hpp"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <unistd.h>
#else
#include <unistd.h>
#endif

namespace alphashape::platform {
namespace {

constexpr double kBytesPerGb = 1024.0 * 1024.0 * 1024.0;

/// The ALPHASHAPE_MEMORY_BUDGET_MB override, or 0 when unset/unparseable.
std::uint64_t budget_override_bytes() {
    const std::string text = get_env("ALPHASHAPE_MEMORY_BUDGET_MB");
    if (text.empty()) {
        return 0;
    }
    // std::strtod rather than std::stod: a malformed value must read as "unset",
    // not throw out of a task's setup.
    char* end = nullptr;
    const double megabytes = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || !std::isfinite(megabytes) || !(megabytes > 0.0)) {
        return 0;
    }
    return static_cast<std::uint64_t>(megabytes * 1024.0 * 1024.0);
}

}  // namespace

std::uint64_t total_physical_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (::GlobalMemoryStatusEx(&status) == 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(status.ullTotalPhys);
#elif defined(__APPLE__)
    std::uint64_t bytes = 0;
    std::size_t length = sizeof(bytes);
    if (::sysctlbyname("hw.memsize", &bytes, &length, nullptr, 0) != 0) {
        return 0;
    }
    return bytes;
#else
    const long pages = ::sysconf(_SC_PHYS_PAGES);
    const long page_size = ::sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || page_size <= 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page_size);
#endif
}

std::uint64_t memory_budget_bytes() {
    const std::uint64_t override_bytes = budget_override_bytes();
    if (override_bytes != 0) {
        return override_bytes;
    }
    const std::uint64_t total = total_physical_memory_bytes();
    if (total == 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(static_cast<double>(total) * kTaskMemoryBudgetFraction);
}

std::string format_gb(double bytes) {
    const double gb = bytes / kBytesPerGb;
    // One decimal: "7.9 GB". Round to the nearest tenth first so a value like
    // 7.95 prints 8.0 rather than 7.9.
    const double tenths = std::round(gb * 10.0) / 10.0;
    std::string text = std::to_string(tenths);
    // std::to_string gives 6 decimals; trim to the one we want.
    const std::size_t dot = text.find('.');
    if (dot != std::string::npos) {
        text.resize(dot + 2);
    }
    return text + " GB";
}

}  // namespace alphashape::platform
