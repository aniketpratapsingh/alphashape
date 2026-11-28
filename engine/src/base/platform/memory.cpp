#include "base/platform/memory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "base/platform/env.hpp"

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <unistd.h>
#else
#include <sys/resource.h>
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

std::uint64_t available_physical_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (::GlobalMemoryStatusEx(&status) == 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(status.ullAvailPhys);
#elif defined(__APPLE__)
    // The VM statistics page counts are in the host page size. What the OS can
    // hand out without swapping is the free pages plus the file-backed pages
    // (the file cache, which the kernel drops first) plus purgeable pages.
    //
    // free_count already INCLUDES speculative pages: XNU's host_statistics64
    // reports them inside free_count, and the vm_stat tool prints "Pages free"
    // as free_count - speculative_count. Adding speculative_count on top would
    // double-count them.
    //
    // inactive_count is deliberately NOT counted: it holds anonymous pages that
    // can only be reclaimed by compressing or swapping them, so they are not
    // available without swapping. Compressed pages are likewise not counted,
    // because reclaiming them costs a decompression and is not "free" memory.
    //
    // This approximates the system monitor's "free + cached files" figure.
    ::vm_size_t page_size = 0;
    if (::host_page_size(::mach_host_self(), &page_size) != KERN_SUCCESS) {
        return 0;
    }
    ::mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    ::vm_statistics64_data_t stats{};
    if (::host_statistics64(::mach_host_self(), HOST_VM_INFO64,
                             reinterpret_cast<::host_info64_t>(&stats), &count) != KERN_SUCCESS) {
        return 0;
    }
    const std::uint64_t pages = static_cast<std::uint64_t>(stats.free_count) +
                                static_cast<std::uint64_t>(stats.external_page_count) +
                                static_cast<std::uint64_t>(stats.purgeable_count);
    // "Available" can never exceed the machine's total, so clamp rather than
    // report an impossible figure.
    const std::uint64_t bytes = pages * static_cast<std::uint64_t>(page_size);
    const std::uint64_t total = total_physical_memory_bytes();
    return total != 0 ? std::min(bytes, total) : bytes;
#else
    // MemAvailable is the kernel's own estimate of what a new allocation can
    // get without swapping; it is absent on very old kernels, in which case the
    // answer is "unknown" rather than a guess.
    std::FILE* file = std::fopen("/proc/meminfo", "r");
    if (file == nullptr) {
        return 0;
    }
    unsigned long long kilobytes = 0;
    bool found = false;
    char line[256];
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        if (std::sscanf(line, "MemAvailable: %llu kB", &kilobytes) == 1) {
            found = true;
            break;
        }
    }
    std::fclose(file);
    if (!found) {
        return 0;
    }
    return static_cast<std::uint64_t>(kilobytes) * 1024ULL;
#endif
}

std::uint64_t current_resident_set_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    if (::GetProcessMemoryInfo(::GetCurrentProcess(), &counters,
                               static_cast<DWORD>(sizeof(counters))) == 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(counters.WorkingSetSize);
#elif defined(__APPLE__)
    ::mach_task_basic_info_data_t info{};
    ::mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (::task_info(::mach_task_self(), MACH_TASK_BASIC_INFO,
                    reinterpret_cast<::task_info_t>(&info), &count) != KERN_SUCCESS) {
        return 0;
    }
    return static_cast<std::uint64_t>(info.resident_size);
#else
    // /proc/self/statm: "size resident shared text lib data dt" in pages. Only
    // the resident field is wanted; the virtual size is read and discarded
    // because an assignment-suppressing conversion cannot carry a length
    // modifier under -Wformat=.
    std::FILE* file = std::fopen("/proc/self/statm", "r");
    if (file == nullptr) {
        return 0;
    }
    unsigned long long virtual_pages = 0;
    unsigned long long resident_pages = 0;
    const int read = std::fscanf(file, "%llu %llu", &virtual_pages, &resident_pages);
    std::fclose(file);
    (void)virtual_pages;
    if (read != 2) {
        return 0;
    }
    const long page_size = ::sysconf(_SC_PAGE_SIZE);
    if (page_size <= 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(resident_pages) *
           static_cast<std::uint64_t>(page_size);
#endif
}

std::uint64_t memory_budget_floor_bytes(std::uint64_t total) {
    constexpr std::uint64_t kOneGib = 1024ULL * 1024ULL * 1024ULL;
    const std::uint64_t fraction_of_total =
        static_cast<std::uint64_t>(static_cast<double>(total) * kTaskMemoryBudgetFraction);
    return std::min(kOneGib, fraction_of_total);
}

std::uint64_t combine_memory_budget(std::uint64_t total, std::uint64_t available,
                                    std::uint64_t own_resident) {
    if (total == 0) {
        return 0;
    }
    const std::uint64_t fraction_of_total =
        static_cast<std::uint64_t>(static_cast<double>(total) * kTaskMemoryBudgetFraction);
    if (available == 0) {
        // Availability unknown: keep the old rule, a fraction of total.
        return fraction_of_total;
    }
    const std::uint64_t free_for_us = available + own_resident;
    const std::uint64_t fraction_of_free =
        static_cast<std::uint64_t>(static_cast<double>(free_for_us) *
                                   kAvailableMemoryBudgetFraction);
    const std::uint64_t budget = std::min(fraction_of_total, fraction_of_free);
    return std::max(budget, memory_budget_floor_bytes(total));
}

std::uint64_t memory_budget_bytes() {
    const std::uint64_t override_bytes = budget_override_bytes();
    if (override_bytes != 0) {
        return override_bytes;
    }
    return combine_memory_budget(total_physical_memory_bytes(),
                                 available_physical_memory_bytes(),
                                 current_resident_set_bytes());
}

std::string describe_memory_budget() {
    const std::uint64_t override_bytes = budget_override_bytes();
    if (override_bytes != 0) {
        return "memory budget " + format_gb(static_cast<double>(override_bytes)) +
               " (ALPHASHAPE_MEMORY_BUDGET_MB override)";
    }
    const std::uint64_t total = total_physical_memory_bytes();
    const std::uint64_t available = available_physical_memory_bytes();
    const std::uint64_t own = current_resident_set_bytes();
    const std::uint64_t budget = combine_memory_budget(total, available, own);
    if (total == 0) {
        return "memory budget unknown";
    }
    if (available == 0) {
        return "memory budget " + format_gb(static_cast<double>(budget)) + " of " +
               format_gb(static_cast<double>(total));
    }
    const std::uint64_t fraction_of_total =
        static_cast<std::uint64_t>(static_cast<double>(total) * kTaskMemoryBudgetFraction);
    const std::uint64_t fraction_of_free = static_cast<std::uint64_t>(
        static_cast<double>(available + own) * kAvailableMemoryBudgetFraction);
    if (fraction_of_free < fraction_of_total) {
        return "memory budget " + format_gb(static_cast<double>(budget)) +
               " (limited by available memory: " + format_gb(static_cast<double>(available)) +
               " free of " + format_gb(static_cast<double>(total)) + ")";
    }
    return "memory budget " + format_gb(static_cast<double>(budget)) + " of " +
           format_gb(static_cast<double>(total));
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

std::uint64_t peak_resident_set_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    if (::GetProcessMemoryInfo(::GetCurrentProcess(), &counters,
                               static_cast<DWORD>(sizeof(counters))) == 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
#else
    ::rusage usage{};
    if (::getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
#if defined(__APPLE__)
    // macOS reports ru_maxrss in bytes.
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    // Linux and the BSDs report ru_maxrss in kilobytes.
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
#endif
}

}  // namespace alphashape::platform
