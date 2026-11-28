#pragma once

// Physical memory introspection, so a task can refuse an allocation that cannot
// possibly fit before it turns into an out-of-memory kill or a bare
// std::bad_alloc. This is the only place the engine asks the OS how much RAM the
// machine has.

#include <cstdint>
#include <string>

namespace alphashape::platform {

/// Total physical memory in bytes, or 0 when the platform cannot report it.
std::uint64_t total_physical_memory_bytes();

/// Physical memory the OS could hand this process right now without swapping,
/// or 0 when the platform cannot report it. This is the machine's free memory,
/// not this process's: it already excludes what other programs hold.
///
/// Linux reads MemAvailable from /proc/meminfo; macOS sums the free,
/// file-backed and purgeable page counts (free already includes speculative
/// pages, and inactive anonymous pages are not reclaimable without
/// compression/swap, so neither is added; compressed pages are NOT counted as
/// available); Windows reads ullAvailPhys from GlobalMemoryStatusEx.
std::uint64_t available_physical_memory_bytes();

/// Resident set size of this process in bytes, or 0 when the platform cannot
/// report it. This is the memory the process holds NOW, as opposed to
/// peak_resident_set_bytes()'s high-water mark.
std::uint64_t current_resident_set_bytes();

/// The fraction of physical memory a single task may plan to use. Planning
/// against 100% would ignore the OS, the rest of the process (the document, the
/// other assets) and whatever else the user is running.
inline constexpr double kTaskMemoryBudgetFraction = 0.60;

/// The fraction of (available + this process's resident) memory a single task
/// may plan to use when the machine is busy. The process's own resident memory
/// is added back because memory this process already holds (the loaded
/// document, an earlier stage's buffers) is not "used by others", so it is
/// still available to the task.
inline constexpr double kAvailableMemoryBudgetFraction = 0.85;

/// The smallest budget a task is ever given, so a momentarily busy machine does
/// not produce an absurd plan: min(1 GiB, kTaskMemoryBudgetFraction x total).
/// A task that cannot fit even this is refused with an explanation rather than
/// silently planned at a useless scale.
std::uint64_t memory_budget_floor_bytes(std::uint64_t total);

/// The pure combination rule behind memory_budget_bytes(), exposed so it can be
/// unit-tested without touching the machine:
///
///   * total == 0            -> 0 (nothing known, no limit);
///   * available == 0        -> kTaskMemoryBudgetFraction x total (unknown
///                              availability keeps the old behaviour);
///   * otherwise             -> min(kTaskMemoryBudgetFraction x total,
///                                 kAvailableMemoryBudgetFraction x
///                                 (available + own_resident)),
///                              raised to memory_budget_floor_bytes(total).
std::uint64_t combine_memory_budget(std::uint64_t total, std::uint64_t available,
                                    std::uint64_t own_resident);

/// The byte budget a single task should stay under: the ALPHASHAPE_MEMORY_BUDGET_MB
/// override when it parses as a positive number, otherwise
/// combine_memory_budget(total_physical_memory_bytes(),
/// available_physical_memory_bytes(), current_resident_set_bytes()). Returns 0
/// when nothing is known (no limit).
///
/// The override is read on every call, never cached: tests set and clear it, and
/// a cached value would leak between them.
std::uint64_t memory_budget_bytes();

/// A one-line explanation of the current budget for a log or diagnostic, e.g.
/// "memory budget 4.1 GB (limited by available memory: 4.8 GB free of 16.0 GB)"
/// or "memory budget 9.6 GB of 16.0 GB". Names the override when it is set.
std::string describe_memory_budget();

/// A human-readable GB figure (one decimal) for a byte count, e.g. "7.9 GB".
/// Used in the "this machine has Y GB" diagnostics.
std::string format_gb(double bytes);

/// Peak resident set size of this process in bytes, or 0 when the platform
/// cannot report it. This is the process-wide high-water mark, not the peak of
/// one task: the job history records it as an upper bound on a run's memory.
std::uint64_t peak_resident_set_bytes();

}  // namespace alphashape::platform
