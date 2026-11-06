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

/// The fraction of physical memory a single task may plan to use. Planning
/// against 100% would ignore the OS, the rest of the process (the document, the
/// other assets) and whatever else the user is running.
inline constexpr double kTaskMemoryBudgetFraction = 0.60;

/// The byte budget a single task should stay under: the ALPHASHAPE_MEMORY_BUDGET_MB
/// override when it parses as a positive number, otherwise
/// kTaskMemoryBudgetFraction of total_physical_memory_bytes(). Returns 0 when
/// neither is available (no limit).
///
/// The override is read on every call, never cached: tests set and clear it, and
/// a cached value would leak between them.
std::uint64_t memory_budget_bytes();

/// A human-readable GB figure (one decimal) for a byte count, e.g. "7.9 GB".
/// Used in the "this machine has Y GB" diagnostics.
std::string format_gb(double bytes);

/// Peak resident set size of this process in bytes, or 0 when the platform
/// cannot report it. This is the process-wide high-water mark, not the peak of
/// one task: the job history records it as an upper bound on a run's memory.
std::uint64_t peak_resident_set_bytes();

}  // namespace alphashape::platform
