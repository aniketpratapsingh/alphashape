#pragma once

// Job estimates and measured throughput history.
//
// Estimate is the engine-independent answer the task dialog shows:
// seconds, a relative sigma, peak RAM, product and disk bytes, and whether a
// measured rate or the cold model produced it (SP 4.8).
//
// ThroughputHistory holds the last 20 successful runs per task per project.
// Once a task has three rows, a robust median of its measured seconds per
// camera-megapixel replaces the cold constants and the estimate is marked
// confident; sigma comes from the inter-quartile range of those rates, floored
// at 10 %. Rows survive a restart through ThroughputStore, whose file
// implementation writes `<project>.files/job_history.tsv` atomically (and
// `<x>.psz.history.tsv` beside a .psz project, which has no .files folder).
//
// A run's peak RSS is only recorded when it raised the process-wide high-water
// mark; the median of three such peaks then replaces the cold RAM model.
//
// No Qt types in this header.

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common/tasks/params.hpp"
#include "common/tasks/registry.hpp"

namespace alphashape {
class Chunk;
}

namespace alphashape::gui {

/// One completed run, as the history keeps it.
struct ThroughputRow {
    std::string task;
    double cameras = 0.0;
    double megapixels = 0.0;  ///< working-resolution megapixels per camera
    double seconds = 0.0;
    /// Process-wide peak-RSS high-water mark at the moment the run ended, but
    /// only when this run raised it (`after > before`). 0 means "unknown": the
    /// counter is shared by the whole process, so a run that stayed below an
    /// earlier high-water mark says nothing about its own footprint.
    std::uint64_t peak_rss_bytes = 0;
};

/// Peak RSS a run may claim: `after` only when it raised the process-wide
/// high-water mark, otherwise 0 ("unknown"). Pure so the rule is testable.
std::uint64_t recorded_peak_rss(std::uint64_t before, std::uint64_t after);

/// Median of the nonzero `peak_rss_bytes` in `rows`, scaled by the ratio of
/// `current_units` (cameras * megapixels) to the median units of the rows that
/// recorded a peak. Returns `fallback` when fewer than three rows recorded one
/// (the cold model is more trustworthy than two samples).
std::uint64_t measured_ram_from_history(const std::vector<ThroughputRow>& rows,
                                        double current_units, std::uint64_t fallback);

/// The plain numbers the estimator needs from a chunk. Copied under the
/// document lock so a modal dialog never keeps a `Chunk*` (and the lock) alive
/// while it waits for the user. Nothing here borrows from the chunk.
struct ChunkFacts {
    struct CameraFacts {
        bool enabled = false;
        bool has_frame = false;
        bool has_keypoints = false;
        /// Sensor width * height, or the linked frame's size when the sensor
        /// has no dimensions.
        double pixels = 0.0;
    };
    std::vector<CameraFacts> cameras;
    /// Total width * height over the active depth-map set (0 when none).
    double depth_map_pixels = 0.0;
    double depth_map_count = 0.0;
    /// Active elevation raster cells, 0 when the chunk has no DEM.
    std::uint64_t dem_cells = 0;
    /// Active dense point-cloud point count, 0 when the chunk has none.
    std::uint64_t point_count = 0;
};

/// Snapshot the estimator inputs from `chunk`. Call this inside read_document().
ChunkFacts chunk_facts(const Chunk& chunk);

/// The camera count estimate_task uses: every camera when none is enabled yet
/// (before the first alignment), otherwise the enabled ones. The task dialog's
/// RAM warning uses the same rule so it never says "0-image block".
double estimate_camera_count(const std::string& task, const ChunkFacts& facts);

/// Bounded per-task history: at most the newest 20 rows for each task.
class ThroughputHistory {
public:
    static constexpr std::size_t kMaxRowsPerTask = 20;

    /// Append a row, dropping the oldest row of its task when the cap is hit.
    /// Rows with a task name that is empty, or non-finite numbers, are ignored.
    void add(ThroughputRow row);

    /// Rows for one task, oldest first. Null when the task has none.
    const std::vector<ThroughputRow>* rows(const std::string& task) const;

    /// Every retained row, in task-name then insertion order.
    std::vector<ThroughputRow> all() const;

    std::size_t size() const;
    bool empty() const { return size() == 0; }
    void clear() { by_task_.clear(); }

private:
    std::map<std::string, std::vector<ThroughputRow>> by_task_;
};

/// Persistence boundary. A later job.db can implement the same two calls.
class ThroughputStore {
public:
    virtual ~ThroughputStore() = default;

    /// Read the retained history. A missing or unreadable file is an empty
    /// history, never an exception.
    virtual ThroughputHistory load() = 0;

    /// Append one row. Returns false when the row could not be persisted; the
    /// in-memory history is the caller's concern.
    virtual bool append(const ThroughputRow& row) = 0;
};

/// `<base>.files/job_history.tsv` for a project path, where `base` is the path
/// with its final extension removed (the ProjectIO side-car folder). A `.psz`
/// project has no such folder, so its history is `<path>.history.tsv` beside
/// the manifest instead. Empty for no path.
std::string history_path_for_project(const std::string& project_path);

/// Atomic TSV implementation: load parses skip-malformed lines, append rewrites
/// the retained history through a temporary file and renames it into place.
/// The temporary is removed when the write or rename fails.
class FileThroughputStore final : public ThroughputStore {
public:
    explicit FileThroughputStore(std::string path);

    const std::string& path() const { return path_; }
    ThroughputHistory load() override;
    bool append(const ThroughputRow& row) override;

private:
    bool write(const ThroughputHistory& history);

    std::string path_;
};

/// The estimate row's content, exactly SP 4.8. `basis` is either
/// "measured (last N runs)", "cold model" or "cold model (generic)".
struct Estimate {
    double seconds = 0.0;
    double seconds_sigma_rel = 0.35;
    std::uint64_t peak_ram_bytes = 0;
    std::uint64_t disk_bytes = 0;
    std::uint64_t output_bytes = 0;
    bool confident = false;  ///< true when a measured rate was used
    std::string basis;       ///< "measured (last 3 runs)" | "cold model"
};

/// Estimate one run. Never throws: an unknown task, an empty chunk or an
/// unavailable history all fall back to the generic cold model. Takes O(rows)
/// in the task's history (bounded at 20) and is expected well under 5 ms.
Estimate estimate_task(const TaskRegistry& registry, const std::string& task,
                       const Params& params, const ChunkFacts& facts,
                       const ThroughputHistory& history);

/// Convenience overload for callers that already hold the chunk under the lock.
Estimate estimate_task(const TaskRegistry& registry, const std::string& task,
                       const Params& params, const Chunk& chunk,
                       const ThroughputHistory& history);

/// The history row a finished run contributes. Cameras and megapixels use the
/// same working-resolution rule as estimate_task, so a later estimate compares
/// like with like.
ThroughputRow throughput_row_for(const std::string& task, const Params& params,
                                 const Chunk& chunk, double seconds,
                                 std::uint64_t peak_rss_bytes);

/// The RAM budget the warning compares against: memory_budget_bytes() when it
/// is nonzero, otherwise total_physical_memory_bytes(). Exposed so the dialog
/// and the tests use the same rule.
std::uint64_t estimate_memory_budget();

/// True when the estimate should show the "same settings on an N-image block"
/// warning (peak RAM strictly above 75 % of `budget`). A zero budget disables
/// the warning because there is nothing to compare against.
bool needs_ram_warning(const Estimate& estimate, std::uint64_t budget);

}  // namespace alphashape::gui
