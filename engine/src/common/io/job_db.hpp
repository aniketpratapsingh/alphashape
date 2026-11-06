#pragma once

// Job database (M11 §4.9, WP-11.10).
//
// One table: jobs(id INTEGER PRIMARY KEY, kind TEXT, payload TEXT, state TEXT,
// created TEXT, updated TEXT). Timestamps are ISO-8601 UTC. Every operation is
// serialised by a mutex, so one JobDb may be shared across threads.
//
// SQLite is optional at configure time. Without it JobDb is a JSON-lines
// append-only log replayed on open; ids stay monotonic across reopen because
// removed rows remain in the log. JobDbFallback is always compiled so the
// fallback is tested directly on every lane, whether or not SQLite was found.

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace alphashape {

struct JobRecord {
    std::int64_t id = 0;
    std::string kind;
    std::string payload;
    std::string state;
    std::string created;
    std::string updated;
};

/// The JSON-lines backend. Exposed so tests exercise it even on a build that
/// selected SQLite for JobDb. Writes are flushed and fsynced.
class JobDbFallback {
public:
    explicit JobDbFallback(std::string path);
    JobDbFallback(const JobDbFallback&) = delete;
    JobDbFallback& operator=(const JobDbFallback&) = delete;
    JobDbFallback(JobDbFallback&&) = delete;
    JobDbFallback& operator=(JobDbFallback&&) = delete;

    std::int64_t insert_job(const std::string& kind, const std::string& payload);
    void set_state(std::int64_t id, const std::string& state);
    std::vector<JobRecord> list_jobs(
        const std::optional<std::string>& state = std::nullopt) const;
    bool remove_job(std::int64_t id);

private:
    void replay();

    mutable std::mutex mutex_;
    std::string path_;
    std::vector<JobRecord> records_;
    std::int64_t next_id_ = 1;
};

/// The configured job database: SQLite when found at configure time, the
/// JSON-lines fallback otherwise. Same API either way.
class JobDb {
public:
    /// Opens (or creates) the database at `path`. Throws Error when the parent
    /// directory or the file cannot be written.
    static JobDb open(std::string path);

    JobDb(JobDb&&) noexcept;
    JobDb& operator=(JobDb&&) noexcept;
    JobDb(const JobDb&) = delete;
    JobDb& operator=(const JobDb&) = delete;
    ~JobDb();

    std::int64_t insert_job(const std::string& kind, const std::string& payload);
    void set_state(std::int64_t id, const std::string& state);
    std::vector<JobRecord> list_jobs(
        const std::optional<std::string>& state = std::nullopt) const;
    bool remove_job(std::int64_t id);

    /// "sqlite3" when built with SQLite, "jsonl" for the fallback.
    static std::string backend();

    /// Where job.db lives for a project, following the lock-file rules: beside
    /// the lock inside `<base>.files/` for a directory project, and
    /// `<x>.psz.jobs.db` beside a single-file `.psz` (never a `x.files/`).
    static std::string path_for_project(const std::string& project_path);

private:
    struct Impl;
    explicit JobDb(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

}  // namespace alphashape
