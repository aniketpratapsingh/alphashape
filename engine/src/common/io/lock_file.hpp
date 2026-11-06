#pragma once

// Writer lock for a project container (WP-11.9).
//
// Opening a project for writing takes `<base>.files/lock`; a read-only open
// does not. The file records the writer's pid, host, start time and a random
// token, so a second writer can name the holder and a lock left behind by a
// crash can be recognised:
//
//   * same host, pid no longer running  -> stale, broken automatically;
//   * another host, older than max_age  -> stale, broken automatically;
//   * otherwise                         -> ProjectIOError naming the holder.
//
// The lock is opt-in: only explicit acquire() callers (the GUI's project open,
// tests) take it, so every existing CLI and library call behaves as before.

#include <string>

namespace alphashape {

struct LockOptions {
    /// Age after which a lock written on another host may be broken. A lock on
    /// this host is judged by its pid instead, however old it is.
    long long max_age_seconds = 24 * 60 * 60;
};

class ProjectLock {
public:
    ProjectLock() = default;
    ~ProjectLock();

    ProjectLock(const ProjectLock&) = delete;
    ProjectLock& operator=(const ProjectLock&) = delete;
    ProjectLock(ProjectLock&& other) noexcept;
    ProjectLock& operator=(ProjectLock&& other) noexcept;

    /// Take the writer lock for `project_path` (the .psx/.psz path, not the
    /// .files directory). Creates the .files directory when needed. Throws
    /// ProjectIOError when a live writer holds the lock.
    static ProjectLock acquire(const std::string& project_path,
                               const LockOptions& options = LockOptions());

    /// `<base>.files/lock` for a project path.
    static std::string lock_path(const std::string& project_path);

    /// Remove the lock file if this object still owns it. Idempotent.
    void release();

    bool held() const { return !path_.empty(); }
    const std::string& path() const { return path_; }

private:
    std::string path_;
    std::string token_;
};

/// The human-readable holder of an existing lock file, for error messages, or
/// "" when the file is absent or unreadable.
std::string lock_holder(const std::string& lock_path);

}  // namespace alphashape
