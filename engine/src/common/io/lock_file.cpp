#include "common/io/lock_file.hpp"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>

#include "base/platform/env.hpp"
#include "base/platform/process.hpp"
#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"

namespace alphashape {

namespace fs = std::filesystem;

namespace {

struct LockRecord {
    long long pid = 0;
    std::string host;
    long long started = 0;
    std::string token;
};

std::int64_t now_seconds() {
    const auto since_epoch = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(since_epoch).count());
}

std::string strip_extension(const fs::path& path) {
    fs::path out = path;
    out.replace_extension();
    return platform::path_to_utf8(out);
}

std::string make_token() {
    std::random_device device;
    // One draw per statement (the engine's rng convention).
    const std::uint64_t high = static_cast<std::uint64_t>(device()) << 32U;
    const std::uint64_t low = static_cast<std::uint64_t>(device());
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << (high | low);
    return out.str();
}

LockRecord parse_record(const std::string& text) {
    LockRecord record;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t equals = line.find('=');
        if (equals == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, equals);
        const std::string value = line.substr(equals + 1);
        if (key == "pid") {
            try {
                record.pid = std::stoll(value);
            } catch (...) {
                record.pid = 0;
            }
        } else if (key == "started") {
            try {
                record.started = std::stoll(value);
            } catch (...) {
                record.started = 0;
            }
        } else if (key == "host") {
            record.host = value;
        } else if (key == "token") {
            record.token = value;
        }
    }
    return record;
}

bool read_record(const std::string& path, LockRecord* record) {
    std::ifstream in(platform::path_from_utf8(path), std::ios::binary);
    if (!in) {
        return false;
    }
    std::ostringstream text;
    text << in.rdbuf();
    *record = parse_record(text.str());
    return true;
}

std::string describe(const LockRecord& record) {
    std::string text = "pid " + std::to_string(record.pid);
    text += record.host.empty() ? " on an unknown host" : " on '" + record.host + "'";
    if (record.started > 0) {
        text += " since " + std::to_string(record.started);
    }
    return text;
}

/// True when the recorded holder is gone: a dead pid on this host, or a lock
/// from another host that is older than `max_age_seconds`.
bool is_stale(const LockRecord& record, const std::string& local_host,
              const LockOptions& options, std::int64_t now) {
    // A record without a token is an exclusive-open lock whose writer has not
    // finished writing yet; it is live, not stale.
    if (record.token.empty()) {
        return false;
    }
    if (!record.host.empty() && !local_host.empty() && record.host == local_host) {
        return !platform::process_is_alive(record.pid);
    }
    if (record.started <= 0) {
        return true;
    }
    return now - record.started > options.max_age_seconds;
}

enum class CreateStatus { Created, Exists, Failed };

bool is_read_only_errno(int value) {
#ifdef EROFS
    return value == EACCES || value == EPERM || value == EROFS;
#else
    return value == EACCES || value == EPERM;
#endif
}

/// Create `path` only when it does not exist. The primary path is a hard link:
/// atomic, and the record is complete when it appears. Without hard links the
/// fallback is an exclusive open, never a rename (rename replaces an existing
/// destination on POSIX, so it is not an exclusive create).
CreateStatus create_exclusive(const std::string& path, const std::string& text,
                              std::string* error, bool* read_only) {
    const fs::path target = platform::path_from_utf8(path);
    fs::path temp = target;
    temp += ".tmp.";
    // Token, not only the pid: two acquisitions in one process must not write
    // over each other's staging file.
    temp += make_token();
    {
        errno = 0;
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) {
            *error = "cannot write '" + platform::path_to_utf8(temp) + "'";
            *read_only = is_read_only_errno(errno);
            return CreateStatus::Failed;
        }
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!out) {
            *error = "short write to '" + platform::path_to_utf8(temp) + "'";
            std::error_code ignored;
            fs::remove(temp, ignored);
            return CreateStatus::Failed;
        }
    }
    std::error_code ec;
    fs::create_hard_link(temp, target, ec);
    if (!ec) {
        std::error_code ignored;
        fs::remove(temp, ignored);
        return CreateStatus::Created;
    }
    if (ec == std::errc::permission_denied) {
        *read_only = true;
    }
    if (fs::exists(target)) {
        std::error_code ignored;
        fs::remove(temp, ignored);
        *error = "another writer created the lock first";
        return CreateStatus::Exists;
    }
    std::string exclusive_error;
    const bool created = platform::create_file_exclusive(path, text, &exclusive_error);
    // Capture errno before any filesystem call below can overwrite it.
    const int exclusive_errno = errno;
    std::error_code ignored;
    fs::remove(temp, ignored);
    if (created) {
        return CreateStatus::Created;
    }
    if (fs::exists(target)) {
        *error = "another writer created the lock first";
        return CreateStatus::Exists;
    }
    *error = exclusive_error;
    *read_only = *read_only || is_read_only_errno(exclusive_errno);
    return CreateStatus::Failed;
}

bool same_record(const LockRecord& a, const LockRecord& b) {
    return a.pid == b.pid && a.host == b.host && a.token == b.token;
}

enum class BreakStatus { Broken, Vanished, Changed };

/// Break a stale lock without ever deleting a record that another writer
/// installed in the meantime: rename it aside, re-read the sideline and compare
/// it with the record that was judged stale. A different record is put back
/// when the path is still free, and is otherwise reported as the live holder.
BreakStatus break_stale_lock(const std::string& lock_path, const LockRecord& judged,
                             std::string* holder) {
    const fs::path path = platform::path_from_utf8(lock_path);
    fs::path sideline = path;
    sideline += ".stale.";
    sideline += make_token();
    std::error_code ec;
    fs::rename(path, sideline, ec);
    if (ec) {
        return BreakStatus::Vanished;
    }
    LockRecord moved;
    if (!read_record(platform::path_to_utf8(sideline), &moved)) {
        std::error_code ignored;
        fs::remove(sideline, ignored);
        return BreakStatus::Vanished;
    }
    if (same_record(moved, judged)) {
        std::error_code ignored;
        fs::remove(sideline, ignored);
        return BreakStatus::Broken;
    }
    std::error_code exists_ec;
    if (!fs::exists(path, exists_ec)) {
        std::error_code put_back;
        fs::rename(sideline, path, put_back);
        if (!put_back) {
            *holder = describe(moved);
            return BreakStatus::Changed;
        }
    }
    std::error_code ignored;
    fs::remove(sideline, ignored);
    *holder = describe(moved);
    return BreakStatus::Changed;
}

}  // namespace

std::string ProjectLock::lock_path(const std::string& project_path) {
    const fs::path project = platform::path_from_utf8(project_path);
    fs::path lock;
    if (project.extension() == ".psz") {
        // A single-file project owns no .files directory, so its lock sits
        // beside it as "<path>.lock".
        lock = project;
        lock += ".lock";
    } else {
        const fs::path files_dir(
            platform::path_from_utf8(strip_extension(project) + ".files"));
        lock = files_dir / "lock";
    }
    lock.make_preferred();
    return platform::path_to_utf8(lock);
}

ProjectLock ProjectLock::acquire(const std::string& project_path,
                                 const LockOptions& options) {
    return acquire_impl(project_path, options, nullptr);
}

ProjectLock ProjectLock::acquire_assuming_stale(const std::string& project_path,
                                                const std::string& stale_record,
                                                const LockOptions& options) {
    return acquire_impl(project_path, options, &stale_record);
}

ProjectLock ProjectLock::acquire_impl(const std::string& project_path,
                                      const LockOptions& options,
                                      const std::string* assumed_stale) {
    ProjectLock lock;
    lock.path_ = lock_path(project_path);
    const fs::path path = platform::path_from_utf8(lock.path_);
    const fs::path parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        fs::create_directories(parent, ec);
        if (ec) {
            const bool read_only = ec == std::errc::permission_denied ||
                                   ec == std::errc::read_only_file_system;
            const std::string message = "cannot write lock in '" +
                                        platform::path_to_utf8(parent) + "': " + ec.message();
            if (read_only) {
                throw ProjectReadOnlyError(message + "; the project is read-only");
            }
            throw ProjectIOError(message);
        }
    }

    const std::string local = platform::hostname();
    for (int attempt = 0; attempt < 4; ++attempt) {
        LockRecord existing;
        bool have_record = false;
        if (attempt == 0 && assumed_stale != nullptr) {
            existing = parse_record(*assumed_stale);
            have_record = true;
        } else {
            have_record = read_record(lock.path_, &existing);
        }
        if (have_record) {
            if (!is_stale(existing, local, options, now_seconds())) {
                throw ProjectIOError("project '" + project_path +
                                     "' is locked by another writer (" + describe(existing) +
                                     "); remove '" + lock.path_ +
                                     "' only if that writer is gone");
            }
            std::string holder;
            const BreakStatus broken = break_stale_lock(lock.path_, existing, &holder);
            if (broken == BreakStatus::Changed) {
                if (holder.empty()) {
                    holder = lock_holder(lock.path_);
                }
                throw ProjectIOError("project '" + project_path +
                                     "' is locked by another writer (" + holder + ")");
            }
            if (broken == BreakStatus::Vanished) {
                continue;
            }
        }
        lock.token_ = make_token();
        std::ostringstream text;
        text << "pid=" << platform::process_id() << '\n'
             << "host=" << local << '\n'
             << "started=" << now_seconds() << '\n'
             << "token=" << lock.token_ << '\n';
        std::string error;
        bool read_only = false;
        const CreateStatus status =
            create_exclusive(lock.path_, text.str(), &error, &read_only);
        if (status == CreateStatus::Created) {
            return lock;
        }
        if (status == CreateStatus::Failed) {
            if (read_only) {
                throw ProjectReadOnlyError("cannot write lock in '" +
                                           platform::path_to_utf8(parent) + "': " + error +
                                           "; the project is read-only");
            }
            throw ProjectIOError("cannot write lock '" + lock.path_ + "': " + error);
        }
        // Exists: another writer won the race; re-read who it is on the next
        // pass, but never remove that record.
    }
    throw ProjectIOError("project '" + project_path + "' could not be locked");
}

void ProjectLock::release() {
    if (path_.empty()) {
        return;
    }
    // Only remove the lock this object owns: a stale-break and re-acquire by
    // another process must not have its file deleted from under it.
    LockRecord record;
    if (read_record(path_, &record) && record.token == token_) {
        std::error_code ignored;
        fs::remove(platform::path_from_utf8(path_), ignored);
    }
    path_.clear();
    token_.clear();
}

ProjectLock::~ProjectLock() {
    release();
}

ProjectLock::ProjectLock(ProjectLock&& other) noexcept
    : path_(std::move(other.path_)), token_(std::move(other.token_)) {
    other.path_.clear();
    other.token_.clear();
}

ProjectLock& ProjectLock::operator=(ProjectLock&& other) noexcept {
    if (this != &other) {
        release();
        path_ = std::move(other.path_);
        token_ = std::move(other.token_);
        other.path_.clear();
        other.token_.clear();
    }
    return *this;
}

std::string lock_holder(const std::string& lock_path) {
    LockRecord record;
    if (!read_record(lock_path, &record)) {
        return std::string();
    }
    return describe(record);
}

}  // namespace alphashape
