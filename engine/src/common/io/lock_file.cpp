#include "common/io/lock_file.hpp"

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
    if (!record.host.empty() && !local_host.empty() && record.host == local_host) {
        return !platform::process_is_alive(record.pid);
    }
    if (record.started <= 0) {
        return true;
    }
    return now - record.started > options.max_age_seconds;
}

/// Create `path` only when it does not exist. A hard link is atomic and fails
/// when the destination already exists on every filesystem that supports it; a
/// filesystem without hard links falls back to a checked rename.
bool create_exclusive(const std::string& path, const std::string& text, std::string* error) {
    const fs::path target = platform::path_from_utf8(path);
    fs::path temp = target;
    temp += ".tmp";
    temp += std::to_string(platform::process_id());
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) {
            *error = "cannot write '" + platform::path_to_utf8(temp) + "'";
            return false;
        }
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!out) {
            *error = "short write to '" + platform::path_to_utf8(temp) + "'";
            std::error_code ignored;
            fs::remove(temp, ignored);
            return false;
        }
    }
    std::error_code ec;
    fs::create_hard_link(temp, target, ec);
    if (!ec) {
        std::error_code ignored;
        fs::remove(temp, ignored);
        return true;
    }
    if (fs::exists(target)) {
        std::error_code ignored;
        fs::remove(temp, ignored);
        *error = "another writer created the lock first";
        return false;
    }
    // Hard links are unavailable here; the destination is absent, so a rename
    // is still a safe creation.
    fs::rename(temp, target, ec);
    if (ec) {
        std::error_code ignored;
        fs::remove(temp, ignored);
        *error = ec.message();
        return false;
    }
    return true;
}

}  // namespace

std::string ProjectLock::lock_path(const std::string& project_path) {
    const fs::path project = platform::path_from_utf8(project_path);
    const fs::path files_dir(platform::path_from_utf8(strip_extension(project) + ".files"));
    return platform::path_to_utf8(files_dir / "lock");
}

ProjectLock ProjectLock::acquire(const std::string& project_path,
                                 const LockOptions& options) {
    ProjectLock lock;
    lock.path_ = lock_path(project_path);
    const fs::path path = platform::path_from_utf8(lock.path_);
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    if (ec) {
        throw ProjectIOError("cannot create '" +
                             platform::path_to_utf8(path.parent_path()) + "': " + ec.message());
    }

    for (int attempt = 0; attempt < 2; ++attempt) {
        LockRecord existing;
        if (read_record(lock.path_, &existing)) {
            const std::string local = platform::hostname();
            if (!is_stale(existing, local, options, now_seconds())) {
                throw ProjectIOError("project '" + project_path +
                                     "' is locked by another writer (" + describe(existing) +
                                     "); remove '" + lock.path_ + "' only if that writer is gone");
            }
            // Stale: the holder is provably gone, so the lock can be replaced.
            std::error_code ignored;
            fs::remove(path, ignored);
        }

        lock.token_ = make_token();
        std::ostringstream text;
        text << "pid=" << platform::process_id() << '\n'
             << "host=" << platform::hostname() << '\n'
             << "started=" << now_seconds() << '\n'
             << "token=" << lock.token_ << '\n';
        std::string error;
        if (create_exclusive(lock.path_, text.str(), &error)) {
            return lock;
        }
        // A concurrent writer won the race; report who it is on the next pass.
        if (attempt == 0) {
            continue;
        }
        throw ProjectIOError("project '" + project_path +
                             "' is locked by another writer (" +
                             lock_holder(lock.path_) + "): " + error);
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
