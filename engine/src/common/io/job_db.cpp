#include "common/io/job_db.hpp"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "base/platform/env.hpp"
#include "base/utils/error.hpp"

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#if defined(ALPHASHAPE_USE_SQLITE3)
#include <sqlite3.h>
#endif

namespace alphashape {

namespace {

namespace fs = std::filesystem;

/// ISO-8601 UTC, second precision: "2026-11-06T12:34:56Z".
std::string iso8601_utc_now() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#if defined(_WIN32)
    (void)::gmtime_s(&utc, &now);
#else
    (void)::gmtime_r(&now, &utc);
#endif
    char text[32];
    const std::size_t written =
        std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return std::string(text, written);
}

/// Quoted JSON string; control characters and the two escapes JSON needs.
std::string json_quote(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 2);
    out.push_back('"');
    for (const char raw : text) {
        const unsigned char c = static_cast<unsigned char>(raw);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20U) {
                    char escaped[7];
                    std::snprintf(escaped, sizeof(escaped), "\\u%04x",
                                  static_cast<unsigned>(c));
                    out += escaped;
                } else {
                    out.push_back(raw);
                }
        }
    }
    out.push_back('"');
    return out;
}

/// Decodes a JSON string starting at `start` (the opening quote).
std::string json_string_at(const std::string& line, std::size_t start) {
    if (start >= line.size() || line[start] != '"') {
        return std::string();
    }
    std::string out;
    for (std::size_t i = start + 1; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            break;
        }
        if (c != '\\') {
            out.push_back(c);
            continue;
        }
        if (++i >= line.size()) {
            break;
        }
        switch (line[i]) {
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if (i + 4 >= line.size()) {
                    return out;
                }
                unsigned value = 0;
                for (int digit = 0; digit < 4; ++digit) {
                    const char hex = line[i + 1 + static_cast<std::size_t>(digit)];
                    value <<= 4U;
                    if (hex >= '0' && hex <= '9') {
                        value |= static_cast<unsigned>(hex - '0');
                    } else if (hex >= 'a' && hex <= 'f') {
                        value |= static_cast<unsigned>(hex - 'a' + 10);
                    } else if (hex >= 'A' && hex <= 'F') {
                        value |= static_cast<unsigned>(hex - 'A' + 10);
                    }
                }
                i += 4;
                out.push_back(static_cast<char>(value & 0xFFU));
                break;
            }
            default: out.push_back(line[i]); break;
        }
    }
    return out;
}

/// A flat JSON object as key -> value text. Strings are decoded; numbers keep
/// their spelling, so the two getters below share one parser and a key that
/// appears inside a payload cannot be mistaken for a top-level key.
std::map<std::string, std::string> parse_object(const std::string& line) {
    std::map<std::string, std::string> values;
    std::size_t i = line.find('{');
    if (i == std::string::npos) {
        return values;
    }
    ++i;
    for (;;) {
        while (i < line.size() &&
               (line[i] == ' ' || line[i] == '\t' || line[i] == ',')) {
            ++i;
        }
        if (i >= line.size() || line[i] == '}') {
            break;
        }
        if (line[i] != '"') {
            break;
        }
        const std::size_t key_start = i;
        ++i;
        while (i < line.size() && line[i] != '"') {
            const std::size_t step = line[i] == '\\' ? std::size_t{2} : std::size_t{1};
            i += step;
        }
        if (i >= line.size()) {
            break;
        }
        const std::string key = json_string_at(line, key_start);
        ++i;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
            ++i;
        }
        if (i >= line.size() || line[i] != ':') {
            break;
        }
        ++i;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
            ++i;
        }
        if (i < line.size() && line[i] == '"') {
            values[key] = json_string_at(line, i);
            ++i;
            while (i < line.size() && line[i] != '"') {
                const std::size_t step =
                    line[i] == '\\' ? std::size_t{2} : std::size_t{1};
                i += step;
            }
            if (i < line.size()) {
                ++i;
            }
        } else {
            const std::size_t start = i;
            while (i < line.size() && line[i] != ',' && line[i] != '}') {
                ++i;
            }
            std::size_t end = i;
            while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t')) {
                --end;
            }
            values[key] = line.substr(start, end - start);
        }
    }
    return values;
}

std::string json_get_string(const std::string& line, const std::string& key) {
    const std::map<std::string, std::string> values = parse_object(line);
    const auto found = values.find(key);
    return found == values.end() ? std::string() : found->second;
}

std::int64_t json_get_int(const std::string& line, const std::string& key) {
    const std::map<std::string, std::string> values = parse_object(line);
    const auto found = values.find(key);
    if (found == values.end()) {
        return 0;
    }
    std::int64_t value = 0;
    bool negative = false;
    const std::string& text = found->second;
    std::size_t i = 0;
    if (i < text.size() && text[i] == '-') {
        negative = true;
        ++i;
    }
    for (; i < text.size() && text[i] >= '0' && text[i] <= '9'; ++i) {
        value = value * 10 + static_cast<std::int64_t>(text[i] - '0');
    }
    return negative ? -value : value;
}

/// Appends one line and fsyncs it before returning.
void append_synced(const std::string& path, const std::string& line) {
#if defined(_WIN32)
    const std::wstring wide = platform::path_from_utf8(path).wstring();
    std::FILE* file = ::_wfopen(wide.c_str(), L"ab");
#else
    std::FILE* file = std::fopen(path.c_str(), "ab");
#endif
    if (file == nullptr) {
        throw Error("job.db: cannot open '" + path + "' for append");
    }
    const std::string record = line + "\n";
    const std::size_t written =
        std::fwrite(record.data(), std::size_t{1}, record.size(), file);
    const bool ok = written == record.size() && std::fflush(file) == 0;
#if defined(_WIN32)
    const bool synced = ok && ::_commit(::_fileno(file)) == 0;
#else
    const bool synced = ok && ::fsync(::fileno(file)) == 0;
#endif
    std::fclose(file);
    if (!synced) {
        throw Error("job.db: cannot write '" + path + "'");
    }
}

}  // namespace

JobDbFallback::JobDbFallback(std::string path) : path_(std::move(path)) {
    replay();
}

void JobDbFallback::replay() {
    std::ifstream in(platform::path_from_utf8(path_), std::ios::binary);
    if (!in) {
        return;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::string op = json_get_string(line, "op");
        const std::int64_t id = json_get_int(line, "id");
        if (op == "insert") {
            JobRecord record;
            record.id = id;
            record.kind = json_get_string(line, "kind");
            record.payload = json_get_string(line, "payload");
            record.state = json_get_string(line, "state");
            record.created = json_get_string(line, "created");
            record.updated = json_get_string(line, "updated");
            records_.push_back(std::move(record));
            if (id >= next_id_) {
                next_id_ = id + 1;
            }
        } else if (op == "update") {
            for (JobRecord& record : records_) {
                if (record.id == id) {
                    record.state = json_get_string(line, "state");
                    record.updated = json_get_string(line, "updated");
                    break;
                }
            }
        } else if (op == "remove") {
            for (std::size_t i = 0; i < records_.size(); ++i) {
                if (records_[i].id == id) {
                    records_.erase(records_.begin() + static_cast<std::ptrdiff_t>(i));
                    break;
                }
            }
        }
    }
}

std::int64_t JobDbFallback::insert_job(const std::string& kind,
                                       const std::string& payload) {
    std::lock_guard<std::mutex> lock(mutex_);
    JobRecord record;
    record.id = next_id_;
    record.kind = kind;
    record.payload = payload;
    record.state = "pending";
    record.created = iso8601_utc_now();
    record.updated = record.created;

    std::ostringstream line;
    line << "{\"op\":\"insert\",\"id\":" << record.id << ",\"kind\":"
         << json_quote(record.kind) << ",\"payload\":" << json_quote(record.payload)
         << ",\"state\":" << json_quote(record.state) << ",\"created\":"
         << json_quote(record.created) << ",\"updated\":" << json_quote(record.updated)
         << '}';
    append_synced(path_, line.str());
    records_.push_back(std::move(record));
    return next_id_++;
}

void JobDbFallback::set_state(std::int64_t id, const std::string& state) {
    std::lock_guard<std::mutex> lock(mutex_);
    JobRecord* found = nullptr;
    for (JobRecord& record : records_) {
        if (record.id == id) {
            found = &record;
            break;
        }
    }
    if (found == nullptr) {
        throw Error("job.db: no job with id " + std::to_string(id));
    }
    const std::string updated = iso8601_utc_now();
    std::ostringstream line;
    line << "{\"op\":\"update\",\"id\":" << id << ",\"state\":" << json_quote(state)
         << ",\"updated\":" << json_quote(updated) << '}';
    append_synced(path_, line.str());
    found->state = state;
    found->updated = updated;
}

std::vector<JobRecord> JobDbFallback::list_jobs(
    const std::optional<std::string>& state) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<JobRecord> out;
    for (const JobRecord& record : records_) {
        if (!state.has_value() || record.state == *state) {
            out.push_back(record);
        }
    }
    return out;
}

bool JobDbFallback::remove_job(std::int64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].id != id) {
            continue;
        }
        const std::string line =
            "{\"op\":\"remove\",\"id\":" + std::to_string(id) + "}";
        append_synced(path_, line);
        records_.erase(records_.begin() + static_cast<std::ptrdiff_t>(i));
        return true;
    }
    return false;
}

#if defined(ALPHASHAPE_USE_SQLITE3)
namespace {

/// A prepared statement that finalises itself.
class Statement {
public:
    Statement(sqlite3* db, const char* sql) {
        if (::sqlite3_prepare_v2(db, sql, -1, &stmt_, nullptr) != SQLITE_OK) {
            throw Error("job.db: prepare failed: " + std::string(::sqlite3_errmsg(db)));
        }
    }
    ~Statement() { ::sqlite3_finalize(stmt_); }
    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;

    sqlite3_stmt* get() const { return stmt_; }

    void bind_text(int index, const std::string& text) {
        if (::sqlite3_bind_text(stmt_, index, text.data(),
                                static_cast<int>(text.size()),
                                SQLITE_TRANSIENT) != SQLITE_OK) {
            throw Error("job.db: bind failed");
        }
    }
    void bind_int(int index, std::int64_t value) {
        if (::sqlite3_bind_int64(stmt_, index, value) != SQLITE_OK) {
            throw Error("job.db: bind failed");
        }
    }
    void step_done(sqlite3* db) {
        if (::sqlite3_step(stmt_) != SQLITE_DONE) {
            throw Error("job.db: write failed: " + std::string(::sqlite3_errmsg(db)));
        }
    }

private:
    sqlite3_stmt* stmt_ = nullptr;
};

std::string column_text(sqlite3_stmt* stmt, int column) {
    const unsigned char* text = ::sqlite3_column_text(stmt, column);
    if (text == nullptr) {
        return std::string();
    }
    const int bytes = ::sqlite3_column_bytes(stmt, column);
    return std::string(reinterpret_cast<const char*>(text),
                       static_cast<std::size_t>(bytes));
}

}  // namespace
#endif

struct JobDb::Impl {
    explicit Impl(std::string database_path) : path(std::move(database_path)) {
#if defined(ALPHASHAPE_USE_SQLITE3)
        if (::sqlite3_open_v2(path.c_str(), &db,
                              SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                              nullptr) != SQLITE_OK) {
            const std::string reason =
                db != nullptr ? ::sqlite3_errmsg(db) : "cannot open database";
            if (db != nullptr) {
                ::sqlite3_close(db);
                db = nullptr;
            }
            throw Error("job.db: " + reason + " ('" + path + "')");
        }
        (void)::sqlite3_busy_timeout(db, 5000);
        char* message = nullptr;
        if (::sqlite3_exec(db, "PRAGMA journal_mode=WAL", nullptr, nullptr,
                           &message) != SQLITE_OK) {
            const std::string reason = message != nullptr ? message : "WAL failed";
            ::sqlite3_free(message);
            throw Error("job.db: " + reason);
        }
        if (::sqlite3_exec(db,
                           "CREATE TABLE IF NOT EXISTS jobs("
                           "id INTEGER PRIMARY KEY, kind TEXT, payload TEXT, "
                           "state TEXT, created TEXT, updated TEXT)",
                           nullptr, nullptr, &message) != SQLITE_OK) {
            const std::string reason = message != nullptr ? message : "schema failed";
            ::sqlite3_free(message);
            throw Error("job.db: " + reason);
        }
#else
        fallback = std::make_unique<JobDbFallback>(path);
#endif
    }

#if defined(ALPHASHAPE_USE_SQLITE3)
    ~Impl() {
        if (db != nullptr) {
            ::sqlite3_close(db);
        }
    }
#endif

    std::string path;
    mutable std::mutex mutex;
#if defined(ALPHASHAPE_USE_SQLITE3)
    sqlite3* db = nullptr;
#else
    std::unique_ptr<JobDbFallback> fallback;
#endif
};

JobDb::JobDb(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
JobDb::~JobDb() = default;
JobDb::JobDb(JobDb&&) noexcept = default;
JobDb& JobDb::operator=(JobDb&&) noexcept = default;

JobDb JobDb::open(std::string path) {
    const fs::path target = platform::path_from_utf8(path);
    const fs::path parent = target.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        fs::create_directories(parent, ec);
        if (ec) {
            throw Error("job.db: cannot create '" + platform::path_to_utf8(parent) +
                        "': " + ec.message());
        }
    }
    return JobDb(std::make_unique<Impl>(std::move(path)));
}

std::string JobDb::backend() {
#if defined(ALPHASHAPE_USE_SQLITE3)
    return "sqlite3";
#else
    return "jsonl";
#endif
}

std::string JobDb::path_for_project(const std::string& project_path) {
    const fs::path project = platform::path_from_utf8(project_path);
    fs::path database;
    if (project.extension() == ".psz") {
        // A .psz owns no .files directory, so its jobs database sits beside it.
        database = project;
        database += ".jobs.db";
    } else {
        fs::path base = project;
        base.replace_extension();
        database = fs::path(platform::path_from_utf8(platform::path_to_utf8(base) +
                                                     ".files")) /
                   "job.db";
    }
    database.make_preferred();
    return platform::path_to_utf8(database);
}

std::int64_t JobDb::insert_job(const std::string& kind,
                               const std::string& payload) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#if defined(ALPHASHAPE_USE_SQLITE3)
    const std::string now = iso8601_utc_now();
    Statement statement(impl_->db,
                        "INSERT INTO jobs(kind,payload,state,created,updated) "
                        "VALUES(?,?,?,?,?)");
    statement.bind_text(1, kind);
    statement.bind_text(2, payload);
    statement.bind_text(3, "pending");
    statement.bind_text(4, now);
    statement.bind_text(5, now);
    statement.step_done(impl_->db);
    return static_cast<std::int64_t>(::sqlite3_last_insert_rowid(impl_->db));
#else
    return impl_->fallback->insert_job(kind, payload);
#endif
}

void JobDb::set_state(std::int64_t id, const std::string& state) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#if defined(ALPHASHAPE_USE_SQLITE3)
    Statement statement(impl_->db,
                        "UPDATE jobs SET state=?, updated=? WHERE id=?");
    statement.bind_text(1, state);
    statement.bind_text(2, iso8601_utc_now());
    statement.bind_int(3, id);
    statement.step_done(impl_->db);
    if (::sqlite3_changes(impl_->db) == 0) {
        throw Error("job.db: no job with id " + std::to_string(id));
    }
#else
    impl_->fallback->set_state(id, state);
#endif
}

std::vector<JobRecord> JobDb::list_jobs(
    const std::optional<std::string>& state) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#if defined(ALPHASHAPE_USE_SQLITE3)
    std::string sql =
        "SELECT id,kind,payload,state,created,updated FROM jobs";
    if (state.has_value()) {
        sql += " WHERE state=?";
    }
    sql += " ORDER BY id";
    Statement statement(impl_->db, sql.c_str());
    if (state.has_value()) {
        statement.bind_text(1, *state);
    }
    std::vector<JobRecord> out;
    for (;;) {
        const int rc = ::sqlite3_step(statement.get());
        if (rc == SQLITE_DONE) {
            break;
        }
        if (rc != SQLITE_ROW) {
            throw Error("job.db: read failed: " +
                        std::string(::sqlite3_errmsg(impl_->db)));
        }
        JobRecord record;
        record.id = ::sqlite3_column_int64(statement.get(), 0);
        record.kind = column_text(statement.get(), 1);
        record.payload = column_text(statement.get(), 2);
        record.state = column_text(statement.get(), 3);
        record.created = column_text(statement.get(), 4);
        record.updated = column_text(statement.get(), 5);
        out.push_back(std::move(record));
    }
    return out;
#else
    return impl_->fallback->list_jobs(state);
#endif
}

bool JobDb::remove_job(std::int64_t id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#if defined(ALPHASHAPE_USE_SQLITE3)
    Statement statement(impl_->db, "DELETE FROM jobs WHERE id=?");
    statement.bind_int(1, id);
    statement.step_done(impl_->db);
    return ::sqlite3_changes(impl_->db) > 0;
#else
    return impl_->fallback->remove_job(id);
#endif
}

}  // namespace alphashape
