// WP-11.10: the job database. JobDb.* runs on whichever backend the configure
// selected; JobDbFallback.* always runs, so the JSON-lines path is covered on
// every lane even when SQLite is present.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "base/platform/env.hpp"
#include "base/platform/temp.hpp"
#include "common/io/job_db.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;
namespace plat = platform;

/// Removes the database and SQLite's side files when the test ends, including
/// an early ASSERT return.
class TempDb {
public:
    TempDb() : path_(platform::unique_temp_path("alphashape_jobdb_", ".db")) {}
    ~TempDb() {
        std::error_code ec;
        fs::remove(fs::path(path_), ec);
        fs::remove(fs::path(path_ + "-wal"), ec);
        fs::remove(fs::path(path_ + "-shm"), ec);
    }
    TempDb(const TempDb&) = delete;
    TempDb& operator=(const TempDb&) = delete;

    const std::string& path() const { return path_; }

private:
    std::string path_;
};

TEST(JobDb, InsertListUpdateRemove) {
    RecordProperty("backend", JobDb::backend());
    TempDb temp;
    JobDb db = JobDb::open(temp.path());

    const std::int64_t id = db.insert_job("build", "{\"a\":1}");
    EXPECT_GT(id, 0);
    std::vector<JobRecord> jobs = db.list_jobs();
    ASSERT_EQ(jobs.size(), 1U);
    EXPECT_EQ(jobs[0].id, id);
    EXPECT_EQ(jobs[0].kind, "build");
    EXPECT_EQ(jobs[0].payload, "{\"a\":1}");
    EXPECT_EQ(jobs[0].state, "pending");
    EXPECT_FALSE(jobs[0].created.empty());
    EXPECT_FALSE(jobs[0].updated.empty());

    db.set_state(id, "running");
    jobs = db.list_jobs(std::string("running"));
    ASSERT_EQ(jobs.size(), 1U);
    EXPECT_EQ(jobs[0].state, "running");
    EXPECT_TRUE(db.list_jobs(std::string("pending")).empty());

    EXPECT_TRUE(db.remove_job(id));
    EXPECT_TRUE(db.list_jobs().empty());
    EXPECT_FALSE(db.remove_job(id));
}

TEST(JobDb, ConcurrentInserts) {
    TempDb temp;
    JobDb db = JobDb::open(temp.path());
    std::mutex mutex;
    std::vector<std::int64_t> ids;
    std::vector<std::thread> workers;
    for (int thread = 0; thread < 4; ++thread) {
        workers.emplace_back([&db, &mutex, &ids, thread]() {
            for (int i = 0; i < 100; ++i) {
                const std::int64_t id =
                    db.insert_job("t" + std::to_string(thread), std::to_string(i));
                std::lock_guard<std::mutex> lock(mutex);
                ids.push_back(id);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    ASSERT_EQ(ids.size(), 400U);
    const std::set<std::int64_t> unique(ids.begin(), ids.end());
    EXPECT_EQ(unique.size(), 400U);
    EXPECT_EQ(db.list_jobs().size(), 400U);
}

TEST(JobDb, ReopenKeepsRows) {
    TempDb temp;
    std::int64_t first = 0;
    std::int64_t second = 0;
    {
        JobDb db = JobDb::open(temp.path());
        first = db.insert_job("a", "1");
        second = db.insert_job("b", "2");
        ASSERT_EQ(db.list_jobs().size(), 2U);
    }
    JobDb db = JobDb::open(temp.path());
    const std::vector<JobRecord> jobs = db.list_jobs();
    ASSERT_EQ(jobs.size(), 2U);
    EXPECT_EQ(jobs[0].id, first);
    EXPECT_EQ(jobs[1].id, second);
    const std::int64_t third = db.insert_job("c", "3");
    EXPECT_GT(third, second);
    EXPECT_EQ(db.list_jobs().size(), 3U);
}

TEST(JobDbFallback, InsertListUpdateRemove) {
    TempDb temp;
    JobDbFallback db(temp.path());

    const std::int64_t id = db.insert_job("build", "{\"a\":1}");
    EXPECT_GT(id, 0);
    std::vector<JobRecord> jobs = db.list_jobs();
    ASSERT_EQ(jobs.size(), 1U);
    EXPECT_EQ(jobs[0].id, id);
    EXPECT_EQ(jobs[0].kind, "build");
    EXPECT_EQ(jobs[0].payload, "{\"a\":1}");
    EXPECT_EQ(jobs[0].state, "pending");

    db.set_state(id, "running");
    jobs = db.list_jobs(std::string("running"));
    ASSERT_EQ(jobs.size(), 1U);
    EXPECT_EQ(jobs[0].state, "running");
    EXPECT_TRUE(db.list_jobs(std::string("pending")).empty());

    EXPECT_TRUE(db.remove_job(id));
    EXPECT_TRUE(db.list_jobs().empty());
    EXPECT_FALSE(db.remove_job(id));
}

TEST(JobDbFallback, ConcurrentInserts) {
    TempDb temp;
    JobDbFallback db(temp.path());
    std::mutex mutex;
    std::vector<std::int64_t> ids;
    std::vector<std::thread> workers;
    for (int thread = 0; thread < 4; ++thread) {
        workers.emplace_back([&db, &mutex, &ids, thread]() {
            for (int i = 0; i < 100; ++i) {
                const std::int64_t id =
                    db.insert_job("t" + std::to_string(thread), std::to_string(i));
                std::lock_guard<std::mutex> lock(mutex);
                ids.push_back(id);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    ASSERT_EQ(ids.size(), 400U);
    const std::set<std::int64_t> unique(ids.begin(), ids.end());
    EXPECT_EQ(unique.size(), 400U);
    EXPECT_EQ(db.list_jobs().size(), 400U);
}

TEST(JobDbFallback, ReopenKeepsRows) {
    TempDb temp;
    std::int64_t first = 0;
    std::int64_t second = 0;
    {
        JobDbFallback db(temp.path());
        first = db.insert_job("a", "1");
        second = db.insert_job("b", "2");
        ASSERT_EQ(db.list_jobs().size(), 2U);
    }
    JobDbFallback db(temp.path());
    const std::vector<JobRecord> jobs = db.list_jobs();
    ASSERT_EQ(jobs.size(), 2U);
    EXPECT_EQ(jobs[0].id, first);
    EXPECT_EQ(jobs[1].id, second);
    const std::int64_t third = db.insert_job("c", "3");
    EXPECT_GT(third, second);
    EXPECT_EQ(db.list_jobs().size(), 3U);
}

/// Appends bytes verbatim: used to plant a torn (unterminated) record.
void append_raw(const std::string& path, const std::string& text) {
    std::ofstream out(plat::path_from_utf8(path),
                      std::ios::binary | std::ios::app);
    out << text;
}

TEST(JobDb, IdsNotReusedAfterRemove) {
    TempDb temp;
    std::int64_t removed = 0;
    {
        JobDb db = JobDb::open(temp.path());
        db.insert_job("a", "1");
        db.insert_job("b", "2");
        removed = db.insert_job("c", "3");
        ASSERT_TRUE(db.remove_job(removed));
    }
    JobDb db = JobDb::open(temp.path());
    const std::int64_t next = db.insert_job("d", "4");
    EXPECT_GT(next, removed);
    EXPECT_EQ(db.list_jobs().size(), 3U);
}

TEST(JobDbFallback, IdsNotReusedAfterRemove) {
    TempDb temp;
    std::int64_t removed = 0;
    {
        JobDbFallback db(temp.path());
        db.insert_job("a", "1");
        db.insert_job("b", "2");
        removed = db.insert_job("c", "3");
        ASSERT_TRUE(db.remove_job(removed));
    }
    JobDbFallback db(temp.path());
    const std::int64_t next = db.insert_job("d", "4");
    EXPECT_GT(next, removed);
    EXPECT_EQ(db.list_jobs().size(), 3U);
}

TEST(JobDbFallback, TornRemoveIsIgnored) {
    TempDb temp;
    std::int64_t id = 0;
    {
        JobDbFallback db(temp.path());
        id = db.insert_job("build", "{}");
    }
    // A crash between operands leaves an unterminated fragment at the tail.
    append_raw(temp.path(), "{\"op\":\"remove\",\"id\":" + std::to_string(id));
    JobDbFallback db(temp.path());
    const std::vector<JobRecord> jobs = db.list_jobs();
    ASSERT_EQ(jobs.size(), 1U);
    EXPECT_EQ(jobs[0].id, id);
}

TEST(JobDbFallback, TornInsertIsIgnored) {
    TempDb temp;
    std::int64_t first = 0;
    {
        JobDbFallback db(temp.path());
        first = db.insert_job("build", "{}");
    }
    append_raw(temp.path(), "{\"op\":\"insert\",\"id\":2,\"kind\":\"torn\"");
    {
        JobDbFallback db(temp.path());
        ASSERT_EQ(db.list_jobs().size(), 1U);
        // The torn record was truncated, so the next append starts a fresh line
        // and the new id cannot collide with the id the fragment named.
        const std::int64_t next = db.insert_job("build2", "{}");
        EXPECT_GT(next, first);
    }
    JobDbFallback db(temp.path());
    const std::vector<JobRecord> jobs = db.list_jobs();
    ASSERT_EQ(jobs.size(), 2U);
    EXPECT_NE(jobs[0].id, jobs[1].id);
}

TEST(JobDb, PathForProject) {
    const fs::path base =
        plat::path_from_utf8(plat::unique_temp_path("alphashape_paths_", ""));
    const std::string dir_db =
        JobDb::path_for_project(plat::path_to_utf8(base / "survey.psx"));
    EXPECT_EQ(plat::path_from_utf8(dir_db), base / "survey.files" / "job.db");
    EXPECT_FALSE(fs::exists(base / "survey.files"));

    const std::string file_db =
        JobDb::path_for_project(plat::path_to_utf8(base / "survey.psz"));
    EXPECT_EQ(plat::path_from_utf8(file_db), base / "survey.psz.jobs.db");
    EXPECT_FALSE(fs::exists(base / "survey.files"));
}

}  // namespace
}  // namespace alphashape

