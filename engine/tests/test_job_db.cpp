// WP-11.10: the job database. JobDb.* runs on whichever backend the configure
// selected; JobDbFallback.* always runs, so the JSON-lines path is covered on
// every lane even when SQLite is present.

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "base/platform/temp.hpp"
#include "common/io/job_db.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

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

}  // namespace
}  // namespace alphashape
