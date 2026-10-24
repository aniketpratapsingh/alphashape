#pragma once

// A task that blocks until released, so a GUI test can observe the "job running" state and
// then let it end. The wait is on a condition variable rather than a 1 ms poll: a spinning
// worker would burn a core for as long as the test needs the job alive, which on a shared CI
// machine slows every other lane.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "base/utils/error.hpp"
#include "common/tasks/task.hpp"

namespace alphashape::gui::testing {

class Gate {
public:
    void open() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            open_ = true;
        }
        ready_.notify_all();
    }

    /// Wait until open() or the timeout; true when it was opened.
    bool wait(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return ready_.wait_for(lock, timeout, [this]() { return open_; });
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_;
    bool open_ = false;
};

class HoldTask : public AbstractTask {
public:
    HoldTask(std::shared_ptr<Gate> release, std::shared_ptr<std::atomic<bool>> started)
        : release_(std::move(release)), started_(std::move(started)) {}
    std::string name() const override { return "HoldTask"; }
    void declare_params(Params&) const override {}
    TargetType target_type() const override { return TargetType::Chunk; }
    void apply(Document&, Params&, const Progress& progress) override {
        started_->store(true);
        while (!release_->wait(std::chrono::milliseconds(20))) {
            if (progress.is_cancelled()) {
                throw CancelledError("cancelled");
            }
        }
    }

private:
    std::shared_ptr<Gate> release_;
    std::shared_ptr<std::atomic<bool>> started_;
};

}  // namespace alphashape::gui::testing
