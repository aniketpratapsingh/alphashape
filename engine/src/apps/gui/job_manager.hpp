#pragma once

// JobManager: runs engine tasks on a worker thread so the window stays responsive.
//
// One job runs at a time; further submissions wait in FIFO order. A job moves
// Pending -> Running -> Succeeded | Failed | Cancelled. The worker holds the document's
// exclusive DocumentLock for the whole task (see GuiApp::document_lock()); the GUI disables the
// actions that would touch the document for as long as busy() is true.
//
// All public functions and all signals belong to the GUI thread. The worker posts its progress
// and its result back as queued calls, so nothing GUI-side runs on the worker.
//
// Cancellation is cooperative: cancel() raises the flag behind Progress::cancelled, which the
// tasks poll at their stage / work-item boundaries and answer by throwing. A job that was asked
// to cancel and then threw (or was still pending) ends Cancelled; one that finished anyway
// ends Succeeded.

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <QElapsedTimer>
#include <QObject>
#include <QString>

#include "common/tasks/params.hpp"
#include "common/tasks/registry.hpp"

namespace alphashape::gui {

class GuiApp;

class JobManager : public QObject {
    Q_OBJECT

public:
    using JobId = quint64;

    enum class State { Pending, Running, Succeeded, Failed, Cancelled };
    Q_ENUM(State)

    /// What the manager remembers about a job (also after it finished).
    struct Job {
        JobId id = 0;
        QString task;
        State state = State::Pending;
        double percent = 0.0;  ///< overall, 0..100, never decreases
        QString stage;         ///< current stage name, "" before the first report
        QString message;       ///< failure reason (Failed) or short note
        qint64 elapsed_ms = 0; ///< set when the job finished
    };

    explicit JobManager(GuiApp& app);
    ~JobManager() override;

    /// Queue `task_name` with `params`. `chunk_id` >= 0 makes that chunk the active one for the
    /// run; -1 keeps the active chunk. Returns at once. Problems with the task itself (unknown
    /// name, bad parameter values) are reported by the job ending Failed, not by throwing.
    /// Throws alphashape::Error only when the manager is shutting down.
    JobId submit(const std::string& task_name, Params params, std::int32_t chunk_id = -1);

    /// Ask the running job to stop. No-op when nothing runs. Queued jobs stay queued.
    void cancel();
    /// Cancel one job: a pending job is dropped at once, the running one is asked to stop.
    /// False when the id is unknown or already finished.
    bool cancel(JobId id);
    /// Cancel the running job and drop every pending one.
    void cancel_all();

    /// Cancel everything and block until the worker thread has ended (the window uses this when
    /// the user quits while a job runs). The manager stays usable.
    void cancel_all_and_wait();
    /// Same, and refuse further submissions. Called when the application goes away.
    void shutdown();

    /// True while a job is pending or running: the document must not be touched meanwhile.
    bool busy() const { return !queue_.empty(); }
    std::optional<JobId> running_job() const;
    std::size_t pending_count() const;
    std::optional<Job> job(JobId id) const;
    /// State of a job; an id the manager never issued (or has forgotten) reports Cancelled.
    State state(JobId id) const;

    /// Replace the registry tasks are looked up in (default: the built-in one). Tests register
    /// their own tasks this way. Not allowed while busy.
    void set_registry(const TaskRegistry* registry);

    static QString state_name(State state);

Q_SIGNALS:
    void job_state_changed(quint64 id, JobManager::State state);
    /// Overall percent (0..100, monotonic per job), the stage name, and an ETA in milliseconds
    /// (-1 until it can be estimated).
    void job_progress(quint64 id, double percent, const QString& stage, qint64 eta_ms);
    /// A job reached a final state. `message` is the failure reason for Failed.
    void job_finished(quint64 id, JobManager::State state, const QString& message);
    /// Emitted when busy() flips.
    void busy_changed(bool busy);

private:
    struct Request {
        Job job;
        std::string task_name;
        Params params;
        std::int32_t chunk_id = -1;
    };

    /// State shared with the worker thread.
    struct Worker {
        std::atomic<bool> cancel{false};
        std::mutex mutex;  // guards the fields below
        std::string stage;
        double fraction = 0.0;
        bool progress_posted = false;
        bool done = false;
        std::string error;  // empty on success
        std::thread thread;
    };

    GuiApp& app_;
    const TaskRegistry* registry_;
    std::deque<Request> queue_;  ///< front = the running job (once started)
    std::vector<Job> history_;   ///< finished jobs, oldest first, bounded
    std::shared_ptr<Worker> worker_;
    std::vector<std::string> stages_;  ///< stage names of the running task
    QElapsedTimer clock_;
    JobId next_id_ = 1;
    bool shutting_down_ = false;

    void start_next();
    void run_worker(const std::shared_ptr<Worker>& worker, Request request,
                    std::shared_ptr<Document> document);
    void deliver_progress(const std::shared_ptr<Worker>& worker);
    void worker_done(const std::shared_ptr<Worker>& worker);
    void finish_current(State final_state, const QString& message);
    void set_state(Request& request, State next);
};

}  // namespace alphashape::gui
