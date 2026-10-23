#include "apps/gui/job_manager.hpp"

#include <algorithm>
#include <exception>
#include <utility>

#include <QMetaObject>

#include "apps/gui/app.hpp"
#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/core/document.hpp"

namespace alphashape::gui {

namespace {
constexpr std::size_t kMaxHistory = 200;
/// Progress is never reported as exactly "done" before the task returns.
constexpr double kMinEtaPercent = 2.0;
}  // namespace

JobManager::JobManager(GuiApp& app) : QObject(nullptr), app_(app), registry_(&builtin_registry()) {}

JobManager::~JobManager() {
    shutdown();
}

QString JobManager::state_name(State state) {
    switch (state) {
        case State::Pending: return QStringLiteral("pending");
        case State::Running: return QStringLiteral("running");
        case State::Succeeded: return QStringLiteral("succeeded");
        case State::Failed: return QStringLiteral("failed");
        case State::Cancelled: return QStringLiteral("cancelled");
    }
    return QStringLiteral("?");
}

void JobManager::set_registry(const TaskRegistry* registry) {
    if (registry == nullptr) {
        throw ValueError("the job manager needs a task registry");
    }
    if (busy()) {
        throw Error("cannot change the task registry while a job is active");
    }
    registry_ = registry;
}

// --- Queries -------------------------------------------------------------------------------

std::optional<JobManager::JobId> JobManager::running_job() const {
    if (!queue_.empty() && queue_.front().job.state == State::Running) {
        return queue_.front().job.id;
    }
    return std::nullopt;
}

std::size_t JobManager::pending_count() const {
    return queue_.empty() ? 0 : queue_.size() - (running_job() ? 1 : 0);
}

std::optional<JobManager::Job> JobManager::job(JobId id) const {
    for (const Request& request : queue_) {
        if (request.job.id == id) {
            return request.job;
        }
    }
    for (const Job& finished : history_) {
        if (finished.id == id) {
            return finished;
        }
    }
    return std::nullopt;
}

JobManager::State JobManager::state(JobId id) const {
    const std::optional<Job> found = job(id);
    return found ? found->state : State::Cancelled;
}

// --- Submitting and cancelling ---------------------------------------------------------------

JobManager::JobId JobManager::submit(const std::string& task_name, Params params,
                                     std::int32_t chunk_id) {
    if (shutting_down_) {
        throw Error("the job manager is shutting down");
    }
    Request request;
    request.job.id = next_id_++;
    request.job.task = QString::fromStdString(task_name);
    request.task_name = task_name;
    request.params = std::move(params);
    request.chunk_id = chunk_id;
    const JobId id = request.job.id;

    const bool was_idle = queue_.empty();
    queue_.push_back(std::move(request));
    if (was_idle) {
        Q_EMIT busy_changed(true);
    }
    Q_EMIT job_state_changed(id, State::Pending);
    if (was_idle) {
        start_next();
    }
    return id;
}

void JobManager::cancel() {
    if (worker_ && running_job()) {
        worker_->cancel.store(true);
        Q_EMIT app_.log_message(LogLevel::Info,
                                tr("Cancelling %1...").arg(queue_.front().job.task));
    }
}

bool JobManager::cancel(JobId id) {
    if (running_job() && queue_.front().job.id == id) {
        cancel();
        return true;
    }
    for (auto it = queue_.begin(); it != queue_.end(); ++it) {
        if (it->job.id != id) {
            continue;
        }
        Job dropped = it->job;
        queue_.erase(it);
        dropped.state = State::Cancelled;
        history_.push_back(dropped);
        if (history_.size() > kMaxHistory) {
            history_.erase(history_.begin());
        }
        Q_EMIT job_state_changed(id, State::Cancelled);
        Q_EMIT job_finished(id, State::Cancelled, QString());
        return true;
    }
    return false;
}

void JobManager::cancel_all() {
    std::vector<JobId> pending;
    for (std::size_t i = running_job() ? 1 : 0; i < queue_.size(); ++i) {
        pending.push_back(queue_[i].job.id);
    }
    for (const JobId id : pending) {
        cancel(id);
    }
    cancel();
}

void JobManager::shutdown() {
    shutting_down_ = true;
    cancel_all_and_wait();
}

void JobManager::cancel_all_and_wait() {
    cancel_all();
    if (worker_) {
        worker_->cancel.store(true);
        const std::shared_ptr<Worker> worker = worker_;
        if (worker->thread.joinable()) {
            worker->thread.join();
        }
        worker_done(worker);
    }
}

// --- Running ---------------------------------------------------------------------------------

void JobManager::set_state(Request& request, State next) {
    request.job.state = next;
    Q_EMIT job_state_changed(request.job.id, next);
}

void JobManager::start_next() {
    if (queue_.empty() || worker_) {
        return;
    }
    Request& front = queue_.front();
    stages_.clear();
    if (const TaskEntry* entry = registry_->find(front.task_name)) {
        for (const Stage& stage : entry->task->stages()) {
            stages_.push_back(stage.name);
        }
    }
    clock_.start();
    worker_ = std::make_shared<Worker>();
    set_state(front, State::Running);
    Q_EMIT app_.log_message(LogLevel::Info, tr("Started %1").arg(front.job.task));

    // The worker gets its own copy of the request, and keeps the document alive.
    std::shared_ptr<Worker> worker = worker_;
    worker->thread = std::thread([this, worker, request = front, document = app_.shared_document()]() mutable {
        run_worker(worker, std::move(request), std::move(document));
    });
}

void JobManager::run_worker(const std::shared_ptr<Worker>& worker, Request request,
                            std::shared_ptr<Document> document) {
    std::string error;
    try {
        std::unique_lock<std::shared_mutex> lock = app_.document_lock().write();
        const std::int32_t previous_active = document->active_chunk_id();
        if (request.chunk_id >= 0) {
            if (document->workspace().find_chunk(request.chunk_id) == nullptr) {
                throw Error("there is no chunk with id " + std::to_string(request.chunk_id));
            }
            document->set_active_chunk_id(request.chunk_id);
        }

        Progress progress;
        progress.report = [this, &worker](const std::string& stage, double fraction) {
            bool post = false;
            {
                std::lock_guard<std::mutex> guard(worker->mutex);
                worker->stage = stage;
                worker->fraction = fraction;
                post = !worker->progress_posted;
                worker->progress_posted = true;
            }
            if (post) {
                QMetaObject::invokeMethod(
                    this, [this, worker]() { deliver_progress(worker); }, Qt::QueuedConnection);
            }
        };
        progress.cancelled = [&worker]() { return worker->cancel.load(); };

        try {
            run_task(*registry_, *document, request.task_name, request.params, progress);
        } catch (...) {
            // A task that did not finish must leave the document as it found it.
            document->set_active_chunk_id(previous_active);
            throw;
        }
    } catch (const std::exception& e) {
        error = e.what();
        if (error.empty()) {
            error = "the task failed";
        }
    } catch (...) {
        error = "the task failed with an unknown error";
    }
    {
        std::lock_guard<std::mutex> guard(worker->mutex);
        worker->error = std::move(error);
        worker->done = true;
    }
    QMetaObject::invokeMethod(
        this, [this, worker]() { worker_done(worker); }, Qt::QueuedConnection);
}

void JobManager::deliver_progress(const std::shared_ptr<Worker>& worker) {
    std::string stage;
    double fraction = 0.0;
    {
        std::lock_guard<std::mutex> guard(worker->mutex);
        stage = worker->stage;
        fraction = worker->fraction;
        worker->progress_posted = false;
    }
    if (worker != worker_ || queue_.empty()) {
        return;  // a late update for a job that already ended
    }
    Job& current = queue_.front().job;
    fraction = std::clamp(fraction, 0.0, 1.0);

    double percent = current.percent;
    const auto at = std::find(stages_.begin(), stages_.end(), stage);
    if (at != stages_.end()) {
        percent = 100.0 * (static_cast<double>(at - stages_.begin()) + fraction) /
                  static_cast<double>(stages_.size());
    } else if (stages_.empty()) {
        percent = 100.0 * fraction;
    }
    // Overall progress never goes backwards, whatever order a task reports in.
    current.percent = std::clamp(std::max(current.percent, percent), 0.0, 100.0);

    const QString stage_name = QString::fromStdString(stage);
    if (stage_name != current.stage) {
        current.stage = stage_name;
        Q_EMIT app_.log_message(LogLevel::Info,
                                tr("%1: stage %2").arg(current.task, stage_name));
    }
    qint64 eta = -1;
    if (current.percent >= kMinEtaPercent && current.percent < 100.0) {
        eta = static_cast<qint64>(static_cast<double>(clock_.elapsed()) *
                                  (100.0 - current.percent) / current.percent);
    }
    Q_EMIT job_progress(current.id, current.percent, current.stage, eta);
}

void JobManager::worker_done(const std::shared_ptr<Worker>& worker) {
    if (worker != worker_) {
        return;  // already finalised by shutdown()
    }
    if (worker->thread.joinable()) {
        worker->thread.join();
    }
    std::string error;
    {
        std::lock_guard<std::mutex> guard(worker->mutex);
        error = worker->error;
    }
    const bool cancelled = worker->cancel.load();
    worker_.reset();
    if (error.empty()) {
        finish_current(State::Succeeded, QString());
    } else if (cancelled) {
        finish_current(State::Cancelled, QString());
    } else {
        finish_current(State::Failed, QString::fromStdString(error));
    }
}

void JobManager::finish_current(State final_state, const QString& message) {
    Request request = std::move(queue_.front());
    queue_.pop_front();
    Job done = std::move(request.job);
    done.state = final_state;
    done.message = message;
    done.elapsed_ms = clock_.elapsed();

    if (final_state == State::Succeeded) {
        // The task changed the document in ways the undo stack cannot reverse.
        app_.task_changed_document();
        done.percent = 100.0;
        Q_EMIT job_progress(done.id, 100.0, done.stage, 0);
    }
    history_.push_back(done);
    if (history_.size() > kMaxHistory) {
        history_.erase(history_.begin());
    }

    switch (final_state) {
        case State::Succeeded:
            Q_EMIT app_.log_message(LogLevel::Info, tr("%1 finished").arg(done.task));
            break;
        case State::Failed:
            Q_EMIT app_.log_message(LogLevel::Error,
                                    tr("%1 failed: %2").arg(done.task, message));
            break;
        default:
            Q_EMIT app_.log_message(LogLevel::Warn, tr("%1 cancelled").arg(done.task));
            break;
    }
    Q_EMIT job_state_changed(done.id, final_state);
    Q_EMIT job_finished(done.id, final_state, message);

    if (queue_.empty()) {
        Q_EMIT busy_changed(false);
    } else if (!shutting_down_) {
        start_next();
    }
}

}  // namespace alphashape::gui
