// Jobs.*: the job manager (non-blocking run, progress, cancel) and its window wiring, on the
// offscreen harness. Real engine tasks run on the small camera fixtures; the cancel / queue /
// lock tests use a gated test task registered in a private registry, so they never depend on how
// fast a machine is.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <QAction>
#include <QLabel>
#include <QProgressBar>
#include <QStatusBar>
#include <QToolButton>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/commands/model_commands.hpp"
#include "common/tasks/registry.hpp"
#include "harness.hpp"

using namespace alphashape;
using namespace alphashape::gui;
using alphashape::gui::testing::GuiTest;
using alphashape::gui::testing::pump_until;

namespace {

constexpr int kWaitMs = 60000;  // upper bound for polling, not a delay

/// Shared between a test and its GateTask.
struct Gate {
    std::atomic<bool> started{false};
    std::atomic<bool> release{false};
    std::mutex mutex;
    std::vector<std::string> order;  ///< labels of the runs that completed
};

/// Holds at a gate until released or cancelled, then reports a few progress values (one of them
/// a regression, to check the manager keeps overall progress monotonic), logs a line and
/// stamps the chunk.
class GateTask : public AbstractTask {
public:
    explicit GateTask(std::shared_ptr<Gate> gate) : gate_(std::move(gate)) {}

    std::string name() const override { return "GateTask"; }
    void declare_params(Params& params) const override {
        params.declare({"label", ParamType::String, std::string{"run"}, "Which run this is."});
    }
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override { return {Stage{"a", 0}, Stage{"b", 0}}; }

    void apply(Document& document, Params& params, const Progress& progress) override {
        const std::string label = params.get_string("label");
        progress.update("a", 0.0);
        gate_->started = true;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!gate_->release.load()) {
            if (progress.is_cancelled()) {
                if (label == "cancel_then_fail") {
                    // A real failure that merely arrives after the cancel request.
                    throw Error("gate task broke while stopping");
                }
                // The engine's cancellation exception; the document is touched first, as a task
                // that stopped half way through its work would have.
                if (label == "mutate") {
                    document.workspace().add_chunk("Partial");
                }
                throw CancelledError("GateTask was cancelled");
            }
            if (std::chrono::steady_clock::now() > deadline) {
                throw Error("GateTask timed out waiting for the test");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (label == "fail") {
            throw Error("gate task failed on purpose");
        }
        if (label == "mutate_fail") {
            document.workspace().add_chunk("Partial");
            throw Error("gate task failed after mutating");
        }
        progress.update("a", 0.5);
        alphashape::log(LogLevel::Info) << "gate halfway " << label;
        progress.update("a", 1.0);
        progress.update("b", 0.5);
        progress.update("a", 0.2);  // a task going backwards must not move the bar back
        progress.update("b", 1.0);
        document.active_chunk()->meta().set("test/marker", label);
        std::lock_guard<std::mutex> guard(gate_->mutex);
        gate_->order.push_back(label);
    }

private:
    std::shared_ptr<Gate> gate_;
};

struct Finished {
    quint64 id;
    JobManager::State state;
    QString message;
};

class JobsTest : public GuiTest {
protected:
    void SetUp() override {
        GuiTest::SetUp();
        gate_ = std::make_shared<Gate>();
        registry_.register_task(std::make_shared<GateTask>(gate_));

        Chunk& added = app().document().workspace().add_chunk("Chunk 1");
        chunk_id_ = added.id();
        app().document().set_active_chunk_id(chunk_id_);

        JobManager& jobs = app().jobs();
        QObject::connect(&jobs, &JobManager::job_state_changed, &app(),
                         [this](quint64 id, JobManager::State state) {
                             states_.emplace_back(id, state);
                         });
        QObject::connect(&jobs, &JobManager::job_progress, &app(),
                         [this](quint64, double percent, const QString&, qint64) {
                             percents_.push_back(percent);
                         });
        QObject::connect(&jobs, &JobManager::job_finished, &app(),
                         [this](quint64 id, JobManager::State state, const QString& message) {
                             finished_.push_back({id, state, message});
                         });
        QObject::connect(&app(), &GuiApp::log_message, &app(),
                         [this](alphashape::LogLevel, const QString& text) {
                             logs_.push_back(text);
                         });
        window().set_error_handler([this](const QString& title, const QString& message) {
            errors_.emplace_back(title, message);
        });
    }

    void TearDown() override {
        // The app (and its job manager, which points at registry_) goes first; registry_ is a
        // member of this class and is only destroyed afterwards.
        GuiTest::TearDown();
    }

    void use_gate_registry() { app().jobs().set_registry(&registry_); }

    Params gate_params(const char* label = "run") {
        Params params = make_params(registry_.require("GateTask"));
        params.set("label", ParamValue{std::string(label)});
        return params;
    }

    void wait_started() {
        ASSERT_TRUE(pump_until([&] { return gate_->started.load(); }, kWaitMs));
    }
    void wait_finished(std::size_t count = 1) {
        ASSERT_TRUE(pump_until([&] { return finished_.size() >= count; }, kWaitMs));
        // busy_changed(false) follows job_finished on the same turn when the queue is empty.
        ASSERT_TRUE(pump_until([&] { return !app().jobs().busy() || finished_.size() > count; },
                               kWaitMs));
    }
    bool saw_log(const QString& needle) const {
        for (const QString& line : logs_) {
            if (line.contains(needle)) {
                return true;
            }
        }
        return false;
    }
    Chunk& chunk() { return *app().document().active_chunk(); }

    /// Two photos in the chunk, added directly (no job) before the test starts.
    void add_fixture_photos() {
        Params params = make_params(builtin_registry().require("AddPhotos"));
        const std::string dir = ALPHASHAPE_GUI_DATA_DIR;
        params.set("filenames", ParamValue{std::vector<std::string>{dir + "/cam_a1.jpg",
                                                                   dir + "/cam_a2.jpg"}});
        run_task(builtin_registry(), app().document(), "AddPhotos", params);
    }

    std::shared_ptr<Gate> gate_;
    TaskRegistry registry_;
    std::int32_t chunk_id_ = -1;
    std::vector<std::pair<quint64, JobManager::State>> states_;
    std::vector<double> percents_;
    std::vector<Finished> finished_;
    std::vector<QString> logs_;
    std::vector<std::pair<QString, QString>> errors_;
};

}  // namespace

TEST_F(JobsTest, RealTaskRunsOffTheGuiThreadAndSucceeds) {
    add_fixture_photos();
    app().execute(make_add_chunk_command(app().document().workspace(), "Extra", -1));
    ASSERT_TRUE(app().undo_stack().can_undo());
    ASSERT_EQ(chunk().tie_points().keypoint_sets.size(), 0U);

    Params params = make_params(builtin_registry().require("DetectFeatures"));
    const quint64 id = app().jobs().submit("DetectFeatures", params, chunk_id_);
    EXPECT_TRUE(app().jobs().busy());  // submit returned without waiting for the task

    wait_finished();
    ASSERT_EQ(finished_.size(), 1U);
    EXPECT_EQ(finished_[0].id, id);
    EXPECT_EQ(finished_[0].state, JobManager::State::Succeeded) << finished_[0].message.toStdString();
    EXPECT_FALSE(app().jobs().busy());
    EXPECT_EQ(chunk().tie_points().keypoint_sets.size(), 2U);
    EXPECT_TRUE(errors_.empty());

    // There is no reversal command for tasks, so the undo history is dropped and the project
    // is marked as changed.
    EXPECT_FALSE(app().undo_stack().can_undo());
    EXPECT_TRUE(app().dirty());
    EXPECT_FALSE(action("action.edit.undo")->isEnabled());

    // States went Pending -> Running -> Succeeded.
    ASSERT_EQ(states_.size(), 3U);
    EXPECT_EQ(states_[0].second, JobManager::State::Pending);
    EXPECT_EQ(states_[1].second, JobManager::State::Running);
    EXPECT_EQ(states_[2].second, JobManager::State::Succeeded);
    ASSERT_FALSE(percents_.empty());
    EXPECT_DOUBLE_EQ(percents_.back(), 100.0);
    EXPECT_TRUE(saw_log(QStringLiteral("DetectFeatures")));
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(JobsTest, ProgressIsMonotonicAndReportsStages) {
    use_gate_registry();
    QStringList stages;
    QObject::connect(&app().jobs(), &JobManager::job_progress, &app(),
                     [&](quint64, double, const QString& stage, qint64) { stages << stage; });
    app().jobs().submit("GateTask", gate_params());
    wait_started();
    gate_->release = true;
    wait_finished();

    ASSERT_EQ(finished_[0].state, JobManager::State::Succeeded);
    ASSERT_FALSE(percents_.empty());
    for (std::size_t i = 1; i < percents_.size(); ++i) {
        EXPECT_GE(percents_[i], percents_[i - 1]) << "at update " << i;
    }
    EXPECT_GE(percents_.front(), 0.0);
    EXPECT_DOUBLE_EQ(percents_.back(), 100.0);
    EXPECT_TRUE(stages.contains(QStringLiteral("a")));
    EXPECT_EQ(chunk().meta().find("test/marker") != nullptr, true);
    EXPECT_TRUE(saw_log(QStringLiteral("gate halfway run")));  // engine log line forwarded
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(JobsTest, CancelMidRunEndsCancelledAndInvalidatesTheHistory) {
    use_gate_registry();
    app().execute(make_add_chunk_command(app().document().workspace(), "Extra", -1));
    const Chunk before = chunk();
    const std::size_t chunks_before = app().document().workspace().chunks().size();

    const quint64 id = app().jobs().submit("GateTask", gate_params());
    wait_started();
    ASSERT_EQ(app().jobs().state(id), JobManager::State::Running);

    // The status strip button is wired to the same cancel.
    auto* button = widget<QToolButton>(QStringLiteral("status.cancel"));
    ASSERT_NE(button, nullptr);
    EXPECT_TRUE(button->isVisibleTo(&window()));
    button->click();
    wait_finished();

    ASSERT_EQ(finished_.size(), 1U);
    EXPECT_EQ(finished_[0].state, JobManager::State::Cancelled);
    EXPECT_TRUE(errors_.empty());  // a cancel is not an error
    EXPECT_TRUE(chunk() == before);
    EXPECT_EQ(app().document().workspace().chunks().size(), chunks_before);
    // The task had started, so it may have changed the document: dirty, and no stale undo.
    EXPECT_TRUE(app().dirty());
    EXPECT_FALSE(app().undo_stack().can_undo());
    EXPECT_FALSE(action("action.edit.undo")->isEnabled());
    EXPECT_FALSE(button->isVisibleTo(&window()));
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(JobsTest, CancelAfterPartialMutationMarksDirtyAndClearsUndo) {
    use_gate_registry();
    app().execute(make_add_chunk_command(app().document().workspace(), "Extra", -1));
    const std::size_t chunks_before = app().document().workspace().chunks().size();

    const quint64 id = app().jobs().submit("GateTask", gate_params("mutate"));
    wait_started();
    app().jobs().cancel(id);
    wait_finished();

    ASSERT_EQ(finished_.size(), 1U);
    EXPECT_EQ(finished_[0].state, JobManager::State::Cancelled);
    EXPECT_EQ(app().document().workspace().chunks().size(), chunks_before + 1);  // partial
    EXPECT_TRUE(app().dirty());
    EXPECT_FALSE(app().undo_stack().can_undo());
    EXPECT_FALSE(app().jobs().busy());
}

TEST_F(JobsTest, FailureAfterPartialMutationMarksDirtyAndClearsUndo) {
    use_gate_registry();
    app().execute(make_add_chunk_command(app().document().workspace(), "Extra", -1));
    const std::size_t chunks_before = app().document().workspace().chunks().size();

    app().jobs().submit("GateTask", gate_params("mutate_fail"));
    wait_started();
    gate_->release = true;
    wait_finished();

    ASSERT_EQ(finished_.size(), 1U);
    EXPECT_EQ(finished_[0].state, JobManager::State::Failed);
    EXPECT_EQ(app().document().workspace().chunks().size(), chunks_before + 1);
    EXPECT_TRUE(app().dirty());
    EXPECT_FALSE(app().undo_stack().can_undo());
}

TEST_F(JobsTest, FailureBeforeAnyTaskCodeLeavesDirtyAndUndoAlone) {
    use_gate_registry();
    app().execute(make_add_chunk_command(app().document().workspace(), "Extra", -1));
    const bool dirty_before = app().dirty();

    app().jobs().submit("NoSuchTask", Params{});
    app().jobs().submit("GateTask", gate_params(), /*chunk_id=*/999);
    wait_finished(2);

    EXPECT_EQ(finished_[0].state, JobManager::State::Failed);
    EXPECT_EQ(finished_[1].state, JobManager::State::Failed);
    EXPECT_EQ(app().dirty(), dirty_before);
    EXPECT_TRUE(app().undo_stack().can_undo());
}

TEST_F(JobsTest, ARealErrorAfterCancelIsFailedNotCancelled) {
    use_gate_registry();
    const quint64 id = app().jobs().submit("GateTask", gate_params("cancel_then_fail"));
    wait_started();
    app().jobs().cancel(id);
    wait_finished();

    ASSERT_EQ(finished_.size(), 1U);
    EXPECT_EQ(finished_[0].state, JobManager::State::Failed);
    EXPECT_TRUE(finished_[0].message.contains(QStringLiteral("broke while stopping")));
    EXPECT_EQ(errors_.size(), 1U);
}

TEST_F(JobsTest, StateOfAnUnknownIdIsUnknown) {
    EXPECT_EQ(app().jobs().state(424242), JobManager::State::Unknown);
    EXPECT_FALSE(app().jobs().job(424242).has_value());
}

TEST_F(JobsTest, ReadDocumentRunsUnderTheSharedLockAndLogSinkKeepsWorking) {
    const std::string name = app().read_document([](const Document& d) { return d.path(); });
    EXPECT_TRUE(name.empty());
    EXPECT_TRUE(app().document_lock().try_write().owns_lock());  // the read lock was released
}

TEST_F(JobsTest, FailingRealTaskEndsFailedAndCallsTheErrorHandler) {
    add_fixture_photos();
    const Chunk before = chunk();
    Params params = make_params(builtin_registry().require("DetectFeatures"));
    params.set("descriptor_type", ParamValue{std::string("bogus")});
    app().jobs().submit("DetectFeatures", params, chunk_id_);
    wait_finished();

    ASSERT_EQ(finished_.size(), 1U);
    EXPECT_EQ(finished_[0].state, JobManager::State::Failed);
    EXPECT_TRUE(finished_[0].message.contains(QStringLiteral("bogus")));
    ASSERT_EQ(errors_.size(), 1U);
    EXPECT_TRUE(errors_[0].second.contains(QStringLiteral("bogus")));
    EXPECT_TRUE(chunk() == before);
    EXPECT_FALSE(app().jobs().busy());
    EXPECT_TRUE(saw_log(QStringLiteral("failed")));
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(JobsTest, UnknownTaskAndUnknownChunkFailWithoutCrashing) {
    app().jobs().submit("NoSuchTask", Params{});
    wait_finished(1);
    EXPECT_EQ(finished_[0].state, JobManager::State::Failed);

    use_gate_registry();
    app().jobs().submit("GateTask", gate_params(), /*chunk_id=*/999);
    wait_finished(2);
    EXPECT_EQ(finished_[1].state, JobManager::State::Failed);
    EXPECT_TRUE(finished_[1].message.contains(QStringLiteral("999")));
    EXPECT_EQ(errors_.size(), 2U);
    EXPECT_EQ(app().document().active_chunk_id(), chunk_id_);  // untouched
}

TEST_F(JobsTest, MutatingActionsAreDisabledWhileRunningAndComeBack) {
    use_gate_registry();
    app().execute(make_add_chunk_command(app().document().workspace(), "Extra", -1));
    const char* const names[] = {"action.file.new", "action.file.open", "action.file.save",
                                 "action.file.save_as", "action.file.close",
                                 "action.edit.undo"};
    for (const char* name : names) {
        EXPECT_TRUE(action(QString::fromLatin1(name))->isEnabled()) << name;
    }
    EXPECT_TRUE(app().document_lock().try_read().owns_lock());

    app().jobs().submit("GateTask", gate_params());
    wait_started();
    for (const char* name : names) {
        EXPECT_FALSE(action(QString::fromLatin1(name))->isEnabled()) << name;
    }
    // Even a caller that skips the actions is refused, and the worker owns the document lock.
    EXPECT_THROW(app().new_document(""), Error);
    EXPECT_THROW(app().execute(make_add_chunk_command(app().document().workspace(), "No", -1)),
                 Error);
    EXPECT_THROW(app().undo(), Error);
    EXPECT_FALSE(app().document_lock().try_read().owns_lock());

    // The status strip shows the stage text, the bar and the cancel button.
    EXPECT_TRUE(widget<QProgressBar>(QStringLiteral("status.progress"))->isVisibleTo(&window()));
    EXPECT_TRUE(widget<QToolButton>(QStringLiteral("status.cancel"))->isVisibleTo(&window()));
    EXPECT_TRUE(window().statusBar()->findChild<QLabel*>(QStringLiteral("status.jobs"))
                    ->text().contains(QStringLiteral("GateTask")));

    gate_->release = true;
    wait_finished();
    for (const char* name : {"action.file.new", "action.file.open", "action.file.save",
                             "action.file.save_as", "action.file.close"}) {
        EXPECT_TRUE(action(QString::fromLatin1(name))->isEnabled()) << name;
    }
    EXPECT_TRUE(app().document_lock().try_read().owns_lock());
    EXPECT_FALSE(widget<QProgressBar>(QStringLiteral("status.progress"))->isVisibleTo(&window()));
    EXPECT_NO_THROW(app().new_document(""));
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(JobsTest, JobsRunOneAtATimeInSubmissionOrder) {
    use_gate_registry();
    const quint64 first = app().jobs().submit("GateTask", gate_params("first"));
    const quint64 second = app().jobs().submit("GateTask", gate_params("second"));
    const quint64 third = app().jobs().submit("GateTask", gate_params("third"));
    wait_started();
    EXPECT_EQ(app().jobs().state(first), JobManager::State::Running);
    EXPECT_EQ(app().jobs().state(second), JobManager::State::Pending);
    EXPECT_EQ(app().jobs().pending_count(), 2U);

    // A pending job can be dropped without ever running.
    EXPECT_TRUE(app().jobs().cancel(third));
    EXPECT_EQ(app().jobs().state(third), JobManager::State::Cancelled);
    EXPECT_EQ(app().jobs().pending_count(), 1U);

    gate_->release = true;
    ASSERT_TRUE(pump_until([&] { return !app().jobs().busy(); }, kWaitMs));
    ASSERT_EQ(finished_.size(), 3U);
    EXPECT_EQ(finished_[0].id, third);  // dropped at once
    EXPECT_EQ(finished_[1].id, first);
    EXPECT_EQ(finished_[2].id, second);
    EXPECT_EQ(finished_[1].state, JobManager::State::Succeeded);
    EXPECT_EQ(finished_[2].state, JobManager::State::Succeeded);
    const std::vector<std::string> expected{"first", "second"};
    EXPECT_EQ(gate_->order, expected);
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(JobsTest, QuittingWhileRunningAsksToCancel) {
    use_gate_registry();
    app().jobs().submit("GateTask", gate_params());
    wait_started();

    QString asked_about;
    bool answer = false;
    window().set_cancel_jobs_handler([&](const QString& task) {
        asked_about = task;
        return answer;
    });
    EXPECT_FALSE(window().close());  // "no": the window and the job stay
    EXPECT_EQ(asked_about, QStringLiteral("GateTask"));
    EXPECT_TRUE(app().jobs().busy());

    answer = true;
    EXPECT_TRUE(window().close());
    EXPECT_FALSE(app().jobs().busy());
    ASSERT_EQ(finished_.size(), 1U);
    EXPECT_EQ(finished_[0].state, JobManager::State::Cancelled);
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(JobsTest, DestroyingTheAppWithAJobRunningEndsItCleanly) {
    use_gate_registry();
    app().jobs().submit("GateTask", gate_params());
    wait_started();
    rebuild(/*reset_settings=*/true);  // must cancel and join, not crash or hang
    EXPECT_FALSE(app().jobs().busy());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}
