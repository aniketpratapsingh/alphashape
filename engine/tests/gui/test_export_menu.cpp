// ExportMenu.*: the File > Export submenu. Each entry is enabled only when the active chunk
// holds that asset, opens the schema-generated TaskDialog on a suggested destination, and runs
// the engine's export task through the JobManager. The end-to-end test writes a real file.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <QAction>
#include <QApplication>
#include <QDialog>
#include <QDir>
#include <QFileInfo>
#include <QLineEdit>
#include <QMenu>
#include <QStatusBar>
#include <QStringList>
#include <QTimer>

#include "apps/gui/export_controller.hpp"
#include "apps/gui/main_window.hpp"
#include "apps/gui/task_dialog.hpp"
#include "apps/gui/workflow.hpp"
#include "base/utils/error.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/io/project_io.hpp"
#include "common/tasks/registry.hpp"
#include "harness.hpp"
#include "hold_task.hpp"

using namespace alphashape;
using namespace alphashape::gui;
using alphashape::gui::testing::Gate;
using alphashape::gui::testing::GuiTest;
using alphashape::gui::testing::HoldTask;
using alphashape::gui::testing::pump_until;

namespace {

constexpr int kPumpTimeoutMs = 15000;

class ExportTest : public GuiTest {
protected:
    void SetUp() override {
        GuiTest::SetUp();
        Chunk& added = app().document().workspace().add_chunk("Chunk 1");
        chunk_id_ = added.id();
        app().document().set_active_chunk_id(chunk_id_);
        // A failing job would open a modal error box that never closes offscreen.
        window().set_error_handler([this](const QString& title, const QString& message) {
            errors_.emplace_back(title, message);
        });
        window().exports().set_submit_handler(
            [this](const std::string& task, Params params, std::int32_t target)
                -> std::optional<quint64> {
                submitted_.push_back(Submitted{task, std::move(params), target});
                return std::nullopt;  // never queued: this test asserts the request only
            });
        window().exports().update_enabled();
        ready_ = true;
    }

    /// Leave the job manager idle and pointed at the built-in registry, whatever the test did.
    void TearDown() override {
        if (!ready_) {
            GuiTest::TearDown();
            return;
        }
        hold_release_->open();
        if (!pump_until([this]() { return !app().jobs().busy(); }, kPumpTimeoutMs)) {
            app().jobs().cancel_all_and_wait();
        }
        app().jobs().set_registry(&builtin_registry());
        GuiTest::TearDown();
    }

    Chunk& chunk() { return *app().document().active_chunk(); }
    ExportController& exports() { return window().exports(); }
    /// Whether the action with this objectName ("action.export.model") is enabled. Going
    /// through the window means a missing action fails the test instead of dereferencing null.
    bool enabled(const char* object_name) { return action(object_name)->isEnabled(); }
    void refresh() { exports().update_enabled(); }

    struct Submitted {
        std::string task;
        Params params;
        std::int32_t chunk_id = -2;
    };

    std::int32_t chunk_id_ = -1;
    std::vector<Submitted> submitted_;
    std::vector<std::pair<QString, QString>> errors_;
    std::shared_ptr<Gate> hold_release_ = std::make_shared<Gate>();
    /// A member, not a local: the job manager keeps the pointer until TearDown restores the
    /// built-in registry.
    TaskRegistry hold_registry_;
    bool ready_ = false;
};

void give_model(Chunk& chunk) {
    Mesh& mesh = *chunk.add_model().data;
    mesh.vertices = {Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0),
                     Eigen::Vector3d(0.0, 1.0, 0.0)};
    mesh.faces = {Eigen::Vector3i(0, 1, 2)};
}

void give_point_cloud(Chunk& chunk) {
    PointCloud& cloud = chunk.point_cloud();
    cloud.points = {Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0)};
    cloud.normals = {Eigen::Vector3f(0.0F, 0.0F, 1.0F), Eigen::Vector3f(0.0F, 0.0F, 1.0F)};
}

void give_elevation(Chunk& chunk) {
    Elevation& dem = *chunk.add_elevation().data;
    dem.raster.grid.width = 2;
    dem.raster.grid.height = 2;
    dem.raster.grid.pixel_size_x = 1.0;
    dem.raster.grid.pixel_size_y = 1.0;
    dem.raster.channels = 1;
    dem.raster.data.assign(4, 1.0F);
}

void give_orthomosaic(Chunk& chunk) {
    Orthomosaic& ortho = *chunk.add_orthomosaic().data;
    ortho.raster.grid.width = 2;
    ortho.raster.grid.height = 2;
    ortho.raster.grid.pixel_size_x = 1.0;
    ortho.raster.grid.pixel_size_y = 1.0;
    ortho.raster.channels = 4;
    ortho.raster.data.assign(16, static_cast<std::uint8_t>(200));
    for (std::size_t i = 3; i < ortho.raster.data.size(); i += 4) {
        ortho.raster.data[i] = 255;  // alpha: every cell observed
    }
}

/// A project on disk, so the export lands in this test's own scratch directory. The chunk is
/// made active first: the exporters address the ACTIVE chunk, so a save that left no active id
/// would load a document with nothing to export.
QString save_project(const QString& path) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    ProjectIO::save(document, path.toStdString());
    return path;
}

/// Accept the real TaskDialog as soon as it is up, and count how often that happened. A dialog
/// that refuses to accept (an invalid preset would be a bug) is rejected after a short grace
/// period, so a broken preset fails the assertions instead of hanging the test.
QTimer* accept_dialogs(QWidget* owner, std::shared_ptr<int> accepted) {
    auto* timer = new QTimer(owner);
    timer->setInterval(5);
    auto attempts = std::make_shared<int>(0);
    QObject::connect(timer, &QTimer::timeout, timer, [accepted, attempts]() {
        for (QWidget* widget : QApplication::topLevelWidgets()) {
            auto* dialog = qobject_cast<QDialog*>(widget);
            if (dialog != nullptr && dialog->objectName() == QStringLiteral("dialog.task") &&
                dialog->isVisible()) {
                if (*attempts < 100) {
                    ++*attempts;
                    dialog->accept();
                    ++*accepted;
                } else {
                    dialog->reject();
                }
            }
        }
    });
    timer->start();
    return timer;
}

}  // namespace

TEST_F(ExportTest, MenuListsAllFourExports) {
    QMenu* export_menu = menu("menu.file.export");
    ASSERT_NE(export_menu, nullptr);
    QStringList names;
    for (QAction* entry : export_menu->actions()) {
        if (!entry->isSeparator()) {
            names << entry->objectName();
        }
    }
    EXPECT_EQ(names, (QStringList{"action.export.model", "action.export.point_cloud",
                                  "action.export.dem", "action.export.orthomosaic"}));
    for (const ExportStep& step : ExportController::steps()) {
        EXPECT_NE(builtin_registry().find(step.task), nullptr) << step.task;
    }
}

TEST_F(ExportTest, ActionsAreDisabledForAMissingAssetAndSayWhy) {
    for (const char* name : {"action.export.model", "action.export.point_cloud",
                             "action.export.dem", "action.export.orthomosaic"}) {
        QAction* entry = action(name);
        ASSERT_NE(entry, nullptr);
        EXPECT_FALSE(entry->isEnabled()) << name;
        EXPECT_TRUE(entry->toolTip().contains(QStringLiteral("no"))) << name;
    }
    EXPECT_EQ(exports().unavailable_reason("ExportModel"),
              QStringLiteral("This project has no model to export."));

    give_model(chunk());
    refresh();
    EXPECT_TRUE(enabled("action.export.model"));
    EXPECT_FALSE(enabled("action.export.point_cloud"));
    EXPECT_TRUE(exports().unavailable_reason("ExportModel").isEmpty());

    give_point_cloud(chunk());
    give_elevation(chunk());
    give_orthomosaic(chunk());
    refresh();
    for (const char* name : {"action.export.model", "action.export.point_cloud",
                             "action.export.dem", "action.export.orthomosaic"}) {
        EXPECT_TRUE(action(name)->isEnabled()) << name;
    }
    for (const ExportStep& step : ExportController::steps()) {
        EXPECT_TRUE(exports().unavailable_reason(step.task).isEmpty()) << step.task;
    }
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(ExportTest, AnEmptyAssetDoesNotCount) {
    // add_model() creates a slot with an empty payload; that is not a model to export.
    chunk().add_model();
    chunk().add_elevation();
    chunk().add_orthomosaic();
    refresh();
    EXPECT_FALSE(enabled("action.export.model"));
    EXPECT_FALSE(enabled("action.export.dem"));
    EXPECT_FALSE(enabled("action.export.orthomosaic"));
}

TEST_F(ExportTest, NothingIsEnabledWithoutAnActiveChunk) {
    app().document().set_active_chunk_id(-1);
    refresh();
    for (const ExportStep& step : ExportController::steps()) {
        EXPECT_FALSE(exports().action(step.task)->isEnabled()) << step.task;
        EXPECT_EQ(exports().unavailable_reason(step.task),
                  QStringLiteral("There is no active chunk."));
    }
}

TEST_F(ExportTest, AnExportTheMenuDoesNotOfferCannotRun) {
    EXPECT_EQ(exports().action("BuildModel"), nullptr);
    EXPECT_EQ(exports().unavailable_reason("BuildModel"),
              QStringLiteral("This export is not available."));
    EXPECT_FALSE(exports().run("BuildModel"));
    EXPECT_FALSE(exports().run("NoSuchTask"));
    EXPECT_TRUE(submitted_.empty());
    EXPECT_TRUE(errors_.empty());
}

TEST_F(ExportTest, TriggeringSubmitsTheTaskWithASuggestedPath) {
    give_model(chunk());
    refresh();
    std::vector<std::string> asked;
    Presets seen;
    exports().set_params_handler([&](const std::string& task, const Presets& presets)
                                     -> std::optional<Params> {
        asked.push_back(task);
        seen = presets;
        // The params handler owns applying the presets (the default one does it on the real
        // dialog); here it is the only thing that fills the required path.
        Params params = make_params(builtin_registry().require(task));
        for (const auto& [name, value] : presets) {
            params.set(name, value);
        }
        return params;
    });

    ASSERT_TRUE(trigger("action.export.model"));
    ASSERT_EQ(asked, (std::vector<std::string>{"ExportModel"}));
    ASSERT_EQ(submitted_.size(), 1U);
    EXPECT_EQ(submitted_[0].task, "ExportModel");
    EXPECT_EQ(submitted_[0].chunk_id, chunk_id_);
    // The destination is preset, so accepting the dialog writes a named file without typing.
    const std::string path = submitted_[0].params.get_string("path");
    EXPECT_FALSE(path.empty());
    EXPECT_TRUE(QFileInfo(QString::fromStdString(path)).fileName().endsWith(".ply")) << path;
    EXPECT_EQ(submitted_[0].params.get_string("format"), "ply");
    ASSERT_EQ(seen.size(), 2U);
    EXPECT_EQ(seen[0].first, "path");
    EXPECT_EQ(seen[1].first, "format");
    EXPECT_TRUE(errors_.empty());
}

TEST_F(ExportTest, ADestinationWithoutAnExtensionGetsTheTasksOwn) {
    give_point_cloud(chunk());
    refresh();
    exports().set_params_handler(
        [](const std::string& task, const Presets&) -> std::optional<Params> {
            Params params = make_params(builtin_registry().require(task));
            params.set("path", std::string("out/cloud"));
            return params;
        });
    ASSERT_TRUE(trigger("action.export.point_cloud"));
    ASSERT_EQ(submitted_.size(), 1U);
    EXPECT_EQ(submitted_[0].params.get_string("path"), "out/cloud.ply");
}

TEST_F(ExportTest, ACancelledDialogSubmitsNothing) {
    give_model(chunk());
    refresh();
    exports().set_params_handler([](const std::string&, const Presets&)
                                     -> std::optional<Params> { return std::nullopt; });
    ASSERT_TRUE(trigger("action.export.model"));
    EXPECT_TRUE(submitted_.empty());
    EXPECT_TRUE(errors_.empty());
}

TEST_F(ExportTest, ActionsAreDisabledWhileAJobRuns) {
    give_model(chunk());
    refresh();
    ASSERT_TRUE(enabled("action.export.model"));

    auto started = std::make_shared<std::atomic<bool>>(false);
    hold_registry_.register_task(std::make_shared<HoldTask>(hold_release_, started));
    app().jobs().set_registry(&hold_registry_);
    app().jobs().submit("HoldTask", Params{}, chunk_id_);
    ASSERT_TRUE(pump_until([&]() { return started->load(); }, kPumpTimeoutMs));

    for (const ExportStep& step : ExportController::steps()) {
        EXPECT_FALSE(exports().action(step.task)->isEnabled()) << step.task;
        EXPECT_EQ(exports().action(step.task)->toolTip(), QStringLiteral("A job is running."));
    }
    EXPECT_EQ(exports().unavailable_reason("ExportModel"), QStringLiteral("A job is running."));
    EXPECT_FALSE(exports().run("ExportModel"));
    EXPECT_TRUE(submitted_.empty());

    hold_release_->open();
    ASSERT_TRUE(pump_until([&]() { return !app().jobs().busy(); }, kPumpTimeoutMs));
    app().jobs().set_registry(&builtin_registry());
    // busy_changed drives update_enabled(), so the menu is enabled again by itself.
    EXPECT_TRUE(enabled("action.export.model"));
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(ExportTest, TheRealDialogShowsTheSuggestedPathAndSubmitsIt) {
    const QString project = save_project(scratch("site.psx"));
    ASSERT_TRUE(window().open_project(project));
    ASSERT_EQ(app().document().workspace().chunks().size(), 1U);
    const std::int32_t opened_chunk = app().document().active_chunk_id();
    ASSERT_GE(opened_chunk, 0);
    give_model(*app().document().active_chunk());
    refresh();
    ASSERT_TRUE(enabled("action.export.model"));

    std::optional<Params> submitted_params;
    exports().set_submit_handler([&](const std::string& task, Params params,
                                     std::int32_t target) -> std::optional<quint64> {
        EXPECT_EQ(task, "ExportModel");
        EXPECT_EQ(target, opened_chunk);
        submitted_params = std::move(params);
        return std::nullopt;
    });
    // No params handler stub: a timer accepts the real dialog as soon as it is up. This also
    // proves the preset destination passes the dialog's own required-path check.
    auto accepted = std::make_shared<int>(0);
    QTimer* timer = accept_dialogs(&window(), accepted);
    const bool ran = trigger("action.export.model");
    timer->stop();
    timer->deleteLater();
    ASSERT_TRUE(ran);
    EXPECT_EQ(*accepted, 1) << "the real dialog was shown exactly once";
    ASSERT_TRUE(submitted_params.has_value())
        << "the real dialog was accepted and its parameters submitted";
    // The dialog opened on the preset destination in the project's own folder.
    const QString path = QString::fromStdString(submitted_params->get_string("path"));
    EXPECT_EQ(QFileInfo(path).absolutePath(), QFileInfo(project).absolutePath());
    EXPECT_EQ(QFileInfo(path).fileName(), QStringLiteral("site_model.ply"));
    EXPECT_EQ(submitted_params->get_string("format"), "ply");
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(ExportTest, ExportRunsThroughTheJobManagerAndWritesTheFile) {
    const QString project = save_project(scratch("written.psx"));
    ASSERT_TRUE(window().open_project(project));
    give_model(*app().document().active_chunk());
    refresh();
    // The production submit path: whatever the dialog accepted goes to the JobManager.
    exports().set_submit_handler([this](const std::string& task, Params params,
                                        std::int32_t target) -> std::optional<quint64> {
        return app().jobs().submit(task, std::move(params), target);
    });

    auto state = std::make_shared<int>(-1);
    QObject::connect(&app().jobs(), &JobManager::job_finished, &app(),
                     [state](quint64, JobManager::State result, const QString&) {
                         *state = static_cast<int>(result);
                     });
    auto written = std::make_shared<std::optional<QString>>();
    QObject::connect(&window().exports(), &ExportController::export_written, &window(),
                     [written](const QString&, const QString& path) { *written = path; });

    // The whole chain: menu action -> real dialog (accepted by the timer) -> JobManager ->
    // ExportModel -> PLY on disk -> status note with Reveal.
    auto accepted = std::make_shared<int>(0);
    QTimer* timer = accept_dialogs(&window(), accepted);
    const bool ran = trigger("action.export.model");
    timer->stop();
    timer->deleteLater();
    ASSERT_TRUE(ran);
    ASSERT_TRUE(pump_until([&]() { return *state >= 0; }, kPumpTimeoutMs));
    EXPECT_EQ(*state, static_cast<int>(JobManager::State::Succeeded))
        << (errors_.empty() ? std::string() : errors_[0].second.toStdString());
    ASSERT_TRUE(written->has_value()) << "the controller reported the file it wrote";
    const QString path = **written;
    EXPECT_TRUE(QFileInfo::exists(path)) << path.toStdString();
    EXPECT_GT(QFileInfo(path).size(), 0);
    EXPECT_EQ(QFileInfo(path).fileName(), QStringLiteral("written_model.ply"));

    // The status strip says what happened; the Reveal action is there to show the folder.
    EXPECT_TRUE(window().statusBar()->currentMessage().contains(QStringLiteral("Exported")));
    auto* reveal = action("action.export.reveal");
    ASSERT_NE(reveal, nullptr);
    EXPECT_EQ(reveal->text(), QStringLiteral("Reveal"));
    EXPECT_FALSE(reveal->toolTip().isEmpty());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(ExportTest, TheRealDialogOffersTheTasksFormatsAsAFilter) {
    give_model(chunk());
    refresh();
    // Capture the dialog while it is up, instead of accepting it blindly.
    QString filter;
    QString suggested;
    auto timer = std::make_shared<QTimer>();
    timer->setInterval(5);
    QObject::connect(timer.get(), &QTimer::timeout, timer.get(), [&]() {
        for (QWidget* widget : QApplication::topLevelWidgets()) {
            auto* dialog = qobject_cast<TaskDialog*>(widget);
            if (dialog == nullptr || !dialog->isVisible()) {
                continue;
            }
            filter = dialog->path_filter("path");
            suggested = dialog->editor("path") == nullptr
                            ? QString()
                            : qobject_cast<QLineEdit*>(dialog->editor("path"))->text();
            dialog->reject();
        }
    });
    timer->start();
    ASSERT_TRUE(trigger("action.export.model"));
    timer->stop();
    EXPECT_TRUE(filter.contains(QStringLiteral("*.ply"))) << filter.toStdString();
    EXPECT_TRUE(filter.contains(QStringLiteral("All files"))) << filter.toStdString();
    EXPECT_TRUE(suggested.endsWith(QStringLiteral("Untitled_model.ply")))
        << suggested.toStdString();
    EXPECT_TRUE(submitted_.empty()) << "the dialog was cancelled";
}

TEST_F(ExportTest, SuggestPathUsesTheProjectFolderOrDocuments) {
    const ExportStep step = ExportController::find_step("ExportDem").value();
    const QString beside = ExportController::suggest_path(QStringLiteral("/tmp/site.psx"),
                                                          QStringLiteral("site"), step);
    EXPECT_EQ(beside, QStringLiteral("/tmp/site_dem.tif"));

    // A project that was never saved has no folder; the Documents location is the fallback.
    const QString unsaved =
        ExportController::suggest_path(QString(), QStringLiteral("draft"), step);
    EXPECT_EQ(unsaved, QDir(ExportController::default_directory())
                           .filePath(QStringLiteral("draft_dem.tif")));
    EXPECT_FALSE(ExportController::default_directory().isEmpty());
    // A task this menu does not offer has no step.
    EXPECT_FALSE(ExportController::find_step("BuildModel").has_value());
}

TEST_F(ExportTest, ASubmitFailureIsReportedNotThrown) {
    give_model(chunk());
    refresh();
    exports().set_params_handler(
        [](const std::string& task, const Presets&) -> std::optional<Params> {
            return make_params(builtin_registry().require(task));
        });
    exports().set_submit_handler([](const std::string&, Params, std::int32_t)
                                     -> std::optional<quint64> { throw Error("shutting down"); });
    ASSERT_TRUE(trigger("action.export.model"));
    ASSERT_EQ(errors_.size(), 1U);
    EXPECT_EQ(errors_[0].second, QStringLiteral("shutting down"));
    // A failed submission leaves the menu usable.
    EXPECT_TRUE(enabled("action.export.model"));
}
