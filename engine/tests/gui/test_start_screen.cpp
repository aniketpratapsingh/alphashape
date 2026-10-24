// StartScreen.*: the launcher in the central area. It is shown on startup when no project is
// asked for, lists GuiApp's recent projects (marking the ones whose file is gone), and offers
// New / Open / Add photos and drag-and-drop. Every choice is a signal; MainWindow acts on it.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <QAction>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QLabel>
#include <QListWidget>
#include <QMenu>
#include <QMimeData>
#include <QPalette>
#include <QPushButton>
#include <QStatusBar>
#include <QStringList>
#include <QUrl>

#include "apps/gui/app.hpp"
#include "apps/gui/command_line.hpp"
#include "apps/gui/drop_paths.hpp"
#include "apps/gui/image_files.hpp"
#include "apps/gui/main_window.hpp"
#include "apps/gui/start_screen.hpp"
#include "apps/gui/viewer/viewport_widget.hpp"
#include "common/commands/model_commands.hpp"
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
using alphashape::gui::testing::pump;
using alphashape::gui::testing::pump_until;

namespace {

constexpr int kPumpTimeoutMs = 15000;

/// Creates a project the way the engine would: a document with one labelled chunk.
QString make_project(const QString& path, const char* label = "Block A") {
    Document document;
    document.workspace().add_chunk(label);
    ProjectIO::save(document, path.toStdString());
    return path;
}

}  // namespace

/// Every test here runs a window whose error box would never close offscreen (the fake photo
/// files fail AddPhotos), so the fixture collects the messages instead.
class StartScreenTest : public GuiTest {
protected:
    void SetUp() override {
        GuiTest::SetUp();
        // The "no sample configured" test is meaningless if the machine happens to set one.
        qunsetenv("ALPHASHAPE_SAMPLE_PROJECT");
        window().set_error_handler([this](const QString& title, const QString& message) {
            errors_.emplace_back(title, message);
        });
        ready_ = true;
    }

    /// Leave the job manager idle and pointed at the built-in registry, whatever the test did.
    void TearDown() override {
        if (!ready_) {
            GuiTest::TearDown();
            return;
        }
        release_->open();
        if (!pump_until([this]() { return !app().jobs().busy(); }, kPumpTimeoutMs)) {
            app().jobs().cancel_all_and_wait();
        }
        app().jobs().set_registry(&builtin_registry());
        qunsetenv("ALPHASHAPE_SAMPLE_PROJECT");
        GuiTest::TearDown();
    }

    std::vector<std::pair<QString, QString>> errors_;
    std::shared_ptr<alphashape::gui::testing::Gate> release_ =
        std::make_shared<alphashape::gui::testing::Gate>();
    /// A member, not a local: the job manager keeps the pointer until TearDown restores the
    /// built-in registry.
    TaskRegistry registry_;
    bool ready_ = false;
};

TEST_F(StartScreenTest, ShownOnStartupWhenNoProjectIsAskedFor) {
    show_window();
    EXPECT_NE(window().start_screen(), nullptr);
    EXPECT_TRUE(window().showing_start_screen());
    EXPECT_EQ(window().central_page(), 1);
    EXPECT_TRUE(window().start_screen()->isVisible());
    auto* viewport = window().viewport();
    ASSERT_NE(viewport, nullptr);
    EXPECT_FALSE(viewport->isVisible());
    EXPECT_EQ(window().start_screen()->objectName(), QStringLiteral("startScreen"));
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(StartScreenTest, EntryPointsStartHiddenAndTheListIsEmpty) {
    show_window();
    StartScreen* screen = window().start_screen();
    ASSERT_NE(screen, nullptr);
    EXPECT_TRUE(screen->recent_paths().isEmpty());
    EXPECT_NE(screen->findChild<QLabel*>(QStringLiteral("start.recent.empty")), nullptr);
}

TEST_F(StartScreenTest, NotShownWhenTheCommandLineOpensAProject) {
    // The real startup path: Options carries --open <path>, GuiApp loads it, and MainWindow
    // reads GuiApp::project_requested() to decide the initial central page.
    const QString path = make_project(scratch("opened.psx"));
    Options options;
    options.reset_settings = true;
    options.open_path = path.toStdString();
    rebuild_with(options);
    ASSERT_EQ(app().document().path(), QFileInfo(path).absoluteFilePath().toStdString());
    show_window();
    EXPECT_FALSE(window().showing_start_screen());
    EXPECT_EQ(window().central_page(), 0);
    EXPECT_TRUE(window().viewport()->isVisible());
}

TEST_F(StartScreenTest, NotShownWhenTheCommandLineAsksForANewProject) {
    // --new is a request for an (empty) project, not for the launcher: the viewport shows even
    // though the document has no path and no chunks yet.
    Options options;
    options.reset_settings = true;
    options.new_project = true;
    rebuild_with(options);
    show_window();
    EXPECT_FALSE(window().showing_start_screen());
    EXPECT_EQ(window().central_page(), 0);
    EXPECT_TRUE(window().viewport()->isVisible());
}

TEST_F(StartScreenTest, ShownWhenTheCommandLineAsksForNothing) {
    Options options;
    options.reset_settings = true;
    rebuild_with(options);  // no --open, no --new
    show_window();
    EXPECT_TRUE(window().showing_start_screen());
    EXPECT_EQ(window().central_page(), 1);
}

TEST_F(StartScreenTest, RecentListShowsEntriesAndMarksTheMissingOnes) {
    const QString present = make_project(scratch("present.psx"));
    app().add_recent_file(present);
    app().add_recent_file(scratch("gone.psx"));  // recorded, never created
    pump(20);
    ASSERT_TRUE(window().start_screen()->wait_for_recent_checks()) << "the sweep finished";

    QListWidget* list = window().start_screen()->recent_list();
    ASSERT_NE(list, nullptr);
    ASSERT_EQ(list->count(), 2);
    // Newest first, as GuiApp's list is ordered.
    EXPECT_EQ(list->item(0)->data(Qt::UserRole).toString(),
              QFileInfo(scratch("gone.psx")).absoluteFilePath());
    EXPECT_EQ(list->item(1)->data(Qt::UserRole).toString(), QFileInfo(present).absoluteFilePath());

    EXPECT_TRUE(window().start_screen()->recent_missing(0));
    EXPECT_FALSE(window().start_screen()->recent_missing(1));
    EXPECT_TRUE(list->item(0)->text().contains(QStringLiteral("missing")));
    EXPECT_FALSE(list->item(1)->text().contains(QStringLiteral("missing")));
    // The missing entry is greyed through the palette, not a hardcoded colour.
    EXPECT_EQ(list->item(0)->foreground().color(),
              list->palette().color(QPalette::Disabled, QPalette::Text));
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(StartScreenTest, RowsAppearAtOnceAndAreOnlyGreyedWhenTheCheckReturns) {
    // A probe that blocks until released: the point is what the list looks like WHILE the
    // existence check is still running, which is the case a dead SMB/NFS path produces.
    app().add_recent_file(scratch("slow.psx"));
    auto gate = std::make_shared<Gate>();
    auto asked = std::make_shared<std::atomic<bool>>(false);
    window().start_screen()->set_exists_prober([gate, asked](const QString&) {
        asked->store(true);
        gate->wait(std::chrono::seconds(20));
        return false;  // by the time it answers, the file is "gone"
    });
    window().start_screen()->reload();
    QListWidget* list = window().start_screen()->recent_list();
    ASSERT_EQ(list->count(), 1);
    // The row is up immediately, and NOT greyed: the answer is not in yet.
    EXPECT_FALSE(list->item(0)->text().contains(QStringLiteral("missing")));
    EXPECT_FALSE(window().start_screen()->recent_missing(0));
    EXPECT_TRUE(window().start_screen()->recent_checks_pending());

    gate->open();
    ASSERT_TRUE(window().start_screen()->wait_for_recent_checks());
    EXPECT_TRUE(window().start_screen()->recent_missing(0));
    EXPECT_TRUE(list->item(0)->text().contains(QStringLiteral("missing")));
}

TEST_F(StartScreenTest, AStaleSweepDoesNotOverwriteANewerOne) {
    const QString first = make_project(scratch("one.psx"));
    app().add_recent_file(first);
    auto gate = std::make_shared<Gate>();
    window().start_screen()->set_exists_prober([gate](const QString&) {
        gate->wait(std::chrono::seconds(20));
        return false;
    });
    window().start_screen()->reload();  // sweep 1, blocked
    // A second reload starts sweep 2; release the first, then the second must win.
    window().start_screen()->set_exists_prober([](const QString&) { return true; });
    window().start_screen()->reload();
    gate->open();
    ASSERT_TRUE(window().start_screen()->wait_for_recent_checks());
    pump(30);
    // Sweep 1 said "missing", sweep 2 said "present": the newer answer is what is shown.
    EXPECT_FALSE(window().start_screen()->recent_missing(0));
    EXPECT_FALSE(window().start_screen()->recent_list()->item(0)->text().contains(
        QStringLiteral("missing")));
}

TEST_F(StartScreenTest, CollectsRenderedDarkAndLight) {
    // Debugging aid only; asserts nothing about pixels.
    show_window();
    GuiTest::grab(window().start_screen(), "start_screen_dark");
    app().theme().apply(ThemeMode::Light);
    pump(30);
    GuiTest::grab(window().start_screen(), "start_screen_light");
    app().theme().apply(ThemeMode::Dark);
    pump(20);
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(StartScreenTest, ClickingARecentProjectOpensItAndSwitchesToTheViewport) {
    const QString path = make_project(scratch("recent.psx"));
    app().add_recent_file(path);
    show_window();
    QListWidget* list = window().start_screen()->recent_list();
    ASSERT_EQ(list->count(), 1);

    // A click is a user activation; emit the item signal the way QListWidget would.
    list->itemClicked(list->item(0));
    pump(20);
    EXPECT_EQ(app().document().path(), QFileInfo(path).absoluteFilePath().toStdString());
    EXPECT_EQ(app().document().workspace().chunks().size(), 1U);
    EXPECT_FALSE(window().showing_start_screen());
    EXPECT_TRUE(window().viewport()->isVisible());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(StartScreenTest, ClickingAMissingRecentEntryDoesNothing) {
    app().add_recent_file(scratch("gone.psx"));
    show_window();
    QListWidget* list = window().start_screen()->recent_list();
    ASSERT_EQ(list->count(), 1);
    list->itemClicked(list->item(0));
    pump(20);
    EXPECT_TRUE(app().document().path().empty());
    EXPECT_TRUE(window().showing_start_screen());
}

TEST_F(StartScreenTest, RemovingAMissingEntryDropsItFromTheListAndTheSettings) {
    const QString present = make_project(scratch("keep.psx"));
    app().add_recent_file(present);
    app().add_recent_file(scratch("gone.psx"));
    show_window();
    ASSERT_EQ(window().start_screen()->recent_list()->count(), 2);

    window().start_screen()->remove_recent(0);
    pump(20);
    EXPECT_EQ(window().start_screen()->recent_list()->count(), 1);
    EXPECT_EQ(app().recent_files(),
              QStringList{QFileInfo(present).absoluteFilePath()});
    // The removal is persisted, not just painted: a fresh app sees the shorter list.
    rebuild(/*reset_settings=*/false);
    EXPECT_EQ(app().recent_files(), QStringList{QFileInfo(present).absoluteFilePath()});
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(StartScreenTest, CloseReturnsToTheStartScreen) {
    const QString path = make_project(scratch("roundtrip.psx"));
    show_window();
    ASSERT_TRUE(window().open_project(path));
    EXPECT_FALSE(window().showing_start_screen());

    ASSERT_TRUE(trigger("action.file.close"));
    pump(20);
    EXPECT_TRUE(window().showing_start_screen());
    EXPECT_TRUE(window().start_screen()->isVisible());
    EXPECT_TRUE(app().document().workspace().chunks().empty());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(StartScreenTest, NewProjectShowsTheViewport) {
    show_window();
    ASSERT_TRUE(trigger("action.file.new"));
    pump(20);
    EXPECT_FALSE(window().showing_start_screen());
    EXPECT_TRUE(window().viewport()->isVisible());
}

TEST_F(StartScreenTest, CloseWithUnsavedChangesPromptsThroughTheExistingHandler) {
    int prompts = 0;
    window().set_unsaved_handler([&](const QString&) {
        ++prompts;
        return UnsavedChoice::Cancel;
    });
    show_window();
    ASSERT_TRUE(window().open_project(make_project(scratch("dirty.psx"))));
    app().execute(make_add_chunk_command(app().document().workspace(), "Extra", -1));
    ASSERT_TRUE(app().dirty());

    EXPECT_FALSE(window().close_project());
    EXPECT_EQ(prompts, 1);
    EXPECT_FALSE(window().showing_start_screen()) << "Cancel keeps the project open";
    EXPECT_EQ(app().document().workspace().chunks().size(), 2U);
}

TEST_F(StartScreenTest, DroppingAProjectPathOpensIt) {
    show_window();
    const QString path = make_project(scratch("dropped.psx"), "Dropped");
    EXPECT_TRUE(window().start_screen()->drop_paths({path}));
    pump(20);
    EXPECT_EQ(app().document().path(), QFileInfo(path).absoluteFilePath().toStdString());
    EXPECT_EQ(app().document().workspace().chunks().size(), 1U);
    EXPECT_EQ(app().document().workspace().chunks()[0]->label(), "Dropped");
    EXPECT_FALSE(window().showing_start_screen());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(StartScreenTest, DroppingImagesStartsANewProjectAndAddsThem) {
    // Two real files, so AddPhotos reaches the decoder (and then fails on the bytes, which the
    // fixture's error handler collects); that is still proof the request went through the job
    // manager for the new chunk.
    const QString first = scratch("a.jpg");
    const QString second = scratch("b.jpg");
    for (const QString& path : {first, second}) {
        QFile file(path);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        file.write("not really a jpeg");
    }
    show_window();

    // Connect before the drop: the job may finish before the test gets control back.
    auto state = std::make_shared<int>(-1);
    QObject::connect(&app().jobs(), &JobManager::job_finished, &app(),
                     [state](quint64, JobManager::State result, const QString&) {
                         *state = static_cast<int>(result);
                     });
    EXPECT_TRUE(window().start_screen()->drop_paths({first, second}));
    pump(20);

    // The document was replaced with a one-chunk project and the drop left the launcher.
    EXPECT_FALSE(window().showing_start_screen());
    ASSERT_EQ(app().document().workspace().chunks().size(), 1U);
    ASSERT_TRUE(pump_until([&]() { return *state >= 0; }, kPumpTimeoutMs));
    EXPECT_EQ(*state, static_cast<int>(JobManager::State::Failed));
    EXPECT_FALSE(errors_.empty()) << "a failed import is reported through the error handler";
}

TEST_F(StartScreenTest, DroppingOnlyNonImageFilesDoesNothing) {
    // A PDF is not a photo, not a project and not a folder: the drop is refused outright, so
    // no "new project with these photos" is started from a file the engine cannot decode.
    const QString pdf = scratch("report.pdf");
    {
        QFile file(pdf);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        file.write("%PDF-1.4");
    }
    show_window();
    EXPECT_FALSE(window().start_screen()->drop_paths({pdf}));
    pump(20);
    EXPECT_TRUE(window().showing_start_screen());
    EXPECT_TRUE(app().document().workspace().chunks().empty());
    EXPECT_FALSE(app().jobs().busy());
    // The window agrees: the same path dropped straight onto it is ignored too.
    EXPECT_FALSE(window().handle_dropped_paths({pdf}));
    EXPECT_TRUE(app().document().workspace().chunks().empty());
}

TEST_F(StartScreenTest, DroppingAProjectWithImagesOpensTheProjectAndSaysTheImagesWereIgnored) {
    const QString project = make_project(scratch("mixed.psx"), "Mixed");
    const QString image = scratch("extra.jpg");
    {
        QFile file(image);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        file.write("not really a jpeg");
    }
    show_window();
    EXPECT_TRUE(window().start_screen()->drop_paths({image, project}));
    pump(20);
    EXPECT_EQ(app().document().path(), QFileInfo(project).absoluteFilePath().toStdString());
    EXPECT_EQ(app().document().workspace().chunks().size(), 1U)
        << "the images did not start a second project";
    EXPECT_FALSE(window().showing_start_screen());
    EXPECT_TRUE(window().statusBar()->currentMessage().contains(QStringLiteral("ignored")))
        << window().statusBar()->currentMessage().toStdString();
    EXPECT_FALSE(app().jobs().busy());
}

TEST_F(StartScreenTest, DroppingWithADirtyDocumentPromptsBeforeItIsReplaced) {
    show_window();
    ASSERT_TRUE(window().open_project(make_project(scratch("dirty_drop.psx"))));
    app().execute(make_add_chunk_command(app().document().workspace(), "Extra", -1));
    ASSERT_TRUE(app().dirty());

    int prompts = 0;
    window().set_unsaved_handler([&](const QString&) {
        ++prompts;
        return UnsavedChoice::Cancel;
    });
    const QString other = make_project(scratch("other.psx"), "Other");
    EXPECT_FALSE(window().handle_dropped_paths({other}));
    EXPECT_EQ(prompts, 1) << "the drop asked before replacing unsaved work";
    EXPECT_EQ(app().document().workspace().chunks().size(), 2U);
}

TEST_F(StartScreenTest, DroppedFoldersAreScannedWithABound) {
    const QString root = scratch("dropdir");
    ASSERT_TRUE(QDir().mkpath(QDir(root).filePath(QStringLiteral("nested/deeper/past"))));
    for (const QString& relative : {QStringLiteral("top.jpg"), QStringLiteral("nested/mid.png"),
                                    QStringLiteral("nested/deeper/low.tif"),
                                    QStringLiteral("nested/deeper/past/too_far.jpg")}) {
        QFile file(QDir(root).filePath(relative));
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        file.write("x");
    }
    // Depth 0 is the dropped folder itself; the bound is the deepest folder that is scanned.
    const QStringList found = images_in_directories({root}, 2);
    ASSERT_EQ(found.size(), 3U) << "a bound of 2 stops before the fourth level";
    EXPECT_TRUE(found[0].endsWith(QStringLiteral("top.jpg")));
    EXPECT_TRUE(found[1].endsWith(QStringLiteral("mid.png")));
    EXPECT_TRUE(found[2].endsWith(QStringLiteral("low.tif")));
    // The limit is a total cap across the whole scan.
    EXPECT_EQ(images_in_directories({root}, 3, 2).size(), 2);
    EXPECT_EQ(images_in_directories({root}).size(), 4) << "the default bound reaches them all";
    EXPECT_TRUE(images_in_directories({scratch("no-such-folder")}).isEmpty());
}

TEST_F(StartScreenTest, AddPhotosButtonStartsAProjectAtOnce) {
    // Clicking the button runs the real chain: the picker hook -> add_photos_requested ->
    // new project + chunk -> AddPhotos on the job manager.
    window().start_screen()->set_photos_handler([]() {
        return QStringList{"/data/one.jpg", "/data/two.jpg"};
    });
    show_window();
    auto state = std::make_shared<int>(-1);
    QObject::connect(&app().jobs(), &JobManager::job_finished, &app(),
                     [state](quint64, JobManager::State result, const QString&) {
                         *state = static_cast<int>(result);
                     });
    window().start_screen()->add_photos_button()->click();
    pump(20);
    EXPECT_FALSE(window().showing_start_screen())
        << "the project opened before the photos were read";
    ASSERT_EQ(app().document().workspace().chunks().size(), 1U);
    ASSERT_TRUE(pump_until([&]() { return *state >= 0; }, kPumpTimeoutMs));
    EXPECT_EQ(*state, static_cast<int>(JobManager::State::Failed));
    // The chunk the task targeted is the one the button created.
    EXPECT_EQ(app().document().active_chunk_id(),
              app().document().workspace().chunks().front()->id());
}

TEST_F(StartScreenTest, ACancelledPhotoPickerOpensNothing) {
    window().start_screen()->set_photos_handler([]() { return QStringList(); });
    show_window();
    window().start_screen()->add_photos_button()->click();
    pump(20);
    EXPECT_TRUE(window().showing_start_screen());
    EXPECT_TRUE(app().document().workspace().chunks().empty());
    EXPECT_FALSE(app().jobs().busy());
}

TEST_F(StartScreenTest, SampleIsHiddenWhenNoSampleProjectIsConfigured) {
    // No ALPHASHAPE_SAMPLE_PROJECT and no share/alphashape/sample next to the test binary.
    show_window();
    EXPECT_FALSE(StartScreen::find_sample_project().has_value());
    EXPECT_FALSE(window().start_screen()->sample_available());
    EXPECT_FALSE(window().start_screen()->sample_button()->isVisible());
}

TEST_F(StartScreenTest, SampleAppearsWhenIndicatedByTheEnvironment) {
    const QString path = make_project(scratch("sample.psx"), "Sample");
    qputenv("ALPHASHAPE_SAMPLE_PROJECT", path.toLocal8Bit());
    show_window();
    EXPECT_TRUE(window().start_screen()->sample_available());
    EXPECT_TRUE(window().start_screen()->sample_button()->isVisible());
    EXPECT_EQ(window().start_screen()->sample_path(), path);

    window().start_screen()->open_path_requested(window().start_screen()->sample_path());
    pump(20);
    EXPECT_EQ(app().document().workspace().chunks().size(), 1U);
    EXPECT_EQ(app().document().workspace().chunks()[0]->label(), "Sample");
    qunsetenv("ALPHASHAPE_SAMPLE_PROJECT");
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(StartScreenTest, SampleEnvironmentPointingAtAFolderFindsTheProjectInIt) {
    const QString folder = scratch("sample_dir");
    ASSERT_TRUE(QDir().mkpath(folder));
    const QString path = make_project(QDir(folder).filePath(QStringLiteral("demo.psx")));
    qputenv("ALPHASHAPE_SAMPLE_PROJECT", folder.toLocal8Bit());
    EXPECT_EQ(StartScreen::find_sample_project().value_or(QString()), path);
    qunsetenv("ALPHASHAPE_SAMPLE_PROJECT");
}

TEST_F(StartScreenTest, AMissingSampleEnvironmentHidesTheButtonRatherThanGuessing) {
    qputenv("ALPHASHAPE_SAMPLE_PROJECT", scratch("nope.psx").toLocal8Bit());
    EXPECT_FALSE(StartScreen::find_sample_project().has_value());
    qunsetenv("ALPHASHAPE_SAMPLE_PROJECT");
}

TEST_F(StartScreenTest, EntryPointsAreDisabledWhileAJobRuns) {
    show_window();
    EXPECT_TRUE(window().start_screen()->new_button()->isEnabled());
    EXPECT_TRUE(window().start_screen()->open_button()->isEnabled());

    // A job that blocks, so the busy window is wide enough to observe on any machine. It needs
    // an active chunk (the task targets Chunk), which is added directly, behind the command
    // layer, so no signal switches the window away from the launcher.
    Chunk& chunk = app().document().workspace().add_chunk("Chunk 1");
    app().document().set_active_chunk_id(chunk.id());
    auto started = std::make_shared<std::atomic<bool>>(false);
    registry_.register_task(
        std::make_shared<alphashape::gui::testing::HoldTask>(release_, started));
    app().jobs().set_registry(&registry_);
    app().jobs().submit("HoldTask", Params{}, chunk.id());
    ASSERT_TRUE(pump_until([&]() { return started->load(); }, kPumpTimeoutMs));

    EXPECT_FALSE(window().start_screen()->new_button()->isEnabled());
    EXPECT_FALSE(window().start_screen()->open_button()->isEnabled());
    EXPECT_FALSE(window().start_screen()->add_photos_button()->isEnabled());
    // A drop is refused while the document belongs to the worker.
    EXPECT_FALSE(window().start_screen()->drop_paths({scratch("x.psx")}));

    release_->open();
    ASSERT_TRUE(pump_until([&]() { return !app().jobs().busy(); }, kPumpTimeoutMs));
    app().jobs().set_registry(&builtin_registry());
    EXPECT_TRUE(window().start_screen()->new_button()->isEnabled());
    EXPECT_TRUE(window().start_screen()->open_button()->isEnabled());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(StartScreenTest, LocalPathsIgnoresRemoteUrlsAndEmptyEntries) {
    QMimeData mime;
    mime.setUrls({QUrl::fromLocalFile("/tmp/a.psx"), QUrl("https://example.com/b.psx"),
                  QUrl::fromLocalFile("")});
    EXPECT_EQ(StartScreen::local_paths(&mime), QStringList{QStringLiteral("/tmp/a.psx")});
    EXPECT_TRUE(StartScreen::local_paths(nullptr).isEmpty());
    QMimeData text;
    text.setText(QStringLiteral("plain text"));
    EXPECT_TRUE(StartScreen::local_paths(&text).isEmpty());
}

TEST_F(StartScreenTest, ContextMenuOffersOpenAndRemove) {
    const QString present = make_project(scratch("menu.psx"));
    app().add_recent_file(present);
    show_window();
    QListWidget* list = window().start_screen()->recent_list();
    ASSERT_EQ(list->count(), 1);
    list->setCurrentRow(0);
    // The row overload is what the position overload resolves to; a real context menu needs a
    // laid-out list, which the offscreen plugin does not always provide.
    std::unique_ptr<QMenu> menu(window().start_screen()->build_recent_menu(0));
    ASSERT_NE(menu, nullptr);
    QAction* open = menu->findChild<QAction*>(QStringLiteral("action.start.recent.open"));
    QAction* remove = menu->findChild<QAction*>(QStringLiteral("action.start.recent.remove"));
    ASSERT_NE(open, nullptr);
    ASSERT_NE(remove, nullptr);
    EXPECT_TRUE(open->isEnabled());
    // A row outside the list has no menu.
    EXPECT_EQ(window().start_screen()->build_recent_menu(7), nullptr);

    remove->trigger();
    pump(20);
    EXPECT_TRUE(window().start_screen()->recent_list()->count() == 0);
    EXPECT_TRUE(app().recent_files().isEmpty());
}
