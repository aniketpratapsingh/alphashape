// MainWindow.*: the window shell (menus, actions, docks, layout persistence, file operations,
// undo/redo wiring, unsaved-changes prompt), driven through the offscreen harness.

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include <QAction>
#include <QDir>
#include <QDockWidget>
#include <QFile>
#include <QFileInfo>
#include <QKeySequence>
#include <QLabel>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QRegularExpression>
#include <QStatusBar>
#include <QStringList>

#include "apps/gui/viewport_placeholder.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/document.hpp"
#include "common/io/project_io.hpp"
#include "harness.hpp"

using namespace alphashape;
using namespace alphashape::gui;
using alphashape::gui::testing::GuiTest;
using alphashape::gui::testing::pump;
using alphashape::gui::testing::pump_until;

namespace {

using MainWindowTest = GuiTest;

void add_chunk(GuiApp& app, const char* label) {
    app.execute(make_add_chunk_command(app.document().workspace(), label, -1));
}

// Creates a project the way the engine would: a document with two labelled chunks, saved
// through ProjectIO.
QString make_engine_project(const QString& path) {
    Document document;
    document.workspace().add_chunk("Block A");
    document.workspace().add_chunk("Block B");
    ProjectIO::save(document, path.toStdString());
    return path;
}

}  // namespace

TEST_F(MainWindowTest, ConstructsWithoutQtWarnings) {
    show_window();
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(MainWindowTest, EveryMenuAndActionExistsWithAStableName) {
    const QStringList menus = {
        "menu.file", "menu.file.recent", "menu.edit", "menu.view", "menu.view.theme",
        "menu.workflow", "menu.help",
    };
    for (const QString& name : menus) {
        EXPECT_NE(menu(name), nullptr) << name.toStdString();
    }
    const QStringList actions = {
        "action.file.new", "action.file.open", "action.file.recent.0",
        "action.file.recent.9", "action.file.recent.clear", "action.file.save",
        "action.file.save_as", "action.file.close", "action.file.quit", "action.edit.undo",
        "action.edit.redo", "action.edit.preferences", "action.view.dock.workspace",
        "action.view.dock.properties", "action.view.dock.photos", "action.view.dock.log",
        "action.view.theme.light", "action.view.theme.dark", "action.view.reset_layout",
        "action.workflow.add_photos", "action.workflow.build_orthomosaic", "action.help.about",
        "action.help.repository",
    };
    for (const QString& name : actions) {
        EXPECT_NE(action(name), nullptr) << name.toStdString();
    }

    // The menu bar holds exactly these menus, in this order.
    QStringList titles;
    for (QAction* entry : window().menuBar()->actions()) {
        titles << entry->text().remove(QLatin1Char('&'));
    }
    EXPECT_EQ(titles, (QStringList{"File", "Edit", "View", "Workflow", "Help"}));

    // Preferences is present but disabled for now.
    EXPECT_FALSE(action("action.edit.preferences")->isEnabled());

    // Standard shortcuts.
    EXPECT_EQ(action("action.file.new")->shortcut(), QKeySequence(QKeySequence::New));
    EXPECT_EQ(action("action.file.open")->shortcut(), QKeySequence(QKeySequence::Open));
    EXPECT_EQ(action("action.file.save")->shortcut(), QKeySequence(QKeySequence::Save));
    EXPECT_EQ(action("action.edit.undo")->shortcut(), QKeySequence(QKeySequence::Undo));
    EXPECT_EQ(action("action.edit.redo")->shortcut(), QKeySequence(QKeySequence::Redo));
}

TEST_F(MainWindowTest, DefaultDockLayout) {
    show_window();
    EXPECT_EQ(window().dockWidgetArea(dock("dock.workspace")), Qt::LeftDockWidgetArea);
    EXPECT_EQ(window().dockWidgetArea(dock("dock.properties")), Qt::RightDockWidgetArea);
    EXPECT_EQ(window().dockWidgetArea(dock("dock.photos")), Qt::BottomDockWidgetArea);
    EXPECT_EQ(window().dockWidgetArea(dock("dock.log")), Qt::BottomDockWidgetArea);
    EXPECT_TRUE(window().tabifiedDockWidgets(dock("dock.photos")).contains(dock("dock.log")));

    auto* viewport = dynamic_cast<ViewportPlaceholder*>(window().centralWidget());
    ASSERT_NE(viewport, nullptr);
    EXPECT_EQ(viewport->objectName(), QStringLiteral("viewport"));
    EXPECT_EQ(viewport->canvas_mode(), ThemeMode::Dark);

    EXPECT_NE(window().statusBar()->findChild<QLabel*>("status.jobs"), nullptr);
}

TEST_F(MainWindowTest, ViewportFollowsTheThemeSwitch) {
    show_window();
    ASSERT_TRUE(action("action.view.theme.dark")->isChecked());
    trigger("action.view.theme.light");
    auto* viewport = dynamic_cast<ViewportPlaceholder*>(window().centralWidget());
    ASSERT_NE(viewport, nullptr);
    EXPECT_EQ(app().theme().mode(), ThemeMode::Light);
    EXPECT_EQ(viewport->canvas_mode(), ThemeMode::Light);
    EXPECT_TRUE(action("action.view.theme.light")->isChecked());
    EXPECT_FALSE(action("action.view.theme.dark")->isChecked());
    trigger("action.view.theme.dark");
    EXPECT_EQ(viewport->canvas_mode(), ThemeMode::Dark);
    GuiTest::grab(&window(), "main_window_dark");
}

TEST_F(MainWindowTest, DockTogglesHideAndShowTheirDocks) {
    show_window();
    for (const char* name : {"workspace", "properties", "photos", "log"}) {
        QDockWidget* target = dock(QStringLiteral("dock.") + name);
        QAction* toggle = action(QStringLiteral("action.view.dock.") + name);
        ASSERT_NE(target, nullptr);
        ASSERT_NE(toggle, nullptr);
        EXPECT_TRUE(toggle->isCheckable());
        // Photos and Log share a tab strip; make the one under test the visible tab.
        target->raise();
        pump(20);
        ASSERT_TRUE(target->isVisible()) << name;
        EXPECT_TRUE(toggle->isChecked()) << name;

        toggle->trigger();
        pump(20);
        EXPECT_FALSE(target->isVisible()) << name;
        EXPECT_FALSE(toggle->isChecked()) << name;

        toggle->trigger();
        pump(20);
        EXPECT_TRUE(target->isVisible()) << name;
        EXPECT_TRUE(toggle->isChecked()) << name;
    }
}

TEST_F(MainWindowTest, LayoutSavesRestoresAndResets) {
    show_window();
    // Rearrange: Workspace to the right, Properties closed, Log closed.
    window().addDockWidget(Qt::RightDockWidgetArea, dock("dock.workspace"));
    dock("dock.properties")->hide();
    dock("dock.log")->hide();
    pump(20);
    window().save_layout();

    // A new window with the same settings comes back the way it was left.
    rebuild(/*reset_settings=*/false);
    show_window();
    EXPECT_EQ(window().dockWidgetArea(dock("dock.workspace")), Qt::RightDockWidgetArea);
    EXPECT_FALSE(dock("dock.properties")->isVisible());
    EXPECT_FALSE(dock("dock.log")->isVisible());
    EXPECT_TRUE(dock("dock.photos")->isVisible());

    // Reset layout brings back the default.
    trigger("action.view.reset_layout");
    pump(30);
    EXPECT_EQ(window().dockWidgetArea(dock("dock.workspace")), Qt::LeftDockWidgetArea);
    EXPECT_EQ(window().dockWidgetArea(dock("dock.properties")), Qt::RightDockWidgetArea);
    EXPECT_TRUE(dock("dock.workspace")->isVisible());
    EXPECT_TRUE(dock("dock.properties")->isVisible());
    EXPECT_FALSE(dock("dock.workspace")->isFloating());
    EXPECT_TRUE(window().tabifiedDockWidgets(dock("dock.photos")).contains(dock("dock.log")));
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(MainWindowTest, NewSaveAsCloseOpenRoundTrip) {
    const QString path = scratch("survey.psx");
    window().set_save_path_handler([&](const QString&) { return path; });
    window().set_open_path_handler([&]() { return path; });
    window().set_unsaved_handler([](const QString&) {
        ADD_FAILURE() << "no prompt expected on a clean document";
        return UnsavedChoice::Cancel;
    });

    ASSERT_TRUE(trigger("action.file.new"));
    EXPECT_TRUE(app().document().path().empty());
    add_chunk(app(), "Block A");
    add_chunk(app(), "Block B");
    EXPECT_TRUE(app().dirty());
    EXPECT_TRUE(window().isWindowModified());

    ASSERT_TRUE(trigger("action.file.save_as"));
    EXPECT_FALSE(app().dirty());
    EXPECT_FALSE(window().isWindowModified());
    EXPECT_TRUE(QFile::exists(path));
    EXPECT_EQ(app().document_title(), QStringLiteral("survey"));
    EXPECT_TRUE(window().windowTitle().startsWith(QStringLiteral("survey")));
    EXPECT_EQ(app().recent_files().value(0), QFileInfo(path).absoluteFilePath());

    ASSERT_TRUE(trigger("action.file.close"));
    EXPECT_TRUE(app().document().path().empty());
    EXPECT_TRUE(app().document().workspace().chunks().empty());
    EXPECT_FALSE(app().undo_stack().can_undo());

    ASSERT_TRUE(trigger("action.file.open"));
    ASSERT_EQ(app().document().workspace().chunks().size(), 2U);
    EXPECT_EQ(app().document().workspace().chunks()[0]->label(), "Block A");
    EXPECT_EQ(app().document().workspace().chunks()[1]->label(), "Block B");
    EXPECT_EQ(app().document_title(), QStringLiteral("survey"));
    EXPECT_FALSE(app().dirty());

    // Save on an already named project writes in place and keeps the name.
    add_chunk(app(), "Block C");
    ASSERT_TRUE(trigger("action.file.save"));
    EXPECT_FALSE(app().dirty());
    EXPECT_EQ(app().document().path(), path.toStdString());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(MainWindowTest, OpensAProjectCreatedByTheEngine) {
    const QString path = make_engine_project(scratch("engine.psx"));
    ASSERT_TRUE(window().open_project(path));
    ASSERT_EQ(app().document().workspace().chunks().size(), 2U);
    EXPECT_EQ(app().document().workspace().chunks()[1]->label(), "Block B");
    EXPECT_FALSE(app().dirty());
}

TEST_F(MainWindowTest, SaveWithoutAPathAsksWhereAndCancelKeepsTheDocumentDirty) {
    int asked = 0;
    window().set_save_path_handler([&](const QString& suggested) {
        ++asked;
        EXPECT_EQ(suggested, QStringLiteral("Untitled.psx"));
        return QString();
    });
    add_chunk(app(), "A");
    EXPECT_FALSE(window().save_project());
    EXPECT_EQ(asked, 1);
    EXPECT_TRUE(app().dirty());

    // A name typed without an extension gets .psx.
    window().set_save_path_handler([&](const QString&) { return scratch("noext"); });
    EXPECT_TRUE(window().save_project());
    EXPECT_TRUE(QFile::exists(scratch("noext.psx")));
}

TEST_F(MainWindowTest, FailedOpenIsReportedAndLeavesTheDocumentAlone) {
    add_chunk(app(), "Keep me");
    window().set_unsaved_handler([](const QString&) { return UnsavedChoice::Discard; });
    QStringList errors;
    window().set_error_handler([&](const QString& title, const QString& message) {
        errors << title + QStringLiteral(": ") + message;
    });
    EXPECT_FALSE(window().open_project(scratch("missing.psx")));
    EXPECT_EQ(errors.size(), 1);
    EXPECT_EQ(app().document().workspace().chunks().size(), 1U);
    EXPECT_FALSE(app().recent_files().contains(QFileInfo(scratch("missing.psx")).absoluteFilePath()));
}

TEST_F(MainWindowTest, UndoRedoActionsFollowTheUndoStack) {
    QAction* undo = action("action.edit.undo");
    QAction* redo = action("action.edit.redo");
    EXPECT_FALSE(undo->isEnabled());
    EXPECT_FALSE(redo->isEnabled());
    EXPECT_EQ(undo->text(), QStringLiteral("Undo"));

    add_chunk(app(), "A");
    EXPECT_TRUE(undo->isEnabled());
    EXPECT_FALSE(redo->isEnabled());
    EXPECT_EQ(undo->text(), QStringLiteral("Undo Add Chunk"));
    EXPECT_EQ(app().document().workspace().chunks().size(), 1U);

    ASSERT_TRUE(trigger("action.edit.undo"));
    EXPECT_FALSE(undo->isEnabled());
    EXPECT_TRUE(redo->isEnabled());
    EXPECT_EQ(undo->text(), QStringLiteral("Undo"));
    EXPECT_EQ(redo->text(), QStringLiteral("Redo Add Chunk"));
    EXPECT_TRUE(app().document().workspace().chunks().empty());
    EXPECT_FALSE(app().dirty());  // back at the loaded state

    ASSERT_TRUE(trigger("action.edit.redo"));
    EXPECT_TRUE(undo->isEnabled());
    EXPECT_FALSE(redo->isEnabled());
    EXPECT_EQ(app().document().workspace().chunks().size(), 1U);
    EXPECT_TRUE(app().dirty());

    // Replacing the document clears the history and the actions follow.
    app().new_document("");
    EXPECT_FALSE(undo->isEnabled());
}

TEST_F(MainWindowTest, DirtyFlagFollowsTheCleanPointOfTheHistory) {
    const QString path = scratch("dirty.psx");
    window().set_save_path_handler([&](const QString&) { return path; });
    add_chunk(app(), "A");
    ASSERT_TRUE(window().save_project());  // clean point at history index 1
    EXPECT_FALSE(app().dirty());

    add_chunk(app(), "B");
    EXPECT_TRUE(app().dirty());
    app().undo();
    EXPECT_FALSE(app().dirty());  // undone back to the saved state
    app().undo();
    EXPECT_TRUE(app().dirty());   // index 0, before the saved state

    // A new edit C from index 0 puts the history at index 1 again, but it is a different
    // history: the saved state is unreachable, so the document must be dirty right away.
    add_chunk(app(), "C");
    EXPECT_TRUE(app().dirty());
    app().undo();
    EXPECT_TRUE(app().dirty());
    app().redo();  // index 1 once more
    EXPECT_TRUE(app().dirty());
    app().undo();
    app().redo();
    EXPECT_TRUE(app().dirty());  // stays dirty until the next save

    ASSERT_TRUE(window().save_project());
    EXPECT_FALSE(app().dirty());
}

TEST_F(MainWindowTest, DroppingTheCleanPointOffTheUndoCapMakesTheDocumentDirty) {
    window().set_save_path_handler([&](const QString&) { return scratch("cap.psx"); });
    for (std::size_t i = 0; i < kMaxUndoDepth; ++i) {
        add_chunk(app(), "X");
    }
    ASSERT_TRUE(window().save_project());  // clean point at the cap
    EXPECT_FALSE(app().dirty());

    // One more edit pushes the oldest command off the bottom; the history length (and so the
    // position) is unchanged, but the saved state is gone.
    add_chunk(app(), "Y");
    EXPECT_GT(app().undo_stack().dropped_count(), 0U);
    EXPECT_TRUE(app().dirty());
    app().undo();
    app().redo();
    EXPECT_TRUE(app().dirty());
    ASSERT_TRUE(window().save_project());
    EXPECT_FALSE(app().dirty());
}

TEST_F(MainWindowTest, OpenPromptsExactlyOnceAndNotWhenTheFileDialogIsCancelled) {
    int prompts = 0;
    window().set_unsaved_handler([&](const QString&) {
        ++prompts;
        return UnsavedChoice::Discard;
    });
    const QString other = make_engine_project(scratch("other.psx"));
    QString chosen = other;
    int dialogs = 0;
    window().set_open_path_handler([&]() {
        ++dialogs;
        return chosen;
    });

    add_chunk(app(), "Dirty");
    ASSERT_TRUE(window().open_project());
    EXPECT_EQ(dialogs, 1);
    EXPECT_EQ(prompts, 1);
    EXPECT_EQ(app().document().workspace().chunks().size(), 2U);

    // Cancelling the file dialog does nothing and does not prompt.
    prompts = 0;
    add_chunk(app(), "Dirty again");
    chosen.clear();
    EXPECT_FALSE(window().open_project());
    EXPECT_EQ(prompts, 0);
    EXPECT_TRUE(app().dirty());
    EXPECT_EQ(app().document().workspace().chunks().size(), 3U);

    // The recent-files path (open_project(path)) prompts once as well.
    EXPECT_TRUE(window().open_project(other));
    EXPECT_EQ(prompts, 1);
}

TEST_F(MainWindowTest, SaveChoiceThatFailsAbortsAndReportsTheError) {
    // The project path is a directory, so writing it fails.
    const QString blocked = scratch("blocked.psx");
    ASSERT_TRUE(QDir().mkpath(blocked));
    app().new_document(blocked.toStdString());
    add_chunk(app(), "Unsaved");
    window().set_unsaved_handler([](const QString&) { return UnsavedChoice::Save; });
    int errors = 0;
    window().set_error_handler([&](const QString&, const QString&) { ++errors; });

    EXPECT_FALSE(window().new_project());
    EXPECT_EQ(errors, 1);
    EXPECT_TRUE(app().dirty());
    EXPECT_EQ(app().document().workspace().chunks().size(), 1U);
    EXPECT_EQ(app().document().path(), blocked.toStdString());

    EXPECT_FALSE(window().close_project());
    EXPECT_EQ(errors, 2);
    EXPECT_EQ(app().document().workspace().chunks().size(), 1U);
}

TEST_F(MainWindowTest, SaveChoiceThenCancelledSaveAsAbortsTheOperation) {
    add_chunk(app(), "Unsaved");
    window().set_unsaved_handler([](const QString&) { return UnsavedChoice::Save; });
    int asked = 0;
    window().set_save_path_handler([&](const QString&) {
        ++asked;
        return QString();
    });
    const QString other = make_engine_project(scratch("elsewhere.psx"));
    EXPECT_FALSE(window().new_project());
    EXPECT_FALSE(window().open_project(other));
    EXPECT_FALSE(window().close_project());
    EXPECT_EQ(asked, 3);
    EXPECT_TRUE(app().dirty());
    EXPECT_EQ(app().document().workspace().chunks().size(), 1U);
}

TEST_F(MainWindowTest, DiscardReplacesTheDocumentOnNewOpenAndClose) {
    window().set_unsaved_handler([](const QString&) { return UnsavedChoice::Discard; });
    const QString other = make_engine_project(scratch("discard.psx"));

    add_chunk(app(), "A");
    ASSERT_TRUE(window().new_project());
    EXPECT_TRUE(app().document().workspace().chunks().empty());
    EXPECT_FALSE(app().dirty());

    add_chunk(app(), "B");
    ASSERT_TRUE(window().open_project(other));
    EXPECT_EQ(app().document().workspace().chunks().size(), 2U);
    EXPECT_FALSE(app().dirty());

    add_chunk(app(), "C");
    ASSERT_TRUE(window().close_project());
    EXPECT_TRUE(app().document().workspace().chunks().empty());
    EXPECT_FALSE(app().dirty());
}

TEST_F(MainWindowTest, FailedOpenKeepsThePreviousDocumentAndItsDirtyState) {
    add_chunk(app(), "Keep me");
    window().set_unsaved_handler([](const QString&) { return UnsavedChoice::Discard; });
    window().set_error_handler([](const QString&, const QString&) {});
    EXPECT_FALSE(window().open_project(scratch("missing.psx")));
    EXPECT_TRUE(app().dirty());
    EXPECT_EQ(app().document().workspace().chunks().size(), 1U);
    EXPECT_EQ(app().document().workspace().chunks()[0]->label(), "Keep me");
    app().undo();  // the history survived too
    EXPECT_TRUE(app().document().workspace().chunks().empty());
}

TEST_F(MainWindowTest, SaveAsRefreshesTheTitleWithoutReplacingTheDocument) {
    int replaced = 0;
    QObject::connect(&app(), &GuiApp::document_replaced, [&]() { ++replaced; });
    window().set_save_path_handler([&](const QString&) { return scratch("titled.psx"); });
    add_chunk(app(), "A");
    ASSERT_TRUE(window().save_project_as());
    EXPECT_EQ(replaced, 0);
    EXPECT_TRUE(window().windowTitle().contains(QStringLiteral("titled")));
}

TEST_F(MainWindowTest, ReplacingADirtyDocumentEmitsDirtyChanged) {
    std::vector<bool> seen;
    QObject::connect(&app(), &GuiApp::dirty_changed, [&](bool d) { seen.push_back(d); });
    add_chunk(app(), "A");
    app().new_document("");
    ASSERT_EQ(seen.size(), 2U);
    EXPECT_TRUE(seen[0]);
    EXPECT_FALSE(seen[1]);

    add_chunk(app(), "B");
    app().open_document(make_engine_project(scratch("replace.psx")).toStdString());
    ASSERT_EQ(seen.size(), 4U);
    EXPECT_FALSE(seen[3]);
}

TEST_F(MainWindowTest, UnsavedChangesPromptOnCloseNewAndOpen) {
    int prompts = 0;
    UnsavedChoice answer = UnsavedChoice::Cancel;
    QString asked_about;
    window().set_unsaved_handler([&](const QString& name) {
        ++prompts;
        asked_about = name;
        return answer;
    });
    show_window();

    // Clean document: nothing is asked.
    EXPECT_TRUE(window().new_project());
    EXPECT_EQ(prompts, 0);

    app().new_document("draft.psx");  // a named, clean project
    EXPECT_FALSE(app().dirty());
    add_chunk(app(), "A");

    // Cancel stops each operation and keeps the work.
    EXPECT_FALSE(window().new_project());
    EXPECT_EQ(prompts, 1);
    EXPECT_EQ(asked_about, QStringLiteral("draft"));
    EXPECT_FALSE(window().open_project(make_engine_project(scratch("x.psx"))));
    EXPECT_FALSE(window().close_project());
    EXPECT_EQ(app().document().workspace().chunks().size(), 1U);

    // Closing the window asks too, and Cancel keeps it open.
    const int before_close = prompts;
    EXPECT_FALSE(window().close());
    EXPECT_EQ(prompts, before_close + 1);
    EXPECT_TRUE(window().isVisible());

    // Discard lets the window close.
    answer = UnsavedChoice::Discard;
    EXPECT_TRUE(window().close());
    EXPECT_FALSE(window().isVisible());
}

TEST_F(MainWindowTest, SavingFromThePromptSavesThenProceeds) {
    const QString path = scratch("prompted.psx");
    window().set_save_path_handler([&](const QString&) { return path; });
    window().set_unsaved_handler([](const QString&) { return UnsavedChoice::Save; });
    show_window();
    add_chunk(app(), "A");
    EXPECT_TRUE(window().close());
    EXPECT_TRUE(QFile::exists(path));
    EXPECT_FALSE(window().isVisible());
}

TEST_F(MainWindowTest, RecentFilesAreCappedDeduplicatedAndShownInTheMenu) {
    for (int i = 0; i < 12; ++i) {
        app().add_recent_file(scratch(QStringLiteral("p%1.psx").arg(i)));
    }
    app().add_recent_file(scratch("p5.psx"));  // moves to the front
    const QStringList recent = app().recent_files();
    ASSERT_EQ(recent.size(), GuiApp::kMaxRecentFiles);
    EXPECT_EQ(recent.first(), QFileInfo(scratch("p5.psx")).absoluteFilePath());

    int visible = 0;
    for (int i = 0; i < GuiApp::kMaxRecentFiles; ++i) {
        if (action(QStringLiteral("action.file.recent.%1").arg(i))->isVisible()) {
            ++visible;
        }
    }
    EXPECT_EQ(visible, GuiApp::kMaxRecentFiles);
    EXPECT_EQ(action("action.file.recent.0")->text(), QStringLiteral("p5.psx"));

    // Choosing an entry opens that project.
    const QString real = make_engine_project(scratch("recent.psx"));
    app().add_recent_file(real);
    ASSERT_TRUE(trigger("action.file.recent.0"));
    EXPECT_EQ(app().document().workspace().chunks().size(), 2U);

    ASSERT_TRUE(trigger("action.file.recent.clear"));
    EXPECT_TRUE(app().recent_files().isEmpty());
    EXPECT_FALSE(menu("menu.file.recent")->isEnabled());
}

TEST_F(MainWindowTest, AboutTextNamesTheVersionAndLicence) {
    const QString text = window().about_text();
    EXPECT_TRUE(text.contains(QStringLiteral("alphashape")));
    EXPECT_TRUE(text.contains(QStringLiteral("General Public License")));
    EXPECT_TRUE(text.contains(QStringLiteral("github.com/aniketpratapsingh/alphashape")));
    EXPECT_TRUE(text.contains(QRegularExpression(QStringLiteral("\\d+\\.\\d+"))));
}
