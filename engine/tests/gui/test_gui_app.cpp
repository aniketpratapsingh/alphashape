#include <filesystem>

#include <gtest/gtest.h>

#include <QTemporaryDir>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "apps/gui/app.hpp"
#include "apps/gui/main_window.hpp"
#include "base/utils/error.hpp"
#include "common/commands/command.hpp"
#include "common/core/document.hpp"
#include "common/io/lock_file.hpp"
#include "common/io/project_io.hpp"
#include "harness.hpp"

using alphashape::gui::GuiApp;
using alphashape::gui::Options;
using alphashape::gui::ThemeMode;

TEST(GuiApp, ThemeChoicePersistsButCommandLineOverrideDoesNot) {
    Options reset;
    reset.reset_settings = true;
    {
        GuiApp app(reset);
        EXPECT_EQ(app.theme().mode(), ThemeMode::Dark);
        app.set_theme(ThemeMode::Light);
    }
    {
        GuiApp app{Options{}};
        EXPECT_EQ(app.theme().mode(), ThemeMode::Light);  // remembered
    }
    {
        Options forced;
        forced.theme = ThemeMode::Dark;
        GuiApp app(forced);
        EXPECT_EQ(app.theme().mode(), ThemeMode::Dark);
    }
    {
        GuiApp app{Options{}};
        EXPECT_EQ(app.theme().mode(), ThemeMode::Light);  // --theme did not overwrite it
    }
    {
        GuiApp app(reset);
        EXPECT_EQ(app.theme().mode(), ThemeMode::Dark);  // --reset-settings forgot it
    }
}

TEST(GuiApp, NewProjectGetsItsPathAndFreshUndoStack) {
    Options options;
    options.new_project = true;
    options.new_path = "site.psx";
    GuiApp app(options);
    EXPECT_EQ(app.document().path(), "site.psx");
    EXPECT_EQ(app.document_title(), QStringLiteral("site"));
    EXPECT_FALSE(app.undo_stack().can_undo());

    app.new_document("");
    EXPECT_EQ(app.document_title(), QStringLiteral("Untitled"));
}

TEST(GuiApp, OpeningAMissingProjectThrowsAndKeepsTheDocument) {
    GuiApp app{Options{}};
    EXPECT_THROW(app.open_document("/nonexistent/dir/none.psx"), alphashape::Error);
    EXPECT_EQ(app.document_title(), QStringLiteral("Untitled"));

    Options options;
    options.open_path = "/nonexistent/dir/none.psx";
    EXPECT_THROW(GuiApp{options}, alphashape::Error);
}

using GuiAppWindow = alphashape::gui::testing::GuiTest;

TEST_F(GuiAppWindow, MainWindowTitleFollowsTheDocument) {
    EXPECT_TRUE(window().windowTitle().startsWith(QStringLiteral("Untitled")));
    app().new_document("survey.psx");
    EXPECT_TRUE(window().windowTitle().startsWith(QStringLiteral("survey")));
}

namespace {
namespace fs = std::filesystem;

/// A one-chunk project on disk, so open_document() has something to lock.
void write_seed_project(const std::string& path) {
    alphashape::Document document;
    alphashape::Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    alphashape::ProjectIO::save(document, path);
}
}  // namespace

TEST(GuiApp, SaveAsSwapsTheLockAndKeepsUndo) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    const fs::path root(dir.path().toStdString());
    const std::string old = (root / "old.psx").string();
    const std::string next = (root / "new.psx").string();
    write_seed_project(old);

    GuiApp app{Options{}};
    app.open_document(old);
    const auto chunk_id = app.document().workspace().chunks().front()->id();
    app.execute(alphashape::make_set_chunk_label_command(chunk_id, app.document(),
                                                         "Renamed"));
    EXPECT_EQ(app.document().workspace().chunks().front()->label(), "Renamed");

    const std::string old_lock = alphashape::ProjectLock::lock_path(old);
    ASSERT_TRUE(fs::exists(old_lock));
    app.save_document_as(next);

    EXPECT_FALSE(fs::exists(old_lock));  // the swap releases the old lock
    EXPECT_TRUE(fs::exists(alphashape::ProjectLock::lock_path(next)));
    EXPECT_EQ(app.document().path(), next);

    // Save As must not cost the history: undo still reverts the edit.
    EXPECT_TRUE(app.undo_stack().can_undo());
    app.undo();
    EXPECT_EQ(app.document().workspace().chunks().front()->label(), "Chunk 1");
}

TEST(GuiApp, ReadOnlyProjectOpensWithoutLock) {
#if defined(_WIN32)
    GTEST_SKIP() << "POSIX directory-permission test";
#else
    if (::geteuid() == 0) {
        GTEST_SKIP() << "root ignores directory permissions";
    }
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    const fs::path root(dir.path().toStdString());
    const std::string path = (root / "ro.psx").string();
    write_seed_project(path);
    const fs::path files = root / "ro.files";
    std::error_code ec;
    fs::permissions(files, fs::perms::owner_read | fs::perms::owner_exec,
                    fs::perm_options::replace, ec);
    ASSERT_FALSE(ec) << ec.message();

    GuiApp app{Options{}};
    EXPECT_NO_THROW(app.open_document(path));
    EXPECT_TRUE(app.opened_read_only());
    EXPECT_FALSE(fs::exists(alphashape::ProjectLock::lock_path(path)));

    fs::permissions(files, fs::perms::owner_all, fs::perm_options::replace, ec);
    EXPECT_FALSE(ec) << ec.message();
#endif
}
