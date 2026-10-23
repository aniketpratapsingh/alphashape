#include <gtest/gtest.h>

#include "apps/gui/app.hpp"
#include "apps/gui/main_window.hpp"
#include "base/utils/error.hpp"
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
