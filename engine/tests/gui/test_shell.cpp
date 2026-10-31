// GuiShell.*: the shell as a user starts it. StartsOffscreen and BadUsageExits64 run the real
// alphashape-gui binary; NoQtWarnings builds the app and window in-process.

#include <gtest/gtest.h>

#include <QApplication>
#include <QProcess>
#include <QStringList>
#include <QtGlobal>

#include "apps/gui/app.hpp"
#include "apps/gui/main_window.hpp"
#include "harness.hpp"

using namespace alphashape::gui;

namespace {

struct RunResult {
    bool finished = false;
    int exit_code = -1;
    QString out;
    QString err;
};

RunResult run_gui(const QStringList& args, int timeout_ms) {
    QProcess process;
    process.start(QStringLiteral(ALPHASHAPE_GUI_EXE), args);
    RunResult result;
    result.finished = process.waitForFinished(timeout_ms);
    if (!result.finished) {
        process.kill();
        process.waitForFinished(5000);
    }
    result.exit_code = process.exitCode();
    result.out = QString::fromUtf8(process.readAllStandardOutput());
    result.err = QString::fromUtf8(process.readAllStandardError());
    return result;
}

}  // namespace

TEST(GuiShell, StartsOffscreen) {
    const RunResult r = run_gui({QStringLiteral("--offscreen-smoke")}, 20000);
    ASSERT_TRUE(r.finished) << "did not finish within 20 s; stderr: " << r.err.toStdString();
    EXPECT_EQ(r.exit_code, 0) << r.err.toStdString();
    EXPECT_TRUE(r.out.contains(QStringLiteral("gui ready"))) << r.out.toStdString();
}

using GuiShellWindow = alphashape::gui::testing::GuiTest;

TEST_F(GuiShellWindow, NoQtWarnings) {
    show_window();
    app().theme().apply(ThemeMode::Light);
    alphashape::gui::testing::pump();
    app().theme().apply(ThemeMode::Dark);
    alphashape::gui::testing::pump();
    window().close();
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST(GuiShell, QtOptionsAreAcceptedAndStripped) {
    const RunResult r = run_gui({QStringLiteral("-platform"), QStringLiteral("offscreen"),
                                 QStringLiteral("--offscreen-smoke")}, 20000);
    ASSERT_TRUE(r.finished);
    EXPECT_EQ(r.exit_code, 0) << r.err.toStdString();
    EXPECT_TRUE(r.out.contains(QStringLiteral("gui ready"))) << r.out.toStdString();
}

TEST(GuiShell, BadUsageExits64) {
    const QList<QStringList> bad = {
        {QStringLiteral("--bogus")},
        {QStringLiteral("--theme"), QStringLiteral("purple")},
        {QStringLiteral("--open")},
        {QStringLiteral("--log-level"), QStringLiteral("loud")},
        {QStringLiteral("--new"), QStringLiteral("a.psx"), QStringLiteral("--open"), QStringLiteral("b.psx")},
    };
    for (const QStringList& args : bad) {
        const RunResult r = run_gui(args, 20000);
        ASSERT_TRUE(r.finished) << args.join(QLatin1Char(' ')).toStdString();
        EXPECT_EQ(r.exit_code, 64) << args.join(QLatin1Char(' ')).toStdString();
        EXPECT_TRUE(r.err.contains(QStringLiteral("Usage:")));
    }
}

TEST(GuiShell, HelpAndVersionExitZero) {
    const RunResult help = run_gui({QStringLiteral("--help")}, 20000);
    ASSERT_TRUE(help.finished);
    EXPECT_EQ(help.exit_code, 0);
    EXPECT_TRUE(help.out.contains(QStringLiteral("--offscreen-smoke")));

    const RunResult version = run_gui({QStringLiteral("--version")}, 20000);
    ASSERT_TRUE(version.finished);
    EXPECT_EQ(version.exit_code, 0);
    // The program identifies itself on a line of its own. Checked line-wise
    // rather than as a prefix of the whole capture: on Windows the GUI-subsystem
    // binary's console adoption can leave a UTF-8 BOM or a blank line ahead of
    // the text, which says nothing about whether --version printed its line.
    const QStringList lines = version.out.split(QLatin1Char('\n'));
    bool identified = false;
    for (const QString& line : lines) {
        const QString trimmed = line.trimmed();
        if (trimmed.startsWith(QStringLiteral("alphashape-gui "))) {
            identified = true;
        }
    }
    EXPECT_TRUE(identified) << "version output: " << version.out.toStdString();
}

TEST(GuiShell, OpeningAMissingProjectExits1) {
    const RunResult r = run_gui({QStringLiteral("--offscreen-smoke"), QStringLiteral("--open"),
                                 QStringLiteral("/nonexistent/dir/none.psx")}, 20000);
    ASSERT_TRUE(r.finished);
    EXPECT_EQ(r.exit_code, 1) << r.err.toStdString();
}
