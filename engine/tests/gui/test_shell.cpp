// GuiShell.*: the shell as a user starts it. StartsOffscreen and BadUsageExits64 run the real
// alphashape-gui binary; NoQtWarnings builds the app and window in-process.

#include <gtest/gtest.h>

#include <QApplication>
#include <QElapsedTimer>
#include <QMutex>
#include <QProcess>
#include <QStringList>
#include <QtGlobal>

#include "apps/gui/app.hpp"
#include "apps/gui/main_window.hpp"

using namespace alphashape::gui;

namespace {

struct RunResult {
    bool finished = false;
    int exit_code = -1;
    QString out;
    QString err;
    qint64 millis = 0;
};

RunResult run_gui(const QStringList& args, int timeout_ms) {
    QProcess process;
    QElapsedTimer timer;
    timer.start();
    process.start(QStringLiteral(ALPHASHAPE_GUI_EXE), args);
    RunResult result;
    result.finished = process.waitForFinished(timeout_ms);
    result.millis = timer.elapsed();
    if (!result.finished) {
        process.kill();
        process.waitForFinished(5000);
    }
    result.exit_code = process.exitCode();
    result.out = QString::fromUtf8(process.readAllStandardOutput());
    result.err = QString::fromUtf8(process.readAllStandardError());
    return result;
}

// Collects every message Qt would have printed at warning level or worse.
QStringList g_messages;
QMutex g_messages_mutex;
QtMessageHandler g_previous_handler = nullptr;

// Messages that come from Qt's platform layer rather than from our code and carry no action:
//  - the offscreen QPA plugin has no window-manager size hints and says so for every top-level;
//  - the font database reports (with a timing that varies per machine) that it resolved the
//    generic "sans-serif" of the style sheet's font list through an alias;
//  - the offscreen plugin on Windows has no native font database and complains about the
//    missing Qt font directory (the real "windows" platform plugin does not).
bool is_known_platform_noise(const QString& message) {
    return message.startsWith(QLatin1String("This plugin does not support propagateSizeHints")) ||
           message.startsWith(QLatin1String("Populating font family aliases took")) ||
           message.startsWith(QLatin1String("QFontDatabase: Cannot find font directory"));
}

void collecting_handler(QtMsgType type, const QMessageLogContext& context, const QString& message) {
    if ((type == QtWarningMsg || type == QtCriticalMsg || type == QtFatalMsg) &&
        is_known_platform_noise(message)) {
        return;
    }
    if (type == QtWarningMsg || type == QtCriticalMsg || type == QtFatalMsg) {
        QMutexLocker lock(&g_messages_mutex);
        g_messages << message;
        return;
    }
    if (g_previous_handler) {
        g_previous_handler(type, context, message);
    }
}

}  // namespace

TEST(GuiShell, StartsOffscreen) {
    const RunResult r = run_gui({QStringLiteral("--offscreen-smoke")}, 5000);
    ASSERT_TRUE(r.finished) << "did not finish within 5 s; stderr: " << r.err.toStdString();
    EXPECT_EQ(r.exit_code, 0) << r.err.toStdString();
    EXPECT_TRUE(r.out.contains(QStringLiteral("gui ready"))) << r.out.toStdString();
    EXPECT_LE(r.millis, 5000);
}

TEST(GuiShell, NoQtWarnings) {
    g_previous_handler = qInstallMessageHandler(collecting_handler);
    {
        Options options;
        options.reset_settings = true;
        GuiApp app(options);
        MainWindow window(app);
        window.show();
        QApplication::processEvents();
        app.theme().apply(ThemeMode::Light);
        QApplication::processEvents();
        app.theme().apply(ThemeMode::Dark);
        QApplication::processEvents();
        window.close();
    }
    qInstallMessageHandler(g_previous_handler);
    EXPECT_TRUE(g_messages.isEmpty()) << g_messages.join(QLatin1Char('\n')).toStdString();
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
    EXPECT_TRUE(version.out.startsWith(QStringLiteral("alphashape-gui ")));
}

TEST(GuiShell, OpeningAMissingProjectExits1) {
    const RunResult r = run_gui({QStringLiteral("--offscreen-smoke"), QStringLiteral("--open"),
                                 QStringLiteral("/nonexistent/dir/none.psx")}, 20000);
    ASSERT_TRUE(r.finished);
    EXPECT_EQ(r.exit_code, 1) << r.err.toStdString();
}
