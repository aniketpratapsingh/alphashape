// Main for the viewport render tests.
//
// These are the only GUI tests that need a platform integration with a graphics API: producing
// pixels requires a QRhi, and the offscreen QPA plugin reports that it has no RHI-based
// rendering at all (Qt's QRhiWidget warns outright on it). So this binary runs on the platform
// Qt would pick anyway (cocoa, windows, xcb) and, on a display-less Linux host, runs itself
// under `xvfb-run`, which GitHub's ubuntu image ships.
//
// Choosing that path is a two-step decision, because loading the xcb plugin on a host without
// its runtime libraries makes Qt abort rather than return an error:
//   1. `--probe-platform` creates the QGuiApplication and exits; it is run in a child process, so
//      a missing plugin is an exit code and not a crash of this one.
//   2. Only when the probe succeeds is the real run re-executed under xvfb-run. Otherwise the
//      tests run here on the offscreen plugin, where they exercise the whole recording path on
//      QRhi's Null backend and say so in their output.
//
// "No GPU in this container" therefore never fails the suite.

#include <cstdio>

#include <gtest/gtest.h>

#include <QByteArray>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QGuiApplication>
#include <QProcess>
#include <QProcessEnvironment>
#include <QString>
#include <QStringList>
#include <QTemporaryDir>
#include <QtGlobal>

namespace {

/// Marks the child of the xvfb re-exec, so a child that somehow still has no display cannot
/// start another one.
constexpr const char* kReexecMarker = "ALPHASHAPE_VIEWER_RENDER_REEXEC";

/// Absolute path of this executable. QCoreApplication::applicationFilePath() is only meaningful
/// once an application object exists, and these helpers run before that, so argv[0] is resolved
/// instead.
QString self_path(const char* argv0) {
    const QFileInfo info(QString::fromLocal8Bit(argv0 != nullptr ? argv0 : ""));
    if (info.isAbsolute()) {
        return info.absoluteFilePath();
    }
    return QFileInfo(QDir::current(), info.filePath()).absoluteFilePath();
}

bool has_display() {
#if defined(Q_OS_LINUX)
    return !qEnvironmentVariableIsEmpty("DISPLAY") ||
           !qEnvironmentVariableIsEmpty("WAYLAND_DISPLAY");
#else
    return true;  // macOS and Windows sessions always have a compositor
#endif
}

/// Run an external program with the inherited streams and return its exit code (-1 when it could
/// not be started).
int run_forwarded(const QString& program, const QStringList& arguments,
                  const QProcessEnvironment& environment) {
    QProcess process;
    process.setProcessEnvironment(environment);
    // Inherit the streams, so ctest captures the child's output as its own.
    process.setProcessChannelMode(QProcess::ForwardedChannels);
    process.start(program, arguments);
    if (!process.waitForStarted(5000)) {
        return -1;
    }
    process.waitForFinished(-1);
    return process.exitStatus() == QProcess::NormalExit ? process.exitCode() : -1;
}

/// True when `xvfb-run <self> --probe-platform` starts a QGuiApplication successfully. The
/// probe's own diagnostics (the platform name, or Qt's plugin error) go to our stderr.
bool xvfb_platform_works(const QString& self) {
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert(QString::fromLatin1(kReexecMarker), QStringLiteral("1"));
    environment.remove(QStringLiteral("QT_QPA_PLATFORM"));
    const QStringList arguments{QStringLiteral("-a"),
                                QStringLiteral("--server-args=-screen 0 1280x1024x24"), self,
                                QStringLiteral("--probe-platform")};
    return run_forwarded(QStringLiteral("xvfb-run"), arguments, environment) == 0;
}

/// Re-run this binary (with its own arguments) under xvfb-run; returns the child's exit code, or
/// -1 when xvfb-run could not be started.
int reexec_under_xvfb(const QString& self, int argc, char** argv) {
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert(QString::fromLatin1(kReexecMarker), QStringLiteral("1"));
    environment.remove(QStringLiteral("QT_QPA_PLATFORM"));
    QStringList arguments{QStringLiteral("-a"),
                          QStringLiteral("--server-args=-screen 0 1280x1024x24"), self};
    for (int i = 1; i < argc; ++i) {
        arguments << QString::fromLocal8Bit(argv[i]);
    }
    return run_forwarded(QStringLiteral("xvfb-run"), arguments, environment);
}

}  // namespace

int main(int argc, char** argv) {
    bool probe = false;
    bool list_tests = false;
    for (int i = 1; i < argc; ++i) {
        const QByteArray argument(argv[i]);
        probe = probe || argument == "--probe-platform";
        list_tests = list_tests || argument == "--gtest_list_tests";
    }

    // ctest asks for the test list at build time (gtest_discover_tests), on a host that has no
    // display and no reason to start one. Listing needs no window, so go straight to the
    // offscreen plugin there.
    if (list_tests) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }

    const bool needs_platform = !list_tests && !has_display() &&
                                qEnvironmentVariableIsEmpty(kReexecMarker);

    if (needs_platform) {
        const QString self = self_path(argc > 0 ? argv[0] : nullptr);
        // Step 1: can a real platform start here at all? Only then is the extra xvfb pass worth
        // making; otherwise the offscreen plugin below is the honest choice.
        if (!self.isEmpty() && xvfb_platform_works(self)) {
            const int child = reexec_under_xvfb(self, argc, argv);
            if (child >= 0) {
                return child;
            }
        }
        // No display and no usable platform: stay headless. The tests then use QRhi's Null
        // backend and report that they did.
        qputenv("QT_QPA_PLATFORM", "offscreen");
        std::fputs("viewer render tests: no display platform available; running headless "
                   "(QRhi Null backend, no pixel assertions)\n",
                   stdout);
    }

    QTemporaryDir settings_dir;
    const QString settings_file =
        QDir(settings_dir.path())
            .filePath(QStringLiteral("settings-%1.ini").arg(QCoreApplication::applicationPid()));
    qputenv("ALPHASHAPE_GUI_SETTINGS_FILE", settings_file.toLocal8Bit());

    QGuiApplication app(argc, argv);
    if (probe) {
        std::printf("viewer render probe: platform %s\n",
                    QGuiApplication::platformName().toUtf8().constData());
        std::fflush(stdout);
        return 0;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
