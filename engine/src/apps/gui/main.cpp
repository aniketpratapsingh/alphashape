// alphashape-gui: the desktop front end.
//
// Exit codes: 0 success, 1 runtime error (for example --open on an unreadable project),
// 64 bad usage (EX_USAGE, as alphashape-cli).

#include <cstdio>
#include <cstddef>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

#include <QApplication>
#include <QByteArray>
#include <QStringList>
#include <QString>
#include <QtGlobal>

#include "apps/gui/app.hpp"
#include "apps/gui/command_line.hpp"
#include "apps/gui/main_window.hpp"
#include "base/platform/console.hpp"
#include "base/utils/log.hpp"

#ifndef ALPHASHAPE_GUI_VERSION
#define ALPHASHAPE_GUI_VERSION "0.0.0"
#endif

int main(int argc, char** argv) {
    using namespace alphashape;
    using namespace alphashape::gui;

    // On Windows the installed executable is a GUI-subsystem binary (no console
    // window beside the app), and Windows gives such a process no standard
    // handles even when a terminal started it. Adopting the parent's console
    // keeps --help, --version, --offscreen-smoke and the log visible from a
    // command prompt; a redirection the caller set up is left untouched. It is
    // a no-op off Windows.
    (void)platform::attach_parent_console();

    // The smoke run must work on a machine with no display server. Only the literal flag is
    // looked for here; the real parse happens below, once Qt has removed its own options.
    bool wants_smoke = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--offscreen-smoke") {
            wants_smoke = true;
        }
    }
    bool headless = false;
#if defined(Q_OS_UNIX) && !defined(Q_OS_MACOS)
    headless = qEnvironmentVariableIsEmpty("DISPLAY") && qEnvironmentVariableIsEmpty("WAYLAND_DISPLAY");
#endif
    bool forced_offscreen_for_headless = false;
    if ((wants_smoke || headless) && qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        forced_offscreen_for_headless = headless && !wants_smoke;
        // Also lets --help / --version / bad usage run on a display-less machine.
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }

    // The QApplication goes first: it strips Qt's own options (-platform, -style, ...) from
    // the argument list, and QCoreApplication::arguments() is Unicode-correct on Windows.
    QApplication qapp(argc, argv);
    QApplication::setOrganizationName(QStringLiteral("alphashape"));
    QApplication::setApplicationName(QStringLiteral("alphashape"));
    QApplication::setApplicationVersion(QStringLiteral(ALPHASHAPE_GUI_VERSION));

    std::vector<std::string> args;
    const QStringList qt_args = QCoreApplication::arguments();
    for (qsizetype i = 1; i < qt_args.size(); ++i) {
        const QByteArray utf8 = qt_args[i].toUtf8();
        args.emplace_back(utf8.constData(), static_cast<std::size_t>(utf8.size()));
    }
    const ParseResult parsed = parse_command_line(args);
    if (!parsed.ok()) {
        std::fprintf(stderr, "alphashape-gui: %s\n\n%s", parsed.error.c_str(), usage_text());
        return 64;
    }
    const Options& options = parsed.options;
    if (options.action == Options::Action::Help) {
        std::fputs(usage_text(), stdout);
        return 0;
    }
    if (options.action == Options::Action::Version) {
        std::printf("alphashape-gui %s\n", ALPHASHAPE_GUI_VERSION);
        return 0;
    }
    if (forced_offscreen_for_headless && !options.offscreen_smoke) {
        // We only picked the offscreen platform so that parsing could run; a real session needs one.
        std::fputs("alphashape-gui: no display available (set DISPLAY or QT_QPA_PLATFORM)\n", stderr);
        return 1;
    }
    if (options.log_level) {
        Log::set_level(*options.log_level);
    }

    try {
        GuiApp app(options);
        MainWindow window(app);
        window.show();
        if (options.offscreen_smoke) {
            QApplication::processEvents();
            std::puts("gui ready");
            std::fflush(stdout);
            return 0;
        }
        return QApplication::exec();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "alphashape-gui: %s\n", e.what());
        return 1;
    }
}
