// alphashape-gui: the desktop front end.
//
// Exit codes: 0 success, 1 runtime error (for example --open on an unreadable project),
// 64 bad usage (EX_USAGE, as alphashape-cli).

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include <QApplication>
#include <QString>
#include <QtGlobal>

#include "apps/gui/app.hpp"
#include "apps/gui/command_line.hpp"
#include "apps/gui/main_window.hpp"
#include "base/utils/log.hpp"

#ifndef ALPHASHAPE_GUI_VERSION
#define ALPHASHAPE_GUI_VERSION "0.0.0"
#endif

int main(int argc, char** argv) {
    using namespace alphashape;
    using namespace alphashape::gui;

    const std::vector<std::string> args(argv + 1, argv + argc);
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
    if (options.log_level) {
        Log::set_level(*options.log_level);
    }
    // The smoke run must work on a machine with no display server.
    if (options.offscreen_smoke && qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }

    QApplication qapp(argc, argv);
    QApplication::setOrganizationName(QStringLiteral("alphashape"));
    QApplication::setApplicationName(QStringLiteral("alphashape"));
    QApplication::setApplicationVersion(QStringLiteral(ALPHASHAPE_GUI_VERSION));

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
