#pragma once

// Command line of alphashape-gui. Parsed after the QApplication has stripped Qt's own options;
// a bad command line exits 64 (EX_USAGE, the same code alphashape-cli uses). Arguments are
// UTF-8 (converted from QCoreApplication::arguments()).

#include <optional>
#include <string>
#include <vector>

#include "base/utils/log.hpp"
#include "apps/gui/theme/theme_mode.hpp"

namespace alphashape::gui {

struct Options {
    enum class Action { Run, Help, Version };

    Action action = Action::Run;
    /// --new [path]: start with an empty project; `path` (may be empty) is where it will be saved.
    bool new_project = false;
    /// Path given to --new, or empty.
    std::string new_path;
    /// --open <path> (or a lone positional argument): project to load at startup.
    std::optional<std::string> open_path;
    /// --theme: overrides the stored preference for this session only.
    std::optional<ThemeMode> theme;
    bool reset_settings = false;
    std::optional<LogLevel> log_level;
    /// Construct the app and an empty main window, print "gui ready", exit 0.
    bool offscreen_smoke = false;
};

struct ParseResult {
    Options options;
    /// Empty on success; otherwise a one-line description of what was wrong.
    std::string error;

    bool ok() const { return error.empty(); }
};

/// `args` excludes argv[0] and any Qt options; strings are UTF-8.
ParseResult parse_command_line(const std::vector<std::string>& args);

/// Full --help text.
const char* usage_text();

/// "trace|debug|info|warn|error|off" -> level.
std::optional<LogLevel> log_level_from_name(const std::string& name);

}  // namespace alphashape::gui
