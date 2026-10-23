#include "apps/gui/command_line.hpp"

#include <cstddef>

namespace alphashape::gui {

namespace {

constexpr const char* kUsage =
    "Usage:\n"
    "  alphashape-gui [options] [project]\n"
    "\n"
    "Options:\n"
    "  --new [path]         start with an empty project (optionally to be saved at path)\n"
    "  --open <path>        open a project (a lone positional argument does the same)\n"
    "  --theme dark|light   use this theme for the session (overrides the stored choice)\n"
    "  --reset-settings     discard stored settings before starting\n"
    "  --log-level <lvl>    trace, debug, info, warn, error or off (default info)\n"
    "  --offscreen-smoke    build the window headlessly, print \"gui ready\" and exit\n"
    "  --version            print the version and exit\n"
    "  --help               print this text and exit\n";

bool looks_like_option(const std::string& arg) {
    return arg.size() > 1 && arg[0] == '-';
}

}  // namespace

const char* usage_text() { return kUsage; }

std::optional<LogLevel> log_level_from_name(const std::string& name) {
    if (name == "trace") return LogLevel::Trace;
    if (name == "debug") return LogLevel::Debug;
    if (name == "info") return LogLevel::Info;
    if (name == "warn") return LogLevel::Warn;
    if (name == "error") return LogLevel::Error;
    if (name == "off") return LogLevel::Off;
    return std::nullopt;
}

ParseResult parse_command_line(const std::vector<std::string>& args) {
    ParseResult result;
    Options& opts = result.options;
    auto fail = [&result](std::string message) {
        result.error = std::move(message);
        return result;
    };

    for (std::size_t i = 0; i < args.size(); ++i) {
        std::string arg = args[i];
        // Accept --name=value as well as --name value.
        std::optional<std::string> inline_value;
        if (arg.rfind("--", 0) == 0) {
            const std::size_t eq = arg.find('=');
            if (eq != std::string::npos) {
                inline_value = arg.substr(eq + 1);
                arg.resize(eq);
            }
        }
        auto take_value = [&]() -> std::optional<std::string> {
            if (inline_value) {
                return inline_value;
            }
            if (i + 1 < args.size()) {
                return args[++i];
            }
            return std::nullopt;
        };

        if (arg == "--help" || arg == "-h") {
            opts.action = Options::Action::Help;
        } else if (arg == "--version") {
            opts.action = Options::Action::Version;
        } else if (arg == "--new") {
            if (opts.new_project) {
                return fail("--new given twice");
            }
            opts.new_project = true;
            // The path is optional: take the next token only if it is not an option.
            if (inline_value) {
                opts.new_path = *inline_value;
            } else if (i + 1 < args.size() && !looks_like_option(args[i + 1])) {
                opts.new_path = args[++i];
            }
        } else if (arg == "--open") {
            const auto value = take_value();
            if (!value || value->empty()) {
                return fail("--open needs a path");
            }
            if (opts.open_path) {
                return fail("more than one project to open");
            }
            opts.open_path = *value;
        } else if (arg == "--theme") {
            const auto value = take_value();
            const auto mode = value ? theme_from_name(*value) : std::nullopt;
            if (!mode) {
                return fail("--theme must be dark or light");
            }
            opts.theme = mode;
        } else if (arg == "--reset-settings") {
            if (inline_value) {
                return fail("--reset-settings takes no value");
            }
            opts.reset_settings = true;
        } else if (arg == "--log-level") {
            const auto value = take_value();
            const auto level = value ? log_level_from_name(*value) : std::nullopt;
            if (!level) {
                return fail("--log-level must be trace, debug, info, warn, error or off");
            }
            opts.log_level = level;
        } else if (arg == "--offscreen-smoke") {
            if (inline_value) {
                return fail("--offscreen-smoke takes no value");
            }
            opts.offscreen_smoke = true;
        } else if (looks_like_option(arg)) {
            return fail("unknown option " + arg);
        } else {
            if (opts.open_path) {
                return fail("more than one project to open");
            }
            opts.open_path = arg;
        }
    }

    if (opts.new_project && opts.open_path) {
        return fail("--new and --open are mutually exclusive");
    }
    return result;
}

}  // namespace alphashape::gui
