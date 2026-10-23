#pragma once

// Minimal logging facade. Deliberately tiny: a level, a stream, thread-safe.
// Progress reporting for long-running tasks is separate (see tasks/progress.hpp)
// because it reaches the GUI as a callback, not as log text.

#include <functional>
#include <mutex>
#include <sstream>
#include <string>

namespace alphashape {

enum class LogLevel { Trace = 0, Debug, Info, Warn, Error, Off };

class Log {
public:
    static void set_level(LogLevel level);
    static LogLevel level();

    /// Thread-safe write of one already-formatted line.
    static void write(LogLevel level, const std::string& message);

    /// Receives every emitted line instead of the console. Called with the logger's mutex held
    /// from whichever thread logs, so it must be quick and must not log itself. An empty
    /// function restores console output. A front end uses this to show engine log lines.
    using Sink = std::function<void(LogLevel level, const std::string& message)>;
    static void set_sink(Sink sink);

    /// True if a message at this level would be emitted; lets callers avoid
    /// building expensive strings.
    static bool enabled(LogLevel level);

private:
    static LogLevel level_;
    static std::mutex mutex_;
    static Sink sink_;
};

namespace detail {
/// Accumulates a line and emits it on destruction, so callers can stream:
///   log(LogLevel::Info) << "loaded " << n << " cameras";
class LogLine {
public:
    explicit LogLine(LogLevel level) : level_(level) {}
    ~LogLine() { Log::write(level_, stream_.str()); }

    LogLine(const LogLine&) = delete;
    LogLine& operator=(const LogLine&) = delete;

    template <typename T>
    LogLine& operator<<(const T& value) {
        stream_ << value;
        return *this;
    }

private:
    LogLevel level_;
    std::ostringstream stream_;
};
}  // namespace detail

inline detail::LogLine log(LogLevel level) { return detail::LogLine(level); }

}  // namespace alphashape
