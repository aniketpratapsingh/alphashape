#include "base/utils/log.hpp"

#include <iostream>
#include <utility>

namespace alphashape {

LogLevel Log::level_ = LogLevel::Info;
std::mutex Log::mutex_;
Log::Sink Log::sink_;

void Log::set_level(LogLevel level) { level_ = level; }

LogLevel Log::level() { return level_; }

void Log::set_sink(Sink sink) {
    std::lock_guard<std::mutex> lock(mutex_);
    sink_ = std::move(sink);
}

bool Log::enabled(LogLevel level) { return level >= level_; }

namespace {
const char* level_name(LogLevel level) {
    switch (level) {
        case LogLevel::Trace: return "trace";
        case LogLevel::Debug: return "debug";
        case LogLevel::Info:  return "info";
        case LogLevel::Warn:  return "warn";
        case LogLevel::Error: return "error";
        case LogLevel::Off:   return "off";
    }
    return "?";
}
}  // namespace

void Log::write(LogLevel level, const std::string& message) {
    if (!enabled(level)) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (sink_) {
        sink_(level, message);
        return;
    }
    std::ostream& out = (level >= LogLevel::Warn) ? std::cerr : std::cout;
    out << '[' << level_name(level) << "] " << message << '\n';
    out.flush();
}

}  // namespace alphashape
