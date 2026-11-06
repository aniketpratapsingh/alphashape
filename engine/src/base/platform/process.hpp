#pragma once

// Spawn a child process, capture its combined stdout+stderr and wait for it,
// without a shell. Used by tests (the Platform suite's child that prints and
// exits 3) and by the video path, which shells out to ffmpeg.
//
// POSIX: fork + execvp in the child with a pipe; Windows: CreateProcessW over
// a redirected pipe. Both read until EOF, so a child that writes more than the
// pipe buffer does not deadlock.

#include <string>
#include <utility>
#include <vector>

namespace alphashape::platform {

struct ProcessResult {
    /// True when the child was started; false when it could not be (then
    /// `output` carries the reason and `exit_code` is -1).
    bool started = false;
    /// True when the timeout killed the child; `exit_code` then reports what
    /// the kill produced (POSIX 128 + SIGKILL).
    bool timed_out = false;
    /// Exit status. On POSIX a child killed by a signal reports 128 + signo,
    /// matching the shell convention; a Windows process reports its exit code.
    int exit_code = -1;
    /// Everything the child wrote to stdout and stderr, interleaved in arrival
    /// order (both are redirected to one pipe).
    std::string output;
};

/// Extra environment for the child and an optional wall-clock limit.
struct ProcessOptions {
    /// Added to (or overriding) the inherited environment.
    std::vector<std::pair<std::string, std::string>> env;
    /// Kill the child after this many milliseconds; <= 0 means no limit.
    int timeout_ms = 0;
};

/// Runs `argv[0]` with `argv` (which must include argv[0] and a nullptr-like
/// end is added by the implementation) and waits for it.
ProcessResult run_process_capture(const std::vector<std::string>& argv);

/// Same, with extra environment and a timeout. On timeout the child and (on
/// POSIX) its process group are killed.
ProcessResult run_process_capture(const std::vector<std::string>& argv,
                                  const ProcessOptions& options);

/// Resolves `name` on PATH to an absolute path, or "" when it is not found.
/// A name that already contains a path separator is returned unchanged if it
/// names an executable.
std::string find_executable(const std::string& name);

/// True when a process with `pid` is running (or exists but belongs to another
/// user). Used by the project lock file to tell "the writer is alive" from "the
/// writer died and left a stale lock". A pid <= 0 is never alive.
bool process_is_alive(long long pid);

/// The local host name, or "" when the platform cannot report it. Stored in a
/// lock file so a lock from another machine is never broken as if its pid were
/// local.
std::string hostname();

/// Create `path` and write `text` to it only when the file does not already
/// exist. POSIX uses open(O_CREAT|O_EXCL); Windows uses CreateFileW(CREATE_NEW).
/// Returns false with `error` set when the file exists or cannot be written.
bool create_file_exclusive(const std::string& path, const std::string& text,
                           std::string* error);

}  // namespace alphashape::platform
