#pragma once

// Spawn a child process, capture its combined stdout+stderr and wait for it,
// without a shell. Used by tests (the Platform suite's child that prints and
// exits 3) and by the video path, which shells out to ffmpeg.
//
// POSIX: fork + execvp in the child with a pipe; Windows: CreateProcessW over
// a redirected pipe. Both read until EOF, so a child that writes more than the
// pipe buffer does not deadlock.

#include <string>
#include <vector>

namespace alphashape::platform {

struct ProcessResult {
    /// True when the child was started; false when it could not be (then
    /// `output` carries the reason and `exit_code` is -1).
    bool started = false;
    /// Exit status. On POSIX a child killed by a signal reports 128 + signo,
    /// matching the shell convention; a Windows process reports its exit code.
    int exit_code = -1;
    /// Everything the child wrote to stdout and stderr, interleaved in arrival
    /// order (both are redirected to one pipe).
    std::string output;
};

/// Runs `argv[0]` with `argv` (which must include argv[0] and a nullptr-like
/// end is added by the implementation) and waits for it.
ProcessResult run_process_capture(const std::vector<std::string>& argv);

/// True when a process with `pid` is running (or exists but belongs to another
/// user). Used by the project lock file to tell "the writer is alive" from "the
/// writer died and left a stale lock". A pid <= 0 is never alive.
bool process_is_alive(long long pid);

/// The local host name, or "" when the platform cannot report it. Stored in a
/// lock file so a lock from another machine is never broken as if its pid were
/// local.
std::string hostname();

}  // namespace alphashape::platform
