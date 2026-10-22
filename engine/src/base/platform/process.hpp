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

}  // namespace alphashape::platform
