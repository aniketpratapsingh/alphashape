#pragma once

// Terminal behaviour behind one interface: is a stream a terminal, are we
// writing to a console or a pipe, and how do we get UTF-8 out of a Windows
// console.
//
// The progress bar (CLI, Python binder) draws only when stderr is a terminal.
// Before this header the CLI spelled that as isatty(STDERR_FILENO), which does
// not exist on Windows.

#include <cstdio>

namespace alphashape::platform {

enum class Stream { Stdout, Stderr };

/// True when the stream is attached to a terminal (a console on Windows; a tty
/// on POSIX). False when redirected to a file or a pipe.
bool is_tty(Stream stream);

/// Puts the CRT's stdout/stderr into binary mode so no "\n" is rewritten to
/// "\r\n". The engine writes project archives and byte-identical products that
/// must not pass through text translation. A console stream is unaffected on
/// POSIX (there is no text mode).
void set_binary_mode(std::FILE* stream);

/// Installs UTF-8 as the console code page and enables
/// ENABLE_VIRTUAL_TERMINAL_PROCESSING so ANSI escapes (used by the progress
/// line) work on a Windows console. A no-op on POSIX, where the locale is the
/// caller's business. Safe to call more than once.
void setup_console_utf8();

}  // namespace alphashape::platform
