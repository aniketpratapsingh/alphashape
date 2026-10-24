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

/// Gives a GUI-subsystem Windows program the console it was started from.
///
/// Windows starts a /SUBSYSTEM:WINDOWS process with no standard handles even
/// when a console launched it, so `alphashape.exe --version` in a terminal
/// would print nothing anywhere. AttachConsole(ATTACH_PARENT_PROCESS) adopts
/// the parent's console and the CRT streams are reopened onto CONOUT$.
///
/// Returns false when there is no console to attach to (a shortcut, Explorer,
/// a service). On POSIX this returns whether the streams are usable and does
/// nothing else: a POSIX GUI program already inherits its parent's streams.
/// Redirected streams are left alone -- a pipe or a file handed over by
/// CreateProcess is inherited and valid, and replacing it with CONOUT$ would
/// lose the redirection the caller asked for. Call this first thing in main().
bool attach_parent_console();

}  // namespace alphashape::platform
