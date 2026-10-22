#include "base/platform/console.hpp"

#include <cstdio>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "base/platform/env.hpp"

namespace alphashape::platform {

namespace {

#if defined(_WIN32)
int win_handle_for(Stream stream) {
    return stream == Stream::Stdout ? _fileno(stdout) : _fileno(stderr);
}
#else
int posix_fd_for(Stream stream) { return stream == Stream::Stdout ? STDOUT_FILENO : STDERR_FILENO; }
#endif

}  // namespace

bool is_tty(Stream stream) {
#if defined(_WIN32)
    const int fd = win_handle_for(stream);
    if (fd < 0) {
        return false;
    }
    // A stream is interactive when its handle is a character device: a console,
    // or a Windows terminal pty (which OpenSSH and ConPTY both report as a
    // character device). A pipe or a file is not.
    const intptr_t raw = _get_osfhandle(fd);
    if (raw == -1) {
        return false;
    }
    const HANDLE handle = reinterpret_cast<HANDLE>(raw);
    DWORD mode = 0;
    if (::GetConsoleMode(handle, &mode) != 0) {
        return true;
    }
    const DWORD type = ::GetFileType(handle);
    return type == FILE_TYPE_CHAR;
#else
    const int fd = posix_fd_for(stream);
    return ::isatty(fd) != 0;
#endif
}

void set_binary_mode(std::FILE* stream) {
    if (stream == nullptr) {
        return;
    }
#if defined(_WIN32)
    (void)_setmode(_fileno(stream), _O_BINARY);
#else
    // POSIX has no text mode; the call exists so call sites read the same on
    // every platform and nobody reintroduces a platform #ifdef.
    (void)stream;
#endif
}

void setup_console_utf8() {
#if defined(_WIN32)
    (void)::SetConsoleOutputCP(CP_UTF8);
    (void)::SetConsoleCP(CP_UTF8);
    // Enable ANSI escapes on the attached consoles. When the streams are pipes
    // (CI, a redirected log) there is no console and this is a no-op.
    for (DWORD std_handle : {static_cast<DWORD>(STD_OUTPUT_HANDLE),
                             static_cast<DWORD>(STD_ERROR_HANDLE)}) {
        const HANDLE handle = ::GetStdHandle(std_handle);
        if (handle == nullptr || handle == INVALID_HANDLE_VALUE) {
            continue;
        }
        DWORD mode = 0;
        if (::GetConsoleMode(handle, &mode) == 0) {
            continue;
        }
        (void)::SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
#endif
}

}  // namespace alphashape::platform
