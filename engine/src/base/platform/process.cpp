#include "base/platform/process.hpp"

#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "base/platform/env.hpp"

namespace alphashape::platform {

#if !defined(_WIN32)
namespace {

/// Drains `fd` until EOF into `out`. Called in the parent after the child's
/// inherited descriptors are closed, so EOF means the child exited or closed
/// the pipe deliberately.
void read_until_eof(int fd, std::string* out) {
    char buffer[4096];
    for (;;) {
        const ssize_t got = ::read(fd, buffer, sizeof(buffer));
        if (got > 0) {
            out->append(buffer, static_cast<std::size_t>(got));
            continue;
        }
        if (got == 0) {
            return;
        }
        if (errno == EINTR) {
            continue;
        }
        return;
    }
}

}  // namespace

ProcessResult run_process_capture(const std::vector<std::string>& argv) {
    ProcessResult result;
    if (argv.empty() || argv[0].empty()) {
        result.output = "run_process_capture: empty argv";
        return result;
    }

    int fds[2] = {-1, -1};
    if (::pipe(fds) != 0) {
        result.output = std::string("pipe failed: ") + std::strerror(errno);
        return result;
    }

    // Build the argv block before forking: only async-signal-safe calls are
    // allowed between fork and exec.
    std::vector<char*> raw;
    raw.reserve(argv.size() + 1);
    for (const std::string& arg : argv) {
        raw.push_back(const_cast<char*>(arg.c_str()));
    }
    raw.push_back(nullptr);

    const pid_t pid = ::fork();
    if (pid < 0) {
        const std::string detail = std::string("fork failed: ") + std::strerror(errno);
        ::close(fds[0]);
        ::close(fds[1]);
        result.output = detail;
        return result;
    }
    if (pid == 0) {
        // Child: stderr follows stdout into the pipe, and stdin is /dev/null so
        // a child that reads does not hang the wait.
        ::dup2(fds[1], STDOUT_FILENO);
        ::dup2(fds[1], STDERR_FILENO);
        const int dev_null = ::open("/dev/null", O_RDONLY);
        if (dev_null >= 0) {
            ::dup2(dev_null, STDIN_FILENO);
            ::close(dev_null);
        }
        ::close(fds[0]);
        ::close(fds[1]);
        ::execvp(raw[0], raw.data());
        // exec failed: report it on the pipe (which becomes the child output)
        // and exit with the conventional "command not found" status.
        const std::string message =
            std::string("cannot execute '") + argv[0] + "': " + std::strerror(errno) + "\n";
        (void)!::write(STDERR_FILENO, message.data(), message.size());
        ::_exit(127);
    }

    ::close(fds[1]);
    read_until_eof(fds[0], &result.output);
    ::close(fds[0]);

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result.exit_code = 128 + WTERMSIG(status);
    } else {
        result.exit_code = -1;
    }
    result.started = true;
    return result;
}

#else  // _WIN32

namespace {

/// Quotes one argument for the CommandLineToArgvW convention that every
/// Windows program (including the MSVC CRT) parses: wrap in quotes when the
/// argument is empty or contains a space/tab/quote, and escape embedded quotes
/// and the backslashes that precede one.
std::wstring quote_argument(const std::wstring& arg) {
    const bool needs_quotes =
        arg.empty() || arg.find_first_of(L" \t\"") != std::wstring::npos;
    if (!needs_quotes) {
        return arg;
    }
    std::wstring out;
    out.reserve(arg.size() + 2);
    out.push_back(L'"');
    std::size_t backslashes = 0;
    for (const wchar_t c : arg) {
        if (c == L'\\') {
            ++backslashes;
            continue;
        }
        if (c == L'"') {
            // 2n+1 backslashes then a quote: n backslashes stay literal and the
            // quote is escaped.
            out.append(backslashes * 2 + 1, L'\\');
            out.push_back(L'"');
            backslashes = 0;
            continue;
        }
        out.append(backslashes, L'\\');
        backslashes = 0;
        out.push_back(c);
    }
    // Trailing backslashes would escape the closing quote, so double them.
    out.append(backslashes * 2, L'\\');
    out.push_back(L'"');
    return out;
}

std::wstring build_command_line(const std::vector<std::string>& argv, std::string* error) {
    std::wstring line;
    for (std::size_t i = 0; i < argv.size(); ++i) {
        const std::wstring wide = from_utf8(argv[i]);
        if (wide.empty()) {
            *error = "run_process_capture: argument " + std::to_string(i) + " is not valid UTF-8";
            return std::wstring();
        }
        if (i != 0) {
            line.push_back(L' ');
        }
        line += quote_argument(wide);
    }
    return line;
}

}  // namespace

ProcessResult run_process_capture(const std::vector<std::string>& argv) {
    ProcessResult result;
    if (argv.empty() || argv[0].empty()) {
        result.output = "run_process_capture: empty argv";
        return result;
    }

    std::string conversion_error;
    std::wstring command_line = build_command_line(argv, &conversion_error);
    if (!conversion_error.empty()) {
        result.output = conversion_error;
        return result;
    }

    // Inheritable write end, non-inheritable read end. The child gets stdout and
    // stderr on the write end; the parent keeps the read end and closes its copy
    // of the write end, so EOF means the child exited.
    SECURITY_ATTRIBUTES attributes{};
    attributes.nLength = sizeof(attributes);
    attributes.bInheritHandle = TRUE;
    HANDLE read_end = nullptr;
    HANDLE write_end = nullptr;
    if (::CreatePipe(&read_end, &write_end, &attributes, 0) == 0) {
        result.output = "CreatePipe failed (error " +
                        std::to_string(static_cast<unsigned long>(::GetLastError())) + ")";
        return result;
    }
    (void)::SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = ::GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = write_end;
    startup.hStdError = write_end;
    PROCESS_INFORMATION process{};

    // CreateProcessW may modify the command line buffer, hence the mutable
    // vector. The application name is null so the first token is resolved
    // through the standard search path exactly as a shell would resolve it
    // (a bare "cmd.exe", a PATH entry, or an absolute path).
    std::vector<wchar_t> mutable_line(command_line.begin(), command_line.end());
    mutable_line.push_back(L'\0');

    const BOOL created = ::CreateProcessW(nullptr, mutable_line.data(), nullptr, nullptr,
                                          TRUE, 0, nullptr, nullptr, &startup, &process);
    ::CloseHandle(write_end);

    if (created == 0) {
        const DWORD code = ::GetLastError();
        ::CloseHandle(read_end);
        result.output = "CreateProcess '" + argv[0] + "' failed (error " +
                        std::to_string(static_cast<unsigned long>(code)) + ")";
        return result;
    }

    // Read before WaitForSingleObject: a child that fills the pipe would
    // otherwise block forever while the parent waits.
    char buffer[4096];
    for (;;) {
        DWORD got = 0;
        const BOOL ok = ::ReadFile(read_end, buffer, sizeof(buffer), &got, nullptr);
        if (!ok || got == 0) {
            break;
        }
        result.output.append(buffer, got);
    }
    ::CloseHandle(read_end);

    (void)::WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 0;
    (void)::GetExitCodeProcess(process.hProcess, &exit_code);
    ::CloseHandle(process.hThread);
    ::CloseHandle(process.hProcess);

    result.exit_code = static_cast<int>(exit_code);
    result.started = true;
    return result;
}

#endif

}  // namespace alphashape::platform
