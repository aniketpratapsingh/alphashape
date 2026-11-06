#include "base/platform/process.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
#endif

#include "base/platform/env.hpp"

namespace alphashape::platform {

#if !defined(_WIN32)
namespace {

/// True when the descriptor is readable, EOF, or the deadline expired.
/// Returns false on a poll error other than EINTR.
bool wait_readable(int fd, int wait_ms, bool* eof, std::string* out) {
    struct pollfd entry{};
    entry.fd = fd;
    entry.events = POLLIN;
    const int rc = ::poll(&entry, 1, wait_ms);
    if (rc < 0) {
        return errno == EINTR;
    }
    if (rc == 0) {
        return true;
    }
    if ((entry.revents & (POLLIN | POLLHUP | POLLERR)) == 0) {
        return true;
    }
    char buffer[4096];
    for (;;) {
        const ssize_t got = ::read(fd, buffer, sizeof(buffer));
        if (got > 0) {
            out->append(buffer, static_cast<std::size_t>(got));
            continue;
        }
        if (got == 0) {
            *eof = true;
        }
        break;
    }
    return true;
}

std::string find_executable_posix(const std::string& name) {
    if (name.empty()) {
        return std::string();
    }
    if (name.find('/') != std::string::npos) {
        return ::access(name.c_str(), X_OK) == 0 ? name : std::string();
    }
    const char* path = std::getenv("PATH");
    if (path == nullptr) {
        return std::string();
    }
    const std::string dirs(path);
    std::size_t start = 0;
    for (;;) {
        const std::size_t colon = dirs.find(':', start);
        const std::string dir =
            dirs.substr(start, colon == std::string::npos ? std::string::npos
                                                          : colon - start);
        const std::string candidate = (dir.empty() ? "." : dir) + "/" + name;
        if (::access(candidate.c_str(), X_OK) == 0) {
            return candidate;
        }
        if (colon == std::string::npos) {
            break;
        }
        start = colon + 1;
    }
    return std::string();
}

/// The inherited environment with `extra` applied (replace or append).
std::vector<std::string> build_environment(
    const std::vector<std::pair<std::string, std::string>>& extra) {
    std::vector<std::string> entries;
    for (char** item = environ; item != nullptr && *item != nullptr; ++item) {
        entries.emplace_back(*item);
    }
    for (const auto& [key, value] : extra) {
        const std::string prefix = key + "=";
        bool replaced = false;
        for (std::string& entry : entries) {
            if (entry.compare(0, prefix.size(), prefix) == 0) {
                entry = prefix + value;
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            entries.push_back(prefix + value);
        }
    }
    return entries;
}

}  // namespace

ProcessResult run_process_capture(const std::vector<std::string>& argv) {
    return run_process_capture(argv, ProcessOptions());
}

ProcessResult run_process_capture(const std::vector<std::string>& argv,
                                  const ProcessOptions& options) {
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

    // Build everything the child needs before forking: only async-signal-safe
    // calls are allowed between fork and exec.
    const std::string executable = find_executable_posix(argv[0]);
    if (executable.empty()) {
        ::close(fds[0]);
        ::close(fds[1]);
        result.output = "cannot execute '" + argv[0] + "': not found on PATH\n";
        result.exit_code = 127;
        return result;
    }
    std::vector<char*> raw;
    raw.reserve(argv.size() + 1);
    for (const std::string& arg : argv) {
        raw.push_back(const_cast<char*>(arg.c_str()));
    }
    raw.push_back(nullptr);
    std::vector<std::string> environment = build_environment(options.env);
    std::vector<char*> envp;
    envp.reserve(environment.size() + 1);
    for (std::string& entry : environment) {
        envp.push_back(entry.data());
    }
    envp.push_back(nullptr);

    const pid_t pid = ::fork();
    if (pid < 0) {
        const std::string detail = std::string("fork failed: ") + std::strerror(errno);
        ::close(fds[0]);
        ::close(fds[1]);
        result.output = detail;
        return result;
    }
    if (pid == 0) {
        // Child: its own process group so a timeout can kill descendants too.
        (void)::setpgid(0, 0);
        ::dup2(fds[1], STDOUT_FILENO);
        ::dup2(fds[1], STDERR_FILENO);
        const int dev_null = ::open("/dev/null", O_RDONLY);
        if (dev_null >= 0) {
            ::dup2(dev_null, STDIN_FILENO);
            ::close(dev_null);
        }
        ::close(fds[0]);
        ::close(fds[1]);
        ::execve(executable.c_str(), raw.data(), envp.data());
        const std::string message =
            std::string("cannot execute '") + argv[0] + "': " + std::strerror(errno) + "\n";
        (void)!::write(STDERR_FILENO, message.data(), message.size());
        ::_exit(127);
    }

    (void)::setpgid(pid, pid);  // close the child's race to install the group
    ::close(fds[1]);
    (void)::fcntl(fds[0], F_SETFL, ::fcntl(fds[0], F_GETFL, 0) | O_NONBLOCK);

    const auto started = std::chrono::steady_clock::now();
    const auto deadline =
        options.timeout_ms > 0
            ? started + std::chrono::milliseconds(options.timeout_ms)
            : std::chrono::steady_clock::time_point::max();

    bool eof = false;
    int status = 0;
    bool reaped = false;
    while (!(reaped && eof)) {
        int wait_ms = 50;
        if (deadline != std::chrono::steady_clock::time_point::max()) {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now());
            const long long remaining = left.count();
            if (remaining <= 0) {
                if (!reaped) {
                    result.timed_out = true;
                    (void)::kill(-pid, SIGKILL);
                    (void)::kill(pid, SIGKILL);
                }
                break;
            }
            wait_ms = remaining > 50 ? 50 : static_cast<int>(remaining);
        }
        if (!eof) {
            (void)wait_readable(fds[0], wait_ms, &eof, &result.output);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
        }
        if (!reaped) {
            const pid_t waited = ::waitpid(pid, &status, WNOHANG);
            reaped = waited == pid || (waited < 0 && errno == ECHILD);
        }
    }
    ::close(fds[0]);
    if (!reaped) {
        while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
        }
    }
    if (result.timed_out) {
        result.exit_code = 128 + SIGKILL;
        result.started = true;
        return result;
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

/// The inherited environment with `extra` applied, as a CreateProcessW block.
std::wstring build_environment_block(
    const std::vector<std::pair<std::string, std::string>>& extra) {
    std::vector<std::wstring> entries;
    LPWCH block = ::GetEnvironmentStringsW();
    if (block != nullptr) {
        for (LPWCH item = block; *item != L'\0'; item += std::wcslen(item) + 1) {
            entries.emplace_back(item);
        }
        (void)::FreeEnvironmentStringsW(block);
    }
    for (const auto& [key, value] : extra) {
        const std::wstring prefix = from_utf8(key) + L"=";
        const std::wstring entry = prefix + from_utf8(value);
        bool replaced = false;
        for (std::wstring& existing : entries) {
            if (existing.compare(0, prefix.size(), prefix) == 0) {
                existing = entry;
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            entries.push_back(entry);
        }
    }
    std::sort(entries.begin(), entries.end());
    std::wstring out;
    for (const std::wstring& entry : entries) {
        out += entry;
        out.push_back(L'\0');
    }
    out.push_back(L'\0');
    return out;
}

}  // namespace

ProcessResult run_process_capture(const std::vector<std::string>& argv) {
    return run_process_capture(argv, ProcessOptions());
}

ProcessResult run_process_capture(const std::vector<std::string>& argv,
                                  const ProcessOptions& options) {
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

    // A job object kills the whole tree on timeout; TerminateProcess alone
    // leaves python's own children running.
    HANDLE job = ::CreateJobObjectW(nullptr, nullptr);
    if (job != nullptr) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (::SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits,
                                      static_cast<DWORD>(sizeof(limits))) == 0) {
            ::CloseHandle(job);
            job = nullptr;
        }
    }

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
    std::wstring environment = build_environment_block(options.env);

    const BOOL created =
        ::CreateProcessW(nullptr, mutable_line.data(), nullptr, nullptr, TRUE, 0,
                         environment.empty() ? nullptr : environment.data(), nullptr,
                         &startup, &process);
    ::CloseHandle(write_end);

    if (created == 0) {
        const DWORD code = ::GetLastError();
        ::CloseHandle(read_end);
        if (job != nullptr) {
            ::CloseHandle(job);
        }
        result.output = "CreateProcess '" + argv[0] + "' failed (error " +
                        std::to_string(static_cast<unsigned long>(code)) + ")";
        return result;
    }
    if (job != nullptr && ::AssignProcessToJobObject(job, process.hProcess) == 0) {
        ::CloseHandle(job);
        job = nullptr;
    }

    // Read concurrently: a child that fills the pipe would otherwise block
    // forever while the parent waits.
    std::thread reader([&result, read_end]() {
        char buffer[4096];
        for (;;) {
            DWORD got = 0;
            const BOOL ok = ::ReadFile(read_end, buffer, sizeof(buffer), &got, nullptr);
            if (!ok || got == 0) {
                break;
            }
            result.output.append(buffer, got);
        }
    });

    const DWORD timeout =
        options.timeout_ms > 0 ? static_cast<DWORD>(options.timeout_ms) : INFINITE;
    if (::WaitForSingleObject(process.hProcess, timeout) == WAIT_TIMEOUT) {
        result.timed_out = true;
        const UINT kill_code = static_cast<UINT>(1);
        if (job != nullptr) {
            (void)::TerminateJobObject(job, kill_code);
        } else {
            (void)::TerminateProcess(process.hProcess, kill_code);
        }
    }
    (void)::WaitForSingleObject(process.hProcess, INFINITE);
    reader.join();
    ::CloseHandle(read_end);

    DWORD exit_code = 0;
    (void)::GetExitCodeProcess(process.hProcess, &exit_code);
    ::CloseHandle(process.hThread);
    ::CloseHandle(process.hProcess);
    if (job != nullptr) {
        ::CloseHandle(job);
    }

    result.exit_code = static_cast<int>(exit_code);
    result.started = true;
    return result;
}

#endif

std::string find_executable(const std::string& name) {
    if (name.empty()) {
        return std::string();
    }
#if defined(_WIN32)
    const std::wstring wide = from_utf8(name);
    if (wide.empty()) {
        return std::string();
    }
    wchar_t buffer[32768];
    const DWORD capacity =
        static_cast<DWORD>(sizeof(buffer) / sizeof(buffer[0]));
    const DWORD length = ::SearchPathW(nullptr, wide.c_str(), L".exe", capacity,
                                       buffer, nullptr);
    if (length == 0 || length >= capacity) {
        return std::string();
    }
    return to_utf8(std::wstring(buffer, length));
#else
    return find_executable_posix(name);
#endif
}

bool process_is_alive(long long pid) {
    if (pid <= 0) {
        return false;
    }
#if defined(_WIN32)
    const DWORD id = static_cast<DWORD>(pid);
    HANDLE handle = ::OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, id);
    if (handle == nullptr) {
        // Access denied means the process exists but belongs to another user;
        // ERROR_INVALID_PARAMETER means no such process.
        return ::GetLastError() == ERROR_ACCESS_DENIED;
    }
    DWORD exit_code = 0;
    const BOOL ok = ::GetExitCodeProcess(handle, &exit_code);
    ::CloseHandle(handle);
    return ok != 0 && exit_code == STILL_ACTIVE;
#else
    // A pid outside pid_t would wrap (possibly negative, which names a process
    // group), so it cannot be a live process.
    if (pid > static_cast<long long>(std::numeric_limits<pid_t>::max())) {
        return false;
    }
    // Signal 0 performs the permission and existence checks without sending
    // anything. EPERM means the process exists but belongs to another user.
    const int rc = ::kill(static_cast<pid_t>(pid), 0);
    return rc == 0 || errno == EPERM;
#endif
}

bool create_file_exclusive(const std::string& path, const std::string& text,
                           std::string* error) {
#if defined(_WIN32)
    const std::wstring wide = from_utf8(path);
    if (wide.empty()) {
        *error = "path is not valid UTF-8";
        return false;
    }
    HANDLE handle = ::CreateFileW(wide.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                                  FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        const DWORD code = ::GetLastError();
        *error = "CreateFile failed (error " +
                 std::to_string(static_cast<unsigned long>(code)) + ")";
        return false;
    }
    std::size_t written = 0;
    while (written < text.size()) {
        const std::size_t left = text.size() - written;
        const DWORD chunk =
            static_cast<DWORD>(left > 0x7fffffffU ? 0x7fffffffU : left);
        DWORD put = 0;
        const BOOL ok = ::WriteFile(handle, text.data() + written, chunk, &put, nullptr);
        if (ok == 0 || put == 0) {
            ::CloseHandle(handle);
            (void)::DeleteFileW(wide.c_str());
            *error = "short write";
            return false;
        }
        written += static_cast<std::size_t>(put);
    }
    ::CloseHandle(handle);
    return true;
#else
    const int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd < 0) {
        *error = std::strerror(errno);
        return false;
    }
    std::size_t written = 0;
    while (written < text.size()) {
        const ssize_t put =
            ::write(fd, text.data() + written, text.size() - written);
        if (put < 0) {
            if (errno == EINTR) {
                continue;
            }
            *error = std::strerror(errno);
            ::close(fd);
            (void)::unlink(path.c_str());
            return false;
        }
        if (put == 0) {
            *error = "short write";
            ::close(fd);
            (void)::unlink(path.c_str());
            return false;
        }
        written += static_cast<std::size_t>(put);
    }
    ::close(fd);
    return true;
#endif
}

std::string hostname() {
#if defined(_WIN32)
    wchar_t buffer[256];
    DWORD size = static_cast<DWORD>(sizeof(buffer) / sizeof(buffer[0]));
    if (::GetComputerNameW(buffer, &size) == 0) {
        return std::string();
    }
    return to_utf8(std::wstring(buffer, size));
#else
    char buffer[256];
    if (::gethostname(buffer, sizeof(buffer)) != 0) {
        return std::string();
    }
    buffer[sizeof(buffer) - 1] = '\0';
    return std::string(buffer);
#endif
}

}  // namespace alphashape::platform
