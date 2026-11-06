// posix_spawn_file_actions_addclosefrom_np and pipe2 are GNU extensions; the
// macro must precede every system header.
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "base/platform/process.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <limits>
#include <mutex>
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
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

#if defined(__APPLE__) && !defined(POSIX_SPAWN_CLOEXEC_DEFAULT)
// <spawn.h> hides the flag below __DARWIN_C_FULL; the value is stable in Darwin.
#define POSIX_SPAWN_CLOEXEC_DEFAULT 0x4000
#endif
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

/// Serialises pipe creation and spawn on platforms whose pipe() has no
/// atomic close-on-exec flag: between pipe() and fcntl() another spawn could
/// otherwise inherit the descriptors.
std::mutex& spawn_mutex() {
    static std::mutex mutex;
    return mutex;
}

/// Creates the capture pipe with close-on-exec on both ends, so neither end
/// leaks into an unrelated concurrently spawned child.
bool create_capture_pipe(int fds[2], std::string* error) {
#if defined(__linux__)
    if (::pipe2(fds, O_CLOEXEC) != 0) {
        *error = std::string("pipe failed: ") + std::strerror(errno);
        return false;
    }
#else
    if (::pipe(fds) != 0) {
        *error = std::string("pipe failed: ") + std::strerror(errno);
        return false;
    }
    // macOS has no pipe2; the flag is set while spawn_mutex() is held so no
    // other spawn can observe the window.
    for (int index = 0; index < 2; ++index) {
        (void)::fcntl(fds[index], F_SETFD, FD_CLOEXEC);
    }
#endif
    return true;
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

    // Held until posix_spawn returns: macOS sets FD_CLOEXEC after pipe().
    std::unique_lock<std::mutex> spawn_lock{spawn_mutex()};
    int fds[2] = {-1, -1};
    std::string pipe_error;
    if (!create_capture_pipe(fds, &pipe_error)) {
        result.output = pipe_error;
        return result;
    }

    // Resolve the executable in the parent: posix_spawn reports an exec failure
    // as an error code, so the child needs no fallback path of its own.
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

    posix_spawn_file_actions_t actions;
    (void)::posix_spawn_file_actions_init(&actions);
    (void)::posix_spawn_file_actions_adddup2(&actions, fds[1], STDOUT_FILENO);
    (void)::posix_spawn_file_actions_adddup2(&actions, fds[1], STDERR_FILENO);
    (void)::posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null",
                                             O_RDONLY, 0);
    // Only 0/1/2 may reach the script: close both pipe ends explicitly, then
    // close every other descriptor where the platform supports it in one step.
    if (fds[0] > STDERR_FILENO) {
        (void)::posix_spawn_file_actions_addclose(&actions, fds[0]);
    }
    if (fds[1] > STDERR_FILENO) {
        (void)::posix_spawn_file_actions_addclose(&actions, fds[1]);
    }
    short spawn_flags = POSIX_SPAWN_SETPGROUP;
#if defined(__APPLE__) && defined(POSIX_SPAWN_CLOEXEC_DEFAULT)
    spawn_flags = static_cast<short>(spawn_flags | POSIX_SPAWN_CLOEXEC_DEFAULT);
#elif defined(__linux__) && defined(__GLIBC_PREREQ)
#if __GLIBC_PREREQ(2, 34)
    (void)::posix_spawn_file_actions_addclosefrom_np(&actions, 3);
#endif
#endif

    posix_spawnattr_t attributes;
    (void)::posix_spawnattr_init(&attributes);
    (void)::posix_spawnattr_setflags(&attributes, spawn_flags);
    // pgroup 0: the child leads a new group, so a timeout kills descendants.
    (void)::posix_spawnattr_setpgroup(&attributes, 0);

    pid_t pid = 0;
    const int spawn_error =
        ::posix_spawn(&pid, executable.c_str(), &actions, &attributes,
                      raw.data(), envp.data());
    (void)::posix_spawnattr_destroy(&attributes);
    (void)::posix_spawn_file_actions_destroy(&actions);
    spawn_lock.unlock();
    if (spawn_error != 0) {
        ::close(fds[0]);
        ::close(fds[1]);
        // posix_spawn reports an exec failure as an errno in the parent.
        result.output = "cannot execute '" + argv[0] + "': " +
                        std::strerror(spawn_error) + "\n";
        result.exit_code = 127;
        return result;
    }
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
        // An empty argument is valid and must reach the child as ""; only a
        // non-empty string that failed to convert is an error.
        if (wide.empty() && !argv[i].empty()) {
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

    // Both pipe ends are created non-inheritable. Only the child's two write
    // handles (and a NUL stdin) are named in PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
    // so no unrelated inheritable handle crosses into the child even when two
    // run_process_capture calls overlap.
    HANDLE read_end = nullptr;
    HANDLE write_end = nullptr;
    if (::CreatePipe(&read_end, &write_end, nullptr, 0) == 0) {
        result.output = "CreatePipe failed (error " +
                        std::to_string(static_cast<unsigned long>(::GetLastError())) + ")";
        return result;
    }
    (void)::SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0);
    (void)::SetHandleInformation(write_end, HANDLE_FLAG_INHERIT,
                                 HANDLE_FLAG_INHERIT);
    HANDLE null_input = ::CreateFileW(L"NUL", GENERIC_READ,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (null_input == INVALID_HANDLE_VALUE) {
        const DWORD code = ::GetLastError();
        ::CloseHandle(read_end);
        ::CloseHandle(write_end);
        result.output = "CreateFile NUL failed (error " +
                        std::to_string(static_cast<unsigned long>(code)) + ")";
        return result;
    }
    (void)::SetHandleInformation(null_input, HANDLE_FLAG_INHERIT,
                                 HANDLE_FLAG_INHERIT);

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

    // Exactly these handles may cross the boundary: NUL stdin plus the one
    // write end shared by stdout and stderr.
    HANDLE inherited[2] = {null_input, write_end};
    SIZE_T attribute_size = 0;
    (void)::InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_size);
    std::vector<unsigned char> attribute_storage(attribute_size);
    auto* attribute_list =
        reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attribute_storage.data());
    const bool initialized =
        attribute_size != 0 &&
        ::InitializeProcThreadAttributeList(attribute_list, 1, 0,
                                            &attribute_size) != 0;
    const bool listed =
        initialized &&
        ::UpdateProcThreadAttribute(attribute_list, 0,
                                    PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited,
                                    sizeof(inherited), nullptr, nullptr) != 0;
    if (!listed) {
        const DWORD code = ::GetLastError();
        if (initialized) {
            ::DeleteProcThreadAttributeList(attribute_list);
        }
        ::CloseHandle(read_end);
        ::CloseHandle(write_end);
        ::CloseHandle(null_input);
        if (job != nullptr) {
            ::CloseHandle(job);
        }
        result.output = "InitializeProcThreadAttributeList failed (error " +
                        std::to_string(static_cast<unsigned long>(code)) + ")";
        return result;
    }

    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = null_input;
    startup.StartupInfo.hStdOutput = write_end;
    startup.StartupInfo.hStdError = write_end;
    startup.lpAttributeList = attribute_list;
    PROCESS_INFORMATION process{};

    // CreateProcessW may modify the command line buffer, hence the mutable
    // vector. The application name is null so the first token is resolved
    // through the standard search path exactly as a shell would resolve it
    // (a bare "cmd.exe", a PATH entry, or an absolute path).
    std::vector<wchar_t> mutable_line(command_line.begin(), command_line.end());
    mutable_line.push_back(L'\0');
    std::wstring environment = build_environment_block(options.env);

    // CREATE_UNICODE_ENVIRONMENT is required when a Unicode environment block
    // is supplied; without it CreateProcessW rejects the block (error 87).
    const BOOL created =
        ::CreateProcessW(nullptr, mutable_line.data(), nullptr, nullptr, TRUE,
                         EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT,
                         environment.empty() ? nullptr : environment.data(), nullptr,
                         &startup.StartupInfo, &process);
    ::DeleteProcThreadAttributeList(attribute_list);
    ::CloseHandle(write_end);
    ::CloseHandle(null_input);

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
