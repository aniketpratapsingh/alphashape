#include "base/platform/paths.hpp"

#include <string>

#if defined(_WIN32)
#include <windows.h>

#include <vector>
#else
#include <dlfcn.h>
#endif

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#elif !defined(_WIN32)
#include <unistd.h>
#endif

#include "base/platform/env.hpp"

namespace alphashape::platform {

#if defined(_WIN32)
namespace {

/// GetModuleFileNameW with a growing buffer. It reports the length it copied
/// and truncates silently when the buffer is too small (the pre-Vista contract
/// it still documents), so a full buffer means "grow and ask again" rather
/// than "this is the whole path".
std::string module_file_name(HMODULE module) {
    std::vector<wchar_t> buffer(MAX_PATH + 1);
    for (;;) {
        const DWORD copied = ::GetModuleFileNameW(module, buffer.data(),
                                                  static_cast<DWORD>(buffer.size()));
        if (copied == 0) {
            return std::string();
        }
        if (copied < buffer.size() - 1) {
            return to_utf8(std::wstring(buffer.data(), copied));
        }
        // Possibly truncated: double and retry. A path that fills the buffer
        // exactly twice is pathological, but the loop is bounded by the path
        // limit Windows itself enforces.
        if (buffer.size() >= 65536) {
            return to_utf8(std::wstring(buffer.data(), copied));
        }
        buffer.resize(buffer.size() * 2);
    }
}

}  // namespace
#endif

std::string executable_path() {
#if defined(_WIN32)
    return module_file_name(nullptr);
#elif defined(__APPLE__)
    // _NSGetExecutablePath wants the buffer size in bytes on input and writes
    // the size it needed on failure, which is the documented two-call dance.
    uint32_t size = 0;
    if (_NSGetExecutablePath(nullptr, &size) != -1 || size == 0) {
        return std::string();
    }
    std::string buffer(size, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        return std::string();
    }
    return std::string(buffer.c_str());
#else
    // /proc/self/exe is a symlink to the running image; readlink does not
    // terminate the string, and the link target can change between the size
    // probe and the read, so retry while the result fills the buffer.
    for (std::size_t size = 256; size <= 65536; size *= 2) {
        std::string buffer(size, '\0');
        const ssize_t length = ::readlink("/proc/self/exe", buffer.data(), buffer.size());
        if (length <= 0) {
            return std::string();
        }
        if (static_cast<std::size_t>(length) < buffer.size()) {
            buffer.resize(static_cast<std::size_t>(length));
            return buffer;
        }
    }
    return std::string();
#endif
}

std::string module_path(const void* anchor) {
#if defined(_WIN32)
    if (anchor == nullptr) {
        return std::string();
    }
    HMODULE module = nullptr;
    // FROM_ADDRESS resolves the module that owns an address; UNCHANGED_REFCOUNT
    // keeps the loader's reference count alone, so there is nothing to free.
    if (::GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                 GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                             static_cast<LPCWSTR>(anchor), &module) == 0) {
        return std::string();
    }
    return module_file_name(module);
#else
    if (anchor == nullptr) {
        return std::string();
    }
    Dl_info info{};
    if (::dladdr(anchor, &info) == 0 || info.dli_fname == nullptr) {
        return std::string();
    }
    return std::string(info.dli_fname);
#endif
}

}  // namespace alphashape::platform
