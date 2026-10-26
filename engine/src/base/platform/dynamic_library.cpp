#include "base/platform/dynamic_library.hpp"

#include <utility>

#if defined(_WIN32)
#include <windows.h>

// In every SDK this project builds against (Windows 10+ / KB2533623), but a
// build configured with an older NTDDI_VERSION would fail to compile without
// this rather than fall back to the vulnerable search order.
#ifndef LOAD_LIBRARY_SEARCH_SYSTEM32
#define LOAD_LIBRARY_SEARCH_SYSTEM32 0x00000800
#endif
#else
#include <dlfcn.h>
#endif

#include "base/platform/env.hpp"

namespace alphashape::platform {

DynamicLibrary::~DynamicLibrary() { close(); }

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : handle_(other.handle_), path_(std::move(other.path_)) {
    other.handle_ = nullptr;
}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
    if (this != &other) {
        close();
        handle_ = other.handle_;
        path_ = std::move(other.path_);
        other.handle_ = nullptr;
    }
    return *this;
}

#if defined(_WIN32)
namespace {

std::string last_error_text() {
    const DWORD code = ::GetLastError();
    if (code == 0) {
        return "no error reported";
    }
    wchar_t* buffer = nullptr;
    const DWORD length = ::FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<wchar_t*>(&buffer), 0, nullptr);
    std::string text;
    if (length != 0 && buffer != nullptr) {
        text = to_utf8(std::wstring(buffer, length));
        // FormatMessage appends a CRLF; the message is reused inside one line.
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) {
            text.pop_back();
        }
        ::LocalFree(buffer);
    }
    if (text.empty()) {
        text = "error " + std::to_string(static_cast<unsigned long>(code));
    } else {
        text += " (error " + std::to_string(static_cast<unsigned long>(code)) + ")";
    }
    return text;
}

}  // namespace
#endif

DynamicLibrary DynamicLibrary::open(const std::string& path, std::string* error) {
    if (path.empty()) {
        if (error != nullptr) {
            *error = "empty library path";
        }
        return DynamicLibrary();
    }
#if defined(_WIN32)
    const std::wstring wide = from_utf8(path);
    if (wide.empty()) {
        if (error != nullptr) {
            *error = "'" + path + "' is not a valid UTF-8 path";
        }
        return DynamicLibrary();
    }
    // Two deliberately different rules:
    //
    //   * A bare name ("vulkan-1.dll") is a system component, so it is loaded
    //     from System32 only. Plain LoadLibraryW would search the application
    //     directory and the current working directory first, so a planted
    //     vulkan-1.dll beside the exe (or in whatever directory the user
    //     happened to start us in) would win -- classic DLL planting, and it
    //     would also shadow the graphics driver's loader. This covers the
    //     ALPHASHAPE_VULKAN_LIBRARY override as well: an override that names a
    //     different loader must be an absolute path, because a bare name is
    //     always resolved against System32.
    //   * A path with a directory in it is loaded from there with its own
    //     dependencies beside it (the SDK's or a relocated loader's layout).
    //
    // SetDefaultDllDirectories is deliberately not called process-wide: Qt and
    // the vcpkg runtime DLLs still have to resolve out of the application
    // directory.
    const bool has_directory = path.find('\\') != std::string::npos || path.find('/') != std::string::npos;
    HMODULE module = nullptr;
    if (has_directory) {
        module = ::LoadLibraryExW(wide.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    } else {
        module = ::LoadLibraryExW(wide.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    }
    if (module == nullptr) {
        if (error != nullptr) {
            *error = "LoadLibrary '" + path + "' failed: " + last_error_text();
        }
        return DynamicLibrary();
    }
    return DynamicLibrary(reinterpret_cast<void*>(module), path);
#else
    void* handle = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        if (error != nullptr) {
            const char* detail = ::dlerror();
            *error = "dlopen '" + path + "' failed: " + (detail != nullptr ? detail : "unknown error");
        }
        return DynamicLibrary();
    }
    return DynamicLibrary(handle, path);
#endif
}

void* DynamicLibrary::symbol(const std::string& name, std::string* error) const {
    if (handle_ == nullptr) {
        if (error != nullptr) {
            *error = "no library is open";
        }
        return nullptr;
    }
    if (name.empty()) {
        if (error != nullptr) {
            *error = "empty symbol name";
        }
        return nullptr;
    }
#if defined(_WIN32)
    void* address = reinterpret_cast<void*>(
        ::GetProcAddress(reinterpret_cast<HMODULE>(handle_), name.c_str()));
    if (address == nullptr && error != nullptr) {
        *error = "'" + path_ + "' has no symbol '" + name + "': " + last_error_text();
    }
    return address;
#else
    // dlsym returns void*; POSIX guarantees the conversion is valid for a
    // function pointer in the platform's ABI (the tree already relies on it).
    dlerror();  // clear any stale error before probing
    void* address = ::dlsym(handle_, name.c_str());
    if (address == nullptr && error != nullptr) {
        const char* detail = ::dlerror();
        *error = "'" + path_ + "' has no symbol '" + name + "': " +
                 (detail != nullptr ? detail : "unknown error");
    }
    return address;
#endif
}

void DynamicLibrary::close() {
    if (handle_ == nullptr) {
        return;
    }
#if defined(_WIN32)
    // Failure here loses a module refcount; there is nothing a caller can do,
    // and the process exit reclaims it.
    (void)::FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
    (void)::dlclose(handle_);
#endif
    handle_ = nullptr;
}

}  // namespace alphashape::platform
