#pragma once

// Runtime loading of a shared library, so the engine never links libvulkan /
// vulkan-1.dll and can report why the GPU layer is unavailable instead of
// failing to start.
//
// POSIX: dlopen(RTLD_NOW | RTLD_LOCAL) / dlsym / dlclose.
// Windows: LoadLibraryExW with LOAD_WITH_ALTERED_SEARCH_PATH, so a loader
// resolved next to the executable wins over a stale one in the system
// directories; GetProcAddress / FreeLibrary.
//
// The handle is an opaque void* so no caller needs the platform header. A
// null handle is never valid.

#include <string>

namespace alphashape::platform {

class DynamicLibrary {
public:
    DynamicLibrary() = default;
    ~DynamicLibrary();

    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;
    DynamicLibrary(DynamicLibrary&& other) noexcept;
    DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;

    /// Opens `path` (a UTF-8 path, or a bare library name so the platform's own
    /// search applies). On failure returns "" — the path form is more useful
    /// than a name that may then be searched again. An error message is
    /// written to *error when it is non-null.
    static DynamicLibrary open(const std::string& path, std::string* error);

    /// Resolves one symbol. Returns nullptr when the library is closed or the
    /// symbol is absent; *error is filled if it is non-null.
    void* symbol(const std::string& name, std::string* error) const;

    void close();
    bool is_open() const { return handle_ != nullptr; }
    const std::string& path() const { return path_; }

private:
    DynamicLibrary(void* handle, std::string path) : handle_(handle), path_(std::move(path)) {}

    void* handle_ = nullptr;
    std::string path_;
};

}  // namespace alphashape::platform
