#pragma once

// Runtime loading of a shared library, so the engine never links libvulkan /
// vulkan-1.dll and can report why the GPU layer is unavailable instead of
// failing to start.
//
// POSIX: dlopen(RTLD_NOW | RTLD_LOCAL) / dlsym / dlclose.
// Windows: a path containing a directory loads with
// LOAD_WITH_ALTERED_SEARCH_PATH, so its dependencies resolve beside it; a bare
// name is treated as a system component and loads with
// LOAD_LIBRARY_SEARCH_SYSTEM32, so neither the application directory nor the
// current working directory is ever searched for it (no DLL planting). An
// alternative loader named by bare name is therefore resolved against System32
// only -- pass an absolute path for a private one. GetProcAddress / FreeLibrary.
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

    /// Opens `path`: a UTF-8 path, or a bare library name (Windows: System32
    /// only; POSIX: the platform's own search). On failure returns a closed
    /// handle with *error set when it is non-null.
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
