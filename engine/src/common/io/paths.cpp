#include "common/io/paths.hpp"

#include <algorithm>
#include <filesystem>

#include "base/platform/env.hpp"

namespace alphashape {

namespace fs = std::filesystem;

namespace {

std::string with_forward_slashes(const fs::path& path) {
    std::string text = platform::path_to_utf8(path);
    // Only Windows uses '\' as a separator; on POSIX a backslash is an
    // ordinary character in a file name and must survive storage.
    if constexpr (fs::path::preferred_separator == '\\') {
        std::replace(text.begin(), text.end(), '\\', '/');
    }
    return text;
}

/// A Windows drive prefix ("C:") or UNC prefix ("//server") is absolute even
/// when this build is running on POSIX, so a project written on Windows is
/// never mistaken for a relative path.
bool has_windows_root(const std::string& path) {
    if (path.size() >= 3 && path[1] == ':' && path[2] == '/' &&
        ((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z'))) {
        return true;
    }
    return path.size() >= 2 && path[0] == '/' && path[1] == '/';
}

}  // namespace

bool path_is_absolute(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    if (has_windows_root(path)) {
        return true;
    }
    return fs::path(path).is_absolute();
}

std::string path_for_storage(const std::string& path, const std::string& project_dir,
                             bool absolute_paths) {
    if (path.empty() || project_dir.empty()) {
        // An in-memory archive has no project location to normalise against,
        // and its bytes must not change: keep the model's path verbatim.
        return path;
    }
    const std::string normalised = with_forward_slashes(platform::path_from_utf8(path));
    if (absolute_paths) {
        // absolute_paths covers every stored path: a path that is relative in
        // memory is resolved against the project directory first, so the file
        // records a location a reader on any working directory can open.
        if (path_is_absolute(normalised)) {
            return normalised;
        }
        fs::path resolved =
            (platform::path_from_utf8(project_dir) /
             platform::path_from_utf8(path)).lexically_normal();
        return with_forward_slashes(resolved);
    }
    if (!path_is_absolute(normalised)) {
        return normalised;
    }
    // lexically_relative() returns an empty path when the two locations do not
    // share a root (another Windows drive, or a different POSIX root). That is
    // exactly the case where the path stays absolute.
    const fs::path relative =
        platform::path_from_utf8(normalised).lexically_relative(
            platform::path_from_utf8(project_dir));
    if (relative.empty() || relative.is_absolute()) {
        return normalised;
    }
    return with_forward_slashes(relative);
}

std::string path_from_storage(const std::string& stored, const std::string& project_dir) {
    if (stored.empty()) {
        return std::string();
    }
    fs::path parsed = platform::path_from_utf8(stored);
    // The in-memory model uses the platform's preferred separators (Windows
    // code that built a path with backslashes must compare equal after a round
    // trip); only the stored form is forced to '/'. u8string() preserves the
    // separators it was given, so the conversion is explicit.
    parsed.make_preferred();
    if (path_is_absolute(stored) || project_dir.empty()) {
        return platform::path_to_utf8(parsed);
    }
    fs::path joined = (platform::path_from_utf8(project_dir) / parsed).lexically_normal();
    joined.make_preferred();
    return platform::path_to_utf8(joined);
}

}  // namespace alphashape
