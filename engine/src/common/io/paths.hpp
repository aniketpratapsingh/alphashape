#pragma once

// Frame-path storage rules for a project container (WP-11.9).
//
// doc.xml and frames.bin store each frame's `path`. A project is meant to be
// moved together with its images, so the default is a path relative to the
// project file. A path that cannot be expressed relative to the project (a
// different Windows drive or POSIX root) and every path in a project saved with
// `absolute_paths=true` is stored absolutely instead.
//
// Stored paths always use '/' separators, so a project written on Windows opens
// on POSIX and vice versa. A Windows drive-qualified path ("C:/a/b.jpg") is
// kept absolute on any platform; only a same-root path is made relative.

#include <string>

namespace alphashape {

/// Rewrite `path` for storage in `project_dir`. An empty project directory (an
/// unsaved document) or a path that cannot be made relative is stored absolute.
/// Separators are normalised to '/'.
std::string path_for_storage(const std::string& path, const std::string& project_dir,
                             bool absolute_paths);

/// Resolve a stored path against `project_dir`. An absolute stored path is
/// returned unchanged (and still uses '/' separators); a relative one is joined
/// with the project directory. An empty stored path returns empty.
std::string path_from_storage(const std::string& stored, const std::string& project_dir);

/// True when `path` names an absolute location: a POSIX root, a Windows drive
/// or UNC prefix, or (only on Windows) an `std::filesystem` absolute.
bool path_is_absolute(const std::string& path);

}  // namespace alphashape
