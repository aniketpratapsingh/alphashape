#pragma once

// Paths to the running image: the executable, and the shared library or
// extension module that a given code address lives in.
//
// The engine needs both to find its data directory next to itself:
//
//   * <exe_dir>/share/proj            a normal Windows or Linux install
//                                       (bin/alphashape.exe -> share/proj)
//   * <module_dir>/../share/proj      python/alphashape.pyd in the same install
//   * Contents/Resources/share/proj   the macOS app bundle
//
// Every Windows API call for this lives here, so no other directory names
// GetModuleFileName or dladdr (the rule stated in src/base/platform/CMakeLists.txt).

#include <string>

namespace alphashape::platform {

/// The full path of the running executable, as UTF-8, or "" when the platform
/// cannot report it. The path is as the loader named it (it may be relative or
/// contain a symlink).
std::string executable_path();

/// The full path of the dynamic library or extension module that contains
/// `anchor` (any code address inside it), as UTF-8, or "" on failure. For the
/// executable itself this returns the executable path, so one call describes
/// both cases: pass the address of a function in a translation unit that is
/// linked into the image whose location matters.
std::string module_path(const void* anchor);

}  // namespace alphashape::platform
