#pragma once

// Environment variables and UTF-8 / wide-string conversion.
//
// Every path inside the engine is a UTF-8 std::string (that is what
// std::filesystem::u8path / path::u8string consume). Windows APIs are wide, so
// the conversion lives here and nowhere else.

#include <filesystem>
#include <string>

namespace alphashape::platform {

/// UTF-8 view of a wide Windows string. On POSIX the input type does not exist,
/// so the function is only declared under _WIN32.
#if defined(_WIN32)
std::string to_utf8(const std::wstring& wide);
std::wstring from_utf8(const std::string& utf8);
#endif

/// A filesystem path from a UTF-8 string. std::filesystem::path's string
/// constructor decodes its argument in the *native narrow* encoding, which on
/// Windows is the ANSI code page: an install under "C:\Users\Jörg\..." would
/// arrive as mojibake and no file would open. Going through u8string decodes
/// UTF-8 on every platform, which is the encoding the rest of the engine uses.
std::filesystem::path path_from_utf8(const std::string& utf8);

/// The inverse: the path as a UTF-8 std::string. path::string() would narrow
/// with the ANSI code page on Windows, so anything that prints a path or hands
/// one to a C library (PROJ, an environment variable) must use this.
std::string path_to_utf8(const std::filesystem::path& path);

/// Returns "" when the variable is unset. A set-but-empty variable also reads
/// as "": every caller in the engine treats empty as absent (ALPHASHAPE_GPU,
/// VK_DRIVER_FILES, the SDK overrides).
std::string get_env(const char* name);

/// Sets or unsets a variable. An empty value removes it, matching get_env's
/// reading. Returns false when the platform rejected the change.
bool set_env(const char* name, const std::string& value);

}  // namespace alphashape::platform
