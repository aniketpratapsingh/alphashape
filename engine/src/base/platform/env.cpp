#include "base/platform/env.hpp"

#include <cstdlib>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace alphashape::platform {

#if defined(_WIN32)
namespace {

/// MultiByteToWideChar / WideCharToMultiByte wrapper. Both need a two-pass
/// call: the first with a zero destination computes the length in characters
/// *excluding* the terminator, the second fills the buffer (and with an
/// explicit -1 length also writes the terminator we then drop).
std::wstring widen(const std::string& utf8) {
    if (utf8.empty()) {
        return std::wstring();
    }
    const int needed = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(),
                                             static_cast<int>(utf8.size()), nullptr, 0);
    if (needed <= 0) {
        return std::wstring();
    }
    std::wstring wide(static_cast<std::size_t>(needed), L'\0');
    const int written = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(),
                                              static_cast<int>(utf8.size()), wide.data(), needed);
    if (written != needed) {
        return std::wstring();
    }
    return wide;
}

/// Converts `length` wide characters (no terminator: the two calls pass an
/// explicit count, so the size WideCharToMultiByte reports is exactly what the
/// buffer must hold). Passing -1 instead would require room for the NUL and
/// makes the second call fail with ERROR_INSUFFICIENT_BUFFER.
std::string narrow(const wchar_t* wide, std::size_t length) {
    if (wide == nullptr || length == 0) {
        return std::string();
    }
    const int wide_length = static_cast<int>(length);
    const int needed =
        ::WideCharToMultiByte(CP_UTF8, 0, wide, wide_length, nullptr, 0, nullptr, nullptr);
    if (needed <= 0) {
        return std::string();
    }
    std::string utf8(static_cast<std::size_t>(needed), '\0');
    const int written = ::WideCharToMultiByte(CP_UTF8, 0, wide, wide_length, utf8.data(), needed,
                                              nullptr, nullptr);
    if (written != needed) {
        return std::string();
    }
    return utf8;
}

}  // namespace

std::string to_utf8(const std::wstring& wide) { return narrow(wide.c_str(), wide.size()); }

std::wstring from_utf8(const std::string& utf8) { return widen(utf8); }
#endif

std::string get_env(const char* name) {
#if defined(_WIN32)
    const std::wstring wide_name = from_utf8(name);
    if (wide_name.empty()) {
        return std::string();
    }
    // GetEnvironmentVariableW returns 0 both for "unset" and (with
    // ERROR_SUCCESS) for "set to empty"; the two read the same here by design.
    DWORD needed = ::GetEnvironmentVariableW(wide_name.c_str(), nullptr, 0);
    if (needed == 0) {
        return std::string();
    }
    std::wstring buffer(static_cast<std::size_t>(needed), L'\0');
    const DWORD written = ::GetEnvironmentVariableW(wide_name.c_str(), buffer.data(), needed);
    if (written == 0 || written >= needed) {
        return std::string();
    }
    buffer.resize(written);
    return to_utf8(buffer);
#else
    const char* value = std::getenv(name);
    return value != nullptr ? std::string(value) : std::string();
#endif
}

bool set_env(const char* name, const std::string& value) {
#if defined(_WIN32)
    const std::wstring wide_name = from_utf8(name);
    if (wide_name.empty()) {
        return false;
    }
    const std::wstring wide_value = from_utf8(value);
    // A null value deletes the variable; an empty string would set it to "".
    const wchar_t* payload = value.empty() ? nullptr : wide_value.c_str();
    return ::SetEnvironmentVariableW(wide_name.c_str(), payload) != 0;
#else
    // Empty means unset, as documented in the header.
    if (value.empty()) {
        return ::unsetenv(name) == 0;
    }
    return ::setenv(name, value.c_str(), 1) == 0;
#endif
}

}  // namespace alphashape::platform
