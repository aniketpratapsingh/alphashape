#pragma once

#include <optional>
#include <string>

namespace alphashape::gui {

/// The two application themes. Dark is the default.
enum class ThemeMode { Dark, Light };

/// "dark" / "light".
inline const char* theme_name(ThemeMode mode) {
    return mode == ThemeMode::Dark ? "dark" : "light";
}

/// Inverse of theme_name(); nullopt for anything else (case-sensitive).
inline std::optional<ThemeMode> theme_from_name(const std::string& name) {
    if (name == "dark") {
        return ThemeMode::Dark;
    }
    if (name == "light") {
        return ThemeMode::Light;
    }
    return std::nullopt;
}

}  // namespace alphashape::gui
