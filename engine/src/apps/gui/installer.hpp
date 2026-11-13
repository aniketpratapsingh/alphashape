#pragma once

// Installing a verified update, per platform.
//
// The app is unsigned, so nothing here replaces a running bundle behind the
// user's back. The Linux AppImage is replaced in place only after the user
// agrees (its running file is the update target, and the old one is kept as
// .bak). Every path falls back to opening the release page when the platform
// step cannot run. Signature verification is future work; the SHA-256 check
// that has already happened is integrity only.

#include <string>

#include <QString>

#include "common/update/release.hpp"

class QWidget;

namespace alphashape::gui {

struct InstallResult {
    /// A short user-facing sentence describing what happened.
    QString message;
    /// Empty on success; the release page was opened as the fallback otherwise.
    std::string error;
    bool success = false;
};

/// `downloaded` is the verified file. `release` supplies the fallback URL.
/// Asks for consent before launching an installer or replacing a running
/// AppImage (keeping the old file as .bak in that case); when the user declines,
/// nothing is replaced and the result is not a success.
InstallResult install_update(const update::Release& release, const QString& downloaded,
                             const update::Target& target, QWidget* parent);

}  // namespace alphashape::gui
