#include "apps/gui/installer.hpp"

#include <QCoreApplication>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMessageBox>
#include <QProcess>
#include <QUrl>

#include "base/platform/env.hpp"

namespace alphashape::gui {
namespace {

InstallResult fallback(const update::Release& release, const QString& reason) {
    InstallResult result;
    result.error = reason.toStdString();
    result.message = reason;
    if (QDesktopServices::openUrl(QUrl(QString::fromStdString(release.html_url)))) {
        result.message += QCoreApplication::translate(
            "Installer", " The release page has been opened in your browser.");
    }
    return result;
}

InstallResult install_macos(const update::Release& release, const QString& downloaded) {
    if (!QDesktopServices::openUrl(QUrl::fromLocalFile(downloaded))) {
        return fallback(release, QCoreApplication::translate(
                                      "Installer", "Could not open the disk image."));
    }
    InstallResult result;
    result.success = true;
    result.message = QCoreApplication::translate(
        "Installer",
        "The disk image is open. Drag alphashape into Applications, then restart the app. "
        "The build is unsigned, so macOS may ask you to confirm the first launch.");
    return result;
}

InstallResult install_windows(const update::Release& release, const QString& downloaded,
                              QWidget* parent) {
    const auto answer = QMessageBox::question(
        parent, QCoreApplication::translate("Installer", "Install update"),
        QCoreApplication::translate(
            "Installer",
            "alphashape will quit and the installer will start. Continue?"),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (answer != QMessageBox::Yes) {
        return fallback(release, QCoreApplication::translate(
                                      "Installer", "The installer was not launched."));
    }
    const bool started = QProcess::startDetached(downloaded, {});
    if (!started) {
        return fallback(release, QCoreApplication::translate(
                                      "Installer", "Could not start the installer."));
    }
    InstallResult result;
    result.success = true;
    result.message = QCoreApplication::translate(
        "Installer", "The installer has started. alphashape will now quit.");
    QCoreApplication::quit();
    return result;
}

InstallResult install_appimage(const update::Release& release, const QString& downloaded) {
    const QString running = QString::fromStdString(platform::get_env("APPIMAGE"));
    if (running.isEmpty()) {
        return fallback(release, QCoreApplication::translate(
                                      "Installer", "The running AppImage was not found."));
    }
    const QFileInfo info(running);
    const QString staged = info.absoluteFilePath() + QStringLiteral(".new");
    const QString backup = info.absoluteFilePath() + QStringLiteral(".bak");
    QFile::remove(staged);
    if (!QFile::copy(downloaded, staged)) {
        return fallback(release, QCoreApplication::translate(
                                      "Installer", "Could not stage the new AppImage."));
    }
    QFile::setPermissions(staged, QFileDevice::ReadOwner | QFileDevice::WriteOwner |
                                      QFileDevice::ExeOwner | QFileDevice::ReadGroup |
                                      QFileDevice::ExeGroup | QFileDevice::ReadOther |
                                      QFileDevice::ExeOther);
    // Keep the old file until the replacement is in place.
    QFile::remove(backup);
    const bool renamed_old = QFile::rename(info.absoluteFilePath(), backup);
    const bool renamed_new = renamed_old && QFile::rename(staged, info.absoluteFilePath());
    if (!renamed_new) {
        if (renamed_old) {
            QFile::rename(backup, info.absoluteFilePath());
        }
        QFile::remove(staged);
        return fallback(release, QCoreApplication::translate(
                                      "Installer", "Could not replace the running AppImage."));
    }

    const auto answer = QMessageBox::question(
        nullptr, QCoreApplication::translate("Installer", "Update installed"),
        QCoreApplication::translate("Installer",
                                    "The new AppImage is in place. Restart alphashape now?"),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
    InstallResult result;
    result.success = true;
    result.message = QCoreApplication::translate(
        "Installer",
        "The new AppImage is in place. The previous one is kept as a .bak file.");
    if (answer == QMessageBox::Yes && QProcess::startDetached(info.absoluteFilePath(), {})) {
        QCoreApplication::quit();
    }
    return result;
}

InstallResult install_linux_package(const update::Release& release,
                                    const QString& downloaded) {
    if (!QDesktopServices::openUrl(QUrl::fromLocalFile(downloaded))) {
        return fallback(release, QCoreApplication::translate(
                                      "Installer", "Could not open the package."));
    }
    InstallResult result;
    result.success = true;
    result.message = QCoreApplication::translate(
        "Installer",
        "The package is open in your software installer. Finish the installation there, "
        "then restart alphashape.");
    return result;
}

}  // namespace

InstallResult install_update(const update::Release& release, const QString& downloaded,
                             const update::Target& target, QWidget* parent) {
    if (!QFileInfo::exists(downloaded)) {
        return fallback(release, QCoreApplication::translate(
                                      "Installer", "The downloaded update file is missing."));
    }
    switch (target.platform) {
        case update::Platform::MacOS:
            return install_macos(release, downloaded);
        case update::Platform::Windows:
            return install_windows(release, downloaded, parent);
        case update::Platform::Linux:
            return target.appimage ? install_appimage(release, downloaded)
                                   : install_linux_package(release, downloaded);
        case update::Platform::Other:
            return fallback(release, QCoreApplication::translate(
                                          "Installer", "No installer is available."));
    }
    return fallback(release, QCoreApplication::translate("Installer",
                                                          "No installer is available."));
}

}  // namespace alphashape::gui
