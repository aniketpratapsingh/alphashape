#include "apps/gui/update_checker.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <utility>

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QStandardPaths>
#include <QString>
#include <QStringList>

#include "apps/gui/update_source.hpp"
#include "base/platform/env.hpp"
#include "common/update/checksum.hpp"
#include "common/update/release.hpp"
#include "common/update/version.hpp"

#ifndef ALPHASHAPE_GUI_VERSION
#define ALPHASHAPE_GUI_VERSION "0.0.0"
#endif

namespace alphashape::gui {
namespace {

constexpr const char* kAutoKey = "updates/auto_check";
constexpr const char* kChannelKey = "updates/channel";
constexpr const char* kLastCheckKey = "updates/last_check";
constexpr const char* kSkippedKey = "updates/skipped_version";
constexpr const char* kOptOutEnv = "ALPHASHAPE_NO_UPDATE_CHECK";

std::int64_t now_unix() {
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

}  // namespace

std::string running_version() { return ALPHASHAPE_GUI_VERSION; }

UpdateChecker::UpdateChecker(QSettings& settings, std::unique_ptr<UpdateSource> source)
    : settings_(settings), source_(std::move(source)) {}

UpdateChecker::~UpdateChecker() = default;

void UpdateChecker::set_source(std::unique_ptr<UpdateSource> source) {
    source_ = std::move(source);
}

bool UpdateChecker::automatic_checks_enabled() const {
    return settings_.value(QString::fromLatin1(kAutoKey), true).toBool();
}

void UpdateChecker::set_automatic_checks_enabled(bool enabled) {
    settings_.setValue(QString::fromLatin1(kAutoKey), enabled);
    settings_.sync();
}

update::Channel UpdateChecker::channel() const {
    const QString stored = settings_.value(QString::fromLatin1(kChannelKey)).toString();
    if (!stored.isEmpty()) {
        return stored == QStringLiteral("prerelease") ? update::Channel::Prerelease
                                                      : update::Channel::Stable;
    }
    // A pre-release build defaults to the pre-release channel: otherwise it
    // would sit on a channel whose releases are all older than itself.
    if (const auto current = update::Version::parse(running_version());
        current && current->is_prerelease()) {
        return update::Channel::Prerelease;
    }
    return update::Channel::Stable;
}

void UpdateChecker::set_channel(update::Channel channel) {
    settings_.setValue(QString::fromLatin1(kChannelKey),
                       channel == update::Channel::Prerelease ? QStringLiteral("prerelease")
                                                              : QStringLiteral("stable"));
    settings_.sync();
}

bool UpdateChecker::environment_opt_out() {
    return !platform::get_env(kOptOutEnv).empty();
}

bool UpdateChecker::checks_allowed(bool manual) const {
    if (environment_opt_out()) {
        return false;
    }
    return manual || automatic_checks_enabled();
}

std::int64_t UpdateChecker::last_check_unix() const {
    return settings_.value(QString::fromLatin1(kLastCheckKey), Q_INT64_C(0)).toLongLong();
}

void UpdateChecker::record_check(std::int64_t now) {
    settings_.setValue(QString::fromLatin1(kLastCheckKey), static_cast<qint64>(now));
    settings_.sync();
}

bool UpdateChecker::automatic_check_due_now() const {
    return update::automatic_check_due(now_unix(), last_check_unix());
}

std::optional<update::Version> UpdateChecker::skipped_version() const {
    const QString text = settings_.value(QString::fromLatin1(kSkippedKey)).toString();
    if (text.isEmpty()) {
        return std::nullopt;
    }
    return update::Version::parse(text.toStdString());
}

void UpdateChecker::skip_version(const update::Version& version) {
    settings_.setValue(QString::fromLatin1(kSkippedKey),
                       QString::fromStdString(version.to_string()));
    settings_.sync();
}

bool UpdateChecker::check(bool manual, CheckDone done) {
    if (busy_) {
        if (manual) {
            done(CheckResult{CheckResult::Status::Error, {},
                             "an update check is already running"});
        }
        return false;
    }
    if (!checks_allowed(manual)) {
        if (manual) {
            done(CheckResult{CheckResult::Status::Error, {},
                             "update checks are disabled (ALPHASHAPE_NO_UPDATE_CHECK)"});
        }
        return false;
    }
    if (!manual && !automatic_check_due_now()) {
        return false;
    }

    busy_ = true;
    source_->fetch_text(update::kReleasesApiUrl,
                        [this, done](bool ok, const std::string& body,
                                     const std::string& error) {
                            CheckResult result;
                            if (!ok) {
                                result.status = CheckResult::Status::Error;
                                result.error = "could not reach GitHub: " + error;
                                busy_ = false;
                                done(result);
                                return;
                            }
                            try {
                                const auto releases = update::parse_releases(body);
                                result = evaluate(releases);
                                record_check(now_unix());
                            } catch (const std::exception& e) {
                                result.status = CheckResult::Status::Error;
                                result.error =
                                    "could not read the GitHub releases response: " +
                                    std::string(e.what());
                            }
                            busy_ = false;
                            done(result);
                        });
    return true;
}

CheckResult UpdateChecker::evaluate(const std::vector<update::Release>& releases) {
    CheckResult result;
    const auto current = update::Version::parse(running_version());
    if (!current) {
        result.status = CheckResult::Status::Error;
        result.error = "this build reports a version that is not semantic: " + running_version();
        return result;
    }
    const update::Selection selection =
        update::select_release(releases, *current, channel(), skipped_version());
    if (!selection.release) {
        result.status = CheckResult::Status::UpToDate;
        result.skipped = selection.skipped;
        return result;
    }
    const update::Target target = update::host_target();
    if (update::choose_asset(*selection.release, target) == nullptr) {
        result.status = CheckResult::Status::Error;
        result.release = *selection.release;
        result.error = update::no_package_reason(target);
        return result;
    }
    result.status = CheckResult::Status::Available;
    result.release = *selection.release;
    return result;
}

QString UpdateChecker::download_directory() {
    QString base = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
    if (base.isEmpty()) {
        base = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    }
    return QDir(base).filePath(QStringLiteral("updates"));
}

void UpdateChecker::cancel_download() {
    if (source_) {
        source_->cancel();
    }
}

void UpdateChecker::download_and_verify(const update::Release& release,
                                        const update::Asset& asset,
                                        const QString& cache_dir, ProgressCallback progress,
                                        std::function<void(const DownloadResult&)> done) {
    if (!is_allowed_update_download_url(asset.browser_download_url)) {
        done({false, {}, "refusing to download an update from outside GitHub"});
        return;
    }
    const update::Asset* sums = release.find_asset(update::kChecksumsAssetName);
    if (sums == nullptr) {
        done({false, {}, "this release has no SHA256SUMS; refusing to install an "
                         "unverifiable download"});
        return;
    }
    if (!is_allowed_update_metadata_url(sums->browser_download_url)) {
        done({false, {}, "the SHA256SUMS URL is not a GitHub URL"});
        return;
    }
    // The asset name comes from the release JSON and becomes a file name under
    // the cache directory. GitHub only allows plain names, but a compromised or
    // spoofed response must not be able to write outside the cache.
    const QString name = QString::fromStdString(asset.name);
    if (name.isEmpty() || name.contains(QLatin1Char('/')) || name.contains(QLatin1Char('\\')) ||
        name.contains(QStringLiteral("..")) || QFileInfo(name).fileName() != name) {
        done({false, {}, "the update asset has an unsafe file name: " + asset.name});
        return;
    }
    if (!QDir().mkpath(cache_dir)) {
        done({false, {}, "could not create the update cache directory"});
        return;
    }
    const QString path = QDir(cache_dir).filePath(name);
    QFile::remove(path);

    const std::string asset_url = asset.browser_download_url;
    const std::string sums_url = sums->browser_download_url;
    const std::string asset_name = asset.name;

    source_->download_file(
        asset_url, path, asset.size, std::move(progress),
        [this, path, asset_name, sums_url, done](bool ok, const std::string& error) {
            if (!ok) {
                done({false, {}, "download failed: " + error});
                return;
            }
            source_->fetch_text(
                sums_url,
                [path, asset_name, done](bool sums_ok, const std::string& sums_body,
                                         const std::string& sums_error) {
                    if (!sums_ok) {
                        QFile::remove(path);
                        done({false, {},
                              "could not download SHA256SUMS: " + sums_error});
                        return;
                    }
                    const auto entries = update::parse_sha256sums(sums_body);
                    const update::ChecksumEntry* entry =
                        update::find_checksum(entries, asset_name);
                    if (entry == nullptr) {
                        QFile::remove(path);
                        done({false, {},
                              "SHA256SUMS does not list " + asset_name +
                                  "; refusing to install an unverifiable download"});
                        return;
                    }
                    QFile file(path);
                    if (!file.open(QIODevice::ReadOnly)) {
                        QFile::remove(path);
                        done({false, {},
                              "could not read the downloaded file for checksum verification"});
                        return;
                    }
                    QCryptographicHash hash(QCryptographicHash::Sha256);
                    if (!hash.addData(&file)) {
                        file.close();
                        QFile::remove(path);
                        done({false, {},
                              "could not read the downloaded file for checksum verification"});
                        return;
                    }
                    file.close();
                    const std::string actual =
                        QString::fromLatin1(hash.result().toHex()).toStdString();
                    if (!update::checksum_matches(entry->sha256, actual)) {
                        QFile::remove(path);
                        done({false, {},
                              "checksum mismatch for " + asset_name + ": expected " +
                                  entry->sha256 + ", got " + actual});
                        return;
                    }
                    done({true, path, {}});
                });
        });
}

}  // namespace alphashape::gui
