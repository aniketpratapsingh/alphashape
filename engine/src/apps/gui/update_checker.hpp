#pragma once

// UpdateChecker: the GUI policy around the Qt-free update core.
//
// It owns the settings keys (all under "updates/"), the throttle, the 24-hour
// timestamp, the skipped version, the channel, and the download/verify step.
// The network sits behind UpdateSource so tests inject a fake.
//
// Privacy: the only request is a GET of the public GitHub Releases API (and,
// during a user-approved download, the release asset and its SHA256SUMS). No
// identifiers, no usage data, no telemetry. ALPHASHAPE_NO_UPDATE_CHECK=1 or
// --no-update-check disables every request.

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <QSettings>
#include <QString>

#include "common/update/release.hpp"

namespace alphashape::gui {

class UpdateSource;

/// The running build's version, from ALPHASHAPE_GUI_VERSION.
std::string running_version();

struct CheckResult {
    enum class Status { UpToDate, Available, Error };
    Status status = Status::UpToDate;
    update::Release release;
    std::string error;
    bool skipped = false;  ///< up to date because the user skipped that version
};

struct DownloadResult {
    bool ok = false;
    QString path;
    std::string error;
};

class UpdateChecker {
public:
    using CheckDone = std::function<void(const CheckResult&)>;
    using ProgressCallback = std::function<void(qint64 received, qint64 total)>;

    UpdateChecker(QSettings& settings, std::unique_ptr<UpdateSource> source);
    ~UpdateChecker();

    UpdateChecker(const UpdateChecker&) = delete;
    UpdateChecker& operator=(const UpdateChecker&) = delete;

    UpdateSource& source() { return *source_; }
    /// Replace the network seam. The production code installs the QNetwork one
    /// at construction; tests swap in a fake before any check starts.
    void set_source(std::unique_ptr<UpdateSource> source);

    /// Run a check. `manual` bypasses the 24-hour throttle and always reports a
    /// result. An automatic check stays silent when it is disabled, opted out
    /// or throttled (the callback is not invoked). A manual check reports an
    /// error with the reason. Returns true when a request was started.
    bool check(bool manual, CheckDone done);

    bool automatic_checks_enabled() const;
    void set_automatic_checks_enabled(bool enabled);

    update::Channel channel() const;
    void set_channel(update::Channel channel);

    /// ALPHASHAPE_NO_UPDATE_CHECK is set to a non-empty value.
    static bool environment_opt_out();
    /// True when checks may run at all: not opted out and enabled for the mode.
    bool checks_allowed(bool manual) const;

    std::int64_t last_check_unix() const;
    void record_check(std::int64_t now);
    bool automatic_check_due_now() const;

    std::optional<update::Version> skipped_version() const;
    void skip_version(const update::Version& version);

    /// Download `asset` into `cache_dir` and verify it against the release's
    /// SHA256SUMS. A missing checksum, a mismatch or a read failure is an
    /// error and the file is removed. The callback runs exactly once.
    void download_and_verify(const update::Release& release, const update::Asset& asset,
                             const QString& cache_dir, ProgressCallback progress,
                             std::function<void(const DownloadResult&)> done);
    void cancel_download();

    /// The per-user cache directory the verified download is written to.
    static QString download_directory();

private:
    CheckResult evaluate(const std::vector<update::Release>& releases);

    QSettings& settings_;
    std::unique_ptr<UpdateSource> source_;
    bool busy_ = false;
};

}  // namespace alphashape::gui
