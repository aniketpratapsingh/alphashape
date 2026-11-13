#pragma once

// The updater's only network seam. UpdateChecker talks to this interface, so
// every test runs against a fake and never opens a socket. The production
// implementation is NetworkUpdateSource (QNetworkAccessManager).
//
// Security: every URL is checked before a request starts, and every redirect hop
// is checked again (UserVerifiedRedirectPolicy asks this code before following
// each one). HTTPS only, and downloads only from github.com or the two asset CDN
// hosts GitHub redirects release downloads to
// (objects.githubusercontent.com and release-assets.githubusercontent.com).

#include <cstdint>
#include <functional>
#include <string>

#include <QObject>
#include <QPointer>
#include <QString>

class QNetworkAccessManager;
class QNetworkReply;

namespace alphashape::gui {

/// True when `url` may be fetched for release metadata or the checksum file.
bool is_allowed_update_metadata_url(const std::string& url);

/// True when `url` may be used to download an installer. Stricter than the
/// metadata check: the GitHub API itself is not a download host.
bool is_allowed_update_download_url(const std::string& url);

class UpdateSource {
public:
    using TextCallback =
        std::function<void(bool ok, const std::string& body, const std::string& error)>;
    using ProgressCallback = std::function<void(qint64 received, qint64 total)>;
    using FileCallback = std::function<void(bool ok, const std::string& error)>;

    virtual ~UpdateSource() = default;

    /// Fetch a small text resource (the releases JSON or SHA256SUMS).
    virtual void fetch_text(const std::string& url, TextCallback done) = 0;

    /// Download `url` to the file `dest`. The callback runs exactly once;
    /// `cancel()` may have caused it.
    virtual void download_file(const std::string& url, const QString& dest,
                               ProgressCallback progress, FileCallback done) = 0;

    /// Abort the request in flight, if any. The pending callback still runs.
    virtual void cancel() = 0;
};

/// QNetworkAccessManager implementation used by the real application.
class NetworkUpdateSource : public QObject, public UpdateSource {
    Q_OBJECT

public:
    explicit NetworkUpdateSource(QObject* parent = nullptr);
    ~NetworkUpdateSource() override;

    void fetch_text(const std::string& url, TextCallback done) override;
    void download_file(const std::string& url, const QString& dest, ProgressCallback progress,
                       FileCallback done) override;
    void cancel() override;

private:
    void start_reply(const std::string& url, bool download, const QString& dest);

    QNetworkAccessManager* network_ = nullptr;
    QPointer<QNetworkReply> reply_;
    QString download_path_;
    bool cancelled_ = false;
};

}  // namespace alphashape::gui
