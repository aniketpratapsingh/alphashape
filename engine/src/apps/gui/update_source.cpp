#include "apps/gui/update_source.hpp"

#include <algorithm>
#include <utility>

#include <QByteArray>
#include <QFile>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QUrl>

namespace alphashape::gui {
namespace {

/// The largest release metadata response (the releases JSON or SHA256SUMS) we
/// will buffer. Both are tens of kilobytes in practice.
constexpr qint64 kMaxTextBytes = 8 * 1024 * 1024;
/// Slack over the size the release API declares for an asset, so a rounding or
/// chunking difference is not mistaken for a malicious response.
constexpr qint64 kDownloadMarginBytes = 1024 * 1024;
/// GitHub redirects a release download a couple of times at most.
constexpr int kMaxRedirects = 8;

bool allowed_host(const QString& host, bool allow_api) {
    if (host == QStringLiteral("github.com") ||
        host == QStringLiteral("objects.githubusercontent.com") ||
        host == QStringLiteral("release-assets.githubusercontent.com")) {
        return true;
    }
    return allow_api && host == QStringLiteral("api.github.com");
}

bool allowed_url(const std::string& url, bool allow_api) {
    const QUrl parsed(QString::fromStdString(url));
    if (!parsed.isValid() || parsed.scheme() != QStringLiteral("https")) {
        return false;
    }
    // github.com uses the default port; an explicit other port is not GitHub.
    if (parsed.port() != -1 && parsed.port() != 443) {
        return false;
    }
    if (!parsed.userInfo().isEmpty()) {
        return false;
    }
    return allowed_host(parsed.host(), allow_api);
}

QString network_error_text(QNetworkReply* reply) {
    const int status =
        reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    QString text = reply->errorString();
    if (status != 0) {
        text = QStringLiteral("HTTP %1 (%2)").arg(status).arg(reply->errorString());
    }
    return text;
}

}  // namespace

struct NetworkUpdateSource::Transfer {
    bool download = false;
    /// True for a metadata fetch: api.github.com is allowed, and the response
    /// is buffered as text instead of written to a file.
    bool allow_api = false;
    /// The body exceeds `limit`, or a redirect hop was refused; the completion
    /// callback reports it as an error.
    bool too_large = false;
    std::string redirect_error;
    qint64 limit = -1;
    qint64 received = 0;
    int redirects = 0;
    QString dest;
    QByteArray body;
    QFile* file = nullptr;
    UpdateSource::ProgressCallback progress;
    UpdateSource::FileCallback file_done;
    UpdateSource::TextCallback text_done;

    ~Transfer() {
        if (file != nullptr) {
            file->close();
            delete file;
        }
    }
};

bool is_allowed_update_metadata_url(const std::string& url) { return allowed_url(url, true); }

bool is_allowed_update_download_url(const std::string& url) {
    return allowed_url(url, false);
}

NetworkUpdateSource::NetworkUpdateSource(QObject* parent)
    : QObject(parent), network_(new QNetworkAccessManager(this)) {}

NetworkUpdateSource::~NetworkUpdateSource() {
    // No callback runs into a half-destroyed object: drop the connections
    // before aborting, and let the shared Transfer close the file.
    if (reply_ != nullptr) {
        QObject::disconnect(reply_, nullptr, this, nullptr);
        reply_->abort();
        reply_->deleteLater();
        reply_ = nullptr;
    }
    if (transfer_ && transfer_->download && transfer_->file != nullptr) {
        // A quit mid-download must not leave a partial installer behind.
        transfer_->file->close();
        QFile::remove(transfer_->dest);
    }
    transfer_.reset();
}

void NetworkUpdateSource::cancel() {
    cancelled_ = true;
    if (reply_ != nullptr) {
        reply_->abort();
    }
}

void NetworkUpdateSource::fetch_text(const std::string& url, TextCallback done) {
    cancelled_ = false;
    if (!is_allowed_update_metadata_url(url)) {
        done(false, std::string(),
             "refusing to fetch an update URL outside GitHub: " + url);
        return;
    }
    if (reply_ != nullptr || transfer_) {
        done(false, std::string(), "another update request is already running");
        return;
    }
    auto transfer = std::make_shared<Transfer>();
    transfer->allow_api = true;
    transfer->limit = kMaxTextBytes;
    transfer->text_done = std::move(done);
    transfer_ = transfer;
    start_get(url, transfer);
}

void NetworkUpdateSource::download_file(const std::string& url, const QString& dest,
                                        std::int64_t expected_size, ProgressCallback progress,
                                        FileCallback done) {
    cancelled_ = false;
    if (!is_allowed_update_download_url(url)) {
        done(false, "refusing to download an update from outside GitHub: " + url);
        return;
    }
    if (reply_ != nullptr || transfer_) {
        done(false, "another update request is already running");
        return;
    }
    auto transfer = std::make_shared<Transfer>();
    transfer->download = true;
    transfer->dest = dest;
    transfer->progress = std::move(progress);
    transfer->file_done = std::move(done);
    transfer->limit =
        expected_size > 0 ? expected_size + kDownloadMarginBytes : static_cast<qint64>(-1);
    transfer->file = new QFile(dest);
    if (!transfer->file->open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        const std::string error =
            "could not open the download file: " + transfer->file->errorString().toStdString();
        UpdateSource::FileCallback failed = std::move(transfer->file_done);
        transfer.reset();
        failed(false, error);
        return;
    }
    transfer_ = transfer;
    start_get(url, transfer);
}

void NetworkUpdateSource::start_get(const std::string& url,
                                    const std::shared_ptr<Transfer>& transfer) {
    QNetworkRequest request(QUrl(QString::fromStdString(url)));
    // UserVerifiedRedirectPolicy asks this code about every hop before it
    // follows it; the automatic policies would follow a redirect to whatever
    // host the server names.
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                         QNetworkRequest::UserVerifiedRedirectPolicy);
    request.setHeader(QNetworkRequest::UserAgentHeader,
                      QStringLiteral("alphashape-updater"));
    if (transfer->allow_api) {
        request.setRawHeader("Accept", "application/vnd.github+json");
    }
    QNetworkReply* reply = network_->get(request);
    reply_ = reply;
    const int generation = ++generation_;

    connect(reply, &QNetworkReply::readyRead, this, [this, reply, transfer, generation]() {
        if (generation != generation_) {
            return;
        }
        const QByteArray chunk = reply->readAll();
        if (transfer->download) {
            if (transfer->limit >= 0 &&
                transfer->received + chunk.size() > transfer->limit) {
                transfer->too_large = true;
                reply->abort();
                return;
            }
            transfer->file->write(chunk);
            transfer->received += chunk.size();
            return;
        }
        transfer->body.append(chunk);
        if (transfer->body.size() > transfer->limit) {
            transfer->too_large = true;
            reply->abort();
        }
    });
    connect(reply, &QNetworkReply::downloadProgress, this,
            [transfer](qint64 received, qint64 total) {
                if (transfer->progress) {
                    transfer->progress(received, total);
                }
            });
    connect(reply, &QNetworkReply::redirected, this,
            [this, reply, transfer, generation](const QUrl& target) {
                if (generation != generation_) {
                    return;
                }
                const std::string next = target.toString().toStdString();
                if (!allowed_url(next, transfer->allow_api)) {
                    transfer->redirect_error =
                        "refusing to follow a redirect outside GitHub: " + next;
                    reply->abort();
                    return;
                }
                if (++transfer->redirects > kMaxRedirects) {
                    transfer->redirect_error = "too many redirects while fetching an update";
                    reply->abort();
                    return;
                }
                // Let Qt follow this hop. The next hop arrives as another
                // redirected() signal and goes through this same check.
                reply->redirectAllowed();
            });
    connect(reply, &QNetworkReply::finished, this, [this, reply, transfer, generation]() {
        if (generation != generation_) {
            reply->deleteLater();
            return;
        }
        reply_ = nullptr;
        const QByteArray rest = reply->readAll();
        std::string error;
        if (!transfer->redirect_error.empty()) {
            error = transfer->redirect_error;
        } else if (transfer->too_large) {
            error = transfer->download
                        ? "the download exceeded the size the release declares"
                        : "the update response is larger than the 8 MiB limit";
        } else if (cancelled_) {
            error = "the update request was cancelled";
        } else if (reply->error() != QNetworkReply::NoError) {
            error = network_error_text(reply).toStdString();
        }
        const bool ok = error.empty();
        reply->deleteLater();

        if (transfer->download) {
            if (ok) {
                if (transfer->limit >= 0 &&
                    transfer->received + rest.size() > transfer->limit) {
                    error = "the download exceeded the size the release declares";
                } else {
                    transfer->file->write(rest);
                }
            }
            transfer->file->close();
            if (!error.empty()) {
                // Never leave a partial installer behind.
                transfer->file->remove();
            }
            UpdateSource::FileCallback done = std::move(transfer->file_done);
            transfer_.reset();
            done(error.empty(), error);
            return;
        }

        transfer->body.append(rest);
        if (transfer->body.size() > transfer->limit) {
            error = "the update response is larger than the 8 MiB limit";
        }
        std::string text(transfer->body.constData(),
                         static_cast<std::size_t>(transfer->body.size()));
        UpdateSource::TextCallback done = std::move(transfer->text_done);
        transfer_.reset();
        done(error.empty(), std::move(text), error);
    });
}

}  // namespace alphashape::gui
