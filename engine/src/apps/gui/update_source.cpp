#include "apps/gui/update_source.hpp"

#include <algorithm>
#include <utility>

#include <QFile>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QUrl>

namespace alphashape::gui {
namespace {

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

bool is_allowed_update_metadata_url(const std::string& url) { return allowed_url(url, true); }

bool is_allowed_update_download_url(const std::string& url) {
    return allowed_url(url, false);
}

NetworkUpdateSource::NetworkUpdateSource(QObject* parent)
    : QObject(parent), network_(new QNetworkAccessManager(this)) {}

NetworkUpdateSource::~NetworkUpdateSource() = default;

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
    if (reply_ != nullptr) {
        done(false, std::string(), "another update request is already running");
        return;
    }
    QNetworkRequest request(QUrl(QString::fromStdString(url)));
    // UserVerifiedRedirectPolicy asks this code about every hop before it
    // follows it; the automatic policies would follow a redirect to whatever
    // host the server names.
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                         QNetworkRequest::UserVerifiedRedirectPolicy);
    request.setHeader(QNetworkRequest::UserAgentHeader,
                      QStringLiteral("alphashape-updater"));
    request.setRawHeader("Accept", "application/vnd.github+json");
    QNetworkReply* reply = network_->get(request);
    reply_ = reply;

    auto* body = new QByteArray();
    connect(reply, &QNetworkReply::readyRead, reply,
            [reply, body]() { body->append(reply->readAll()); });
    connect(reply, &QNetworkReply::redirected, reply,
            [reply](const QUrl& target) {
                if (!allowed_url(target.toString().toStdString(), true)) {
                    reply->abort();
                    return;
                }
                // The next hop arrives as another redirected() signal and goes
                // through this same check.
                reply->redirectAllowed();
            });
    connect(reply, &QNetworkReply::finished, reply, [this, reply, body, done]() {
        reply_ = nullptr;
        const QByteArray rest = reply->readAll();
        body->append(rest);
        const bool ok = !cancelled_ && reply->error() == QNetworkReply::NoError;
        const std::string error =
            ok ? std::string() : network_error_text(reply).toStdString();
        std::string text(body->constData(), static_cast<std::size_t>(body->size()));
        delete body;
        reply->deleteLater();
        done(ok, std::move(text), error);
    });
}

void NetworkUpdateSource::download_file(const std::string& url, const QString& dest,
                                         ProgressCallback progress, FileCallback done) {
    cancelled_ = false;
    if (!is_allowed_update_download_url(url)) {
        done(false, "refusing to download an update from outside GitHub: " + url);
        return;
    }
    if (reply_ != nullptr) {
        done(false, "another update request is already running");
        return;
    }
    auto* file = new QFile(dest);
    if (!file->open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        const std::string error =
            "could not open the download file: " + file->errorString().toStdString();
        delete file;
        done(false, error);
        return;
    }
    download_path_ = dest;
    QNetworkRequest request(QUrl(QString::fromStdString(url)));
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                         QNetworkRequest::UserVerifiedRedirectPolicy);
    request.setHeader(QNetworkRequest::UserAgentHeader,
                      QStringLiteral("alphashape-updater"));
    QNetworkReply* reply = network_->get(request);
    reply_ = reply;

    connect(reply, &QNetworkReply::readyRead, reply, [reply, file]() {
        file->write(reply->readAll());
    });
    connect(reply, &QNetworkReply::downloadProgress, reply,
            [progress](qint64 received, qint64 total) {
                if (progress) {
                    progress(received, total);
                }
            });
    connect(reply, &QNetworkReply::redirected, reply, [reply](const QUrl& target) {
        if (!allowed_url(target.toString().toStdString(), false)) {
            reply->abort();
            return;
        }
        reply->redirectAllowed();
    });
    connect(reply, &QNetworkReply::finished, reply,
            [this, reply, file, done]() {
        reply_ = nullptr;
        file->write(reply->readAll());
        file->close();
        const bool ok = !cancelled_ && reply->error() == QNetworkReply::NoError;
        std::string error;
        if (!ok) {
            error = network_error_text(reply).toStdString();
            // Never leave a partial installer behind.
            file->remove();
            download_path_.clear();
        }
        delete file;
        reply->deleteLater();
        done(ok, error);
    });
}

}  // namespace alphashape::gui
