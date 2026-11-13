#include "apps/gui/update_dialog.hpp"

#include <QCoreApplication>
#include <QDesktopServices>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QTextBrowser>
#include <QUrl>
#include <QVBoxLayout>

#include <utility>

#include "apps/gui/installer.hpp"

namespace alphashape::gui {

UpdateDialog::UpdateDialog(const CheckResult& result, UpdateChecker& checker, QWidget* parent)
    : QDialog(parent), result_(result), checker_(checker) {
    setObjectName(QStringLiteral("updateDialog"));
    setWindowTitle(tr("Software Update"));
    build_ui();
}

void UpdateDialog::set_open_url_handler(OpenUrlHandler handler) {
    open_url_ = std::move(handler);
}

void UpdateDialog::build_ui() {
    auto* layout = new QVBoxLayout(this);

    version_label_ = new QLabel(this);
    version_label_->setObjectName(QStringLiteral("update.version"));
    version_label_->setText(tr("alphashape %1 is available. You have %2.")
                                .arg(QString::fromStdString(result_.release.version.to_string()),
                                     QString::fromStdString(running_version())));
    version_label_->setWordWrap(true);
    layout->addWidget(version_label_);

    notes_ = new QTextBrowser(this);
    notes_->setObjectName(QStringLiteral("update.notes"));
    notes_->setOpenExternalLinks(true);
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
    notes_->setMarkdown(QString::fromStdString(result_.release.body));
#else
    notes_->setPlainText(QString::fromStdString(result_.release.body));
#endif
    layout->addWidget(notes_, 1);

    status_ = new QLabel(this);
    status_->setObjectName(QStringLiteral("update.status"));
    status_->setWordWrap(true);
    status_->setVisible(false);
    layout->addWidget(status_);

    progress_ = new QProgressBar(this);
    progress_->setObjectName(QStringLiteral("update.progress"));
    progress_->setRange(0, 100);
    progress_->setVisible(false);
    layout->addWidget(progress_);

    auto* buttons = new QHBoxLayout();
    download_ = new QPushButton(tr("Download and Install"), this);
    download_->setObjectName(QStringLiteral("update.download"));
    download_->setDefault(true);
    skip_ = new QPushButton(tr("Skip This Version"), this);
    skip_->setObjectName(QStringLiteral("update.skip"));
    later_ = new QPushButton(tr("Remind Me Later"), this);
    later_->setObjectName(QStringLiteral("update.later"));
    view_ = new QPushButton(tr("View on GitHub"), this);
    view_->setObjectName(QStringLiteral("update.view_github"));
    cancel_ = new QPushButton(tr("Cancel"), this);
    cancel_->setObjectName(QStringLiteral("update.cancel"));
    cancel_->setVisible(false);

    buttons->addWidget(view_);
    buttons->addStretch();
    buttons->addWidget(skip_);
    buttons->addWidget(later_);
    buttons->addWidget(cancel_);
    buttons->addWidget(download_);
    layout->addLayout(buttons);

    connect(download_, &QPushButton::clicked, this, &UpdateDialog::start_download);
    connect(skip_, &QPushButton::clicked, this, [this]() {
        checker_.skip_version(result_.release.version);
        accept();
    });
    connect(later_, &QPushButton::clicked, this, &QDialog::reject);
    connect(view_, &QPushButton::clicked, this, &UpdateDialog::open_release_page);
    connect(cancel_, &QPushButton::clicked, this, [this]() {
        status_->setText(tr("Cancelling the download..."));
        status_->setVisible(true);
        cancel_->setEnabled(false);
        checker_.cancel_download();
    });
}

void UpdateDialog::set_busy(bool busy) {
    download_->setEnabled(!busy);
    skip_->setEnabled(!busy);
    later_->setEnabled(!busy);
    view_->setEnabled(!busy);
    cancel_->setVisible(busy);
    cancel_->setEnabled(busy);
    progress_->setVisible(busy);
}

void UpdateDialog::report_error(const QString& message) {
    status_->setText(message);
    status_->setVisible(true);
}

void UpdateDialog::open_release_page() {
    const QString url = result_.release.html_url.empty()
                            ? QString::fromLatin1(update::kRepositoryUrl)
                            : QString::fromStdString(result_.release.html_url);
    const bool ok = open_url_ ? open_url_(url) : QDesktopServices::openUrl(QUrl(url));
    if (!ok) {
        report_error(tr("Could not open %1").arg(url));
    }
}

void UpdateDialog::start_download() {
    if (downloading_) {
        return;
    }
    const update::Target target = update::host_target();
    const update::Asset* asset = update::choose_asset(result_.release, target);
    if (asset == nullptr) {
        report_error(QString::fromStdString(update::no_package_reason(target)));
        return;
    }

    downloading_ = true;
    status_->setVisible(false);
    progress_->setRange(0, 100);
    progress_->setValue(0);
    set_busy(true);

    checker_.download_and_verify(
        result_.release, *asset, UpdateChecker::download_directory(),
        [this](qint64 received, qint64 total) {
            if (!downloading_) {
                return;
            }
            if (total > 0) {
                progress_->setRange(0, 100);
                progress_->setValue(static_cast<int>(received * 100 / total));
            } else {
                progress_->setRange(0, 0);  // unknown size: busy indicator
            }
        },
        [this](const DownloadResult& download) { finish_download(download); });
}

void UpdateDialog::finish_download(const DownloadResult& download) {
    downloading_ = false;
    set_busy(false);
    progress_->setVisible(false);

    if (!download.ok) {
        report_error(QString::fromStdString(download.error));
        return;
    }

    const InstallResult install =
        install_update(result_.release, download.path, update::host_target(), this);
    if (!install.error.empty()) {
        report_error(install.message);
        return;
    }
    if (!install.message.isEmpty()) {
        QMessageBox::information(this, tr("Update"), install.message);
    }
    accept();
}

}  // namespace alphashape::gui
