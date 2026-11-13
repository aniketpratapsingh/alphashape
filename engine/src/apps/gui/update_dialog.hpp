#pragma once

// UpdateDialog: the "a newer version is available" window.
//
// It shows the version and the release notes (rendered from Markdown), and
// offers Download and Install, Skip This Version, Remind Me Later and View on
// GitHub. The download writes to the per-user cache, verifies SHA-256 against
// the release's SHA256SUMS, and only then hands the file to the platform
// installer. The user consent step for Windows and the AppImage restart live in
// installer.cpp.
//
// Everything network-facing goes through UpdateChecker, so a test constructs
// this dialog with a fake UpdateSource and never opens a socket.

#include <functional>

#include <QDialog>
#include <QString>

#include "apps/gui/update_checker.hpp"

class QLabel;
class QProgressBar;
class QPushButton;
class QTextBrowser;

namespace alphashape::gui {

class UpdateDialog : public QDialog {
    Q_OBJECT

public:
    using OpenUrlHandler = std::function<bool(const QString& url)>;

    UpdateDialog(const CheckResult& result, UpdateChecker& checker,
                 QWidget* parent = nullptr);

    /// Replace QDesktopServices::openUrl (tests assert the URL instead of
    /// launching a browser). Returning false is reported as a failure.
    void set_open_url_handler(OpenUrlHandler handler);

    /// Start the download/verify/install sequence. The Download button calls
    /// this; public so a test can begin it without an event-loop click.
    void start_download();

    QTextBrowser* notes() const { return notes_; }
    QLabel* version_label() const { return version_label_; }
    QLabel* status_label() const { return status_; }
    QProgressBar* progress() const { return progress_; }
    QPushButton* download_button() const { return download_; }
    QPushButton* skip_button() const { return skip_; }
    QPushButton* later_button() const { return later_; }
    QPushButton* view_button() const { return view_; }
    QPushButton* cancel_button() const { return cancel_; }

private:
    void build_ui();
    void set_busy(bool busy);
    void report_error(const QString& message);
    void open_release_page();
    void finish_download(const DownloadResult& result);

    CheckResult result_;
    UpdateChecker& checker_;
    OpenUrlHandler open_url_;

    QLabel* version_label_ = nullptr;
    QTextBrowser* notes_ = nullptr;
    QLabel* status_ = nullptr;
    QProgressBar* progress_ = nullptr;
    QPushButton* download_ = nullptr;
    QPushButton* skip_ = nullptr;
    QPushButton* later_ = nullptr;
    QPushButton* view_ = nullptr;
    QPushButton* cancel_ = nullptr;
    bool downloading_ = false;
};

}  // namespace alphashape::gui
