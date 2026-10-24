#include "apps/gui/app.hpp"

#include <iostream>
#include <memory>

#include <QApplication>
#include <QByteArray>
#include <QMetaObject>
#include <QDir>
#include <QFileInfo>
#include <QString>
#include <QSettings>
#include <QtGlobal>

#include "base/utils/error.hpp"
#include "common/io/project_io.hpp"

namespace alphashape::gui {

namespace {
constexpr const char* kThemeKey = "ui/theme";
constexpr const char* kRecentKey = "files/recent";
constexpr std::size_t kUnreachable = static_cast<std::size_t>(-1);
/// The GuiApp that currently receives engine log lines (the sink is process-wide).
GuiApp* g_log_owner = nullptr;

/// The platform store, or an explicit INI file named by ALPHASHAPE_GUI_SETTINGS_FILE. The file
/// form is a test seam: QSettings' default store is one machine-wide registry key on Windows, so
/// test processes running in parallel would read and clear each other's settings (a neighbour's
/// reset wiped the layout another test had just saved). The environment variable covers the
/// spawned alphashape-gui binary as well as the in-process apps.
std::unique_ptr<QSettings> make_settings() {
    const QByteArray file = qgetenv("ALPHASHAPE_GUI_SETTINGS_FILE");
    if (!file.isEmpty()) {
        return std::make_unique<QSettings>(QString::fromLocal8Bit(file), QSettings::IniFormat);
    }
    return std::make_unique<QSettings>(QStringLiteral("alphashape"),
                                       QStringLiteral("alphashape"));
}
}  // namespace

GuiApp::GuiApp(const Options& options, QObject* parent)
    : QObject(parent),
      settings_(make_settings()),
      document_(std::make_shared<Document>()),
      undo_stack_(document_.get()),
      theme_(this),
      jobs_(*this) {
    qRegisterMetaType<alphashape::LogLevel>("alphashape::LogLevel");
    if (options.reset_settings) {
        settings_->clear();
        settings_->sync();
    }

    recent_ = settings_->value(kRecentKey).toStringList();

    Theme::load_bundled_fonts();
    // Fusion honours the whole QPalette on every platform, so widgets the style sheet does not
    // cover still follow the theme instead of the OS look.
    QApplication::setStyle(QStringLiteral("Fusion"));

    ThemeMode mode = ThemeMode::Dark;
    if (options.theme) {
        mode = *options.theme;
    } else if (const auto stored =
                   theme_from_name(settings_->value(kThemeKey).toString().toStdString())) {
        mode = *stored;
    }
    theme_.apply(mode);

    if (options.open_path) {
        open_document(*options.open_path);
    } else if (options.new_project) {
        new_document(options.new_path);
    }
    project_requested_ = options.open_path.has_value() || options.new_project;

    // Engine log lines may come from the worker thread; hand them to the GUI thread.
    g_log_owner = this;
    Log::set_sink([this](LogLevel level, const std::string& message) {
        // Keep warnings and errors on the console too, so CI logs still show them.
        if (level >= LogLevel::Warn && level != LogLevel::Off) {
            std::cerr << '[' << (level == LogLevel::Warn ? "warn" : "error") << "] " << message
                      << '\n';
        }
        QMetaObject::invokeMethod(
            this,
            [this, level, text = QString::fromStdString(message)]() {
                Q_EMIT log_message(level, text);
            },
            Qt::QueuedConnection);
    });
}

GuiApp::~GuiApp() {
    jobs_.shutdown();  // the worker uses the document and the lock, so end it first
    if (g_log_owner == this) {
        Log::set_sink({});
        g_log_owner = nullptr;
    }
}

void GuiApp::require_idle(const char* what) const {
    if (jobs_.busy()) {
        throw Error(std::string("cannot ") + what + " while a job is running");
    }
}

void GuiApp::task_changed_document() {
    undo_stack_.clear();
    force_dirty_ = true;
    clean_position_ = undo_stack_.position();
    Q_EMIT undo_changed();
    publish_dirty();
}

void GuiApp::set_theme(ThemeMode mode) {
    theme_.apply(mode);
    settings_->setValue(kThemeKey, QString::fromLatin1(theme_name(mode)));
}

void GuiApp::new_document(const std::string& path) {
    require_idle("replace the project");
    undo_stack_.clear();
    document_ = path.empty() ? std::make_shared<Document>() : std::make_shared<Document>(path);
    undo_stack_.bind(document_.get());
    reset_dirty_tracking();
    Q_EMIT document_replaced();
    Q_EMIT undo_changed();
    publish_dirty();
}

void GuiApp::open_document(const std::string& path) {
    require_idle("open a project");
    std::shared_ptr<Document> loaded = ProjectIO::load(path);
    undo_stack_.clear();
    document_ = std::move(loaded);
    undo_stack_.bind(document_.get());
    reset_dirty_tracking();
    add_recent_file(QString::fromStdString(path));
    Q_EMIT document_replaced();
    Q_EMIT undo_changed();
    publish_dirty();
}

void GuiApp::save_document() {
    require_idle("save the project");
    const std::string path = read_document([](const Document& d) { return d.path(); });
    if (path.empty()) {
        throw Error("the project has no file name yet");
    }
    ProjectIO::save(*document_, path);
    reset_dirty_tracking();
    add_recent_file(QString::fromStdString(path));
    publish_dirty();
}

void GuiApp::save_document_as(const std::string& path) {
    require_idle("save the project");
    ProjectIO::save(*document_, path);
    document_->set_path(path);
    reset_dirty_tracking();
    add_recent_file(QString::fromStdString(path));
    Q_EMIT path_changed();
    publish_dirty();
}

bool GuiApp::dirty() const {
    return force_dirty_ || undo_stack_.position() != clean_position_;
}

void GuiApp::mark_dirty() {
    force_dirty_ = true;
    publish_dirty();
}

void GuiApp::reset_dirty_tracking() {
    force_dirty_ = false;
    clean_position_ = undo_stack_.position();
}

void GuiApp::publish_dirty() {
    const bool now = dirty();
    if (now != last_dirty_) {
        last_dirty_ = now;
        Q_EMIT dirty_changed(now);
    }
}

void GuiApp::execute(std::unique_ptr<AbstractCommand> command) {
    require_idle("edit the project");
    const std::size_t before = undo_stack_.position();
    const std::size_t dropped_before = undo_stack_.dropped_count();
    undo_stack_.execute(std::move(command));  // throws without touching the document
    // Executing from inside the history discards the redo branch; if the clean state lived in
    // it, it can never be returned to.
    // The same holds when the depth cap pushed old entries off the bottom.
    if ((before < clean_position_ || undo_stack_.dropped_count() != dropped_before) &&
        clean_position_ != kUnreachable) {
        clean_position_ = kUnreachable;
    }
    Q_EMIT undo_changed();
    publish_dirty();
}

void GuiApp::undo() {
    require_idle("undo");
    undo_stack_.undo();
    Q_EMIT undo_changed();
    publish_dirty();
}

void GuiApp::redo() {
    require_idle("redo");
    undo_stack_.redo();
    Q_EMIT undo_changed();
    publish_dirty();
}

void GuiApp::begin_macro(const std::string& name) {
    require_idle("edit the project");
    undo_stack_.begin_macro(name);
}

void GuiApp::end_macro() {
    // Closing a macro never touches the document, so it is allowed even if a job started in
    // between: the edits the group holds are already applied.
    undo_stack_.end_macro();
    Q_EMIT undo_changed();
    publish_dirty();
}

QStringList GuiApp::recent_files() const {
    return recent_;
}

// The in-memory list is authoritative; QSettings is only where it is persisted. QSettings writes
// lazily and re-reads the file when it looks changed; on Windows a write that loses a race with a
// lock (antivirus, indexer) leaves the file stale and a later re-read then replaced what had just
// been stored, which made the list come back empty or old. Reading our own copy removes that
// dependency on write timing.
void GuiApp::store_recent_files() {
    if (recent_.isEmpty()) {
        settings_->remove(kRecentKey);
    } else {
        settings_->setValue(kRecentKey, recent_);
    }
    settings_->sync();
}

namespace {
// Same normal form for storing and for matching: absolute, '/' separators, "." and ".." resolved.
QString recent_key(const QString& path) {
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}
// Windows paths ignore case, and so must the duplicate check or "C:/x" and "c:/x" list twice.
#ifdef Q_OS_WIN
constexpr Qt::CaseSensitivity kPathCase = Qt::CaseInsensitive;
#else
constexpr Qt::CaseSensitivity kPathCase = Qt::CaseSensitive;
#endif
void remove_path(QStringList& list, const QString& key) {
    for (qsizetype i = list.size(); i-- > 0;) {
        if (list[i].compare(key, kPathCase) == 0) {
            list.removeAt(i);
        }
    }
}
}  // namespace

void GuiApp::add_recent_file(const QString& path) {
    if (path.isEmpty()) {
        return;
    }
    const QString absolute = recent_key(path);
    remove_path(recent_, absolute);
    recent_.prepend(absolute);
    while (recent_.size() > kMaxRecentFiles) {
        recent_.removeLast();
    }
    store_recent_files();
    Q_EMIT recent_files_changed();
}

void GuiApp::remove_recent_file(const QString& path) {
    const qsizetype before = recent_.size();
    remove_path(recent_, recent_key(path));
    if (recent_.size() != before) {
        store_recent_files();
        Q_EMIT recent_files_changed();
    }
}

void GuiApp::clear_recent_files() {
    recent_.clear();
    store_recent_files();
    Q_EMIT recent_files_changed();
}

QString document_title_of(const Document& document) {
    const std::string path = document.path();
    if (path.empty()) {
        return QStringLiteral("Untitled");
    }
    return QFileInfo(QString::fromStdString(path)).completeBaseName();
}

QString GuiApp::document_title() const {
    return read_document([](const Document& d) { return document_title_of(d); });
}

}  // namespace alphashape::gui
