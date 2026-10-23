#include "apps/gui/app.hpp"

#include <iostream>

#include <QApplication>
#include <QMetaObject>
#include <QFileInfo>
#include <QString>

#include "base/utils/error.hpp"
#include "common/io/project_io.hpp"

namespace alphashape::gui {

namespace {
constexpr const char* kThemeKey = "ui/theme";
constexpr const char* kRecentKey = "files/recent";
constexpr std::size_t kUnreachable = static_cast<std::size_t>(-1);
/// The GuiApp that currently receives engine log lines (the sink is process-wide).
GuiApp* g_log_owner = nullptr;
}  // namespace

GuiApp::GuiApp(const Options& options, QObject* parent)
    : QObject(parent),
      settings_(QStringLiteral("alphashape"), QStringLiteral("alphashape")),
      document_(std::make_shared<Document>()),
      undo_stack_(document_.get()),
      theme_(this),
      jobs_(*this) {
    qRegisterMetaType<alphashape::LogLevel>("alphashape::LogLevel");
    if (options.reset_settings) {
        settings_.clear();
        settings_.sync();
    }

    Theme::load_bundled_fonts();
    // Fusion honours the whole QPalette on every platform, so widgets the style sheet does not
    // cover still follow the theme instead of the OS look.
    QApplication::setStyle(QStringLiteral("Fusion"));

    ThemeMode mode = ThemeMode::Dark;
    if (options.theme) {
        mode = *options.theme;
    } else if (const auto stored =
                   theme_from_name(settings_.value(kThemeKey).toString().toStdString())) {
        mode = *stored;
    }
    theme_.apply(mode);

    if (options.open_path) {
        open_document(*options.open_path);
    } else if (options.new_project) {
        new_document(options.new_path);
    }

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
    settings_.setValue(kThemeKey, QString::fromLatin1(theme_name(mode)));
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

QStringList GuiApp::recent_files() const {
    return settings_.value(kRecentKey).toStringList();
}

void GuiApp::add_recent_file(const QString& path) {
    if (path.isEmpty()) {
        return;
    }
    const QString absolute = QFileInfo(path).absoluteFilePath();
    QStringList list = recent_files();
    list.removeAll(absolute);
    list.prepend(absolute);
    while (list.size() > kMaxRecentFiles) {
        list.removeLast();
    }
    settings_.setValue(kRecentKey, list);
    Q_EMIT recent_files_changed();
}

void GuiApp::remove_recent_file(const QString& path) {
    QStringList list = recent_files();
    if (list.removeAll(QFileInfo(path).absoluteFilePath()) > 0) {
        settings_.setValue(kRecentKey, list);
        Q_EMIT recent_files_changed();
    }
}

void GuiApp::clear_recent_files() {
    settings_.remove(kRecentKey);
    Q_EMIT recent_files_changed();
}

QString GuiApp::document_title() const {
    const std::string path = read_document([](const Document& d) { return d.path(); });
    if (path.empty()) {
        return QStringLiteral("Untitled");
    }
    return QFileInfo(QString::fromStdString(path)).completeBaseName();
}

}  // namespace alphashape::gui
