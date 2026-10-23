#include "apps/gui/app.hpp"

#include <QApplication>
#include <QFileInfo>
#include <QString>

#include "base/utils/error.hpp"
#include "common/io/project_io.hpp"

namespace alphashape::gui {

namespace {
constexpr const char* kThemeKey = "ui/theme";
constexpr const char* kRecentKey = "files/recent";
constexpr std::size_t kUnreachable = static_cast<std::size_t>(-1);
}  // namespace

GuiApp::GuiApp(const Options& options, QObject* parent)
    : QObject(parent),
      settings_(QStringLiteral("alphashape"), QStringLiteral("alphashape")),
      document_(std::make_shared<Document>()),
      undo_stack_(document_.get()),
      theme_(this) {
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
}

GuiApp::~GuiApp() = default;

void GuiApp::set_theme(ThemeMode mode) {
    theme_.apply(mode);
    settings_.setValue(kThemeKey, QString::fromLatin1(theme_name(mode)));
}

void GuiApp::new_document(const std::string& path) {
    undo_stack_.clear();
    document_ = path.empty() ? std::make_shared<Document>() : std::make_shared<Document>(path);
    undo_stack_.bind(document_.get());
    reset_dirty_tracking();
    Q_EMIT document_replaced();
    Q_EMIT undo_changed();
    publish_dirty();
}

void GuiApp::open_document(const std::string& path) {
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
    if (document_->path().empty()) {
        throw Error("the project has no file name yet");
    }
    ProjectIO::save(*document_, document_->path());
    reset_dirty_tracking();
    add_recent_file(QString::fromStdString(document_->path()));
    publish_dirty();
}

void GuiApp::save_document_as(const std::string& path) {
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
    undo_stack_.undo();
    Q_EMIT undo_changed();
    publish_dirty();
}

void GuiApp::redo() {
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
    if (document_->path().empty()) {
        return QStringLiteral("Untitled");
    }
    return QFileInfo(QString::fromStdString(document_->path())).completeBaseName();
}

}  // namespace alphashape::gui
