#include "apps/gui/app.hpp"

#include <QApplication>
#include <QFileInfo>
#include <QString>

#include "common/io/project_io.hpp"

namespace alphashape::gui {

namespace {
constexpr const char* kThemeKey = "ui/theme";
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
    emit document_replaced();
}

void GuiApp::open_document(const std::string& path) {
    std::shared_ptr<Document> loaded = ProjectIO::load(path);
    undo_stack_.clear();
    document_ = std::move(loaded);
    undo_stack_.bind(document_.get());
    emit document_replaced();
}

QString GuiApp::document_title() const {
    if (document_->path().empty()) {
        return QStringLiteral("Untitled");
    }
    return QFileInfo(QString::fromStdString(document_->path())).completeBaseName();
}

}  // namespace alphashape::gui
