#include "apps/gui/theme/theme.hpp"

#include <QApplication>
#include <QColor>
#include <QFile>
#include <QFontDatabase>
#include <QRgb>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"

namespace alphashape::gui {

namespace {

// One column of the table in qt-mapping.md section 1. The comments name the token each value
// comes from; the Theme.PaletteMatchesTokens test re-reads tokens.json and compares.
struct PaletteColors {
    QRgb window;            // surface.app
    QRgb window_text;       // text.primary
    QRgb base;              // dark surface.panel, light surface.raised
    QRgb alternate_base;
    QRgb text;              // text.primary
    QRgb placeholder_text;  // text.disabled
    QRgb button;
    QRgb button_text;       // text.primary
    QRgb bright_text;
    QRgb highlight;         // selection.bg
    QRgb highlight_inactive;  // selection.bgInactive
    QRgb highlighted_text;  // text.primary
    QRgb accent;            // accent.base
    QRgb link;
    QRgb link_visited;      // status.processing
    QRgb tool_tip_base;
    QRgb tool_tip_text;     // text.primary
    QRgb light;
    QRgb midlight;
    QRgb mid;
    QRgb dark;
    QRgb shadow;
    QRgb disabled_text;     // text.disabled
};

constexpr PaletteColors kDark = {
    0xFF0D1114, 0xFFE8EEF1, 0xFF12171B, 0xFF171E23, 0xFFE8EEF1, 0xFF6F7E88,
    0xFF171E23, 0xFFE8EEF1, 0xFFFFFFFF, 0xFF1E3F3C, 0xFF1A2B2E, 0xFFE8EEF1,
    0xFF35BDAF, 0xFF6FD3C8, 0xFFA78BFA, 0xFF1D252B, 0xFFE8EEF1, 0xFF33414B,
    0xFF28323A, 0xFF33414B, 0xFF28323A, 0xFF000000, 0xFF6F7E88,
};

constexpr PaletteColors kLight = {
    0xFFEDF0F1, 0xFF101619, 0xFFFFFFFF, 0xFFF2F5F5, 0xFF101619, 0xFF76838C,
    0xFFFFFFFF, 0xFF101619, 0xFF000000, 0xFFCFEAE6, 0xFFE3F1EF, 0xFF101619,
    0xFF0E7C72, 0xFF0B6E66, 0xFF6D4AC0, 0xFFFFFFFF, 0xFF101619, 0xFFFFFFFF,
    0xFFF2F5F5, 0xFFC7D0D4, 0xFFDDE3E5, 0xFFA3B0B6, 0xFF76838C,
};

QString read_resource(const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        throw Error("missing embedded resource " + path.toStdString());
    }
    return QString::fromUtf8(file.readAll());
}

}  // namespace

Theme::Theme(QObject* parent) : QObject(parent) {}

QPalette Theme::palette_for(ThemeMode mode) {
    const PaletteColors& c = (mode == ThemeMode::Dark) ? kDark : kLight;
    QPalette p;
    // setColor(role, ...) writes all three colour groups; the group-specific differences follow.
    p.setColor(QPalette::Window, QColor::fromRgb(c.window));
    p.setColor(QPalette::WindowText, QColor::fromRgb(c.window_text));
    p.setColor(QPalette::Base, QColor::fromRgb(c.base));
    p.setColor(QPalette::AlternateBase, QColor::fromRgb(c.alternate_base));
    p.setColor(QPalette::Text, QColor::fromRgb(c.text));
    p.setColor(QPalette::PlaceholderText, QColor::fromRgb(c.placeholder_text));
    p.setColor(QPalette::Button, QColor::fromRgb(c.button));
    p.setColor(QPalette::ButtonText, QColor::fromRgb(c.button_text));
    p.setColor(QPalette::BrightText, QColor::fromRgb(c.bright_text));
    p.setColor(QPalette::Highlight, QColor::fromRgb(c.highlight));
    p.setColor(QPalette::HighlightedText, QColor::fromRgb(c.highlighted_text));
    p.setColor(QPalette::Accent, QColor::fromRgb(c.accent));
    p.setColor(QPalette::Link, QColor::fromRgb(c.link));
    p.setColor(QPalette::LinkVisited, QColor::fromRgb(c.link_visited));
    p.setColor(QPalette::ToolTipBase, QColor::fromRgb(c.tool_tip_base));
    p.setColor(QPalette::ToolTipText, QColor::fromRgb(c.tool_tip_text));
    p.setColor(QPalette::Light, QColor::fromRgb(c.light));
    p.setColor(QPalette::Midlight, QColor::fromRgb(c.midlight));
    p.setColor(QPalette::Mid, QColor::fromRgb(c.mid));
    p.setColor(QPalette::Dark, QColor::fromRgb(c.dark));
    p.setColor(QPalette::Shadow, QColor::fromRgb(c.shadow));

    // Inactive: unfocused selection is visibly calmer.
    p.setColor(QPalette::Inactive, QPalette::Highlight, QColor::fromRgb(c.highlight_inactive));

    // Disabled: same hues, disabled text colour.
    const QColor disabled = QColor::fromRgb(c.disabled_text);
    p.setColor(QPalette::Disabled, QPalette::WindowText, disabled);
    p.setColor(QPalette::Disabled, QPalette::Text, disabled);
    p.setColor(QPalette::Disabled, QPalette::ButtonText, disabled);
    return p;
}

QString Theme::style_sheet_for(ThemeMode mode) {
    return read_resource(mode == ThemeMode::Dark
                             ? QStringLiteral(":/alphashape/theme/alphashape-dark.qss")
                             : QStringLiteral(":/alphashape/theme/alphashape-light.qss"));
}

int Theme::load_bundled_fonts() {
    static const char* const kFonts[] = {
        ":/alphashape/fonts/Inter-Variable.ttf",
        ":/alphashape/fonts/JetBrainsMono-Variable.ttf",
    };
    int loaded = 0;
    for (const char* path : kFonts) {
        if (QFontDatabase::addApplicationFont(QString::fromLatin1(path)) >= 0) {
            ++loaded;
        } else {
            alphashape::log(LogLevel::Warn) << "could not load bundled font " << path
                                << "; falling back to system fonts";
        }
    }
    return loaded;
}

void Theme::apply(ThemeMode mode) {
    // Read the sheet first: if the resource is missing, nothing has been half-applied.
    const QString sheet = style_sheet_for(mode);
    const bool first = !applied_;
    const ThemeMode previous_viewport = viewport_mode();
    const bool mode_changed = first || mode != mode_;

    mode_ = mode;
    applied_ = true;
    QApplication::setPalette(palette_for(mode));
    qApp->setStyleSheet(sheet);

    if (mode_changed) {
        emit changed(mode);
    }
    if (viewport_mode() != previous_viewport) {
        emit viewport_changed(viewport_mode());
    }
}

ThemeMode Theme::viewport_mode() const {
    switch (viewport_background_) {
        case ViewportBackground::Dark: return ThemeMode::Dark;
        case ViewportBackground::Light: return ThemeMode::Light;
        case ViewportBackground::Follow: break;
    }
    return mode_;
}

void Theme::set_viewport_background(ViewportBackground background) {
    const ThemeMode before = viewport_mode();
    viewport_background_ = background;
    if (viewport_mode() != before) {
        emit viewport_changed(viewport_mode());
    }
}

}  // namespace alphashape::gui
