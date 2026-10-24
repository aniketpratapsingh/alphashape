#pragma once

// The application theme: a QPalette (OS-integrated roles) plus the compiled style sheet.
// design/system/qt-mapping.md is the specification; design/system/tokens.json is the source
// of the colours. The sheets are generated from it by tools/theme/compile_qss.py and compiled
// into the binary as Qt resources.

#include <QColor>
#include <QObject>
#include <QPalette>
#include <QString>

#include "apps/gui/theme/theme_mode.hpp"

namespace alphashape::gui {

/// View > Viewport background. The 3-D viewport is its own instrument pane: "Follow theme"
/// maps the dark chrome to the dark canvas and the light chrome to the light canvas, while
/// Dark / Light pin it regardless of the theme (a light UI over a dark canvas is the usual
/// choice for imagery work).
enum class ViewportBackground { Follow, Dark, Light };

class Theme : public QObject {
    Q_OBJECT

public:
    explicit Theme(QObject* parent = nullptr);

    /// The mode last passed to apply() (Dark before the first call).
    ThemeMode mode() const { return mode_; }

    /// Install the palette and style sheet on the QApplication. Safe to call repeatedly:
    /// this is also how the theme switches at runtime. Emits changed() when the mode differs
    /// from the one in force (and on the first call).
    void apply(ThemeMode mode);

    ViewportBackground viewport_background() const { return viewport_background_; }
    void set_viewport_background(ViewportBackground background);

    /// The theme the viewport canvas is drawn in right now.
    ThemeMode viewport_mode() const;

    /// Palette for a mode, per qt-mapping.md section 1 (Active, Inactive and Disabled groups).
    static QPalette palette_for(ThemeMode mode);

    /// Status colours (tokens status.success / warning / error) for a mode. Panes that paint
    /// text by status (the Log console, the photo badges) take them from here so the tokens
    /// stay the single source.
    static QColor status_success(ThemeMode mode);
    static QColor status_warning(ThemeMode mode);
    static QColor status_error(ThemeMode mode);

    /// The compiled style sheet for a mode, read from the embedded resources.
    static QString style_sheet_for(ThemeMode mode);

    /// Register the bundled Inter and JetBrains Mono (SIL OFL 1.1, licence texts alongside).
    /// Returns how many families were registered; the QSS falls back to system fonts for any
    /// that are missing.
    static int load_bundled_fonts();

Q_SIGNALS:
    void changed(ThemeMode mode);
    void viewport_changed(ThemeMode mode);

private:
    ThemeMode mode_ = ThemeMode::Dark;
    bool applied_ = false;
    ViewportBackground viewport_background_ = ViewportBackground::Follow;
};

}  // namespace alphashape::gui
