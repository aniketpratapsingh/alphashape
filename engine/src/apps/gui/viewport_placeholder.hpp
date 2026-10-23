#pragma once

#include <QWidget>

#include "apps/gui/theme/theme_mode.hpp"

namespace alphashape::gui {

class Theme;

/// Stand-in for the 3-D view: a canvas in the theme's viewport mode with the label "3-D view".
/// The real viewer replaces it later; the canvas colours here mirror the `viewport.bg` and
/// `viewport.bgGradientTo` tokens of design/system/tokens.json.
class ViewportPlaceholder : public QWidget {
public:
    ViewportPlaceholder(Theme& theme, QWidget* parent = nullptr);

    ThemeMode canvas_mode() const { return mode_; }

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    ThemeMode mode_;
};

}  // namespace alphashape::gui
