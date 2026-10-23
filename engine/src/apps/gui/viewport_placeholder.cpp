#include "apps/gui/viewport_placeholder.hpp"

#include <QColor>
#include <QLinearGradient>
#include <QPaintEvent>
#include <QPainter>

#include "apps/gui/theme/theme.hpp"
#include "apps/gui/theme/viewport_colors.hpp"

namespace alphashape::gui {

ViewportPlaceholder::ViewportPlaceholder(Theme& theme, QWidget* parent)
    : QWidget(parent), mode_(theme.viewport_mode()) {
    setObjectName(QStringLiteral("viewport"));
    setMinimumSize(200, 150);
    connect(&theme, &Theme::viewport_changed, this, [this](ThemeMode mode) {
        mode_ = mode;
        update();
    });
}

void ViewportPlaceholder::paintEvent(QPaintEvent*) {
    const bool dark = mode_ == ThemeMode::Dark;
    QPainter painter(this);
    QLinearGradient gradient(0, 0, 0, height());
    namespace vc = viewport_colors;
    gradient.setColorAt(0.0, QColor(dark ? vc::dark::background : vc::light::background));
    gradient.setColorAt(1.0, QColor(dark ? vc::dark::gradient_to : vc::light::gradient_to));
    painter.fillRect(rect(), gradient);
    painter.setPen(QColor(dark ? vc::dark::label : vc::light::label));
    painter.drawText(rect(), Qt::AlignCenter, tr("3-D view"));
}

}  // namespace alphashape::gui
