#include "apps/gui/viewport_placeholder.hpp"

#include <QColor>
#include <QLinearGradient>
#include <QPaintEvent>
#include <QPainter>

#include "apps/gui/theme/theme.hpp"

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
    gradient.setColorAt(0.0, QColor(dark ? "#1D252B" : "#D9E0E3"));
    gradient.setColorAt(1.0, QColor(dark ? "#141B20" : "#EDF1F2"));
    painter.fillRect(rect(), gradient);
    painter.setPen(QColor(dark ? "#B9C8D2" : "#2C3A42"));
    painter.drawText(rect(), Qt::AlignCenter, tr("3-D view"));
}

}  // namespace alphashape::gui
