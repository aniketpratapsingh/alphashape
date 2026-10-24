#include "apps/gui/viewer/viewport_widget.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

#include <QColor>
#include <QGuiApplication>
#include <QKeyEvent>
#include <QLabel>
#include <QLinearGradient>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QWheelEvent>

#include <rhi/qrhi.h>

#include "apps/gui/app.hpp"
#include "apps/gui/theme/theme.hpp"
#include "apps/gui/theme/viewport_colors.hpp"
#include "apps/gui/workspace_model.hpp"
#include "common/core/document.hpp"

namespace alphashape::gui {

namespace {

/// Platforms that cannot host a QRhi, by their QPA name. "offscreen" is the one CI and the tests
/// use and the one Qt's offscreen plugin itself reports RhiBasedRendering=false for; "minimal"
/// is the other display-less plugin Qt ships. Every real platform (cocoa, windows, xcb, wayland,
/// offscreen-with-EGL) is left to QRhiWidget, which reports through renderFailed() if the
/// device still cannot be created.
bool platform_has_rhi() {
    const QString platform = QGuiApplication::platformName();
    return platform != QLatin1String("offscreen") && platform != QLatin1String("minimal");
}

}  // namespace

/// The render canvas. It holds no state of its own: the camera, the renderer and the scene live
/// on the ViewportWidget, which outlives it (the canvas is a child).
///
/// No Q_OBJECT: it declares no signal or slot, and keeping moc out of this file keeps the RHI
/// headers in exactly one translation unit.
class ViewportWidget::Canvas : public QRhiWidget {
public:
    explicit Canvas(ViewportWidget& owner) : QRhiWidget(&owner), owner_(owner) {}

    void initialize(QRhiCommandBuffer* cb) override {
        static_cast<void>(cb);
        QRhiRenderTarget* target = renderTarget();
        if (target == nullptr) {
            owner_.on_render_failed(
                ViewportWidget::tr("the rendering device did not provide a render target"));
            return;
        }
        if (!owner_.renderer_.ensure(rhi(), target->renderPassDescriptor(),
                                     target->sampleCount())) {
            owner_.on_render_failed(owner_.renderer_.error());
        }
    }

    void render(QRhiCommandBuffer* cb) override {
        QRhiRenderTarget* target = renderTarget();
        if (target == nullptr) {
            return;
        }
        owner_.renderer_.render(cb, target, owner_.camera_, target->pixelSize());
    }

    void releaseResources() override { owner_.renderer_.release(); }

private:
    ViewportWidget& owner_;
};

std::optional<std::int32_t> selected_camera(const WorkspaceSelection* selection) {
    if (selection == nullptr) {
        return std::nullopt;
    }
    for (const NodeRef& ref : selection->items()) {
        if (ref.kind == NodeKind::Camera) {
            return ref.item_id;
        }
    }
    return std::nullopt;
}

ViewportWidget::ViewportWidget(GuiApp& app, Theme& theme, QWidget* parent)
    : QWidget(parent), app_(app), theme_(theme), mode_(theme.viewport_mode()) {
    setObjectName(QStringLiteral("viewport"));
    setMinimumSize(200, 150);
    setFocusPolicy(Qt::StrongFocus);

    if (platform_has_rhi()) {
        canvas_ = new Canvas(*this);
        canvas_->setObjectName(QStringLiteral("viewport.canvas"));
        canvas_->installEventFilter(this);
        QObject::connect(canvas_, &QRhiWidget::renderFailed, this, [this]() {
            on_render_failed(tr("the graphics device could not be created"));
        });
    }

    message_ = new QLabel(this);
    message_->setObjectName(QStringLiteral("viewport.message"));
    message_->setAlignment(Qt::AlignCenter);
    message_->setWordWrap(true);
    message_->setMargin(24);
    message_->setAttribute(Qt::WA_TransparentForMouseEvents);
    message_->hide();

    QObject::connect(&theme_, &Theme::viewport_changed, this, &ViewportWidget::on_theme_changed);
    QObject::connect(&app_, &GuiApp::document_replaced, this, [this]() { rebuild_scene(); });
    QObject::connect(&app_, &GuiApp::undo_changed, this, [this]() { rebuild_scene(); });
    // A task changes the document without commands: after every job, whatever its outcome, the
    // scene is rebuilt. Only the job that leaves the manager idle does it (a queued follow-up
    // would relock the document).
    QObject::connect(&app_.jobs(), &JobManager::job_finished, this,
                     [this](quint64, JobManager::State, const QString&) { rebuild_scene(); });

    place_children();
    rebuild_scene();
}

ViewportWidget::~ViewportWidget() {
    // The canvas is a child and is destroyed with the widget; its resources belong to a QRhi the
    // canvas owns, so dropping them here (while that QRhi is still alive) is the safe order.
    if (canvas_ != nullptr) {
        canvas_->releaseResources();
        delete canvas_;
        canvas_ = nullptr;
    }
}

void ViewportWidget::set_selection(WorkspaceSelection* selection) {
    if (selection_ != nullptr) {
        QObject::disconnect(selection_, nullptr, this, nullptr);
    }
    selection_ = selection;
    if (selection_ != nullptr) {
        QObject::connect(selection_, &WorkspaceSelection::changed, this,
                         [this]() { rebuild_scene(); });
    }
    rebuild_scene();
}

void ViewportWidget::rebuild_scene() {
    if (app_.jobs().busy()) {
        // The document belongs to the worker; serve this when the job finishes.
        stale_ = true;
        return;
    }
    stale_ = false;
    BuildOptions options = options_;
    options.grid = show_grid_;
    options.axes = show_axes_;
    options.highlight_camera = selected_camera(selection_);
    const SceneStyle style = scene_style_for(mode_);
    scene_ = app_.read_document(
        [&](const Document& document) { return build_scene(document, style, options); });
    renderer_.set_scene(scene_);
    renderer_.set_background(style.background_bottom);
    if (scene_.has_bounds) {
        camera_.set_scene_bounds(scene_.bounds_min, scene_.bounds_max);
        if (fit_pending_) {
            fit_pending_ = false;
            fit_scene();
            return;  // fit_scene() already requested the frame
        }
    }
    update_message();
    request_frame();
}

void ViewportWidget::refresh_scene() { rebuild_scene(); }

void ViewportWidget::on_theme_changed(ThemeMode mode) {
    mode_ = mode;
    rebuild_scene();
    update_message();
    request_frame();
}

void ViewportWidget::on_render_failed(const QString& reason) {
    render_error_ = reason;
    update_message();
    update();
}

void ViewportWidget::update_message() {
    QString text;
    if (!render_error_.isEmpty()) {
        text = tr("3-D view unavailable: %1").arg(render_error_);
    } else if (canvas_ == nullptr) {
        text = tr("3-D view unavailable: this platform provides no graphics backend.");
    }
    message_->setText(text);
    message_->setVisible(!text.isEmpty());
    if (message_->isVisible()) {
        message_->raise();
    }
}

void ViewportWidget::place_children() {
    if (canvas_ != nullptr) {
        canvas_->setGeometry(rect());
    }
    if (message_ != nullptr) {
        message_->setGeometry(rect());
    }
}

void ViewportWidget::request_frame() {
    if (canvas_ != nullptr) {
        canvas_->update();
    }
    update();
}

int ViewportWidget::canvas_height() const {
    const int height = canvas_ != nullptr ? canvas_->height() : this->height();
    return std::max(height, 1);
}

void ViewportWidget::fit_scene() {
    if (!camera_.has_scene()) {
        return;
    }
    const int width = std::max(canvas_ != nullptr ? canvas_->width() : this->width(), 1);
    const int height = canvas_height();
    camera_.fit(static_cast<double>(width) / static_cast<double>(height));
    request_frame();
}

void ViewportWidget::toggle_projection() {
    camera_.toggle_projection();
    Q_EMIT projection_changed(camera_.orthographic());
    request_frame();
}

void ViewportWidget::set_orthographic(bool on) {
    if (camera_.orthographic() == on) {
        return;
    }
    camera_.set_orthographic(on);
    Q_EMIT projection_changed(on);
    request_frame();
}

void ViewportWidget::set_show_grid(bool on) {
    if (show_grid_ == on) {
        return;
    }
    show_grid_ = on;
    rebuild_scene();
}

void ViewportWidget::set_show_axes(bool on) {
    if (show_axes_ == on) {
        return;
    }
    show_axes_ = on;
    rebuild_scene();
}

void ViewportWidget::handle_mouse_press(const QPoint& position, Qt::MouseButton button) {
    last_mouse_ = position;
    dragging_ = true;
    drag_button_ = button;
    setFocus();
}

void ViewportWidget::handle_mouse_move(const QPoint& position, Qt::MouseButtons buttons) {
    if (!dragging_) {
        return;
    }
    const QPoint delta = position - last_mouse_;
    last_mouse_ = position;
    if (delta.isNull()) {
        return;
    }
    const bool pan = drag_button_ == Qt::RightButton || drag_button_ == Qt::MiddleButton ||
                     (buttons & Qt::MiddleButton) != 0;
    if (pan) {
        camera_.pan(static_cast<double>(delta.x()), static_cast<double>(delta.y()),
                    canvas_height());
    } else {
        camera_.orbit(static_cast<double>(delta.x()), static_cast<double>(delta.y()));
    }
    request_frame();
}

void ViewportWidget::handle_wheel(int angle_delta) {
    // One wheel notch (120 units) is one dolly step; a trackpad sends fractions of that.
    camera_.dolly(static_cast<double>(angle_delta) / 120.0);
    request_frame();
}

void ViewportWidget::handle_key(int key) {
    switch (key) {
        case Qt::Key_F:
            fit_scene();
            break;
        case Qt::Key_5:
        case Qt::Key_P:
            toggle_projection();
            break;
        default:
            break;
    }
}

bool ViewportWidget::eventFilter(QObject* watched, QEvent* event) {
    if (watched != canvas_) {
        return QWidget::eventFilter(watched, event);
    }
    switch (event->type()) {
        case QEvent::MouseButtonPress: {
            auto* mouse = static_cast<QMouseEvent*>(event);
            if (mouse->button() == Qt::LeftButton || mouse->button() == Qt::RightButton ||
                mouse->button() == Qt::MiddleButton) {
                handle_mouse_press(mouse->position().toPoint(), mouse->button());
            }
            return false;
        }
        case QEvent::MouseMove: {
            auto* mouse = static_cast<QMouseEvent*>(event);
            handle_mouse_move(mouse->position().toPoint(), mouse->buttons());
            return false;
        }
        case QEvent::MouseButtonRelease:
            dragging_ = false;
            drag_button_ = Qt::NoButton;
            return false;
        case QEvent::Wheel: {
            auto* wheel = static_cast<QWheelEvent*>(event);
            handle_wheel(wheel->angleDelta().y());
            return true;
        }
        case QEvent::KeyPress: {
            handle_key(static_cast<QKeyEvent*>(event)->key());
            return true;
        }
        default:
            break;
    }
    return QWidget::eventFilter(watched, event);
}

void ViewportWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    place_children();
    // A resize changes the aspect the camera frames with; keep the scene fitted while it still
    // fills exactly the way fit() left it.
    if (camera_.has_scene()) {
        request_frame();
    }
}

void ViewportWidget::paintEvent(QPaintEvent*) {
    const bool dark = mode_ == ThemeMode::Dark;
    QPainter painter(this);
    QLinearGradient gradient(0, 0, 0, height());
    namespace vc = viewport_colors;
    gradient.setColorAt(0.0, QColor(dark ? vc::dark::background : vc::light::background));
    gradient.setColorAt(1.0, QColor(dark ? vc::dark::gradient_to : vc::light::gradient_to));
    painter.fillRect(rect(), gradient);
}

}  // namespace alphashape::gui
