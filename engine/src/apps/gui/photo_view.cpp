#include "apps/gui/photo_view.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <QAction>
#include <QColor>
#include <QContextMenuEvent>
#include <QFont>
#include <QFontMetrics>
#include <QGraphicsItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QGraphicsSimpleTextItem>
#include <QImage>
#include <QLineF>
#include <QMenu>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPalette>
#include <QPen>
#include <QPoint>
#include <QPolygonF>
#include <QPixmap>
#include <QRectF>
#include <QResizeEvent>
#include <QScrollBar>
#include <QStyleOptionGraphicsItem>
#include <QWheelEvent>

#include "apps/gui/app.hpp"
#include "apps/gui/job_manager.hpp"
#include "apps/gui/theme/theme.hpp"
#include "common/algorithm/markers/project_markers.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/io/camera_image.hpp"
#include "image/image.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::gui {

namespace {

/// Longest side of the pixmap kept in memory. A 24 MP frame is downscaled for display; the
/// scene is scaled back up so every coordinate stays a full-resolution pixel.
constexpr int kMaxDisplaySide = 2048;

/// Flag colours. Placed (a measured pixel) is green; predicted (projected from the marker's
/// estimated position) is a muted blue, so a hint never reads as a measurement.
QColor placed_colour() { return QColor(64, 200, 96); }
QColor predicted_colour() { return QColor(110, 150, 230); }

/// The chunk with `chunk_id` in a document held under the shared read lock. Document has no
/// const find_chunk(), so the workspaces are walked.
const Chunk* find_chunk(const Document& document, std::int32_t chunk_id) {
    for (const std::shared_ptr<Workspace>& workspace : document.workspaces()) {
        if (workspace == nullptr) {
            continue;
        }
        for (const std::shared_ptr<Chunk>& chunk : workspace->chunks()) {
            if (chunk != nullptr && chunk->id() == chunk_id) {
                return chunk.get();
            }
        }
    }
    return nullptr;
}

/// The flag's own font, in device pixels (the item ignores the view transform).
QFont flag_font() {
    QFont font;
    font.setPixelSize(11);
    font.setBold(true);
    return font;
}

}  // namespace

/// One marker drawn over the image: a pole with a pennant and the marker's label.
///
/// The item ignores the view transform, so the flag keeps a constant on-screen size however far
/// the image is zoomed; its position is still a full-resolution pixel in scene coordinates.
class MarkerFlagItem : public QGraphicsItem {
public:
    MarkerFlagItem(std::int32_t marker_id, QString label, bool placed)
        : marker_id_(marker_id), label_(std::move(label)), placed_(placed) {
        setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
        setAcceptedMouseButtons(Qt::NoButton);  // the view drives the drag
        setZValue(10.0);
        const QFontMetrics metrics(flag_font());
        bounds_ = QRectF(-4.0, -22.0,
                         20.0 + static_cast<double>(metrics.horizontalAdvance(label_)), 28.0);
    }

    std::int32_t marker_id() const { return marker_id_; }

    QRectF boundingRect() const override { return bounds_; }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) override {
        painter->save();
        painter->setRenderHint(QPainter::Antialiasing, true);
        const QColor colour = placed_ ? placed_colour() : predicted_colour();
        painter->setPen(QPen(colour, 1.6));
        painter->setBrush(colour);

        // Pole, pennant, and the exact pixel the flag anchors at.
        painter->drawLine(QPointF(0.0, 0.0), QPointF(0.0, -16.0));
        QPolygonF pennant;
        pennant << QPointF(0.0, -16.0) << QPointF(12.0, -12.0) << QPointF(0.0, -8.0);
        painter->drawPolygon(pennant);
        painter->drawEllipse(QPointF(0.0, 0.0), 2.6, 2.6);

        painter->setFont(flag_font());
        painter->drawText(QPointF(15.0, -8.0), label_);
        painter->restore();
    }

private:
    std::int32_t marker_id_ = -1;
    QString label_;
    bool placed_ = false;
    QRectF bounds_;
};

PhotoView::PhotoView(GuiApp& app, QWidget* parent) : QGraphicsView(parent), app_(app) {
    setObjectName(QStringLiteral("photo.view"));
    error_handler_ = [this](const QString& title, const QString& message) {
        QMessageBox::warning(this, title, message);
    };

    scene_ = new QGraphicsScene(this);
    setScene(scene_);
    setRenderHint(QPainter::Antialiasing, true);
    setRenderHint(QPainter::SmoothPixmapTransform, true);
    setDragMode(QGraphicsView::NoDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorViewCenter);
    setAlignment(Qt::AlignCenter);
    setFocusPolicy(Qt::StrongFocus);
    update_background();

    connect(&app_, &GuiApp::undo_changed, this, &PhotoView::refresh);
    connect(&app_, &GuiApp::document_replaced, this, [this]() { reset_view(); });
    // A different active chunk means a different frame for the same camera id, so the image is
    // reloaded, not just the flags.
    connect(&app_, &GuiApp::active_chunk_changed, this, [this](qint32) {
        if (app_.jobs().busy()) {
            stale_ = true;
            return;
        }
        if (camera_id_ >= 0) {
            set_camera(camera_id_);
        }
    });
    connect(&app_.jobs(), &JobManager::busy_changed, this, [this](bool busy) {
        if (busy || !stale_) {
            return;
        }
        // A finished job may have changed the frame itself (a calibration, a film transform),
        // so the image is reloaded rather than only the flags.
        if (camera_id_ >= 0) {
            set_camera(camera_id_);
        } else {
            refresh();
        }
    });
    connect(&app_.theme(), &Theme::changed, this, [this](ThemeMode) { update_background(); });
    reset_view();
}

PhotoView::~PhotoView() = default;

void PhotoView::set_error_handler(ErrorHandler handler) {
    error_handler_ = std::move(handler);
}

void PhotoView::report(const QString& title, const QString& message) {
    if (error_handler_) {
        error_handler_(title, message);
    }
}

std::int32_t PhotoView::active_chunk_id() const {
    return app_.read_document([](const Document& document) {
        const Chunk* chunk = document.active_chunk();
        return chunk == nullptr ? -1 : chunk->id();
    });
}

QPointF PhotoView::scene_to_pixel(const QPointF& scene) const { return scene; }

QPointF PhotoView::pixel_to_scene(const QPointF& pixel) const { return pixel; }

bool PhotoView::has_image() const { return image_item_ != nullptr; }

int PhotoView::marker_items_count() const {
    int count = 0;
    for (QGraphicsItem* item : scene_->items()) {
        if (dynamic_cast<MarkerFlagItem*>(item) != nullptr) {
            ++count;
        }
    }
    return count;
}

void PhotoView::update_background() {
    // The image is the subject, so the surround follows the theme's window colour rather than
    // the viewport's canvas choice: a light theme gets a light surround.
    const QPalette palette = app_.theme().palette_for(app_.theme().mode());
    setBackgroundBrush(palette.color(QPalette::Window));
}

void PhotoView::show_message(const QString& text) {
    if (message_item_ == nullptr) {
        message_item_ = scene_->addSimpleText(text);
        message_item_->setBrush(QColor(200, 200, 200));
        QFont font = message_item_->font();
        font.setPixelSize(14);
        message_item_->setFont(font);
    } else {
        message_item_->setText(text);
    }
    message_item_->setPos(12.0, 12.0);
    scene_->setSceneRect(0.0, 0.0, 640.0, 480.0);
}

void PhotoView::clear_marker_items() {
    for (QGraphicsItem* item : scene_->items()) {
        if (dynamic_cast<MarkerFlagItem*>(item) != nullptr) {
            scene_->removeItem(item);
            delete item;
        }
    }
    drag_item_ = nullptr;
}

bool PhotoView::load_image() {
    if (image_item_ != nullptr) {
        scene_->removeItem(image_item_);
        delete image_item_;
        image_item_ = nullptr;
    }
    if (message_item_ != nullptr) {
        scene_->removeItem(message_item_);
        delete message_item_;
        message_item_ = nullptr;
    }
    image_size_ = QSizeF();

    if (camera_id_ < 0) {
        show_message(tr("No photo selected"));
        return false;
    }
    QImage display;
    QString failure;
    const bool loaded = app_.read_document([&](const Document& document) {
        const Chunk* chunk = document.active_chunk();
        if (chunk == nullptr) {
            failure = tr("No chunk is active");
            return false;
        }
        const Camera* camera = nullptr;
        for (const Camera& candidate : chunk->cameras()) {
            if (candidate.id == camera_id_) {
                camera = &candidate;
                break;
            }
        }
        if (camera == nullptr) {
            failure = tr("The active chunk has no camera %1").arg(camera_id_);
            return false;
        }
        try {
            const Image decoded = load_camera_image(document, *chunk, *camera);
            if (decoded.empty()) {
                failure = tr("The photo is empty");
                return false;
            }
            const int width = decoded.width();
            const int height = decoded.height();
            const int channels = decoded.channels();
            if (width <= 0 || height <= 0 || (channels != 1 && channels != 3)) {
                failure = tr("The photo has an unsupported pixel layout");
                return false;
            }
            const QImage::Format format =
                channels == 1 ? QImage::Format_Grayscale8 : QImage::Format_RGB888;
            // QImage does not own the bytes: copy() detaches them before `decoded` dies.
            const QImage framed(decoded.data(), width, height, width * channels, format);
            image_size_ = QSizeF(static_cast<double>(width), static_cast<double>(height));
            display = (width > kMaxDisplaySide || height > kMaxDisplaySide)
                          ? framed.scaled(kMaxDisplaySide, kMaxDisplaySide, Qt::KeepAspectRatio,
                                          Qt::SmoothTransformation)
                                .copy()
                          : framed.copy();
            return true;
        } catch (const std::exception& error) {
            failure = QString::fromUtf8(error.what());
            return false;
        }
    });

    if (!loaded || display.isNull()) {
        show_message(failure.isEmpty() ? tr("The photo could not be loaded") : failure);
        return false;
    }

    image_item_ = scene_->addPixmap(QPixmap::fromImage(display));
    image_item_->setTransformationMode(Qt::SmoothTransformation);
    // Scene coordinates are full-resolution pixels: the downscaled pixmap is scaled back up.
    const double scale = image_size_.width() / static_cast<double>(display.width());
    image_item_->setScale(scale);
    image_item_->setPos(0.0, 0.0);
    image_item_->setZValue(0.0);
    scene_->setSceneRect(0.0, 0.0, image_size_.width(), image_size_.height());
    return true;
}

bool PhotoView::set_camera(std::int32_t camera_id) {
    if (app_.jobs().busy()) {
        return false;
    }
    const bool exists = app_.read_document([camera_id](const Document& document) {
        const Chunk* chunk = document.active_chunk();
        if (chunk == nullptr) {
            return false;
        }
        for (const Camera& camera : chunk->cameras()) {
            if (camera.id == camera_id) {
                return true;
            }
        }
        return false;
    });
    // Re-opening the same camera (a finished job, a chunk switch) keeps the user's zoom; a
    // different camera starts framed.
    const bool same_camera = camera_id_ == camera_id && image_item_ != nullptr;
    camera_id_ = camera_id;
    if (!same_camera) {
        fit_mode_ = true;
    }
    clear_marker_items();
    const bool loaded = load_image();
    refresh();
    if (loaded && !same_camera) {
        fit_image();
    }
    // True when the camera exists: the view is showing it, even if its image could not be
    // decoded (the message in the scene says why).
    return exists;
}

void PhotoView::reset_view() {
    camera_id_ = -1;
    fit_mode_ = true;
    clear_marker_items();
    load_image();
}

void PhotoView::fit_image() {
    if (image_item_ == nullptr) {
        return;
    }
    // Before the first layout pass the viewport has no size; fitting then would compute a
    // degenerate scale. The resize handler refits once there is room.
    if (viewport()->width() <= 0 || viewport()->height() <= 0) {
        fit_mode_ = true;
        return;
    }
    fitInView(image_item_, Qt::KeepAspectRatio);
    fit_mode_ = true;
}

void PhotoView::refresh() {
    if (app_.jobs().busy()) {
        stale_ = true;
        return;
    }
    stale_ = false;
    clear_marker_items();
    if (image_item_ == nullptr || camera_id_ < 0) {
        return;
    }
    struct Flag {
        std::int32_t marker_id = -1;
        QString label;
        QPointF pixel;
        bool placed = false;
    };
    std::vector<Flag> flags;
    app_.read_document([&](const Document& document) {
        const Chunk* chunk = document.active_chunk();
        if (chunk == nullptr) {
            return;
        }
        const Camera* camera = nullptr;
        for (const Camera& candidate : chunk->cameras()) {
            if (candidate.id == camera_id_) {
                camera = &candidate;
                break;
            }
        }
        if (camera == nullptr) {
            return;
        }
        // The calibration a predicted flag is projected through. A chunk built without sensors
        // (a digital-only test, an imported block) has none, so only measured flags are drawn.
        std::optional<Calibration> calibration;
        try {
            if (const Sensor* sensor = find_effective_sensor(*chunk, *camera)) {
                calibration = sensor_pipeline_calibration(*sensor);
            }
        } catch (const std::exception&) {
            calibration.reset();
        }

        for (const Marker& marker : chunk->markers()) {
            Flag flag;
            flag.marker_id = marker.id;
            flag.label = QString::fromStdString(marker.label);
            bool found = false;
            for (const MarkerProjection& projection : marker.projections) {
                if (projection.camera_id == camera_id_) {
                    flag.pixel = QPointF(projection.pixel.x(), projection.pixel.y());
                    flag.placed = true;
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (!calibration.has_value()) {
                    continue;
                }
                const std::optional<Eigen::Vector2d> predicted =
                    project_marker(marker, *camera, *calibration, /*clip_to_image=*/false);
                if (!predicted.has_value()) {
                    continue;
                }
                flag.pixel = QPointF(predicted->x(), predicted->y());
                flag.placed = false;
            }
            flags.push_back(std::move(flag));
        }
    });

    for (const Flag& flag : flags) {
        auto* item = new MarkerFlagItem(flag.marker_id, flag.label, flag.placed);
        item->setPos(flag.pixel);
        scene_->addItem(item);
    }
}

QPointF PhotoView::clamp_to_image(const QPointF& pixel) const {
    if (image_size_.isEmpty()) {
        return pixel;
    }
    return QPointF(std::clamp(pixel.x(), 0.0, image_size_.width()),
                   std::clamp(pixel.y(), 0.0, image_size_.height()));
}

std::string PhotoView::free_marker_label(std::int32_t chunk_id) const {
    return app_.read_document([chunk_id](const Document& document) {
        const Chunk* chunk = find_chunk(document, chunk_id);
        if (chunk == nullptr) {
            return std::string("point 1");
        }
        for (std::int32_t n = 1;; ++n) {
            const std::string candidate = "point " + std::to_string(n);
            bool taken = false;
            for (const Marker& marker : chunk->markers()) {
                if (marker.label == candidate) {
                    taken = true;
                    break;
                }
            }
            if (!taken) {
                return candidate;
            }
        }
    });
}

bool PhotoView::is_last_projection(std::int32_t chunk_id, std::int32_t marker_id) const {
    return app_.read_document([chunk_id, marker_id](const Document& document) {
        const Chunk* chunk = find_chunk(document, chunk_id);
        if (chunk == nullptr) {
            return false;
        }
        for (const Marker& marker : chunk->markers()) {
            if (marker.id != marker_id) {
                continue;
            }
            // A marker that carries a reference is a control point: dropping its last
            // projection must not throw the coordinate away with it.
            return marker.projections.size() <= 1U && !marker.reference.has_value();
        }
        return false;
    });
}

std::vector<std::pair<std::int32_t, QString>> PhotoView::unplaced_markers() const {
    std::vector<std::pair<std::int32_t, QString>> result;
    if (camera_id_ < 0) {
        return result;
    }
    app_.read_document([&](const Document& document) {
        const Chunk* chunk = document.active_chunk();
        if (chunk == nullptr) {
            return;
        }
        for (const Marker& marker : chunk->markers()) {
            bool placed_here = false;
            for (const MarkerProjection& projection : marker.projections) {
                if (projection.camera_id == camera_id_) {
                    placed_here = true;
                    break;
                }
            }
            if (!placed_here) {
                result.emplace_back(marker.id, QString::fromStdString(marker.label));
            }
        }
    });
    return result;
}

// --- Edits ---------------------------------------------------------------------------------

bool PhotoView::add_marker_at(const QPointF& pixel) {
    if (app_.jobs().busy() || camera_id_ < 0) {
        return false;
    }
    const std::int32_t chunk_id = active_chunk_id();
    if (chunk_id < 0) {
        return false;
    }
    const QPointF target = clamp_to_image(pixel);
    try {
        // The label variant creates the marker and its first projection in one command, so a
        // single undo removes both.
        app_.execute(make_add_marker_with_projection_command(
            chunk_id, free_marker_label(chunk_id), camera_id_,
            Eigen::Vector2d(target.x(), target.y()), /*pinned=*/true));
        return true;
    } catch (const std::exception& error) {
        report(tr("Add marker"), QString::fromUtf8(error.what()));
        return false;
    }
}

bool PhotoView::place_marker_at(std::int32_t marker_id, const QPointF& pixel) {
    if (app_.jobs().busy() || camera_id_ < 0) {
        return false;
    }
    const std::int32_t chunk_id = active_chunk_id();
    if (chunk_id < 0) {
        return false;
    }
    const QPointF target = clamp_to_image(pixel);
    try {
        app_.execute(make_add_marker_with_projection_command(
            chunk_id, marker_id, camera_id_, Eigen::Vector2d(target.x(), target.y()),
            /*pinned=*/true));
        return true;
    } catch (const std::exception& error) {
        report(tr("Place marker"), QString::fromUtf8(error.what()));
        return false;
    }
}

bool PhotoView::move_projection(std::int32_t marker_id, const QPointF& pixel) {
    if (app_.jobs().busy() || camera_id_ < 0) {
        return false;
    }
    const std::int32_t chunk_id = active_chunk_id();
    if (chunk_id < 0) {
        return false;
    }
    const QPointF target = clamp_to_image(pixel);
    try {
        app_.execute(make_move_marker_projection_command(
            chunk_id, marker_id, camera_id_, Eigen::Vector2d(target.x(), target.y())));
        return true;
    } catch (const std::exception& error) {
        report(tr("Move marker"), QString::fromUtf8(error.what()));
        return false;
    }
}

bool PhotoView::remove_projection(std::int32_t marker_id) {
    if (app_.jobs().busy() || camera_id_ < 0) {
        return false;
    }
    const std::int32_t chunk_id = active_chunk_id();
    if (chunk_id < 0) {
        return false;
    }
    try {
        app_.execute(make_remove_marker_projection_command(
            chunk_id, marker_id, camera_id_,
            /*delete_marker=*/is_last_projection(chunk_id, marker_id)));
        return true;
    } catch (const std::exception& error) {
        report(tr("Remove marker projection"), QString::fromUtf8(error.what()));
        return false;
    }
}

// --- Input ---------------------------------------------------------------------------------

MarkerFlagItem* PhotoView::flag_at(const QPoint& view_pos) const {
    MarkerFlagItem* best = nullptr;
    double best_distance = 0.0;
    for (QGraphicsItem* item : scene_->items()) {
        auto* flag = dynamic_cast<MarkerFlagItem*>(item);
        if (flag == nullptr) {
            continue;
        }
        // The flag is drawn in device pixels around its anchor, so hit-test in view space.
        const QPoint anchor = mapFromScene(flag->scenePos());
        const double ax = static_cast<double>(anchor.x());
        const double ay = static_cast<double>(anchor.y());
        const QRectF hit(ax - 5.0, ay - 22.0, 44.0, 30.0);
        const QPointF point(static_cast<double>(view_pos.x()),
                            static_cast<double>(view_pos.y()));
        if (!hit.contains(point)) {
            continue;
        }
        const double distance = QLineF(QPointF(ax, ay), point).length();
        if (best == nullptr || distance < best_distance) {
            best = flag;
            best_distance = distance;
        }
    }
    return best;
}

void PhotoView::wheelEvent(QWheelEvent* event) {
    if (image_item_ == nullptr) {
        QGraphicsView::wheelEvent(event);
        return;
    }
    const double steps = event->angleDelta().y() / 120.0;
    if (steps == 0.0) {
        event->ignore();
        return;
    }
    // AnchorUnderMouse keeps the pixel under the cursor fixed while the scale changes.
    const double factor = std::pow(1.2, steps);
    scale(factor, factor);
    fit_mode_ = false;
    event->accept();
}

void PhotoView::mousePressEvent(QMouseEvent* event) {
    const QPoint pos = event->position().toPoint();
    if (event->button() == Qt::LeftButton && !app_.jobs().busy()) {
        if (MarkerFlagItem* flag = flag_at(pos)) {
            drag_item_ = flag;
            drag_origin_ = flag->scenePos();
            drag_moved_ = false;
            event->accept();
            return;
        }
    }
    if (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton) {
        panning_ = true;
        pan_last_ = pos;
        setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }
    QGraphicsView::mousePressEvent(event);
}

void PhotoView::mouseMoveEvent(QMouseEvent* event) {
    const QPoint pos = event->position().toPoint();
    if (drag_item_ != nullptr) {
        const QPointF target = clamp_to_image(mapToScene(pos));
        drag_item_->setPos(target);
        drag_moved_ = drag_moved_ || (target - drag_origin_).manhattanLength() > 0.5;
        event->accept();
        return;
    }
    if (panning_) {
        const QPoint delta = pos - pan_last_;
        pan_last_ = pos;
        horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
        verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
        event->accept();
        return;
    }
    QGraphicsView::mouseMoveEvent(event);
}

void PhotoView::mouseReleaseEvent(QMouseEvent* event) {
    if (drag_item_ != nullptr && event->button() == Qt::LeftButton) {
        MarkerFlagItem* flag = drag_item_;
        drag_item_ = nullptr;
        if (drag_moved_) {
            const QPointF target = clamp_to_image(mapToScene(event->position().toPoint()));
            // The command rebuilds the flags through undo_changed; a failure leaves the flag
            // where the drag put it, so refresh() restores the document's position.
            if (!move_projection(flag->marker_id(), target)) {
                refresh();
            }
        }
        drag_moved_ = false;
        event->accept();
        return;
    }
    if (panning_ && (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton)) {
        panning_ = false;
        unsetCursor();
        event->accept();
        return;
    }
    QGraphicsView::mouseReleaseEvent(event);
}

void PhotoView::contextMenuEvent(QContextMenuEvent* event) {
    if (app_.jobs().busy() || image_item_ == nullptr) {
        event->ignore();
        return;
    }
    const QPointF pixel = clamp_to_image(mapToScene(event->pos()));
    MarkerFlagItem* flag = flag_at(event->pos());

    QMenu menu(this);
    menu.setObjectName(QStringLiteral("menu.photo.context"));

    QAction* add = menu.addAction(tr("Add Marker"));
    add->setObjectName(QStringLiteral("action.photo.add_marker"));
    connect(add, &QAction::triggered, this, [this, pixel]() { add_marker_at(pixel); });

    const std::vector<std::pair<std::int32_t, QString>> unplaced = unplaced_markers();
    QMenu* place = menu.addMenu(tr("Place Marker"));
    place->setObjectName(QStringLiteral("menu.photo.place"));
    place->setEnabled(!unplaced.empty());
    for (const auto& entry : unplaced) {
        QAction* action = place->addAction(entry.second);
        action->setObjectName(QStringLiteral("action.photo.place.%1").arg(entry.first));
        const std::int32_t marker_id = entry.first;
        connect(action, &QAction::triggered, this,
                [this, marker_id, pixel]() { place_marker_at(marker_id, pixel); });
    }

    if (flag != nullptr) {
        menu.addSeparator();
        QAction* remove = menu.addAction(tr("Remove Marker projection"));
        remove->setObjectName(QStringLiteral("action.photo.remove_projection"));
        const std::int32_t marker_id = flag->marker_id();
        connect(remove, &QAction::triggered, this,
                [this, marker_id]() { remove_projection(marker_id); });
    }

    menu.exec(event->globalPos());
    event->accept();
}
void PhotoView::resizeEvent(QResizeEvent* event) {
    QGraphicsView::resizeEvent(event);
    if (fit_mode_) {
        fit_image();
    }
}

}  // namespace alphashape::gui
