#pragma once

// PhotoView: one camera's image with its markers, the "Photo" tab of the central area.
//
// The view shows the frame the camera is linked to (decoded through the engine's image IO,
// so a film camera shows its film-normalised resampling) and draws every marker of the active
// chunk as a flag:
//   * green  - the marker has a projection in this camera (a measured pixel);
//   * blue   - the marker has an estimated position but no projection here, so the flag sits
//              where project_marker() predicts it (a hint, not a measurement).
//
// Scene coordinates ARE full-resolution image pixels: the pixmap is downscaled for display but
// the item is scaled back up, so a click at scene (x, y) is pixel (x, y) of the original frame.
// That keeps every marker coordinate in the same space the engine and the project file use.
//
// Editing goes through the command layer, so every placement, move and removal is one undo
// entry. While a job owns the document the view refuses edits and only marks itself stale; it
// rebuilds when the manager goes idle.
//
// The public add/place/move/remove methods are what the context menu and the drag handler call,
// so a test can exercise the whole edit path without synthesising mouse events.

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <QGraphicsView>
#include <QPoint>
#include <QPointF>
#include <QSizeF>
#include <QString>

class QContextMenuEvent;
class QGraphicsPixmapItem;
class QGraphicsScene;
class QGraphicsSimpleTextItem;
class QMouseEvent;
class QResizeEvent;
class QWheelEvent;

namespace alphashape::gui {

class GuiApp;
class MarkerFlagItem;

class PhotoView : public QGraphicsView {
    Q_OBJECT

public:
    explicit PhotoView(GuiApp& app, QWidget* parent = nullptr);
    ~PhotoView() override;

    /// Show `camera_id`'s frame. False when the active chunk has no such camera (or a job owns
    /// the document); the view then shows nothing. A camera whose image cannot be decoded still
    /// opens, with a message in place of the picture.
    bool set_camera(std::int32_t camera_id);
    std::int32_t camera_id() const { return camera_id_; }
    /// Forget the camera and show the "no photo selected" message (a replaced document).
    void reset_view();

    /// Number of marker flags in the scene (placed and predicted together).
    int marker_items_count() const;
    /// True when a frame is decoded and shown.
    bool has_image() const;

    /// Scene coordinates are full-resolution pixels; these exist so a caller never has to know
    /// that, and so a future display transform stays inside this class.
    QPointF scene_to_pixel(const QPointF& scene) const;
    QPointF pixel_to_scene(const QPointF& pixel) const;

    /// Create a marker with one projection at `pixel` in this camera (one undo entry).
    bool add_marker_at(const QPointF& pixel);
    /// Add (or replace) `marker_id`'s projection in this camera at `pixel`.
    bool place_marker_at(std::int32_t marker_id, const QPointF& pixel);
    /// Move `marker_id`'s projection in this camera to `pixel`; a predicted flag becomes placed.
    bool move_projection(std::int32_t marker_id, const QPointF& pixel);
    /// Remove `marker_id`'s projection in this camera. The marker itself goes when this was its
    /// last projection, so a stray click does not leave an empty marker behind.
    bool remove_projection(std::int32_t marker_id);

    /// Rebuild the flags from the document. A no-op while a job runs (the view marks itself
    /// stale and rebuilds when the job finishes).
    void refresh();
    /// Frame the whole image in the viewport.
    void fit_image();

    /// Told about a command that failed (title, message). Default: a warning box.
    using ErrorHandler = std::function<void(const QString& title, const QString& message)>;
    void set_error_handler(ErrorHandler handler);

protected:
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void contextMenuEvent(QContextMenuEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    /// Decode the camera's frame into the scene. Returns false when the camera does not exist.
    bool load_image();
    void clear_marker_items();
    void show_message(const QString& text);
    /// Match the surround to the theme.
    void update_background();
    void report(const QString& title, const QString& message);
    /// The flag under a viewport position, or nullptr.
    MarkerFlagItem* flag_at(const QPoint& view_pos) const;
    /// Clamp a pixel to the image rectangle (no-op without an image).
    QPointF clamp_to_image(const QPointF& pixel) const;
    /// The active chunk id, or -1.
    std::int32_t active_chunk_id() const;
    /// The lowest free default marker label ("point N") in `chunk_id`.
    std::string free_marker_label(std::int32_t chunk_id) const;
    /// True when `marker_id` has no other projection than this camera's.
    bool is_last_projection(std::int32_t chunk_id, std::int32_t marker_id) const;
    /// Markers of the active chunk with no projection in this camera, for the Place submenu.
    std::vector<std::pair<std::int32_t, QString>> unplaced_markers() const;

    GuiApp& app_;
    QGraphicsScene* scene_ = nullptr;
    QGraphicsPixmapItem* image_item_ = nullptr;
    QGraphicsSimpleTextItem* message_item_ = nullptr;
    std::int32_t camera_id_ = -1;
    QSizeF image_size_;
    bool fit_mode_ = true;
    bool stale_ = false;
    ErrorHandler error_handler_;

    /// Drag state: the flag being moved and where it started.
    MarkerFlagItem* drag_item_ = nullptr;
    QPointF drag_origin_;
    bool drag_moved_ = false;
    /// Pan state: the last viewport position while the hand drag is active.
    bool panning_ = false;
    QPoint pan_last_;
};

}  // namespace alphashape::gui
