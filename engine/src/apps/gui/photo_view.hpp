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
// Decoding a frame is expensive, so the display pixmap of the last few cameras is kept in a
// small cache keyed by everything that decides what the picture looks like (camera, chunk,
// frame path, file modification time, recorded size, and the film transform a scanned frame is
// resampled through). Re-opening a photo, or refreshing its markers after an edit, therefore
// does not decode again; a job that changed the frame (a calibration, a film transform) does,
// because its key no longer matches.
//
// Editing goes through the command layer, so every placement, move and removal is one undo
// entry. While a job owns the document the view refuses edits and rebuilds when the manager
// goes idle.
//
// The public add/place/move/remove methods are what the context menu and the drag handler call,
// so a test can exercise the whole edit path without synthesising mouse events.

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <QGraphicsView>
#include <QPoint>
#include <QPointF>
#include <QPixmap>
#include <QSizeF>
#include <QString>

class QContextMenuEvent;
class QGraphicsPixmapItem;
class QGraphicsScene;
class QGraphicsSimpleTextItem;
class QMouseEvent;
class QResizeEvent;
class QWheelEvent;

namespace alphashape {
class Camera;
class Chunk;
class Document;
}  // namespace alphashape

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
    /// The scene position (a full-resolution pixel) of every marker flag, keyed by marker id.
    /// A test hook: it lets a test prove a flag sits on the pixel the document stores.
    std::vector<std::pair<std::int32_t, QPointF>> marker_flag_positions() const;
    /// True when a frame is decoded and shown.
    bool has_image() const;
    /// How many times a frame has been decoded from disk (a cache miss). Diagnostic, used by
    /// the tests to prove a re-open or a marker refresh does not decode again.
    int image_loads() const { return image_loads_; }

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

    /// Rebuild the flags from the document. A no-op while a job runs (the view rebuilds when
    /// the job finishes).
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
    /// Everything that decides what a camera's display pixmap looks like. Two frames with the
    /// same key decode to the same picture, so the cache may serve the second from the first.
    struct ImageKey {
        std::int32_t camera_id = -1;
        std::int32_t chunk_id = -1;
        QString path;
        qint64 mtime = 0;
        int width = 0;
        int height = 0;
        /// The film geometry a scanned frame is resampled through. A calibration rewrites it,
        /// so the key changes and the frame reloads.
        bool film_enabled = false;
        double film_resolution = 0.0;
        double film_size_x = 0.0;
        double film_size_y = 0.0;
        /// The per-scan film transform, flattened; empty for a digital camera.
        std::array<double, 9> film_transform{};
        bool has_film_transform = false;

        bool operator==(const ImageKey&) const = default;
    };

    /// One cached display pixmap and the full-resolution size it stands for.
    struct CachedImage {
        ImageKey key;
        QPixmap pixmap;
        QSizeF full_size;
    };

    /// Decode the camera's frame into the scene. Returns false when the camera does not exist.
    /// Serves the display pixmap from the cache when the key matches.
    bool load_image();
    /// The key for `camera` in `chunk`, or nullopt when the camera has no linked frame.
    static std::optional<ImageKey> image_key(const Document& document, const Chunk& chunk,
                                             const Camera& camera);
    /// The cached image for `key`, or nullptr. Moves the hit to the front of the LRU order.
    const CachedImage* cached_image(const ImageKey& key);
    /// Store `image` under `key`, evicting the least recently used beyond the cap.
    void store_image(ImageKey key, QPixmap pixmap, QSizeF full_size);
    /// The view scale fitInView() would choose for the current viewport, or 0 when unknown.
    double fit_scale() const;
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
    ErrorHandler error_handler_;

    /// Display pixmaps of the last few cameras, most recently used first.
    std::vector<CachedImage> image_cache_;
    /// Frames decoded from disk (a cache miss). Diagnostic for the tests.
    int image_loads_ = 0;

    /// Drag state: the flag being moved and where it started.
    MarkerFlagItem* drag_item_ = nullptr;
    QPointF drag_origin_;
    bool drag_moved_ = false;
    /// Pan state: the last viewport position while the hand drag is active.
    bool panning_ = false;
    QPoint pan_last_;
};

}  // namespace alphashape::gui
