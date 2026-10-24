#pragma once

// ViewportWidget: the 3-D view in the centre of the main window.
//
// Two layers, deliberately separable:
//   * CameraController (viewer/camera_controller.hpp) - the orbit camera, pure math, unit-tested.
//   * ViewportRenderer (viewer/viewport_renderer.hpp) - Scene -> QRhi draw calls.
// ViewportWidget owns both, plus the Scene rebuild rule and the input handling.
//
// THE CANVAS IS OPTIONAL. The canvas is a QRhiWidget, which needs a platform that offers
// RHI-based rendering. The offscreen QPA plugin (CI, the GUI tests, --offscreen-smoke) reports
// that it does not, and warns if a QRhiWidget is created at all, so the canvas is only created
// when the platform can provide one. Without it the widget paints the theme's gradient plus a
// plain message - which is also what it shows when the canvas reports renderFailed(). No code
// path here can crash because there is no GPU.
//
// REFRESH RULE (same as the panes): the document is never read while a job runs. Rebuilds are
// requested on job_finished, document_replaced and undo_changed; a request during a job only
// marks the scene stale, and the rebuild happens when the manager goes idle.
//
// Rendering is on demand: no timer runs. update() is called only when the scene, the camera, the
// view options or the theme changed.

#include <cstdint>
#include <optional>

#include <QPoint>
#include <QString>
#include <QWidget>

#include "apps/gui/theme/theme_mode.hpp"
#include "apps/gui/viewer/camera_controller.hpp"
#include "apps/gui/viewer/scene.hpp"
#include "apps/gui/viewer/viewport_renderer.hpp"

QT_BEGIN_NAMESPACE
class QKeyEvent;
class QLabel;
class QPaintEvent;
class QResizeEvent;
QT_END_NAMESPACE

namespace alphashape::gui {

class GuiApp;
class Theme;
class WorkspaceSelection;

class ViewportWidget : public QWidget {
    Q_OBJECT

public:
    /// `app` supplies the document and the job manager; `theme` the canvas colours.
    ViewportWidget(GuiApp& app, Theme& theme, QWidget* parent = nullptr);
    ~ViewportWidget() override;

    /// The theme mode the canvas is drawn in right now ("View > Viewport background").
    ThemeMode canvas_mode() const { return mode_; }

    /// True when a QRhiWidget canvas could be created on this platform.
    bool rendering_available() const { return canvas_ != nullptr; }

    CameraController& camera() { return camera_; }
    const CameraController& camera() const { return camera_; }

    /// What the last frame drew (zeros until one was rendered).
    const ViewportRenderer::Stats& stats() const { return renderer_.stats(); }
    /// Why the canvas could not render; empty when it can.
    QString render_error() const { return render_error_; }
    /// Backend the canvas runs on ("Metal", "Vulkan", ... or "none").
    QString backend_name() const { return renderer_.backend_name(); }

    void fit_scene();
    void toggle_projection();
    void set_orthographic(bool on);

    /// Grid and axes toggles (both default on).
    void set_show_grid(bool on);
    void set_show_axes(bool on);
    bool show_grid() const { return show_grid_; }
    bool show_axes() const { return show_axes_; }

    /// The Workspace selection: a selected camera is drawn in the highlight colour.
    void set_selection(WorkspaceSelection* selection);

    /// True when the built scene has neither points nor a mesh (an empty or unaligned chunk).
    bool scene_is_empty() const { return renderer_.scene().empty(); }
    /// The scene the renderer holds. There is exactly one: the widget does not keep a second
    /// copy, so a rebuild moves the fresh scene into the renderer instead of duplicating it.
    const Scene& scene() const { return renderer_.scene(); }

    /// Rebuild the scene from the document now (a no-op while a job runs: it marks the scene
    /// stale instead). Call it when the document changed without a job.
    void refresh_scene();
    bool scene_stale() const { return stale_; }

    /// Request one frame: the canvas renders on its next paint, the placeholder repaints now.
    void request_frame();

Q_SIGNALS:
    /// The projection changed (the View menu keeps its checkable entry in step).
    void projection_changed(bool orthographic);

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

private:
    /// The QRhiWidget canvas. Defined in the .cpp, the only file that needs the RHI headers.
    class Canvas;

    void rebuild_scene();
    /// Update the camera highlight after the Workspace selection changed: a colour-only change,
    /// so it restyles the existing scene instead of rebuilding it.
    void reselect_camera();
    std::int32_t active_chunk_id() const;
    void place_children();
    void update_message();
    void on_render_failed(const QString& reason);
    void on_render_succeeded();
    void on_theme_changed(ThemeMode mode);
    void handle_mouse_press(const QPoint& position, Qt::MouseButton button);
    void handle_mouse_move(const QPoint& position, Qt::MouseButtons buttons);
    void handle_wheel(int angle_delta);
    void handle_key(int key);
    int canvas_height() const;

    GuiApp& app_;
    Theme& theme_;
    ThemeMode mode_;
    CameraController camera_;
    ViewportRenderer renderer_;
    BuildOptions options_;
    /// What the last build highlighted, so a selection change that alters nothing is a no-op.
    std::optional<std::int32_t> last_highlight_;

    Canvas* canvas_ = nullptr;
    QLabel* message_ = nullptr;
    WorkspaceSelection* selection_ = nullptr;
    QString render_error_;

    bool show_grid_ = true;
    bool show_axes_ = true;
    bool stale_ = false;
    /// Frame the scene once, the first time it has something in it.
    bool fit_pending_ = true;
    QPoint last_mouse_;
    bool dragging_ = false;
    Qt::MouseButton drag_button_ = Qt::NoButton;
};

/// The camera id (in the active chunk) the Workspace selection names, if it names one.
/// A camera node belonging to a different chunk yields nullopt: ids are per chunk.
std::optional<std::int32_t> selected_camera(const WorkspaceSelection* selection,
                                            std::int32_t active_chunk_id);

}  // namespace alphashape::gui
