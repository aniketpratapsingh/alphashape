#pragma once

// PhotosPane: the Photos dock. An icon list of the active chunk's cameras over a PhotosModel,
// with a filter box, a context menu and Delete to enable/disable the selected photos.
//
// Every edit goes through the command layer, so it is undoable: enabling or disabling several
// photos is one macro entry named after the action ("Disable 3 photos"), never one entry per
// photo.
//
// The list selection is mirrored into the WorkspaceSelection the Workspace tree and the
// Properties pane already share, so selecting a photo updates the property sheet. Selecting a
// camera in the tree selects its row here in turn.
//
// Deferred (not in this work package): persisting thumbnails into the project
// (thumbnails.zip), rotating photos, masks and the depth / normal display modes.
//
// Jobs: while a job is active the document belongs to the worker. The pane never reads it,
// every mutating action is refused, and the model rebuilds when the job finishes.

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <QWidget>

#include "apps/gui/photos_model.hpp"

class QAction;
class QLabel;
class QLineEdit;
class QListView;
class QMenu;

namespace alphashape::gui {

class GuiApp;
class WorkspaceSelection;

class PhotosPane : public QWidget {
    Q_OBJECT

public:
    PhotosPane(GuiApp& app, WorkspaceSelection& selection, QWidget* parent = nullptr);

    PhotosModel& model() { return *model_; }
    QListView* view() { return view_; }
    QLineEdit* filter_edit() { return filter_; }

    /// Told about a command that failed (title, message). Default: a warning box.
    using ErrorHandler = std::function<void(const QString& title, const QString& message)>;
    void set_error_handler(ErrorHandler handler);

    /// Set the filter programmatically (the filter box follows).
    void set_filter(const QString& text);

    /// Camera ids selected in the list, in row order.
    std::vector<std::int32_t> selected_camera_ids() const;

    /// Select one camera (clearing the rest). False when the filter hides it.
    bool select_camera(std::int32_t camera_id);

    /// Enable or disable `camera_ids` as one undo entry. Returns false if refused or failed
    /// (nothing to do, a job running, or the command refused).
    bool set_cameras_enabled(const std::vector<std::int32_t>& camera_ids, bool enabled);
    /// The Delete action: disable the selected photos.
    bool disable_selected();
    bool enable_selected();

    /// The context menu for the current selection. Entries are named "action.photos.<verb>";
    /// disabled ones stay listed.
    std::unique_ptr<QMenu> build_context_menu();

private:
    void on_view_selection_changed();
    void on_shared_selection_changed();
    void on_context_menu_requested(const QPoint& pos);
    void report(const QString& title, const QString& message);
    /// Reports and returns true when a job is active (the document must not be touched).
    bool refuse_while_busy(const QString& title);
    void update_actions();
    void update_summary();

    GuiApp& app_;
    WorkspaceSelection& selection_;
    PhotosModel* model_ = nullptr;
    QListView* view_ = nullptr;
    QLineEdit* filter_ = nullptr;
    QLabel* summary_ = nullptr;
    QAction* enable_action_ = nullptr;
    QAction* disable_action_ = nullptr;
    ErrorHandler error_handler_;
    bool syncing_ = false;
};

}  // namespace alphashape::gui
