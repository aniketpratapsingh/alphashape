#pragma once

// ReferencePane: the Reference dock. Two tables (cameras, markers) over the
// active chunk's reference table, a toolbar and a summary line.
//
// Every cell edit is one undoable command (see ReferenceModel). The toolbar
// opens the existing task dialogs for OptimizeCameras, ImportReference,
// ExportReference and ConvertReference; the Settings action opens the reference
// settings dialog. "Place on photo" opens the camera the Workspace tree has
// selected (through the shared WorkspaceSelection), falling back to the first
// camera with a frame when nothing is selected.
//
// The pane never reads the document while a job owns it: the models mark
// themselves stale and rebuild when the job finishes.

#include <cstdint>
#include <string>

#include <QString>
#include <QWidget>

#include "apps/gui/reference_model.hpp"

class QAction;
class QLabel;
class QTableView;
class QToolBar;

namespace alphashape::gui {

class GuiApp;
class WorkspaceSelection;

class ReferencePane : public QWidget {
    Q_OBJECT

public:
    /// `selection` is the shared Workspace selection the tree, the Properties sheet and the
    /// Photos list publish into; "Place on photo" reads the selected camera from it.
    ReferencePane(GuiApp& app, WorkspaceSelection& selection, QWidget* parent = nullptr);

    ReferenceModel& cameras() { return *camera_model_; }
    ReferenceModel& markers() { return *marker_model_; }
    QTableView* camera_view() { return camera_view_; }
    QTableView* marker_view() { return marker_view_; }
    QToolBar* toolbar() { return toolbar_; }
    QLabel* status_label() { return status_; }
    /// An action by objectName ("action.reference.optimize", ...), or nullptr.
    QAction* action(const QString& object_name) const;

    /// Rebuild both tables and the status line from the document.
    void refresh();

Q_SIGNALS:
    /// "Place on photo" was triggered: the window opens `camera_id` in the Photo tab, where
    /// markers are placed. The pane never opens the tab itself.
    void place_on_photo_requested(qint32 camera_id);

    /// "Show photos with this marker" was triggered: the window filters the Photos pane to the
    /// cameras that see `marker_id`, labelled `label`. The pane never touches the Photos dock.
    void show_marker_photos_requested(qint32 marker_id, const QString& label);

private:
    void create_toolbar();
    void create_tables();
    void run_task(const std::string& task);
    void update_status();
    void update_actions();
    /// The camera "Place on photo" should open: the selected camera of the active chunk, else
    /// the first camera that has a frame, else -1.
    std::int32_t camera_for_photo() const;
    /// The marker row the "Show photos" action should use: the selected marker row, else -1.
    std::int32_t selected_marker_row() const;

    GuiApp& app_;
    WorkspaceSelection& selection_;
    ReferenceModel* camera_model_ = nullptr;
    ReferenceModel* marker_model_ = nullptr;
    QTableView* camera_view_ = nullptr;
    QTableView* marker_view_ = nullptr;
    QToolBar* toolbar_ = nullptr;
    QLabel* status_ = nullptr;
    QAction* optimize_ = nullptr;
    QAction* import_ = nullptr;
    QAction* export_ = nullptr;
    QAction* convert_ = nullptr;
    QAction* import_markers_ = nullptr;
    QAction* pick_ = nullptr;
    QAction* show_photos_ = nullptr;
};

}  // namespace alphashape::gui
