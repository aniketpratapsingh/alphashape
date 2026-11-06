#pragma once

// ReferencePane: the Reference dock. Two tables (cameras, markers) over the
// active chunk's reference table, a toolbar and a summary line.
//
// Every cell edit is one undoable command (see ReferenceModel). The toolbar
// opens the existing task dialogs for OptimizeCameras, ImportReference,
// ExportReference and ConvertReference; the reference-settings dialog is a
// later work package, so its action is a disabled placeholder, as is picking a
// reference in the viewport.
//
// The pane never reads the document while a job owns it: the models mark
// themselves stale and rebuild when the job finishes.

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

class ReferencePane : public QWidget {
    Q_OBJECT

public:
    explicit ReferencePane(GuiApp& app, QWidget* parent = nullptr);

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

private:
    void create_toolbar();
    void create_tables();
    void run_task(const std::string& task);
    void update_status();
    void update_actions();

    GuiApp& app_;
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
};

}  // namespace alphashape::gui
