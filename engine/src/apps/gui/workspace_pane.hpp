#pragma once

// WorkspacePane: the tree in the Workspace dock. It publishes the selection through a
// WorkspaceSelection and offers a context menu whose entries are all engine commands, so every
// edit goes through GuiApp::execute() and is undoable.
//
// Not offered, because the engine has no command for them: renaming a chunk and removing
// cameras. Making a chunk active (double-click) sets the document's active id directly and
// marks the project dirty; it is not on the undo stack.

#include <functional>
#include <cstdint>
#include <memory>
#include <vector>

#include <QList>
#include <QString>
#include <QWidget>

#include "apps/gui/workspace_model.hpp"

class QMenu;
class QTreeView;

namespace alphashape::gui {

class GuiApp;

class WorkspacePane : public QWidget {
    Q_OBJECT

public:
    WorkspacePane(GuiApp& app, WorkspaceSelection& selection, QWidget* parent = nullptr);

    WorkspaceModel& model() { return *model_; }
    QTreeView* tree() { return tree_; }

    /// Told about a command that failed (title, message). Default: a warning box.
    using ErrorHandler = std::function<void(const QString& title, const QString& message)>;
    void set_error_handler(ErrorHandler handler);

    /// Select exactly this node (and make it current). False when it is not in the tree.
    bool select(const NodeRef& ref);
    /// Extend the selection with one more node.
    bool add_to_selection(const NodeRef& ref);

    // --- The operations behind the context menu. Each returns false if refused or failed. ---
    bool add_chunk();
    bool remove_chunk(std::int32_t chunk_id);
    bool set_cameras_enabled(std::int32_t chunk_id, const std::vector<std::int32_t>& camera_ids,
                             bool enabled);
    /// Make a chunk the active one. Not undoable (see the file comment).
    bool activate_chunk(std::int32_t chunk_id);

    /// The context menu for the current selection (or for empty space when it is empty).
    /// Entries are named "action.workspace.<verb>"; disabled ones stay listed.
    std::unique_ptr<QMenu> build_context_menu();

private:
    void on_tree_selection_changed();
    void on_context_menu_requested(const QPoint& pos);
    void on_activated(const QModelIndex& index);
    /// Cameras covered by the selection (camera nodes, camera groups, the Cameras group), all
    /// from one chunk. Empty when the selection spans chunks or holds no cameras.
    std::vector<std::int32_t> selected_cameras(std::int32_t& chunk_id) const;
    void report(const QString& title, const QString& message);

    GuiApp& app_;
    WorkspaceSelection& selection_;
    WorkspaceModel* model_ = nullptr;
    QTreeView* tree_ = nullptr;
    ErrorHandler error_handler_;
};

}  // namespace alphashape::gui
