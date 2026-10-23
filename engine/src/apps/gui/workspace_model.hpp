#pragma once

// WorkspaceModel: the Workspace tree (Document > Chunks > per-chunk groups > items) as a
// QAbstractItemModel, plus WorkspaceSelection, the small shared object through which the tree
// pane tells the Properties pane what is selected.
//
// The model never holds pointers into the engine Document. Every node is identified by a
// NodeRef (kind plus chunk id plus item id), so the document may be replaced or edited
// underneath it and the tree is simply rebuilt: a full reset after the document was swapped,
// a keyed in-place merge after execute/undo/redo (which keeps expansion and selection).

#include <cstdint>
#include <memory>

#include <QAbstractItemModel>
#include <QList>
#include <QObject>

namespace alphashape::gui {

class GuiApp;

namespace detail {
struct TreeNode;
}

enum class NodeKind {
    Document,
    Chunk,
    Cameras,        ///< the "Cameras (aligned/total)" group
    CameraGroup,    ///< a camera group inside Cameras; item_id = group id
    Camera,         ///< item_id = camera id
    Sensors,
    Sensor,         ///< item_id = index into Chunk::sensors()
    TiePoints,
    DepthMaps,      ///< group node
    DepthMapSet,    ///< item_id = asset id
    PointClouds,
    PointCloud,
    Models,
    Model,
    TiledModels,
    TiledModel,
    Elevations,
    Elevation,
    Orthomosaics,
    Orthomosaic,
    Markers,
    Marker,         ///< item_id = marker id
    ScaleBars,
    ScaleBar,       ///< item_id = scalebar id
    Shapes,
};

/// Stable identity of a tree node across rebuilds.
struct NodeRef {
    NodeKind kind = NodeKind::Document;
    std::int32_t chunk_id = -1;
    std::int32_t item_id = -1;

    bool operator==(const NodeRef&) const = default;
};

/// True for the group nodes that only hold other nodes (Cameras, Sensors, Point clouds, ...).
bool is_group_kind(NodeKind kind);

class WorkspaceModel : public QAbstractItemModel {
    Q_OBJECT

public:
    enum Role {
        KindRole = Qt::UserRole + 1,  ///< int(NodeKind)
        ChunkIdRole,                  ///< chunk id, -1 above chunk level
        ItemIdRole,                   ///< see NodeKind
    };

    explicit WorkspaceModel(GuiApp& app, QObject* parent = nullptr);
    ~WorkspaceModel() override;

    // QAbstractItemModel
    QModelIndex index(int row, int column, const QModelIndex& parent = {}) const override;
    QModelIndex parent(const QModelIndex& child) const override;
    int rowCount(const QModelIndex& parent = {}) const override;
    int columnCount(const QModelIndex& parent = {}) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    Qt::ItemFlags flags(const QModelIndex& index) const override;

    NodeRef ref_at(const QModelIndex& index) const;
    /// Invalid index when no such node is in the tree.
    QModelIndex index_for(const NodeRef& ref) const;

    /// Re-read the document and update the tree in place. Called on execute/undo/redo; call it
    /// after changing the document by other means.
    /// While a job is active the document must not be read: refresh() then only marks the tree
    /// stale() and the rebuild happens when the job finishes.
    void refresh();
    bool stale() const { return stale_; }

Q_SIGNALS:
    /// The tree was fully rebuilt because a job finished (the previous selection is gone).
    void reset_after_job();

private:
    void refresh_after_job();
    std::unique_ptr<detail::TreeNode> build_fresh();
    bool stale_ = false;
    using Node = detail::TreeNode;

    void reset();
    void merge_children(Node& parent, std::vector<std::unique_ptr<Node>>& fresh);
    void update_node(Node& old_node, Node& fresh);
    QModelIndex index_of(const Node* node) const;

    GuiApp& app_;
    std::unique_ptr<Node> root_;
};

/// What is selected in the Workspace tree. Items are in tree order; current() is the focus
/// item (the one the Properties pane describes when several are selected).
class WorkspaceSelection : public QObject {
    Q_OBJECT

public:
    explicit WorkspaceSelection(QObject* parent = nullptr);

    const QList<NodeRef>& items() const { return items_; }
    bool has_current() const { return has_current_; }
    NodeRef current() const { return current_; }

    /// Replace the selection. Emits changed() only when something differs.
    void set(const QList<NodeRef>& items, const NodeRef& current);
    void set(const QList<NodeRef>& items);
    void clear();

Q_SIGNALS:
    void changed();

private:
    QList<NodeRef> items_;
    NodeRef current_;
    bool has_current_ = false;
};

}  // namespace alphashape::gui
