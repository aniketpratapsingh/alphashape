#include "apps/gui/workspace_model.hpp"

#include <algorithm>
#include <cstddef>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <QBrush>
#include <QFont>
#include <QGuiApplication>
#include <QPalette>
#include <QString>

#include "apps/gui/app.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"

namespace alphashape::gui {

namespace detail {

struct TreeNode {
    NodeRef ref;
    QString text;
    QString tooltip;
    bool dim = false;   ///< greyed: an empty group or a disabled camera
    bool bold = false;  ///< the active chunk / the active asset
    TreeNode* parent = nullptr;
    int row = 0;
    std::vector<std::unique_ptr<TreeNode>> children;
};

}  // namespace detail

bool is_group_kind(NodeKind kind) {
    switch (kind) {
        case NodeKind::Cameras:
        case NodeKind::Sensors:
        case NodeKind::TiePoints:
        case NodeKind::DepthMaps:
        case NodeKind::PointClouds:
        case NodeKind::Models:
        case NodeKind::TiledModels:
        case NodeKind::Elevations:
        case NodeKind::Orthomosaics:
        case NodeKind::Markers:
        case NodeKind::ScaleBars:
        case NodeKind::Shapes:
            return true;
        default:
            return false;
    }
}

namespace {

using Node = detail::TreeNode;
using NodePtr = std::unique_ptr<Node>;

QString qs(const std::string& text) { return QString::fromStdString(text); }

QString count(std::size_t n) { return QString::number(static_cast<qulonglong>(n)); }

NodePtr make_node(NodeKind kind, std::int32_t chunk_id, std::int32_t item_id, QString text) {
    auto node = std::make_unique<Node>();
    node->ref = NodeRef{kind, chunk_id, item_id};
    node->text = std::move(text);
    return node;
}

Node* adopt(Node& parent, NodePtr child) {
    child->parent = &parent;
    child->row = static_cast<int>(parent.children.size());
    parent.children.push_back(std::move(child));
    return parent.children.back().get();
}

void renumber(Node& parent, std::size_t from) {
    for (std::size_t i = from; i < parent.children.size(); ++i) {
        parent.children[i]->row = static_cast<int>(i);
    }
}

bool is_aligned(const Camera& camera) { return camera.pose != Transform::Identity(); }

/// A group node whose text is "<name> (<summary>)", or "<name> (none)" greyed when empty.
Node* add_group(Node& chunk_node, NodeKind kind, std::int32_t chunk_id, const QString& name,
                const QString& summary, bool empty) {
    NodePtr node = make_node(kind, chunk_id, -1,
                             name + QStringLiteral(" (") +
                                 (empty ? QObject::tr("none") : summary) + QLatin1Char(')'));
    node->dim = empty;
    return adopt(chunk_node, std::move(node));
}

template <class Slot, class Summary>
void add_asset_group(Node& chunk_node, std::int32_t chunk_id, NodeKind group_kind,
                     NodeKind item_kind, const QString& name, const std::vector<Slot>& slots,
                     std::int32_t active_id, Summary summary) {
    const Slot* active = find_asset(slots, active_id);
    if (active == nullptr && !slots.empty()) {
        active = &slots.front();
    }
    Node* group = add_group(chunk_node, group_kind, chunk_id, name,
                            active != nullptr ? summary(*active) : QString(), slots.empty());
    for (const Slot& slot : slots) {
        NodePtr item = make_node(item_kind, chunk_id, slot.id,
                                 qs(slot.label) + QStringLiteral(" (") + summary(slot) +
                                     QLatin1Char(')'));
        item->bold = (&slot == active);
        item->dim = !slot.enabled;
        adopt(*group, std::move(item));
    }
}

QString raster_size(int width, int height) {
    return QStringLiteral("%1x%2").arg(width).arg(height);
}

void build_cameras(Node& chunk_node, const Chunk& chunk) {
    const std::int32_t chunk_id = chunk.id();
    const auto& cameras = chunk.cameras();
    std::size_t aligned = 0;
    for (const Camera& camera : cameras) {
        aligned += is_aligned(camera) ? 1U : 0U;
    }
    Node* group = add_group(chunk_node, NodeKind::Cameras, chunk_id, QObject::tr("Cameras"),
                            count(aligned) + QLatin1Char('/') + count(cameras.size()),
                            cameras.empty());

    auto camera_node = [&](const Camera& camera) {
        NodePtr node = make_node(NodeKind::Camera, chunk_id, camera.id, qs(camera.label));
        node->dim = !camera.enabled;
        node->tooltip = QStringLiteral("%1, %2")
                            .arg(is_aligned(camera) ? QObject::tr("aligned") : QObject::tr("not aligned"),
                                 camera.enabled ? QObject::tr("enabled") : QObject::tr("disabled"));
        return node;
    };

    std::unordered_map<std::int32_t, const Camera*> by_id;
    by_id.reserve(cameras.size());
    for (const Camera& camera : cameras) {
        by_id.emplace(camera.id, &camera);
    }
    std::unordered_set<std::int32_t> grouped;
    for (const CameraGroup& cg : chunk.camera_groups()) {
        NodePtr gnode = make_node(NodeKind::CameraGroup, chunk_id, cg.id, QString());
        Node* raw = adopt(*group, std::move(gnode));
        for (std::int32_t member : cg.members) {
            const auto it = by_id.find(member);
            if (it == by_id.end() || !grouped.insert(member).second) {
                continue;
            }
            adopt(*raw, camera_node(*it->second));
        }
        raw->text = qs(cg.label) + QStringLiteral(" (") + count(raw->children.size()) +
                    QLatin1Char(')');
    }
    for (const Camera& camera : cameras) {
        if (grouped.find(camera.id) == grouped.end()) {
            adopt(*group, camera_node(camera));
        }
    }
}

NodePtr build_chunk(const Document& document, const Chunk& chunk) {
    const std::int32_t id = chunk.id();
    NodePtr chunk_node = make_node(NodeKind::Chunk, id, -1, qs(chunk.label()));
    chunk_node->bold = (id == document.active_chunk_id());
    Node& cn = *chunk_node;

    build_cameras(cn, chunk);

    const auto& sensors = chunk.sensors();
    Node* sensor_group = add_group(cn, NodeKind::Sensors, id, QObject::tr("Sensors"),
                                   count(sensors.size()), sensors.empty());
    for (std::size_t i = 0; i < sensors.size(); ++i) {
        const QString label = sensors[i].label.empty()
                                  ? QObject::tr("Sensor %1").arg(i + 1)
                                  : qs(sensors[i].label);
        adopt(*sensor_group,
              make_node(NodeKind::Sensor, id, static_cast<std::int32_t>(i), label));
    }

    const TiePoints& tie = chunk.tie_points();
    add_group(cn, NodeKind::TiePoints, id, QObject::tr("Tie points"), count(tie.tracks.size()),
              tie.tracks.empty() && tie.total_keypoints() == 0);

    add_asset_group(cn, id, NodeKind::DepthMaps, NodeKind::DepthMapSet,
                    QObject::tr("Depth maps"), chunk.depth_map_sets(),
                    chunk.active_depth_maps_id(), [](const DepthMapsAsset& slot) {
                        return QObject::tr("%1 maps").arg(slot.data ? slot.data->size() : std::size_t{0});
                    });
    add_asset_group(cn, id, NodeKind::PointClouds, NodeKind::PointCloud,
                    QObject::tr("Point cloud"), chunk.point_clouds(),
                    chunk.active_point_cloud_id(), [](const PointCloudAsset& slot) {
                        return QObject::tr("%1 points").arg(slot.data ? slot.data->points.size() : std::size_t{0});
                    });
    add_asset_group(cn, id, NodeKind::Models, NodeKind::Model, QObject::tr("Model"),
                    chunk.models(), chunk.active_model_id(), [](const ModelAsset& slot) {
                        return QObject::tr("%1 faces").arg(slot.data ? slot.data->faces.size() : std::size_t{0});
                    });
    add_asset_group(cn, id, NodeKind::TiledModels, NodeKind::TiledModel,
                    QObject::tr("Tiled model"), chunk.tiled_models(),
                    chunk.active_tiled_model_id(), [](const TiledModelAsset& slot) {
                        return QObject::tr("%1 KiB").arg(slot.data ? slot.data->payload.size() / 1024 : std::size_t{0});
                    });
    add_asset_group(cn, id, NodeKind::Elevations, NodeKind::Elevation, QObject::tr("DEM"),
                    chunk.elevations(), chunk.active_elevation_id(),
                    [](const ElevationAsset& slot) {
                        return slot.data ? raster_size(slot.data->raster.grid.width,
                                                       slot.data->raster.grid.height)
                                         : QString();
                    });
    add_asset_group(cn, id, NodeKind::Orthomosaics, NodeKind::Orthomosaic,
                    QObject::tr("Orthomosaic"), chunk.orthomosaics(),
                    chunk.active_orthomosaic_id(), [](const OrthomosaicAsset& slot) {
                        return slot.data ? raster_size(slot.data->raster.grid.width,
                                                       slot.data->raster.grid.height)
                                         : QString();
                    });

    Node* markers = add_group(cn, NodeKind::Markers, id, QObject::tr("Markers"),
                              count(chunk.markers().size()), chunk.markers().empty());
    for (const Marker& marker : chunk.markers()) {
        adopt(*markers, make_node(NodeKind::Marker, id, marker.id, qs(marker.label)));
    }
    Node* bars = add_group(cn, NodeKind::ScaleBars, id, QObject::tr("Scale bars"),
                           count(chunk.scalebars().size()), chunk.scalebars().empty());
    for (const Scalebar& bar : chunk.scalebars()) {
        adopt(*bars, make_node(NodeKind::ScaleBar, id, bar.id, qs(bar.label)));
    }
    add_group(cn, NodeKind::Shapes, id, QObject::tr("Shapes"),
              count(chunk.shapes().features.size()), chunk.shapes().features.empty());
    return chunk_node;
}

NodePtr build_root(GuiApp& app) {
    const Document& document = app.document();
    auto root = std::make_unique<Node>();
    Node* doc_node = adopt(*root, make_node(NodeKind::Document, -1, -1, app.document_title()));
    for (const auto& workspace : document.workspaces()) {
        for (const auto& chunk : workspace->chunks()) {
            adopt(*doc_node, build_chunk(document, *chunk));
        }
    }
    return root;
}

using Key = std::tuple<int, std::int32_t, std::int32_t>;
Key key_of(const NodeRef& ref) { return {static_cast<int>(ref.kind), ref.chunk_id, ref.item_id}; }

}  // namespace

// --- WorkspaceModel ----------------------------------------------------------------------

WorkspaceModel::WorkspaceModel(GuiApp& app, QObject* parent)
    : QAbstractItemModel(parent), app_(app), root_(build_root(app)) {
    connect(&app_, &GuiApp::document_replaced, this, &WorkspaceModel::reset);
    connect(&app_, &GuiApp::undo_changed, this, &WorkspaceModel::refresh);
    connect(&app_, &GuiApp::path_changed, this, &WorkspaceModel::refresh);
}

WorkspaceModel::~WorkspaceModel() = default;

void WorkspaceModel::reset() {
    beginResetModel();
    root_ = build_root(app_);
    endResetModel();
}

QModelIndex WorkspaceModel::index_of(const Node* node) const {
    if (node == nullptr || node == root_.get()) {
        return {};
    }
    return createIndex(node->row, 0, const_cast<Node*>(node));
}

QModelIndex WorkspaceModel::index(int row, int column, const QModelIndex& parent) const {
    if (column != 0 || row < 0) {
        return {};
    }
    const Node* node = parent.isValid() ? static_cast<const Node*>(parent.internalPointer())
                                        : root_.get();
    if (static_cast<std::size_t>(row) >= node->children.size()) {
        return {};
    }
    return createIndex(row, 0, node->children[static_cast<std::size_t>(row)].get());
}

QModelIndex WorkspaceModel::parent(const QModelIndex& child) const {
    if (!child.isValid()) {
        return {};
    }
    const Node* node = static_cast<const Node*>(child.internalPointer());
    return index_of(node->parent);
}

int WorkspaceModel::rowCount(const QModelIndex& parent) const {
    if (parent.column() > 0) {
        return 0;
    }
    const Node* node = parent.isValid() ? static_cast<const Node*>(parent.internalPointer())
                                        : root_.get();
    return static_cast<int>(node->children.size());
}

int WorkspaceModel::columnCount(const QModelIndex&) const { return 1; }

Qt::ItemFlags WorkspaceModel::flags(const QModelIndex& index) const {
    if (!index.isValid()) {
        return Qt::NoItemFlags;
    }
    return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

QVariant WorkspaceModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || index.model() != this) {
        return {};
    }
    const Node* node = static_cast<const Node*>(index.internalPointer());
    switch (role) {
        case Qt::DisplayRole:
            return node->text;
        case Qt::ToolTipRole:
            return node->tooltip.isEmpty() ? QVariant() : QVariant(node->tooltip);
        case Qt::FontRole:
            if (node->bold) {
                QFont font;
                font.setBold(true);
                return font;
            }
            return {};
        case Qt::ForegroundRole:
            if (node->dim) {
                return QBrush(QGuiApplication::palette().color(QPalette::Disabled, QPalette::Text));
            }
            return {};
        case KindRole:
            return static_cast<int>(node->ref.kind);
        case ChunkIdRole:
            return node->ref.chunk_id;
        case ItemIdRole:
            return node->ref.item_id;
        default:
            return {};
    }
}

NodeRef WorkspaceModel::ref_at(const QModelIndex& index) const {
    if (!index.isValid() || index.model() != this) {
        return {};
    }
    return static_cast<const Node*>(index.internalPointer())->ref;
}

QModelIndex WorkspaceModel::index_for(const NodeRef& ref) const {
    std::vector<const Node*> stack{root_.get()};
    while (!stack.empty()) {
        const Node* node = stack.back();
        stack.pop_back();
        if (node != root_.get() && node->ref == ref) {
            return index_of(node);
        }
        // Narrow to the right chunk quickly: a chunk-level node never holds another chunk's items.
        for (const auto& child : node->children) {
            if (child->ref.kind == NodeKind::Chunk && ref.kind != NodeKind::Document &&
                child->ref.chunk_id != ref.chunk_id) {
                continue;
            }
            stack.push_back(child.get());
        }
    }
    return {};
}

void WorkspaceModel::refresh() {
    NodePtr fresh = build_root(app_);
    merge_children(*root_, fresh->children);
}

void WorkspaceModel::update_node(Node& old_node, Node& fresh) {
    if (old_node.text != fresh.text || old_node.tooltip != fresh.tooltip ||
        old_node.dim != fresh.dim || old_node.bold != fresh.bold) {
        old_node.text = fresh.text;
        old_node.tooltip = fresh.tooltip;
        old_node.dim = fresh.dim;
        old_node.bold = fresh.bold;
        const QModelIndex idx = index_of(&old_node);
        Q_EMIT dataChanged(idx, idx);
    }
    merge_children(old_node, fresh.children);
}

void WorkspaceModel::merge_children(Node& parent, std::vector<NodePtr>& fresh) {
    auto& old = parent.children;
    const QModelIndex parent_index = index_of(&parent);

    if (old.empty() && fresh.empty()) {
        return;
    }
    if (old.empty()) {
        beginInsertRows(parent_index, 0, static_cast<int>(fresh.size()) - 1);
        for (NodePtr& node : fresh) {
            adopt(parent, std::move(node));
        }
        endInsertRows();
        return;
    }
    if (fresh.empty()) {
        beginRemoveRows(parent_index, 0, static_cast<int>(old.size()) - 1);
        old.clear();
        endRemoveRows();
        return;
    }

    bool same = old.size() == fresh.size();
    for (std::size_t i = 0; same && i < old.size(); ++i) {
        same = old[i]->ref == fresh[i]->ref;
    }
    if (same) {
        for (std::size_t i = 0; i < old.size(); ++i) {
            update_node(*old[i], *fresh[i]);
        }
        return;
    }

    // 1. Drop nodes that are gone.
    std::set<Key> wanted;
    for (const NodePtr& node : fresh) {
        wanted.insert(key_of(node->ref));
    }
    for (std::size_t i = old.size(); i-- > 0;) {
        if (wanted.count(key_of(old[i]->ref)) == 0) {
            beginRemoveRows(parent_index, static_cast<int>(i), static_cast<int>(i));
            old.erase(old.begin() + static_cast<std::ptrdiff_t>(i));
            renumber(parent, i);
            endRemoveRows();
        }
    }
    // 2. Walk the new order, inserting what is new and re-positioning what moved.
    for (std::size_t i = 0; i < fresh.size(); ++i) {
        if (i < old.size() && old[i]->ref == fresh[i]->ref) {
            update_node(*old[i], *fresh[i]);
            continue;
        }
        for (std::size_t j = i + 1; j < old.size(); ++j) {
            if (old[j]->ref == fresh[i]->ref) {
                beginRemoveRows(parent_index, static_cast<int>(j), static_cast<int>(j));
                old.erase(old.begin() + static_cast<std::ptrdiff_t>(j));
                renumber(parent, j);
                endRemoveRows();
                break;
            }
        }
        beginInsertRows(parent_index, static_cast<int>(i), static_cast<int>(i));
        fresh[i]->parent = &parent;
        old.insert(old.begin() + static_cast<std::ptrdiff_t>(i), std::move(fresh[i]));
        renumber(parent, i);
        endInsertRows();
    }
    // 3. Anything left over can only come from duplicate ids in a malformed document.
    if (old.size() > fresh.size()) {
        beginRemoveRows(parent_index, static_cast<int>(fresh.size()),
                        static_cast<int>(old.size()) - 1);
        old.resize(fresh.size());
        endRemoveRows();
    }
}

// --- WorkspaceSelection ------------------------------------------------------------------

WorkspaceSelection::WorkspaceSelection(QObject* parent) : QObject(parent) {}

void WorkspaceSelection::set(const QList<NodeRef>& items, const NodeRef& current) {
    const bool has = !items.isEmpty();
    if (items == items_ && has == has_current_ && (!has || current == current_)) {
        return;
    }
    items_ = items;
    has_current_ = has;
    current_ = has ? current : NodeRef{};
    Q_EMIT changed();
}

void WorkspaceSelection::set(const QList<NodeRef>& items) {
    set(items, items.isEmpty() ? NodeRef{} : items.front());
}

void WorkspaceSelection::clear() { set({}, {}); }

}  // namespace alphashape::gui
