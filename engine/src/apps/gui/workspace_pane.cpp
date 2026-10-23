#include "apps/gui/workspace_pane.hpp"

#include <algorithm>
#include <exception>
#include <vector>

#include <QAbstractItemView>
#include <QAction>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QMenu>
#include <QMessageBox>
#include <QTreeView>
#include <QVBoxLayout>

#include "apps/gui/app.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/document.hpp"

namespace alphashape::gui {

WorkspacePane::WorkspacePane(GuiApp& app, WorkspaceSelection& selection, QWidget* parent)
    : QWidget(parent), app_(app), selection_(selection) {
    setObjectName(QStringLiteral("pane.workspace"));
    error_handler_ = [this](const QString& title, const QString& message) {
        QMessageBox::warning(this, title, message);
    };

    model_ = new WorkspaceModel(app_, this);
    tree_ = new QTreeView(this);
    tree_->setObjectName(QStringLiteral("tree.workspace"));
    tree_->setModel(model_);
    tree_->setHeaderHidden(true);
    tree_->setSelectionMode(QAbstractItemView::ExtendedSelection);
    tree_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    tree_->setContextMenuPolicy(Qt::CustomContextMenu);
    tree_->setUniformRowHeights(true);
    tree_->setExpandsOnDoubleClick(false);  // double-click activates a chunk instead
    tree_->setAnimated(false);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(tree_);

    // Open the project and its chunks so the first look shows something.
    auto expand_top = [this]() {
        const QModelIndex doc = model_->index(0, 0);
        tree_->expand(doc);
        for (int i = 0; i < model_->rowCount(doc); ++i) {
            tree_->expand(model_->index(i, 0, doc));
        }
    };
    expand_top();
    connect(&app_, &GuiApp::document_replaced, this, expand_top);
    connect(model_, &QAbstractItemModel::rowsInserted, this,
            [this](const QModelIndex& inserted_under, int first, int last) {
                // A chunk that was just added opens by itself; deeper inserts keep user state.
                if (inserted_under.isValid() && model_->ref_at(inserted_under).kind == NodeKind::Document) {
                    for (int i = first; i <= last; ++i) {
                        tree_->expand(model_->index(i, 0, inserted_under));
                    }
                }
            });

    connect(tree_->selectionModel(), &QItemSelectionModel::selectionChanged, this,
            &WorkspacePane::on_tree_selection_changed);
    connect(tree_->selectionModel(), &QItemSelectionModel::currentChanged, this,
            &WorkspacePane::on_tree_selection_changed);
    connect(model_, &QAbstractItemModel::modelReset, this,
            &WorkspacePane::on_tree_selection_changed);
    connect(tree_, &QTreeView::customContextMenuRequested, this,
            &WorkspacePane::on_context_menu_requested);
    connect(tree_, &QTreeView::doubleClicked, this, &WorkspacePane::on_activated);
}

void WorkspacePane::set_error_handler(ErrorHandler handler) {
    error_handler_ = std::move(handler);
}

void WorkspacePane::report(const QString& title, const QString& message) {
    if (error_handler_) {
        error_handler_(title, message);
    }
}

// --- Selection ---------------------------------------------------------------------------

void WorkspacePane::on_tree_selection_changed() {
    QList<NodeRef> items;
    QModelIndexList rows = tree_->selectionModel()->selectedIndexes();
    // Tree order: by path of rows from the root.
    auto path = [](QModelIndex index) {
        std::vector<int> rows_up;
        for (; index.isValid(); index = index.parent()) {
            rows_up.push_back(index.row());
        }
        std::reverse(rows_up.begin(), rows_up.end());
        return rows_up;
    };
    std::sort(rows.begin(), rows.end(), [&](const QModelIndex& a, const QModelIndex& b) {
        return path(a) < path(b);
    });
    for (const QModelIndex& index : rows) {
        items.append(model_->ref_at(index));
    }
    NodeRef current;
    const QModelIndex cur = tree_->selectionModel()->currentIndex();
    if (cur.isValid() && items.contains(model_->ref_at(cur))) {
        current = model_->ref_at(cur);
    } else if (!items.isEmpty()) {
        current = items.front();
    }
    selection_.set(items, current);
}

bool WorkspacePane::select(const NodeRef& ref) {
    const QModelIndex index = model_->index_for(ref);
    if (!index.isValid()) {
        return false;
    }
    tree_->selectionModel()->setCurrentIndex(
        index, QItemSelectionModel::ClearAndSelect | QItemSelectionModel::Current);
    return true;
}

bool WorkspacePane::add_to_selection(const NodeRef& ref) {
    const QModelIndex index = model_->index_for(ref);
    if (!index.isValid()) {
        return false;
    }
    tree_->selectionModel()->select(index, QItemSelectionModel::Select);
    return true;
}

// --- Operations --------------------------------------------------------------------------

bool WorkspacePane::add_chunk() {
    try {
        const std::size_t existing = app_.document().workspace().chunks().size();
        app_.execute(make_add_chunk_command(
            app_.document().workspace(), tr("Chunk %1").arg(existing + 1).toStdString(), -1));
        return true;
    } catch (const std::exception& error) {
        report(tr("Add chunk"), QString::fromUtf8(error.what()));
        return false;
    }
}

bool WorkspacePane::remove_chunk(std::int32_t chunk_id) {
    try {
        app_.execute(make_remove_chunk_command(chunk_id));
        return true;
    } catch (const std::exception& error) {
        report(tr("Remove chunk"), QString::fromUtf8(error.what()));
        return false;
    }
}

bool WorkspacePane::set_cameras_enabled(std::int32_t chunk_id,
                                        const std::vector<std::int32_t>& camera_ids,
                                        bool enabled) {
    if (camera_ids.empty()) {
        return false;
    }
    try {
        app_.execute(make_set_camera_enabled_command(chunk_id, camera_ids, enabled));
        return true;
    } catch (const std::exception& error) {
        report(enabled ? tr("Enable cameras") : tr("Disable cameras"),
               QString::fromUtf8(error.what()));
        return false;
    }
}

bool WorkspacePane::activate_chunk(std::int32_t chunk_id) {
    Document& document = app_.document();
    if (document.workspace().find_chunk(chunk_id) == nullptr) {
        return false;
    }
    if (document.active_chunk_id() == chunk_id) {
        return true;
    }
    document.set_active_chunk_id(chunk_id);
    app_.mark_dirty();
    model_->refresh();
    return true;
}

void WorkspacePane::on_activated(const QModelIndex& index) {
    const NodeRef ref = model_->ref_at(index);
    if (ref.kind == NodeKind::Chunk) {
        activate_chunk(ref.chunk_id);
    } else if (model_->hasChildren(index)) {
        tree_->setExpanded(index, !tree_->isExpanded(index));
    }
}

std::vector<std::int32_t> WorkspacePane::selected_cameras(std::int32_t& chunk_id) const {
    chunk_id = -1;
    std::vector<std::int32_t> ids;
    const Document& document = app_.document();
    for (const NodeRef& ref : selection_.items()) {
        if (ref.kind != NodeKind::Camera && ref.kind != NodeKind::CameraGroup &&
            ref.kind != NodeKind::Cameras) {
            return {};
        }
        if (chunk_id != -1 && chunk_id != ref.chunk_id) {
            return {};
        }
        chunk_id = ref.chunk_id;
        const Chunk* chunk = nullptr;
        for (const auto& workspace : document.workspaces()) {
            for (const auto& candidate : workspace->chunks()) {
                if (candidate->id() == ref.chunk_id) {
                    chunk = candidate.get();
                }
            }
        }
        if (chunk == nullptr) {
            return {};
        }
        if (ref.kind == NodeKind::Camera) {
            ids.push_back(ref.item_id);
        } else if (ref.kind == NodeKind::Cameras) {
            for (const Camera& camera : chunk->cameras()) {
                ids.push_back(camera.id);
            }
        } else {
            for (const CameraGroup& group : chunk->camera_groups()) {
                if (group.id == ref.item_id) {
                    ids.insert(ids.end(), group.members.begin(), group.members.end());
                }
            }
        }
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

// --- Context menu ------------------------------------------------------------------------

std::unique_ptr<QMenu> WorkspacePane::build_context_menu() {
    auto menu = std::make_unique<QMenu>(this);
    menu->setObjectName(QStringLiteral("menu.workspace.context"));
    auto entry = [&](const QString& name, const QString& text, bool enabled, auto&& fn) {
        QAction* action = menu->addAction(text);
        action->setObjectName(QStringLiteral("action.workspace.") + name);
        action->setEnabled(enabled);
        connect(action, &QAction::triggered, this, fn);
        return action;
    };

    const QList<NodeRef>& items = selection_.items();
    const std::size_t chunk_count = app_.document().workspace().chunks().size();

    entry(QStringLiteral("add_chunk"), tr("Add chunk"), true, [this]() { add_chunk(); });

    if (items.size() == 1 && items.front().kind == NodeKind::Chunk) {
        const std::int32_t id = items.front().chunk_id;
        entry(QStringLiteral("set_active_chunk"), tr("Set as active chunk"),
              app_.document().active_chunk_id() != id, [this, id]() { activate_chunk(id); });
        // The engine refuses to remove the last chunk; say so up front instead of failing.
        entry(QStringLiteral("remove_chunk"), tr("Remove chunk"), chunk_count > 1,
              [this, id]() { remove_chunk(id); });
    }

    std::int32_t camera_chunk = -1;
    const std::vector<std::int32_t> cameras = selected_cameras(camera_chunk);
    if (!cameras.empty()) {
        menu->addSeparator();
        const Chunk* chunk = app_.document().workspace().find_chunk(camera_chunk);
        bool any_disabled = false;
        bool any_enabled = false;
        if (chunk != nullptr) {
            for (const Camera& camera : chunk->cameras()) {
                if (std::binary_search(cameras.begin(), cameras.end(), camera.id)) {
                    (camera.enabled ? any_enabled : any_disabled) = true;
                }
            }
        }
        entry(QStringLiteral("enable_cameras"), tr("Enable cameras"), any_disabled,
              [this, camera_chunk, cameras]() { set_cameras_enabled(camera_chunk, cameras, true); });
        entry(QStringLiteral("disable_cameras"), tr("Disable cameras"), any_enabled,
              [this, camera_chunk, cameras]() { set_cameras_enabled(camera_chunk, cameras, false); });
    }
    return menu;
}

void WorkspacePane::on_context_menu_requested(const QPoint& pos) {
    // Right-click on an unselected row selects it first, like every file manager.
    const QModelIndex index = tree_->indexAt(pos);
    if (index.isValid() && !tree_->selectionModel()->isSelected(index)) {
        tree_->selectionModel()->setCurrentIndex(
            index, QItemSelectionModel::ClearAndSelect | QItemSelectionModel::Current);
    }
    std::unique_ptr<QMenu> menu = build_context_menu();
    menu->exec(tree_->viewport()->mapToGlobal(pos));
}

}  // namespace alphashape::gui
