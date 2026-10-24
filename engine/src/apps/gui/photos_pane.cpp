#include "apps/gui/photos_pane.hpp"

#include <algorithm>
#include <exception>
#include <string>
#include <utility>

#include <QAbstractItemView>
#include <QAction>
#include <QColor>
#include <QFont>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QItemSelectionModel>
#include <QKeySequence>
#include <QLabel>
#include <QLineEdit>
#include <QListView>
#include <QMenu>
#include <QMessageBox>
#include <QPainter>
#include <QPainterPath>
#include <QSize>
#include <QStyle>
#include <QStyledItemDelegate>
#include <QVariant>
#include <QVBoxLayout>

#include "apps/gui/app.hpp"
#include "apps/gui/theme/theme.hpp"
#include "apps/gui/workspace_model.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"

namespace alphashape::gui {

namespace {

/// Geometry of one photo cell. The thumbnail box is the icon area above the caption.
constexpr int kCellWidth = 172;
constexpr int kCellHeight = 158;
constexpr int kThumbWidth = 160;
constexpr int kThumbHeight = 112;
constexpr int kPadding = 5;
constexpr int kCaptionHeight = 17;
constexpr int kBadgeHeight = 15;

/// A font a little smaller than `base`. The style sheet sizes text in pixels, so the
/// point size is often -1 (and setPointSizeF(-1) would warn); scale whichever unit the font
/// actually carries.
QFont smaller_font(const QFont& base, double points, bool bold) {
    QFont font = base;
    if (font.pointSizeF() > 0.0) {
        font.setPointSizeF(std::max(font.pointSizeF() - points, 5.0));
    } else if (font.pixelSize() > 0) {
        const auto delta = static_cast<int>(points * 1.4 + 0.5);
        font.setPixelSize(std::max(font.pixelSize() - delta, 6));
    }
    font.setBold(bold);
    return font;
}

/// Paints one photo: thumbnail, caption, and the state badges the model reports.
class PhotoDelegate : public QStyledItemDelegate {
public:
    PhotoDelegate(Theme& theme, QObject* parent) : QStyledItemDelegate(parent), theme_(theme) {}
    ~PhotoDelegate() override = default;

    QSize sizeHint(const QStyleOptionViewItem&, const QModelIndex&) const override {
        return {kCellWidth, kCellHeight};
    }

    void paint(QPainter* painter, const QStyleOptionViewItem& option,
               const QModelIndex& index) const override {
        painter->save();
        painter->setRenderHint(QPainter::Antialiasing, true);
        painter->setRenderHint(QPainter::SmoothPixmapTransform, true);

        const QRect cell = option.rect.adjusted(2, 2, -2, -2);
        const bool selected = (option.state & QStyle::State_Selected) != 0;
        const QPalette& palette = option.palette;
        const QColor surface = palette.color(QPalette::Base);
        const QColor border = selected ? palette.color(QPalette::Accent)
                                       : palette.color(QPalette::Mid);

        QPainterPath frame;
        frame.addRoundedRect(QRectF(cell), 3.0, 3.0);
        painter->fillPath(frame, surface);
        painter->setPen(QPen(border, selected ? 1.5 : 1.0));
        painter->drawPath(frame);

        const QRect thumb_area(cell.left() + kPadding, cell.top() + kPadding, kThumbWidth,
                              kThumbHeight);
        const QVariant decoration = index.data(Qt::DecorationRole);
        const QPixmap thumbnail = decoration.canConvert<QPixmap>() ? decoration.value<QPixmap>()
                                                                  : QPixmap();
        if (!thumbnail.isNull()) {
            const QPixmap scaled = thumbnail.scaled(thumb_area.size(), Qt::KeepAspectRatio,
                                                    Qt::SmoothTransformation);
            const QRect target(thumb_area.center().x() - scaled.width() / 2,
                               thumb_area.center().y() - scaled.height() / 2, scaled.width(),
                               scaled.height());
            painter->drawPixmap(target, scaled);
        } else {
            const auto state = static_cast<PhotosModel::ThumbnailState>(
                index.data(PhotosModel::ThumbnailStateRole).toInt());
            const QString text = state == PhotosModel::ThumbnailState::Missing
                                     ? QObject::tr("no preview")
                                     : QObject::tr("loading...");
            painter->setPen(palette.color(QPalette::Disabled, QPalette::Text));
            painter->setFont(smaller_font(option.font, 1.0, false));
            painter->drawText(thumb_area, Qt::AlignCenter, text);
        }

        // Caption.
        const QFont caption = smaller_font(option.font, 0.5, selected);
        painter->setFont(caption);
        const bool enabled = index.data(PhotosModel::EnabledRole).toBool();
        painter->setPen(selected ? palette.color(QPalette::Accent)
                                 : palette.color(enabled ? QPalette::Active : QPalette::Disabled,
                                                 QPalette::Text));
        const QRect caption_area(cell.left() + kPadding, thumb_area.bottom() + 2, kThumbWidth,
                                 kCaptionHeight);
        const QFontMetrics metrics(caption);
        painter->drawText(caption_area,
                          Qt::AlignLeft | Qt::AlignVCenter,
                          metrics.elidedText(index.data(Qt::DisplayRole).toString(),
                                             Qt::ElideMiddle, kThumbWidth));

        // Badges, right to left along the top of the thumbnail: amber for "not aligned", red
        // for "disabled", the two states the engine distinguishes. The kind decides the colour, so
        // a translation of the text can never be mistaken for the other badge. Text and kind come
        // from the same role pair and always have the same length.
        const QStringList badges = index.data(PhotosModel::BadgesRole).toStringList();
        const QVariantList badge_kinds = index.data(PhotosModel::BadgeKindsRole).toList();
        const QFont badge_font = smaller_font(option.font, 2.0, true);
        painter->setFont(badge_font);
        const QFontMetrics badge_metrics(badge_font);
        int right = thumb_area.right() - 2;
        const int badge_count =
            std::min(static_cast<int>(badges.size()), static_cast<int>(badge_kinds.size()));
        for (int i = 0; i < badge_count; ++i) {
            const QString& badge = badges[i];
            const auto kind = static_cast<PhotosModel::Badge>(badge_kinds[i].toInt());
            const bool aligned_warning = kind == PhotosModel::NotAlignedBadge;
            const QColor colour = aligned_warning ? Theme::status_warning(theme_.mode())
                                                  : Theme::status_error(theme_.mode());
            const int width = badge_metrics.horizontalAdvance(badge) + 10;
            const QRect chip(right - width + 1, thumb_area.top() + 2, width, kBadgeHeight);
            QPainterPath chip_path;
            chip_path.addRoundedRect(QRectF(chip), kBadgeHeight / 2.0, kBadgeHeight / 2.0);
            painter->fillPath(chip_path, palette.color(QPalette::Base));
            painter->setPen(QPen(colour, 1.0));
            painter->drawPath(chip_path);
            painter->setPen(colour);
            painter->drawText(chip, Qt::AlignCenter, badge);
            right = chip.left() - 4;
        }
        painter->restore();
    }

private:
    Theme& theme_;
};

/// Pairs GuiApp::begin_macro() with end_macro() for the duration of a scope.
///
/// The macro is opened by the constructor and closed by the destructor, so an early return or an
/// exception thrown by the body can never leave the undo stack with an open group (which would
/// hold every later edit hostage and refuse undo/redo). If the body threw after some commands
/// were applied, the destructor still closes the group: those edits become one reversible
/// transaction that the caller reports as failed, instead of an unbalanced stack.
class MacroGuard {
public:
    MacroGuard(GuiApp& app, const std::string& name) : app_(app) {
        app_.begin_macro(name);
        open_ = true;
    }

    ~MacroGuard() {
        if (!open_) {
            return;
        }
        // Closing never touches the document (see GuiApp::end_macro), so the exception path is
        // safe. The catch only covers an unbalanced stack, which cannot happen from here.
        try {
            app_.end_macro();
        } catch (...) {
        }
    }

    MacroGuard(const MacroGuard&) = delete;
    MacroGuard& operator=(const MacroGuard&) = delete;

private:
    GuiApp& app_;
    bool open_ = false;
};

}  // namespace

PhotosPane::PhotosPane(GuiApp& app, WorkspaceSelection& selection, QWidget* parent)
    : QWidget(parent), app_(app), selection_(selection) {
    setObjectName(QStringLiteral("pane.photos"));
    error_handler_ = [this](const QString& title, const QString& message) {
        QMessageBox::warning(this, title, message);
    };

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    auto* header = new QWidget(this);
    header->setObjectName(QStringLiteral("photos.header"));
    auto* header_layout = new QHBoxLayout(header);
    header_layout->setContentsMargins(6, 4, 6, 4);
    header_layout->setSpacing(6);

    filter_ = new QLineEdit(header);
    filter_->setObjectName(QStringLiteral("photos.filter"));
    filter_->setPlaceholderText(tr("Filter photos"));
    filter_->setClearButtonEnabled(true);
    filter_->setFixedWidth(180);
    filter_->setToolTip(tr("Show only photos whose name contains this text"));
    header_layout->addWidget(filter_);

    summary_ = new QLabel(header);
    summary_->setObjectName(QStringLiteral("photos.summary"));
    header_layout->addWidget(summary_);
    header_layout->addStretch(1);
    layout->addWidget(header);

    model_ = new PhotosModel(app_, this);
    view_ = new QListView(this);
    view_->setObjectName(QStringLiteral("view.photos"));
    view_->setModel(model_);
    view_->setViewMode(QListView::IconMode);
    view_->setResizeMode(QListView::Adjust);
    view_->setMovement(QListView::Static);
    view_->setFlow(QListView::LeftToRight);
    view_->setWrapping(true);
    view_->setUniformItemSizes(true);
    view_->setGridSize(QSize(kCellWidth, kCellHeight));
    view_->setSelectionMode(QAbstractItemView::ExtendedSelection);
    view_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    view_->setContextMenuPolicy(Qt::CustomContextMenu);
    view_->setSpacing(0);
    view_->setWordWrap(false);
    view_->setItemDelegate(new PhotoDelegate(app_.theme(), view_));
    layout->addWidget(view_, 1);

    // Delete disables, like every photo browser: the common action rather than deletion. The
    // actions belong to the list, not the pane, so Delete still edits text in the filter box
    // (whose child it is not).
    enable_action_ = new QAction(tr("Enable photos"), this);
    enable_action_->setObjectName(QStringLiteral("action.photos.enable"));
    connect(enable_action_, &QAction::triggered, this, [this]() { enable_selected(); });
    disable_action_ = new QAction(tr("Disable photos"), this);
    disable_action_->setObjectName(QStringLiteral("action.photos.disable"));
    disable_action_->setShortcut(QKeySequence::Delete);
    disable_action_->setShortcutContext(Qt::WidgetWithChildrenShortcut);
    connect(disable_action_, &QAction::triggered, this, [this]() { disable_selected(); });
    view_->addAction(enable_action_);
    view_->addAction(disable_action_);

    connect(filter_, &QLineEdit::textChanged, this, [this](const QString& text) {
        model_->set_filter(text);  // the reset that follows re-applies the shared selection
        update_summary();
    });
    connect(view_->selectionModel(), &QItemSelectionModel::selectionChanged, this,
            &PhotosPane::on_view_selection_changed);
    connect(&selection_, &WorkspaceSelection::changed, this,
            &PhotosPane::on_shared_selection_changed);
    connect(view_, &QListView::customContextMenuRequested, this,
            &PhotosPane::on_context_menu_requested);
    connect(&app_, &GuiApp::undo_changed, this, [this]() {
        // The model refreshes itself on undo_changed; the pane only re-reads the flags its
        // actions depend on.
        update_actions();
        update_summary();
    });
    connect(&app_, &GuiApp::document_replaced, this, [this]() {
        view_->selectionModel()->clear();
        update_actions();
        update_summary();
    });
    // Every rebuild (a finished job, an undo that changed the camera set, the filter, a replaced
    // document) clears the list's selection as part of the reset. Re-apply the shared selection
    // from the identity the tree and the Properties sheet are holding, the way the Workspace
    // tree restores its view after its own reset.
    connect(model_, &PhotosModel::reset_completed, this, [this]() {
        on_shared_selection_changed();
        update_actions();
        update_summary();
    });
    connect(&app_.jobs(), &JobManager::busy_changed, this, [this](bool) { update_actions(); });
    // The badges are painted from the theme's status colours, so a theme switch needs a repaint.
    connect(&app_.theme(), &Theme::changed, this,
            [this](ThemeMode) { view_->viewport()->update(); });

    update_actions();
    update_summary();
}

void PhotosPane::set_error_handler(ErrorHandler handler) { error_handler_ = std::move(handler); }

void PhotosPane::report(const QString& title, const QString& message) {
    if (error_handler_) {
        error_handler_(title, message);
    }
}

bool PhotosPane::refuse_while_busy(const QString& title) {
    if (!app_.jobs().busy()) {
        return false;
    }
    report(title, tr("A task is running. Wait for it to finish or cancel it first."));
    return true;
}

void PhotosPane::set_filter(const QString& text) {
    if (filter_->text() == text) {
        return;
    }
    filter_->setText(text);  // textChanged does the rest
}

void PhotosPane::update_summary() {
    if (filter_->text().trimmed().isEmpty()) {
        summary_->setText(tr("%1 photos").arg(model_->total_count()));
    } else {
        summary_->setText(tr("%1 of %2 photos").arg(model_->rowCount()).arg(model_->total_count()));
    }
}

void PhotosPane::update_actions() {
    const bool busy = app_.jobs().busy();
    const std::vector<std::int32_t> selected = selected_camera_ids();
    bool any_enabled = false;
    bool any_disabled = false;
    if (!busy && !selected.empty() && model_->chunk_id() >= 0) {
        app_.read_document([&](const Document& document) {
            const Chunk* chunk = nullptr;
            for (const auto& workspace : document.workspaces()) {
                for (const auto& candidate : workspace->chunks()) {
                    if (candidate->id() == model_->chunk_id()) {
                        chunk = candidate.get();
                    }
                }
            }
            if (chunk == nullptr) {
                return;
            }
            for (const Camera& camera : chunk->cameras()) {
                if (std::find(selected.begin(), selected.end(), camera.id) != selected.end()) {
                    (camera.enabled ? any_enabled : any_disabled) = true;
                }
            }
        });
    }
    enable_action_->setEnabled(!busy && any_disabled);
    disable_action_->setEnabled(!busy && any_enabled);
}

// --- Selection -----------------------------------------------------------------------------

std::vector<std::int32_t> PhotosPane::selected_camera_ids() const {
    std::vector<std::int32_t> ids;
    QModelIndexList rows = view_->selectionModel()->selectedIndexes();
    std::sort(rows.begin(), rows.end(), [](const QModelIndex& a, const QModelIndex& b) {
        return a.row() < b.row();
    });
    ids.reserve(static_cast<std::size_t>(rows.size()));
    for (const QModelIndex& index : rows) {
        const int id = index.data(PhotosModel::CameraIdRole).toInt();
        if (id >= 0) {
            ids.push_back(id);
        }
    }
    return ids;
}

void PhotosPane::on_view_selection_changed() {
    if (syncing_) {
        return;
    }
    // The reset that a rebuild performs drops the list's selection (QAbstractItemView::reset).
    // That is the rebuild talking, not the user deselecting, so never publish it: the shared
    // selection the tree and the Properties sheet read must survive, and the pane re-applies it
    // from reset_completed().
    if (model_->resetting()) {
        return;
    }
    syncing_ = true;
    const int chunk_id = model_->chunk_id();
    QList<NodeRef> items;
    for (const std::int32_t id : selected_camera_ids()) {
        items.append(NodeRef{NodeKind::Camera, chunk_id, id});
    }
    selection_.set(items, items.isEmpty() ? NodeRef{} : items.front());
    syncing_ = false;
    update_actions();
}

void PhotosPane::on_shared_selection_changed() {
    if (syncing_) {
        return;
    }
    // Mirror a camera selection in the tree into the list; a selection of anything else clears it.
    // An empty shared selection is not a command to clear the list: the reset above re-syncs from
    // whatever the tree is holding and leaves an untouched list alone.
    const NodeRef current = selection_.current();
    const bool camera = selection_.has_current() && current.kind == NodeKind::Camera &&
                        current.chunk_id == model_->chunk_id();
    syncing_ = true;
    if (camera) {
        const int row = model_->row_for_camera(current.item_id);
        if (row >= 0) {
            view_->selectionModel()->setCurrentIndex(
                model_->index(row, 0),
                QItemSelectionModel::ClearAndSelect | QItemSelectionModel::Current);
        } else {
            view_->selectionModel()->clear();
        }
    } else if (!selection_.items().isEmpty()) {
        view_->selectionModel()->clear();
    }
    syncing_ = false;
    update_actions();
}

bool PhotosPane::select_camera(std::int32_t camera_id) {
    const int row = model_->row_for_camera(camera_id);
    if (row < 0) {
        return false;
    }
    view_->selectionModel()->setCurrentIndex(model_->index(row, 0),
                                             QItemSelectionModel::ClearAndSelect |
                                                 QItemSelectionModel::Current);
    return true;
}

// --- Operations ----------------------------------------------------------------------------

bool PhotosPane::set_cameras_enabled(const std::vector<std::int32_t>& camera_ids, bool enabled) {
    if (camera_ids.empty() || model_->chunk_id() < 0) {
        return false;
    }
    if (refuse_while_busy(enabled ? tr("Enable photos") : tr("Disable photos"))) {
        return false;
    }
    const std::int32_t chunk_id = model_->chunk_id();
    // One undo entry for the whole selection, named after what the user did, so the Edit menu
    // reads "Undo Disable 3 photos". The engine changes the flag on every camera in one command;
    // the macro gives that command the action's name instead of the engine's.
    const int count = static_cast<int>(camera_ids.size());
    const QString name = count == 1 ? (enabled ? tr("Enable photo") : tr("Disable photo"))
                                    : (enabled ? tr("Enable %1 photos") : tr("Disable %1 photos"))
                                          .arg(count);
    try {
        // One macro for the whole selection: the guard closes it however this function leaves,
        // so a refused command cannot leave an open group behind.
        const MacroGuard macro(app_, name.toStdString());
        app_.execute(make_set_camera_enabled_command(chunk_id, camera_ids, enabled));
        return true;
    } catch (const std::exception& error) {
        report(enabled ? tr("Enable photos") : tr("Disable photos"),
               QString::fromUtf8(error.what()));
        return false;
    }
}

bool PhotosPane::disable_selected() {
    return set_cameras_enabled(selected_camera_ids(), false);
}

bool PhotosPane::enable_selected() { return set_cameras_enabled(selected_camera_ids(), true); }

// --- Context menu --------------------------------------------------------------------------

std::unique_ptr<QMenu> PhotosPane::build_context_menu() {
    auto menu = std::make_unique<QMenu>(this);
    menu->setObjectName(QStringLiteral("menu.photos.context"));
    // While a job runs the document is off limits: the mutating entries stay listed but
    // disabled, and nothing here reads the document.
    const bool busy = app_.jobs().busy();
    auto entry = [&](const QString& name, const QString& text, bool enabled, auto&& fn) {
        QAction* action = menu->addAction(text);
        action->setObjectName(QStringLiteral("action.photos.") + name);
        action->setEnabled(enabled);
        connect(action, &QAction::triggered, this, fn);
        return action;
    };
    entry(QStringLiteral("select_all"), tr("Select all"), !busy && model_->rowCount() > 0,
          [this]() { view_->selectAll(); });
    menu->addSeparator();
    entry(QStringLiteral("enable"), tr("Enable"), enable_action_->isEnabled(), [this]() {
        enable_selected();
    });
    entry(QStringLiteral("disable"), tr("Disable"), disable_action_->isEnabled(), [this]() {
        disable_selected();
    });
    return menu;
}

void PhotosPane::on_context_menu_requested(const QPoint& pos) {
    // Right-click on an unselected cell selects it first, like every file manager.
    const QModelIndex index = view_->indexAt(pos);
    if (index.isValid() && !view_->selectionModel()->isSelected(index)) {
        view_->selectionModel()->setCurrentIndex(
            index, QItemSelectionModel::ClearAndSelect | QItemSelectionModel::Current);
    }
    std::unique_ptr<QMenu> menu = build_context_menu();
    menu->exec(view_->viewport()->mapToGlobal(pos));
}

}  // namespace alphashape::gui
