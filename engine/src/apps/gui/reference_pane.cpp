#include "apps/gui/reference_pane.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>

#include <QAction>
#include <QAbstractItemModel>
#include <QAbstractItemView>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QLabel>
#include <QMessageBox>
#include <QString>
#include <QStyleOptionViewItem>
#include <QStyledItemDelegate>
#include <QTableView>
#include <QToolBar>
#include <QVBoxLayout>

#include "apps/gui/app.hpp"
#include "apps/gui/job_manager.hpp"
#include "apps/gui/reference_settings_dialog.hpp"
#include "apps/gui/task_dialog.hpp"
#include "apps/gui/workspace_model.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/tasks/registry.hpp"
#include "model/crs/coordinate_system.hpp"

namespace alphashape::gui {
namespace {

QString rmse_text(double value) {
    return std::isfinite(value) ? QString::number(value, 'g', 6) : QStringLiteral("n/a");
}

/// A double editor for source coordinates and accuracy. The default delegate
/// only builds a spin box when the model returns a double, and these columns
/// return locale-formatted text, so the editor is created here and the value is
/// read from the display text.
class NumberDelegate : public QStyledItemDelegate {
public:
    explicit NumberDelegate(double minimum, QObject* parent = nullptr)
        : QStyledItemDelegate(parent), minimum_(minimum) {}

    QWidget* createEditor(QWidget* parent, const QStyleOptionViewItem&,
                          const QModelIndex&) const override {
        auto* spin = new QDoubleSpinBox(parent);
        spin->setRange(minimum_, 1e12);
        spin->setDecimals(10);
        spin->setKeyboardTracking(false);
        return spin;
    }

    void setEditorData(QWidget* editor, const QModelIndex& index) const override {
        if (auto* spin = qobject_cast<QDoubleSpinBox*>(editor)) {
            spin->setValue(index.data(Qt::DisplayRole).toString().toDouble());
            return;
        }
        QStyledItemDelegate::setEditorData(editor, index);
    }

    void setModelData(QWidget* editor, QAbstractItemModel* model,
                      const QModelIndex& index) const override {
        if (auto* spin = qobject_cast<QDoubleSpinBox*>(editor)) {
            spin->interpretText();
            model->setData(index, spin->value(), Qt::EditRole);
            return;
        }
        QStyledItemDelegate::setModelData(editor, model, index);
    }

private:
    double minimum_;
};

}  // namespace

ReferencePane::ReferencePane(GuiApp& app, WorkspaceSelection& selection, QWidget* parent)
    : QWidget(parent), app_(app), selection_(selection) {
    setObjectName(QStringLiteral("pane.reference"));
    camera_model_ = new ReferenceModel(app_, ReferenceModel::Section::Cameras, this);
    marker_model_ = new ReferenceModel(app_, ReferenceModel::Section::Markers, this);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    toolbar_ = new QToolBar(this);
    toolbar_->setObjectName(QStringLiteral("toolbar.reference"));
    toolbar_->setToolButtonStyle(Qt::ToolButtonTextOnly);
    layout->addWidget(toolbar_);
    create_toolbar();
    create_tables();
    status_ = new QLabel(this);
    status_->setObjectName(QStringLiteral("reference.status"));
    status_->setContentsMargins(6, 2, 6, 2);
    layout->addWidget(status_);

    connect(camera_model_, &ReferenceModel::refreshed, this,
            &ReferencePane::update_status);
    connect(marker_model_, &ReferenceModel::refreshed, this,
            &ReferencePane::update_status);
    connect(&app_.jobs(), &JobManager::busy_changed, this, [this](bool) {
        update_actions();
    });
    connect(&app_, &GuiApp::document_replaced, this,
            [this]() { update_actions(); });
    update_actions();
    refresh();
}

void ReferencePane::create_tables() {
    const auto make_view = [this](ReferenceModel* model, const QString& name) {
        auto* view = new QTableView(this);
        view->setObjectName(name);
        view->setModel(model);
        view->setSelectionBehavior(QAbstractItemView::SelectRows);
        view->setEditTriggers(QAbstractItemView::DoubleClicked |
                              QAbstractItemView::SelectedClicked |
                              QAbstractItemView::EditKeyPressed);
        view->verticalHeader()->setVisible(false);
        view->horizontalHeader()->setStretchLastSection(true);
        view->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
        view->horizontalHeader()->setSectionResizeMode(
            static_cast<int>(ReferenceModel::LabelColumn), QHeaderView::Stretch);
        auto* coordinates = new NumberDelegate(-1e12, view);
        auto* accuracy = new NumberDelegate(1e-9, view);
        for (int column : {static_cast<int>(ReferenceModel::SourceXColumn),
                           static_cast<int>(ReferenceModel::SourceYColumn),
                           static_cast<int>(ReferenceModel::SourceZColumn)}) {
            view->setItemDelegateForColumn(column, coordinates);
        }
        for (int column : {static_cast<int>(ReferenceModel::AccuracyXColumn),
                           static_cast<int>(ReferenceModel::AccuracyYColumn),
                           static_cast<int>(ReferenceModel::AccuracyZColumn)}) {
            view->setItemDelegateForColumn(column, accuracy);
        }
        return view;
    };
    camera_view_ = make_view(camera_model_, QStringLiteral("reference.cameras"));
    marker_view_ = make_view(marker_model_, QStringLiteral("reference.markers"));
    // "Show photos with this marker" is only meaningful with a marker row selected.
    connect(marker_view_->selectionModel(), &QItemSelectionModel::selectionChanged, this,
            [this]() { update_actions(); });
    auto* layout = qobject_cast<QVBoxLayout*>(this->layout());
    auto* camera_title = new QLabel(tr("Cameras"), this);
    camera_title->setContentsMargins(6, 2, 6, 0);
    layout->addWidget(camera_title);
    layout->addWidget(camera_view_, 1);
    auto* marker_title = new QLabel(tr("Markers"), this);
    marker_title->setContentsMargins(6, 2, 6, 0);
    layout->addWidget(marker_title);
    layout->addWidget(marker_view_, 1);
}

void ReferencePane::create_toolbar() {
    const auto make = [this](const QString& name, const QString& text) {
        auto* entry = new QAction(text, this);
        entry->setObjectName(name);
        toolbar_->addAction(entry);
        return entry;
    };
    optimize_ = make(QStringLiteral("action.reference.optimize"), tr("Optimize"));
    optimize_->setToolTip(tr("Optimize the camera alignment"));
    connect(optimize_, &QAction::triggered, this,
            [this]() { run_task("OptimizeCameras"); });

    import_ = make(QStringLiteral("action.reference.import"), tr("Import"));
    import_->setToolTip(tr("Import reference coordinates"));
    connect(import_, &QAction::triggered, this,
            [this]() { run_task("ImportReference"); });

    export_ = make(QStringLiteral("action.reference.export"), tr("Export"));
    export_->setToolTip(tr("Export the reference table"));
    connect(export_, &QAction::triggered, this,
            [this]() { run_task("ExportReference"); });

    convert_ = make(QStringLiteral("action.reference.convert"), tr("Convert"));
    if (builtin_registry().find("ConvertReference") != nullptr) {
        convert_->setToolTip(tr("Convert the reference coordinates to another CRS"));
        connect(convert_, &QAction::triggered, this,
                [this]() { run_task("ConvertReference"); });
    } else {
        convert_->setEnabled(false);
        convert_->setToolTip(tr("ConvertReference is not available in this build"));
    }

    QAction* settings = make(QStringLiteral("action.reference.settings"), tr("Settings"));
    settings->setToolTip(tr("Edit the reference settings"));
    connect(settings, &QAction::triggered, this, [this]() {
        if (app_.jobs().busy()) {
            return;
        }
        ReferenceSettingsDialog dialog(app_, this);
        dialog.exec();
        refresh();
    });

    import_markers_ = make(QStringLiteral("action.reference.import_markers"),
                           tr("Import Markers"));
    if (builtin_registry().find("ImportMarkers") != nullptr) {
        import_markers_->setToolTip(
            tr("Import marker projections from a CSV or XML file"));
        connect(import_markers_, &QAction::triggered, this,
                [this]() { run_task("ImportMarkers"); });
    } else {
        import_markers_->setEnabled(false);
        import_markers_->setToolTip(tr("ImportMarkers is not available in this build"));
    }

    // Placing a marker is done on the photo, not in this table: the action is a hint that
    // switches the central area to the Photo tab. It keeps its objectName so a shortcut or a
    // test that named it still finds it.
    pick_ = make(QStringLiteral("action.reference.pick"), tr("Place on photo"));
    pick_->setToolTip(tr("Open the selected photo to place markers on it"));
    connect(pick_, &QAction::triggered, this, [this]() {
        if (app_.jobs().busy()) {
            return;
        }
        const std::int32_t camera_id = camera_for_photo();
        if (camera_id >= 0) {
            Q_EMIT place_on_photo_requested(camera_id);
        }
    });

    // "Show photos with this marker": filters the Photos dock to the cameras that see the
    // selected marker row, so the user can find the photos where a control point is visible.
    show_photos_ = make(QStringLiteral("action.reference.show_photos"),
                        tr("Show photos with this marker"));
    show_photos_->setToolTip(
        tr("Filter the Photos list to the photos that see the selected marker"));
    connect(show_photos_, &QAction::triggered, this, [this]() {
        if (app_.jobs().busy()) {
            return;
        }
        const std::int32_t row = selected_marker_row();
        if (row < 0) {
            return;
        }
        const std::int32_t marker_id = marker_model_->row_id(row);
        if (marker_id < 0) {
            return;
        }
        Q_EMIT show_marker_photos_requested(
            marker_id, QString::fromStdString(marker_model_->row(row).label));
    });
}

std::int32_t ReferencePane::selected_marker_row() const {
    if (marker_view_ == nullptr || marker_view_->selectionModel() == nullptr) {
        return -1;
    }
    const QModelIndexList rows = marker_view_->selectionModel()->selectedRows();
    if (rows.isEmpty()) {
        return -1;
    }
    return rows.front().row();
}

std::int32_t ReferencePane::camera_for_photo() const {
    // The camera the Workspace tree (or the Photos list) has selected, when it belongs to the
    // active chunk. The tree publishes into the shared selection; the engine's per-camera
    // `selected` flag is not maintained by the GUI, so it is not consulted.
    const std::int32_t active = app_.read_document([](const Document& document) {
        const Chunk* chunk = document.active_chunk();
        return chunk == nullptr ? std::int32_t{-1} : chunk->id();
    });
    if (active < 0) {
        return -1;
    }
    for (const NodeRef& ref : selection_.items()) {
        if (ref.kind == NodeKind::Camera && ref.chunk_id == active) {
            return ref.item_id;
        }
    }
    // Nothing selected: fall back to the first camera that actually has a frame, so the Photo
    // tab opens on a picture rather than an empty message.
    return app_.read_document([active](const Document& document) {
        const Chunk* chunk = document.active_chunk();
        if (chunk == nullptr || chunk->id() != active) {
            return std::int32_t{-1};
        }
        for (const Camera& camera : chunk->cameras()) {
            for (const std::shared_ptr<Frame>& frame : chunk->frames()) {
                if (frame != nullptr && frame->camera_id() == camera.id) {
                    return camera.id;
                }
            }
        }
        return chunk->cameras().empty() ? std::int32_t{-1} : chunk->cameras().front().id;
    });
}

QAction* ReferencePane::action(const QString& object_name) const {
    return findChild<QAction*>(object_name);
}

void ReferencePane::refresh() {
    camera_model_->refresh();
    marker_model_->refresh();
    update_actions();
    update_status();
}

void ReferencePane::run_task(const std::string& task) {
    if (app_.jobs().busy() || builtin_registry().find(task) == nullptr) {
        return;
    }
    const std::int32_t chunk_id = app_.read_document([](const Document& document) {
        const Chunk* chunk = document.active_chunk();
        return chunk == nullptr ? -1 : chunk->id();
    });
    if (chunk_id < 0) {
        return;
    }
    try {
        TaskDialog dialog(task, builtin_registry(), app_.settings(), this);
        if (dialog.exec() != QDialog::Accepted) {
            return;
        }
        app_.jobs().submit(task, dialog.params(), chunk_id);
    } catch (const std::exception& error) {
        QMessageBox::critical(this, tr("Reference"), QString::fromUtf8(error.what()));
    }
}

void ReferencePane::update_status() {
    const reference::ReferenceSummary& summary = camera_model_->summary();
    QString crs = tr("Local coordinates");
    app_.read_document([&](const Document& document) {
        const Chunk* chunk = document.active_chunk();
        if (chunk == nullptr) {
            return;
        }
        const crs::CoordinateSystem cs = chunk->coordinate_system();
        if (cs.is_set() && !cs.is_local()) {
            crs = QString::fromStdString(cs.name());
        }
    });
    status_->setText(
        tr("%1 markers  |  camera RMSE %2 m  |  control RMSE %3 m  |  check RMSE %4 m  |  %5")
            .arg(QString::number(static_cast<qulonglong>(summary.marker_count)),
                 rmse_text(summary.camera_rmse_m), rmse_text(summary.control_rmse_m),
                 rmse_text(summary.check_rmse_m), crs));
}

void ReferencePane::update_actions() {
    const bool busy = app_.jobs().busy();
    const bool idle_chunk =
        !busy && app_.read_document([](const Document& document) {
            return document.active_chunk() != nullptr;
        });
    for (QAction* entry : {optimize_, import_, export_}) {
        if (entry != nullptr) {
            entry->setEnabled(idle_chunk);
        }
    }
    if (import_markers_ != nullptr &&
        builtin_registry().find("ImportMarkers") != nullptr) {
        import_markers_->setEnabled(idle_chunk);
    }
    if (pick_ != nullptr) {
        pick_->setEnabled(idle_chunk);
    }
    if (show_photos_ != nullptr) {
        show_photos_->setEnabled(idle_chunk && selected_marker_row() >= 0);
    }
    if (QAction* settings = action(QStringLiteral("action.reference.settings"))) {
        settings->setEnabled(idle_chunk);
    }
    if (convert_ != nullptr && builtin_registry().find("ConvertReference") != nullptr) {
        convert_->setEnabled(idle_chunk);
    }
}

}  // namespace alphashape::gui
