#include "apps/gui/reference_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <QModelIndex>

#include "apps/gui/app.hpp"
#include "apps/gui/job_manager.hpp"
#include "common/algorithm/reference/reference_table.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "model/crs/transform.hpp"

namespace alphashape::gui {
namespace {

bool row_before(const reference::ReferenceRow& a,
                const reference::ReferenceRow& b) {
    // Descending |error|; a row without an error sorts after every measured one.
    if (a.error_m.has_value() != b.error_m.has_value()) {
        return a.error_m.has_value();
    }
    if (!a.error_m.has_value()) {
        return a.id < b.id;
    }
    return *a.error_m > *b.error_m;
}

}  // namespace

ReferenceModel::ReferenceModel(GuiApp& app, Section section, QObject* parent)
    : QAbstractTableModel(parent), app_(app), section_(section) {
    connect(&app_, &GuiApp::undo_changed, this, &ReferenceModel::refresh);
    connect(&app_, &GuiApp::document_replaced, this, &ReferenceModel::refresh);
    connect(&app_, &GuiApp::active_chunk_changed, this, &ReferenceModel::refresh);
    connect(&app_.jobs(), &JobManager::job_finished, this,
            [this](quint64, JobManager::State, const QString&) {
                refresh_after_job();
            });
    connect(&app_.jobs(), &JobManager::busy_changed, this, [this](bool busy) {
        if (!busy && stale_) {
            refresh();
        }
    });
}

void ReferenceModel::refresh_after_job() {
    // A task may change camera poses or marker positions without a command.
    refresh();
}

void ReferenceModel::refresh() {
    if (app_.jobs().busy()) {
        stale_ = true;
        return;
    }
    std::vector<reference::ReferenceRow> rows;
    reference::ReferenceSummary summary;
    std::int32_t chunk_id = -1;
    bool geographic = false;
    app_.read_document([&](const Document& document) {
        const Chunk* chunk = document.active_chunk();
        if (chunk == nullptr) {
            return;
        }
        chunk_id = chunk->id();
        const crs::CoordinateSystem display = chunk->coordinate_system();
        geographic =
            display.is_set() && !display.is_local() &&
            (display.kind() == crs::CrsKind::Geographic2D ||
             display.kind() == crs::CrsKind::Geographic3D ||
             display.kind() == crs::CrsKind::Compound);
        const reference::ReferenceTable table =
            reference::build_reference_table(*chunk, chunk->coordinate_system());
        summary = table.summary();
        const bool cameras = section_ == Section::Cameras;
        for (const reference::ReferenceRow& entry : table.rows) {
            const bool camera = entry.kind == reference::ReferenceRowKind::Camera;
            if (camera == cameras) {
                rows.push_back(entry);
            }
        }
    });
    std::stable_sort(rows.begin(), rows.end(), row_before);

    beginResetModel();
    rows_ = std::move(rows);
    summary_ = summary;
    chunk_id_ = chunk_id;
    geographic_ = geographic;
    stale_ = false;
    endResetModel();
    Q_EMIT refreshed();
}

bool ReferenceModel::has_column(int column) const {
    if (column < 0 || column >= static_cast<int>(ColumnCount)) {
        return false;
    }
    return section_ == Section::Markers || column != static_cast<int>(PinnedColumn);
}

bool ReferenceModel::source_axis(int column, int* axis) {
    switch (column) {
        case SourceXColumn: *axis = 0; return true;
        case SourceYColumn: *axis = 1; return true;
        case SourceZColumn: *axis = 2; return true;
        default: return false;
    }
}

int ReferenceModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) {
        return 0;
    }
    return static_cast<int>(rows_.size());
}

int ReferenceModel::columnCount(const QModelIndex& parent) const {
    if (parent.isValid()) {
        return 0;
    }
    return section_ == Section::Markers ? static_cast<int>(ColumnCount)
                                        : static_cast<int>(PinnedColumn);
}

const reference::ReferenceRow& ReferenceModel::row(int row_index) const {
    static const reference::ReferenceRow empty;
    if (row_index < 0 || row_index >= static_cast<int>(rows_.size())) {
        return empty;
    }
    return rows_[static_cast<std::size_t>(row_index)];
}

std::optional<double> ReferenceModel::error_m(int row_index) const {
    return row(row_index).error_m;
}

std::int32_t ReferenceModel::row_id(int row_index) const {
    return row(row_index).id;
}

namespace {

/// Fixed decimals by unit: 8 for degrees, 3 for metres.
QString number(double value, int decimals) {
    return QString::number(value, 'f', decimals);
}

QString component(const std::optional<Eigen::Vector3d>& vector, int axis,
                  int decimals) {
    if (!vector.has_value()) {
        return QString();
    }
    return number((*vector)[axis], decimals);
}

}  // namespace

QVariant ReferenceModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || !has_column(index.column())) {
        return QVariant();
    }
    const reference::ReferenceRow& item = rows_[static_cast<std::size_t>(index.row())];
    const int column = index.column();
    if (role == Qt::CheckStateRole) {
        if (column == static_cast<int>(LabelColumn)) {
            return item.enabled ? Qt::Checked : Qt::Unchecked;
        }
        if (column == static_cast<int>(PinnedColumn)) {
            return item.pinned ? Qt::Checked : Qt::Unchecked;
        }
        return QVariant();
    }
    if (role == Qt::TextAlignmentRole && column != static_cast<int>(LabelColumn) &&
        column != static_cast<int>(PinnedColumn)) {
        return static_cast<int>(Qt::AlignRight | Qt::AlignVCenter);
    }
    if (role != Qt::DisplayRole) {
        return QVariant();
    }
    const int source_decimals = geographic_ ? 8 : 3;
    switch (column) {
        case LabelColumn: return QString::fromStdString(item.label);
        case SourceXColumn: return component(item.source, 0, source_decimals);
        case SourceYColumn: return component(item.source, 1, source_decimals);
        case SourceZColumn: return component(item.source, 2, source_decimals);
        case AccuracyColumn: return number(item.sigma, 3);
        case EstimatedXColumn: return component(item.estimated, 0, source_decimals);
        case EstimatedYColumn: return component(item.estimated, 1, source_decimals);
        case EstimatedZColumn: return component(item.estimated, 2, source_decimals);
        case ErrorColumn:
            // ENU metres, or display metres for a projected CRS.
            return item.error_m.has_value() ? number(*item.error_m, 3) : QString();
        case ProjectionsColumn:
            return QString::number(static_cast<qulonglong>(item.projection_count));
        default: return QVariant();
    }
}

QVariant ReferenceModel::headerData(int section, Qt::Orientation orientation,
                                    int role) const {
    if (role != Qt::DisplayRole || orientation != Qt::Horizontal ||
        !has_column(section)) {
        return QVariant();
    }
    switch (section) {
        case LabelColumn: return tr("Label");
        case SourceXColumn: return tr("Source X");
        case SourceYColumn: return tr("Source Y");
        case SourceZColumn: return tr("Source Z");
        case AccuracyColumn: return tr("Accuracy");
        case EstimatedXColumn: return tr("Estimated X");
        case EstimatedYColumn: return tr("Estimated Y");
        case EstimatedZColumn: return tr("Estimated Z");
        case ErrorColumn: return tr("Error");
        case ProjectionsColumn: return tr("Projections");
        case PinnedColumn: return tr("Pinned");
        default: return QVariant();
    }
}

Qt::ItemFlags ReferenceModel::flags(const QModelIndex& index) const {
    if (!index.isValid() || !has_column(index.column())) {
        return Qt::NoItemFlags;
    }
    Qt::ItemFlags result = Qt::ItemIsSelectable | Qt::ItemIsEnabled;
    const int column = index.column();
    if (column == static_cast<int>(LabelColumn) ||
        column == static_cast<int>(PinnedColumn)) {
        result |= Qt::ItemIsUserCheckable;
    }
    int axis = 0;
    if (source_axis(column, &axis) || column == static_cast<int>(AccuracyColumn)) {
        result |= Qt::ItemIsEditable;
    }
    return result;
}

bool ReferenceModel::can_edit() const {
    return chunk_id_ >= 0 && !app_.jobs().busy();
}

bool ReferenceModel::setData(const QModelIndex& index, const QVariant& value,
                             int role) {
    if (!can_edit() || !index.isValid() || !has_column(index.column())) {
        return false;
    }
    const reference::ReferenceRow& item =
        rows_[static_cast<std::size_t>(index.row())];
    const int column = index.column();
    if (role == Qt::CheckStateRole) {
        const bool on = value.toInt() == static_cast<int>(Qt::Checked);
        if (column == static_cast<int>(LabelColumn)) {
            if (item.enabled == on) {
                return false;
            }
            if (item.kind == reference::ReferenceRowKind::Camera) {
                app_.execute(
                    make_set_camera_reference_enabled_command(chunk_id_, item.id, on));
            } else {
                const auto role = on ? reference::MarkerReferenceRole::Gcp
                                     : reference::MarkerReferenceRole::CheckPoint;
                app_.execute(make_set_marker_role_command(chunk_id_, item.id, role));
            }
            return true;
        }
        if (column == static_cast<int>(PinnedColumn)) {
            if (item.pinned == on) {
                return false;
            }
            app_.execute(make_set_marker_pinned_command(chunk_id_, item.id, on));
            return true;
        }
        return false;
    }
    if (role != Qt::EditRole) {
        return false;
    }
    bool ok = false;
    const double typed = value.toDouble(&ok);
    if (!ok) {
        return false;
    }
    const bool camera = item.kind == reference::ReferenceRowKind::Camera;
    if (column == static_cast<int>(AccuracyColumn)) {
        if (!(typed > 0.0)) {
            return false;
        }
        const Eigen::Vector3d accuracy = Eigen::Vector3d::Constant(typed);
        if (camera) {
            app_.execute(make_set_camera_reference_command(chunk_id_, item.id,
                                                          item.source, accuracy));
        } else {
            app_.execute(make_set_marker_reference_command(chunk_id_, item.id,
                                                          item.source, accuracy));
        }
        return true;
    }
    int axis = 0;
    if (!source_axis(column, &axis)) {
        return false;
    }
    Eigen::Vector3d location = item.source.value_or(Eigen::Vector3d::Zero());
    location[static_cast<Eigen::Index>(axis)] = typed;
    if (!location.allFinite()) {
        return false;
    }
    // The row is displayed in the chunk CRS, but the stored reference lives in
    // the camera / marker slot CRS. When the two differ, convert the edited
    // display value back through the inverse of the display transform rather
    // than writing display units into the slot.
    CRS slot;
    crs::CoordinateSystem display;
    bool found = false;
    app_.read_document([&](const Document& document) {
        const Chunk* chunk = document.active_chunk();
        if (chunk == nullptr || chunk->id() != chunk_id_) {
            return;
        }
        found = true;
        display = chunk->coordinate_system();
        slot = camera ? chunk->camera_crs() : chunk->marker_crs();
    });
    if (!found) {
        return false;
    }
    const crs::CoordinateSystem slot_cs = slot;
    if (display.is_set() && !display.is_local() && slot_cs.is_set() &&
        !slot_cs.is_local() && !(slot_cs == display)) {
        try {
            const crs::CrsTransform xform = crs::CrsTransform::create(display, slot_cs);
            const crs::CrsPoint point =
                xform.forward(crs::CrsPoint{location.x(), location.y(), location.z()});
            location = Eigen::Vector3d(point.x, point.y, point.z);
        } catch (const std::exception&) {
            return false;  // no conversion: refuse rather than write the wrong units
        }
    }
    if (!location.allFinite()) {
        return false;
    }
    if (camera) {
        app_.execute(make_set_camera_reference_command(chunk_id_, item.id, location,
                                                      item.sigma_xyz));
    } else {
        app_.execute(make_set_marker_reference_command(chunk_id_, item.id, location,
                                                      item.sigma_xyz));
    }
    return true;
}

}  // namespace alphashape::gui
