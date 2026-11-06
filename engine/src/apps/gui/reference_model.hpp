#pragma once

// ReferenceModel: one section of the active chunk's reference table (the
// cameras, or the markers and check points) as a flat table.
//
// The model holds no pointers into the engine Document. Each refresh rebuilds
// the rows from reference_table.cpp, keyed by camera / marker id, so a
// replaced or edited document is picked up whole. Rows are sorted by
// descending |error|, with rows that have no error last.
//
// Every cell edit goes through the command layer, one command per edit:
// source coordinates and accuracy through the camera / marker reference
// factory, the enabled checkbox through the camera / marker flag factory, and
// pinned through the marker pinned factory. The undo stack therefore restores
// exactly the value the user changed.
//
// Concurrency: the model never reads the document while a job owns it. It
// marks itself stale and rebuilds when the job finishes.

#include <cstdint>
#include <optional>
#include <vector>

#include <QAbstractTableModel>
#include <QString>
#include <QVariant>

#include "common/algorithm/reference/reference_table.hpp"

class QModelIndex;

namespace alphashape::gui {

class GuiApp;

class ReferenceModel : public QAbstractTableModel {
    Q_OBJECT

public:
    enum class Section { Cameras, Markers };

    enum Column {
        LabelColumn = 0,
        SourceXColumn,
        SourceYColumn,
        SourceZColumn,
        AccuracyColumn,
        EstimatedXColumn,
        EstimatedYColumn,
        EstimatedZColumn,
        ErrorColumn,
        ProjectionsColumn,
        PinnedColumn,
        ColumnCount
    };

    ReferenceModel(GuiApp& app, Section section, QObject* parent = nullptr);

    /// Rebuild every row from the active chunk. A no-op while a job is running.
    void refresh();

    Section section() const { return section_; }
    const reference::ReferenceSummary& summary() const { return summary_; }
    /// The row at `row`, or a default row when out of range.
    const reference::ReferenceRow& row(int row) const;
    std::optional<double> error_m(int row) const;
    std::int32_t row_id(int row) const;
    std::int32_t chunk_id() const { return chunk_id_; }
    /// True when a refresh was requested during a job and is waiting for it.
    bool stale() const { return stale_; }

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation orientation,
                        int role = Qt::DisplayRole) const override;
    Qt::ItemFlags flags(const QModelIndex& index) const override;
    bool setData(const QModelIndex& index, const QVariant& value,
                 int role = Qt::EditRole) override;

Q_SIGNALS:
    /// Emitted after every rebuild (a document-change signal or a finished job).
    void refreshed();

private:
    bool can_edit() const;
    void refresh_after_job();
    /// True when `column` belongs to this section (markers add Pinned).
    bool has_column(int column) const;
    /// Axis 0..2 when `column` is a source component.
    static bool source_axis(int column, int* axis);

    GuiApp& app_;
    Section section_;
    std::vector<reference::ReferenceRow> rows_;
    reference::ReferenceSummary summary_;
    std::int32_t chunk_id_ = -1;
    /// The display CRS is geographic, so source / estimated values are degrees.
    bool geographic_ = false;
    bool stale_ = false;
};

}  // namespace alphashape::gui
