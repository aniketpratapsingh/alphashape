#pragma once

// CrsPicker: chooses a coordinate reference system from the engine's curated
// built-in list, or from a pasted WKT definition. The widget reports validity
// through is_valid(); a containing dialog disables its OK button when false.
// Everything is gated on the PROJ-backed engine (alphashape_crs).

#include <optional>

#include <QString>
#include <QWidget>

#include "model/crs/coordinate_system.hpp"

class QCheckBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPlainTextEdit;
class QTabWidget;

namespace alphashape::gui {

class CrsPicker : public QWidget {
    Q_OBJECT

public:
    explicit CrsPicker(QWidget* parent = nullptr);

    /// Select `cs` (or the LOCAL entry) without emitting selection_changed.
    void set_coordinate_system(const crs::CoordinateSystem& cs);
    bool is_valid() const;
    QString error() const;
    /// The chosen CRS; LOCAL when nothing valid is selected.
    crs::CoordinateSystem coordinate_system() const;

    QLineEdit* filter_edit() { return filter_; }
    QLineEdit* epsg_edit() { return epsg_; }
    QListWidget* list_widget() { return list_; }
    QPlainTextEdit* wkt_edit() { return wkt_; }
    QLabel* error_label() { return error_; }
    QTabWidget* tabs() { return tabs_; }

Q_SIGNALS:
    void selection_changed();

private:
    void build_search_tab();
    void build_wkt_tab();
    void apply_filter(const QString& text);
    void refresh_validity();
    int selected_epsg() const;
    /// The code typed into the EPSG field, or -1 when it is empty/not a number.
    int typed_epsg() const;

    QTabWidget* tabs_ = nullptr;
    QLineEdit* filter_ = nullptr;
    QLineEdit* epsg_ = nullptr;
    QListWidget* list_ = nullptr;
    QPlainTextEdit* wkt_ = nullptr;
    QLabel* error_ = nullptr;
    QCheckBox* vertical_ = nullptr;
    QLineEdit* vertical_name_ = nullptr;
    /// The vertical component restored by set_coordinate_system, kept verbatim
    /// (grid, offset, reference frame) so a round trip loses nothing.
    std::optional<crs::VerticalCrs> vertical_state_;
    QString error_text_;
    bool valid_ = false;
};

}  // namespace alphashape::gui
