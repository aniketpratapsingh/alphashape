#pragma once

// Marker tool dialogs (WP-14.22b): Detect Markers with a live target preview,
// and the marker-sheet printer. Detect Fiducials uses the plain TaskDialog.
//
// DetectMarkersDialog is a TaskDialog with a preview of the selected target
// family, so the fields still come from the engine schema rather than a copy
// of it. PrintMarkersDialog builds a SheetOptions from its fields and renders
// a live page preview; it never mutates the document, and an invalid layout
// becomes inline text instead of an exception.

#include <optional>
#include <string>
#include <vector>

#include <QDialog>
#include <QImage>
#include <QString>

#include "apps/gui/task_dialog.hpp"
#include "common/algorithm/markers/marker_sheet.hpp"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSpinBox;

namespace alphashape::gui {

/// DetectMarkers' parameters (built by TaskDialog from the task schema) plus a
/// preview of the selected target. A family with no printable sheet shows a
/// text placeholder instead.
class DetectMarkersDialog : public TaskDialog {
    Q_OBJECT

public:
    DetectMarkersDialog(const TaskRegistry& registry, QSettings& settings,
                        QWidget* parent = nullptr);

    QLabel* preview_label() const { return preview_; }
    /// The image last drawn into the preview; null while a placeholder shows.
    const QImage& preview_image() const { return preview_image_; }

private:
    void refresh_preview();

    QLabel* preview_ = nullptr;
    QImage preview_image_;
};

/// Print Markers: a marker-sheet layout (no document mutation) with a live
/// first-page preview. The PDF path comes from an injectable hook, and write
/// failures are reported through the modal error hook.
class PrintMarkersDialog : public QDialog {
    Q_OBJECT

public:
    explicit PrintMarkersDialog(QWidget* parent = nullptr);

    /// The layout the fields currently describe.
    SheetOptions options() const;
    /// Pages the current layout needs; 0 while the layout is invalid.
    int page_count() const { return page_count_; }
    bool layout_valid() const { return valid_; }

    QComboBox* target_editor() const { return target_; }
    QDoubleSpinBox* target_size_editor() const { return target_size_; }
    QDoubleSpinBox* quiet_zone_editor() const { return quiet_zone_; }
    QSpinBox* first_code_editor() const { return first_code_; }
    QSpinBox* count_editor() const { return count_; }
    QCheckBox* corner_marks_editor() const { return corner_marks_; }
    QCheckBox* labels_editor() const { return labels_; }
    QSpinBox* board_cols_editor() const { return board_cols_; }
    QSpinBox* board_rows_editor() const { return board_rows_; }
    QDoubleSpinBox* square_size_editor() const { return square_size_; }
    QLabel* page_count_label() const { return page_label_; }
    QLabel* preview_label() const { return preview_; }
    QLabel* error_label() const { return error_; }
    QPushButton* save_button() const { return save_; }
    const QImage& preview_image() const { return preview_image_; }

    /// Recompute the layout and redraw the preview. Never throws.
    void refresh();
    /// Ask for a path through the hook and write the PDF. Never throws.
    void save_pdf();

private:
    QWidget* build_fields();
    void sync_chessboard_fields();
    /// The target the combo currently selects.
    SheetTarget selected_target() const;

    QComboBox* target_ = nullptr;
    QComboBox* paper_ = nullptr;
    QDoubleSpinBox* target_size_ = nullptr;
    QDoubleSpinBox* quiet_zone_ = nullptr;
    QSpinBox* first_code_ = nullptr;
    QSpinBox* count_ = nullptr;
    QCheckBox* corner_marks_ = nullptr;
    QCheckBox* labels_ = nullptr;
    QSpinBox* board_cols_ = nullptr;
    QSpinBox* board_rows_ = nullptr;
    QDoubleSpinBox* square_size_ = nullptr;
    QLabel* page_label_ = nullptr;
    QLabel* preview_ = nullptr;
    QLabel* error_ = nullptr;
    QPushButton* save_ = nullptr;
    QImage preview_image_;
    int page_count_ = 0;
    bool valid_ = false;
};

}  // namespace alphashape::gui
