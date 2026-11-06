#pragma once

// CalibrationDialog: camera calibration for one sensor (WP-14.22a). Sensor
// list on the left, Initial / Adjusted / GNSS-INS tabs on the right. Every
// edit is one undoable command from the model command layer, so the dialog
// never writes a sensor directly.

#include <cstdint>

#include <QDialog>
#include <QHash>
#include <QString>

#include "common/core/types.hpp"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QListWidget;
class QPushButton;
class QTabWidget;
class QTableWidget;
class QWidget;

namespace alphashape::gui {

class GuiApp;

class CalibrationDialog : public QDialog {
    Q_OBJECT

public:
    explicit CalibrationDialog(GuiApp& app, QWidget* parent = nullptr);

    /// Point the dialog at a chunk. The first sensor is selected.
    void set_chunk(std::int32_t chunk_id);
    void select_sensor(int index);
    int sensor_index() const { return sensor_index_; }

    QListWidget* sensor_list() { return sensor_list_; }
    QTabWidget* tabs() { return tabs_; }
    QDoubleSpinBox* initial_editor(const QString& param) const;
    QCheckBox* fit_check(const QString& param) const;
    QLabel* adjusted_value(const QString& param) const;
    QTableWidget* correlation_table() { return correlation_; }

    /// Re-read the selected sensor from the document and reload every tab.
    void refresh();

private:
    void build_sensor_list();
    QWidget* build_initial_tab();
    QWidget* build_adjusted_tab();
    QWidget* build_gnss_tab();
    void load_sensor(int index);
    void update_adjusted(const Sensor& sensor);
    void commit_calibration();
    void commit_fixed_params();
    void commit_physical();
    void commit_antenna();
    void run_optimize();
    Sensor read_sensor(std::int32_t& chunk_id) const;

    GuiApp& app_;
    std::int32_t chunk_id_ = -1;
    int sensor_index_ = -1;
    Sensor sensor_;
    QListWidget* sensor_list_ = nullptr;
    QTabWidget* tabs_ = nullptr;
    QComboBox* type_ = nullptr;
    QDoubleSpinBox* pixel_size_ = nullptr;
    QDoubleSpinBox* focal_length_ = nullptr;
    QDoubleSpinBox* antenna_offset_[3] = {nullptr, nullptr, nullptr};
    QCheckBox* boresight_enabled_ = nullptr;
    QDoubleSpinBox* boresight_[3] = {nullptr, nullptr, nullptr};
    QPushButton* optimize_ = nullptr;
    QTableWidget* correlation_ = nullptr;
    QHash<QString, QDoubleSpinBox*> initial_;
    QHash<QString, QCheckBox*> fit_;
    QHash<QString, QLabel*> adjusted_;
    QHash<QString, QLabel*> sigma_;
};

}  // namespace alphashape::gui
