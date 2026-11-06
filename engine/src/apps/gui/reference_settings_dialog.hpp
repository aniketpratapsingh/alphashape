#pragma once

// ReferenceSettingsDialog: the chunk's reference settings (WP-14.22a). CRS
// pickers for the camera, marker and world slots; the default accuracies; the
// control-point toggle. OK applies every change as one undoable macro. Only
// fields the engine actually has are shown.

#include <cstdint>
#include <optional>

#include <QDialog>
#include <QString>

#include "common/core/reference_settings.hpp"
#include "common/core/types.hpp"

class QCheckBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;

namespace alphashape::gui {

class GuiApp;

class ReferenceSettingsDialog : public QDialog {
    Q_OBJECT

public:
    explicit ReferenceSettingsDialog(GuiApp& app, QWidget* parent = nullptr);

    /// The editor for an accuracy key: camera_location, camera_rotation,
    /// marker_location, marker_projection, scalebar or tiepoint.
    QDoubleSpinBox* accuracy_editor(const QString& key) const;

    /// Set one CRS slot (0 camera, 1 marker, 2 world) as if the picker had
    /// returned it. The change is only applied by accept(), like a UI edit.
    void set_crs_slot(int slot, std::optional<CRS> value);

public Q_SLOTS:
    void accept() override;

private:
    void build_crs_rows();
    void build_accuracy_rows();
    void refresh_crs_labels();
    void choose_crs(int slot);
    std::optional<CRS>& slot_ref(int slot);
    ReferenceSettings settings_from_editors() const;

    GuiApp& app_;
    std::int32_t chunk_id_ = -1;
    CRS chunk_crs_;
    ReferenceSettings initial_settings_;
    std::optional<CRS> initial_camera_crs_;
    std::optional<CRS> initial_marker_crs_;
    std::optional<CRS> initial_world_crs_;
    std::optional<CRS> camera_crs_;
    std::optional<CRS> marker_crs_;
    std::optional<CRS> world_crs_;
    QCheckBox* follow_camera_ = nullptr;
    QCheckBox* follow_marker_ = nullptr;
    QCheckBox* follow_world_ = nullptr;
    QLabel* camera_label_ = nullptr;
    QLabel* marker_label_ = nullptr;
    QLabel* world_label_ = nullptr;
    QPushButton* camera_button_ = nullptr;
    QPushButton* marker_button_ = nullptr;
    QPushButton* world_button_ = nullptr;
    QDoubleSpinBox* camera_location_ = nullptr;
    QDoubleSpinBox* camera_rotation_ = nullptr;
    QDoubleSpinBox* marker_location_ = nullptr;
    QDoubleSpinBox* marker_projection_ = nullptr;
    QDoubleSpinBox* scalebar_ = nullptr;
    QDoubleSpinBox* tiepoint_ = nullptr;
    QCheckBox* use_gcp_ = nullptr;
};

}  // namespace alphashape::gui
