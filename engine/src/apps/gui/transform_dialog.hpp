#pragma once

// TransformDialog: numeric object move/scale/rotate (WP-14.15, GUI-20).
//
// Six values in the chunk's local frame: centre X/Y/Z, uniform scale, and yaw/pitch/roll in
// degrees (R = Rz(yaw) * Ry(pitch) * Rx(roll), applied before the translation). The fields use
// QLocale::c() so a decimal point is always ".", whatever the system locale is.
//
// Every edit is one undoable make_set_chunk_transform_command; the dialog never writes the
// chunk directly. "Reset" restores identity (no rotation, unit scale, zero translation) and
// leaves the region alone; because it is a normal command, undo brings the previous transform
// back.

#include <cstdint>

#include <Eigen/Core>
#include <QDialog>
#include <QString>

#include "common/core/types.hpp"

class QLineEdit;

namespace alphashape::gui {

class GuiApp;

/// The six editable values.
struct TransformValues {
    Eigen::Vector3d centre = Eigen::Vector3d::Zero();
    double scale = 1.0;
    double yaw_degrees = 0.0;
    double pitch_degrees = 0.0;
    double roll_degrees = 0.0;
};

/// Build a transform from the six values (degrees).
Transform transform_from_values(const TransformValues& values);

/// Decompose a similarity transform into the six values. A non-uniform or degenerate matrix is
/// reported with the mean column length as its scale.
TransformValues transform_values(const Transform& transform);

class TransformDialog : public QDialog {
    Q_OBJECT

public:
    TransformDialog(GuiApp& app, std::int32_t chunk_id, QWidget* parent = nullptr);

    /// Load the chunk's current transform into the fields.
    void load_from_chunk();

    /// The transform the fields describe. Call only after a validation pass; unparsable fields
    /// keep their last valid value.
    Transform transform_from_fields() const;
    TransformValues values_from_fields() const;
    void set_values(const TransformValues& values);

    /// Read the fields, validate them, and push one command. False when a field is not a finite
    /// number or the chunk no longer exists (nothing is pushed then).
    bool apply();

    /// Put identity in the fields and push one identity command (the region is untouched).
    bool reset_transform();

    /// Field editors, by name: "centre.x", "centre.y", "centre.z", "scale", "yaw", "pitch",
    /// "roll". Exposed for the offscreen tests.
    QLineEdit* field(const QString& name) const;

private:
    /// Parse one editor; false and a red border when it is not a finite number.
    bool read_field(const QLineEdit* edit, double& value) const;
    bool read_values(TransformValues& values) const;
    /// Push `transform` with the chunk's current region as one command.
    bool push_transform(const Transform& transform);

    GuiApp& app_;
    std::int32_t chunk_id_ = -1;
    QLineEdit* centre_x_ = nullptr;
    QLineEdit* centre_y_ = nullptr;
    QLineEdit* centre_z_ = nullptr;
    QLineEdit* scale_ = nullptr;
    QLineEdit* yaw_ = nullptr;
    QLineEdit* pitch_ = nullptr;
    QLineEdit* roll_ = nullptr;
};

}  // namespace alphashape::gui
