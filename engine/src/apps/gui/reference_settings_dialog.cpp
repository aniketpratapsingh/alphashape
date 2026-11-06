#include "apps/gui/reference_settings_dialog.hpp"

#include <exception>
#include <string>

#include <QCheckBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>

#include "apps/gui/app.hpp"
#include "apps/gui/crs_picker.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"

namespace alphashape::gui {
namespace {

/// A modal wrapper around CrsPicker. OK stays disabled while the picker says
/// the current input is not a usable definition.
class PickerDialog : public QDialog {
public:
    PickerDialog(QWidget* parent, const crs::CoordinateSystem& current)
        : QDialog(parent) {
        setObjectName(QStringLiteral("crs.dialog"));
        setWindowTitle(tr("Select Coordinate System"));
        auto* layout = new QVBoxLayout(this);
        picker_ = new CrsPicker(this);
        picker_->set_coordinate_system(current);
        layout->addWidget(picker_, 1);
        buttons_ = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
                                        this);
        layout->addWidget(buttons_);
        connect(buttons_, &QDialogButtonBox::accepted, this, &QDialog::accept);
        connect(buttons_, &QDialogButtonBox::rejected, this, &QDialog::reject);
        connect(picker_, &CrsPicker::selection_changed, this,
                [this]() { update_ok(); });
        update_ok();
        resize(560, 480);
    }

    crs::CoordinateSystem chosen() const { return picker_->coordinate_system(); }

private:
    void update_ok() {
        buttons_->button(QDialogButtonBox::Ok)->setEnabled(picker_->is_valid());
    }

    CrsPicker* picker_ = nullptr;
    QDialogButtonBox* buttons_ = nullptr;
};

}  // namespace

ReferenceSettingsDialog::ReferenceSettingsDialog(GuiApp& app, QWidget* parent)
    : QDialog(parent), app_(app) {
    setObjectName(QStringLiteral("dialog.reference_settings"));
    setWindowTitle(tr("Reference Settings"));
    auto* layout = new QVBoxLayout(this);
    app_.read_document([this](const Document& document) {
        const Chunk* chunk = document.active_chunk();
        if (chunk == nullptr) {
            return;
        }
        chunk_id_ = chunk->id();
        chunk_crs_ = chunk->crs();
        initial_settings_ = chunk->reference_settings();
        initial_camera_crs_ = chunk->explicit_camera_crs();
        initial_marker_crs_ = chunk->explicit_marker_crs();
        initial_world_crs_ = chunk->explicit_world_crs();
    });
    camera_crs_ = initial_camera_crs_;
    marker_crs_ = initial_marker_crs_;
    world_crs_ = initial_world_crs_;
    build_crs_rows();
    build_accuracy_rows();
    use_gcp_ = new QCheckBox(tr("Use markers as control points"), this);
    use_gcp_->setObjectName(QStringLiteral("reference.use_gcp"));
    use_gcp_->setChecked(initial_settings_.use_gcp);
    layout->addWidget(use_gcp_);
    auto* buttons =
        new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    layout->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::accepted, this,
            &ReferenceSettingsDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    if (chunk_id_ < 0) {
        buttons->button(QDialogButtonBox::Ok)->setEnabled(false);
    }
    resize(560, 540);
}

void ReferenceSettingsDialog::build_crs_rows() {
    auto* group = new QGroupBox(tr("Coordinate systems"), this);
    auto* grid = new QGridLayout(group);
    const auto add_row = [this, grid, group](
                             int row, const QString& key, const QString& title,
                             std::optional<CRS>& slot, QCheckBox*& follow,
                             QLabel*& label, QPushButton*& button) {
        label = new QLabel(group);
        label->setObjectName(QStringLiteral("reference.crs.") + key +
                             QStringLiteral("_name"));
        grid->addWidget(new QLabel(title, group), row, 0);
        grid->addWidget(label, row, 1);
        follow = new QCheckBox(tr("Follow chunk CRS"), group);
        follow->setObjectName(QStringLiteral("reference.crs.") + key +
                              QStringLiteral("_follow"));
        follow->setChecked(!slot.has_value());
        grid->addWidget(follow, row, 2);
        button = new QPushButton(tr("Set..."), group);
        button->setObjectName(QStringLiteral("reference.crs.") + key +
                              QStringLiteral("_browse"));
        button->setEnabled(!follow->isChecked());
        grid->addWidget(button, row, 3);
        const int index = row;
        connect(follow, &QCheckBox::toggled, this, [this, index, button](bool on) {
            if (on) {
                slot_ref(index).reset();
            }
            button->setEnabled(!on);
            refresh_crs_labels();
        });
        connect(button, &QPushButton::clicked, this,
                [this, index]() { choose_crs(index); });
    };
    add_row(0, QStringLiteral("camera"), tr("Camera:"), camera_crs_, follow_camera_,
            camera_label_, camera_button_);
    add_row(1, QStringLiteral("marker"), tr("Marker:"), marker_crs_, follow_marker_,
            marker_label_, marker_button_);
    add_row(2, QStringLiteral("world"), tr("World:"), world_crs_, follow_world_,
            world_label_, world_button_);
    qobject_cast<QVBoxLayout*>(layout())->addWidget(group);
    refresh_crs_labels();
}

void ReferenceSettingsDialog::refresh_crs_labels() {
    const auto text = [this](const std::optional<CRS>& slot) {
        const CRS& value = slot.has_value() ? *slot : chunk_crs_;
        if (value.epsg > 0) {
            return QStringLiteral("EPSG:%1").arg(value.epsg);
        }
        if (value.is_set()) {
            return tr("Custom WKT");
        }
        return tr("Not set");
    };
    camera_label_->setText(text(camera_crs_));
    marker_label_->setText(text(marker_crs_));
    world_label_->setText(text(world_crs_));
}

std::optional<CRS>& ReferenceSettingsDialog::slot_ref(int slot) {
    return slot == 0 ? camera_crs_ : (slot == 1 ? marker_crs_ : world_crs_);
}

void ReferenceSettingsDialog::choose_crs(int slot) {
    std::optional<CRS>& target = slot_ref(slot);
    const CRS& base = target.has_value() ? *target : chunk_crs_;
    crs::CoordinateSystem current;
    try {
        current = crs::CoordinateSystem(base);
    } catch (const std::exception&) {
        current = crs::CoordinateSystem::local();
    }
    PickerDialog dialog(this, current);
    if (dialog.exec() != QDialog::Accepted) {
        return;
    }
    try {
        const crs::CoordinateSystem chosen = dialog.chosen();
        if (chosen.is_local() || !chosen.is_set()) {
            target.reset();
        } else {
            target = CRS(chosen);
        }
    } catch (const std::exception& error) {
        QMessageBox::critical(this, tr("Coordinate System"),
                              QString::fromUtf8(error.what()));
        return;
    }
    refresh_crs_labels();
}

void ReferenceSettingsDialog::build_accuracy_rows() {
    auto* group = new QGroupBox(tr("Default accuracies"), this);
    auto* form = new QFormLayout(group);
    const auto add = [form, group](const QString& key, const QString& title,
                                   double value, double maximum, int decimals,
                                   QDoubleSpinBox*& editor) {
        editor = new QDoubleSpinBox(group);
        editor->setObjectName(QStringLiteral("reference.") + key);
        editor->setRange(1e-9, maximum);
        editor->setDecimals(decimals);
        editor->setValue(value);
        form->addRow(title, editor);
    };
    add(QStringLiteral("camera_location"), tr("Camera position (m):"),
        initial_settings_.camera_location_accuracy_m, 1e6, 6, camera_location_);
    add(QStringLiteral("camera_rotation"), tr("Camera rotation (deg):"),
        initial_settings_.camera_rotation_accuracy_deg, 360.0, 6, camera_rotation_);
    add(QStringLiteral("marker_location"), tr("Marker position (m):"),
        initial_settings_.marker_location_accuracy_m, 1e6, 6, marker_location_);
    add(QStringLiteral("marker_projection"), tr("Marker projection (px):"),
        initial_settings_.marker_projection_accuracy_px, 1e6, 6, marker_projection_);
    add(QStringLiteral("scalebar"), tr("Scale bar length (m):"),
        initial_settings_.scalebar_accuracy_m, 1e6, 6, scalebar_);
    add(QStringLiteral("tiepoint"), tr("Tie point (px):"),
        initial_settings_.tiepoint_accuracy_px, 1e6, 6, tiepoint_);
    qobject_cast<QVBoxLayout*>(layout())->addWidget(group);
}

QDoubleSpinBox* ReferenceSettingsDialog::accuracy_editor(const QString& key) const {
    if (key == QStringLiteral("camera_location")) {
        return camera_location_;
    }
    if (key == QStringLiteral("camera_rotation")) {
        return camera_rotation_;
    }
    if (key == QStringLiteral("marker_location")) {
        return marker_location_;
    }
    if (key == QStringLiteral("marker_projection")) {
        return marker_projection_;
    }
    if (key == QStringLiteral("scalebar")) {
        return scalebar_;
    }
    if (key == QStringLiteral("tiepoint")) {
        return tiepoint_;
    }
    return nullptr;
}

ReferenceSettings ReferenceSettingsDialog::settings_from_editors() const {
    ReferenceSettings settings;
    settings.camera_location_accuracy_m = camera_location_->value();
    settings.camera_rotation_accuracy_deg = camera_rotation_->value();
    settings.marker_location_accuracy_m = marker_location_->value();
    settings.marker_projection_accuracy_px = marker_projection_->value();
    settings.scalebar_accuracy_m = scalebar_->value();
    settings.tiepoint_accuracy_px = tiepoint_->value();
    settings.use_gcp = use_gcp_->isChecked();
    return settings;
}

void ReferenceSettingsDialog::accept() {
    if (chunk_id_ < 0) {
        return;
    }
    const ReferenceSettings settings = settings_from_editors();
    const bool camera_changed = camera_crs_ != initial_camera_crs_;
    const bool marker_changed = marker_crs_ != initial_marker_crs_;
    const bool world_changed = world_crs_ != initial_world_crs_;
    if (!camera_changed && !marker_changed && !world_changed &&
        settings == initial_settings_) {
        QDialog::accept();
        return;
    }
    app_.begin_macro("Reference Settings");
    bool applied = true;
    try {
        if (camera_changed) {
            app_.execute(make_set_chunk_crs_slot_command(chunk_id_, CrsSlot::Camera,
                                                         camera_crs_));
        }
        if (marker_changed) {
            app_.execute(make_set_chunk_crs_slot_command(chunk_id_, CrsSlot::Marker,
                                                         marker_crs_));
        }
        if (world_changed) {
            app_.execute(make_set_chunk_crs_slot_command(chunk_id_, CrsSlot::World,
                                                         world_crs_));
        }
        if (settings != initial_settings_) {
            app_.execute(
                make_set_reference_settings_command(chunk_id_, settings));
        }
    } catch (const std::exception& error) {
        applied = false;
        QMessageBox::critical(this, tr("Reference Settings"),
                              QString::fromUtf8(error.what()));
    }
    app_.end_macro();
    if (applied) {
        QDialog::accept();
    }
}

}  // namespace alphashape::gui



