#include "apps/gui/transform_dialog.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <memory>

#include <Eigen/Geometry>

#include <QDialogButtonBox>
#include <QDoubleValidator>
#include <QFormLayout>
#include <QLabel>
#include <QLineEdit>
#include <QLocale>
#include <QVBoxLayout>

#include "apps/gui/app.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"

namespace alphashape::gui {

namespace {

Eigen::Matrix3d rotation_from_degrees(double yaw_degrees, double pitch_degrees,
                                      double roll_degrees) {
    const double to_radians = 3.14159265358979323846 / 180.0;
    const double y = yaw_degrees * to_radians;
    const double p = pitch_degrees * to_radians;
    const double r = roll_degrees * to_radians;
    const Eigen::Matrix3d rz = Eigen::AngleAxisd(y, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Matrix3d ry = Eigen::AngleAxisd(p, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Matrix3d rx = Eigen::AngleAxisd(r, Eigen::Vector3d::UnitX()).toRotationMatrix();
    return rz * ry * rx;
}

QString number_text(double value) {
    return QLocale::c().toString(value, 'f', 9);
}

QLineEdit* make_edit(const QString& object_name, QWidget* parent) {
    auto* edit = new QLineEdit(parent);
    edit->setObjectName(object_name);
    edit->setAlignment(Qt::AlignRight);
    auto* validator =
        new QDoubleValidator(-1.0e18, 1.0e18, 9, edit);
    validator->setLocale(QLocale::c());
    validator->setNotation(QDoubleValidator::StandardNotation);
    edit->setValidator(validator);
    return edit;
}

const Chunk* find_chunk(const Document& document, std::int32_t chunk_id) {
    for (const std::shared_ptr<Chunk>& chunk : document.workspace().chunks()) {
        if (chunk != nullptr && chunk->id() == chunk_id) {
            return chunk.get();
        }
    }
    return nullptr;
}

}  // namespace

Transform transform_from_values(const TransformValues& values) {
    Transform transform = Transform::Identity();
    const Eigen::Matrix3d rotation =
        rotation_from_degrees(values.yaw_degrees, values.pitch_degrees, values.roll_degrees);
    transform.topLeftCorner<3, 3>() = rotation * values.scale;
    transform.topRightCorner<3, 1>() = values.centre;
    return transform;
}

TransformValues transform_values(const Transform& transform) {
    TransformValues values;
    values.centre = transform.topRightCorner<3, 1>();
    const Eigen::Matrix3d linear = transform.topLeftCorner<3, 3>();
    const double sx = linear.col(0).norm();
    const double sy = linear.col(1).norm();
    const double sz = linear.col(2).norm();
    values.scale = (sx + sy + sz) / 3.0;
    if (!(values.scale > 0.0)) {
        values.scale = 1.0;
        return values;
    }
    Eigen::Matrix3d rotation = linear / values.scale;
    const double sin_pitch = std::clamp(-rotation(2, 0), -1.0, 1.0);
    const double pitch = std::asin(sin_pitch);
    double yaw = 0.0;
    double roll = 0.0;
    if (std::abs(rotation(2, 0)) < 1.0 - 1e-12) {
        yaw = std::atan2(rotation(1, 0), rotation(0, 0));
        roll = std::atan2(rotation(2, 1), rotation(2, 2));
    } else {
        yaw = std::atan2(-rotation(0, 1), rotation(1, 1));
    }
    const double to_degrees = 180.0 / 3.14159265358979323846;
    values.yaw_degrees = yaw * to_degrees;
    values.pitch_degrees = pitch * to_degrees;
    values.roll_degrees = roll * to_degrees;
    return values;
}

TransformDialog::TransformDialog(GuiApp& app, std::int32_t chunk_id, QWidget* parent)
    : QDialog(parent), app_(app), chunk_id_(chunk_id) {
    setObjectName(QStringLiteral("transformDialog"));
    setWindowTitle(tr("Transform Object"));

    centre_x_ = make_edit(QStringLiteral("transform.centre.x"), this);
    centre_y_ = make_edit(QStringLiteral("transform.centre.y"), this);
    centre_z_ = make_edit(QStringLiteral("transform.centre.z"), this);
    scale_ = make_edit(QStringLiteral("transform.scale"), this);
    yaw_ = make_edit(QStringLiteral("transform.yaw"), this);
    pitch_ = make_edit(QStringLiteral("transform.pitch"), this);
    roll_ = make_edit(QStringLiteral("transform.roll"), this);

    auto* form = new QFormLayout;
    form->addRow(tr("Centre X"), centre_x_);
    form->addRow(tr("Centre Y"), centre_y_);
    form->addRow(tr("Centre Z"), centre_z_);
    form->addRow(tr("Scale"), scale_);
    form->addRow(tr("Yaw (deg)"), yaw_);
    form->addRow(tr("Pitch (deg)"), pitch_);
    form->addRow(tr("Roll (deg)"), roll_);

    auto* hint = new QLabel(
        tr("Rotation order: yaw about Z, then pitch about Y, then roll about X."), this);
    hint->setWordWrap(true);

    auto* buttons = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel | QDialogButtonBox::Apply |
            QDialogButtonBox::Reset,
        this);
    buttons->setObjectName(QStringLiteral("transform.buttons"));

    auto* layout = new QVBoxLayout(this);
    layout->addLayout(form);
    layout->addWidget(hint);
    layout->addWidget(buttons);

    connect(buttons, &QDialogButtonBox::clicked, this, [this, buttons](QAbstractButton* button) {
        switch (buttons->standardButton(button)) {
            case QDialogButtonBox::Apply:
                static_cast<void>(apply());
                break;
            case QDialogButtonBox::Reset:
                static_cast<void>(reset_transform());
                break;
            case QDialogButtonBox::Ok:
                if (apply()) {
                    accept();
                }
                break;
            default:
                reject();
                break;
        }
    });

    load_from_chunk();
}

void TransformDialog::load_from_chunk() {
    Transform transform = Transform::Identity();
    static_cast<void>(app_.read_document([&](const Document& document) {
        const Chunk* chunk = find_chunk(document, chunk_id_);
        if (chunk != nullptr) {
            transform = chunk->transform();
        }
    }));
    set_values(transform_values(transform));
}

QLineEdit* TransformDialog::field(const QString& name) const {
    if (name == QLatin1String("centre.x")) {
        return centre_x_;
    }
    if (name == QLatin1String("centre.y")) {
        return centre_y_;
    }
    if (name == QLatin1String("centre.z")) {
        return centre_z_;
    }
    if (name == QLatin1String("scale")) {
        return scale_;
    }
    if (name == QLatin1String("yaw")) {
        return yaw_;
    }
    if (name == QLatin1String("pitch")) {
        return pitch_;
    }
    if (name == QLatin1String("roll")) {
        return roll_;
    }
    return nullptr;
}

void TransformDialog::set_values(const TransformValues& values) {
    centre_x_->setText(number_text(values.centre.x()));
    centre_y_->setText(number_text(values.centre.y()));
    centre_z_->setText(number_text(values.centre.z()));
    scale_->setText(number_text(values.scale));
    yaw_->setText(number_text(values.yaw_degrees));
    pitch_->setText(number_text(values.pitch_degrees));
    roll_->setText(number_text(values.roll_degrees));
}

bool TransformDialog::read_field(const QLineEdit* edit, double& value) const {
    bool ok = false;
    const QString text = edit->text().trimmed();
    value = QLocale::c().toDouble(text, &ok);
    if (!ok || !std::isfinite(value)) {
        const_cast<QLineEdit*>(edit)->setStyleSheet(
            QStringLiteral("border: 1px solid #d9534f;"));
        return false;
    }
    const_cast<QLineEdit*>(edit)->setStyleSheet(QString());
    return true;
}

bool TransformDialog::read_values(TransformValues& values) const {
    bool valid = true;
    valid = read_field(centre_x_, values.centre.x()) && valid;
    valid = read_field(centre_y_, values.centre.y()) && valid;
    valid = read_field(centre_z_, values.centre.z()) && valid;
    valid = read_field(scale_, values.scale) && valid;
    valid = read_field(yaw_, values.yaw_degrees) && valid;
    valid = read_field(pitch_, values.pitch_degrees) && valid;
    valid = read_field(roll_, values.roll_degrees) && valid;
    return valid;
}

TransformValues TransformDialog::values_from_fields() const {
    TransformValues values;
    static_cast<void>(read_values(values));
    return values;
}

Transform TransformDialog::transform_from_fields() const {
    return transform_from_values(values_from_fields());
}

bool TransformDialog::push_transform(const Transform& transform) {
    Region region;
    bool found = false;
    static_cast<void>(app_.read_document([&](const Document& document) {
        const Chunk* chunk = find_chunk(document, chunk_id_);
        if (chunk != nullptr) {
            region = chunk->region();
            found = true;
        }
    }));
    if (!found) {
        return false;
    }
    try {
        app_.execute(make_set_chunk_transform_command(chunk_id_, transform, region));
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

bool TransformDialog::apply() {
    TransformValues values;
    if (!read_values(values)) {
        return false;
    }
    return push_transform(transform_from_values(values));
}

bool TransformDialog::reset_transform() {
    TransformValues values;
    set_values(values);
    return push_transform(Transform::Identity());
}

