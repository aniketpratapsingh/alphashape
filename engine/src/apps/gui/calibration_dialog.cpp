#include "apps/gui/calibration_dialog.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <QAbstractItemView>
#include <QCheckBox>
#include <QColor>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSplitter>
#include <QStringList>
#include <QTabWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>
#include <QWidget>

#include "apps/gui/app.hpp"
#include "apps/gui/job_manager.hpp"
#include "apps/gui/task_dialog.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/tasks/registry.hpp"

namespace alphashape::gui {
namespace {

QString sensor_text(const Sensor& sensor) {
    QString text = QString::fromStdString(sensor.label);
    if (text.isEmpty()) {
        text = QObject::tr("Sensor");
    }
    if (sensor.width > 0 && sensor.height > 0) {
        text += QStringLiteral(" (%1 x %2)")
                    .arg(sensor.width)
                    .arg(sensor.height);
    }
    return text;
}

/// The coefficient each kCalibrationParamNames entry names, in the same order.
using CalibrationField = double Calibration::*;
constexpr std::array<CalibrationField, kCalibrationParamCount> kCalibrationFields{
    &Calibration::f,  &Calibration::cx, &Calibration::cy, &Calibration::b1,
    &Calibration::b2, &Calibration::k1, &Calibration::k2, &Calibration::k3,
    &Calibration::k4, &Calibration::p1, &Calibration::p2, &Calibration::p3,
    &Calibration::p4};

bool is_fixed(const std::vector<std::string>& fixed, const char* name) {
    return std::find(fixed.begin(), fixed.end(), name) != fixed.end();
}

/// Workspace has no const find_chunk(), and the panes read the document under
/// the shared lock as a const Document.
const Chunk* find_chunk_in(const Document& document, std::int32_t chunk_id) {
    for (const std::shared_ptr<Chunk>& chunk : document.workspace().chunks()) {
        if (chunk != nullptr && chunk->id() == chunk_id) {
            return chunk.get();
        }
    }
    return nullptr;
}

}  // namespace

CalibrationDialog::CalibrationDialog(GuiApp& app, QWidget* parent)
    : QDialog(parent), app_(app) {
    setObjectName(QStringLiteral("dialog.calibration"));
    setWindowTitle(tr("Camera Calibration"));
    auto* layout = new QVBoxLayout(this);
    auto* split = new QSplitter(Qt::Horizontal, this);
    split->setObjectName(QStringLiteral("calibration.splitter"));
    sensor_list_ = new QListWidget(split);
    sensor_list_->setObjectName(QStringLiteral("calibration.sensors"));
    sensor_list_->setMinimumWidth(190);
    tabs_ = new QTabWidget(split);
    tabs_->setObjectName(QStringLiteral("calibration.tabs"));
    tabs_->addTab(build_initial_tab(), tr("Initial"));
    tabs_->addTab(build_adjusted_tab(), tr("Adjusted"));
    tabs_->addTab(build_gnss_tab(), tr("GNSS/INS"));
    split->addWidget(sensor_list_);
    split->addWidget(tabs_);
    split->setStretchFactor(1, 1);
    layout->addWidget(split, 1);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    optimize_ = buttons->addButton(tr("Optimize..."), QDialogButtonBox::ActionRole);
    optimize_->setObjectName(QStringLiteral("calibration.optimize"));
    layout->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(optimize_, &QPushButton::clicked, this, &CalibrationDialog::run_optimize);
    connect(sensor_list_, &QListWidget::currentRowChanged, this,
            &CalibrationDialog::load_sensor);
    connect(&app_, &GuiApp::undo_changed, this, [this]() {
        if (isVisible()) {
            refresh();
        }
    });
    connect(&app_.jobs(), &JobManager::busy_changed, this, [this](bool) { refresh(); });
    optimize_->setEnabled(builtin_registry().find("OptimizeCameras") != nullptr);
    const std::int32_t active = app_.read_document([](const Document& document) {
        const Chunk* chunk = document.active_chunk();
        return chunk == nullptr ? -1 : chunk->id();
    });
    set_chunk(active);
    resize(760, 620);
}

QWidget* CalibrationDialog::build_initial_tab() {
    auto* page = new QWidget(this);
    page->setObjectName(QStringLiteral("calibration.initial_page"));
    auto* layout = new QVBoxLayout(page);
    auto* geometry = new QGroupBox(tr("Sensor geometry"), page);
    auto* geometry_form = new QFormLayout(geometry);
    type_ = new QComboBox(geometry);
    type_->setObjectName(QStringLiteral("calibration.type"));
    type_->addItem(tr("Frame"));
    type_->addItem(tr("Fisheye"));
    geometry_form->addRow(tr("Type:"), type_);
    const auto spin = [geometry](const QString& name, double maximum, int decimals) {
        auto* editor = new QDoubleSpinBox(geometry);
        editor->setObjectName(name);
        editor->setRange(-maximum, maximum);
        editor->setDecimals(decimals);
        editor->setKeyboardTracking(false);
        return editor;
    };
    pixel_size_ = spin(QStringLiteral("calibration.pixel_size"), 1e6, 3);
    geometry_form->addRow(tr("Pixel size (micrometres):"), pixel_size_);
    focal_length_ = spin(QStringLiteral("calibration.focal_length"), 1e6, 3);
    geometry_form->addRow(tr("Focal length (mm):"), focal_length_);
    layout->addWidget(geometry);

    auto* coefficients = new QGroupBox(tr("Initial calibration"), page);
    auto* form = new QFormLayout(coefficients);
    for (int i = 0; i < kCalibrationParamCount; ++i) {
        const QString name = QString::fromLatin1(kCalibrationParamNames[static_cast<std::size_t>(i)]);
        auto* row = new QWidget(coefficients);
        auto* row_layout = new QHBoxLayout(row);
        row_layout->setContentsMargins(0, 0, 0, 0);
        auto* editor = new QDoubleSpinBox(row);
        editor->setObjectName(QStringLiteral("calibration.initial.") + name);
        editor->setRange(-1e9, 1e9);
        editor->setDecimals(6);
        editor->setKeyboardTracking(false);
        auto* fit = new QCheckBox(tr("Fit"), row);
        fit->setObjectName(QStringLiteral("calibration.fit.") + name);
        row_layout->addWidget(editor, 1);
        row_layout->addWidget(fit);
        form->addRow(name + QStringLiteral(":"), row);
        initial_.insert(name, editor);
        fit_.insert(name, fit);
        connect(editor, &QDoubleSpinBox::editingFinished, this,
                &CalibrationDialog::commit_calibration);
        connect(fit, &QCheckBox::toggled, this,
                [this](bool) { commit_fixed_params(); });
    }
    layout->addWidget(coefficients, 1);
    connect(type_, &QComboBox::currentIndexChanged, this,
            [this](int) { commit_calibration(); });
    connect(pixel_size_, &QDoubleSpinBox::editingFinished, this,
            &CalibrationDialog::commit_physical);
    connect(focal_length_, &QDoubleSpinBox::editingFinished, this,
            &CalibrationDialog::commit_physical);
    return page;
}

QWidget* CalibrationDialog::build_adjusted_tab() {
    auto* page = new QWidget(this);
    page->setObjectName(QStringLiteral("calibration.adjusted_page"));
    auto* layout = new QVBoxLayout(page);
    auto* grid = new QGridLayout();
    grid->addWidget(new QLabel(tr("Parameter"), page), 0, 0);
    grid->addWidget(new QLabel(tr("Adjusted"), page), 0, 1);
    grid->addWidget(new QLabel(tr("Sigma"), page), 0, 2);
    for (int i = 0; i < kCalibrationParamCount; ++i) {
        const QString name =
            QString::fromLatin1(kCalibrationParamNames[static_cast<std::size_t>(i)]);
        auto* value = new QLabel(page);
        value->setObjectName(QStringLiteral("calibration.adjusted.") + name);
        auto* sigma = new QLabel(page);
        sigma->setObjectName(QStringLiteral("calibration.sigma.") + name);
        grid->addWidget(new QLabel(name + QStringLiteral(":"), page), i + 1, 0);
        grid->addWidget(value, i + 1, 1);
        grid->addWidget(sigma, i + 1, 2);
        adjusted_.insert(name, value);
        sigma_.insert(name, sigma);
    }
    grid->setColumnStretch(1, 1);
    grid->setColumnStretch(2, 1);
    layout->addLayout(grid);
    correlation_ = new QTableWidget(page);
    correlation_->setObjectName(QStringLiteral("calibration.correlation"));
    correlation_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    layout->addWidget(correlation_, 1);
    return page;
}

QWidget* CalibrationDialog::build_gnss_tab() {
    auto* page = new QWidget(this);
    page->setObjectName(QStringLiteral("calibration.gnss_page"));
    auto* layout = new QVBoxLayout(page);
    auto* group = new QGroupBox(tr("Antenna lever arm"), page);
    auto* form = new QFormLayout(group);
    const char* axes[3] = {"x", "y", "z"};
    for (int i = 0; i < 3; ++i) {
        auto* editor = new QDoubleSpinBox(group);
        editor->setObjectName(QStringLiteral("calibration.antenna.") +
                             QString::fromLatin1(axes[i]));
        editor->setRange(-1e6, 1e6);
        editor->setDecimals(4);
        editor->setKeyboardTracking(false);
        form->addRow(QStringLiteral("%1 (m):").arg(QString::fromLatin1(axes[i])),
                     editor);
        antenna_offset_[i] = editor;
        connect(editor, &QDoubleSpinBox::editingFinished, this,
                &CalibrationDialog::commit_antenna);
    }
    layout->addWidget(group);

    auto* angles = new QGroupBox(tr("Boresight (body to camera)"), page);
    auto* angles_layout = new QVBoxLayout(angles);
    boresight_enabled_ = new QCheckBox(tr("Specify boresight angles"), angles);
    boresight_enabled_->setObjectName(QStringLiteral("calibration.boresight_enabled"));
    angles_layout->addWidget(boresight_enabled_);
    auto* angles_form = new QFormLayout();
    for (int i = 0; i < 3; ++i) {
        auto* editor = new QDoubleSpinBox(angles);
        editor->setObjectName(QStringLiteral("calibration.boresight.") +
                             QString::fromLatin1(axes[i]));
        editor->setRange(-360.0, 360.0);
        editor->setDecimals(6);
        editor->setKeyboardTracking(false);
        editor->setEnabled(false);
        angles_form->addRow(
            QStringLiteral("%1 (deg):").arg(QString::fromLatin1(axes[i])), editor);
        boresight_[i] = editor;
        connect(editor, &QDoubleSpinBox::editingFinished, this,
                &CalibrationDialog::commit_antenna);
    }
    angles_layout->addLayout(angles_form);
    layout->addWidget(angles);
    layout->addStretch(1);
    connect(boresight_enabled_, &QCheckBox::toggled, this, [this](bool on) {
        for (QDoubleSpinBox* editor : boresight_) {
            editor->setEnabled(on);
        }
        commit_antenna();
    });
    return page;
}

void CalibrationDialog::set_chunk(std::int32_t chunk_id) {
    chunk_id_ = chunk_id;
    build_sensor_list();
}

void CalibrationDialog::build_sensor_list() {
    {
        const QSignalBlocker blocker(sensor_list_);
        sensor_list_->clear();
    }
    sensor_index_ = -1;
    sensor_ = Sensor{};
    std::vector<Sensor> sensors;
    app_.read_document([this, &sensors](const Document& document) {
        const Chunk* chunk = find_chunk_in(document, chunk_id_);
        if (chunk != nullptr) {
            sensors = chunk->sensors();
        }
    });
    for (const Sensor& sensor : sensors) {
        sensor_list_->addItem(sensor_text(sensor));
    }
    if (sensors.empty()) {
        load_sensor(-1);
    } else {
        sensor_list_->setCurrentRow(0);
        load_sensor(0);
    }
}

void CalibrationDialog::select_sensor(int index) {
    if (index < 0 || index >= sensor_list_->count()) {
        return;
    }
    sensor_list_->setCurrentRow(index);
    load_sensor(index);
}

Sensor CalibrationDialog::read_sensor(std::int32_t& chunk_id) const {
    Sensor sensor;
    std::int32_t found = -1;
    app_.read_document([this, &sensor, &found](const Document& document) {
        const Chunk* chunk = find_chunk_in(document, chunk_id_);
        if (chunk == nullptr) {
            return;
        }
        found = chunk->id();
        if (sensor_index_ >= 0 &&
            sensor_index_ < static_cast<int>(chunk->sensors().size())) {
            sensor = chunk->sensors()[static_cast<std::size_t>(sensor_index_)];
        }
    });
    chunk_id = found;
    return sensor;
}

void CalibrationDialog::refresh() {
    if (!app_.jobs().busy()) {
        std::int32_t id = -1;
        sensor_ = read_sensor(id);
        if (id >= 0) {
            chunk_id_ = id;
        }
        load_sensor(sensor_index_);
    }
    optimize_->setEnabled(!app_.jobs().busy() &&
                          builtin_registry().find("OptimizeCameras") != nullptr);
}

void CalibrationDialog::load_sensor(int index) {
    sensor_index_ = index;
    if (index < 0) {
        update_adjusted(Sensor{});
        return;
    }
    std::int32_t id = -1;
    sensor_ = read_sensor(id);
    if (id >= 0) {
        chunk_id_ = id;
    }
    const Calibration& calibration = sensor_.calibration;
    for (int i = 0; i < kCalibrationParamCount; ++i) {
        const std::size_t at = static_cast<std::size_t>(i);
        const QString name = QString::fromLatin1(kCalibrationParamNames[at]);
        if (QDoubleSpinBox* editor = initial_.value(name)) {
            const QSignalBlocker blocker(editor);
            editor->setValue(calibration.*kCalibrationFields[at]);
        }
        if (QCheckBox* check = fit_.value(name)) {
            const QSignalBlocker blocker(check);
            check->setChecked(
                is_fixed(sensor_.fixed_params, kCalibrationParamNames[at]));
        }
    }
    {
        const QSignalBlocker blocker(type_);
        type_->setCurrentIndex(calibration.type == CameraType::Fisheye ? 1 : 0);
    }
    {
        const QSignalBlocker blocker(pixel_size_);
        pixel_size_->setValue(sensor_.pixel_size_um);
    }
    {
        const QSignalBlocker blocker(focal_length_);
        focal_length_->setValue(sensor_.focal_length_mm);
    }
    for (int i = 0; i < 3; ++i) {
        const QSignalBlocker blocker(antenna_offset_[i]);
        antenna_offset_[i]->setValue(
            sensor_.antenna.offset_m[static_cast<Eigen::Index>(i)]);
    }
    const bool has_angles = sensor_.antenna.boresight_deg.has_value();
    {
        const QSignalBlocker blocker(boresight_enabled_);
        boresight_enabled_->setChecked(has_angles);
    }
    for (int i = 0; i < 3; ++i) {
        const QSignalBlocker blocker(boresight_[i]);
        boresight_[i]->setEnabled(has_angles);
        boresight_[i]->setValue(
            has_angles ? (*sensor_.antenna.boresight_deg)[static_cast<Eigen::Index>(i)]
                       : 0.0);
    }
    update_adjusted(sensor_);
}

void CalibrationDialog::update_adjusted(const Sensor& sensor) {
    const Calibration& calibration = sensor.calibration;
    const auto sigma_of = [&calibration](const std::string& name) {
        const auto at = std::find(calibration.covariance_params.begin(),
                                  calibration.covariance_params.end(), name);
        if (at == calibration.covariance_params.end()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const auto index = static_cast<Eigen::Index>(
            std::distance(calibration.covariance_params.begin(), at));
        if (index < 0 || index >= calibration.covariance.rows() ||
            index >= calibration.covariance.cols()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double variance = calibration.covariance(index, index);
        return variance > 0.0 ? std::sqrt(variance) : 0.0;
    };
    for (int i = 0; i < kCalibrationParamCount; ++i) {
        const std::size_t at = static_cast<std::size_t>(i);
        const QString name = QString::fromLatin1(kCalibrationParamNames[at]);
        if (QLabel* label = adjusted_.value(name)) {
            label->setText(QString::number(calibration.*kCalibrationFields[at], 'f', 6));
        }
        if (QLabel* label = sigma_.value(name)) {
            const double sigma = sigma_of(kCalibrationParamNames[at]);
            label->setText(std::isfinite(sigma) ? QString::number(sigma, 'f', 6)
                                                : QStringLiteral("n/a"));
        }
    }
    correlation_->clear();
    const int count = static_cast<int>(calibration.covariance_params.size());
    const bool has_covariance =
        count > 0 && calibration.covariance.rows() == count &&
        calibration.covariance.cols() == count;
    correlation_->setVisible(has_covariance);
    if (!has_covariance) {
        return;
    }
    correlation_->setRowCount(count);
    correlation_->setColumnCount(count);
    QStringList headers;
    for (const std::string& name : calibration.covariance_params) {
        headers << QString::fromStdString(name);
    }
    correlation_->setHorizontalHeaderLabels(headers);
    correlation_->setVerticalHeaderLabels(headers);
    for (int row = 0; row < count; ++row) {
        for (int column = 0; column < count; ++column) {
            const double variance_row =
                calibration.covariance(static_cast<Eigen::Index>(row),
                                       static_cast<Eigen::Index>(row));
            const double variance_column =
                calibration.covariance(static_cast<Eigen::Index>(column),
                                       static_cast<Eigen::Index>(column));
            double correlation = std::numeric_limits<double>::quiet_NaN();
            if (row == column) {
                correlation = 1.0;
            } else if (variance_row > 0.0 && variance_column > 0.0) {
                correlation = calibration.covariance(
                                  static_cast<Eigen::Index>(row),
                                  static_cast<Eigen::Index>(column)) /
                              std::sqrt(variance_row * variance_column);
            }
            auto* item = new QTableWidgetItem(QString::number(correlation, 'f', 3));
            if (row != column && std::isfinite(correlation) &&
                std::abs(correlation) > 0.9) {
                item->setBackground(QColor(180, 60, 60));
            }
            correlation_->setItem(row, column, item);
        }
    }
}

void CalibrationDialog::commit_calibration() {
    if (chunk_id_ < 0 || sensor_index_ < 0) {
        return;
    }
    Calibration calibration = sensor_.calibration;
    for (int i = 0; i < kCalibrationParamCount; ++i) {
        const std::size_t at = static_cast<std::size_t>(i);
        const QString name = QString::fromLatin1(kCalibrationParamNames[at]);
        if (QDoubleSpinBox* editor = initial_.value(name)) {
            calibration.*kCalibrationFields[at] = editor->value();
        }
    }
    calibration.type =
        type_->currentIndex() == 1 ? CameraType::Fisheye : CameraType::Frame;
    if (calibration.same_model(sensor_.calibration)) {
        return;
    }
    try {
        app_.execute(make_set_sensor_calibration_command(chunk_id_, sensor_index_,
                                                         calibration));
        sensor_.calibration = calibration;
        sensor_.type = calibration.type;
        sensor_.width = calibration.width;
        sensor_.height = calibration.height;
    } catch (const std::exception& error) {
        QMessageBox::critical(this, tr("Camera Calibration"),
                              QString::fromUtf8(error.what()));
        refresh();
        return;
    }
    update_adjusted(sensor_);
}

void CalibrationDialog::commit_fixed_params() {
    if (chunk_id_ < 0 || sensor_index_ < 0) {
        return;
    }
    std::vector<std::string> fixed;
    for (int i = 0; i < kCalibrationParamCount; ++i) {
        const std::size_t at = static_cast<std::size_t>(i);
        const QString name = QString::fromLatin1(kCalibrationParamNames[at]);
        if (QCheckBox* check = fit_.value(name); check != nullptr && check->isChecked()) {
            fixed.emplace_back(kCalibrationParamNames[at]);
        }
    }
    if (fixed == sensor_.fixed_params) {
        return;
    }
    try {
        app_.execute(make_set_sensor_fixed_params_command(chunk_id_, sensor_index_,
                                                          fixed));
        sensor_.fixed_params = fixed;
    } catch (const std::exception& error) {
        QMessageBox::critical(this, tr("Camera Calibration"),
                              QString::fromUtf8(error.what()));
        refresh();
    }
}

void CalibrationDialog::commit_physical() {
    if (chunk_id_ < 0 || sensor_index_ < 0) {
        return;
    }
    const double pixel_size = pixel_size_->value();
    const double focal_length = focal_length_->value();
    if (pixel_size == sensor_.pixel_size_um &&
        focal_length == sensor_.focal_length_mm) {
        return;
    }
    try {
        app_.execute(make_set_sensor_physical_command(
            chunk_id_, sensor_index_, pixel_size, focal_length));
        sensor_.pixel_size_um = pixel_size;
        sensor_.focal_length_mm = focal_length;
    } catch (const std::exception& error) {
        QMessageBox::critical(this, tr("Camera Calibration"),
                              QString::fromUtf8(error.what()));
        refresh();
    }
}

void CalibrationDialog::commit_antenna() {
    if (chunk_id_ < 0 || sensor_index_ < 0) {
        return;
    }
    Antenna antenna;
    for (int i = 0; i < 3; ++i) {
        antenna.offset_m[static_cast<Eigen::Index>(i)] = antenna_offset_[i]->value();
    }
    if (boresight_enabled_->isChecked()) {
        Eigen::Vector3d angles = Eigen::Vector3d::Zero();
        for (int i = 0; i < 3; ++i) {
            angles[static_cast<Eigen::Index>(i)] = boresight_[i]->value();
        }
        antenna.boresight_deg = angles;
    }
    if (antenna == sensor_.antenna) {
        return;
    }
    try {
        app_.execute(
            make_set_sensor_antenna_command(chunk_id_, sensor_index_, antenna));
        sensor_.antenna = antenna;
    } catch (const std::exception& error) {
        QMessageBox::critical(this, tr("Camera Calibration"),
                              QString::fromUtf8(error.what()));
        refresh();
    }
}

void CalibrationDialog::run_optimize() {
    if (chunk_id_ < 0 || app_.jobs().busy() ||
        builtin_registry().find("OptimizeCameras") == nullptr) {
        return;
    }
    try {
        TaskDialog dialog("OptimizeCameras", builtin_registry(), app_.settings(),
                          this);
        if (dialog.exec() != QDialog::Accepted) {
            return;
        }
        app_.jobs().submit("OptimizeCameras", dialog.params(), chunk_id_);
    } catch (const std::exception& error) {
        QMessageBox::critical(this, tr("Camera Calibration"),
                              QString::fromUtf8(error.what()));
    }
}

QDoubleSpinBox* CalibrationDialog::initial_editor(const QString& param) const {
    return initial_.value(param);
}

QCheckBox* CalibrationDialog::fit_check(const QString& param) const {
    return fit_.value(param);
}

QLabel* CalibrationDialog::adjusted_value(const QString& param) const {
    return adjusted_.value(param);
}

}  // namespace alphashape::gui



