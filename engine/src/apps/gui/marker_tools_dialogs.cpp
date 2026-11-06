#include "apps/gui/marker_tools_dialogs.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QIODevice>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QSaveFile>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QWidget>

#include "apps/gui/modal_hooks.hpp"
#include "base/utils/error.hpp"
#include "common/tasks/registry.hpp"

namespace alphashape::gui {
namespace {

constexpr double kTargetPreviewDpi = 30.0;
constexpr double kSheetPreviewDpi = 24.0;
constexpr int kPreviewWidth = 200;
constexpr int kSheetPreviewWidth = 220;

/// An 8-bit marker-sheet raster as a QImage the label can show.
QImage to_qimage(const Image& source) {
    if (source.empty() || source.width() <= 0 || source.height() <= 0) {
        return {};
    }
    const int width = source.width();
    const int height = source.height();
    const int channels = source.channels();
    if (channels == 1) {
        const QImage view(source.data(), width, height, width,
                          QImage::Format_Grayscale8);
        return view.copy();
    }
    if (channels == 3) {
        const QImage view(source.data(), width, height, width * 3,
                          QImage::Format_RGB888);
        return view.copy();
    }
    if (channels == 4) {
        const QImage view(source.data(), width, height, width * 4,
                          QImage::Format_RGBA8888);
        return view.copy();
    }
    return {};
}

/// The sheet family a DetectMarkers target_type selects, when one exists.
/// parse_sheet_target covers the lower-case script aliases as well as the
/// schema's capitalised names.
std::optional<SheetTarget> preview_target(const QString& type) {
    return parse_sheet_target(type.toLower().toStdString());
}

}  // namespace

DetectMarkersDialog::DetectMarkersDialog(const TaskRegistry& registry,
                                         QSettings& settings, QWidget* parent)
    : TaskDialog("DetectMarkers", registry, settings, parent) {
    setWindowTitle(tr("Detect Markers"));
    preview_ = new QLabel(this);
    preview_->setObjectName(QStringLiteral("markerPreview"));
    preview_->setAlignment(Qt::AlignCenter);
    preview_->setWordWrap(true);
    preview_->setMinimumWidth(kPreviewWidth + 20);
    if (auto* root = qobject_cast<QVBoxLayout*>(layout())) {
        // The button box is the last item TaskDialog adds; keep it there.
        root->insertWidget(root->count() - 1, preview_);
    }
    if (auto* target = qobject_cast<QComboBox*>(editor("target_type"))) {
        connect(target, qOverload<int>(&QComboBox::currentIndexChanged), this,
                [this](int) { refresh_preview(); });
    }
    if (auto* inverted = qobject_cast<QCheckBox*>(editor("inverted"))) {
        connect(inverted, &QCheckBox::toggled, this,
                [this](bool) { refresh_preview(); });
    }
    // The base Reset action loads values with the editor signals blocked.
    if (auto* reset = findChild<QPushButton*>(QStringLiteral("task.reset"))) {
        connect(reset, &QPushButton::clicked, this,
                [this]() { refresh_preview(); });
    }
    refresh_preview();
}

void DetectMarkersDialog::set_value(const std::string& param,
                                    const ParamValue& value) {
    TaskDialog::set_value(param, value);
    refresh_preview();
}

void DetectMarkersDialog::refresh_preview() {
    preview_image_ = QImage();
    if (preview_ == nullptr) {
        return;
    }
    const auto* target = qobject_cast<QComboBox*>(editor("target_type"));
    const QString type = target != nullptr ? target->currentText() : QString();
    const std::optional<SheetTarget> sheet = preview_target(type);
    if (!sheet) {
        preview_->setPixmap(QPixmap());
        preview_->setText(tr("No preview for %1").arg(type));
        return;
    }
    SheetOptions options;
    options.target = *sheet;
    // A single target keeps the raster cheap; the preview only has to show the
    // pattern the decoder will look for, not a complete page.
    options.count = 1;
    options.first_code = 1;
    options.labels = false;
    try {
        preview_image_ = to_qimage(
            marker_sheet_raster(options, kTargetPreviewDpi, 0));
    } catch (const std::exception&) {
        // No layout for this family at the preview size: show the placeholder.
        preview_image_ = QImage();
    }
    if (preview_image_.isNull()) {
        preview_->setPixmap(QPixmap());
        preview_->setText(tr("No preview for %1").arg(type));
        return;
    }
    if (const auto* inverted = qobject_cast<QCheckBox*>(editor("inverted"));
        inverted != nullptr && inverted->isChecked()) {
        preview_image_.invertPixels();
    }
    preview_->setText(QString());
    preview_->setPixmap(QPixmap::fromImage(preview_image_)
                            .scaledToWidth(kPreviewWidth, Qt::SmoothTransformation));
}

namespace {

const std::vector<SheetTarget>& all_sheet_targets() {
    static const std::vector<SheetTarget> targets{
        SheetTarget::Circular12, SheetTarget::Circular14, SheetTarget::Circular16,
        SheetTarget::Circular20, SheetTarget::Cross,    SheetTarget::AprilTag36h11,
        SheetTarget::Chessboard};
    return targets;
}

}  // namespace

PrintMarkersDialog::PrintMarkersDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("Print Markers"));
    auto* root = new QVBoxLayout(this);
    root->addWidget(build_fields());

    auto* preview_row = new QHBoxLayout();
    preview_ = new QLabel(this);
    preview_->setObjectName(QStringLiteral("print.preview"));
    preview_->setAlignment(Qt::AlignCenter);
    preview_->setMinimumSize(kSheetPreviewWidth + 20, 260);
    preview_row->addWidget(preview_);
    auto* side = new QVBoxLayout();
    page_label_ = new QLabel(this);
    page_label_->setObjectName(QStringLiteral("print.pages"));
    error_ = new QLabel(this);
    error_->setObjectName(QStringLiteral("print.error"));
    error_->setWordWrap(true);
    side->addWidget(page_label_);
    side->addWidget(error_);
    side->addStretch(1);
    preview_row->addLayout(side, 1);
    root->addLayout(preview_row);

    auto* buttons = new QDialogButtonBox(this);
    save_ = buttons->addButton(tr("Save PDF..."), QDialogButtonBox::AcceptRole);
    save_->setObjectName(QStringLiteral("print.save"));
    buttons->addButton(tr("Close"), QDialogButtonBox::RejectRole);
    connect(buttons, &QDialogButtonBox::accepted, this, &PrintMarkersDialog::save_pdf);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    root->addWidget(buttons);

    connect(target_, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int) {
                sync_chessboard_fields();
                refresh();
            });
    connect(paper_, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int) { refresh(); });
    for (QDoubleSpinBox* spin : {target_size_, quiet_zone_, square_size_}) {
        connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
                [this](double) { refresh(); });
    }
    for (QSpinBox* spin : {first_code_, count_, board_cols_, board_rows_}) {
        connect(spin, qOverload<int>(&QSpinBox::valueChanged), this,
                [this](int) { refresh(); });
    }
    for (QCheckBox* box : {corner_marks_, labels_}) {
        connect(box, &QCheckBox::toggled, this, [this](bool) { refresh(); });
    }
    sync_chessboard_fields();
    refresh();
}

QWidget* PrintMarkersDialog::build_fields() {
    auto* host = new QWidget(this);
    auto* form = new QFormLayout(host);

    target_ = new QComboBox(host);
    target_->setObjectName(QStringLiteral("print.target"));
    for (SheetTarget target : all_sheet_targets()) {
        target_->addItem(QString::fromStdString(sheet_target_name(target)),
                         static_cast<int>(target));
    }
    form->addRow(tr("Target"), target_);

    paper_ = new QComboBox(host);
    paper_->setObjectName(QStringLiteral("print.paper"));
    paper_->addItems({QStringLiteral("A4"), QStringLiteral("Letter")});
    form->addRow(tr("Paper"), paper_);

    target_size_ = new QDoubleSpinBox(host);
    target_size_->setObjectName(QStringLiteral("print.target_size"));
    target_size_->setDecimals(2);
    target_size_->setRange(1.0, 500.0);
    target_size_->setValue(25.0);
    target_size_->setSuffix(tr(" mm"));
    form->addRow(tr("Target size"), target_size_);

    quiet_zone_ = new QDoubleSpinBox(host);
    quiet_zone_->setObjectName(QStringLiteral("print.quiet_zone"));
    quiet_zone_->setDecimals(2);
    quiet_zone_->setRange(0.0, 100.0);
    quiet_zone_->setValue(5.0);
    quiet_zone_->setSuffix(tr(" mm"));
    form->addRow(tr("Quiet zone"), quiet_zone_);

    first_code_ = new QSpinBox(host);
    first_code_->setObjectName(QStringLiteral("print.first_code"));
    first_code_->setRange(0, 1000000);
    first_code_->setValue(1);
    form->addRow(tr("First code"), first_code_);

    count_ = new QSpinBox(host);
    count_->setObjectName(QStringLiteral("print.count"));
    count_->setRange(0, 1000000);
    count_->setValue(0);
    count_->setSpecialValueText(tr("Fill first page"));
    form->addRow(tr("Count"), count_);

    corner_marks_ = new QCheckBox(tr("Corner marks"), host);
    corner_marks_->setObjectName(QStringLiteral("print.corner_marks"));
    corner_marks_->setChecked(true);
    form->addRow(QString(), corner_marks_);

    labels_ = new QCheckBox(tr("Code labels"), host);
    labels_->setObjectName(QStringLiteral("print.labels"));
    labels_->setChecked(true);
    form->addRow(QString(), labels_);

    board_cols_ = new QSpinBox(host);
    board_cols_->setObjectName(QStringLiteral("print.board_cols"));
    board_cols_->setRange(3, 50);
    board_cols_->setValue(9);
    form->addRow(tr("Board columns"), board_cols_);

    board_rows_ = new QSpinBox(host);
    board_rows_->setObjectName(QStringLiteral("print.board_rows"));
    board_rows_->setRange(3, 50);
    board_rows_->setValue(6);
    form->addRow(tr("Board rows"), board_rows_);

    square_size_ = new QDoubleSpinBox(host);
    square_size_->setObjectName(QStringLiteral("print.square_size"));
    square_size_->setDecimals(2);
    square_size_->setRange(1.0, 200.0);
    square_size_->setValue(25.0);
    square_size_->setSuffix(tr(" mm"));
    form->addRow(tr("Square size"), square_size_);
    return host;
}

SheetTarget PrintMarkersDialog::selected_target() const {
    return static_cast<SheetTarget>(target_->currentData().toInt());
}

void PrintMarkersDialog::sync_chessboard_fields() {
    const bool chessboard = selected_target() == SheetTarget::Chessboard;
    // A chessboard has no codes; the code fields stay visible but inert, and
    // the board geometry takes over.
    for (QWidget* field : {static_cast<QWidget*>(target_size_),
                           static_cast<QWidget*>(quiet_zone_),
                           static_cast<QWidget*>(first_code_),
                           static_cast<QWidget*>(count_)}) {
        field->setEnabled(!chessboard);
    }
    for (QWidget* field : {static_cast<QWidget*>(board_cols_),
                           static_cast<QWidget*>(board_rows_),
                           static_cast<QWidget*>(square_size_)}) {
        field->setEnabled(chessboard);
    }
    // Corner marks and the footer apply to both kinds of sheet.
    corner_marks_->setEnabled(true);
    labels_->setEnabled(true);
}

SheetOptions PrintMarkersDialog::options() const {
    SheetOptions options;
    options.target = selected_target();
    options.paper = paper_->currentText().toStdString();
    options.target_size_mm = target_size_->value();
    options.quiet_zone_mm = quiet_zone_->value();
    options.first_code = first_code_->value();
    options.count = count_->value();
    options.corner_marks = corner_marks_->isChecked();
    options.labels = labels_->isChecked();
    options.board_cols = board_cols_->value();
    options.board_rows = board_rows_->value();
    options.square_mm = square_size_->value();
    return options;
}

void PrintMarkersDialog::refresh() {
    if (page_label_ == nullptr) {
        return;
    }
    const SheetOptions layout = options();
    try {
        const int pages = marker_sheet_page_count(layout);
        const Image raster = marker_sheet_raster(layout, kSheetPreviewDpi, 0);
        preview_image_ = to_qimage(raster);
        page_count_ = pages;
        valid_ = true;
        page_label_->setText(pages == 1 ? tr("1 page") : tr("%1 pages").arg(pages));
        error_->setText(QString());
        save_->setEnabled(true);
        if (preview_image_.isNull()) {
            preview_->setPixmap(QPixmap());
            preview_->setText(tr("No preview"));
        } else {
            preview_->setText(QString());
            preview_->setPixmap(
                QPixmap::fromImage(preview_image_)
                    .scaledToWidth(kSheetPreviewWidth, Qt::SmoothTransformation));
        }
    } catch (const std::exception& error) {
        // An invalid layout is normal while the user is typing: say what is
        // wrong, keep the dialog alive, and refuse to save.
        preview_image_ = QImage();
        page_count_ = 0;
        valid_ = false;
        page_label_->setText(tr("No pages"));
        error_->setText(QString::fromUtf8(error.what()));
        save_->setEnabled(false);
        preview_->setPixmap(QPixmap());
        preview_->setText(tr("No preview"));
    }
}

void PrintMarkersDialog::save_pdf() {
    if (!layout_valid()) {
        refresh();
    }
    if (!layout_valid()) {
        return;
    }
    const SheetOptions layout = options();
    const QString suggested =
        QStringLiteral("markers-") +
        QString::fromStdString(sheet_target_name(layout.target)) +
        QStringLiteral(".pdf");
    const QString path =
        modal_hooks().save_file(this, tr("Save Marker Sheet"), suggested,
                                tr("PDF files (*.pdf);;All files (*)"));
    if (path.isEmpty()) {
        return;
    }
    try {
        const std::vector<std::uint8_t> pdf = marker_sheet_pdf(layout);
        // QSaveFile writes a temporary file beside the target and renames it
        // on commit; a failed write removes the temporary file on destruction.
        QSaveFile output(path);
        if (!output.open(QIODevice::WriteOnly)) {
            throw Error("could not open " + path.toStdString() + ": " +
                        output.errorString().toStdString());
        }
        const qint64 size = static_cast<qint64>(pdf.size());
        if (output.write(reinterpret_cast<const char*>(pdf.data()), size) != size) {
            throw Error("could not write " + path.toStdString() + ": " +
                        output.errorString().toStdString());
        }
        if (!output.commit()) {
            throw Error("could not finish " + path.toStdString() + ": " +
                        output.errorString().toStdString());
        }
    } catch (const std::exception& error) {
        report_error(this, tr("Could not save marker sheet"),
                     QString::fromUtf8(error.what()));
    }
}

}  // namespace alphashape::gui
