#include "apps/gui/preferences_dialog.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLabel>
#include <QPushButton>
#include <QSettings>
#include <QVBoxLayout>

namespace alphashape::gui {

namespace {
/// QSettings key for the Add Photos "load XMP lens calibration" option. The
/// same key is read by MainWindow and WorkflowController when they build the
/// AddPhotos parameters.
constexpr const char* kLoadXmpCalibrationKey = "photos/load_xmp_calibration";
}  // namespace

PreferencesDialog::PreferencesDialog(UpdateChecker& checker, QSettings& settings, QWidget* parent)
    : QDialog(parent), checker_(checker), settings_(settings) {
    setObjectName(QStringLiteral("preferencesDialog"));
    setWindowTitle(tr("Preferences"));

    auto* layout = new QVBoxLayout(this);

    auto* heading = new QLabel(tr("Updates"), this);
    heading->setObjectName(QStringLiteral("preferences.updates_heading"));
    layout->addWidget(heading);

    auto* form = new QFormLayout();
    auto_check_ = new QCheckBox(tr("Automatically check for updates"), this);
    auto_check_->setObjectName(QStringLiteral("preferences.auto_updates"));
    auto_check_->setChecked(checker_.automatic_checks_enabled());
    form->addRow(auto_check_);

    channel_ = new QComboBox(this);
    channel_->setObjectName(QStringLiteral("preferences.channel"));
    channel_->addItem(tr("Stable"), static_cast<int>(update::Channel::Stable));
    channel_->addItem(tr("Include pre-releases"),
                      static_cast<int>(update::Channel::Prerelease));
    channel_->setCurrentIndex(
        channel_->findData(static_cast<int>(checker_.channel())));
    form->addRow(tr("Channel:"), channel_);
    layout->addLayout(form);

    auto* privacy = new QLabel(
        tr("Checks contact github.com only: one request for the public release list, and "
           "the installer download when you approve it. No usage data is sent."),
        this);
    privacy->setObjectName(QStringLiteral("preferences.privacy"));
    privacy->setWordWrap(true);
    layout->addWidget(privacy);

    check_now_ = new QPushButton(tr("Check Now"), this);
    check_now_->setObjectName(QStringLiteral("preferences.check_now"));
    layout->addWidget(check_now_);

    auto* photos_heading = new QLabel(tr("Photos"), this);
    photos_heading->setObjectName(QStringLiteral("preferences.photos_heading"));
    layout->addWidget(photos_heading);

    load_xmp_calibration_ =
        new QCheckBox(tr("Load the DJI XMP lens calibration when adding photos"), this);
    load_xmp_calibration_->setObjectName(QStringLiteral("preferences.load_xmp_calibration"));
    load_xmp_calibration_->setToolTip(
        tr("Seed each sensor's initial calibration from the DJI factory lens calibration "
           "(DewarpData) in the photo's XMP. On by default; when it is off, or a photo "
           "carries no such calibration, the EXIF focal length is used instead."));
    load_xmp_calibration_->setChecked(
        settings_.value(QString::fromLatin1(kLoadXmpCalibrationKey), true).toBool());
    layout->addWidget(load_xmp_calibration_);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    buttons->setObjectName(QStringLiteral("preferences.buttons"));
    layout->addWidget(buttons);

    connect(auto_check_, &QCheckBox::toggled, this, [this](bool checked) {
        checker_.set_automatic_checks_enabled(checked);
    });
    connect(channel_, &QComboBox::currentIndexChanged, this, [this](int index) {
        const auto channel = static_cast<update::Channel>(channel_->itemData(index).toInt());
        checker_.set_channel(channel);
    });
    connect(load_xmp_calibration_, &QCheckBox::toggled, this, [this](bool checked) {
        settings_.setValue(QString::fromLatin1(kLoadXmpCalibrationKey), checked);
    });
    connect(check_now_, &QPushButton::clicked, this, &PreferencesDialog::check_requested);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::accept);
}

}  // namespace alphashape::gui
