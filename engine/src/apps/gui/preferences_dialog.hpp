#pragma once

// Preferences: update settings and the Add Photos import options. Values
// persist immediately in QSettings under "updates/..." and "photos/...".
//
// Check Now asks the window to run a manual check; it never starts a request
// from the dialog itself, so a test can wire the signal without a network.

#include <QDialog>

#include "apps/gui/update_checker.hpp"

class QCheckBox;
class QComboBox;
class QPushButton;
class QSettings;

namespace alphashape::gui {

class PreferencesDialog : public QDialog {
    Q_OBJECT

public:
    /// `settings` holds the Add Photos import options; the update settings live
    /// in the checker's own store.
    PreferencesDialog(UpdateChecker& checker, QSettings& settings, QWidget* parent = nullptr);

    QCheckBox* auto_check() const { return auto_check_; }
    QComboBox* channel() const { return channel_; }
    QPushButton* check_now() const { return check_now_; }
    /// "Load the DJI XMP lens calibration when adding photos".
    QCheckBox* load_xmp_calibration() const { return load_xmp_calibration_; }

Q_SIGNALS:
    void check_requested();

private:
    UpdateChecker& checker_;
    QSettings& settings_;
    QCheckBox* auto_check_ = nullptr;
    QComboBox* channel_ = nullptr;
    QPushButton* check_now_ = nullptr;
    QCheckBox* load_xmp_calibration_ = nullptr;
};

}  // namespace alphashape::gui
