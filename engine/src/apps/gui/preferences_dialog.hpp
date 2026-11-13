#pragma once

// Preferences: update settings only (the rest of the app has no preferences
// yet). Values persist immediately in QSettings under "updates/...".
//
// Check Now asks the window to run a manual check; it never starts a request
// from the dialog itself, so a test can wire the signal without a network.

#include <QDialog>

#include "apps/gui/update_checker.hpp"

class QCheckBox;
class QComboBox;
class QPushButton;

namespace alphashape::gui {

class PreferencesDialog : public QDialog {
    Q_OBJECT

public:
    explicit PreferencesDialog(UpdateChecker& checker, QWidget* parent = nullptr);

    QCheckBox* auto_check() const { return auto_check_; }
    QComboBox* channel() const { return channel_; }
    QPushButton* check_now() const { return check_now_; }

Q_SIGNALS:
    void check_requested();

private:
    UpdateChecker& checker_;
    QCheckBox* auto_check_ = nullptr;
    QComboBox* channel_ = nullptr;
    QPushButton* check_now_ = nullptr;
};

}  // namespace alphashape::gui
