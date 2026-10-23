// Shared main for the GUI tests: one offscreen QApplication, with QSettings redirected to a
// scratch directory so no test reads or writes the developer's real settings.

#include <gtest/gtest.h>

#include <QApplication>
#include <QSettings>
#include <QTemporaryDir>
#include <QtGlobal>

int main(int argc, char** argv) {
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }
    QTemporaryDir settings_dir;
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settings_dir.path());

    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
