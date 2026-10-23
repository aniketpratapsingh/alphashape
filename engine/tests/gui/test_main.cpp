// Shared main for the GUI tests: one offscreen QApplication and a scratch directory. Every
// GuiApp here — and every alphashape-gui this binary spawns — is pointed at a settings file of
// its own (see GuiApp's ALPHASHAPE_GUI_SETTINGS_FILE), because QSettings' default store is one
// machine-wide registry key on Windows and parallel ctest processes would share it.

#include <gtest/gtest.h>

#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QString>
#include <QTemporaryDir>
#include <QtGlobal>

int main(int argc, char** argv) {
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }
    QTemporaryDir settings_dir;
    const QString file = QDir(settings_dir.path())
                             .filePath(QStringLiteral("settings-%1.ini")
                                           .arg(QCoreApplication::applicationPid()));
    qputenv("ALPHASHAPE_GUI_SETTINGS_FILE", file.toLocal8Bit());

    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
