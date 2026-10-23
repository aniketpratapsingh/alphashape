#include "apps/gui/main_window.hpp"

#include <QCloseEvent>
#include <QSettings>
#include <QStatusBar>
#include <QWidget>

#include "apps/gui/app.hpp"

namespace alphashape::gui {

namespace {
constexpr const char* kGeometryKey = "window/geometry";
}  // namespace

MainWindow::MainWindow(GuiApp& app, QWidget* parent) : QMainWindow(parent), app_(app) {
    setObjectName(QStringLiteral("mainWindow"));

    auto* central = new QWidget(this);
    central->setObjectName(QStringLiteral("appBackground"));
    setCentralWidget(central);
    statusBar()->showMessage(tr("Ready"));

    resize(1280, 800);
    restoreGeometry(app_.settings().value(kGeometryKey).toByteArray());

    update_title();
    connect(&app_, &GuiApp::document_replaced, this, &MainWindow::update_title);
}

void MainWindow::update_title() {
    setWindowTitle(tr("%1 - alphashape").arg(app_.document_title()));
}

void MainWindow::closeEvent(QCloseEvent* event) {
    app_.settings().setValue(kGeometryKey, saveGeometry());
    QMainWindow::closeEvent(event);
}

}  // namespace alphashape::gui
