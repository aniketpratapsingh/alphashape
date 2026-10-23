#pragma once

#include <QMainWindow>

namespace alphashape::gui {

class GuiApp;

/// The application window. Phase 1 starts with an empty shell: the docks, stage rail and
/// viewport are added by later work packages. It remembers its geometry between runs.
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(GuiApp& app, QWidget* parent = nullptr);

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    void update_title();

    GuiApp& app_;
};

}  // namespace alphashape::gui
