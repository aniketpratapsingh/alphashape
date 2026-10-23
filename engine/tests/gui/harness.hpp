#pragma once

// Offscreen GUI test harness.
//
// test_main.cpp creates the one QApplication of the test binary (offscreen platform, QSettings
// redirected to a scratch directory). GuiTest builds a GuiApp and a MainWindow on top of it and
// offers the helpers every GUI test needs: find actions, menus, docks and widgets by objectName,
// trigger an action, pump the event loop with a timeout, capture Qt warnings, and (for
// debugging only) grab a widget to a PNG.

#include <functional>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include <QAction>
#include <QDockWidget>
#include <QMenu>
#include <QString>
#include <QStringList>
#include <QTemporaryDir>

#include "apps/gui/app.hpp"
#include "apps/gui/main_window.hpp"

namespace alphashape::gui::testing {

/// Collects every message Qt prints at warning level or worse (minus a short list of known
/// platform-layer noise) while it is alive. Other messages go to the previous handler.
class QtMessageCapture {
public:
    QtMessageCapture();
    ~QtMessageCapture();
    QtMessageCapture(const QtMessageCapture&) = delete;
    QtMessageCapture& operator=(const QtMessageCapture&) = delete;

    QStringList messages() const;
    bool empty() const { return messages().isEmpty(); }
    /// The messages joined for use in an assertion message.
    std::string text() const;
};

/// Process the Qt event loop until `done()` is true or `timeout_ms` has passed. Returns the
/// final value of `done()`.
bool pump_until(const std::function<bool()>& done, int timeout_ms = 2000);

/// Process events for about `ms` milliseconds (including deferred deletes and posted events).
void pump(int ms = 30);

/// Fixture: a GuiApp and a MainWindow on fresh settings. Derive from it, or use TEST_F with it
/// directly.
class GuiTest : public ::testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;

    /// Destroy the window and the app and build new ones. With `reset_settings` false the new
    /// instances read what the old ones stored (used to test persistence).
    void rebuild(bool reset_settings = false);

    GuiApp& app() { return *app_; }
    MainWindow& window() { return *window_; }

    /// Look up by objectName anywhere under the window. A missing object fails the test (once,
    /// with the name) and returns nullptr.
    QAction* action(const QString& object_name);
    QMenu* menu(const QString& object_name);
    QDockWidget* dock(const QString& object_name);
    template <typename T>
    T* widget(const QString& object_name) {
        T* found = window_->findChild<T*>(object_name);
        if (found == nullptr) {
            ADD_FAILURE() << "no widget named " << object_name.toStdString();
        }
        return found;
    }

    /// Trigger a named action. Fails the test, and does nothing, if it is missing or disabled
    /// (QAction::trigger would silently ignore a disabled action).
    bool trigger(const QString& object_name);

    /// Show the window and let the first layout pass happen.
    void show_window();

    /// Qt warnings seen since SetUp (construction included).
    const QtMessageCapture& warnings() const { return *capture_; }

    /// A path inside this test's scratch directory.
    QString scratch(const QString& name) const;

    /// The settings file every GuiApp of this test uses (see rebuild()). It lives in the scratch
    /// directory, so it is removed with it and no other test process can see it.
    QString settings_path() const;

    /// Debugging aid: save a screenshot of `widget` as `<name>.png` when the environment
    /// variable ALPHASHAPE_GUI_GRAB_DIR names a directory. Returns true when a file was
    /// written. Never used for assertions.
    static bool grab(QWidget* target, const QString& name);

private:
    std::unique_ptr<QtMessageCapture> capture_;
    std::unique_ptr<QTemporaryDir> scratch_;
    std::unique_ptr<GuiApp> app_;
    std::unique_ptr<MainWindow> window_;
};

}  // namespace alphashape::gui::testing
