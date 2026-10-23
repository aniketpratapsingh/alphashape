#pragma once

#include <exception>
#include <functional>

#include <QList>
#include <QMainWindow>
#include <QString>

class QAction;
class QActionGroup;
class QDockWidget;
class QLabel;
class QMenu;

namespace alphashape::gui {

class GuiApp;
class ViewportPlaceholder;

/// What the user chose when asked about unsaved changes.
enum class UnsavedChoice { Save, Discard, Cancel };

/// The application window: a viewport in the centre, the Workspace dock on the left, the
/// Properties dock on the right, the Photos and Log docks (tabbed) along the bottom, a menu
/// bar and a status strip. The docks hold placeholders until their own work packages land.
///
/// Every action, menu and dock has a stable objectName ("action.file.save", "menu.view",
/// "dock.workspace", ...), which the tests and later shortcut/command code rely on.
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(GuiApp& app, QWidget* parent = nullptr);

    // --- Interaction hooks. Defaults show the usual Qt dialogs; tests replace them. ---

    /// Asked when the project has unsaved changes and the user is about to lose them
    /// (New, Open, Close, window close). The argument is the project name.
    using UnsavedHandler = std::function<UnsavedChoice(const QString& project_name)>;
    void set_unsaved_handler(UnsavedHandler handler);
    /// Returns the project file to open ("" = cancelled).
    using OpenPathHandler = std::function<QString()>;
    void set_open_path_handler(OpenPathHandler handler);
    /// Returns the file to save to, given a suggestion ("" = cancelled).
    using SavePathHandler = std::function<QString(const QString& suggested)>;
    void set_save_path_handler(SavePathHandler handler);
    /// Told about a failed file operation (title, message). Default: a critical message box.
    using ErrorHandler = std::function<void(const QString& title, const QString& message)>;
    void set_error_handler(ErrorHandler handler);

    // --- Project operations (what the File menu does). Each returns false if cancelled or
    // failed; failures are reported through the error handler. ---

    /// Ask what to do with unsaved changes; true when it is fine to replace the document.
    bool confirm_discard();
    bool new_project();
    bool open_project();                      ///< asks for a path through the open handler
    bool open_project(const QString& path);
    bool save_project();
    bool save_project_as();
    bool close_project();

    // --- Layout ---

    /// Persist geometry and dock layout to the settings.
    void save_layout();
    /// Restore the default dock arrangement (View > Reset layout).
    void reset_layout();

    QDockWidget* dock(const QString& object_name) const;
    /// Text of the About box (version and licence).
    QString about_text() const;

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    void create_docks();
    void create_actions();
    void create_menus();
    void create_status_strip();
    void apply_default_layout();
    void update_title();
    void update_undo_actions();
    void update_recent_menu();
    void show_about();
    void report_error(const QString& title, const std::exception& error);
    QAction* make_action(const QString& object_name, const QString& text,
                         const QKeySequence& shortcut = {});

    GuiApp& app_;
    UnsavedHandler unsaved_handler_;
    OpenPathHandler open_path_handler_;
    SavePathHandler save_path_handler_;
    ErrorHandler error_handler_;

    QDockWidget* workspace_dock_ = nullptr;
    QDockWidget* properties_dock_ = nullptr;
    QDockWidget* photos_dock_ = nullptr;
    QDockWidget* log_dock_ = nullptr;
    ViewportPlaceholder* viewport_ = nullptr;
    QLabel* jobs_label_ = nullptr;

    QMenu* recent_menu_ = nullptr;
    QList<QAction*> recent_actions_;  ///< fixed pool, shown or hidden as the list changes
    QAction* recent_clear_ = nullptr;
    QAction* undo_action_ = nullptr;
    QAction* redo_action_ = nullptr;
    QAction* theme_light_ = nullptr;
    QAction* theme_dark_ = nullptr;
};

}  // namespace alphashape::gui
