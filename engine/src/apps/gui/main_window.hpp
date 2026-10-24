#pragma once

#include <exception>
#include <functional>

#include <QList>
#include <QMainWindow>
#include <QString>
#include <QStringList>

class QAction;
class QActionGroup;
class QDockWidget;
class QLabel;
class QProgressBar;
class QStackedWidget;
class QToolButton;
class QMenu;

namespace alphashape::gui {

class ExportController;
class GuiApp;
class LogPane;
class PhotosPane;
class PropertiesPane;
class StartScreen;
class ViewportWidget;
class WorkflowController;
class WorkspacePane;
class WorkspaceSelection;

/// What the user chose when asked about unsaved changes.
enum class UnsavedChoice { Save, Discard, Cancel };

/// The application window: a viewport in the centre, the Workspace dock on the left, the
/// Properties dock on the right, the Photos and Log docks (tabbed) along the bottom, a menu
/// bar and a status strip. The docks hold placeholders until their own work packages land.
///
/// The central area is a two-page stack: the 3-D viewport and the start screen. The start screen
/// shows while no project is open (on startup without a file argument, and after File > Close);
/// anything that opens or starts a project switches to the viewport. It is deliberately a
/// Widgets page rather than a QML view, so the application needs no QML runtime.
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

    /// Asked when the window is closed while a job runs; true = cancel the job and quit. The
    /// argument is the running task's name.
    using CancelJobsHandler = std::function<bool(const QString& task_name)>;
    void set_cancel_jobs_handler(CancelJobsHandler handler);

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
    /// The Workflow menu's controller (task actions and their hooks).
    WorkflowController& workflow() { return *workflow_; }
    /// The File > Export submenu's controller.
    ExportController& exports() { return *export_controller_; }
    /// The central-area start screen ("startScreen"), shown while no project is open.
    StartScreen* start_screen() const { return start_screen_; }
    /// The 3-D view: page 0 of the central stack ("viewport").
    ViewportWidget* viewport() const { return viewport_; }
    /// The central-area stack: page 0 is the viewport ("viewport"), page 1 the start screen.
    /// Exposed so a test can say which page is showing without guessing by visibility.
    QStackedWidget* central_stack() const { return stack_; }
    /// The central page currently showing (0 = viewport, 1 = start screen).
    int central_page() const;
    /// True when the start screen is the visible central page.
    bool showing_start_screen() const;

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
    void update_job_ui();
    void on_job_progress(double percent, const QString& stage, qint64 eta_ms);
    /// `task` is the finished job's name; an export posts its own, more specific note, so the
    /// generic "Task finished" would only overwrite it.
    void on_job_finished(int state, const QString& message, const QString& task);
    void show_about();
    void report_error(const QString& title, const std::exception& error);
    QAction* make_action(const QString& object_name, const QString& text,
                         const QKeySequence& shortcut = {});

    // --- Start screen and exports ---

    /// Show the start screen page (no project open) or the viewport page.
    void show_start_page();
    void show_viewport_page();
    /// Pick the page the document implies: the start screen for an empty, unsaved project.
    void update_central_page();
    /// Start a project from `files`: a new document with one chunk, then AddPhotos through the
    /// job manager (the same path the Workflow menu's AddPhotos uses).
    bool new_project_from_photos(const QStringList& files);
    /// Handle a drop on the start screen: a project file opens it, anything else is photos.
    bool handle_dropped_paths(const QStringList& paths);
    /// Run AddPhotos on the active chunk for `files`.
    bool add_photos_to_active_chunk(const QStringList& files);
    /// Status-bar note with a Reveal action after an export finished.
    void note_export_written(const QString& path);

    GuiApp& app_;
    UnsavedHandler unsaved_handler_;
    OpenPathHandler open_path_handler_;
    SavePathHandler save_path_handler_;
    ErrorHandler error_handler_;
    CancelJobsHandler cancel_jobs_handler_;

    QDockWidget* workspace_dock_ = nullptr;
    QDockWidget* properties_dock_ = nullptr;
    QDockWidget* photos_dock_ = nullptr;
    QDockWidget* log_dock_ = nullptr;
    QStackedWidget* stack_ = nullptr;   ///< central area: viewport / start screen
    ViewportWidget* viewport_ = nullptr;
    StartScreen* start_screen_ = nullptr;
    WorkspaceSelection* selection_ = nullptr;  ///< tree -> Properties, Photos panes and viewport
    WorkspacePane* workspace_pane_ = nullptr;
    PropertiesPane* properties_pane_ = nullptr;
    PhotosPane* photos_pane_ = nullptr;
    LogPane* log_pane_ = nullptr;
    WorkflowController* workflow_ = nullptr;
    ExportController* export_controller_ = nullptr;
    QLabel* jobs_label_ = nullptr;          ///< stage / idle text ("status.jobs")
    QProgressBar* jobs_progress_ = nullptr; ///< "status.progress", visible while a job runs
    QToolButton* jobs_cancel_ = nullptr;    ///< "status.cancel", visible while a job runs
    QList<QAction*> project_actions_;       ///< actions that touch the document
    /// The "Reveal" action of the last export note, and the status-strip button showing it.
    QAction* reveal_action_ = nullptr;
    QToolButton* reveal_button_ = nullptr;

    QMenu* recent_menu_ = nullptr;
    QMenu* export_menu_ = nullptr;
    /// True while the start screen is the intended central page (no project open). Explicit
    /// rather than derived from the document: "File > New" on an empty project must stay on
    /// the viewport, and "-new" at startup is a request for an empty project, not for the
    /// start screen.
    bool start_page_ = false;
    QList<QAction*> recent_actions_;  ///< fixed pool, shown or hidden as the list changes
    QAction* recent_clear_ = nullptr;
    QAction* undo_action_ = nullptr;
    QAction* redo_action_ = nullptr;
    QAction* theme_light_ = nullptr;
    QAction* theme_dark_ = nullptr;
};

}  // namespace alphashape::gui
