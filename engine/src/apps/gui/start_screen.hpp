#pragma once

// StartScreen: what the central area shows while no project is open (on startup without a file
// argument, and after File > Close).
//
// It offers the three ways into a project - New project, Open..., Add photos... - the recent
// projects GuiApp remembers (entries whose file is gone stay listed, greyed, and can be removed
// from the list), an optional Open sample entry when a sample project is configured, and
// drag-and-drop of a project file (opens it) or of images / folders (new project + Add photos).
//
// Whether a recent entry is missing is answered on a worker thread: a recent list can name a
// stale SMB/NFS path, and one stat() on a dead mount freezes the GUI for as long as the network
// takes to time out. Entries therefore show as "unknown" (not greyed) and the whole list is
// built at once; the answers arrive later and are applied only if no newer sweep was started.
//
// The widget owns no state: it reads GuiApp's recent list, rebuilds it on
// recent_files_changed, and reports every user choice as a signal. MainWindow decides what a
// choice means (it owns the unsaved-changes prompt, the window stack and the workflow), so the
// same actions behave the same whichever view is showing.
//
// Colours and sizes come from the theme (QSS roles plus the palette); nothing is hardcoded.

#include <atomic>
#include <functional>
#include <memory>
#include <optional>

#include <QHash>
#include <QList>
#include <QString>
#include <QStringList>
#include <QWidget>

class QDragEnterEvent;
class QDragLeaveEvent;
class QDragMoveEvent;
class QDropEvent;
class QLabel;
class QListWidget;
class QListWidgetItem;
class QMenu;
class QMimeData;
class QPoint;
class QPushButton;
class QShowEvent;

namespace alphashape::gui {

class GuiApp;

class StartScreen : public QWidget {
    Q_OBJECT

public:
    explicit StartScreen(GuiApp& app, QWidget* parent = nullptr);
    /// Tells an outstanding existence sweep not to call back into a destroyed screen.
    ~StartScreen() override;

    /// Rebuild the recent list and re-check whether the sample project is available. Connected
    /// to GuiApp::recent_files_changed; also called whenever the screen is shown.
    void reload();

    QListWidget* recent_list() const { return recent_; }
    /// The paths in the recent list, in display order.
    QStringList recent_paths() const;
    /// True when the entry at `row` is known to name a file that is not on disk. False while its
    /// existence is still being checked on the worker: an entry is never greyed on a guess.
    bool recent_missing(int row) const;
    /// Block until the outstanding existence sweep has been applied (tests, and the
    /// close/quit path). Returns false when it did not finish within the timeout.
    bool wait_for_recent_checks(int timeout_ms = 5000);

    /// How the worker decides whether a path exists. Replaced by a slow probe in the tests.
    using ExistsProber = std::function<bool(const QString&)>;
    void set_exists_prober(ExistsProber prober);
    /// Drop the entry at `row` from the list (and from GuiApp's persisted list).
    void remove_recent(int row);

    /// Context menu for the entry under `position` (in list coordinates): "Open" and "Remove
    /// from Recent Projects" as "action.start.recent.open" / "action.start.recent.remove".
    /// Null when no entry is there. The caller owns the menu.
    QMenu* build_recent_menu(const QPoint& position);
    /// The same menu for a known row (what the position overload resolves to).
    QMenu* build_recent_menu(int row);
    /// The row an item position names, or -1.
    int row_at(const QPoint& position) const;

    /// The sample project, when one is configured and present: ALPHASHAPE_SAMPLE_PROJECT (a
    /// project file, or a directory holding one), else share/alphashape/sample next to the
    /// installed binary. Never walks the source tree: a release build without the data simply
    /// hides the button.
    static std::optional<QString> find_sample_project();
    bool sample_available() const { return !sample_path_.isEmpty(); }
    QString sample_path() const { return sample_path_; }

    /// The local files and folders a drag carries ("" entries and remote URLs dropped).
    static QStringList local_paths(const QMimeData* mime);

    /// Asked for the photos to add (empty = cancelled). Default: a file dialog.
    using PhotosHandler = std::function<QStringList()>;
    void set_photos_handler(PhotosHandler handler);

    // For tests and MainWindow.
    QPushButton* new_button() const { return new_button_; }
    QPushButton* open_button() const { return open_button_; }
    QPushButton* add_photos_button() const { return add_photos_button_; }
    QPushButton* sample_button() const { return sample_button_; }

    /// Enable or disable the entry points (the window disables them while a job runs).
    void set_actions_enabled(bool enabled);
    bool actions_enabled() const { return actions_enabled_; }

    /// True while an existence sweep is on the worker. The screen is fully usable meanwhile:
    /// this is only what a test (or the quit path) waits on.
    bool recent_checks_pending() const { return pending_sweeps_ > 0; }

    /// Drop a project / photo paths onto the screen from code (the user's drag-and-drop, and
    /// the tests, which cannot synthesise platform drags on the offscreen plugin). Accepts the
    /// drop the way Qt would and emits paths_dropped(). False when nothing usable was dropped.
    bool drop_paths(const QStringList& paths);

Q_SIGNALS:
    void new_requested();
    void open_requested();
    /// The user picked photos to add to a new project.
    void add_photos_requested(const QStringList& files);
    /// A project path was chosen (a recent entry or the sample).
    void open_path_requested(const QString& path);
    /// The recent entry for `path` was removed from the list.
    void recent_removed(const QString& path);
    /// Files were dropped on the screen; MainWindow sorts them into "open this project" or
    /// "start a project with these photos".
    void paths_dropped(const QStringList& paths);

protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dragMoveEvent(QDragMoveEvent* event) override;
    void dragLeaveEvent(QDragLeaveEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    /// Re-read the recent list and the sample path whenever the launcher comes back on screen:
    /// a project may have been opened and closed meanwhile, or its file deleted, and the
    /// sample environment could have been set after construction.
    void showEvent(QShowEvent* event) override;

private:
    void build();
    void on_recent_clicked(QListWidgetItem* item);
    void on_context_menu(const QPoint& position);
    void on_add_photos();
    void on_sample();

    /// Re-apply the stored missing/known state of one row (text, grey colour, tooltip).
    void paint_recent_row(QListWidgetItem* item, const QString& path);
    /// Start the worker sweep that answers "does this recent entry exist?" for `paths`. The
    /// answers are applied only if no newer sweep was started meanwhile.
    void start_recent_sweep(const QStringList& paths);
    void apply_recent_sweep(int generation, const QStringList& paths, const QList<bool>& missing);

    GuiApp& app_;
    PhotosHandler photos_handler_;
    ExistsProber exists_prober_;
    /// Answers already known, path -> missing. Survives a reload, so a theme switch repaints
    /// the grey state at once instead of dropping back to "unknown".
    QHash<QString, bool> known_recent_;
    int sweep_generation_ = 0;
    int pending_sweeps_ = 0;
    /// Shared with the outstanding sweeps; cleared in the destructor so a worker that finishes
    /// afterwards does not touch this object.
    std::shared_ptr<std::atomic<bool>> alive_ = std::make_shared<std::atomic<bool>>(true);

    QPushButton* new_button_ = nullptr;
    QPushButton* open_button_ = nullptr;
    QPushButton* add_photos_button_ = nullptr;
    QPushButton* sample_button_ = nullptr;
    QLabel* recent_title_ = nullptr;
    QListWidget* recent_ = nullptr;
    QLabel* empty_ = nullptr;
    QString sample_path_;
    bool actions_enabled_ = true;
    /// dragEnter's verdict, reused by dragMove (which fires continuously).
    bool drag_accepts_ = false;
};

}  // namespace alphashape::gui
