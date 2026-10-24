#include "apps/gui/main_window.hpp"

#include <exception>
#include <optional>
#include <utility>

#include <QAction>
#include <QActionGroup>
#include <QCloseEvent>
#include <QDesktopServices>
#include <QDialog>
#include <QDockWidget>
#include <QFileDialog>
#include <QFileInfo>
#include <QKeySequence>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QPair>
#include <QProgressBar>
#include <QMessageBox>
#include <QSettings>
#include <QStatusBar>
#include <QToolButton>
#include <QUrl>
#include <QWidget>

#include "apps/gui/app.hpp"
#include "apps/gui/properties_pane.hpp"
#include "apps/gui/viewer/viewport_widget.hpp"
#include "apps/gui/workspace_pane.hpp"
#include "apps/gui/workflow.hpp"

#ifndef ALPHASHAPE_GUI_VERSION
#define ALPHASHAPE_GUI_VERSION "0.0.0"
#endif

namespace alphashape::gui {

namespace {

constexpr const char* kGeometryKey = "window/geometry";
constexpr const char* kStateKey = "window/state";
/// Bump when the set of docks changes, so an old saved layout is ignored instead of misapplied.
constexpr int kLayoutVersion = 1;
constexpr const char* kRepositoryUrl = "https://github.com/aniketpratapsingh/alphashape";
constexpr const char* kProjectFilter = "alphashape projects (*.psx);;All files (*)";

QDockWidget* make_dock(QMainWindow* parent, const QString& object_name, const QString& title,
                       const QString& placeholder) {
    auto* dock = new QDockWidget(title, parent);
    dock->setObjectName(object_name);
    dock->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetMovable |
                      QDockWidget::DockWidgetFloatable);
    auto* label = new QLabel(placeholder, dock);
    label->setObjectName(QStringLiteral("panel.") + object_name.mid(5));
    label->setAlignment(Qt::AlignCenter);
    label->setEnabled(false);
    dock->setWidget(label);
    dock->setMinimumWidth(120);
    return dock;
}

}  // namespace

MainWindow::MainWindow(GuiApp& app, QWidget* parent) : QMainWindow(parent), app_(app) {
    setObjectName(QStringLiteral("mainWindow"));

    unsaved_handler_ = [this](const QString& name) {
        QMessageBox box(QMessageBox::Warning, tr("Unsaved changes"),
                        tr("Save changes to \"%1\" before closing?").arg(name),
                        QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel, this);
        box.setInformativeText(tr("Your changes will be lost if you don't save them."));
        box.setDefaultButton(QMessageBox::Save);
        switch (box.exec()) {
            case QMessageBox::Save:
                return UnsavedChoice::Save;
            case QMessageBox::Discard:
                return UnsavedChoice::Discard;
            default:
                return UnsavedChoice::Cancel;
        }
    };
    open_path_handler_ = [this]() {
        const QString start = app_.recent_files().isEmpty()
                                  ? QString()
                                  : QFileInfo(app_.recent_files().first()).absolutePath();
        return QFileDialog::getOpenFileName(this, tr("Open project"), start,
                                            tr(kProjectFilter));
    };
    save_path_handler_ = [this](const QString& suggested) {
        // A default suffix makes the dialog itself append ".psx", so its own overwrite
        // confirmation sees the real target name.
        QFileDialog dialog(this, tr("Save project"), suggested, tr(kProjectFilter));
        dialog.setAcceptMode(QFileDialog::AcceptSave);
        dialog.setDefaultSuffix(QStringLiteral("psx"));
        if (dialog.exec() != QDialog::Accepted || dialog.selectedFiles().isEmpty()) {
            return QString();
        }
        return dialog.selectedFiles().first();
    };
    cancel_jobs_handler_ = [this](const QString& task) {
        QMessageBox box(QMessageBox::Question, tr("Job running"),
                        tr("%1 is still running. Cancel it and quit?").arg(task),
                        QMessageBox::Yes | QMessageBox::No, this);
        box.setDefaultButton(QMessageBox::No);
        return box.exec() == QMessageBox::Yes;
    };
    error_handler_ = [this](const QString& title, const QString& message) {
        QMessageBox::critical(this, title, message);
    };

    viewport_ = new ViewportWidget(app_, app_.theme(), this);
    setCentralWidget(viewport_);

    create_docks();
    create_actions();
    create_menus();
    create_status_strip();

    // The viewport highlights a selected camera, so it follows the tree selection.
    viewport_->set_selection(selection_);

    resize(1280, 800);
    apply_default_layout();
    restoreGeometry(app_.settings().value(kGeometryKey).toByteArray());
    restoreState(app_.settings().value(kStateKey).toByteArray(), kLayoutVersion);

    update_title();
    update_undo_actions();
    update_recent_menu();
    connect(&app_, &GuiApp::document_replaced, this, &MainWindow::update_title);
    connect(&app_, &GuiApp::path_changed, this, &MainWindow::update_title);
    connect(&app_, &GuiApp::dirty_changed, this, [this](bool) { update_title(); });
    connect(&app_, &GuiApp::undo_changed, this, &MainWindow::update_undo_actions);
    connect(&app_, &GuiApp::recent_files_changed, this, &MainWindow::update_recent_menu);

    JobManager& jobs = app_.jobs();
    connect(&jobs, &JobManager::busy_changed, this, [this](bool) {
        update_job_ui();
        update_undo_actions();
    });
    connect(&jobs, &JobManager::job_state_changed, this,
            [this](quint64, JobManager::State) { update_job_ui(); });
    connect(&jobs, &JobManager::job_progress, this,
            [this](quint64, double percent, const QString& stage, qint64 eta_ms) {
                on_job_progress(percent, stage, eta_ms);
            });
    connect(&jobs, &JobManager::job_finished, this,
            [this](quint64, JobManager::State state, const QString& message) {
                on_job_finished(static_cast<int>(state), message);
            });
    update_job_ui();
}

// --- Hooks -------------------------------------------------------------------------------

void MainWindow::set_unsaved_handler(UnsavedHandler handler) {
    unsaved_handler_ = std::move(handler);
}
void MainWindow::set_open_path_handler(OpenPathHandler handler) {
    open_path_handler_ = std::move(handler);
}
void MainWindow::set_save_path_handler(SavePathHandler handler) {
    save_path_handler_ = std::move(handler);
}
void MainWindow::set_error_handler(ErrorHandler handler) {
    error_handler_ = std::move(handler);
}

void MainWindow::set_cancel_jobs_handler(CancelJobsHandler handler) {
    cancel_jobs_handler_ = std::move(handler);
}

void MainWindow::report_error(const QString& title, const std::exception& error) {
    error_handler_(title, QString::fromUtf8(error.what()));
}

// --- Construction ------------------------------------------------------------------------

void MainWindow::create_docks() {
    workspace_dock_ = make_dock(this, QStringLiteral("dock.workspace"), tr("Workspace"),
                                tr("Workspace"));
    properties_dock_ = make_dock(this, QStringLiteral("dock.properties"), tr("Properties"),
                                 tr("Properties"));
    photos_dock_ = make_dock(this, QStringLiteral("dock.photos"), tr("Photos"), tr("Photos"));
    log_dock_ = make_dock(this, QStringLiteral("dock.log"), tr("Log"), tr("Log"));

    // The Workspace tree and the Properties sheet replace their placeholders.
    selection_ = new WorkspaceSelection(this);
    selection_->setObjectName(QStringLiteral("selection.workspace"));
    workspace_pane_ = new WorkspacePane(app_, *selection_, workspace_dock_);
    properties_pane_ = new PropertiesPane(app_, *selection_, properties_dock_);
    delete workspace_dock_->widget();
    workspace_dock_->setWidget(workspace_pane_);
    delete properties_dock_->widget();
    properties_dock_->setWidget(properties_pane_);
}

void MainWindow::apply_default_layout() {
    const struct {
        QDockWidget* dock;
        Qt::DockWidgetArea area;
    } placement[] = {
        {workspace_dock_, Qt::LeftDockWidgetArea},
        {properties_dock_, Qt::RightDockWidgetArea},
        {photos_dock_, Qt::BottomDockWidgetArea},
        {log_dock_, Qt::BottomDockWidgetArea},
    };
    for (const auto& item : placement) {
        item.dock->setFloating(false);
        addDockWidget(item.area, item.dock);
        item.dock->show();
    }
    tabifyDockWidget(photos_dock_, log_dock_);
    photos_dock_->raise();
    resizeDocks({workspace_dock_, properties_dock_}, {268, 320}, Qt::Horizontal);
    resizeDocks({photos_dock_}, {160}, Qt::Vertical);
}

QAction* MainWindow::make_action(const QString& object_name, const QString& text,
                                 const QKeySequence& shortcut) {
    auto* action = new QAction(text, this);
    action->setObjectName(object_name);
    if (!shortcut.isEmpty()) {
        action->setShortcut(shortcut);
    }
    return action;
}

void MainWindow::create_actions() {
    // Project actions.
    connect(make_action(QStringLiteral("action.file.new"), tr("&New"), QKeySequence::New),
            &QAction::triggered, this, [this]() { new_project(); });
    connect(make_action(QStringLiteral("action.file.open"), tr("&Open..."), QKeySequence::Open),
            &QAction::triggered, this, [this]() { open_project(); });
    connect(make_action(QStringLiteral("action.file.save"), tr("&Save"), QKeySequence::Save),
            &QAction::triggered, this, [this]() { save_project(); });
    connect(make_action(QStringLiteral("action.file.save_as"), tr("Save &As..."),
                        QKeySequence::SaveAs),
            &QAction::triggered, this, [this]() { save_project_as(); });
    connect(make_action(QStringLiteral("action.file.close"), tr("&Close"), QKeySequence::Close),
            &QAction::triggered, this, [this]() { close_project(); });
    QAction* quit =
        make_action(QStringLiteral("action.file.quit"), tr("&Quit"), QKeySequence::Quit);
    quit->setMenuRole(QAction::QuitRole);
    connect(quit, &QAction::triggered, this, [this]() { close(); });

    // The recent-files pool: created once and shown or hidden, never destroyed, so an entry
    // can safely open a project (and so rebuild the list) from its own triggered() slot.
    for (int i = 0; i < GuiApp::kMaxRecentFiles; ++i) {
        QAction* entry = make_action(QStringLiteral("action.file.recent.%1").arg(i), QString());
        entry->setVisible(false);
        connect(entry, &QAction::triggered, this, [this, entry]() {
            open_project(entry->data().toString());
        });
        recent_actions_.append(entry);
    }
    recent_clear_ = make_action(QStringLiteral("action.file.recent.clear"), tr("Clear Menu"));
    connect(recent_clear_, &QAction::triggered, &app_, &GuiApp::clear_recent_files);

    // Edit.
    undo_action_ = make_action(QStringLiteral("action.edit.undo"), tr("Undo"), QKeySequence::Undo);
    connect(undo_action_, &QAction::triggered, this, [this]() {
        try {
            app_.undo();
        } catch (const std::exception& e) {
            report_error(tr("Undo failed"), e);
        }
    });
    redo_action_ = make_action(QStringLiteral("action.edit.redo"), tr("Redo"), QKeySequence::Redo);
    connect(redo_action_, &QAction::triggered, this, [this]() {
        try {
            app_.redo();
        } catch (const std::exception& e) {
            report_error(tr("Redo failed"), e);
        }
    });
    QAction* preferences = make_action(QStringLiteral("action.edit.preferences"),
                                       tr("Preferences..."), QKeySequence::Preferences);
    preferences->setMenuRole(QAction::PreferencesRole);
    preferences->setEnabled(false);

    // View.
    auto* theme_group = new QActionGroup(this);
    theme_group->setExclusive(true);
    theme_light_ = make_action(QStringLiteral("action.view.theme.light"), tr("Light"));
    theme_dark_ = make_action(QStringLiteral("action.view.theme.dark"), tr("Dark"));
    for (QAction* action : {theme_light_, theme_dark_}) {
        action->setCheckable(true);
        theme_group->addAction(action);
    }
    const auto sync_theme = [this](ThemeMode mode) {
        theme_light_->setChecked(mode == ThemeMode::Light);
        theme_dark_->setChecked(mode == ThemeMode::Dark);
    };
    sync_theme(app_.theme().mode());
    connect(&app_.theme(), &Theme::changed, this, sync_theme);
    connect(theme_light_, &QAction::triggered, this, [this]() { app_.set_theme(ThemeMode::Light); });
    connect(theme_dark_, &QAction::triggered, this, [this]() { app_.set_theme(ThemeMode::Dark); });
    connect(make_action(QStringLiteral("action.view.reset_layout"), tr("Reset Layout")),
            &QAction::triggered, this, [this]() { reset_layout(); });

    // Viewport: frame the scene, switch projection, show or hide the grid and axes. The viewport
    // itself handles F (fit) and 5 (projection) while it has focus; these menu entries carry no
    // application-wide shortcut, because a bare key would also fire while typing in a pane.
    QAction* fit = make_action(QStringLiteral("action.view.fit_scene"), tr("&Fit Scene"));
    fit->setToolTip(tr("Frame the whole scene (F)"));
    QAction* orthographic = make_action(QStringLiteral("action.view.orthographic"),
                                        tr("&Orthographic"));
    orthographic->setCheckable(true);
    orthographic->setToolTip(tr("Toggle the perspective / orthographic projection (5)"));
    QAction* grid = make_action(QStringLiteral("action.view.show_grid"), tr("Show &Grid"));
    grid->setCheckable(true);
    grid->setChecked(true);
    QAction* axes = make_action(QStringLiteral("action.view.show_axes"), tr("Show &Axes"));
    axes->setCheckable(true);
    axes->setChecked(true);
    connect(fit, &QAction::triggered, this, [this]() { viewport_->fit_scene(); });
    connect(orthographic, &QAction::triggered, this,
            [this](bool on) { viewport_->set_orthographic(on); });
    connect(grid, &QAction::triggered, this, [this](bool on) { viewport_->set_show_grid(on); });
    connect(axes, &QAction::triggered, this, [this](bool on) { viewport_->set_show_axes(on); });
    // Keep the checkable entries in step when the viewport changes the state itself (a key).
    connect(viewport_, &ViewportWidget::projection_changed, orthographic,
            &QAction::setChecked);

    // Help.
    QAction* about = make_action(QStringLiteral("action.help.about"), tr("&About alphashape"));
    about->setMenuRole(QAction::AboutRole);
    connect(about, &QAction::triggered, this, [this]() { show_about(); });
    connect(make_action(QStringLiteral("action.help.repository"), tr("Project &Repository")),
            &QAction::triggered, this,
            []() { QDesktopServices::openUrl(QUrl(QString::fromLatin1(kRepositoryUrl))); });

    const QPair<QDockWidget*, QString> toggles[] = {
        {workspace_dock_, QStringLiteral("workspace")},
        {properties_dock_, QStringLiteral("properties")},
        {photos_dock_, QStringLiteral("photos")},
        {log_dock_, QStringLiteral("log")},
    };
    for (const auto& toggle : toggles) {
        toggle.first->toggleViewAction()->setObjectName(
            QStringLiteral("action.view.dock.") + toggle.second);
    }
}

void MainWindow::create_menus() {
    const auto act = [this](const char* name) {
        return findChild<QAction*>(QString::fromLatin1(name));
    };
    const auto menu = [this](const QString& object_name, const QString& title) {
        QMenu* m = menuBar()->addMenu(title);
        m->setObjectName(object_name);
        return m;
    };

    QMenu* file = menu(QStringLiteral("menu.file"), tr("&File"));
    file->addAction(act("action.file.new"));
    file->addAction(act("action.file.open"));
    recent_menu_ = file->addMenu(tr("Open &Recent"));
    recent_menu_->setObjectName(QStringLiteral("menu.file.recent"));
    for (QAction* entry : recent_actions_) {
        recent_menu_->addAction(entry);
    }
    recent_menu_->addSeparator();
    recent_menu_->addAction(recent_clear_);
    file->addSeparator();
    file->addAction(act("action.file.save"));
    file->addAction(act("action.file.save_as"));
    file->addSeparator();
    file->addAction(act("action.file.close"));
    file->addAction(act("action.file.quit"));

    QMenu* edit = menu(QStringLiteral("menu.edit"), tr("&Edit"));
    edit->addAction(undo_action_);
    edit->addAction(redo_action_);
    edit->addSeparator();
    edit->addAction(act("action.edit.preferences"));

    QMenu* view = menu(QStringLiteral("menu.view"), tr("&View"));
    for (QDockWidget* pane : {workspace_dock_, properties_dock_, photos_dock_, log_dock_}) {
        view->addAction(pane->toggleViewAction());
    }
    view->addSeparator();
    QMenu* theme_menu = view->addMenu(tr("&Theme"));
    theme_menu->setObjectName(QStringLiteral("menu.view.theme"));
    theme_menu->addAction(theme_light_);
    theme_menu->addAction(theme_dark_);
    view->addSeparator();
    view->addAction(act("action.view.fit_scene"));
    view->addAction(act("action.view.orthographic"));
    view->addAction(act("action.view.show_grid"));
    view->addAction(act("action.view.show_axes"));
    view->addSeparator();
    view->addAction(act("action.view.reset_layout"));

    QMenu* workflow_menu = menu(QStringLiteral("menu.workflow"), tr("&Workflow"));
    workflow_ = new WorkflowController(app_, workflow_menu, this);
    workflow_->set_error_handler(
        [this](const QString& title, const QString& message) { error_handler_(title, message); });

    QMenu* help = menu(QStringLiteral("menu.help"), tr("&Help"));
    help->addAction(act("action.help.about"));
    help->addAction(act("action.help.repository"));
}

void MainWindow::create_status_strip() {
    statusBar()->setObjectName(QStringLiteral("statusStrip"));
    statusBar()->showMessage(tr("Ready"));
    // Right side: stage text, a progress bar and a cancel button. The bar and the button only
    // show while a job is active.
    jobs_label_ = new QLabel(tr("No jobs running"), this);
    jobs_label_->setObjectName(QStringLiteral("status.jobs"));
    jobs_progress_ = new QProgressBar(this);
    jobs_progress_->setObjectName(QStringLiteral("status.progress"));
    jobs_progress_->setRange(0, 100);
    jobs_progress_->setFixedWidth(160);
    jobs_progress_->setTextVisible(true);
    jobs_cancel_ = new QToolButton(this);
    jobs_cancel_->setObjectName(QStringLiteral("status.cancel"));
    jobs_cancel_->setText(tr("Cancel"));
    jobs_cancel_->setToolTip(tr("Cancel the running job"));
    connect(jobs_cancel_, &QToolButton::clicked, &app_.jobs(), qOverload<>(&JobManager::cancel));
    statusBar()->addPermanentWidget(jobs_label_);
    statusBar()->addPermanentWidget(jobs_progress_);
    statusBar()->addPermanentWidget(jobs_cancel_);
    jobs_progress_->setVisible(false);
    jobs_cancel_->setVisible(false);
}

// --- Jobs ----------------------------------------------------------------------------------

namespace {
QString format_eta(qint64 eta_ms) {
    const qint64 seconds = (eta_ms + 999) / 1000;
    if (seconds < 60) {
        return QObject::tr("%1 s").arg(seconds);
    }
    if (seconds < 3600) {
        return QObject::tr("%1 min").arg((seconds + 59) / 60);
    }
    return QObject::tr("%1 h %2 min").arg(seconds / 3600).arg((seconds % 3600 + 59) / 60);
}
}  // namespace

void MainWindow::update_job_ui() {
    const JobManager& jobs = app_.jobs();
    const bool busy = jobs.busy();

    // Everything that edits, replaces or saves the document waits for the job.
    for (const char* name : {"action.file.new", "action.file.open", "action.file.save",
                             "action.file.save_as", "action.file.close"}) {
        if (QAction* item = findChild<QAction*>(QString::fromLatin1(name))) {
            item->setEnabled(!busy);
        }
    }
    update_recent_menu();

    jobs_progress_->setVisible(busy);
    jobs_cancel_->setVisible(busy);
    if (!busy) {
        jobs_label_->setText(tr("No jobs running"));
        jobs_label_->setToolTip(QString());
        return;
    }
    const std::optional<JobManager::JobId> running = jobs.running_job();
    const std::optional<JobManager::Job> current =
        running ? jobs.job(*running) : std::optional<JobManager::Job>{};
    QString text = current ? current->task : tr("Waiting");
    if (current && !current->stage.isEmpty()) {
        text += QStringLiteral(": ") + current->stage;
    }
    if (jobs.pending_count() > 0) {
        text += tr(" (+%1 queued)").arg(static_cast<qulonglong>(jobs.pending_count()));
    }
    jobs_label_->setText(text);
    jobs_progress_->setValue(current ? static_cast<int>(current->percent) : 0);
}

void MainWindow::on_job_progress(double percent, const QString& stage, qint64 eta_ms) {
    jobs_progress_->setValue(static_cast<int>(percent));
    update_job_ui();
    jobs_label_->setToolTip(eta_ms >= 0 ? tr("%1 remaining").arg(format_eta(eta_ms)) : stage);
}

void MainWindow::on_job_finished(int state, const QString& message) {
    switch (static_cast<JobManager::State>(state)) {
        case JobManager::State::Failed:
            error_handler_(tr("Task failed"), message);
            break;
        case JobManager::State::Succeeded:
            statusBar()->showMessage(tr("Task finished"), 4000);
            break;
        case JobManager::State::Cancelled:
            statusBar()->showMessage(tr("Task cancelled"), 4000);
            break;
        default:
            break;
    }
}

// --- Project operations ------------------------------------------------------------------

bool MainWindow::confirm_discard() {
    if (!app_.dirty()) {
        return true;
    }
    switch (unsaved_handler_(app_.document_title())) {
        case UnsavedChoice::Save:
            return save_project();
        case UnsavedChoice::Discard:
            return true;
        case UnsavedChoice::Cancel:
            break;
    }
    return false;
}

bool MainWindow::new_project() {
    if (!confirm_discard()) {
        return false;
    }
    app_.new_document("");
    statusBar()->showMessage(tr("New project"), 4000);
    return true;
}

bool MainWindow::open_project() {
    // Ask for the file first: cancelling the dialog must not prompt about unsaved changes.
    const QString path = open_path_handler_();
    if (path.isEmpty()) {
        return false;
    }
    return open_project(path);
}

bool MainWindow::open_project(const QString& path) {
    if (!confirm_discard()) {
        return false;
    }
    try {
        app_.open_document(path.toStdString());
    } catch (const std::exception& e) {
        app_.remove_recent_file(path);
        report_error(tr("Could not open project"), e);
        return false;
    }
    statusBar()->showMessage(tr("Opened %1").arg(path), 4000);
    return true;
}

bool MainWindow::save_project() {
    // The menu action is disabled while a job runs; a programmatic or shortcut call is not.
    if (app_.jobs().busy()) {
        statusBar()->showMessage(tr("A task is running. Wait for it to finish first."), 4000);
        return false;
    }
    const QString current = app_.read_document(
        [](const Document& document) { return QString::fromStdString(document.path()); });
    if (current.isEmpty()) {
        return save_project_as();
    }
    try {
        app_.save_document();
    } catch (const std::exception& e) {
        report_error(tr("Could not save project"), e);
        return false;
    }
    statusBar()->showMessage(tr("Saved"), 4000);
    return true;
}

bool MainWindow::save_project_as() {
    if (app_.jobs().busy()) {
        statusBar()->showMessage(tr("A task is running. Wait for it to finish first."), 4000);
        return false;
    }
    const auto current = app_.read_document([](const Document& document) {
        return QPair<QString, QString>(QString::fromStdString(document.path()),
                                       document_title_of(document));
    });
    const QString suggested =
        current.first.isEmpty() ? current.second + QStringLiteral(".psx") : current.first;
    QString path = save_path_handler_(suggested);
    if (path.isEmpty()) {
        return false;
    }
    if (QFileInfo(path).suffix().isEmpty()) {
        path += QStringLiteral(".psx");
    }
    try {
        app_.save_document_as(path.toStdString());
    } catch (const std::exception& e) {
        report_error(tr("Could not save project"), e);
        return false;
    }
    statusBar()->showMessage(tr("Saved %1").arg(path), 4000);
    return true;
}

bool MainWindow::close_project() {
    if (!confirm_discard()) {
        return false;
    }
    app_.new_document("");
    statusBar()->showMessage(tr("Project closed"), 4000);
    return true;
}

// --- Layout, title, menus ------------------------------------------------------------------

void MainWindow::save_layout() {
    app_.settings().setValue(kGeometryKey, saveGeometry());
    app_.settings().setValue(kStateKey, saveState(kLayoutVersion));
}

void MainWindow::reset_layout() {
    apply_default_layout();
}

QDockWidget* MainWindow::dock(const QString& object_name) const {
    return findChild<QDockWidget*>(object_name);
}

void MainWindow::update_title() {
    setWindowTitle(tr("%1[*] - alphashape").arg(app_.document_title()));
    setWindowModified(app_.dirty());
}

void MainWindow::update_undo_actions() {
    const UndoStack& stack = app_.undo_stack();
    const bool idle = !app_.jobs().busy();  // the document belongs to the job until it ends
    undo_action_->setEnabled(stack.can_undo() && idle);
    undo_action_->setText(stack.can_undo() ? QString::fromStdString(stack.undo_label())
                                           : tr("Undo"));
    redo_action_->setEnabled(stack.can_redo() && idle);
    redo_action_->setText(stack.can_redo() ? QString::fromStdString(stack.redo_label())
                                           : tr("Redo"));
}

void MainWindow::update_recent_menu() {
    const QStringList recent = app_.recent_files();
    for (int i = 0; i < recent_actions_.size(); ++i) {
        QAction* entry = recent_actions_[i];
        const bool used = i < recent.size();
        entry->setVisible(used);
        if (used) {
            entry->setText(QFileInfo(recent[i]).fileName());
            entry->setToolTip(recent[i]);
            entry->setData(recent[i]);
        }
    }
    recent_clear_->setEnabled(!recent.isEmpty());
    recent_menu_->setEnabled(!recent.isEmpty() && !app_.jobs().busy());
}

QString MainWindow::about_text() const {
    return tr("alphashape %1\n\n"
              "An open-source photogrammetry suite.\n"
              "Licensed under the GNU General Public License, version 3 or (at your option) "
              "any later version.\n"
              "Built with Qt %2 (LGPL-3).\n\n%3")
        .arg(QStringLiteral(ALPHASHAPE_GUI_VERSION), QString::fromLatin1(qVersion()),
             QString::fromLatin1(kRepositoryUrl));
}

void MainWindow::show_about() {
    QMessageBox::about(this, tr("About alphashape"), about_text());
}

void MainWindow::closeEvent(QCloseEvent* event) {
    JobManager& jobs = app_.jobs();
    if (jobs.busy()) {
        const std::optional<JobManager::JobId> running = jobs.running_job();
        const std::optional<JobManager::Job> current =
            running ? jobs.job(*running) : std::optional<JobManager::Job>{};
        if (!cancel_jobs_handler_(current ? current->task : tr("A job"))) {
            event->ignore();
            return;
        }
        jobs.cancel_all_and_wait();
    }
    if (!confirm_discard()) {
        event->ignore();
        return;
    }
    save_layout();
    QMainWindow::closeEvent(event);
}

}  // namespace alphashape::gui
