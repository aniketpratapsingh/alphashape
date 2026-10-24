#include "apps/gui/start_screen.hpp"

#include <atomic>
#include <exception>
#include <functional>
#include <memory>
#include <utility>

#include <QAbstractItemView>
#include <QAction>
#include <QBrush>
#include <QByteArray>
#include <QCoreApplication>
#include <QDir>
#include <QDragEnterEvent>
#include <QDragLeaveEvent>
#include <QDropEvent>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QMetaObject>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMenu>
#include <QMimeData>
#include <QModelIndex>
#include <QPalette>
#include <QPushButton>
#include <QRunnable>
#include <QShowEvent>
#include <QThread>
#include <QThreadPool>
#include <QUrl>
#include <QVBoxLayout>
#include <QtGlobal>

#include "apps/gui/app.hpp"
#include "apps/gui/drop_paths.hpp"
#include "apps/gui/image_files.hpp"
#include "apps/gui/theme/theme.hpp"

#ifndef ALPHASHAPE_GUI_VERSION
#define ALPHASHAPE_GUI_VERSION "0.0.0"
#endif

namespace alphashape::gui {

namespace {

constexpr const char* kSampleEnv = "ALPHASHAPE_SAMPLE_PROJECT";
/// Width the content column is capped at, so the screen reads well on a wide window.
constexpr int kContentWidth = 720;

/// True when `path` names an existing project file.
bool project_exists(const QString& path) { return !path.isEmpty() && QFileInfo::exists(path); }

/// The candidate directories search() looks in for `share/alphashape/sample`, most specific
/// first: next to the executable (Linux layout), then the macOS bundle's Resources, then the
/// parent of the binary directory (a prefix layout with bin/ + share/).
QStringList sample_directories() {
    const QDir binary(QCoreApplication::applicationDirPath());
    QStringList roots;
    roots << binary.absoluteFilePath(QStringLiteral("share/alphashape/sample"));
    roots << binary.absoluteFilePath(QStringLiteral("../Resources/share/alphashape/sample"));
    roots << binary.absoluteFilePath(QStringLiteral("../share/alphashape/sample"));
    return roots;
}

/// The first *.psx in `directory`, or "" (the sample ships as one project file in a folder).
QString project_in(const QString& directory) {
    const QDir dir(directory);
    if (!dir.exists()) {
        return QString();
    }
    const QStringList entries = dir.entryList({QStringLiteral("*.psx")}, QDir::Files, QDir::Name);
    return entries.isEmpty() ? QString() : dir.absoluteFilePath(entries.first());
}

/// The off-thread half of the recent-list existence sweep: one stat per entry, then a queued
/// call back on the GUI thread. A QRunnable (not QThreadPool::start on a callable) because
/// that overload is newer than the Qt 6.7 floor this target builds against.
class ExistsSweep : public QRunnable {
public:
    using Apply = std::function<void(int, const QStringList&, const QList<bool>&)>;

    ExistsSweep(int generation, QStringList paths, const StartScreen::ExistsProber& prober,
                Apply apply)
        : generation_(generation), paths_(std::move(paths)), prober_(prober),
          apply_(std::move(apply)) {}

    void run() override {
        QList<bool> missing;
        missing.reserve(paths_.size());
        for (const QString& path : paths_) {
            // A prober that throws must not take the pool down; treat it as "cannot tell" and
            // leave the entry ungreyed (present) rather than claiming it is gone.
            bool exists = true;
            try {
                exists = prober_(path);
            } catch (...) {
                exists = true;
            }
            missing.append(!exists);
        }
        apply_(generation_, paths_, missing);
    }

private:
    int generation_;
    QStringList paths_;
    StartScreen::ExistsProber prober_;
    Apply apply_;
};

}  // namespace

StartScreen::StartScreen(GuiApp& app, QWidget* parent) : QWidget(parent), app_(app) {
    exists_prober_ = [](const QString& path) { return QFileInfo::exists(path); };
    setObjectName(QStringLiteral("startScreen"));
    setAcceptDrops(true);
    setFocusPolicy(Qt::StrongFocus);
    // A plain QWidget only paints a style-sheet background when it is told to; without this the
    // #startScreen rule would silently do nothing.
    setAttribute(Qt::WA_StyledBackground, true);

    photos_handler_ = [this]() {
        return QFileDialog::getOpenFileNames(this, tr("Add Photos"), QString(), image_filter());
    };

    build();
    connect(&app_, &GuiApp::recent_files_changed, this, &StartScreen::reload);
    // The "missing" rows take their grey from the palette, so a theme switch must re-apply it.
    connect(&app_.theme(), &Theme::changed, this, &StartScreen::reload);
    reload();
}

StartScreen::~StartScreen() {
    // An outstanding sweep holds this pointer and the flag; clearing it under the GUI thread's
    // own atomic store is enough — the queued lambda checks the flag on this same thread before
    // it ever dereferences the pointer.
    alive_->store(false);
}

// --- Construction ------------------------------------------------------------------------

void StartScreen::build() {
    auto* outer = new QVBoxLayout(this);
    outer->setContentsMargins(32, 32, 32, 32);
    outer->setSpacing(0);
    outer->addStretch(1);

    // The content column: the title block, the entry points, then the recent list. Centred
    // horizontally and capped in width by the row wrapper below.
    auto* row = new QHBoxLayout();
    row->setSpacing(0);
    auto* column_host = new QWidget(this);
    column_host->setObjectName(QStringLiteral("startScreen.content"));
    column_host->setMaximumWidth(kContentWidth);
    auto* column = new QVBoxLayout(column_host);
    column->setContentsMargins(0, 0, 0, 0);
    column->setSpacing(10);

    auto* title = new QLabel(QStringLiteral("alphashape"), column_host);
    title->setObjectName(QStringLiteral("startTitle"));
    column->addWidget(title);

    auto* version = new QLabel(tr("Version %1").arg(QStringLiteral(ALPHASHAPE_GUI_VERSION)),
                               column_host);
    version->setObjectName(QStringLiteral("startVersion"));
    version->setProperty("role", QStringLiteral("secondary"));
    column->addWidget(version);
    column->addSpacing(12);

    new_button_ = new QPushButton(tr("New project"), column_host);
    new_button_->setObjectName(QStringLiteral("start.new"));
    new_button_->setProperty("role", QStringLiteral("primary"));
    new_button_->setToolTip(tr("Start an empty project"));
    connect(new_button_, &QPushButton::clicked, this, &StartScreen::new_requested);
    column->addWidget(new_button_);

    open_button_ = new QPushButton(tr("Open..."), column_host);
    open_button_->setObjectName(QStringLiteral("start.open"));
    open_button_->setToolTip(tr("Open an existing project file"));
    connect(open_button_, &QPushButton::clicked, this, &StartScreen::open_requested);
    column->addWidget(open_button_);

    add_photos_button_ = new QPushButton(tr("Add photos..."), column_host);
    add_photos_button_->setObjectName(QStringLiteral("start.add_photos"));
    add_photos_button_->setToolTip(tr("Start a new project from a set of photos"));
    connect(add_photos_button_, &QPushButton::clicked, this, &StartScreen::on_add_photos);
    column->addWidget(add_photos_button_);

    sample_button_ = new QPushButton(tr("Open sample project"), column_host);
    sample_button_->setObjectName(QStringLiteral("start.sample"));
    sample_button_->setToolTip(tr("Open the project that ships with alphashape"));
    connect(sample_button_, &QPushButton::clicked, this, &StartScreen::on_sample);
    sample_button_->setVisible(false);
    column->addWidget(sample_button_);

    column->addSpacing(16);

    auto* divider = new QFrame(column_host);
    divider->setObjectName(QStringLiteral("start.divider"));
    divider->setProperty("role", QStringLiteral("divider"));
    divider->setFrameShape(QFrame::HLine);
    divider->setFixedHeight(1);
    column->addWidget(divider);
    column->addSpacing(6);

    recent_title_ = new QLabel(tr("Recent projects"), column_host);
    recent_title_->setObjectName(QStringLiteral("startRecentTitle"));
    recent_title_->setProperty("role", QStringLiteral("label"));
    column->addWidget(recent_title_);

    recent_ = new QListWidget(column_host);
    recent_->setObjectName(QStringLiteral("start.recent"));
    recent_->setSelectionMode(QAbstractItemView::SingleSelection);
    recent_->setContextMenuPolicy(Qt::CustomContextMenu);
    recent_->setUniformItemSizes(true);
    recent_->setMinimumHeight(150);
    connect(recent_, &QListWidget::itemClicked, this, &StartScreen::on_recent_clicked);
    connect(recent_, &QListWidget::customContextMenuRequested, this,
            &StartScreen::on_context_menu);
    column->addWidget(recent_, 1);

    empty_ = new QLabel(tr("No recent projects"), column_host);
    empty_->setObjectName(QStringLiteral("start.recent.empty"));
    empty_->setProperty("role", QStringLiteral("secondary"));
    empty_->setAlignment(Qt::AlignLeft);
    column->addWidget(empty_);

    column->addSpacing(10);
    auto* hint = new QLabel(tr("Drag a project file here to open it, or photos and folders to "
                               "start a new project."),
                            column_host);
    hint->setObjectName(QStringLiteral("startHint"));
    hint->setProperty("role", QStringLiteral("secondary"));
    hint->setWordWrap(true);
    column->addWidget(hint);

    row->addStretch(1);
    row->addWidget(column_host);
    row->addStretch(1);
    outer->addLayout(row);
    outer->addStretch(1);
}

// --- Recent projects ---------------------------------------------------------------------

void StartScreen::reload() {
    const QString selected = recent_->currentItem() == nullptr
                                 ? QString()
                                 : recent_->currentItem()->data(Qt::UserRole).toString();
    recent_->clear();
    const QStringList paths = app_.recent_files();
    for (const QString& path : paths) {
        auto* item = new QListWidgetItem(QFileInfo(path).fileName(), recent_);
        item->setData(Qt::UserRole, path);
        item->setToolTip(path);
        paint_recent_row(item, path);
    }
    // Restore the selection by path (the rows may have moved).
    for (int i = 0; i < recent_->count(); ++i) {
        if (recent_->item(i)->data(Qt::UserRole).toString() == selected && !selected.isEmpty()) {
            recent_->setCurrentRow(i);
            break;
        }
    }
    recent_title_->setVisible(!paths.isEmpty());
    recent_->setVisible(!paths.isEmpty());
    empty_->setVisible(paths.isEmpty());

    sample_path_ = find_sample_project().value_or(QString());
    sample_button_->setVisible(!sample_path_.isEmpty());
    set_actions_enabled(actions_enabled_);

    // The rows are on screen now, un-greyed; whether any is gone is answered off the GUI thread,
    // because a recent entry can name a stale network path and stat()ing it here would freeze
    // the window for as long as the mount takes to time out.
    start_recent_sweep(paths);
}

QStringList StartScreen::recent_paths() const {
    QStringList paths;
    paths.reserve(recent_->count());
    for (int i = 0; i < recent_->count(); ++i) {
        paths << recent_->item(i)->data(Qt::UserRole).toString();
    }
    return paths;
}

bool StartScreen::recent_missing(int row) const {
    QListWidgetItem* item = recent_->item(row);
    if (item == nullptr) {
        return true;
    }
    const QString path = item->data(Qt::UserRole).toString();
    const auto known = known_recent_.constFind(path);
    return known != known_recent_.constEnd() && *known;
}

void StartScreen::set_exists_prober(ExistsProber prober) {
    exists_prober_ = prober ? std::move(prober)
                            : ExistsProber([](const QString& path) { return QFileInfo::exists(path); });
}

bool StartScreen::wait_for_recent_checks(int timeout_ms) {
    QElapsedTimer timer;
    timer.start();
    while (recent_checks_pending()) {
        if (timer.elapsed() >= timeout_ms) {
            return false;
        }
        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
        QThread::msleep(2);
    }
    return true;
}

void StartScreen::paint_recent_row(QListWidgetItem* item, const QString& path) {
    if (item == nullptr) {
        return;
    }
    const auto known = known_recent_.constFind(path);
    if (known == known_recent_.constEnd() || !*known) {
        // Unknown or present: the plain row. An unknown entry is NOT greyed — the answer is not
        // in yet, and greying it would claim something we have not checked.
        item->setForeground(QBrush());
        item->setText(QFileInfo(path).fileName());
        item->setToolTip(path);
        return;
    }
    // Greyed through the palette's disabled text colour, so the theme stays the source of the
    // colour; the suffix says it in text for anyone who cannot see the contrast.
    item->setForeground(palette().color(QPalette::Disabled, QPalette::Text));
    item->setText(QFileInfo(path).fileName() + tr("  (missing)"));
    item->setToolTip(tr("%1\nThis project file is missing.").arg(path));
}

void StartScreen::start_recent_sweep(const QStringList& paths) {
    if (paths.isEmpty()) {
        return;
    }
    // Drop answers for entries that are no longer listed, so the map cannot grow forever.
    for (auto it = known_recent_.begin(); it != known_recent_.end();) {
        if (paths.contains(it.key())) {
            ++it;
        } else {
            it = known_recent_.erase(it);
        }
    }

    const int generation = ++sweep_generation_;
    ++pending_sweeps_;
    // The worker must not touch a QPointer: a widget's QPointer belongs to the GUI thread. It
    // carries a plain flag instead, and the answer is posted to the application object (which
    // lives until the program ends and runs on the GUI thread). The screen is read only inside
    // that queued lambda, on the thread that can also destroy it.
    StartScreen* self = this;
    std::shared_ptr<std::atomic<bool>> alive = alive_;
    auto apply = [self, alive](int sweep, const QStringList& swept, const QList<bool>& missing) {
        QCoreApplication* target = QCoreApplication::instance();
        if (target == nullptr || !alive->load()) {
            return;  // shutting down, or the screen is already gone
        }
        QMetaObject::invokeMethod(
            target,
            [self, alive, sweep, swept, missing]() {
                if (alive->load()) {
                    self->apply_recent_sweep(sweep, swept, missing);
                }
            },
            Qt::QueuedConnection);
    };
    QThreadPool::globalInstance()->start(
        new ExistsSweep(generation, paths, exists_prober_, std::move(apply)));
}

void StartScreen::apply_recent_sweep(int generation, const QStringList& paths,
                                     const QList<bool>& missing) {
    if (pending_sweeps_ > 0) {
        --pending_sweeps_;
    }
    // A newer reload superseded this sweep: its answers describe a list that is no longer shown.
    if (generation != sweep_generation_) {
        return;
    }
    for (qsizetype i = 0; i < paths.size() && i < missing.size(); ++i) {
        known_recent_.insert(paths[i], missing[i]);
    }
    // Repaint from that stored state, every row: the tooltip and the "(missing)" suffix are
    // built here, so it must not depend on what the row happened to carry before.
    for (int row = 0; row < recent_->count(); ++row) {
        paint_recent_row(recent_->item(row), recent_->item(row)->data(Qt::UserRole).toString());
    }
}

void StartScreen::remove_recent(int row) {
    QListWidgetItem* item = recent_->item(row);
    if (item == nullptr) {
        return;
    }
    const QString path = item->data(Qt::UserRole).toString();
    app_.remove_recent_file(path);  // rebuilds the list through recent_files_changed
    Q_EMIT recent_removed(path);
}

void StartScreen::on_recent_clicked(QListWidgetItem* item) {
    if (item == nullptr || !actions_enabled_) {
        return;
    }
    const QString path = item->data(Qt::UserRole).toString();
    if (recent_missing(recent_->row(item))) {
        return;  // the greyed entry is a record of a file that is gone; use Remove to drop it
    }
    // A click is a user action, and opening the file will stat it anyway: a path whose check is
    // still outstanding is verified here rather than refused.
    if (known_recent_.constFind(path) == known_recent_.constEnd() && !project_exists(path)) {
        known_recent_.insert(path, true);
        paint_recent_row(item, path);
        return;
    }
    Q_EMIT open_path_requested(path);
}

void StartScreen::on_context_menu(const QPoint& position) {
    std::unique_ptr<QMenu> menu(build_recent_menu(position));
    if (menu) {
        menu->exec(recent_->viewport()->mapToGlobal(position));
    }
}

int StartScreen::row_at(const QPoint& position) const {
    const QModelIndex index = recent_->indexAt(position);
    return index.isValid() ? index.row() : -1;
}

QMenu* StartScreen::build_recent_menu(const QPoint& position) {
    return build_recent_menu(row_at(position));
}

QMenu* StartScreen::build_recent_menu(int row) {
    if (row < 0 || row >= recent_->count() || !actions_enabled_) {
        return nullptr;
    }
    auto* menu = new QMenu(this);
    menu->setObjectName(QStringLiteral("menu.start.recent"));
    QAction* open = menu->addAction(tr("Open"));
    open->setObjectName(QStringLiteral("action.start.recent.open"));
    open->setEnabled(!recent_missing(row));
    QAction* remove = menu->addAction(tr("Remove from Recent Projects"));
    remove->setObjectName(QStringLiteral("action.start.recent.remove"));
    QObject::connect(open, &QAction::triggered, this,
                     [this, row]() { on_recent_clicked(recent_->item(row)); });
    QObject::connect(remove, &QAction::triggered, this, [this, row]() { remove_recent(row); });
    return menu;
}

// --- Entry points ------------------------------------------------------------------------

void StartScreen::set_actions_enabled(bool enabled) {
    actions_enabled_ = enabled;
    for (QPushButton* button : {new_button_, open_button_, add_photos_button_}) {
        button->setEnabled(enabled);
    }
    if (sample_button_ != nullptr) {
        sample_button_->setEnabled(enabled && !sample_path_.isEmpty());
    }
}

void StartScreen::set_photos_handler(PhotosHandler handler) {
    photos_handler_ = std::move(handler);
}

void StartScreen::on_add_photos() {
    if (!actions_enabled_) {
        return;
    }
    QStringList files;
    try {
        files = photos_handler_ ? photos_handler_() : QStringList();
    } catch (const std::exception&) {
        files.clear();  // a picker that cannot run is a cancelled picker
    }
    if (!files.isEmpty()) {
        Q_EMIT add_photos_requested(files);
    }
}

void StartScreen::on_sample() {
    if (actions_enabled_ && !sample_path_.isEmpty()) {
        Q_EMIT open_path_requested(sample_path_);
    }
}

std::optional<QString> StartScreen::find_sample_project() {
    const QByteArray configured = qgetenv(kSampleEnv);
    if (!configured.isEmpty()) {
        const QString path = QString::fromLocal8Bit(configured);
        // The variable may name the project file, or the folder it lives in.
        if (QFileInfo(path).isDir()) {
            const QString inside = project_in(path);
            if (!inside.isEmpty()) {
                return inside;
            }
        } else if (project_exists(path)) {
            return path;
        }
        return std::nullopt;  // configured but unusable: hide the entry point, do not fall back
    }
    for (const QString& directory : sample_directories()) {
        const QString path = project_in(directory);
        if (!path.isEmpty()) {
            return path;
        }
    }
    return std::nullopt;
}

// --- Drag and drop -----------------------------------------------------------------------

QStringList StartScreen::local_paths(const QMimeData* mime) {
    QStringList paths;
    if (mime == nullptr || !mime->hasUrls()) {
        return paths;
    }
    for (const QUrl& url : mime->urls()) {
        if (!url.isLocalFile()) {
            continue;
        }
        const QString path = url.toLocalFile();
        if (!path.isEmpty()) {
            paths << path;
        }
    }
    return paths;
}

namespace {

/// Whether a drag carries anything the window can open or import. This is decided once, in
/// dragEnter, and cached: a drag-MOVE event fires continuously while the cursor is over the
/// widget, and a stat() per event would stall the cursor over a dead network path.
bool has_usable_drop(const QStringList& paths) {
    for (const QString& path : paths) {
        if (is_project_path(path) || is_image_path(path) || QFileInfo(path).isDir()) {
            return true;
        }
    }
    return false;
}

}  // namespace

void StartScreen::dragEnterEvent(QDragEnterEvent* event) {
    drag_accepts_ = actions_enabled_ && has_usable_drop(local_paths(event->mimeData()));
    if (drag_accepts_) {
        event->acceptProposedAction();
        return;
    }
    QWidget::dragEnterEvent(event);
}

void StartScreen::dragMoveEvent(QDragMoveEvent* event) {
    // The verdict from dragEnter: this event repeats tens of times a second.
    if (drag_accepts_) {
        event->acceptProposedAction();
        return;
    }
    QWidget::dragMoveEvent(event);
}

void StartScreen::dragLeaveEvent(QDragLeaveEvent* event) {
    drag_accepts_ = false;
    QWidget::dragLeaveEvent(event);
}

void StartScreen::dropEvent(QDropEvent* event) {
    const QStringList paths = local_paths(event->mimeData());
    // Re-decide at the drop: a path may have appeared or vanished since the drag started.
    const bool usable = actions_enabled_ && has_usable_drop(paths);
    drag_accepts_ = false;
    if (!usable) {
        QWidget::dropEvent(event);
        return;
    }
    event->acceptProposedAction();
    Q_EMIT paths_dropped(paths);
}

bool StartScreen::drop_paths(const QStringList& paths) {
    if (!actions_enabled_) {
        return false;
    }
    // A project file that exists, an image, or a folder: the same verdict a real drop gets. A
    // path that is gone, or is any other kind of file, is not something this window can act on.
    // The whole drop is forwarded, as drag-and-drop does: MainWindow counts what it ignored and
    // says so in the status bar.
    if (classify_dropped_paths(paths).empty()) {
        return false;
    }
    Q_EMIT paths_dropped(paths);
    return true;
}

void StartScreen::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    reload();
}

}  // namespace alphashape::gui
