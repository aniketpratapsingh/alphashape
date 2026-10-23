#pragma once

// GuiApp: the state that outlives any window. It owns the settings store, the engine
// Document that the windows edit, the UndoStack bound to that document, and the Theme.
// The JobManager runs engine tasks on a worker thread; while it is busy, everything that would
// replace, save or edit the document throws (the window disables those actions up front).
// (The Python host and action host arrive with the WPs that need them.)
//
// A QApplication must exist before a GuiApp is constructed.

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include <QObject>
#include <QSettings>
#include <QStringList>

#include "apps/gui/command_line.hpp"
#include "apps/gui/job_manager.hpp"
#include "apps/gui/theme/theme.hpp"
#include "base/utils/log.hpp"
#include "common/commands/command.hpp"
#include "common/core/document.hpp"
#include "common/core/document_lock.hpp"

namespace alphashape::gui {

class GuiApp : public QObject {
    Q_OBJECT

public:
    /// Applies --reset-settings, loads the bundled fonts, resolves and applies the theme
    /// (--theme, else the stored choice, else dark) and opens or creates the project the
    /// command line asks for. Throws alphashape::Error if --open cannot be read.
    explicit GuiApp(const Options& options, QObject* parent = nullptr);
    ~GuiApp() override;

    QSettings& settings() { return settings_; }
    Theme& theme() { return theme_; }
    Document& document() { return *document_; }
    UndoStack& undo_stack() { return undo_stack_; }
    JobManager& jobs() { return jobs_; }

    /// Held exclusively by a running task; readers that run alongside jobs take it shared.
    /// Rule for GUI panes: refresh from the document only when no job is active, i.e. on
    /// JobManager::job_finished / busy_changed(false); never read it while busy().
    DocumentLock& document_lock() { return document_lock_; }

    /// Run `fn(const Document&)` under the shared document lock and return its result. Use this
    /// for GUI-thread reads (title, path, ...) so they never overlap a running task's writes.
    template <typename Fn>
    auto read_document(Fn&& fn) const {
        const auto lock = document_lock_.read();
        return std::forward<Fn>(fn)(std::as_const(*document_));
    }
    /// The same document, shared with the job worker so it outlives a swap.
    std::shared_ptr<Document> shared_document() { return document_; }

    /// Called by the job manager after a task ended (also Failed / Cancelled once it had started,
    /// since tasks mutate in place and may leave partial changes). Tasks are not reversible through the
    /// command layer (there is no reversal command for them yet), so the undo history, whose
    /// entries assume the pre-task state, is cleared and the document is marked dirty.
    void task_changed_document();

    /// Switch theme at runtime and remember the choice for the next start.
    void set_theme(ThemeMode mode);

    /// Replace the document with an empty one (to be saved at `path`, if non-empty). Clears
    /// the undo history.
    void new_document(const std::string& path);

    /// Replace the document with the project at `path`. On failure the current document is
    /// untouched and alphashape::Error propagates.
    void open_document(const std::string& path);

    /// Write the document to its own path. Throws alphashape::Error on I/O failure, or when
    /// the document has no path yet (use save_document_as()). Marks the document clean.
    void save_document();

    /// Write the document to `path` and make that its path. On failure the path is unchanged
    /// and alphashape::Error propagates. Marks the document clean and records `path` as recent.
    void save_document_as(const std::string& path);

    /// Window title fragment for the current document ("Untitled" when it has no path).
    QString document_title() const;

    /// True when the document differs from what was last loaded or saved. Edits made through
    /// execute()/undo()/redo() maintain it; mark_dirty() covers changes made another way.
    bool dirty() const;
    void mark_dirty();

    /// Run a command through the undo stack, or step the stack. The GUI goes through these (not
    /// through undo_stack() directly) so the dirty flag and undo_changed() stay accurate.
    void execute(std::unique_ptr<AbstractCommand> command);
    void undo();
    void redo();

    /// Most recently used project paths, newest first (at most kMaxRecentFiles).
    static constexpr int kMaxRecentFiles = 10;
    QStringList recent_files() const;
    void add_recent_file(const QString& path);
    void remove_recent_file(const QString& path);
    void clear_recent_files();

Q_SIGNALS:
    /// One line for the Log pane: engine log lines, and job start / stage / end notes. Always
    /// emitted on the GUI thread.
    void log_message(alphashape::LogLevel level, const QString& text);
    /// Emitted after new_document() / open_document() swapped the document.
    void document_replaced();
    /// Emitted when the document keeps its content but gets a new file path (Save As).
    void path_changed();
    /// Emitted when the dirty flag may have changed (always after a change, never spuriously
    /// for a no-op).
    void dirty_changed(bool dirty);
    /// Emitted after execute()/undo()/redo(), and after the document was replaced.
    void undo_changed();
    void recent_files_changed();

private:
    QSettings settings_;
    std::shared_ptr<Document> document_;
    UndoStack undo_stack_;
    Theme theme_;
    mutable DocumentLock document_lock_;
    JobManager jobs_;
    /// Undo position at the last load/save; npos when that state is no longer reachable
    /// (a new edit overwrote the redo branch that led back to it).
    std::size_t clean_position_ = 0;
    bool force_dirty_ = false;
    bool last_dirty_ = false;

    void require_idle(const char* what) const;
    void reset_dirty_tracking();
    void publish_dirty();
};

}  // namespace alphashape::gui

Q_DECLARE_METATYPE(alphashape::LogLevel)
