#pragma once

// GuiApp: the state that outlives any window. It owns the settings store, the engine
// Document that the windows edit, the UndoStack bound to that document, and the Theme.
// (Job manager, Python host and action host arrive with the WPs that need them.)
//
// A QApplication must exist before a GuiApp is constructed.

#include <cstddef>
#include <memory>
#include <string>

#include <QObject>
#include <QSettings>
#include <QStringList>

#include "apps/gui/command_line.hpp"
#include "apps/gui/theme/theme.hpp"
#include "common/commands/command.hpp"
#include "common/core/document.hpp"

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
    /// Undo position at the last load/save; npos when that state is no longer reachable
    /// (a new edit overwrote the redo branch that led back to it).
    std::size_t clean_position_ = 0;
    bool force_dirty_ = false;
    bool last_dirty_ = false;

    void reset_dirty_tracking();
    void publish_dirty();
};

}  // namespace alphashape::gui
