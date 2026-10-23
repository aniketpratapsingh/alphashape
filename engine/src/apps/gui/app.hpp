#pragma once

// GuiApp: the state that outlives any window. It owns the settings store, the engine
// Document that the windows edit, the UndoStack bound to that document, and the Theme.
// (Job manager, Python host and action host arrive with the WPs that need them.)
//
// A QApplication must exist before a GuiApp is constructed.

#include <memory>
#include <string>

#include <QObject>
#include <QSettings>

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

    /// Window title fragment for the current document ("Untitled" when it has no path).
    QString document_title() const;

signals:
    /// Emitted after new_document() / open_document() swapped the document.
    void document_replaced();

private:
    QSettings settings_;
    std::shared_ptr<Document> document_;
    UndoStack undo_stack_;
    Theme theme_;
};

}  // namespace alphashape::gui
