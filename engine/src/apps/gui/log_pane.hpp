#pragma once

// LogPane: the Log dock. Every engine log line and every job note (JobManager publishes them
// through GuiApp::log_message, which is always emitted on the GUI thread) lands here in a
// read-only console, coloured by level, capped at kMaxLines so a long run cannot grow without
// bound. Clear, Copy and a level filter are the actions.
//
// Engine lines reach GuiApp::log_message through the process-wide Log::set_sink that GuiApp
// installs; the sink runs on whichever thread logged and marshals with a queued call. This pane
// only ever sees the GUI-thread signal, so none of it needs a lock.

#include <cstddef>
#include <deque>
#include <functional>

#include <QColor>
#include <QList>
#include <QPlainTextEdit>
#include <QString>
#include <QWidget>

#include "apps/gui/theme/theme_mode.hpp"
#include "base/utils/log.hpp"

class QComboBox;
class QToolButton;

namespace alphashape::gui {

class GuiApp;

/// One console line: what was said and at which level.
struct LogEntry {
    alphashape::LogLevel level = alphashape::LogLevel::Info;
    QString text;
};

/// The console widget itself (no app dependency, so the model of the log can be tested alone).
class LogView : public QPlainTextEdit {
    Q_OBJECT

public:
    /// Lines kept; the oldest are dropped beyond this.
    static constexpr std::size_t kMaxLines = 10000;

    explicit LogView(QWidget* parent = nullptr);

    /// Append one line. A line at a level the filter hides is kept (it shows up when the filter
    /// widens) but not displayed.
    void append(LogEntry entry);
    /// Append a batch; one repaint instead of one per line.
    void append(const QList<LogEntry>& entries);

    /// Show only lines at `lowest` and above. Off hides everything the filter can hide.
    void set_minimum_level(alphashape::LogLevel lowest);
    alphashape::LogLevel minimum_level() const { return minimum_; }

    /// Drop every line (both the kept history and the display).
    void clear_entries();

    /// Which theme's status colours the levels are drawn in. The pane follows Theme::changed.
    void set_theme_mode(alphashape::gui::ThemeMode mode);
    ThemeMode theme_mode() const { return mode_; }

    std::size_t line_count() const { return entries_.size(); }
    const std::deque<LogEntry>& entries() const { return entries_; }
    /// The visible text, one line per entry, without level decoration.
    QString plain_text() const;

Q_SIGNALS:
    /// Lines were added, cleared, or the filter changed what is visible.
    void changed();

private:
    void rebuild_document();
    void apply_line_format(alphashape::LogLevel level, const QString& text);
    bool visible(alphashape::LogLevel level) const;
    QColor colour_for(alphashape::LogLevel level) const;

    std::deque<LogEntry> entries_;
    alphashape::LogLevel minimum_ = alphashape::LogLevel::Debug;
    ThemeMode mode_ = ThemeMode::Dark;
};

class LogPane : public QWidget {
    Q_OBJECT

public:
    explicit LogPane(GuiApp& app, QWidget* parent = nullptr);

    LogView* view() { return view_; }
    QComboBox* level_box() { return level_; }
    QToolButton* clear_button() { return clear_; }
    QToolButton* copy_button() { return copy_; }

    /// Where Copy sends the visible text. Default: the system clipboard. A seam for the tests,
    /// which run on the offscreen platform and have no clipboard of their own.
    using ClipboardWriter = std::function<void(const QString&)>;
    void set_clipboard_writer(ClipboardWriter writer);

    /// Copy the visible console text and return it (empty when there is nothing to copy).
    QString copy_all();
    void set_minimum_level(alphashape::LogLevel level);

private:
    void update_actions();

    GuiApp& app_;
    LogView* view_ = nullptr;
    QComboBox* level_ = nullptr;
    QToolButton* clear_ = nullptr;
    QToolButton* copy_ = nullptr;
    ClipboardWriter clipboard_writer_;
};

}  // namespace alphashape::gui
