#include "apps/gui/log_pane.hpp"

#include <QApplication>
#include <QClipboard>
#include <QComboBox>
#include <QFontDatabase>
#include <QHBoxLayout>
#include <QPalette>
#include <QScrollBar>
#include <QStringList>
#include <QTextCharFormat>
#include <QTextCursor>
#include <QToolButton>
#include <QVBoxLayout>

#include "apps/gui/app.hpp"
#include "apps/gui/theme/theme.hpp"

namespace alphashape::gui {

namespace {

/// Trimmed to what an entry can be at kMaxLines; the widget keeps the same bound so the two
/// can never disagree.
constexpr int kMaxBlocks = static_cast<int>(LogView::kMaxLines);

QString level_name(LogLevel level) {
    switch (level) {
        case LogLevel::Trace: return QStringLiteral("TRACE");
        case LogLevel::Debug: return QStringLiteral("DEBUG");
        case LogLevel::Info: return QStringLiteral("INFO");
        case LogLevel::Warn: return QStringLiteral("WARN");
        case LogLevel::Error: return QStringLiteral("ERROR");
        case LogLevel::Off: return QStringLiteral("OFF");
    }
    return QStringLiteral("?");
}

}  // namespace

// --- LogView -------------------------------------------------------------------------------

LogView::LogView(QWidget* parent) : QPlainTextEdit(parent) {
    setObjectName(QStringLiteral("log.view"));
    setReadOnly(true);
    setUndoRedoEnabled(false);
    setLineWrapMode(QPlainTextEdit::NoWrap);
    setMaximumBlockCount(kMaxBlocks);
    setCenterOnScroll(false);
    QFont console = QFontDatabase::systemFont(QFontDatabase::FixedFont);
    console.setPointSizeF(11.0);
    setFont(console);
}

bool LogView::visible(LogLevel level) const {
    if (minimum_ == LogLevel::Off) {
        return false;
    }
    return level >= minimum_ && level != LogLevel::Off;
}

QColor LogView::colour_for(LogLevel level) const {
    switch (level) {
        case LogLevel::Error:
            return Theme::status_error(mode_);
        case LogLevel::Warn:
            return Theme::status_warning(mode_);
        case LogLevel::Info:
            return palette().color(QPalette::Text);
        default:
            return palette().color(QPalette::Disabled, QPalette::Text);
    }
}

void LogView::set_theme_mode(ThemeMode mode) {
    if (mode == mode_) {
        return;
    }
    mode_ = mode;
    rebuild_document();
}

void LogView::apply_line_format(LogLevel level, const QString& text) {
    QTextCursor cursor(document());
    cursor.movePosition(QTextCursor::End);
    QTextCharFormat format;
    format.setForeground(colour_for(level));
    cursor.insertText(level_name(level) + QStringLiteral("  ") + text + QLatin1Char('\n'),
                      format);
}

void LogView::append(LogEntry entry) {
    if (entries_.size() >= kMaxLines) {
        entries_.pop_front();
    }
    const QString text = entry.text;
    entries_.push_back(std::move(entry));
    if (!visible(entries_.back().level)) {
        return;
    }
    const bool at_bottom =
        verticalScrollBar()->value() >= verticalScrollBar()->maximum() - 4;
    apply_line_format(entries_.back().level, text);
    if (at_bottom) {
        verticalScrollBar()->setValue(verticalScrollBar()->maximum());
    }
}

void LogView::append(const QList<LogEntry>& entries) {
    for (const LogEntry& entry : entries) {
        append(entry);
    }
}

void LogView::rebuild_document() {
    clear();
    QTextCursor cursor(document());
    QTextCharFormat format;
    for (const LogEntry& entry : entries_) {
        if (!visible(entry.level)) {
            continue;
        }
        format.setForeground(colour_for(entry.level));
        cursor.insertText(level_name(entry.level) + QStringLiteral("  ") + entry.text +
                              QLatin1Char('\n'),
                          format);
    }
    verticalScrollBar()->setValue(verticalScrollBar()->maximum());
}

void LogView::set_minimum_level(LogLevel lowest) {
    if (lowest == minimum_) {
        return;
    }
    minimum_ = lowest;
    rebuild_document();
}

void LogView::clear_entries() {
    entries_.clear();
    clear();
}

QString LogView::plain_text() const {
    // What Copy yields: the visible lines (the filter applied) without the level decoration.
    QStringList lines;
    lines.reserve(static_cast<qsizetype>(entries_.size()));
    for (const LogEntry& entry : entries_) {
        if (visible(entry.level)) {
            lines << entry.text;
        }
    }
    return lines.join(QLatin1Char('\n'));
}

// --- LogPane -------------------------------------------------------------------------------

LogPane::LogPane(GuiApp& app, QWidget* parent) : QWidget(parent), app_(app) {
    setObjectName(QStringLiteral("pane.log"));

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    auto* header = new QWidget(this);
    header->setObjectName(QStringLiteral("log.header"));
    auto* header_layout = new QHBoxLayout(header);
    header_layout->setContentsMargins(6, 4, 6, 4);
    header_layout->setSpacing(6);

    level_ = new QComboBox(header);
    level_->setObjectName(QStringLiteral("log.filter"));
    level_->setToolTip(tr("Show lines at this level and above"));
    // A minimum level rather than "only this level": nobody wants to lose the errors when they
    // filter for warnings.
    const struct {
        LogLevel level;
        const char* text;
    } levels[] = {
        {LogLevel::Trace, QT_TRANSLATE_NOOP("alphashape::gui::LogPane", "All levels")},
        {LogLevel::Info, QT_TRANSLATE_NOOP("alphashape::gui::LogPane", "Info and above")},
        {LogLevel::Warn, QT_TRANSLATE_NOOP("alphashape::gui::LogPane", "Warnings and errors")},
        {LogLevel::Error, QT_TRANSLATE_NOOP("alphashape::gui::LogPane", "Errors only")},
    };
    for (const auto& item : levels) {
        level_->addItem(tr(item.text), static_cast<int>(item.level));
    }
    level_->setCurrentIndex(0);  // All levels: nothing the engine emitted is hidden by default
    header_layout->addWidget(level_);
    header_layout->addStretch(1);

    copy_ = new QToolButton(header);
    copy_->setObjectName(QStringLiteral("log.copy"));
    copy_->setText(tr("Copy"));
    copy_->setToolTip(tr("Copy the visible log to the clipboard"));
    header_layout->addWidget(copy_);

    clear_ = new QToolButton(header);
    clear_->setObjectName(QStringLiteral("log.clear"));
    clear_->setText(tr("Clear"));
    clear_->setToolTip(tr("Clear the log"));
    header_layout->addWidget(clear_);
    layout->addWidget(header);

    view_ = new LogView(this);
    view_->set_theme_mode(app_.theme().mode());
    view_->set_minimum_level(static_cast<LogLevel>(level_->currentData().toInt()));
    layout->addWidget(view_, 1);

    connect(&app_.theme(), &Theme::changed, this,
            [this](ThemeMode mode) { view_->set_theme_mode(mode); });
    clipboard_writer_ = [](const QString& text) { QApplication::clipboard()->setText(text); };
    connect(level_, &QComboBox::currentIndexChanged, this, [this](int) {
        set_minimum_level(static_cast<LogLevel>(level_->currentData().toInt()));
        update_actions();
    });
    connect(copy_, &QToolButton::clicked, this, [this]() { copy_all(); });
    connect(clear_, &QToolButton::clicked, this, [this]() {
        view_->clear_entries();
        update_actions();
    });

    // Lines arrive on the GUI thread by contract (GuiApp's sink marshals them), so appending is
    // a plain slot call; a queued connection would also work but loses ordering under load.
    connect(&app_, &GuiApp::log_message, this,
            [this](LogLevel level, const QString& text) {
                view_->append(LogEntry{level, text});
                update_actions();
            });
    update_actions();
}

void LogPane::set_minimum_level(LogLevel level) {
    const int index = level_->findData(static_cast<int>(level));
    if (index >= 0) {
        level_->setCurrentIndex(index);  // currentIndexChanged applies it
    }
}

void LogPane::set_clipboard_writer(ClipboardWriter writer) {
    clipboard_writer_ = std::move(writer);
}

QString LogPane::copy_all() {
    const QString text = view_->plain_text();
    if (!text.isEmpty() && clipboard_writer_) {
        clipboard_writer_(text);
    }
    return text;
}

void LogPane::update_actions() {
    const bool has_lines = view_->line_count() > 0;
    copy_->setEnabled(has_lines);
    clear_->setEnabled(has_lines);
}

}  // namespace alphashape::gui
