#include <gtest/gtest.h>

#include <QApplication>
#include <QCryptographicHash>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QString>

#include "apps/gui/theme/theme.hpp"

using alphashape::gui::Theme;
using alphashape::gui::ThemeMode;
using alphashape::gui::ViewportBackground;

namespace {

// Counts emissions of a signal (QtTest's QSignalSpy would pull in another Qt module).
struct SignalCounter {
    template <typename Sender, typename Signal>
    SignalCounter(Sender* sender, Signal signal) {
        QObject::connect(sender, signal, sender, [this](ThemeMode) { ++n; });
    }
    int count() const { return n; }
    int n = 0;
};

QString repo_path(const char* relative) {
    return QStringLiteral(ALPHASHAPE_REPO_ROOT "/") + QString::fromLatin1(relative);
}

QByteArray read_file(const QString& path) {
    QFile file(path);
    EXPECT_TRUE(file.open(QIODevice::ReadOnly)) << path.toStdString();
    return file.readAll().replace("\r\n", "\n");
}

QJsonObject load_tokens() {
    return QJsonDocument::fromJson(read_file(repo_path("design/system/tokens.json"))).object();
}

// A spec is either "#RRGGBB" or a token path such as "surface.app" under color.<mode>.
QColor resolve(const QJsonObject& tokens, ThemeMode mode, const QString& spec) {
    if (spec.startsWith(QLatin1Char('#'))) {
        return QColor(spec);
    }
    QJsonValue node = tokens.value(QStringLiteral("color"))
                          .toObject()
                          .value(mode == ThemeMode::Dark ? QStringLiteral("dark") : QStringLiteral("light"));
    for (const QString& part : spec.split(QLatin1Char('.'))) {
        node = node.toObject().value(part);
    }
    EXPECT_TRUE(node.isString()) << "no such token: " << spec.toStdString();
    return QColor(node.toString());
}

struct RoleSpec {
    QPalette::ColorRole role;
    const char* dark;
    const char* light;
};

// qt-mapping.md section 1, role by role. Tokens where the mapping names one; literals otherwise.
const RoleSpec kRoles[] = {
    {QPalette::Window, "surface.app", "surface.app"},
    {QPalette::WindowText, "text.primary", "text.primary"},
    {QPalette::Base, "surface.panel", "surface.raised"},
    {QPalette::AlternateBase, "surface.raised", "#F2F5F5"},
    {QPalette::Text, "text.primary", "text.primary"},
    {QPalette::PlaceholderText, "text.disabled", "text.disabled"},
    {QPalette::Button, "surface.raised", "surface.raised"},
    {QPalette::ButtonText, "text.primary", "text.primary"},
    {QPalette::BrightText, "#FFFFFF", "#000000"},
    {QPalette::Highlight, "selection.bg", "selection.bg"},
    {QPalette::HighlightedText, "text.primary", "text.primary"},
    {QPalette::Accent, "accent.base", "accent.base"},
    {QPalette::Link, "text.link", "accent.text"},
    {QPalette::LinkVisited, "status.processing", "status.processing"},
    {QPalette::ToolTipBase, "surface.overlay", "surface.overlay"},
    {QPalette::ToolTipText, "text.primary", "text.primary"},
    {QPalette::Light, "border.default", "#FFFFFF"},
    {QPalette::Midlight, "border.subtle", "#F2F5F5"},
    {QPalette::Mid, "border.default", "border.default"},
    {QPalette::Dark, "border.subtle", "border.subtle"},
    {QPalette::Shadow, "#000000", "border.strong"},
};

}  // namespace

TEST(Theme, PaletteMatchesTokens) {
    const QJsonObject tokens = load_tokens();
    ASSERT_FALSE(tokens.isEmpty());
    for (ThemeMode mode : {ThemeMode::Dark, ThemeMode::Light}) {
        const QPalette palette = Theme::palette_for(mode);
        for (QPalette::ColorGroup group : {QPalette::Active, QPalette::Inactive, QPalette::Disabled}) {
            for (const RoleSpec& spec : kRoles) {
                const char* source = (mode == ThemeMode::Dark) ? spec.dark : spec.light;
                QColor expected = resolve(tokens, mode, QString::fromLatin1(source));
                // Inactive: calmer selection. Disabled: text roles use text.disabled.
                if (group == QPalette::Inactive && spec.role == QPalette::Highlight) {
                    expected = resolve(tokens, mode, QStringLiteral("selection.bgInactive"));
                }
                if (group == QPalette::Disabled &&
                    (spec.role == QPalette::WindowText || spec.role == QPalette::Text ||
                     spec.role == QPalette::ButtonText)) {
                    expected = resolve(tokens, mode, QStringLiteral("text.disabled"));
                }
                const QColor actual = palette.color(group, spec.role);
                EXPECT_EQ(actual.rgb(), expected.rgb())
                    << alphashape::gui::theme_name(mode) << " group " << static_cast<int>(group)
                    << " role " << static_cast<int>(spec.role) << " (" << source << "): got "
                    << actual.name().toStdString() << " want " << expected.name().toStdString();
            }
        }
    }
}

TEST(Theme, QssMatchesTokens) {
    // 1. The checked-in sheets are exactly what the emitter produces from tokens.json.
    QProcess python;
    python.start(QStringLiteral(ALPHASHAPE_PYTHON),
                 {repo_path("tools/theme/compile_qss.py"), QStringLiteral("--check")});
    ASSERT_TRUE(python.waitForFinished(60000)) << "compile_qss.py did not finish";
    EXPECT_EQ(python.exitCode(), 0) << python.readAllStandardError().toStdString();

    // 2. Each sheet records the hash of the tokens it was built from.
    const QByteArray hash =
        QCryptographicHash::hash(read_file(repo_path("design/system/tokens.json")), QCryptographicHash::Sha256).toHex();
    // 3. The sheets compiled into the binary are the checked-in ones.
    for (ThemeMode mode : {ThemeMode::Dark, ThemeMode::Light}) {
        const QString name = QStringLiteral("design/system/alphashape-%1.qss")
                                 .arg(QString::fromLatin1(alphashape::gui::theme_name(mode)));
        const QByteArray on_disk = read_file(repo_path(name.toLatin1().constData()));
        EXPECT_TRUE(on_disk.contains("tokens-sha256: " + hash)) << name.toStdString();
        EXPECT_EQ(Theme::style_sheet_for(mode).toUtf8().replace("\r\n", "\n"), on_disk) << name.toStdString();
    }
}

TEST(Theme, SwitchesAtRuntime) {
    Theme theme;
    SignalCounter changed(&theme, &Theme::changed);

    theme.apply(ThemeMode::Dark);
    EXPECT_EQ(changed.count(), 1);
    EXPECT_EQ(qApp->palette().color(QPalette::Window).rgb(), Theme::palette_for(ThemeMode::Dark).color(QPalette::Window).rgb());
    EXPECT_EQ(qApp->styleSheet(), Theme::style_sheet_for(ThemeMode::Dark));

    theme.apply(ThemeMode::Light);
    EXPECT_EQ(changed.count(), 2);
    EXPECT_EQ(theme.mode(), ThemeMode::Light);
    EXPECT_EQ(qApp->palette().color(QPalette::Window).rgb(), Theme::palette_for(ThemeMode::Light).color(QPalette::Window).rgb());
    EXPECT_EQ(qApp->styleSheet(), Theme::style_sheet_for(ThemeMode::Light));
    EXPECT_NE(Theme::style_sheet_for(ThemeMode::Dark), Theme::style_sheet_for(ThemeMode::Light));

    theme.apply(ThemeMode::Light);  // no change, no signal
    EXPECT_EQ(changed.count(), 2);

    theme.apply(ThemeMode::Dark);
    EXPECT_EQ(changed.count(), 3);
    EXPECT_EQ(qApp->styleSheet(), Theme::style_sheet_for(ThemeMode::Dark));
}

TEST(Theme, ViewportCanStayDarkUnderLightTheme) {
    Theme theme;
    theme.apply(ThemeMode::Light);
    EXPECT_EQ(theme.viewport_mode(), ThemeMode::Light);  // Follow theme

    SignalCounter viewport(&theme, &Theme::viewport_changed);
    theme.set_viewport_background(ViewportBackground::Dark);
    EXPECT_EQ(theme.viewport_mode(), ThemeMode::Dark);
    EXPECT_EQ(viewport.count(), 1);

    theme.apply(ThemeMode::Dark);
    theme.apply(ThemeMode::Light);
    EXPECT_EQ(theme.viewport_mode(), ThemeMode::Dark);
}

TEST(Theme, BundledFontsLoad) {
    EXPECT_EQ(Theme::load_bundled_fonts(), 2);
}
