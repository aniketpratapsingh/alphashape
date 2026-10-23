#include "harness.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QMutex>
#include <QMutexLocker>
#include <QPixmap>
#include <QThread>
#include <QtGlobal>

namespace alphashape::gui::testing {

namespace {

QMutex g_mutex;
QStringList g_messages;
QtMessageHandler g_previous = nullptr;

// Messages that come from Qt's platform layer rather than from our code and carry no action:
//  - the offscreen QPA plugin has no window-manager size hints and says so for every top-level;
//  - the font database reports (with a timing that varies per machine) that it resolved the
//    generic "sans-serif" of the style sheet's font list through an alias;
//  - on Windows only, the offscreen plugin has no native font database and complains about the
//    missing Qt font directory (the real "windows" platform plugin does not).
bool is_known_platform_noise(const QString& message) {
    return message.startsWith(QLatin1String("This plugin does not support propagateSizeHints")) ||
           message.startsWith(QLatin1String("Populating font family aliases took")) ||
#ifdef Q_OS_WIN
           message.startsWith(QLatin1String("QFontDatabase: Cannot find font directory")) ||
#endif
           false;
}

void capturing_handler(QtMsgType type, const QMessageLogContext& context, const QString& message) {
    if (type == QtWarningMsg || type == QtCriticalMsg || type == QtFatalMsg) {
        if (!is_known_platform_noise(message)) {
            QMutexLocker lock(&g_mutex);
            g_messages << message;
        }
        return;
    }
    if (g_previous != nullptr) {
        g_previous(type, context, message);
    }
}

}  // namespace

QtMessageCapture::QtMessageCapture() {
    {
        QMutexLocker lock(&g_mutex);
        g_messages.clear();
    }
    g_previous = qInstallMessageHandler(capturing_handler);
}

QtMessageCapture::~QtMessageCapture() {
    qInstallMessageHandler(g_previous);
    g_previous = nullptr;
}

QStringList QtMessageCapture::messages() const {
    QMutexLocker lock(&g_mutex);
    return g_messages;
}

std::string QtMessageCapture::text() const {
    return messages().join(QLatin1Char('\n')).toStdString();
}

bool pump_until(const std::function<bool()>& done, int timeout_ms) {
    QElapsedTimer timer;
    timer.start();
    while (!done()) {
        if (timer.elapsed() >= timeout_ms) {
            return done();
        }
        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
        QThread::msleep(2);
    }
    return true;
}

void pump(int ms) {
    QElapsedTimer timer;
    timer.start();
    do {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
        QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
        QThread::msleep(2);
    } while (timer.elapsed() < ms);
}

void GuiTest::SetUp() {
    capture_ = std::make_unique<QtMessageCapture>();
    scratch_ = std::make_unique<QTemporaryDir>();
    ASSERT_TRUE(scratch_->isValid());
    rebuild(/*reset_settings=*/true);
}

void GuiTest::TearDown() {
    // Window before app: the window holds a reference to it.
    window_.reset();
    app_.reset();
    scratch_.reset();
    capture_.reset();
}

void GuiTest::rebuild(bool reset_settings) {
    window_.reset();
    app_.reset();
    Options options;
    options.reset_settings = reset_settings;
    app_ = std::make_unique<GuiApp>(options);
    window_ = std::make_unique<MainWindow>(*app_);
}

QAction* GuiTest::action(const QString& object_name) {
    auto* found = window_->findChild<QAction*>(object_name);
    if (found == nullptr) {
        ADD_FAILURE() << "no action named " << object_name.toStdString();
    }
    return found;
}

QMenu* GuiTest::menu(const QString& object_name) {
    return widget<QMenu>(object_name);
}

QDockWidget* GuiTest::dock(const QString& object_name) {
    return widget<QDockWidget>(object_name);
}

bool GuiTest::trigger(const QString& object_name) {
    QAction* found = action(object_name);
    if (found == nullptr) {
        return false;
    }
    if (!found->isEnabled()) {
        ADD_FAILURE() << "action is disabled: " << object_name.toStdString();
        return false;
    }
    found->trigger();
    return true;
}

void GuiTest::show_window() {
    window_->show();
    pump_until([this]() { return window_->isVisible(); });
    pump(30);
}

QString GuiTest::scratch(const QString& name) const {
    return QDir(scratch_->path()).filePath(name);
}

bool GuiTest::grab(QWidget* widget, const QString& name) {
    const QByteArray dir = qgetenv("ALPHASHAPE_GUI_GRAB_DIR");
    if (dir.isEmpty() || widget == nullptr) {
        return false;
    }
    const QString directory = QString::fromLocal8Bit(dir);
    if (!QDir().mkpath(directory)) {
        return false;
    }
    return widget->grab().save(QDir(directory).filePath(name + QStringLiteral(".png")));
}

}  // namespace alphashape::gui::testing
