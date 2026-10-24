#include "apps/gui/image_files.hpp"

#include <QCoreApplication>
#include <QFileInfo>

namespace alphashape::gui {

namespace {

/// The formats the engine decodes (JPEG, PNG, TIFF); PNM is a test format and is not offered.
const char* const kSuffixes[] = {"jpg", "jpeg", "png", "tif", "tiff"};

}  // namespace

QStringList image_suffixes() {
    QStringList suffixes;
    for (const char* suffix : kSuffixes) {
        suffixes << QString::fromLatin1(suffix);
    }
    return suffixes;
}

QStringList image_name_filters() {
    QStringList patterns;
    for (const QString& suffix : image_suffixes()) {
        patterns << QStringLiteral("*.") + suffix;
    }
    return patterns;
}

QString image_filter() {
    // A fixed context: this is a free function, not a QObject, so there is no tr() to call.
    return QCoreApplication::translate("alphashape::gui", "Images (%1);;All files (*)")
        .arg(image_name_filters().join(QLatin1Char(' ')));
}

bool is_image_path(const QString& path) {
    const QString suffix = QFileInfo(path).suffix();
    if (suffix.isEmpty()) {
        return false;
    }
    for (const QString& known : image_suffixes()) {
        if (suffix.compare(known, Qt::CaseInsensitive) == 0) {
            return true;
        }
    }
    return false;
}

}  // namespace alphashape::gui
