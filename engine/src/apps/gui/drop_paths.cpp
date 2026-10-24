#include "apps/gui/drop_paths.hpp"

#include <algorithm>

#include <QDir>
#include <QFileInfo>

#include "apps/gui/image_files.hpp"

namespace alphashape::gui {

bool is_project_path(const QString& path) {
    return QFileInfo(path).suffix().compare(QStringLiteral("psx"), Qt::CaseInsensitive) == 0;
}

DroppedPaths classify_dropped_paths(const QStringList& paths) {
    DroppedPaths found;
    for (const QString& path : paths) {
        if (path.isEmpty()) {
            continue;
        }
        const QFileInfo info(path);
        if (!info.exists()) {
            continue;  // vanished between the drag and the drop
        }
        if (info.isDir()) {
            found.folders << path;
        } else if (is_project_path(path)) {
            found.projects << path;
        } else if (is_image_path(path)) {
            found.images << path;
        }
        // Anything else (a PDF, an archive, ...) is not something this window knows how to open.
    }
    return found;
}

QStringList images_in_directories(const QStringList& folders, int max_depth, int max_images) {
    QStringList found;
    const QStringList filters = image_name_filters();
    if (filters.isEmpty() || max_depth < 0 || max_images <= 0) {
        return found;
    }

    // Breadth-first with an explicit queue and an explicit depth per entry, so the walk order is
    // stable and the bound is enforced without recursion.
    QStringList pending;
    pending.reserve(folders.size());
    for (const QString& folder : folders) {
        pending << folder;
    }
    QList<int> depths;
    depths.fill(0, pending.size());

    for (qsizetype head = 0; head < pending.size(); ++head) {
        const QString directory = pending[head];
        const int depth = depths[head];
        const QDir dir(directory);
        if (!dir.exists()) {
            continue;
        }
        const QStringList entries = dir.entryList(filters, QDir::Files, QDir::Name);
        for (const QString& entry : entries) {
            if (found.size() >= max_images) {
                return found;
            }
            found << dir.absoluteFilePath(entry);
        }
        if (depth >= max_depth) {
            continue;
        }
        const QStringList subdirectories =
            dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
        for (const QString& subdirectory : subdirectories) {
            pending << dir.absoluteFilePath(subdirectory);
            depths << depth + 1;
        }
    }
    return found;
}

}  // namespace alphashape::gui
