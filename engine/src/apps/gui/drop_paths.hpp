#pragma once

// Classification of the paths a drag-and-drop carries, shared by the start screen (which
// decides whether to accept the drag at all) and MainWindow (which acts on it), so the two
// cannot disagree about what a drop means.
//
// A drop names one of three things:
//   * a project file  (.psx)          -> open the project;
//   * image files     (see image_files.hpp) -> start a project with them;
//   * a directory     -> the images it holds, scanned a bounded depth down.
// Anything else is ignored: a PDF dropped on the launcher is not a photo. A path that does not
// exist is ignored too, without a stat() storm — the caller has already accepted the drag.

#include <QString>
#include <QStringList>

namespace alphashape::gui {

/// The usable part of a drop.
struct DroppedPaths {
    QStringList projects;  ///< *.psx files that exist
    QStringList images;    ///< image files that exist
    QStringList folders;   ///< directories that exist

    bool empty() const { return projects.isEmpty() && images.isEmpty() && folders.isEmpty(); }
};

/// Sort `paths` into projects / images / folders, dropping vanished paths and everything that is
/// neither. Pure file-system inspection; no dialogs, no signals.
DroppedPaths classify_dropped_paths(const QStringList& paths);

/// True when `path` is a project file by name (the file need not exist).
bool is_project_path(const QString& path);

/// Every image under `folders`, at most `max_depth` directory levels below each dropped folder
/// (0 = that folder alone, 1 = it and its immediate subfolders), and at most `max_images` in
/// total across all the folders, breadth-first and in name order. The bounds keep a drop of,
/// say, the home directory from walking a whole disk on the GUI thread.
QStringList images_in_directories(const QStringList& folders, int max_depth = 3,
                                  int max_images = 5000);

}  // namespace alphashape::gui
