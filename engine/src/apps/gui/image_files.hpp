#pragma once

// The one list of image-file suffixes the GUI offers and accepts: Add Photos' picker, the
// Workflow menu's picker, the start screen's drag-and-drop and the dropped-folder scan all
// read it from here, so "what counts as a photo" cannot drift between them.
//
// The engine itself does not classify by name: read_image_info()/read_image() sniff the magic
// bytes and ignore the extension (image/io/image_io.hpp). A file NAME is only ever needed on
// the GUI side, where a picker must decide before touching the file.

#include <QString>
#include <QStringList>

namespace alphashape::gui {

/// Lower-case suffixes, no leading dot, in display order.
QStringList image_suffixes();

/// The "Images (...);;All files (*)" filter for a file picker.
QString image_filter();

/// True when `path`'s extension names a supported image format (case-insensitive). Pure string
/// work: it never touches the file system, so it is safe in drag-move handlers.
bool is_image_path(const QString& path);

/// The glob patterns for a directory listing, e.g. "*.jpg".
QStringList image_name_filters();

}  // namespace alphashape::gui
