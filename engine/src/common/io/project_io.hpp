#pragma once

// The on-disk project container.
//
// Layout:
//
//     myproject.psx              the project file (directory form)
//     myproject.files/           literal "{projectname}.files"
//         project.zip            doc.xml + binary entries
//         cache/                 scratch: undo spill, GeoTIFF codec files
//         lock                   present only while a writer holds the project
//
// .psx is a small text manifest pointing at the .files folder. That keeps the
// project openable by path, gives a format marker to sniff, and costs nothing.
//
// A path ending in .psz is the single-file form (WP-11.9): one zip holding
// exactly the entries project.zip holds, so a .psz and a directory project
// carry the same document. Nothing is streamed: a .psz is parsed fully into
// memory on load, which is the same bound the in-memory ZipArchive already
// implies.

#include <memory>
#include <string>

#include "common/core/document.hpp"

namespace alphashape {

struct ReadReport;

struct ProjectIO {
    /// Save to `path`. A ".psz" path writes one archive; anything else writes
    /// the .psx manifest plus "name.files/project.zip". Both go through a temp
    /// file and rename, so a failed save never truncates the previous project.
    ///
    /// `absolute_paths` stores frame paths absolutely instead of relative to
    /// the project file, and records Project/AbsolutePaths accordingly.
    static void save(const Document& document, const std::string& path,
                     bool absolute_paths = false);

    /// Load from a .psz archive, a .psx manifest, or directly from a
    /// project.zip. `report`, when non-null, receives non-fatal warnings
    /// (including photographs that could not be found).
    static std::shared_ptr<Document> load(const std::string& path,
                                          ReadReport* report = nullptr);
};

}  // namespace alphashape
