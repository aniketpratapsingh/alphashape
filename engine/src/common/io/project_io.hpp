#pragma once

// The on-disk project container.
//
// Layout:
//
//     myproject.psx              the project file
//     myproject.files/           literal "{projectname}.files"
//         project.zip            doc.xml + binary entries
//
// .psx is a small text manifest pointing at the .files folder. That keeps the
// project openable by path, gives a format marker to sniff, and costs nothing.

#include <memory>
#include <string>

#include "common/core/document.hpp"

namespace alphashape {

struct ProjectIO {
    /// Save to `path` (normally "name.psx"). Writes "name.files/project.zip"
    /// via a temp file and rename, so a failed save never truncates the
    /// previous project.
    static void save(const Document& document, const std::string& path);

    /// Load from a .psx manifest, or directly from a project.zip.
    static std::shared_ptr<Document> load(const std::string& path);
};

}  // namespace alphashape
