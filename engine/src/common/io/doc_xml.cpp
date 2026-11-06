#include "common/io/doc_xml.hpp"

#include <filesystem>
#include <memory>
#include <new>
#include <string>

#include "base/platform/env.hpp"
#include "base/utils/error.hpp"
#include "common/io/doc_xml_common.hpp"
#include "common/io/paths.hpp"

namespace alphashape {

namespace fs = std::filesystem;

namespace {

/// Turn every stored relative frame path into the absolute path it names. A
/// resolved path that does not exist stays as it was written and produces a
/// warning: the project still loads, and a project whose image tree moved away
/// is readable (the user can relink).
///
/// Only resolving an existing file is deliberate. A path that was relative and
/// is still missing has no better absolute form to hand the caller, and keeping
/// the stored text makes a save/load round trip of an unresolved project
/// byte-stable.
void resolve_frame_paths(Document& document, const std::string& project_dir,
                         ReadReport* report) {
    for (auto& workspace : document.workspaces()) {
        if (workspace == nullptr) {
            continue;
        }
        for (auto& chunk : workspace->chunks()) {
            if (chunk == nullptr) {
                continue;
            }
            for (auto& frame : chunk->frames()) {
                if (frame == nullptr || frame->path().empty()) {
                    continue;
                }
                const std::string stored = frame->path();
                const std::string resolved = path_from_storage(stored, project_dir);
                if (resolved == stored) {
                    continue;  // already absolute
                }
                std::error_code ec;
                if (fs::exists(platform::path_from_utf8(resolved), ec)) {
                    frame->set_path(resolved);
                } else {
                    docxml::add_warning(report, "frame file not found: " + resolved);
                }
            }
        }
    }
}

}  // namespace

// doc.xml is a versioned container. This file is only the dispatcher:
//
//   write            -> always the current format (Document::kFormatVersion)
//   read version 1   -> doc_xml_v1.cpp (the pre-M11 layout, read verbatim)
//   read version 2   -> doc_xml_v2.cpp
//   read version 3   -> doc_xml_v2.cpp (v3 changed only the depth-map sidecar,
//                       ADM1 -> ADM2, which decode_depth_maps dispatches on its
//                       own magic; the XML vocabulary is v2's)
//   read version > 3 -> ProjectIOError ("newer than this build supports")
//   read version 0 or missing -> ProjectIOError
//
// The shared machinery (XML tree and parser, escaping, numeric formatting,
// calibration/meta/CRS helpers, sidecar codecs) lives in doc_xml_common.{hpp,
// cpp} and sidecars.{hpp,cpp}; the two readers share it, so a v1 project and a
// v2 project decode through exactly the same value and sidecar code.

void write_document(const Document& document, ZipArchive& archive, int version,
                    const IoContext& context) {
    if (version != Document::kFormatVersion) {
        throw ProjectIOError("cannot write project format version " +
                             std::to_string(version) + ": this build writes version " +
                             std::to_string(Document::kFormatVersion) + " only");
    }
    docxml::write_v2(document, archive, context);
}

std::shared_ptr<Document> read_document(const ZipArchive& archive, ReadReport* report,
                                        const IoContext& context) {
    if (report != nullptr) {
        report->version = 0;
        report->warnings.clear();
    }
    if (!archive.contains(kDocXmlEntry)) {
        throw ProjectIOError("Invalid project file: no doc.xml");
    }
    const std::string text = archive.get_text(kDocXmlEntry);
    const docxml::XmlNode root = docxml::parse_xml(text);
    if (root.name != "document") {
        throw ProjectIOError("Invalid project file: root element is <" + root.name + ">");
    }
    const int version = docxml::parse_int(docxml::require_attr(root, "version"));
    if (version > Document::kFormatVersion) {
        throw ProjectIOError("project format version " + std::to_string(version) +
                             " is newer than this build supports (" +
                             std::to_string(Document::kFormatVersion) + ")");
    }
    if (report != nullptr) {
        report->version = version;
    }
    // ERROR CONTRACT. read_document promises ProjectIOError for any malformed
    // input. The readers are written to raise it directly, but a structural
    // defect can also surface from the model layer (a duplicate chunk id is a
    // ValueError) or from the allocator (a sidecar whose declared size survives
    // every per-decoder check but cannot be held). Neither says "your project
    // is unreadable", so normalize them here rather than at every call site.
    std::shared_ptr<Document> document;
    try {
        if (version == 1) {
            document = docxml::read_v1(root, archive, report, context);
        } else if (version >= 2 && version <= Document::kFormatVersion) {
            // v2, v3 and v4 share the reader: v3 and v4 are sidecar-encoding
            // bumps only (ADM1 -> ADM2 -> ADM3 for depth maps), and the codec
            // keys off its own magic.
            document = docxml::read_v2(root, archive, report, context);
        } else {
            throw ProjectIOError("Invalid project file: unsupported format version " +
                                 std::to_string(version));
        }
    } catch (const ProjectIOError&) {
        throw;
    } catch (const Error& e) {
        throw ProjectIOError(std::string("Invalid project file: ") + e.what());
    } catch (const std::bad_alloc&) {
        throw ProjectIOError("Invalid project file: a declared size is too large to load");
    }
    if (!context.project_dir.empty()) {
        resolve_frame_paths(*document, context.project_dir, report);
    }
    return document;
}

}  // namespace alphashape
