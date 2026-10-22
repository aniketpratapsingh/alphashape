#include "common/io/doc_xml.hpp"

#include <new>
#include <string>

#include "base/utils/error.hpp"
#include "common/io/doc_xml_common.hpp"

namespace alphashape {

// doc.xml is a versioned container. This file is only the dispatcher:
//
//   write            -> always format v2 (Document::kFormatVersion)
//   read version 1   -> doc_xml_v1.cpp (the pre-M11 layout, read verbatim)
//   read version 2   -> doc_xml_v2.cpp
//   read version > 2 -> ProjectIOError ("newer than this build supports")
//   read version 0 or missing -> ProjectIOError
//
// The shared machinery (XML tree and parser, escaping, numeric formatting,
// calibration/meta/CRS helpers, sidecar codecs) lives in doc_xml_common.{hpp,
// cpp} and sidecars.{hpp,cpp}; the two readers share it, so a v1 project and a
// v2 project decode through exactly the same value and sidecar code.

void write_document(const Document& document, ZipArchive& archive, int version) {
    if (version != Document::kFormatVersion) {
        throw ProjectIOError("cannot write project format version " +
                             std::to_string(version) + ": this build writes version " +
                             std::to_string(Document::kFormatVersion) + " only");
    }
    docxml::write_v2(document, archive);
}

std::shared_ptr<Document> read_document(const ZipArchive& archive, ReadReport* report) {
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
    try {
        if (version == 1) {
            return docxml::read_v1(root, archive, report);
        }
        if (version == 2) {
            return docxml::read_v2(root, archive, report);
        }
    } catch (const ProjectIOError&) {
        throw;
    } catch (const Error& e) {
        throw ProjectIOError(std::string("Invalid project file: ") + e.what());
    } catch (const std::bad_alloc&) {
        throw ProjectIOError("Invalid project file: a declared size is too large to load");
    }
    throw ProjectIOError("Invalid project file: unsupported format version " +
                         std::to_string(version));
}

}  // namespace alphashape
