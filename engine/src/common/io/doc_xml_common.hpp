#pragma once

// Shared machinery for the doc.xml readers and writers.
//
// The archive format has two document versions:
//   * v1 -- the layout produced before M11 (a single active asset per kind,
//           no marker/scalebar/group/shape/trajectory persistence). It is read
//           by doc_xml_v1.cpp and stays byte-compatible with what shipped, so
//           existing projects keep loading exactly as they did.
//   * v2 -- the M11 layout (M11.md §4.4): <document version="2"> carrying
//           every container, all assets of every kind, and per-chunk archives
//           under chunks/<chunk_id>/.
// doc_xml.cpp is only the dispatcher; it picks a reader from the root element's
// version attribute.
//
// Everything in this header is intentionally tiny and dependency-free: an
// ordered XML tree, escaping, and the numeric/string formatting used by both
// writers. Ordering is insertion order (attribute and child vectors), which is
// what makes the writer deterministic -- a load/save of an untouched project
// reproduces the same bytes.

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/io/zip.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/core/types.hpp"
#include "common/io/doc_xml.hpp"

namespace alphashape {

namespace docxml {

/// One XML element. Attributes and children keep insertion order, so a writer
/// that appends in a fixed order always produces the same bytes.
struct XmlNode {
  std::string name;  ///< tag name
  std::vector<std::pair<std::string, std::string>> attrs;
  std::vector<XmlNode> children;
  std::string text;  ///< character data
};

/// Append an empty child element and return it.
XmlNode& add_child(XmlNode& parent, std::string name);

/// Append an attribute (order is preserved).
void set_attr(XmlNode& node, std::string name, std::string value);

/// Look an attribute up; returns nullptr when absent.
const std::string* find_attr(const XmlNode& node, const std::string& name);

/// Attribute or `fallback` when absent.
std::string attr_or(const XmlNode& node, const std::string& name,
                    const std::string& fallback);

/// Attribute access that throws ProjectIOError when the attribute is absent.
const std::string& require_attr(const XmlNode& node, const std::string& name);

/// First child with `name`, or nullptr.
const XmlNode* find_child(const XmlNode& node, const std::string& name);

/// All children named `name`, in document order.
std::vector<const XmlNode*> find_children(const XmlNode& node,
                                          const std::string& name);

// ---- read diagnostics -----------------------------------------------------

/// Append a formatted warning to `report` when one was supplied.
///
/// Readers use this for conditions that are worth surfacing but must not abort
/// loading: attributes written by a newer build, a skipped element, or a
/// missing optional sidecar. A null report (read_document's default) discards
/// the message, so no reader has to branch on it.
void add_warning(ReadReport* report, std::string message);

/// Flag the attributes of `node` that are not listed in `known`.
///
/// This is how an attribute added by a later revision of the same format is
/// handled: it does not fail the load, but the caller learns the project was
/// written by something newer than it understands. Every attribute list is
/// explicit here rather than inferred, so the check stays honest.
void warn_unknown_attrs(const XmlNode& node, std::initializer_list<const char*> known,
                        ReadReport* report);

/// As above for every child element named `known`; used by both versioned
/// readers on the chunk containers.
void warn_unknown_children(const XmlNode& node, std::initializer_list<const char*> known,
                           ReadReport* report);

/// Flag unknown attributes on a <calibration> element. Kept here because the
/// coefficient name list lives beside the reader/writer of that element.
void warn_unknown_calibration_attrs(const XmlNode& node, ReadReport* report);

/// Warn when a frame's recorded path is absolute and no longer present.
///
/// Relative paths cannot be resolved here: their base directory is the project
/// container, which is only known to the caller (§4.9 resolves those), so they
/// are left for that pass. An absolute path is checkable now, and a project
/// whose frames point at deleted files is worth surfacing.
void warn_missing_frame_files(const std::vector<std::shared_ptr<Frame>>& frames,
                              ReadReport* report);

// ---- text -----------------------------------------------------------------

/// XML-escape &, <, >, " and '.
std::string escape(const std::string& in);
/// Inverse of escape(); throws ProjectIOError on unknown entities.
std::string unescape(const std::string& in);

/// Serialise `node` with two-space indentation and LF line endings.
void write_node(const XmlNode& node, std::string& out, int depth);
/// Parse a whole document; returns the root element.
XmlNode parse_xml(const std::string& text);

// ---- numbers --------------------------------------------------------------

/// Shortest round-trippable decimal form ("%.17g", locale independent).
std::string fmt(double v);
double parse_double(const std::string& s);
long long parse_ll(const std::string& s);
std::int32_t parse_i32(const std::string& s);
int parse_int(const std::string& s);
bool parse_bool(const std::string& s);
std::string fmt_bool(bool v);

// ---- vocabulary -----------------------------------------------------------

std::string camera_type_name(CameraType type);
CameraType parse_camera_type(const std::string& s);
std::string join_strings(const std::vector<std::string>& values);
std::vector<std::string> split_strings(const std::string& s);

// ---- value formatting -----------------------------------------------------

std::vector<double> parse_doubles(const std::string& s);
std::string join(const double* values, std::size_t n);
std::string vec3(const Eigen::Vector3d& v);
std::string vec2(double a, double b);
std::pair<double, double> parse_pair(const std::string& s, const char* what);
Eigen::Vector3d parse_vec3(const std::string& s);
std::string matrix4(const Transform& m);
Transform parse_matrix4(const std::string& s);

// ---- small element helpers ------------------------------------------------

void write_meta(const Meta& meta, XmlNode& parent);
void read_meta(const XmlNode* node, Meta& meta);
void write_crs(const char* name, const CRS& crs, XmlNode& parent);
void read_crs(const XmlNode* node, CRS& crs);
void write_calibration(const Calibration& c, XmlNode& parent);
Calibration read_calibration(const XmlNode* node);
void validate_fixed_params(const std::string& sensor_label,
                           const std::vector<std::string>& names);

// ---- versioned entry points ----------------------------------------------

/// Write the whole document in format v2 (the only version written).
void write_v2(const Document& document, ZipArchive& archive);

/// Rebuild a document from a v1 root element (and its legacy sidecar paths).
/// Warnings (unknown attributes, missing optional sidecars) go to `report`.
std::shared_ptr<Document> read_v1(const XmlNode& root, const ZipArchive& archive,
                                  ReadReport* report);

/// Rebuild a document from a v2 root element (chunks/<id>/ sidecar paths).
std::shared_ptr<Document> read_v2(const XmlNode& root, const ZipArchive& archive,
                                  ReadReport* report);

}  // namespace docxml
}  // namespace alphashape
