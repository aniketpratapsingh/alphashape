#include "common/algorithm/reference/reference_xml.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "base/utils/error.hpp"
#include "common/io/doc_xml_common.hpp"

namespace alphashape::reference {
namespace {

using docxml::XmlNode;

std::string require_text(const XmlNode& node, const char* name) {
    const std::string* value = docxml::find_attr(node, name);
    if (value == nullptr) {
        throw ValueError("reference XML: <" + node.name + "> is missing the \"" + name +
                         "\" attribute");
    }
    return *value;
}

double optional_number(const XmlNode& node, const char* name, double fallback) {
    const std::string* value = docxml::find_attr(node, name);
    if (value == nullptr) {
        return fallback;
    }
    return docxml::parse_double(*value);
}

bool optional_bool(const XmlNode& node, const char* name, bool fallback) {
    const std::string* value = docxml::find_attr(node, name);
    if (value == nullptr) {
        return fallback;
    }
    return docxml::parse_bool(*value);
}

std::vector<XmlReferencePosition> read_positions(const XmlNode* container,
                                                 const char* child_name) {
    std::vector<XmlReferencePosition> out;
    if (container == nullptr) {
        return out;
    }
    for (const XmlNode* node : docxml::find_children(*container, child_name)) {
        XmlReferencePosition item;
        item.label = require_text(*node, "label");
        item.position = Eigen::Vector3d(docxml::parse_double(require_text(*node, "x")),
                                        docxml::parse_double(require_text(*node, "y")),
                                        docxml::parse_double(require_text(*node, "z")));
        item.accuracy = optional_number(*node, "sxyz", 0.0);
        item.enabled = optional_bool(*node, "enabled", true);
        if (const std::string* kind = docxml::find_attr(*node, "kind")) {
            item.kind = *kind;
        }
        out.push_back(std::move(item));
    }
    return out;
}

}  // namespace

XmlReferenceDocument parse_reference_xml(const std::string& text) {
    const XmlNode root = docxml::parse_xml(text);
    if (root.name != "reference") {
        throw ValueError("reference XML: root element is <" + root.name +
                         ">, expected <reference>");
    }
    XmlReferenceDocument document;
    document.cameras = read_positions(docxml::find_child(root, "cameras"), "camera");
    document.markers = read_positions(docxml::find_child(root, "markers"), "marker");
    if (const XmlNode* bars = docxml::find_child(root, "scalebars")) {
        for (const XmlNode* node : docxml::find_children(*bars, "scalebar")) {
            XmlReferenceScalebar bar;
            bar.label = require_text(*node, "label");
            bar.distance = docxml::parse_double(require_text(*node, "distance"));
            bar.accuracy = optional_number(*node, "accuracy", 0.0);
            bar.enabled = optional_bool(*node, "enabled", true);
            document.scalebars.push_back(std::move(bar));
        }
    }
    return document;
}

namespace {

void append_position(XmlNode& parent, const char* child_name,
                     const XmlReferencePosition& item) {
    XmlNode& node = docxml::add_child(parent, child_name);
    docxml::set_attr(node, "label", item.label);
    docxml::set_attr(node, "x", docxml::fmt(item.position.x()));
    docxml::set_attr(node, "y", docxml::fmt(item.position.y()));
    docxml::set_attr(node, "z", docxml::fmt(item.position.z()));
    docxml::set_attr(node, "sxyz", docxml::fmt(item.accuracy));
    docxml::set_attr(node, "enabled", docxml::fmt_bool(item.enabled));
    if (!item.kind.empty()) {
        docxml::set_attr(node, "kind", item.kind);
    }
}

}  // namespace

std::string write_reference_xml(const XmlReferenceDocument& document) {
    XmlNode root;
    root.name = "reference";
    XmlNode& cameras = docxml::add_child(root, "cameras");
    for (const XmlReferencePosition& camera : document.cameras) {
        append_position(cameras, "camera", camera);
    }
    XmlNode& markers = docxml::add_child(root, "markers");
    for (const XmlReferencePosition& marker : document.markers) {
        append_position(markers, "marker", marker);
    }
    if (!document.scalebars.empty()) {
        XmlNode& bars = docxml::add_child(root, "scalebars");
        for (const XmlReferenceScalebar& bar : document.scalebars) {
            XmlNode& node = docxml::add_child(bars, "scalebar");
            docxml::set_attr(node, "label", bar.label);
            docxml::set_attr(node, "distance", docxml::fmt(bar.distance));
            docxml::set_attr(node, "accuracy", docxml::fmt(bar.accuracy));
            docxml::set_attr(node, "enabled", docxml::fmt_bool(bar.enabled));
        }
    }
    std::string out;
    docxml::write_node(root, out, 0);
    return out;
}

}  // namespace alphashape::reference
