#include "common/algorithm/reference/markers_io.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "common/algorithm/reference/reference_csv.hpp"
#include "common/io/doc_xml_common.hpp"

namespace alphashape::reference {
namespace {

namespace fs = std::filesystem;

constexpr std::size_t kMissing = std::numeric_limits<std::size_t>::max();

std::string lower(std::string text) {
    for (char& c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

std::string trim(const std::string& text) {
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        --end;
    }
    return text.substr(begin, end - begin);
}

std::string line_context(std::size_t line) {
    return line == 0 ? std::string{} : "line " + std::to_string(line);
}

double finite_number(const std::string& field, std::size_t line) {
    const double value =
        parse_number(field, false, line_context(line).empty() ? std::string("value")
                                                              : line_context(line));
    if (!std::isfinite(value)) {
        throw ValueError((line_context(line).empty() ? std::string("marker file")
                                                     : line_context(line)) +
                         ": coordinate must be finite");
    }
    return value;
}

std::string require_attr(const docxml::XmlNode& node, const char* name) {
    const std::string* value = docxml::find_attr(node, name);
    if (value == nullptr) {
        throw ValueError("marker XML: <" + node.name + "> is missing the \"" + name +
                         "\" attribute");
    }
    return trim(*value);
}

}  // namespace

MarkerIoFormat marker_io_format(const std::string& format, const std::string& path,
                                const char* task) {
    const std::string text = lower(trim(format));
    const auto fail = [&]() {
        throw ValueError(std::string(task) + ": format must be csv or xml, got '" +
                         format + "'");
    };
    if (!text.empty()) {
        if (text == "csv") {
            return MarkerIoFormat::Csv;
        }
        if (text == "xml") {
            return MarkerIoFormat::Xml;
        }
        fail();
    }
    const std::string ext = lower(fs::path(path).extension().string());
    if (ext == ".csv") {
        return MarkerIoFormat::Csv;
    }
    if (ext == ".xml") {
        return MarkerIoFormat::Xml;
    }
    throw ValueError(std::string(task) + ": cannot infer the format of '" + path +
                     "'; set the `format` parameter to csv or xml");
}

std::vector<MarkerIoMarker> parse_marker_csv(const std::string& text) {
    const std::vector<DelimitedRow> rows = parse_delimited(text, ',');
    std::vector<MarkerIoMarker> markers;
    if (rows.empty()) {
        return markers;
    }
    std::size_t marker_col = 0;
    std::size_t camera_col = 1;
    std::size_t u_col = 2;
    std::size_t v_col = 3;
    std::size_t first = 0;
    bool header = false;
    for (const std::string& field : rows.front().fields) {
        const std::string name = lower(trim(field));
        if (name == "marker" || name == "camera" || name == "u" || name == "v") {
            header = true;
        }
    }
    if (header) {
        marker_col = camera_col = u_col = v_col = kMissing;
        for (std::size_t i = 0; i < rows.front().fields.size(); ++i) {
            const std::string name = lower(trim(rows.front().fields[i]));
            if (name == "marker") {
                marker_col = i;
            } else if (name == "camera") {
                camera_col = i;
            } else if (name == "u") {
                u_col = i;
            } else if (name == "v") {
                v_col = i;
            }
        }
        if (marker_col == kMissing || camera_col == kMissing || u_col == kMissing ||
            v_col == kMissing) {
            throw ValueError(
                "ImportMarkers: the CSV header must name marker, camera, u and v");
        }
        first = 1;
    }
    std::map<std::string, std::size_t> by_label;
    for (std::size_t r = first; r < rows.size(); ++r) {
        const DelimitedRow& row = rows[r];
        const std::size_t widest = std::max(std::max(marker_col, camera_col),
                                            std::max(u_col, v_col));
        if (row.fields.size() <= widest) {
            throw ValueError("ImportMarkers: " + line_context(row.line) +
                             ": expected marker, camera, u, v");
        }
        const std::string label = trim(row.fields[marker_col]);
        const std::string camera = trim(row.fields[camera_col]);
        if (label.empty() || camera.empty()) {
            throw ValueError("ImportMarkers: " + line_context(row.line) +
                             ": marker and camera labels must not be empty");
        }
        MarkerIoProjection projection;
        projection.camera = camera;
        projection.uv = Eigen::Vector2d(finite_number(row.fields[u_col], row.line),
                                        finite_number(row.fields[v_col], row.line));
        projection.line = row.line;
        const auto found = by_label.find(label);
        if (found == by_label.end()) {
            MarkerIoMarker marker;
            marker.label = label;
            marker.line = row.line;
            marker.projections.push_back(std::move(projection));
            by_label.emplace(label, markers.size());
            markers.push_back(std::move(marker));
        } else {
            markers[found->second].projections.push_back(std::move(projection));
        }
    }
    return markers;
}

std::vector<MarkerIoMarker> parse_marker_xml(const std::string& text) {
    const docxml::XmlNode root = docxml::parse_xml(text);
    if (root.name != "markers") {
        throw ValueError("marker XML: root element is <" + root.name +
                         ">, expected <markers>");
    }
    std::vector<MarkerIoMarker> markers;
    std::set<std::string> labels;
    for (const docxml::XmlNode* node : docxml::find_children(root, "marker")) {
        MarkerIoMarker marker;
        marker.label = require_attr(*node, "label");
        if (marker.label.empty()) {
            throw ValueError("marker XML: <marker> has an empty label");
        }
        if (!labels.insert(marker.label).second) {
            throw ValueError("marker XML: duplicate marker label '" + marker.label + "'");
        }
        const std::string* x = docxml::find_attr(*node, "x");
        const std::string* y = docxml::find_attr(*node, "y");
        const std::string* z = docxml::find_attr(*node, "z");
        const int present = (x != nullptr ? 1 : 0) + (y != nullptr ? 1 : 0) +
                            (z != nullptr ? 1 : 0);
        if (present != 0 && present != 3) {
            throw ValueError("marker XML: marker '" + marker.label +
                             "' must carry all of x, y, z or none");
        }
        if (present == 3) {
            marker.position = Eigen::Vector3d(finite_number(*x, 0), finite_number(*y, 0),
                                              finite_number(*z, 0));
        }
        for (const docxml::XmlNode* projection :
             docxml::find_children(*node, "projection")) {
            MarkerIoProjection item;
            item.camera = require_attr(*projection, "camera");
            if (item.camera.empty()) {
                throw ValueError("marker XML: marker '" + marker.label +
                                 "' has a projection with an empty camera label");
            }
            item.uv = Eigen::Vector2d(finite_number(require_attr(*projection, "u"), 0),
                                      finite_number(require_attr(*projection, "v"), 0));
            marker.projections.push_back(std::move(item));
        }
        markers.push_back(std::move(marker));
    }
    return markers;
}

std::string write_marker_csv(const std::vector<MarkerIoMarker>& markers, int precision) {
    std::string out = format_delimited({"marker", "camera", "u", "v"}, ',');
    for (const MarkerIoMarker& marker : markers) {
        for (const MarkerIoProjection& projection : marker.projections) {
            out += format_delimited(
                {marker.label, projection.camera,
                 format_number(projection.uv.x(), precision),
                 format_number(projection.uv.y(), precision)},
                ',');
        }
    }
    return out;
}

std::string write_marker_xml(const std::vector<MarkerIoMarker>& markers) {
    docxml::XmlNode root;
    root.name = "markers";
    for (const MarkerIoMarker& marker : markers) {
        docxml::XmlNode& node = docxml::add_child(root, "marker");
        docxml::set_attr(node, "label", marker.label);
        if (marker.position.has_value()) {
            docxml::set_attr(node, "x", docxml::fmt(marker.position->x()));
            docxml::set_attr(node, "y", docxml::fmt(marker.position->y()));
            docxml::set_attr(node, "z", docxml::fmt(marker.position->z()));
        }
        for (const MarkerIoProjection& projection : marker.projections) {
            docxml::XmlNode& child = docxml::add_child(node, "projection");
            docxml::set_attr(child, "camera", projection.camera);
            docxml::set_attr(child, "u", docxml::fmt(projection.uv.x()));
            docxml::set_attr(child, "v", docxml::fmt(projection.uv.y()));
        }
    }
    std::string out;
    docxml::write_node(root, out, 0);
    return out;
}

}  // namespace alphashape::reference
