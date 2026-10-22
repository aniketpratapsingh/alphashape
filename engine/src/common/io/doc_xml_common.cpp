#include "common/io/doc_xml_common.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "base/utils/error.hpp"
#include "common/core/frame.hpp"

namespace alphashape::docxml {

// ---------------------------------------------------------------------------
// Read diagnostics. Warnings are advisory: they never change what is loaded,
// only what the caller is told about it. Each message names the element and
// the offending attribute so a user can locate it in doc.xml.
// ---------------------------------------------------------------------------
void add_warning(ReadReport* report, std::string message) {
    if (report == nullptr) {
        return;
    }
    // A file that repeats the same defect thousands of times (a mutated archive
    // fuzzed through this path) must not turn into an unbounded warning list.
    constexpr std::size_t kMaxWarnings = 256;
    if (report->warnings.size() < kMaxWarnings) {
        report->warnings.push_back(std::move(message));
    } else if (report->warnings.size() == kMaxWarnings) {
        report->warnings.emplace_back("further warnings suppressed");
    }
}

void warn_unknown_attrs(const XmlNode& node, std::initializer_list<const char*> known,
                        ReadReport* report) {
    if (report == nullptr) {
        return;
    }
    for (const auto& [key, value] : node.attrs) {
        bool listed = false;
        for (const char* name : known) {
            if (key == name) {
                listed = true;
                break;
            }
        }
        if (!listed) {
            add_warning(report, "doc.xml: ignoring unknown attribute '" + key +
                                    "' on <" + node.name + ">");
        }
    }
}

void warn_unknown_children(const XmlNode& node, std::initializer_list<const char*> known,
                           ReadReport* report) {
    if (report == nullptr) {
        return;
    }
    for (const XmlNode& child : node.children) {
        bool listed = false;
        for (const char* name : known) {
            if (child.name == name) {
                listed = true;
                break;
            }
        }
        if (!listed) {
            add_warning(report, "doc.xml: ignoring unknown element <" + child.name +
                                    "> inside <" + node.name + ">");
        }
    }
}

void warn_unknown_calibration_attrs(const XmlNode& node, ReadReport* report) {
    warn_unknown_attrs(node,
                       {"type", "width", "height", "f", "cx", "cy", "b1", "b2", "k1",
                        "k2", "k3", "k4", "p1", "p2"},
                       report);
}

void warn_missing_frame_files(const std::vector<std::shared_ptr<Frame>>& frames,
                              ReadReport* report) {
    if (report == nullptr) {
        return;
    }
    std::error_code ec;
    for (const auto& frame : frames) {
        if (report->warnings.size() >= 256) {
            return;  // add_warning() caps the list too; stop stat-ing files
        }
        if (frame == nullptr || frame->path().empty()) {
            continue;
        }
        const std::filesystem::path path(frame->path());
        if (!path.is_absolute()) {
            continue;
        }
        if (!std::filesystem::exists(path, ec)) {
            add_warning(report, "frame file not found: " + frame->path());
        }
        ec.clear();
    }
}

XmlNode& add_child(XmlNode& parent, std::string name) {
    parent.children.push_back(XmlNode{std::move(name), {}, {}, {}});
    return parent.children.back();
}

void set_attr(XmlNode& node, std::string name, std::string value) {
    node.attrs.emplace_back(std::move(name), std::move(value));
}

const std::string* find_attr(const XmlNode& node, const std::string& name) {
    for (const auto& [key, value] : node.attrs) {
        if (key == name) {
            return &value;
        }
    }
    return nullptr;
}

std::string attr_or(const XmlNode& node, const std::string& name,
                    const std::string& fallback) {
    const std::string* value = find_attr(node, name);
    return value != nullptr ? *value : fallback;
}

const std::string& require_attr(const XmlNode& node, const std::string& name) {
    const std::string* value = find_attr(node, name);
    if (value == nullptr) {
        throw ProjectIOError("doc.xml: <" + node.name + "> missing attribute '" +
                             name + "'");
    }
    return *value;
}

const XmlNode* find_child(const XmlNode& node, const std::string& name) {
    for (const auto& child : node.children) {
        if (child.name == name) {
            return &child;
        }
    }
    return nullptr;
}

std::vector<const XmlNode*> find_children(const XmlNode& node,
                                          const std::string& name) {
    std::vector<const XmlNode*> out;
    for (const auto& child : node.children) {
        if (child.name == name) {
            out.push_back(&child);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Escaping and serialisation.
// ---------------------------------------------------------------------------
std::string escape(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (const char c : in) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default: out += c;
        }
    }
    return out;
}

std::string unescape(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        if (in[i] != '&') {
            out += in[i];
            continue;
        }
        const std::size_t semi = in.find(';', i);
        if (semi == std::string::npos) {
            throw ProjectIOError("doc.xml: unterminated entity");
        }
        const std::string entity = in.substr(i + 1, semi - i - 1);
        if (entity == "amp") out += '&';
        else if (entity == "lt") out += '<';
        else if (entity == "gt") out += '>';
        else if (entity == "quot") out += '"';
        else if (entity == "apos") out += '\'';
        else throw ProjectIOError("doc.xml: unknown entity &" + entity + ";");
        i = semi;
    }
    return out;
}

void write_node(const XmlNode& node, std::string& out, int depth) {
    out.append(static_cast<std::size_t>(depth) * 2, ' ');
    out += '<' + node.name;
    for (const auto& [key, value] : node.attrs) {
        out += ' ' + key + "=\"" + escape(value) + '"';
    }
    if (node.children.empty() && node.text.empty()) {
        out += "/>\n";
        return;
    }
    out += '>';
    if (!node.children.empty()) {
        out += '\n';
        for (const auto& child : node.children) {
            write_node(child, out, depth + 1);
        }
        out.append(static_cast<std::size_t>(depth) * 2, ' ');
    } else {
        out += escape(node.text);
    }
    out += "</" + node.name + ">\n";
}

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    XmlNode parse_document() {
        skip_misc();
        XmlNode root = parse_element();
        skip_misc();
        if (pos_ != s_.size()) {
            fail("trailing content after root element");
        }
        return root;
    }

private:
    [[noreturn]] void fail(const std::string& why) const {
        throw ProjectIOError("doc.xml: " + why + " at offset " + std::to_string(pos_));
    }
    bool starts(const char* token) const {
        return s_.compare(pos_, std::strlen(token), token) == 0;
    }
    void skip_ws() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\n' ||
                                    s_[pos_] == '\r' || s_[pos_] == '\t')) {
            ++pos_;
        }
    }
    void skip_until(const char* token) {
        const std::size_t end = s_.find(token, pos_);
        if (end == std::string::npos) {
            fail(std::string("expected '") + token + "'");
        }
        pos_ = end + std::strlen(token);
    }
    /// Prolog, comments and whitespace between elements.
    void skip_misc() {
        for (;;) {
            skip_ws();
            if (starts("<?")) skip_until("?>");
            else if (starts("<!--")) skip_until("-->");
            else return;
        }
    }
    std::string parse_name() {
        const std::size_t start = pos_;
        while (pos_ < s_.size()) {
            const char c = s_[pos_];
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '/' ||
                c == '>' || c == '=') {
                break;
            }
            ++pos_;
        }
        if (pos_ == start) fail("expected a name");
        return s_.substr(start, pos_ - start);
    }
    XmlNode parse_element() {
        // A corrupt or hostile doc.xml can open thousands of nested elements
        // with nothing but "<a>" repeated; without a ceiling the recursive
        // descent exhausts the stack and the process dies instead of raising.
        // Real projects nest at most a handful of levels (chunk > container >
        // element > child), so 64 is far above anything the writer emits.
        constexpr int kMaxDepth = 64;
        if (depth_ >= kMaxDepth) {
            fail("elements nested deeper than " + std::to_string(kMaxDepth));
        }
        ++depth_;
        const DepthGuard guard(depth_);
        if (!starts("<")) fail("expected '<'");
        ++pos_;
        XmlNode node;
        node.name = parse_name();
        for (;;) {
            skip_ws();
            if (starts("/>")) {
                pos_ += 2;
                return node;
            }
            if (starts(">")) {
                ++pos_;
                break;
            }
            std::string key = parse_name();
            skip_ws();
            if (!starts("=")) fail("expected '='");
            ++pos_;
            skip_ws();
            if (!starts("\"")) fail("expected '\"'");
            ++pos_;
            const std::size_t end = s_.find('"', pos_);
            if (end == std::string::npos) fail("unterminated attribute");
            node.attrs.emplace_back(std::move(key), unescape(s_.substr(pos_, end - pos_)));
            pos_ = end + 1;
        }
        std::string text;
        for (;;) {
            if (pos_ >= s_.size()) fail("unterminated element <" + node.name + ">");
            if (starts("</")) {
                pos_ += 2;
                if (parse_name() != node.name) {
                    fail("mismatched closing tag for <" + node.name + ">");
                }
                skip_ws();
                if (!starts(">")) fail("expected '>'");
                ++pos_;
                break;
            }
            if (starts("<!--")) {
                skip_until("-->");
                continue;
            }
            if (starts("<")) {
                node.children.push_back(parse_element());
                continue;
            }
            const std::size_t lt = s_.find('<', pos_);
            text += s_.substr(pos_, lt == std::string::npos ? std::string::npos
                                                           : lt - pos_);
            pos_ = lt == std::string::npos ? s_.size() : lt;
        }
        if (node.children.empty()) {
            node.text = unescape(text);
        }
        return node;
    }

    const std::string& s_;
    std::size_t pos_ = 0;
    int depth_ = 0;

    /// Declared before parse_element() uses it: decrements the nesting depth on
    /// every exit path, including the throw from a malformed child.
    struct DepthGuard {
        int& depth;
        explicit DepthGuard(int& d) : depth(d) {}
        ~DepthGuard() { --depth; }
        DepthGuard(const DepthGuard&) = delete;
        DepthGuard& operator=(const DepthGuard&) = delete;
    };
};

XmlNode parse_xml(const std::string& text) { return Parser(text).parse_document(); }

// ---------------------------------------------------------------------------
// Value formatting. 17 significant digits round-trip any IEEE-754 double
// exactly. (Float keypoint data is binary, in the .kpt sidecars.)
// ---------------------------------------------------------------------------
std::string fmt(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

double parse_double(const std::string& s) {
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || end != s.c_str() + s.size() || errno == ERANGE) {
        throw ProjectIOError("doc.xml: bad number '" + s + "'");
    }
    return v;
}

long long parse_ll(const std::string& s) {
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (s.empty() || end != s.c_str() + s.size() || errno == ERANGE) {
        throw ProjectIOError("doc.xml: bad integer '" + s + "'");
    }
    return v;
}

std::int32_t parse_i32(const std::string& s) {
    return static_cast<std::int32_t>(parse_ll(s));
}
int parse_int(const std::string& s) { return static_cast<int>(parse_ll(s)); }

bool parse_bool(const std::string& s) {
    if (s == "true") return true;
    if (s == "false") return false;
    throw ProjectIOError("doc.xml: bad boolean '" + s + "'");
}

std::string fmt_bool(bool v) { return v ? "true" : "false"; }

std::string camera_type_name(CameraType type) {
    return type == CameraType::Fisheye ? "fisheye" : "frame";
}

CameraType parse_camera_type(const std::string& s) {
    if (s == "frame") return CameraType::Frame;
    if (s == "fisheye") return CameraType::Fisheye;
    throw ProjectIOError("doc.xml: unknown sensor type '" + s + "'");
}

/// Space-separated list of strings (used for the sensor's fixed_params).
std::string join_strings(const std::vector<std::string>& values) {
    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out += ' ';
        out += values[i];
    }
    return out;
}

std::vector<std::string> split_strings(const std::string& s) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && s[i] == ' ') ++i;
        if (i >= s.size()) break;
        const std::size_t j = s.find(' ', i);
        out.push_back(s.substr(i, j == std::string::npos ? std::string::npos : j - i));
        i = j == std::string::npos ? s.size() : j;
    }
    return out;
}

std::vector<double> parse_doubles(const std::string& s) {
    std::vector<double> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && s[i] == ' ') ++i;
        if (i >= s.size()) break;
        const std::size_t j = s.find(' ', i);
        out.push_back(parse_double(
            s.substr(i, j == std::string::npos ? std::string::npos : j - i)));
        i = j == std::string::npos ? s.size() : j;
    }
    return out;
}

std::string join(const double* values, std::size_t n) {
    std::string out;
    for (std::size_t i = 0; i < n; ++i) {
        if (i != 0) out += ' ';
        out += fmt(values[i]);
    }
    return out;
}

std::string vec3(const Eigen::Vector3d& v) { return join(v.data(), 3); }

/// Two doubles, space-separated with %.17g, so every value round-trips.
std::string vec2(double a, double b) {
    const double values[2] = {a, b};
    return join(values, 2);
}

std::pair<double, double> parse_pair(const std::string& s, const char* what) {
    const std::vector<double> v = parse_doubles(s);
    if (v.size() != 2) {
        throw ProjectIOError(std::string("doc.xml: ") + what +
                             " expected 2 numbers, got '" + s + "'");
    }
    return {v[0], v[1]};
}

Eigen::Vector3d parse_vec3(const std::string& s) {
    const auto v = parse_doubles(s);
    if (v.size() != 3) {
        throw ProjectIOError("doc.xml: expected 3 numbers, got '" + s + "'");
    }
    return {v[0], v[1], v[2]};
}

// Row-major, so the text reads like the matrix.
std::string matrix4(const Transform& m) {
    double values[16];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) values[r * 4 + c] = m(r, c);
    }
    return join(values, 16);
}

Transform parse_matrix4(const std::string& s) {
    const auto v = parse_doubles(s);
    if (v.size() != 16) {
        throw ProjectIOError("doc.xml: expected 16 numbers for a transform");
    }
    Transform m;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            m(r, c) = v[static_cast<std::size_t>(r * 4 + c)];
        }
    }
    return m;
}

void write_meta(const Meta& meta, XmlNode& parent) {
    XmlNode& node = add_child(parent, "meta");
    for (const auto& [key, value] : meta.items) {
        XmlNode& prop = add_child(node, "property");
        set_attr(prop, "name", key);
        set_attr(prop, "value", value);
    }
}

void read_meta(const XmlNode* node, Meta& meta) {
    if (node == nullptr) return;
    for (const XmlNode& prop : node->children) {
        if (prop.name == "property") {
            meta.set(require_attr(prop, "name"), require_attr(prop, "value"));
        }
    }
}

void write_crs(const char* name, const CRS& crs, XmlNode& parent) {
    XmlNode& node = add_child(parent, name);
    set_attr(node, "epsg", std::to_string(crs.epsg));
    set_attr(node, "wkt", crs.wkt);
}

void read_crs(const XmlNode* node, CRS& crs) {
    if (node == nullptr) return;
    crs.epsg = parse_int(require_attr(*node, "epsg"));
    crs.wkt = require_attr(*node, "wkt");
}

// Calibration coefficients, in the packed order of kCalibrationParamCount and
// with the attribute name each is written under. Every coefficient gets its own
// attribute so the XML is readable and diffable.
struct CalibrationField {
    const char* name;
    double Calibration::*member;
};

constexpr CalibrationField kCalibrationFields[] = {
    {"f", &Calibration::f},   {"cx", &Calibration::cx}, {"cy", &Calibration::cy},
    {"b1", &Calibration::b1}, {"b2", &Calibration::b2}, {"k1", &Calibration::k1},
    {"k2", &Calibration::k2}, {"k3", &Calibration::k3}, {"k4", &Calibration::k4},
    {"p1", &Calibration::p1}, {"p2", &Calibration::p2},
};

void write_calibration(const Calibration& c, XmlNode& parent) {
    XmlNode& node = add_child(parent, "calibration");
    set_attr(node, "type", camera_type_name(c.type));
    set_attr(node, "width", std::to_string(c.width));
    set_attr(node, "height", std::to_string(c.height));
    for (const CalibrationField& field : kCalibrationFields) {
        set_attr(node, field.name, fmt(c.*(field.member)));
    }
}

Calibration read_calibration(const XmlNode* node) {
    if (node == nullptr) {
        throw ProjectIOError("doc.xml: <sensor> missing <calibration>");
    }
    Calibration c;
    c.type = parse_camera_type(require_attr(*node, "type"));
    c.width = parse_int(require_attr(*node, "width"));
    c.height = parse_int(require_attr(*node, "height"));
    for (const CalibrationField& field : kCalibrationFields) {
        c.*(field.member) = parse_double(require_attr(*node, field.name));
    }
    return c;
}

/// Validate a fixed_params list against the known coefficient names. Unknown
/// names are a hard error rather than silently ignored, so a typo in a
/// hand-edited file cannot quietly disable a constraint.
void validate_fixed_params(const std::string& sensor_label,
                           const std::vector<std::string>& names) {
    for (const std::string& name : names) {
        bool known = false;
        for (const CalibrationField& field : kCalibrationFields) {
            if (name == field.name) {
                known = true;
                break;
            }
        }
        if (!known) {
            throw ProjectIOError("sensor '" + sensor_label +
                                 "': unknown fixed parameter '" + name + "'");
        }
    }
}

}  // namespace alphashape::docxml
