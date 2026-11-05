#include "common/algorithm/calibration/calibration_io.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "common/io/doc_xml_common.hpp"

namespace alphashape {
namespace {

using docxml::XmlNode;

constexpr const char* kPrefix = "calibration file: ";

std::string type_name(CameraType type) {
    return type == CameraType::Frame ? "frame" : "fisheye";
}

CameraType parse_type_name(const std::string& s) {
    if (s == "frame") {
        return CameraType::Frame;
    }
    if (s == "fisheye") {
        return CameraType::Fisheye;
    }
    throw ValueError(std::string(kPrefix) + "unknown camera type '" + s + "'");
}

std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw Error(std::string(kPrefix) + "cannot open '" + path + "' for reading");
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    if (in.bad()) {
        throw Error(std::string(kPrefix) + "cannot read '" + path + "'");
    }
    return buffer.str();
}

void write_text_file(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw Error(std::string(kPrefix) + "cannot open '" + path + "' for writing");
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!out) {
        throw Error(std::string(kPrefix) + "cannot write '" + path + "'");
    }
}

bool is_ws(char ch) {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';
}

std::string trim(const std::string& s) {
    std::size_t begin = 0;
    std::size_t end = s.size();
    while (begin < end && is_ws(s[begin])) {
        ++begin;
    }
    while (end > begin && is_ws(s[end - 1])) {
        --end;
    }
    return s.substr(begin, end - begin);
}

/// Tokenise on whitespace. Used for OpenCV XML <data> elements.
std::vector<std::string> whitespace_tokens(const std::string& text) {
    std::vector<std::string> tokens;
    std::size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && is_ws(text[i])) {
            ++i;
        }
        const std::size_t begin = i;
        while (i < text.size() && !is_ws(text[i])) {
            ++i;
        }
        if (i > begin) {
            tokens.push_back(text.substr(begin, i - begin));
        }
    }
    return tokens;
}

// ---- own XML format -------------------------------------------------------

std::string serialize_xml(const Calibration& c) {
    XmlNode parent;
    try {
        docxml::write_calibration(c, parent);
    } catch (const Error& e) {
        throw ValueError(std::string(kPrefix) + e.what());
    }
    if (parent.children.empty()) {
        throw ValueError(std::string(kPrefix) + "internal: no <calibration> element");
    }
    std::string out = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    docxml::write_node(parent.children.front(), out, 0);
    return out;
}

Calibration parse_xml(const std::string& text) {
    XmlNode root;
    try {
        root = docxml::parse_xml(text);
    } catch (const Error& e) {
        throw ValueError(std::string(kPrefix) + e.what());
    }
    if (root.name != "calibration") {
        throw ValueError(std::string(kPrefix) + "xml: root element is <" + root.name +
                            ">, expected <calibration>");
    }
    try {
        return docxml::read_calibration(&root);
    } catch (const Error& e) {
        throw ValueError(std::string(kPrefix) + e.what());
    }
}

// ---- JSON format ----------------------------------------------------------

std::string serialize_json(const Calibration& c) {
    std::string out = "{\"format\":\"alphashape-calibration\",\"version\":1,\"type\":\"";
    out += type_name(c.type);
    out += "\",\"width\":" + std::to_string(c.width);
    out += ",\"height\":" + std::to_string(c.height);
    const std::pair<const char*, double> terms[] = {
        {"f", c.f},   {"cx", c.cx}, {"cy", c.cy}, {"b1", c.b1}, {"b2", c.b2},
        {"k1", c.k1}, {"k2", c.k2}, {"k3", c.k3}, {"k4", c.k4}, {"p1", c.p1},
        {"p2", c.p2}, {"p3", c.p3}, {"p4", c.p4}};
    for (const auto& term : terms) {
        out += ",\"";
        out += term.first;
        out += "\":";
        out += docxml::fmt(term.second);
    }
    out += "}\n";
    return out;
}

struct JsonValue {
    bool is_string = false;
    double number = 0.0;
    std::string text;
};

/// A strict parser for one flat JSON object whose values are strings or
/// numbers. Nested objects/arrays, duplicate keys and trailing input are
/// rejected; callers validate the key set and value types.
class JsonReader {
public:
    explicit JsonReader(const std::string& text) : text_(text) {}

    std::map<std::string, JsonValue> parse() {
        skip_ws();
        expect('{', "expected '{'");
        std::map<std::string, JsonValue> out;
        skip_ws();
        if (peek() == '}') {
            ++pos_;
            finish();
            return out;
        }
        while (true) {
            skip_ws();
            const std::string key = parse_string();
            skip_ws();
            expect(':', "expected ':'");
            skip_ws();
            JsonValue value;
            if (peek() == '"') {
                value.is_string = true;
                value.text = parse_string();
            } else if (peek() == '{' || peek() == '[') {
                fail("nested values are not supported");
            } else {
                value.number = parse_number();
            }
            if (!out.emplace(key, std::move(value)).second) {
                fail("duplicate key '" + key + "'");
            }
            skip_ws();
            const char ch = peek();
            if (ch == ',') {
                ++pos_;
                continue;
            }
            expect('}', "expected ',' or '}'");
            break;
        }
        finish();
        return out;
    }

private:
    [[noreturn]] void fail(const std::string& message) const {
        throw ValueError(std::string(kPrefix) + "json: " + message + " at offset " +
                         std::to_string(pos_));
    }

    void skip_ws() {
        while (pos_ < text_.size() && is_ws(text_[pos_])) {
            ++pos_;
        }
    }

    char peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }

    void expect(char ch, const std::string& message) {
        if (peek() != ch) {
            fail(message);
        }
        ++pos_;
    }

    void finish() {
        skip_ws();
        if (pos_ != text_.size()) {
            fail("trailing characters");
        }
    }

    std::string parse_string();
    double parse_number();

    const std::string& text_;
    std::size_t pos_ = 0;
};

std::string JsonReader::parse_string() {
    if (peek() != '"') {
        fail("expected a string");
    }
    ++pos_;
    std::string out;
    while (pos_ < text_.size()) {
        const char ch = text_[pos_++];
        if (ch == '"') {
            return out;
        }
        if (ch != '\\') {
            out += ch;
            continue;
        }
        if (pos_ >= text_.size()) {
            fail("unterminated string escape");
        }
        const char esc = text_[pos_++];
        switch (esc) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            default: fail(std::string("unsupported escape \\") + esc);
        }
    }
    fail("unterminated string");
}

double JsonReader::parse_number() {
    const std::size_t begin = pos_;
    while (pos_ < text_.size()) {
        const char ch = text_[pos_];
        const bool digit = ch >= '0' && ch <= '9';
        if (!digit && ch != '+' && ch != '-' && ch != '.' && ch != 'e' && ch != 'E') {
            break;
        }
        ++pos_;
    }
    if (pos_ == begin) {
        fail("expected a number");
    }
    const std::string token = text_.substr(begin, pos_ - begin);
    try {
        return docxml::parse_double(token);
    } catch (const Error&) {
        fail("bad number '" + token + "'");
    }
}

using JsonObject = std::map<std::string, JsonValue>;

const JsonValue& json_required(const JsonObject& object, const char* key) {
    const auto it = object.find(key);
    if (it == object.end()) {
        throw ValueError(std::string(kPrefix) + "json: missing key '" + key + "'");
    }
    return it->second;
}

double json_coefficient(const JsonObject& object, const char* key, bool required) {
    const auto it = object.find(key);
    if (it == object.end()) {
        if (required) {
            throw ValueError(std::string(kPrefix) + "json: missing key '" + key + "'");
        }
        return 0.0;
    }
    if (it->second.is_string || !std::isfinite(it->second.number)) {
        throw ValueError(std::string(kPrefix) + "json: key '" + key +
                         "' must be a finite number");
    }
    return it->second.number;
}

std::string json_string(const JsonObject& object, const char* key) {
    const JsonValue& value = json_required(object, key);
    if (!value.is_string) {
        throw ValueError(std::string(kPrefix) + "json: key '" + key +
                         "' must be a string");
    }
    return value.text;
}

int json_integer(const JsonObject& object, const char* key) {
    const double value = json_coefficient(object, key, true);
    if (!std::isfinite(value) || value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max() || std::floor(value) != value) {
        throw ValueError(std::string(kPrefix) + "json: key '" + key +
                         "' must be an integer");
    }
    return static_cast<int>(value);
}

Calibration parse_json(const std::string& text) {
    const JsonObject object = JsonReader(text).parse();
    static const char* const kKnown[] = {
        "format", "version", "type",  "width", "height", "f",  "cx", "cy", "b1",
        "b2",     "k1",      "k2",    "k3",    "k4",     "p1", "p2", "p3", "p4"};
    for (const auto& entry : object) {
        bool known = false;
        for (const char* key : kKnown) {
            if (entry.first == key) {
                known = true;
                break;
            }
        }
        if (!known) {
            throw ValueError(std::string(kPrefix) + "json: unknown key '" +
                             entry.first + "'");
        }
    }
    if (json_string(object, "format") != "alphashape-calibration") {
        throw ValueError(std::string(kPrefix) + "json: unrecognised format");
    }
    const JsonValue& version = json_required(object, "version");
    if (version.is_string || version.number != 1.0) {
        throw ValueError(std::string(kPrefix) + "json: unsupported version");
    }
    Calibration c;
    c.type = parse_type_name(json_string(object, "type"));
    c.width = json_integer(object, "width");
    c.height = json_integer(object, "height");
    if (c.width <= 0 || c.height <= 0) {
        throw ValueError(std::string(kPrefix) + "json: width and height must be positive");
    }
    c.f = json_coefficient(object, "f", true);
    c.cx = json_coefficient(object, "cx", false);
    c.cy = json_coefficient(object, "cy", false);
    c.b1 = json_coefficient(object, "b1", false);
    c.b2 = json_coefficient(object, "b2", false);
    c.k1 = json_coefficient(object, "k1", false);
    c.k2 = json_coefficient(object, "k2", false);
    c.k3 = json_coefficient(object, "k3", false);
    c.k4 = json_coefficient(object, "k4", false);
    c.p1 = json_coefficient(object, "p1", false);
    c.p2 = json_coefficient(object, "p2", false);
    c.p3 = json_coefficient(object, "p3", false);
    c.p4 = json_coefficient(object, "p4", false);
    return c;
}

// ---- OpenCV common mapping ------------------------------------------------

struct OpenCvModel {
    int width = 0;
    int height = 0;
    double fx = 0.0;
    double skew = 0.0;
    double fy = 0.0;
    double cx = 0.0;  ///< OpenCV principal point, top-left pixel centre at (0, 0)
    double cy = 0.0;
    std::vector<double> distortion;  ///< OpenCV order (k1, k2, P1, P2, k3, ...)
};

std::vector<std::string> to_opencv(const Calibration& c, OpenCvModel& m) {
    if (c.type == CameraType::Fisheye) {
        throw ValueError(std::string(kPrefix) +
                         "a fisheye calibration is not representable in the OpenCV "
                         "pinhole format");
    }
    std::vector<std::string> warnings;
    const std::pair<double, const char*> omitted[] = {
        {c.k4, "k4"}, {c.p3, "p3"}, {c.p4, "p4"}};
    for (const auto& term : omitted) {
        if (term.first != 0.0) {
            warnings.emplace_back(std::string(kPrefix) + term.second +
                                  " is not representable in the OpenCV format; omitted");
        }
    }
    m.width = c.width;
    m.height = c.height;
    // u = W/2 + cx + x'(f+b1) + y'b2, with our top-left pixel centre at
    // (0.5, 0.5); OpenCV's is (0, 0), hence the -0.5 principal-point shift.
    m.fx = c.f + c.b1;
    m.skew = c.b2;
    m.fy = c.f;
    m.cx = 0.5 * static_cast<double>(c.width) + c.cx - 0.5;
    m.cy = 0.5 * static_cast<double>(c.height) + c.cy - 0.5;
    // Our p1 multiplies (r^2 + 2x^2) and our p2 multiplies 2xy; OpenCV's
    // distortion vector is (k1, k2, P1, P2, k3) with P1 on 2xy and P2 on
    // (r^2 + 2x^2), so P1 = our p2 and P2 = our p1.
    m.distortion = {c.k1, c.k2, c.p2, c.p1, c.k3};
    return warnings;
}

Calibration from_opencv(const OpenCvModel& m, std::vector<std::string>& warnings) {
    Calibration c;
    c.type = CameraType::Frame;
    c.width = m.width;
    c.height = m.height;
    c.f = m.fy;
    c.b1 = m.fx - m.fy;
    c.b2 = m.skew;
    c.cx = m.cx - 0.5 * static_cast<double>(m.width) + 0.5;
    c.cy = m.cy - 0.5 * static_cast<double>(m.height) + 0.5;
    const std::vector<double>& d = m.distortion;
    c.k1 = !d.empty() ? d[0] : 0.0;
    c.k2 = d.size() > 1U ? d[1] : 0.0;
    c.p2 = d.size() > 2U ? d[2] : 0.0;  // OpenCV P1
    c.p1 = d.size() > 3U ? d[3] : 0.0;  // OpenCV P2
    c.k3 = d.size() > 4U ? d[4] : 0.0;
    for (std::size_t i = 5; i < d.size(); ++i) {
        if (d[i] != 0.0) {
            warnings.emplace_back(std::string(kPrefix) + "distortion coefficient " +
                                  std::to_string(i + 1) +
                                  " is not representable and was ignored");
        }
    }
    return c;
}

// ---- OpenCV XML -----------------------------------------------------------

std::string opencv_numbers(const std::vector<double>& values) {
    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out += ' ';
        }
        out += docxml::fmt(values[i]);
    }
    return out;
}

std::string serialize_opencv_xml(const OpenCvModel& m) {
    std::string out = "<?xml version=\"1.0\"?>\n<opencv_storage>\n";
    out += "<image_width>" + std::to_string(m.width) + "</image_width>\n";
    out += "<image_height>" + std::to_string(m.height) + "</image_height>\n";
    out += "<camera_matrix type_id=\"opencv-matrix\">\n  <rows>3</rows>\n";
    out += "  <cols>3</cols>\n  <dt>d</dt>\n  <data>\n";
    out += opencv_numbers({m.fx, m.skew, m.cx}) + "\n";
    out += opencv_numbers({0.0, m.fy, m.cy}) + "\n";
    out += opencv_numbers({0.0, 0.0, 1.0}) + "\n";
    out += "  </data>\n</camera_matrix>\n";
    out += "<distortion_coefficients type_id=\"opencv-matrix\">\n";
    out += "  <rows>1</rows>\n  <cols>";
    out += std::to_string(m.distortion.size());
    out += "</cols>\n  <dt>d</dt>\n  <data>\n    ";
    out += opencv_numbers(m.distortion);
    out += "\n  </data>\n</distortion_coefficients>\n</opencv_storage>\n";
    return out;
}

const XmlNode& require_child(const XmlNode& node, const char* name) {
    const XmlNode* child = docxml::find_child(node, name);
    if (child == nullptr) {
        throw ValueError(std::string(kPrefix) + "opencv xml: missing <" + name + ">");
    }
    return *child;
}

double xml_number(const XmlNode& node, const char* name) {
    const std::string text = trim(require_child(node, name).text);
    try {
        return docxml::parse_double(text);
    } catch (const Error& e) {
        throw ValueError(std::string(kPrefix) + "opencv xml: <" + name + ">: " + e.what());
    }
}

int xml_int(const XmlNode& node, const char* name) {
    const double value = xml_number(node, name);
    if (!std::isfinite(value) || value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max() || std::floor(value) != value) {
        throw ValueError(std::string(kPrefix) + "opencv xml: <" + name +
                         "> must be an integer");
    }
    return static_cast<int>(value);
}

std::vector<double> parse_matrix_data(const XmlNode& root, const char* element, int& rows,
                                      int& cols) {
    const XmlNode& matrix = require_child(root, element);
    rows = xml_int(matrix, "rows");
    cols = xml_int(matrix, "cols");
    const XmlNode& data = require_child(matrix, "data");
    std::vector<double> values;
    for (const std::string& token : whitespace_tokens(data.text)) {
        try {
            values.push_back(docxml::parse_double(token));
        } catch (const Error&) {
            throw ValueError(std::string(kPrefix) + "opencv xml: bad number '" + token +
                             "' in <" + element + ">");
        }
        if (!std::isfinite(values.back())) {
            throw ValueError(std::string(kPrefix) + "opencv xml: non-finite value in <" +
                             element + ">");
        }
    }
    return values;
}

bool valid_distortion_count(std::size_t count) {
    return count == 4 || count == 5 || count == 8 || count == 12 || count == 14;
}

Calibration parse_opencv_xml(const std::string& text, int fallback_width,
                             int fallback_height, std::vector<std::string>& warnings) {
    XmlNode root;
    try {
        root = docxml::parse_xml(text);
    } catch (const Error& e) {
        throw ValueError(std::string(kPrefix) + e.what());
    }
    if (root.name != "opencv_storage") {
        throw ValueError(std::string(kPrefix) + "opencv xml: root element is <" +
                         root.name + ">, expected <opencv_storage>");
    }
    int k_rows = 0;
    int k_cols = 0;
    const std::vector<double> k = parse_matrix_data(root, "camera_matrix", k_rows, k_cols);
    if (k_rows != 3 || k_cols != 3 || k.size() != 9) {
        throw ValueError(std::string(kPrefix) +
                         "opencv xml: camera_matrix must be a 3x3 matrix");
    }
    // OpenCV's pinhole camera matrix has no perspective/scale terms beyond the
    // five used here.
    if (k[3] != 0.0 || k[6] != 0.0 || k[7] != 0.0 || k[8] != 1.0) {
        throw ValueError(std::string(kPrefix) +
                         "opencv xml: unsupported camera_matrix (expected K[1][0] = "
                         "K[2][0] = K[2][1] = 0 and K[2][2] = 1)");
    }
    int d_rows = 0;
    int d_cols = 0;
    const std::vector<double> distortion =
        parse_matrix_data(root, "distortion_coefficients", d_rows, d_cols);
    const std::size_t count = distortion.size();
    if ((d_rows != 1 && d_cols != 1) ||
        static_cast<std::size_t>(d_rows) * static_cast<std::size_t>(d_cols) != count ||
        !valid_distortion_count(count)) {
        throw ValueError(std::string(kPrefix) + "opencv xml: distortion_coefficients must "
                         "be a 1xN or Nx1 matrix of 4, 5, 8, 12 or 14 values");
    }
    auto dimension = [&root](const char* name, int fallback) {
        if (docxml::find_child(root, name) != nullptr) {
            const int value = xml_int(root, name);
            if (value <= 0) {
                throw ValueError(std::string(kPrefix) + "opencv xml: <" + name +
                                 "> must be positive");
            }
            return value;
        }
        if (fallback > 0) {
            return fallback;
        }
        throw ValueError(std::string(kPrefix) + "opencv xml: missing <" + name +
                         "> and no fallback image size");
    };
    OpenCvModel m;
    m.width = dimension("image_width", fallback_width);
    m.height = dimension("image_height", fallback_height);
    m.fx = k[0];
    m.skew = k[1];
    m.cx = k[2];
    m.fy = k[4];
    m.cy = k[5];
    m.distortion = distortion;
    return from_opencv(m, warnings);
}

// ---- OpenCV YAML ----------------------------------------------------------

std::string serialize_opencv_yaml(const OpenCvModel& m) {
    std::string out = "%YAML:1.0\n---\n";
    out += "image_width: " + std::to_string(m.width) + "\n";
    out += "image_height: " + std::to_string(m.height) + "\n";
    out += "camera_matrix: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n";
    out += "   data: [ ";
    out += opencv_numbers({m.fx, m.skew, m.cx, 0.0, m.fy, m.cy, 0.0, 0.0, 1.0});
    out += " ]\n";
    out += "distortion_coefficients: !!opencv-matrix\n   rows: 1\n   cols: ";
    out += std::to_string(m.distortion.size());
    out += "\n   dt: d\n   data: [ ";
    out += opencv_numbers(m.distortion);
    out += " ]\n";
    return out;
}

[[noreturn]] void yaml_fail(std::size_t line_index, const std::string& message) {
    throw ValueError(std::string(kPrefix) + "yaml: line " +
                     std::to_string(line_index + 1) + ": " + message);
}

std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    std::size_t begin = 0;
    while (true) {
        std::size_t end = text.find('\n', begin);
        if (end == std::string::npos) {
            end = text.size();
        }
        std::string line = text.substr(begin, end - begin);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(std::move(line));
        if (end == text.size()) {
            break;
        }
        begin = end + 1;
    }
    return lines;
}

bool indented(const std::string& line) {
    return !line.empty() && (line[0] == ' ' || line[0] == '\t');
}

struct YamlMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<double> data;
};

std::vector<double> parse_yaml_data(const std::string& text, std::size_t line_index) {
    const std::string value = trim(text);
    if (value.size() < 2U || value.front() != '[' || value.back() != ']') {
        yaml_fail(line_index, "matrix data must be enclosed in '[' and ']'");
    }
    std::vector<double> numbers;
    std::string token;
    auto flush = [&]() {
        if (token.empty()) {
            return;
        }
        try {
            numbers.push_back(docxml::parse_double(token));
        } catch (const Error&) {
            yaml_fail(line_index, "bad number '" + token + "'");
        }
        token.clear();
    };
    for (const char ch : value.substr(1, value.size() - 2)) {
        if (ch == ',' || is_ws(ch)) {
            flush();
        } else {
            token += ch;
        }
    }
    flush();
    for (const double number : numbers) {
        if (!std::isfinite(number)) {
            yaml_fail(line_index, "non-finite matrix value");
        }
    }
    return numbers;
}

int yaml_int_field(const std::map<std::string, std::string>& fields, const char* key,
                   std::size_t line_index) {
    const auto it = fields.find(key);
    if (it == fields.end()) {
        yaml_fail(line_index, std::string("matrix is missing '") + key + "'");
    }
    long long value = 0;
    try {
        value = docxml::parse_ll(it->second);
    } catch (const Error&) {
        yaml_fail(line_index, std::string("bad integer for '") + key + "'");
    }
    if (value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        yaml_fail(line_index, std::string("'") + key + "' is out of range");
    }
    return static_cast<int>(value);
}

YamlMatrix parse_yaml_matrix_block(const std::vector<std::string>& lines, std::size_t& index,
                                   std::size_t header_line) {
    std::map<std::string, std::string> fields;
    std::string data_text;
    bool have_data = false;
    while (index < lines.size()) {
        const std::string& raw = lines[index];
        if (trim(raw).empty()) {
            ++index;
            continue;
        }
        if (!indented(raw)) {
            break;
        }
        const std::string line = trim(raw);
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) {
            yaml_fail(index, "expected 'key: value' in a matrix block");
        }
        const std::string key = trim(line.substr(0, colon));
        if (key.empty()) {
            yaml_fail(index, "empty matrix field name");
        }
        if (key == "data") {
            data_text = trim(line.substr(colon + 1));
            while (data_text.find(']') == std::string::npos) {
                ++index;
                if (index >= lines.size()) {
                    yaml_fail(header_line, "unterminated matrix data");
                }
                data_text += " " + trim(lines[index]);
            }
            have_data = true;
        } else {
            fields[key] = trim(line.substr(colon + 1));
        }
        ++index;
    }
    YamlMatrix matrix;
    matrix.rows = yaml_int_field(fields, "rows", header_line);
    matrix.cols = yaml_int_field(fields, "cols", header_line);
    const auto dt = fields.find("dt");
    if (dt == fields.end() || dt->second != "d") {
        yaml_fail(header_line, "matrix dt must be 'd'");
    }
    if (!have_data) {
        yaml_fail(header_line, "matrix is missing 'data'");
    }
    matrix.data = parse_yaml_data(data_text, header_line);
    if (matrix.rows <= 0 || matrix.cols <= 0 ||
        static_cast<std::size_t>(matrix.rows) * static_cast<std::size_t>(matrix.cols) !=
            matrix.data.size()) {
        yaml_fail(header_line, "matrix rows x cols does not match its data length");
    }
    return matrix;
}

Calibration parse_opencv_yaml(const std::string& text, int fallback_width,
                              int fallback_height, std::vector<std::string>& warnings) {
    const std::vector<std::string> lines = split_lines(text);
    if (lines.empty() || trim(lines[0]).rfind("%YAML", 0) != 0) {
        throw ValueError(std::string(kPrefix) + "yaml: first line must start with \"%YAML\"");
    }
    std::map<std::string, std::string> scalars;
    std::map<std::string, YamlMatrix> matrices;
    std::size_t index = 1;
    while (index < lines.size()) {
        const std::string& raw = lines[index];
        const std::string line = trim(raw);
        if (line.empty() || line == "---") {
            ++index;
            continue;
        }
        if (indented(raw)) {
            yaml_fail(index, "unexpected indented line");
        }
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) {
            yaml_fail(index, "expected 'key: value'");
        }
        const std::string key = trim(line.substr(0, colon));
        const std::string value = trim(line.substr(colon + 1));
        if (key.empty()) {
            yaml_fail(index, "empty key");
        }
        const std::size_t header_line = index;
        ++index;
        if (value == "!!opencv-matrix") {
            matrices[key] = parse_yaml_matrix_block(lines, index, header_line);
        } else if (value.empty()) {
            yaml_fail(header_line, "key has no value");
        } else {
            scalars[key] = value;  // top-level scalars other than image sizes are ignored
        }
    }
    auto find_matrix = [&matrices](const char* name) -> const YamlMatrix& {
        const auto it = matrices.find(name);
        if (it == matrices.end()) {
            throw ValueError(std::string(kPrefix) + "yaml: missing '" + name + "' matrix");
        }
        return it->second;
    };
    const YamlMatrix& k = find_matrix("camera_matrix");
    if (k.rows != 3 || k.cols != 3 || k.data.size() != 9) {
        throw ValueError(std::string(kPrefix) + "yaml: camera_matrix must be a 3x3 matrix");
    }
    if (k.data[3] != 0.0 || k.data[6] != 0.0 || k.data[7] != 0.0 || k.data[8] != 1.0) {
        throw ValueError(std::string(kPrefix) + "yaml: unsupported camera_matrix");
    }
    const YamlMatrix& d = find_matrix("distortion_coefficients");
    if ((d.rows != 1 && d.cols != 1) ||
        static_cast<std::size_t>(d.rows) * static_cast<std::size_t>(d.cols) !=
            d.data.size() ||
        !valid_distortion_count(d.data.size())) {
        throw ValueError(std::string(kPrefix) + "yaml: distortion_coefficients must be a "
                         "1xN or Nx1 matrix of 4, 5, 8, 12 or 14 values");
    }
    auto dimension = [&scalars](const char* name, int fallback) {
        const auto it = scalars.find(name);
        if (it != scalars.end()) {
            long long value = 0;
            try {
                value = docxml::parse_ll(it->second);
            } catch (const Error&) {
                throw ValueError(std::string(kPrefix) + "yaml: bad integer for '" + name +
                                 "'");
            }
            if (value <= 0 || value > std::numeric_limits<int>::max()) {
                throw ValueError(std::string(kPrefix) + "yaml: '" + name +
                                 "' must be positive");
            }
            return static_cast<int>(value);
        }
        if (fallback > 0) {
            return fallback;
        }
        throw ValueError(std::string(kPrefix) + "yaml: missing '" + name +
                         "' and no fallback image size");
    };
    OpenCvModel m;
    m.width = dimension("image_width", fallback_width);
    m.height = dimension("image_height", fallback_height);
    m.fx = k.data[0];
    m.skew = k.data[1];
    m.cx = k.data[2];
    m.fy = k.data[4];
    m.cy = k.data[5];
    m.distortion = d.data;
    return from_opencv(m, warnings);
}

}  // namespace

std::vector<std::string> save_calibration(const std::string& path, const Calibration& c,
                                          CalibrationFormat format) {
    std::vector<std::string> warnings;
    std::string text;
    switch (format) {
        case CalibrationFormat::Xml:
            text = serialize_xml(c);
            break;
        case CalibrationFormat::Json:
            text = serialize_json(c);
            break;
        case CalibrationFormat::OpenCvXml: {
            OpenCvModel model;
            warnings = to_opencv(c, model);
            text = serialize_opencv_xml(model);
            break;
        }
        case CalibrationFormat::OpenCvYaml: {
            OpenCvModel model;
            warnings = to_opencv(c, model);
            text = serialize_opencv_yaml(model);
            break;
        }
    }
    write_text_file(path, text);
    return warnings;
}

CalibrationFileResult load_calibration(const std::string& path, CalibrationFormat format,
                                       int fallback_width, int fallback_height) {
    const std::string text = read_text_file(path);
    CalibrationFileResult result;
    switch (format) {
        case CalibrationFormat::Xml:
            result.calibration = parse_xml(text);
            break;
        case CalibrationFormat::Json:
            result.calibration = parse_json(text);
            break;
        case CalibrationFormat::OpenCvXml:
            result.calibration =
                parse_opencv_xml(text, fallback_width, fallback_height, result.warnings);
            break;
        case CalibrationFormat::OpenCvYaml:
            result.calibration =
                parse_opencv_yaml(text, fallback_width, fallback_height, result.warnings);
            break;
    }
    return result;
}

}  // namespace alphashape
