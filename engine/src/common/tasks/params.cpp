#include "common/tasks/params.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <sstream>

namespace alphashape {

namespace {

/// Throw unless the variant holds the type the declaration asks for.
void require_type(const std::string& name, ParamType type, const ParamValue& value) {
    bool ok = false;
    switch (type) {
        case ParamType::Bool:       ok = std::holds_alternative<bool>(value); break;
        case ParamType::Int:        ok = std::holds_alternative<std::int64_t>(value); break;
        case ParamType::Float:      ok = std::holds_alternative<double>(value); break;
        case ParamType::String:     ok = std::holds_alternative<std::string>(value); break;
        case ParamType::StringList:
            ok = std::holds_alternative<std::vector<std::string>>(value);
            break;
    }
    if (!ok) {
        throw ValueError("parameter '" + name + "' has the wrong type");
    }
}

// ---------------------------------------------------------------------------
// The list text form: a JSON array of JSON strings.
//
// Why JSON and not a separator character. A list of file names contains
// commas, quotes and backslashes on every platform, so any "split on X" scheme
// needs an escaping rule anyway — and once there is an escaping rule, it may as
// well be a published one. JSON's string grammar is small, universally
// understood, and exactly invertible for any byte sequence, so the doc.xml
// meta, the CLI's --set and the XML reader all agree by construction.
//
// Encoding: `"` `\` and the C0 controls are escaped (`\n`, `\t`, `\r`, `\b`,
// `\f` short forms, `\u00XX` for the rest). Bytes >= 0x80 are passed through
// verbatim: the input is UTF-8 and JSON permits raw UTF-8 in a string, so
// re-encoding them as \u escapes would corrupt nothing but would make the
// round-trip depend on the writer's decoder, which is a place bugs live.
// Everything is byte-oriented, so a non-UTF-8 byte sequence also round-trips
// unchanged even though it is not strictly legal JSON.
// ---------------------------------------------------------------------------

void append_hex4(std::string& out, std::uint32_t value) {
    static const char kHex[] = "0123456789abcdef";
    out += "\\u";
    out += kHex[(value >> 12) & 0xF];
    out += kHex[(value >> 8) & 0xF];
    out += kHex[(value >> 4) & 0xF];
    out += kHex[value & 0xF];
}

std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 2);
    out += '"';
    for (const char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                // Control characters (U+0000..U+001F) must be escaped; every
                // other byte, including all of UTF-8's continuation bytes, is
                // emitted as is.
                if (static_cast<unsigned char>(c) < 0x20) {
                    append_hex4(out, static_cast<std::uint32_t>(static_cast<unsigned char>(c)));
                } else {
                    out += c;
                }
                break;
        }
    }
    out += '"';
    return out;
}

/// Append the UTF-8 encoding of a code point.
void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

/// A hand-written JSON-array-of-strings parser. It accepts exactly the grammar
/// the encoder produces plus insignificant whitespace between tokens, and
/// rejects everything else: the point of the closed grammar is that a
/// malformed value in doc.xml or on the command line is an error, never a
/// silently different list.
class ListParser {
public:
    ListParser(const std::string& text, std::string name)
        : text_(text), name_(std::move(name)) {}

    std::vector<std::string> parse() {
        skip_space();
        expect('[');
        std::vector<std::string> out;
        skip_space();
        if (peek() == ']') {
            ++pos_;
            skip_space();
            expect_end();
            return out;
        }
        for (;;) {
            out.push_back(parse_string());
            skip_space();
            const char c = peek();
            if (c == ',') {
                ++pos_;
                skip_space();
                continue;
            }
            if (c == ']') {
                ++pos_;
                break;
            }
            fail("expected ',' or ']'");
        }
        skip_space();
        expect_end();
        return out;
    }

private:
    char peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }

    void skip_space() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    void expect(char c) {
        if (peek() != c) {
            fail(std::string("expected '") + c + "'");
        }
        ++pos_;
    }

    void expect_end() {
        if (pos_ != text_.size()) {
            fail("trailing characters after the array");
        }
    }

    [[noreturn]] void fail(const std::string& what) const {
        std::string message = "cannot parse '" + text_ + "' as a string list";
        if (!name_.empty()) {
            message += " for parameter '" + name_ + "'";
        }
        throw ValueError(message + ": " + what);
    }

    static int hex_digit(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    std::uint32_t parse_hex4() {
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const int digit = hex_digit(peek());
            if (digit < 0) {
                fail("malformed \\u escape");
            }
            value = (value << 4) | static_cast<std::uint32_t>(digit);
            ++pos_;
        }
        return value;
    }

    std::string parse_string() {
        if (peek() != '"') {
            fail("expected a quoted string");
        }
        ++pos_;
        std::string out;
        for (;;) {
            if (pos_ >= text_.size()) {
                fail("unterminated string");
            }
            const char c = text_[pos_++];
            if (c == '"') {
                return out;
            }
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= text_.size()) {
                fail("unterminated escape");
            }
            const char esc = text_[pos_++];
            switch (esc) {
                case '"':  out += '"'; break;
                case '\\': out += '\\'; break;
                case '/':  out += '/'; break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    std::uint32_t cp = parse_hex4();
                    // A high surrogate must be followed by its low surrogate;
                    // the pair encodes one code point outside the BMP.
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (peek() != '\\' || pos_ + 1 >= text_.size() ||
                            text_[pos_ + 1] != 'u') {
                            fail("high surrogate without a low surrogate");
                        }
                        pos_ += 2;
                        const std::uint32_t low = parse_hex4();
                        if (low < 0xDC00 || low > 0xDFFF) {
                            fail("malformed surrogate pair");
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        fail("unpaired low surrogate");
                    }
                    append_utf8(out, cp);
                    break;
                }
                default:
                    fail("unknown escape");
            }
        }
    }

    const std::string& text_;
    std::string name_;
    std::size_t pos_ = 0;
};

}  // namespace

void Params::declare(const ParamSpec& spec) {
    for (const auto& existing : specs_) {
        if (existing.name == spec.name) {
            throw ValueError("parameter '" + spec.name + "' declared twice");
        }
    }
    // A default that disagrees with the declared type would only surface when
    // a caller reads the parameter, far from the declaration that is wrong.
    if (!std::holds_alternative<std::monostate>(spec.default_value)) {
        require_type(spec.name, spec.type, spec.default_value);
    }
    // The constraints are documentation for the front ends, but a bound or a
    // choice list that contradicts the declaration is a declaration bug, so it
    // is caught here rather than presenting an editor that cannot be satisfied.
    for (const ParamValue* bound : {&spec.minimum, &spec.maximum}) {
        if (!std::holds_alternative<std::monostate>(*bound)) {
            require_type(spec.name, spec.type, *bound);
        }
    }
    if (spec.power_of_two && spec.type != ParamType::Int) {
        throw ValueError("parameter '" + spec.name + "': power_of_two needs an Int parameter");
    }
    if (!spec.choices.empty()) {
        if (spec.type != ParamType::String) {
            throw ValueError("parameter '" + spec.name + "': choices need a String parameter");
        }
        const auto* chosen = std::get_if<std::string>(&spec.default_value);
        const bool listed = chosen != nullptr && std::find(spec.choices.begin(),
                                                          spec.choices.end(), *chosen) !=
                                                      spec.choices.end();
        if (chosen != nullptr && !listed) {
            throw ValueError("parameter '" + spec.name + "': the default '" + *chosen +
                             "' is not one of its choices");
        }
    }
    specs_.push_back(spec);
}

bool Params::declares(const std::string& name) const {
    return find_spec(name) != nullptr;
}

const ParamSpec* Params::find_spec(const std::string& name) const {
    for (const auto& spec : specs_) {
        if (spec.name == name) {
            return &spec;
        }
    }
    return nullptr;
}

void Params::set(const std::string& name, ParamValue value) {
    const ParamSpec* spec = find_spec(name);
    if (spec == nullptr) {
        throw InvalidArgumentError(name);
    }
    if (std::holds_alternative<std::monostate>(value)) {
        values_.erase(name);
        return;
    }
    require_type(name, spec->type, value);
    values_[name] = std::move(value);
}

bool Params::is_set(const std::string& name) const {
    return values_.find(name) != values_.end();
}

ParamValue Params::get(const std::string& name) const {
    const ParamSpec* spec = find_spec(name);
    if (spec == nullptr) {
        throw InvalidArgumentError(name);
    }
    const auto it = values_.find(name);
    if (it != values_.end()) {
        return it->second;
    }
    if (std::holds_alternative<std::monostate>(spec->default_value)) {
        throw ValueError("required parameter '" + name + "' was not provided");
    }
    return spec->default_value;
}

bool Params::get_bool(const std::string& name) const {
    return std::get<bool>(get(name));
}

std::int64_t Params::get_int(const std::string& name) const {
    return std::get<std::int64_t>(get(name));
}

double Params::get_float(const std::string& name) const {
    return std::get<double>(get(name));
}

std::string Params::get_string(const std::string& name) const {
    return std::get<std::string>(get(name));
}

std::vector<std::string> Params::get_string_list(const std::string& name) const {
    return std::get<std::vector<std::string>>(get(name));
}

std::vector<std::string> Params::explicitly_set() const {
    std::vector<std::string> names;
    names.reserve(values_.size());
    for (const auto& [name, value] : values_) {
        (void)value;
        names.push_back(name);
    }
    return names;  // std::map iterates in key order: deterministic output
}

ParamValue parse_param_value(ParamType type, const std::string& text) {
    switch (type) {
        case ParamType::Bool: {
            if (text == "true" || text == "1") return true;
            if (text == "false" || text == "0") return false;
            throw ValueError("cannot parse '" + text + "' as bool");
        }
        case ParamType::Int: {
            std::int64_t value = 0;
            const char* begin = text.data();
            const char* end = begin + text.size();
            const auto [ptr, ec] = std::from_chars(begin, end, value);
            if (ec != std::errc{} || ptr != end) {
                throw ValueError("cannot parse '" + text + "' as int");
            }
            return value;
        }
        case ParamType::Float: {
            try {
                std::size_t consumed = 0;
                const double value = std::stod(text, &consumed);
                if (consumed != text.size()) {
                    throw ValueError("cannot parse '" + text + "' as float");
                }
                return value;
            } catch (const std::invalid_argument&) {
                throw ValueError("cannot parse '" + text + "' as float");
            } catch (const std::out_of_range&) {
                throw ValueError("'" + text + "' is out of range for float");
            }
        }
        case ParamType::String:
            return text;
        case ParamType::StringList:
            return ListParser(text, {}).parse();
    }
    throw ValueError("unknown parameter type");
}

std::string format_param_value(const ParamValue& value) {
    if (std::holds_alternative<std::monostate>(value)) {
        return {};
    }
    if (const auto* b = std::get_if<bool>(&value)) {
        return *b ? "true" : "false";
    }
    if (const auto* i = std::get_if<std::int64_t>(&value)) {
        return std::to_string(*i);
    }
    if (const auto* d = std::get_if<double>(&value)) {
        std::ostringstream out;
        out.precision(17);
        out << *d;
        return out.str();
    }
    if (const auto* list = std::get_if<std::vector<std::string>>(&value)) {
        std::string out = "[";
        for (std::size_t i = 0; i < list->size(); ++i) {
            if (i != 0) {
                out += ',';
            }
            out += json_escape((*list)[i]);
        }
        out += ']';
        return out;
    }
    return std::get<std::string>(value);
}

}  // namespace alphashape
