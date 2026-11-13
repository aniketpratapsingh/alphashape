#include "common/update/json.hpp"

#include <cstdint>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace alphashape::update::json {
namespace {

[[noreturn]] void fail(std::string_view message, std::size_t position) {
    throw std::runtime_error("invalid JSON at byte " + std::to_string(position) + ": " +
                             std::string(message));
}

void append_utf8(std::string& out, std::uint32_t code_point) {
    if (code_point <= 0x7F) {
        out.push_back(static_cast<char>(code_point));
    } else if (code_point <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else if (code_point <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    }
}

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    Value parse_document() {
        skip_space();
        Value value = parse_value();
        skip_space();
        if (position_ != text_.size()) {
            fail("trailing characters", position_);
        }
        return value;
    }

private:
    void skip_space() {
        while (position_ < text_.size()) {
            const char c = text_[position_];
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                break;
            }
            ++position_;
        }
    }

    bool consume(char expected) {
        if (position_ < text_.size() && text_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void expect(char expected) {
        if (!consume(expected)) {
            fail(std::string("expected '") + expected + "'", position_);
        }
    }

    Value parse_value() {
        if (position_ >= text_.size()) {
            fail("unexpected end of input", position_);
        }
        switch (text_[position_]) {
            case 'n': return literal("null", Value::make_null());
            case 't': return literal("true", Value::make_bool(true));
            case 'f': return literal("false", Value::make_bool(false));
            case '"': return Value::make_string(parse_string());
            case '[': return parse_array();
            case '{': return parse_object();
            default: return parse_number();
        }
    }

    Value literal(std::string_view word, Value value) {
        if (text_.substr(position_, word.size()) != word) {
            fail("unexpected token", position_);
        }
        position_ += word.size();
        return value;
    }

    Value parse_array() {
        expect('[');
        Value::Array items;
        skip_space();
        if (consume(']')) {
            return Value::make_array(std::move(items));
        }
        while (true) {
            skip_space();
            items.push_back(parse_value());
            skip_space();
            if (consume(']')) {
                break;
            }
            expect(',');
        }
        return Value::make_array(std::move(items));
    }

    Value parse_object() {
        expect('{');
        Value::Object members;
        skip_space();
        if (consume('}')) {
            return Value::make_object(std::move(members));
        }
        while (true) {
            skip_space();
            if (position_ >= text_.size() || text_[position_] != '"') {
                fail("object key must be a string", position_);
            }
            std::string key = parse_string();
            skip_space();
            expect(':');
            skip_space();
            members.insert_or_assign(std::move(key), parse_value());
            skip_space();
            if (consume('}')) {
                break;
            }
            expect(',');
        }
        return Value::make_object(std::move(members));
    }

    std::uint32_t parse_hex4() {
        if (position_ + 4 > text_.size()) {
            fail("truncated \\u escape", position_);
        }
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[position_++];
            value <<= 4;
            if (c >= '0' && c <= '9') {
                value |= static_cast<std::uint32_t>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                value |= static_cast<std::uint32_t>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                value |= static_cast<std::uint32_t>(c - 'A' + 10);
            } else {
                fail("bad hex digit in \\u escape", position_ - 1);
            }
        }
        return value;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (position_ < text_.size()) {
            const unsigned char c = static_cast<unsigned char>(text_[position_++]);
            if (c == '"') {
                return out;
            }
            if (c == '\\') {
                if (position_ >= text_.size()) {
                    fail("truncated escape", position_);
                }
                const char escape = text_[position_++];
                switch (escape) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        std::uint32_t code_point = parse_hex4();
                        if (code_point >= 0xD800 && code_point <= 0xDBFF) {
                            if (!consume('\\') || !consume('u')) {
                                fail("unpaired surrogate", position_);
                            }
                            const std::uint32_t low = parse_hex4();
                            if (low < 0xDC00 || low > 0xDFFF) {
                                fail("invalid low surrogate", position_);
                            }
                            code_point = 0x10000 + ((code_point - 0xD800) << 10) + (low - 0xDC00);
                        } else if (code_point >= 0xDC00 && code_point <= 0xDFFF) {
                            fail("unpaired surrogate", position_);
                        }
                        append_utf8(out, code_point);
                        break;
                    }
                    default: fail("unknown escape", position_ - 1);
                }
                continue;
            }
            if (c < 0x20) {
                fail("unescaped control character", position_ - 1);
            }
            out.push_back(static_cast<char>(c));
        }
        fail("unterminated string", position_);
    }

    Value parse_number() {
        const std::size_t start = position_;
        if (position_ < text_.size() && text_[position_] == '-') {
            ++position_;
        }
        while (position_ < text_.size()) {
            const char c = text_[position_];
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' ||
                c == '-') {
                ++position_;
            } else {
                break;
            }
        }
        if (position_ == start) {
            fail("expected a value", start);
        }
        // std::from_chars for floating point is not available in Apple's libc++
        // on the deployment target. A stream imbued with the classic locale is
        // locale-independent (unlike strtod), which matters because the GUI
        // process may run with a comma-decimal LC_NUMERIC.
        const std::string token(text_.substr(start, position_ - start));
        std::istringstream stream(token);
        stream.imbue(std::locale::classic());
        double number = 0.0;
        stream >> number;
        if (!stream || !stream.eof()) {
            fail("invalid number", start);
        }
        return Value::make_number(number);
    }

    std::string_view text_;
    std::size_t position_ = 0;
};

}  // namespace

// The private constructors are provided here so the public header stays free
// of parser details.
Value Value::make_null() { return Value(); }
Value Value::make_bool(bool value) {
    Value result;
    result.type_ = Type::Bool;
    result.boolean_ = value;
    return result;
}
Value Value::make_number(double value) {
    Value result;
    result.type_ = Type::Number;
    result.number_ = value;
    return result;
}
Value Value::make_string(std::string value) {
    Value result;
    result.type_ = Type::String;
    result.string_ = std::move(value);
    return result;
}
Value Value::make_array(Array value) {
    Value result;
    result.type_ = Type::Array;
    result.array_ = std::move(value);
    return result;
}
Value Value::make_object(Object value) {
    Value result;
    result.type_ = Type::Object;
    result.object_ = std::move(value);
    return result;
}

Value Value::parse(std::string_view text) { return Parser(text).parse_document(); }

const Value* Value::find(std::string_view key) const {
    if (type_ != Type::Object) {
        return nullptr;
    }
    const auto it = object_.find(key);
    return it == object_.end() ? nullptr : &it->second;
}

std::string Value::string_or(std::string_view key) const {
    const Value* member = find(key);
    return member != nullptr && member->is_string() ? member->string() : std::string();
}

bool Value::bool_or(std::string_view key, bool fallback) const {
    const Value* member = find(key);
    return member != nullptr && member->is_bool() ? member->boolean() : fallback;
}

}  // namespace alphashape::update::json
