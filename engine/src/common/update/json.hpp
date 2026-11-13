#pragma once

// A deliberately small JSON reader for the GitHub Releases API response.
//
// The engine had no JSON library: the project format is XML and the Python
// bridge has no use for JSON, so the updater owns the ~200 lines it needs
// instead of adding a dependency. It is a complete RFC 8259 reader (UTF-8,
// escapes, surrogate pairs); the writer is not needed and not provided.

#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace alphashape::update::json {

enum class Type { Null, Bool, Number, String, Array, Object };

class Value {
public:
    using Array = std::vector<Value>;
    using Object = std::map<std::string, Value, std::less<>>;

    Value() = default;

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool boolean() const { return boolean_; }
    double number() const { return number_; }
    const std::string& string() const { return string_; }
    const Array& array() const { return array_; }
    const Object& object() const { return object_; }

    /// Object member, or nullptr when absent or when this is not an object.
    const Value* find(std::string_view key) const;

    /// A string member, or "" when absent or not a string.
    std::string string_or(std::string_view key) const;
    /// A bool member, or `fallback` when absent or not a bool.
    bool bool_or(std::string_view key, bool fallback) const;

    /// Parse `text`. Throws std::runtime_error with a position when the text
    /// is not valid JSON.
    static Value parse(std::string_view text);

    // Construction helpers for the reader. Public so the reader can live in an
    // anonymous namespace without a second, unrelated Parser class.
    static Value make_null();
    static Value make_bool(bool value);
    static Value make_number(double value);
    static Value make_string(std::string value);
    static Value make_array(Array value);
    static Value make_object(Object value);

private:
    Type type_ = Type::Null;
    bool boolean_ = false;
    double number_ = 0.0;
    std::string string_;
    Array array_;
    Object object_;
};

}  // namespace alphashape::update::json
