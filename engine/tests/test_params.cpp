// Tests for the string-keyed task parameter schema.

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/tasks/params.hpp"

namespace alphashape {
namespace {

/// Run `f`, return the message of the caught `Ex` (or a marker if none / other).
template <typename Ex, typename F>
std::string message_of(F&& f) {
    try {
        f();
    } catch (const Ex& e) {
        return e.what();
    } catch (const std::exception& e) {
        return std::string("<other exception: ") + e.what() + ">";
    } catch (...) {
        return "<non-std exception>";
    }
    return "<no exception>";
}

Params with_int(const std::string& name, std::int64_t default_value) {
    Params p;
    p.declare(ParamSpec{name, ParamType::Int, default_value, ""});
    return p;
}

TEST(Params, DeclaredDefaultIsReturnedWhenUnset) {
    Params p = with_int("limit", 40000);
    EXPECT_TRUE(p.declares("limit"));
    EXPECT_FALSE(p.declares("missing"));
    EXPECT_FALSE(p.is_set("limit"));
    EXPECT_EQ(p.get_int("limit"), 40000);
}

TEST(Params, SetOverridesDefaultAndReportsExplicitlySet) {
    Params p = with_int("limit", 40000);
    p.set("limit", std::int64_t{7});
    EXPECT_TRUE(p.is_set("limit"));
    EXPECT_EQ(p.get_int("limit"), 7);
    EXPECT_EQ(p.explicitly_set(), std::vector<std::string>{"limit"});
}

TEST(Params, ExplicitlySetExcludesDefaultsAndIsSorted) {
    Params p;
    p.declare(ParamSpec{"alpha", ParamType::Int, std::int64_t{1}, ""});
    p.declare(ParamSpec{"beta", ParamType::Int, std::int64_t{2}, ""});
    p.declare(ParamSpec{"gamma", ParamType::Int, std::int64_t{3}, ""});
    // Set out of alphabetical order; only these two are explicit.
    p.set("gamma", std::int64_t{30});
    p.set("beta", std::int64_t{20});
    EXPECT_EQ(p.explicitly_set(), (std::vector<std::string>{"beta", "gamma"}));
}

TEST(Params, UndeclaredKeyRejectedOnSetWithInvalidArgumentError) {
    Params p = with_int("limit", 1);
    const std::string message = message_of<InvalidArgumentError>([&] {
        p.set("bogus", std::int64_t{1});
    });
    EXPECT_NE(message.find("Invalid argument name: bogus"), std::string::npos) << message;
}

TEST(Params, UndeclaredKeyRejectedOnGetWithInvalidArgumentError) {
    Params p = with_int("limit", 1);
    const std::string message = message_of<InvalidArgumentError>([&] {
        (void)p.get("bogus");
    });
    EXPECT_NE(message.find("Invalid argument name: bogus"), std::string::npos) << message;
}

TEST(Params, WrongValueTypeThrowsValueError) {
    Params p = with_int("limit", 1);
    EXPECT_THROW(p.set("limit", std::string("nope")), ValueError);
    EXPECT_THROW(p.set("limit", true), ValueError);
    EXPECT_THROW(p.set("limit", 1.5), ValueError);
}

TEST(Params, DeclaredDefaultOfWrongTypeThrowsAtDeclare) {
    Params p;
    // Type says Int, default says string.
    EXPECT_THROW(p.declare(ParamSpec{"limit", ParamType::Int, std::string("x"), ""}), ValueError);
}

TEST(Params, RequiredParameterUnsetThrowsValueError) {
    Params p;
    p.declare(ParamSpec{"required", ParamType::Int, std::monostate{}, ""});
    EXPECT_THROW(p.get_int("required"), ValueError);
    // Providing it succeeds.
    p.set("required", std::int64_t{5});
    EXPECT_EQ(p.get_int("required"), 5);
}

TEST(Params, DoubleRoundTripsExactly) {
    Params p;
    p.declare(ParamSpec{"d", ParamType::Float, 0.0, ""});
    for (const double value : {0.1, 1e-300, 1.0 / 3.0, -1e-12, 3.141592653589793}) {
        const ParamValue parsed = parse_param_value(ParamType::Float, format_param_value(value));
        EXPECT_EQ(parsed, ParamValue{value}) << "value " << value;
    }
}

TEST(Params, EveryTypeRoundTripsThroughParseFormat) {
    EXPECT_EQ(parse_param_value(ParamType::Bool, format_param_value(true)), ParamValue{true});
    EXPECT_EQ(parse_param_value(ParamType::Bool, format_param_value(false)), ParamValue{false});
    EXPECT_EQ(parse_param_value(ParamType::Int, format_param_value(std::int64_t{-12})),
              ParamValue{std::int64_t{-12}});
    EXPECT_EQ(parse_param_value(ParamType::Float, format_param_value(2.5)), ParamValue{2.5});
    EXPECT_EQ(parse_param_value(ParamType::String, format_param_value(std::string("hi there"))),
              ParamValue{std::string("hi there")});
}

TEST(Params, BoolParsingAcceptsTrueOnesFalseZeros) {
    EXPECT_EQ(parse_param_value(ParamType::Bool, "true"), ParamValue{true});
    EXPECT_EQ(parse_param_value(ParamType::Bool, "1"), ParamValue{true});
    EXPECT_EQ(parse_param_value(ParamType::Bool, "false"), ParamValue{false});
    EXPECT_EQ(parse_param_value(ParamType::Bool, "0"), ParamValue{false});
    EXPECT_THROW(parse_param_value(ParamType::Bool, "yes"), ValueError);
    EXPECT_THROW(parse_param_value(ParamType::Bool, "TRUE"), ValueError);
}

TEST(Params, IntParsingRejectsTrailingGarbage) {
    EXPECT_EQ(parse_param_value(ParamType::Int, "12"), ParamValue{std::int64_t{12}});
    EXPECT_EQ(parse_param_value(ParamType::Int, "-3"), ParamValue{std::int64_t{-3}});
    EXPECT_THROW(parse_param_value(ParamType::Int, "12abc"), ValueError);
    EXPECT_THROW(parse_param_value(ParamType::Int, "abc"), ValueError);
}

TEST(Params, FloatParsingRejectsMalformedText) {
    EXPECT_EQ(parse_param_value(ParamType::Float, "0.1"), ParamValue{0.1});
    EXPECT_THROW(parse_param_value(ParamType::Float, "1.2.3"), ValueError);
    EXPECT_THROW(parse_param_value(ParamType::Float, "1.5x"), ValueError);
}

// ---------------------------------------------------------------------------
// StringList
// ---------------------------------------------------------------------------

/// Round-trip helper: format, parse, and compare against the original.
void expect_list_round_trip(const std::vector<std::string>& list) {
    const std::string text = format_param_value(ParamValue{list});
    EXPECT_EQ(parse_param_value(ParamType::StringList, text), ParamValue{list})
        << "text was " << text;
}

TEST(Params, StringListRoundTripsEmptyList) {
    expect_list_round_trip({});
    EXPECT_EQ(format_param_value((ParamValue{std::vector<std::string>{}})), "[]");
}

TEST(Params, StringListRoundTripsOrdinaryNames) {
    const std::vector<std::string> list{"a.jpg", "dir/b c.jpg", "x"};
    expect_list_round_trip(list);
    EXPECT_EQ(format_param_value(ParamValue{list}), "[\"a.jpg\",\"dir/b c.jpg\",\"x\"]");
}

TEST(Params, StringListRoundTripsCommasAndQuotesAndBackslashes) {
    // A comma is an ordinary character inside a JSON string; the whole reason
    // the text form is JSON rather than a comma-separated list.
    expect_list_round_trip({"a,b", "say \"hi\"", "c:\\path\\to\\file", "back\\\\slash"});
}

TEST(Params, StringListRoundTripsControlCharacters) {
    expect_list_round_trip({"line\nbreak", "tab\there", "cr\r\b\f", std::string("\x01\x1f", 2)});
    // The short escapes must actually be used for the common controls.
    EXPECT_EQ(format_param_value((ParamValue{std::vector<std::string>{"a\nb"}})),
              "[\"a\\nb\"]");
}

TEST(Params, StringListRoundTripsNonAsciiUtf8) {
    // Raw UTF-8 must survive verbatim; the tool must not mangle it into
    // \u escapes or replacement characters.
    const std::vector<std::string> list{"caf\xC3\xA9.jpg", "\xE6\x97\xA5\xE6\x9C\xAC.png",
                                        "\xF0\x9F\x93\xB7.jpg"};
    expect_list_round_trip(list);
    EXPECT_EQ(format_param_value(ParamValue{list}),
              "[\"caf\xC3\xA9.jpg\",\"\xE6\x97\xA5\xE6\x9C\xAC.png\",\"\xF0\x9F\x93\xB7.jpg\"]");
}

TEST(Params, StringListParsingAcceptsWhitespaceBetweenTokens) {
    const ParamValue parsed =
        parse_param_value(ParamType::StringList, "  [ \"a\" ,\n\t\"b\"  ]  ");
    EXPECT_EQ(parsed, (ParamValue{std::vector<std::string>{"a", "b"}}));
}

TEST(Params, StringListParsingAcceptsUnicodeEscapes) {
    EXPECT_EQ(parse_param_value(ParamType::StringList, "[\"\\u0041\\u00e9\"]"),
              (ParamValue{std::vector<std::string>{"A\xC3\xA9"}}));
    // A surrogate pair is one code point outside the BMP: U+1F4F7.
    EXPECT_EQ(parse_param_value(ParamType::StringList, "[\"\\ud83d\\udcf7\"]"),
              (ParamValue{std::vector<std::string>{"\xF0\x9F\x93\xB7"}}));
}

TEST(Params, StringListParsingRejectsMalformedText) {
    for (const char* text : {"", "[", "[\"a\"", "[\"a\" x]", "[\"a\"]junk", "[\"a\",]",
                             "[\"a\", 1]", "[a]", "[\"\\ud83d\"]", "[\"\\q\"]",
                             "[\"\\uZZZZ\"]", "[\"a\"][\"b\"]", "{\"a\":1}"}) {
        EXPECT_THROW(parse_param_value(ParamType::StringList, text), ValueError) << "text: " << text;
    }
}

TEST(Params, StringListRejectsWrongValueTypeOnSet) {
    Params p;
    p.declare(ParamSpec{"files", ParamType::StringList, std::monostate{}, ""});
    // A bare string is the classic bug: the caller meant one name, the binder
    // says list. The schema rejects it rather than iterating the characters.
    EXPECT_THROW(p.set("files", std::string("a.jpg")), ValueError);
    EXPECT_THROW(p.set("files", std::int64_t{1}), ValueError);
    EXPECT_THROW(p.set("files", true), ValueError);
    EXPECT_THROW(p.set("files", 1.5), ValueError);

    p.set("files", std::vector<std::string>{"a.jpg"});
    EXPECT_EQ(p.get_string_list("files"), (std::vector<std::string>{"a.jpg"}));
}

TEST(Params, StringListDeclaredDefaultOfWrongTypeThrowsAtDeclare) {
    Params p;
    EXPECT_THROW(p.declare(ParamSpec{"files", ParamType::StringList, std::string("x"), ""}),
                 ValueError);
    // A list default is accepted and returned when unset.
    p.declare(ParamSpec{"files", ParamType::StringList,
                        std::vector<std::string>{"a", "b"}, ""});
    EXPECT_EQ(p.get_string_list("files"), (std::vector<std::string>{"a", "b"}));
}

TEST(Params, StringListGetterOnScalarThrows) {
    Params p = with_int("limit", 1);
    EXPECT_THROW(p.get_string_list("limit"), std::bad_variant_access);
}

}  // namespace
}  // namespace alphashape
