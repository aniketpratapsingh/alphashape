#pragma once

/// @file reference_csv.hpp
/// @brief Delimited-text machinery for reference import/export (WP-13.2).
///
/// This is deliberately a small, dependency-free layer: RFC 4180 parsing for
/// CSV (quoted fields, doubled quotes, embedded newlines), a whitespace-run
/// reader for TXT, locale-independent number conversion (classic-locale
/// streams, because libc++ does not implement floating-point `from_chars`), and
/// a writer that quotes only when a field needs it.
///
/// A parsed row keeps its 1-based starting line number so an importer can name
/// the offending line in an error, which is what a user needs when a survey
/// export has one malformed record among thousands.

#include <cstddef>
#include <string>
#include <vector>

namespace alphashape::reference {

/// One record of a delimited file. `line` is the 1-based physical line the
/// record started on (a quoted field may span several lines).
struct DelimitedRow {
    std::vector<std::string> fields;
    std::size_t line = 0;

    bool operator==(const DelimitedRow& other) const {
        return fields == other.fields && line == other.line;
    }
};

/// Parse RFC 4180 CSV. `delimiter` is a single character. Records are
/// separated by LF or CRLF; empty records are dropped. A quoted field may
/// contain the delimiter, doubled quotes and newlines.
///
/// @throws ValueError naming the line when a quoted field is not terminated.
std::vector<DelimitedRow> parse_delimited(const std::string& text, char delimiter);

/// Parse whitespace-separated text: every run of spaces and tabs separates
/// fields, LF or CRLF separates records, and empty records are dropped.
std::vector<DelimitedRow> parse_whitespace(const std::string& text);

/// Decode Latin-1 bytes into UTF-8. Bytes below 0x80 are copied verbatim;
/// every other byte becomes the two-byte UTF-8 sequence U+0080..U+00FF.
std::string latin1_to_utf8(const std::string& text);

/// Parse a locale-independent double. When `decimal_comma` is set, a comma is
/// first replaced by a full stop (the delimiter in such a file is `;` or
/// whitespace). The whole field must be consumed; `context` is prefixed to the
/// error so callers can name the file and line.
///
/// @throws ValueError when the field is not a complete number.
double parse_number(const std::string& field, bool decimal_comma,
                    const std::string& context);

/// True when `field` parses completely as a number (used for header detection).
bool is_number(const std::string& field, bool decimal_comma = false);

/// Locale-independent fixed-precision formatting.
std::string format_number(double value, int precision);

/// Quote `value` for a CSV field when it contains `delimiter`, a quote, CR or
/// LF; otherwise return it unchanged.
std::string quote_field(const std::string& value, char delimiter);

/// Join fields with `delimiter`, quoting each as required, and append CRLF.
std::string format_delimited(const std::vector<std::string>& fields, char delimiter);

}  // namespace alphashape::reference
