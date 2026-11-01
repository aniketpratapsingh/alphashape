#include "common/algorithm/reference/reference_csv.hpp"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>

#include "base/utils/error.hpp"

namespace alphashape::reference {
namespace {

std::string line_error(std::size_t line, const std::string& message) {
    return "line " + std::to_string(line) + ": " + message;
}

/// Split the text into physical lines, dropping the CR of a CRLF pair.
std::vector<std::string> physical_lines(const std::string& text) {
    std::vector<std::string> lines;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t nl = text.find('\n', start);
        const std::size_t end = nl == std::string::npos ? text.size() : nl;
        std::string line = text.substr(start, end - start);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(std::move(line));
        if (nl == std::string::npos) {
            break;
        }
        start = nl + 1;
    }
    return lines;
}

bool is_space(char c) { return c == ' ' || c == '\t'; }

}  // namespace

std::vector<DelimitedRow> parse_delimited(const std::string& text, char delimiter) {
    std::vector<DelimitedRow> rows;
    DelimitedRow row;
    row.line = 1;
    std::string field;
    bool quoted = false;
    bool after_quote = false;
    std::size_t line = 1;

    const auto finish_field = [&]() {
        row.fields.push_back(std::move(field));
        field.clear();
        after_quote = false;
    };
    const auto finish_row = [&](bool has_content) {
        finish_field();
        if (has_content || !row.fields.empty()) {
            // A completely empty line yields one empty field; drop it.
            if (!(row.fields.size() == 1 && row.fields[0].empty())) {
                rows.push_back(std::move(row));
            }
        }
        row = DelimitedRow{};
        row.line = line;
    };

    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (quoted) {
            if (c == '"') {
                if (i + 1 < text.size() && text[i + 1] == '"') {
                    field.push_back('"');
                    ++i;
                } else {
                    quoted = false;
                    after_quote = true;
                }
            } else {
                if (c == '\n') {
                    ++line;
                }
                field.push_back(c);
            }
            continue;
        }
        if (c == '\r') {
            continue;
        }
        if (c == '\n') {
            finish_row(!row.fields.empty() || !field.empty());
            ++line;
            row.line = line;
            continue;
        }
        if (c == delimiter) {
            finish_field();
            continue;
        }
        if (c == '"' && field.empty() && !after_quote) {
            quoted = true;
            continue;
        }
        field.push_back(c);
    }
    if (quoted) {
        throw ValueError(line_error(row.line, "unterminated quoted field"));
    }
    if (!field.empty() || !row.fields.empty()) {
        finish_row(true);
    }
    return rows;
}

std::vector<DelimitedRow> parse_whitespace(const std::string& text) {
    std::vector<DelimitedRow> rows;
    const std::vector<std::string> lines = physical_lines(text);
    for (std::size_t li = 0; li < lines.size(); ++li) {
        DelimitedRow row;
        row.line = li + 1;
        std::size_t i = 0;
        while (i < lines[li].size()) {
            while (i < lines[li].size() && is_space(lines[li][i])) {
                ++i;
            }
            if (i >= lines[li].size()) {
                break;
            }
            const std::size_t start = i;
            while (i < lines[li].size() && !is_space(lines[li][i])) {
                ++i;
            }
            row.fields.push_back(lines[li].substr(start, i - start));
        }
        if (!row.fields.empty()) {
            rows.push_back(std::move(row));
        }
    }
    return rows;
}

std::string latin1_to_utf8(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (const char raw : text) {
        const auto byte = static_cast<unsigned char>(raw);
        if (byte < 0x80U) {
            out.push_back(raw);
        } else {
            out.push_back(static_cast<char>(0xC0U | (byte >> 6)));
            out.push_back(static_cast<char>(0x80U | (byte & 0x3FU)));
        }
    }
    return out;
}

double parse_number(const std::string& field, bool decimal_comma,
                    const std::string& context) {
    std::string text = field;
    if (decimal_comma) {
        for (char& c : text) {
            if (c == ',') {
                c = '.';
            }
        }
    }
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end && is_space(text[begin])) {
        ++begin;
    }
    while (end > begin && is_space(text[end - 1])) {
        --end;
    }
    const std::string trimmed = text.substr(begin, end - begin);
    if (trimmed.empty()) {
        throw ValueError(context + ": expected a number, got an empty field");
    }
    std::string lowered = trimmed;
    for (char& c : lowered) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    if (lowered == "nan") {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (lowered == "inf" || lowered == "+inf" || lowered == "infinity") {
        return std::numeric_limits<double>::infinity();
    }
    if (lowered == "-inf" || lowered == "-infinity") {
        return -std::numeric_limits<double>::infinity();
    }
    // A classic-locale stream, not `std::from_chars`: libc++ (macOS) does not
    // implement the floating-point from_chars overload, while the classic
    // locale is equally independent of the process's C locale and always uses
    // "." as the decimal point.
    std::istringstream in(trimmed);
    in.imbue(std::locale::classic());
    double value = 0.0;
    in >> value;
    if (!in || in.peek() != std::char_traits<char>::eof()) {
        throw ValueError(context + ": '" + field + "' is not a number");
    }
    return value;
}

bool is_number(const std::string& field, bool decimal_comma) {
    try {
        (void)parse_number(field, decimal_comma, "");
        return true;
    } catch (const ValueError&) {
        return false;
    }
}

std::string format_number(double value, int precision) {
    if (!std::isfinite(value)) {
        return std::isnan(value) ? "nan" : (value > 0.0 ? "inf" : "-inf");
    }
    // Fixed with `precision` digits after the point is what a survey export
    // means by "6 decimals". The classic locale never consults the C locale
    // and, unlike libc++'s missing floating-point to_chars, is portable.
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(precision < 0 ? 0 : precision) << value;
    return out.str();
}

std::string quote_field(const std::string& value, char delimiter) {
    const bool needs_quotes =
        value.find(delimiter) != std::string::npos ||
        value.find('"') != std::string::npos ||
        value.find('\n') != std::string::npos ||
        value.find('\r') != std::string::npos;
    if (!needs_quotes) {
        return value;
    }
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (const char c : value) {
        if (c == '"') {
            out.push_back('"');
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

std::string format_delimited(const std::vector<std::string>& fields, char delimiter) {
    std::string out;
    for (std::size_t i = 0; i < fields.size(); ++i) {
        if (i != 0) {
            out.push_back(delimiter);
        }
        out += quote_field(fields[i], delimiter);
    }
    out.push_back('\n');
    return out;
}

}  // namespace alphashape::reference
