#include "common/update/version.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <system_error>

namespace alphashape::update {
namespace {

bool all_digits(std::string_view text) {
    return !text.empty() &&
           std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; });
}

std::optional<std::uint64_t> parse_number(std::string_view text) {
    if (!all_digits(text)) {
        return std::nullopt;
    }
    std::uint64_t value = 0;
    const auto* begin = text.data();
    const auto* end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc() || result.ptr != end) {
        return std::nullopt;
    }
    return value;
}

std::vector<std::string_view> split(std::string_view text, char separator) {
    std::vector<std::string_view> parts;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = text.find(separator, start);
        if (end == std::string_view::npos) {
            parts.push_back(text.substr(start));
            break;
        }
        parts.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    return parts;
}

}  // namespace

PrereleaseIdentifier PrereleaseIdentifier::parse(std::string_view token) {
    PrereleaseIdentifier identifier;
    identifier.text = std::string(token);
    if (const auto number = parse_number(token)) {
        identifier.numeric = true;
        identifier.number = *number;
    }
    return identifier;
}

std::strong_ordering PrereleaseIdentifier::compare(const PrereleaseIdentifier& other) const {
    if (numeric != other.numeric) {
        // Numeric identifiers always have lower precedence than alphanumeric ones.
        return numeric ? std::strong_ordering::less : std::strong_ordering::greater;
    }
    if (numeric) {
        return number <=> other.number;
    }
    const int order = text.compare(other.text);
    if (order < 0) {
        return std::strong_ordering::less;
    }
    if (order > 0) {
        return std::strong_ordering::greater;
    }
    return std::strong_ordering::equal;
}

bool PrereleaseIdentifier::operator==(const PrereleaseIdentifier& other) const {
    return compare(other) == 0;
}

std::optional<Version> Version::parse(std::string_view text) {
    if (!text.empty() && (text.front() == 'v' || text.front() == 'V')) {
        text.remove_prefix(1);
    }
    // Build metadata never participates in ordering.
    if (const std::size_t plus = text.find('+'); plus != std::string_view::npos) {
        text = text.substr(0, plus);
    }
    std::string_view prerelease;
    if (const std::size_t dash = text.find('-'); dash != std::string_view::npos) {
        prerelease = text.substr(dash + 1);
        text = text.substr(0, dash);
        if (prerelease.empty()) {
            return std::nullopt;
        }
    }

    const auto parts = split(text, '.');
    if (parts.size() != 3) {
        return std::nullopt;
    }
    std::uint64_t numbers[3] = {0, 0, 0};
    for (std::size_t i = 0; i < 3; ++i) {
        const auto value = parse_number(parts[i]);
        if (!value || *value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            return std::nullopt;
        }
        numbers[i] = *value;
    }

    Version version(static_cast<int>(numbers[0]), static_cast<int>(numbers[1]),
                    static_cast<int>(numbers[2]));
    if (!prerelease.empty()) {
        for (const std::string_view token : split(prerelease, '.')) {
            if (token.empty()) {
                return std::nullopt;
            }
            version.prerelease_.push_back(PrereleaseIdentifier::parse(token));
        }
    }
    return version;
}

std::string Version::to_string() const {
    std::string text = std::to_string(major_) + "." + std::to_string(minor_) + "." +
                       std::to_string(patch_);
    if (!prerelease_.empty()) {
        text += '-';
        for (std::size_t i = 0; i < prerelease_.size(); ++i) {
            if (i != 0) {
                text += '.';
            }
            text += prerelease_[i].text;
        }
    }
    return text;
}

std::strong_ordering Version::compare(const Version& other) const {
    if (major_ != other.major_) {
        return major_ <=> other.major_;
    }
    if (minor_ != other.minor_) {
        return minor_ <=> other.minor_;
    }
    if (patch_ != other.patch_) {
        return patch_ <=> other.patch_;
    }
    // A version without a pre-release outranks one with it.
    if (prerelease_.empty() != other.prerelease_.empty()) {
        return prerelease_.empty() ? std::strong_ordering::greater
                                   : std::strong_ordering::less;
    }
    const std::size_t count = std::min(prerelease_.size(), other.prerelease_.size());
    for (std::size_t i = 0; i < count; ++i) {
        if (const auto order = prerelease_[i].compare(other.prerelease_[i]);
            order != std::strong_ordering::equal) {
            return order;
        }
    }
    return prerelease_.size() <=> other.prerelease_.size();
}

bool Version::operator==(const Version& other) const {
    return compare(other) == std::strong_ordering::equal;
}

}  // namespace alphashape::update
