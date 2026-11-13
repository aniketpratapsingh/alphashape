#pragma once

// Semantic versions and their ordering (https://semver.org/spec/v2.0.0.html).
//
// The updater compares a release tag such as "v0.3.0-rc.1" with the running
// build. Pre-release identifiers are ordered exactly as the specification says:
// numeric identifiers compare numerically, alphanumerics compare in ASCII
// order, numeric is lower than alphanumeric, and a version with a pre-release
// is lower than the same version without one. Build metadata ("+sha") is
// accepted and ignored for ordering, as the specification requires.
//
// Qt-free: this is part of the engine's common layer and is unit-tested with
// canned GitHub release JSON.

#include <compare>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace alphashape::update {

/// One dotted component of a pre-release version.
struct PrereleaseIdentifier {
    bool numeric = false;
    std::uint64_t number = 0;
    std::string text;  ///< the identifier as written (both forms keep it)

    static PrereleaseIdentifier parse(std::string_view token);
    std::strong_ordering compare(const PrereleaseIdentifier& other) const;
    bool operator==(const PrereleaseIdentifier& other) const;
};

class Version {
public:
    Version() = default;
    Version(int major, int minor, int patch) : major_(major), minor_(minor), patch_(patch) {}

    /// Parse "1.2.3", "1.2.3-rc.1", or the same with a leading "v". Build
    /// metadata after "+" is accepted and ignored. Returns nullopt when the
    /// text is not a semantic version.
    static std::optional<Version> parse(std::string_view text);

    int major() const { return major_; }
    int minor() const { return minor_; }
    int patch() const { return patch_; }
    bool is_prerelease() const { return !prerelease_.empty(); }
    const std::vector<PrereleaseIdentifier>& prerelease() const { return prerelease_; }

    /// The canonical spelling, without the leading "v" or build metadata.
    std::string to_string() const;

    std::strong_ordering compare(const Version& other) const;
    bool operator<(const Version& other) const { return compare(other) < 0; }
    bool operator>(const Version& other) const { return compare(other) > 0; }
    bool operator<=(const Version& other) const { return compare(other) <= 0; }
    bool operator>=(const Version& other) const { return compare(other) >= 0; }
    bool operator==(const Version& other) const;
    bool operator!=(const Version& other) const { return !(*this == other); }

private:
    int major_ = 0;
    int minor_ = 0;
    int patch_ = 0;
    std::vector<PrereleaseIdentifier> prerelease_;
};

}  // namespace alphashape::update
