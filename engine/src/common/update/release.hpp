#pragma once

// GitHub Releases: parsing the API response, choosing the release for a
// channel, and choosing the installer asset for the current platform.
//
// Everything here is pure: the caller supplies the JSON text and, for asset
// selection, a Target it can build from the host or from a test. No global
// process state, no network, no Qt.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common/update/version.hpp"

namespace alphashape::update {

/// The repository the updater talks to. Declared here so the GUI and the tests
/// agree on one value.
inline constexpr const char* kReleasesApiUrl =
    "https://api.github.com/repos/aniketpratapsingh/alphashape/releases";
inline constexpr const char* kRepositoryUrl = "https://github.com/aniketpratapsingh/alphashape";
inline constexpr const char* kChecksumsAssetName = "SHA256SUMS";

/// A downloadable file of a release.
struct Asset {
    std::string name;
    std::string browser_download_url;
    /// The size the API declares, or -1 when the response does not carry one.
    /// The downloader caps the transfer a little above it.
    std::int64_t size = -1;
};

/// One entry of the releases list. `version` is parsed from `tag_name`;
/// releases whose tag is not a semantic version (the rolling `nightly`) are
/// omitted by parse_releases().
struct Release {
    std::string tag_name;
    std::string name;
    std::string body;
    std::string html_url;
    bool draft = false;
    bool prerelease = false;
    Version version;
    std::vector<Asset> assets;

    /// The asset whose name is exactly `name`, or nullptr.
    const Asset* find_asset(std::string_view name) const;
};

/// Parse the GitHub Releases API array. Drafts are kept (the selection skips
/// them) so a test can prove they are ignored; entries without a parseable
/// semantic version are dropped. Throws std::runtime_error on malformed JSON
/// or when the top-level value is not an array.
std::vector<Release> parse_releases(std::string_view json);

enum class Channel { Stable, Prerelease };

struct Selection {
    /// The newest acceptable release, when there is a newer one.
    std::optional<Release> release;
    /// True when there is nothing newer (including when the newest one was
    /// skipped by the user).
    bool up_to_date = true;
    /// True when the only candidate is the version the user asked to skip.
    bool skipped = false;
};

/// Newest release newer than `current` that is allowed by `channel`.
/// Drafts are always ignored, and the rolling `nightly` never survives
/// parsing. `skip` names a version the user chose to skip.
Selection select_release(const std::vector<Release>& releases, const Version& current,
                         Channel channel, const std::optional<Version>& skip = std::nullopt);

enum class Platform { Linux, MacOS, Windows, Other };
enum class Architecture { X86_64, Arm64, Other };

struct Target {
    Platform platform = Platform::Other;
    Architecture arch = Architecture::Other;
    /// True when the Linux build is an AppImage (the APPIMAGE variable is set).
    bool appimage = false;
};

/// The host platform and architecture, and whether it runs as an AppImage.
Target host_target();

/// The installer asset for `target`, or nullptr when this build has none. The
/// names follow publish.yml exactly:
///   alphashape-<tag>-windows-x86_64.exe
///   alphashape-<tag>-macos-arm64.dmg
///   alphashape-<tag>-linux-x86_64.AppImage
///   alphashape-<tag>-linux-amd64.deb
const Asset* choose_asset(const Release& release, const Target& target);

/// A user-facing explanation for choose_asset() returning nullptr.
std::string no_package_reason(const Target& target);

/// The automatic check interval.
inline constexpr std::int64_t kCheckIntervalSeconds = 24 * 60 * 60;

/// True when an automatic check is due. A missing (0) or future timestamp is
/// due; a clock that moved backwards must not suppress an update forever.
bool automatic_check_due(std::int64_t now_unix, std::int64_t last_check_unix,
                         std::int64_t interval_seconds = kCheckIntervalSeconds);

}  // namespace alphashape::update
