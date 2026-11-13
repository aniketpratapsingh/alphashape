#include "common/update/release.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "base/platform/env.hpp"
#include "common/update/json.hpp"

namespace alphashape::update {
namespace {

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

Release parse_release(const json::Value& value) {
    Release release;
    release.tag_name = value.string_or("tag_name");
    release.name = value.string_or("name");
    release.body = value.string_or("body");
    release.html_url = value.string_or("html_url");
    release.draft = value.bool_or("draft", false);
    release.prerelease = value.bool_or("prerelease", false);
    if (const auto version = Version::parse(release.tag_name)) {
        release.version = *version;
    }

    if (const json::Value* assets = value.find("assets");
        assets != nullptr && assets->is_array()) {
        for (const json::Value& entry : assets->array()) {
            if (!entry.is_object()) {
                continue;
            }
            Asset asset;
            asset.name = entry.string_or("name");
            asset.browser_download_url = entry.string_or("browser_download_url");
            // "size" is the byte count GitHub reports for the asset. A missing,
            // non-numeric or non-positive value stays -1 ("unknown").
            if (const json::Value* size = entry.find("size");
                size != nullptr && size->is_number() && size->number() > 0) {
                asset.size = static_cast<std::int64_t>(size->number());
            }
            if (!asset.name.empty() && !asset.browser_download_url.empty()) {
                release.assets.push_back(std::move(asset));
            }
        }
    }
    return release;
}

bool version_allowed(const Release& release, Channel channel) {
    if (release.draft) {
        return false;
    }
    if (channel == Channel::Stable && release.prerelease) {
        return false;
    }
    return true;
}

}  // namespace

const Asset* Release::find_asset(std::string_view asset_name) const {
    for (const Asset& asset : assets) {
        if (asset.name == asset_name) {
            return &asset;
        }
    }
    return nullptr;
}

std::vector<Release> parse_releases(std::string_view json_text) {
    const json::Value root = json::Value::parse(json_text);
    if (!root.is_array()) {
        throw std::runtime_error("the GitHub releases response is not a JSON array");
    }
    std::vector<Release> releases;
    releases.reserve(root.array().size());
    for (const json::Value& entry : root.array()) {
        if (!entry.is_object()) {
            continue;
        }
        Release release = parse_release(entry);
        // A rolling tag such as "nightly" has no semantic version and is not a
        // candidate. It also must not turn into a default Version(0,0,0).
        if (!Version::parse(release.tag_name)) {
            continue;
        }
        releases.push_back(std::move(release));
    }
    return releases;
}

Selection select_release(const std::vector<Release>& releases, const Version& current,
                         Channel channel, const std::optional<Version>& skip) {
    Selection selection;
    const Release* best = nullptr;
    for (const Release& release : releases) {
        if (!version_allowed(release, channel)) {
            continue;
        }
        if (!(release.version > current)) {
            continue;
        }
        if (best == nullptr || release.version > best->version) {
            best = &release;
        }
    }
    if (best == nullptr) {
        return selection;
    }
    if (skip && *skip == best->version) {
        selection.skipped = true;
        return selection;
    }
    selection.release = *best;
    selection.up_to_date = false;
    return selection;
}

Target host_target() {
    Target target;
#if defined(__APPLE__)
    target.platform = Platform::MacOS;
#elif defined(_WIN32)
    target.platform = Platform::Windows;
#elif defined(__linux__)
    target.platform = Platform::Linux;
#endif
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm64__)
    target.arch = Architecture::Arm64;
#elif defined(__x86_64__) || defined(_M_X64)
    target.arch = Architecture::X86_64;
#endif
#if defined(__linux__)
    target.appimage = !platform::get_env("APPIMAGE").empty();
#endif
    return target;
}

const Asset* choose_asset(const Release& release, const Target& target) {
    auto pick = [&release](std::string_view suffix) -> const Asset* {
        for (const Asset& asset : release.assets) {
            if (ends_with(asset.name, suffix)) {
                return &asset;
            }
        }
        return nullptr;
    };
    switch (target.platform) {
        case Platform::Windows:
            return target.arch == Architecture::X86_64 ? pick("-windows-x86_64.exe") : nullptr;
        case Platform::MacOS:
            return target.arch == Architecture::Arm64 ? pick("-macos-arm64.dmg") : nullptr;
        case Platform::Linux:
            if (target.arch != Architecture::X86_64) {
                return nullptr;
            }
            return target.appimage ? pick("-linux-x86_64.AppImage") : pick("-linux-amd64.deb");
        case Platform::Other:
            return nullptr;
    }
    return nullptr;
}

std::string no_package_reason(const Target& target) {
    switch (target.platform) {
        case Platform::MacOS:
            if (target.arch == Architecture::X86_64) {
                return "no update package for this platform: alphashape publishes an "
                       "Apple-silicon (arm64) macOS build only";
            }
            return "no macOS arm64 package is attached to this release";
        case Platform::Windows:
            return "no Windows x86-64 package is attached to this release";
        case Platform::Linux:
            if (target.arch != Architecture::X86_64) {
                return "no update package for this platform: the Linux builds are x86-64 only";
            }
            return target.appimage ? "no AppImage is attached to this release"
                                   : "no .deb package is attached to this release";
        case Platform::Other:
            return "no update package for this platform";
    }
    return "no update package for this platform";
}

bool automatic_check_due(std::int64_t now_unix, std::int64_t last_check_unix,
                         std::int64_t interval_seconds) {
    if (last_check_unix <= 0) {
        return true;
    }
    if (now_unix < last_check_unix) {
        return true;
    }
    return now_unix - last_check_unix >= interval_seconds;
}

}  // namespace alphashape::update
