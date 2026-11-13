// Update.*: semantic versions, GitHub Releases parsing, channel and asset
// selection, checksum verification, throttling and skip-version. No network and
// no QApplication: the policy is the engine's common layer.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "common/update/checksum.hpp"
#include "common/update/json.hpp"
#include "common/update/release.hpp"
#include "common/update/version.hpp"

using namespace alphashape::update;

namespace {

Version v(const char* text) {
    const auto parsed = Version::parse(text);
    EXPECT_TRUE(parsed.has_value()) << text;
    return parsed.value_or(Version());
}

const char* kReleasesJson = R"json([
  {
    "tag_name": "v0.2.0",
    "name": "0.2.0",
    "body": "The stable release",
    "html_url": "https://github.com/aniketpratapsingh/alphashape/releases/tag/v0.2.0",
    "draft": false,
    "prerelease": false,
    "assets": [
      {"name": "alphashape-v0.2.0-macos-arm64.dmg",
       "browser_download_url": "https://github.com/aniketpratapsingh/alphashape/releases/download/v0.2.0/alphashape-v0.2.0-macos-arm64.dmg"},
      {"name": "alphashape-v0.2.0-linux-x86_64.AppImage",
       "browser_download_url": "https://github.com/aniketpratapsingh/alphashape/releases/download/v0.2.0/alphashape-v0.2.0-linux-x86_64.AppImage"},
      {"name": "alphashape-v0.2.0-linux-amd64.deb",
       "browser_download_url": "https://github.com/aniketpratapsingh/alphashape/releases/download/v0.2.0/alphashape-v0.2.0-linux-amd64.deb"},
      {"name": "alphashape-v0.2.0-windows-x86_64.exe",
       "browser_download_url": "https://github.com/aniketpratapsingh/alphashape/releases/download/v0.2.0/alphashape-v0.2.0-windows-x86_64.exe"},
      {"name": "SHA256SUMS",
       "browser_download_url": "https://github.com/aniketpratapsingh/alphashape/releases/download/v0.2.0/SHA256SUMS"}
    ]
  },
  {
    "tag_name": "v0.2.1",
    "name": "0.2.1",
    "body": "A patch",
    "html_url": "https://github.com/aniketpratapsingh/alphashape/releases/tag/v0.2.1",
    "draft": false,
    "prerelease": false,
    "assets": []
  },
  {
    "tag_name": "v0.3.0-rc.1",
    "name": "0.3.0 rc 1",
    "body": "**Release candidate**",
    "html_url": "https://github.com/aniketpratapsingh/alphashape/releases/tag/v0.3.0-rc.1",
    "draft": false,
    "prerelease": true,
    "assets": []
  },
  {
    "tag_name": "v0.4.0",
    "name": "Draft 0.4.0",
    "body": "not published",
    "html_url": "https://github.com/aniketpratapsingh/alphashape/releases/tag/v0.4.0",
    "draft": true,
    "prerelease": false,
    "assets": []
  },
  {
    "tag_name": "nightly",
    "name": "Nightly",
    "body": "rolling",
    "html_url": "https://github.com/aniketpratapsingh/alphashape/releases/tag/nightly",
    "draft": false,
    "prerelease": false,
    "assets": []
  }
])json";

Release release_with_assets() {
    Release release;
    release.tag_name = "v0.3.0";
    release.version = v("0.3.0");
    release.assets = {
        {"alphashape-v0.3.0-macos-arm64.dmg", "https://github.com/a.dmg"},
        {"alphashape-v0.3.0-windows-x86_64.exe", "https://github.com/a.exe"},
        {"alphashape-v0.3.0-linux-x86_64.AppImage", "https://github.com/a.AppImage"},
        {"alphashape-v0.3.0-linux-amd64.deb", "https://github.com/a.deb"},
        {"SHA256SUMS", "https://github.com/SHA256SUMS"},
    };
    return release;
}

}  // namespace

TEST(Update, VersionParsesAndOrdersSemantically) {
    EXPECT_EQ(v("0.2.0").to_string(), "0.2.0");
    EXPECT_EQ(v("v0.2.0").to_string(), "0.2.0");
    EXPECT_EQ(v("1.2.3-rc.1").to_string(), "1.2.3-rc.1");
    EXPECT_FALSE(v("1.2.3").is_prerelease());
    EXPECT_TRUE(v("1.2.3-rc.1").is_prerelease());

    EXPECT_LT(v("0.2.0"), v("0.3.0"));
    EXPECT_LT(v("0.2.0"), v("0.2.1"));
    EXPECT_LT(v("0.9.9"), v("1.0.0"));
    EXPECT_EQ(v("1.0.0+build.7"), v("1.0.0"));
}

TEST(Update, PrereleasePrecedenceFollowsTheSpecification) {
    // The specification's own ordering example.
    const std::vector<Version> ordered = {
        v("1.0.0-alpha"), v("1.0.0-alpha.1"), v("1.0.0-alpha.beta"), v("1.0.0-beta"),
        v("1.0.0-beta.2"), v("1.0.0-beta.11"), v("1.0.0-rc.1"), v("1.0.0"),
    };
    for (std::size_t i = 0; i + 1 < ordered.size(); ++i) {
        EXPECT_LT(ordered[i], ordered[i + 1]) << ordered[i].to_string() << " vs "
                                               << ordered[i + 1].to_string();
    }
    EXPECT_LT(v("0.3.0-rc.1"), v("0.3.0"));
    EXPECT_LT(v("0.3.0-rc.1"), v("0.3.0-rc.2"));
    EXPECT_GT(v("0.3.0-rc.10"), v("0.3.0-rc.2"));
    // Numeric identifiers sort below alphanumeric ones.
    EXPECT_LT(v("1.0.0-1"), v("1.0.0-alpha"));
}

TEST(Update, InvalidVersionsAreRejected) {
    EXPECT_FALSE(Version::parse("").has_value());
    EXPECT_FALSE(Version::parse("1.2").has_value());
    EXPECT_FALSE(Version::parse("1.2.3.4").has_value());
    EXPECT_FALSE(Version::parse("1.2.x").has_value());
    EXPECT_FALSE(Version::parse("1.2.3-").has_value());
    EXPECT_FALSE(Version::parse("nightly").has_value());
}

TEST(Update, ParsesReleasesAndDropsUnversionedOnes) {
    const std::vector<Release> releases = parse_releases(kReleasesJson);
    ASSERT_EQ(releases.size(), 4u);  // nightly has no semantic version
    EXPECT_EQ(releases.front().tag_name, "v0.2.0");
    EXPECT_EQ(releases.front().body, "The stable release");
    EXPECT_EQ(releases.front().assets.size(), 5u);
    EXPECT_NE(releases.front().find_asset("SHA256SUMS"), nullptr);
    EXPECT_EQ(releases.front().find_asset("missing"), nullptr);
}

TEST(Update, RejectsMalformedJson) {
    EXPECT_THROW(parse_releases("{not an array}"), std::runtime_error);
    EXPECT_THROW(parse_releases("[{\"tag_name\":}]"), std::runtime_error);
}

TEST(Update, ChannelFilteringPicksTheNewestAllowedRelease) {
    const std::vector<Release> releases = parse_releases(kReleasesJson);

    const Selection stable = select_release(releases, v("0.2.0"), Channel::Stable);
    ASSERT_TRUE(stable.release.has_value());
    EXPECT_EQ(stable.release->tag_name, "v0.2.1");
    EXPECT_FALSE(stable.up_to_date);

    const Selection pre = select_release(releases, v("0.2.0"), Channel::Prerelease);
    ASSERT_TRUE(pre.release.has_value());
    EXPECT_EQ(pre.release->tag_name, "v0.3.0-rc.1");

    // A release candidate is newer than 0.2.1, so nothing older is selected.
    const Selection after_rc =
        select_release(releases, v("0.3.0-rc.1"), Channel::Prerelease);
    EXPECT_TRUE(after_rc.up_to_date);
    EXPECT_FALSE(after_rc.release.has_value());
}

TEST(Update, AlreadyUpToDateWhenNothingIsNewer) {
    const std::vector<Release> releases = parse_releases(kReleasesJson);
    const Selection selection = select_release(releases, v("0.2.1"), Channel::Stable);
    EXPECT_FALSE(selection.release.has_value());
    EXPECT_TRUE(selection.up_to_date);
    EXPECT_FALSE(selection.skipped);

    const Selection current = select_release(releases, v("0.3.0"), Channel::Prerelease);
    EXPECT_TRUE(current.up_to_date);
}

TEST(Update, SkipVersionSuppressesThatRelease) {
    const std::vector<Release> releases = parse_releases(kReleasesJson);
    const Selection skipped =
        select_release(releases, v("0.2.0"), Channel::Stable, v("0.2.1"));
    EXPECT_FALSE(skipped.release.has_value());
    EXPECT_TRUE(skipped.up_to_date);
    EXPECT_TRUE(skipped.skipped);

    // Skipping an older version does not suppress the newest one.
    const Selection other =
        select_release(releases, v("0.2.0"), Channel::Stable, v("0.1.0"));
    ASSERT_TRUE(other.release.has_value());
    EXPECT_EQ(other.release->tag_name, "v0.2.1");
}

TEST(Update, AssetChoiceMatchesEachPlatform) {
    const Release release = release_with_assets();

    Target linux_appimage{Platform::Linux, Architecture::X86_64, true};
    ASSERT_NE(choose_asset(release, linux_appimage), nullptr);
    EXPECT_EQ(choose_asset(release, linux_appimage)->name,
              "alphashape-v0.3.0-linux-x86_64.AppImage");

    Target linux_deb{Platform::Linux, Architecture::X86_64, false};
    ASSERT_NE(choose_asset(release, linux_deb), nullptr);
    EXPECT_EQ(choose_asset(release, linux_deb)->name, "alphashape-v0.3.0-linux-amd64.deb");

    Target macos{Platform::MacOS, Architecture::Arm64, false};
    ASSERT_NE(choose_asset(release, macos), nullptr);
    EXPECT_EQ(choose_asset(release, macos)->name, "alphashape-v0.3.0-macos-arm64.dmg");

    Target windows{Platform::Windows, Architecture::X86_64, false};
    ASSERT_NE(choose_asset(release, windows), nullptr);
    EXPECT_EQ(choose_asset(release, windows)->name,
              "alphashape-v0.3.0-windows-x86_64.exe");

    // An Intel Mac has no published build; the reason is user-facing.
    Target intel_mac{Platform::MacOS, Architecture::X86_64, false};
    EXPECT_EQ(choose_asset(release, intel_mac), nullptr);
    EXPECT_NE(no_package_reason(intel_mac).find("Apple-silicon"), std::string::npos);

    Target linux_arm{Platform::Linux, Architecture::Arm64, false};
    EXPECT_EQ(choose_asset(release, linux_arm), nullptr);
    EXPECT_NE(no_package_reason(linux_arm).find("x86-64"), std::string::npos);

    Target other{Platform::Other, Architecture::Other, false};
    EXPECT_EQ(choose_asset(release, other), nullptr);
}

TEST(Update, Sha256MatchesKnownVectors) {
    EXPECT_EQ(sha256_hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(sha256_hex("abc"),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(sha256_hex("hello"),
              "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
}

TEST(Update, ParsesSha256SumsFormat) {
    const std::string text =
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824  hello.txt\n"
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 *empty\n"
        "\n"
        "not a checksum line\n";
    const std::vector<ChecksumEntry> entries = parse_sha256sums(text);
    ASSERT_EQ(entries.size(), 2u);
    EXPECT_EQ(entries[0].sha256,
              "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
    EXPECT_EQ(entries[0].name, "hello.txt");
    EXPECT_EQ(entries[1].name, "empty");
    EXPECT_NE(find_checksum(entries, "hello.txt"), nullptr);
    EXPECT_EQ(find_checksum(entries, "absent"), nullptr);

    EXPECT_TRUE(checksum_matches(entries[0].sha256,
                                 "2CF24DBA5FB0A30E26E83B2AC5B9E29E1B161E5C1FA7425E73043362938B9824"));
    EXPECT_FALSE(checksum_matches(entries[0].sha256, entries[1].sha256));
}

TEST(Update, VerifiesAndRejectsDownloadedFiles) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("alphashape_update_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(&path)) +
         ".bin");
    {
        std::ofstream file(path, std::ios::binary);
        file << "hello";
    }

    const std::string good =
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824  payload.bin\n";
    const VerifyResult ok = verify_file_sha256(path, "payload.bin", good);
    EXPECT_EQ(ok.status, VerifyStatus::Ok);
    EXPECT_TRUE(ok.error.empty());
    EXPECT_EQ(ok.actual, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");

    const std::string wrong =
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  payload.bin\n";
    const VerifyResult mismatch = verify_file_sha256(path, "payload.bin", wrong);
    EXPECT_EQ(mismatch.status, VerifyStatus::Mismatch);
    EXPECT_FALSE(mismatch.error.empty());

    const VerifyResult missing = verify_file_sha256(path, "other.bin", good);
    EXPECT_EQ(missing.status, VerifyStatus::MissingEntry);
    EXPECT_FALSE(missing.error.empty());

    const VerifyResult unreadable =
        verify_file_sha256(path.string() + ".does-not-exist", "payload.bin", good);
    EXPECT_EQ(unreadable.status, VerifyStatus::ReadError);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(Update, ThrottlesAutomaticChecksToTwentyFourHours) {
    constexpr std::int64_t now = 1'800'000'000;
    EXPECT_TRUE(automatic_check_due(now, 0));
    EXPECT_FALSE(automatic_check_due(now, now - 60));
    EXPECT_FALSE(automatic_check_due(now, now - (kCheckIntervalSeconds - 1)));
    EXPECT_TRUE(automatic_check_due(now, now - kCheckIntervalSeconds));
    EXPECT_TRUE(automatic_check_due(now, now - kCheckIntervalSeconds - 1));
    // A clock that jumped backwards must not suppress checks forever.
    EXPECT_TRUE(automatic_check_due(now, now + 120));
}

TEST(Update, JsonReaderHandlesEscapesAndUnicode) {
    const json::Value value = json::Value::parse(
        R"({"text":"line\nnext \u00e9 \ud83d\ude00","n":1.5,"ok":true,"list":[1,2]})");
    ASSERT_TRUE(value.is_object());
    EXPECT_EQ(value.string_or("text"), "line\nnext \xc3\xa9 \xf0\x9f\x98\x80");
    ASSERT_NE(value.find("n"), nullptr);
    EXPECT_DOUBLE_EQ(value.find("n")->number(), 1.5);
    EXPECT_TRUE(value.bool_or("ok", false));
    ASSERT_NE(value.find("list"), nullptr);
    ASSERT_TRUE(value.find("list")->is_array());
    EXPECT_EQ(value.find("list")->array().size(), 2u);
    EXPECT_FALSE(value.bool_or("missing", false));
    EXPECT_TRUE(value.bool_or("missing", true));
    EXPECT_EQ(value.string_or("missing"), "");
}
