// Update.*: the update action wiring, the update dialog and the preferences
// page, driven through the offscreen harness. Every network call goes through a
// fake UpdateSource; the process also sets ALPHASHAPE_NO_UPDATE_CHECK=1 so the
// silent startup check cannot reach GitHub even in a test that never installs a
// fake.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <QAction>
#include <QByteArray>
#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QTextBrowser>
#include <QtGlobal>
#include <QtTest/QTest>

#include "apps/gui/preferences_dialog.hpp"
#include "apps/gui/update_checker.hpp"
#include "apps/gui/update_dialog.hpp"
#include "apps/gui/update_source.hpp"
#include "common/update/checksum.hpp"
#include "common/update/release.hpp"
#include "harness.hpp"

using namespace alphashape;
using namespace alphashape::gui;
using alphashape::gui::testing::GuiTest;
using alphashape::gui::testing::pump_until;

namespace {

/// A canned UpdateSource. The releases JSON is returned for the API URL and
/// `sums_text` for SHA256SUMS; download_file writes `download_bytes`.
class FakeSource : public UpdateSource {
public:
    std::string releases_json;
    std::string sums_text;
    std::string fail_error;
    bool fail_text = false;
    QByteArray download_bytes;
    int api_calls = 0;

    void fetch_text(const std::string& url, TextCallback done) override {
        if (url.find("SHA256SUMS") != std::string::npos) {
            done(true, sums_text, std::string());
            return;
        }
        ++api_calls;
        if (fail_text) {
            done(false, std::string(), fail_error);
            return;
        }
        done(true, releases_json, std::string());
    }

    void download_file(const std::string&, const QString& dest, std::int64_t /*expected_size*/,
                       ProgressCallback progress, FileCallback done) override {
        QFile file(dest);
        if (!file.open(QIODevice::WriteOnly)) {
            done(false, "cannot write");
            return;
        }
        file.write(download_bytes);
        file.close();
        if (progress) {
            progress(download_bytes.size(), download_bytes.size());
        }
        done(true, std::string());
    }

    void cancel() override { ++cancel_calls; }

    int cancel_calls = 0;
};

/// A source whose download stays in flight until the test completes it, so a
/// test can destroy the dialog between the request and its completion.
class SlowSource : public FakeSource {
public:
    void download_file(const std::string&, const QString&, std::int64_t /*expected_size*/,
                       ProgressCallback progress, FileCallback done) override {
        pending_progress = std::move(progress);
        pending_done = std::move(done);
    }

    bool has_pending() const { return static_cast<bool>(pending_done); }

    void complete(bool ok, const std::string& error) {
        FileCallback done = std::move(pending_done);
        pending_done = nullptr;
        if (done) {
            done(ok, error);
        }
    }

    ProgressCallback pending_progress;
    FileCallback pending_done;
};

const char* kNewReleaseJson = R"json([
  {
    "tag_name": "v0.3.0",
    "name": "0.3.0",
    "body": "## New things\n\n- **Faster** alignment\n- Fixed seams",
    "html_url": "https://github.com/aniketpratapsingh/alphashape/releases/tag/v0.3.0",
    "draft": false,
    "prerelease": false,
    "assets": [
      {"name": "alphashape-v0.3.0-macos-arm64.dmg",
       "browser_download_url": "https://github.com/aniketpratapsingh/alphashape/releases/download/v0.3.0/alphashape-v0.3.0-macos-arm64.dmg"},
      {"name": "alphashape-v0.3.0-windows-x86_64.exe",
       "browser_download_url": "https://github.com/aniketpratapsingh/alphashape/releases/download/v0.3.0/alphashape-v0.3.0-windows-x86_64.exe"},
      {"name": "alphashape-v0.3.0-linux-x86_64.AppImage",
       "browser_download_url": "https://github.com/aniketpratapsingh/alphashape/releases/download/v0.3.0/alphashape-v0.3.0-linux-x86_64.AppImage"},
      {"name": "alphashape-v0.3.0-linux-amd64.deb",
       "browser_download_url": "https://github.com/aniketpratapsingh/alphashape/releases/download/v0.3.0/alphashape-v0.3.0-linux-amd64.deb"},
      {"name": "SHA256SUMS",
       "browser_download_url": "https://github.com/aniketpratapsingh/alphashape/releases/download/v0.3.0/SHA256SUMS"}
    ]
  }
])json";

update::Release release_from_json() {
    const auto releases = update::parse_releases(kNewReleaseJson);
    EXPECT_EQ(releases.size(), 1u);
    return releases.front();
}

/// A dialog-ready result: Available with the 0.3.0 release.
CheckResult available_result() {
    CheckResult result;
    result.status = CheckResult::Status::Available;
    result.release = release_from_json();
    return result;
}

class UpdateGuiTest : public GuiTest {
protected:
    void SetUp() override {
        // Belt and braces: the QApplication-wide setting is in test_main.cpp,
        // and every check in this fixture also sets it again so a previous test
        // cannot leak an unset variable into the silent startup timer.
        qputenv("ALPHASHAPE_NO_UPDATE_CHECK", "1");
        GuiTest::SetUp();
    }
    void TearDown() override {
        qputenv("ALPHASHAPE_NO_UPDATE_CHECK", "1");
        GuiTest::TearDown();
    }
    void allow_checks() { qunsetenv("ALPHASHAPE_NO_UPDATE_CHECK"); }
};

}  // namespace

TEST_F(UpdateGuiTest, UpdateActionExistsAndPreferencesIsEnabled) {
    QAction* check_action = action("action.help.check_updates");
    ASSERT_NE(check_action, nullptr);
    EXPECT_EQ(check_action->menuRole(), QAction::ApplicationSpecificRole);
    EXPECT_TRUE(check_action->isEnabled());
    EXPECT_GE(menu("menu.help")->actions().indexOf(check_action), 0);

    // Preferences stopped being a disabled placeholder when update settings
    // landed in it.
    EXPECT_TRUE(action("action.edit.preferences")->isEnabled());
}

TEST_F(UpdateGuiTest, ManualCheckReportsANewerVersion) {
    allow_checks();
    auto fake = std::make_unique<FakeSource>();
    fake->releases_json = kNewReleaseJson;
    window().set_update_source(std::move(fake));

    CheckResult seen;
    bool called = false;
    bool automatic = true;
    window().set_update_result_handler([&](const CheckResult& result, bool is_automatic) {
        seen = result;
        automatic = is_automatic;
        called = true;
    });

    trigger("action.help.check_updates");
    EXPECT_TRUE(pump_until([&]() { return called; }));
    EXPECT_FALSE(automatic);
    EXPECT_EQ(seen.status, CheckResult::Status::Available);
    EXPECT_EQ(seen.release.version.to_string(), "0.3.0");
}

TEST_F(UpdateGuiTest, ManualCheckBypassesTheThrottle) {
    allow_checks();
    auto fake_raw = new FakeSource();
    fake_raw->releases_json = kNewReleaseJson;
    window().set_update_source(std::unique_ptr<UpdateSource>(fake_raw));

    int results = 0;
    window().set_update_result_handler([&](const CheckResult&, bool) { ++results; });

    // Pretend the automatic check just ran. A manual check must still run.
    window().update_checker().record_check(QDateTime::currentSecsSinceEpoch());
    EXPECT_FALSE(window().update_checker().automatic_check_due_now());

    trigger("action.help.check_updates");
    EXPECT_TRUE(pump_until([&]() { return results == 1; }));
    trigger("action.help.check_updates");
    EXPECT_TRUE(pump_until([&]() { return results == 2; }));
    EXPECT_EQ(fake_raw->api_calls, 2);
}

TEST_F(UpdateGuiTest, AlreadyUpToDateIsReported) {
    allow_checks();
    auto fake = std::make_unique<FakeSource>();
    fake->releases_json = R"json([{"tag_name":"v0.1.0","name":"old","body":"",
      "html_url":"https://github.com/aniketpratapsingh/alphashape/releases/tag/v0.1.0",
      "draft":false,"prerelease":false,"assets":[]}])json";
    window().set_update_source(std::move(fake));

    CheckResult seen;
    window().set_update_result_handler(
        [&](const CheckResult& result, bool) { seen = result; });
    trigger("action.help.check_updates");
    // The fake is synchronous; the callback has already run.
    EXPECT_EQ(seen.status, CheckResult::Status::UpToDate);
}

TEST_F(UpdateGuiTest, NetworkFailureIsReportedAsAnError) {
    allow_checks();
    auto fake = std::make_unique<FakeSource>();
    fake->fail_text = true;
    fake->fail_error = "connection refused";
    window().set_update_source(std::move(fake));

    CheckResult seen;
    window().set_update_result_handler(
        [&](const CheckResult& result, bool) { seen = result; });
    trigger("action.help.check_updates");
    EXPECT_EQ(seen.status, CheckResult::Status::Error);
    EXPECT_NE(seen.error.find("connection refused"), std::string::npos);
}

TEST_F(UpdateGuiTest, SkippedVersionSuppressesTheSameRelease) {
    allow_checks();
    auto fake = std::make_unique<FakeSource>();
    fake->releases_json = kNewReleaseJson;
    window().set_update_source(std::move(fake));
    window().update_checker().skip_version(update::Version::parse("0.3.0").value());

    CheckResult seen;
    window().set_update_result_handler(
        [&](const CheckResult& result, bool) { seen = result; });
    trigger("action.help.check_updates");
    EXPECT_EQ(seen.status, CheckResult::Status::UpToDate);
    EXPECT_TRUE(seen.skipped);
}

TEST_F(UpdateGuiTest, OptOutBlocksEvenManualChecks) {
    qputenv("ALPHASHAPE_NO_UPDATE_CHECK", "1");
    auto fake_raw = new FakeSource();
    fake_raw->releases_json = kNewReleaseJson;
    window().set_update_source(std::unique_ptr<UpdateSource>(fake_raw));

    CheckResult seen;
    window().set_update_result_handler(
        [&](const CheckResult& result, bool) { seen = result; });
    trigger("action.help.check_updates");
    EXPECT_EQ(seen.status, CheckResult::Status::Error);
    EXPECT_NE(seen.error.find("disabled"), std::string::npos);
    EXPECT_EQ(fake_raw->api_calls, 0);
}

TEST_F(UpdateGuiTest, UpdateDialogShowsVersionNotesAndButtons) {
    QSettings settings(scratch("dialog-settings.ini"), QSettings::IniFormat);
    UpdateChecker checker(settings, std::make_unique<FakeSource>());
    UpdateDialog dialog(available_result(), checker);

    ASSERT_NE(dialog.notes(), nullptr);
    EXPECT_GE(dialog.notes()->toPlainText().indexOf(QStringLiteral("New things")), 0);
    ASSERT_NE(dialog.version_label(), nullptr);
    EXPECT_GE(dialog.version_label()->text().indexOf(QStringLiteral("0.3.0")), 0);
    ASSERT_NE(dialog.download_button(), nullptr);
    EXPECT_EQ(dialog.download_button()->text(), QStringLiteral("Download and Install"));
    ASSERT_NE(dialog.skip_button(), nullptr);
    ASSERT_NE(dialog.later_button(), nullptr);
    ASSERT_NE(dialog.view_button(), nullptr);
    // The dialog itself is not shown, so isVisible() is false for every child;
    // isHidden() says whether the widget was explicitly hidden.
    EXPECT_TRUE(dialog.progress()->isHidden());
    EXPECT_TRUE(dialog.cancel_button()->isHidden());
}

TEST_F(UpdateGuiTest, SkippingFromTheDialogPersistsTheVersion) {
    QSettings settings(scratch("skip-settings.ini"), QSettings::IniFormat);
    UpdateChecker checker(settings, std::make_unique<FakeSource>());
    UpdateDialog dialog(available_result(), checker);

    dialog.skip_button()->click();
    EXPECT_EQ(dialog.result(), QDialog::Accepted);
    ASSERT_TRUE(checker.skipped_version().has_value());
    EXPECT_EQ(checker.skipped_version()->to_string(), "0.3.0");
    EXPECT_EQ(settings.value("updates/skipped_version").toString().toStdString(), "0.3.0");
}

TEST_F(UpdateGuiTest, RemindMeLaterClosesWithoutSkipping) {
    QSettings settings(scratch("later-settings.ini"), QSettings::IniFormat);
    UpdateChecker checker(settings, std::make_unique<FakeSource>());
    UpdateDialog dialog(available_result(), checker);

    dialog.later_button()->click();
    EXPECT_EQ(dialog.result(), QDialog::Rejected);
    EXPECT_FALSE(checker.skipped_version().has_value());
}

TEST_F(UpdateGuiTest, ViewOnGitHubUsesTheReleaseUrl) {
    QSettings settings(scratch("url-settings.ini"), QSettings::IniFormat);
    UpdateChecker checker(settings, std::make_unique<FakeSource>());
    UpdateDialog dialog(available_result(), checker);

    QString opened;
    dialog.set_open_url_handler([&](const QString& url) {
        opened = url;
        return true;
    });
    dialog.view_button()->click();
    EXPECT_EQ(opened, QStringLiteral(
                          "https://github.com/aniketpratapsingh/alphashape/releases/tag/v0.3.0"));
}

TEST_F(UpdateGuiTest, DownloadWithoutAPlatformAssetReportsTheReason) {
    QSettings settings(scratch("noasset-settings.ini"), QSettings::IniFormat);
    UpdateChecker checker(settings, std::make_unique<FakeSource>());
    CheckResult result;
    result.status = CheckResult::Status::Available;
    result.release = release_from_json();
    result.release.assets.clear();
    UpdateDialog dialog(result, checker);

    dialog.download_button()->click();
    EXPECT_FALSE(dialog.status_label()->isHidden());
    EXPECT_FALSE(dialog.status_label()->text().isEmpty());
}

TEST_F(UpdateGuiTest, DownloadVerifiesTheChecksum) {
    QSettings settings(scratch("verify-settings.ini"), QSettings::IniFormat);
    auto fake = std::make_unique<FakeSource>();
    fake->download_bytes = QByteArrayLiteral("installer payload");
    fake->sums_text =
        QCryptographicHash::hash(fake->download_bytes, QCryptographicHash::Sha256)
            .toHex()
            .toStdString() +
        "  payload.bin\n";
    UpdateChecker checker(settings, std::move(fake));

    const auto release = release_from_json();
    update::Asset asset{"payload.bin",
                        "https://github.com/aniketpratapsingh/alphashape/releases/download/"
                        "v0.3.0/payload.bin"};
    DownloadResult seen;
    bool called = false;
    checker.download_and_verify(release, asset, scratch("cache"), {},
                                [&](const DownloadResult& result) {
                                    seen = result;
                                    called = true;
                                });
    ASSERT_TRUE(called);
    EXPECT_TRUE(seen.ok) << seen.error;
    EXPECT_TRUE(QFile::exists(seen.path));
    EXPECT_TRUE(seen.error.empty());
}

TEST_F(UpdateGuiTest, ChecksumMismatchRefusesTheDownload) {
    QSettings settings(scratch("mismatch-settings.ini"), QSettings::IniFormat);
    auto fake = std::make_unique<FakeSource>();
    fake->download_bytes = QByteArrayLiteral("installer payload");
    fake->sums_text =
        "0000000000000000000000000000000000000000000000000000000000000000  payload.bin\n";
    UpdateChecker checker(settings, std::move(fake));

    const auto release = release_from_json();
    update::Asset asset{"payload.bin",
                        "https://github.com/aniketpratapsingh/alphashape/releases/download/"
                        "v0.3.0/payload.bin"};
    DownloadResult seen;
    bool called = false;
    const QString cache = scratch("cache-mismatch");
    checker.download_and_verify(release, asset, cache, {},
                                [&](const DownloadResult& result) {
                                    seen = result;
                                    called = true;
                                });
    ASSERT_TRUE(called);
    EXPECT_FALSE(seen.ok);
    EXPECT_NE(seen.error.find("checksum mismatch"), std::string::npos);
    // A refused download is never left behind.
    EXPECT_FALSE(QFile::exists(QDir(cache).filePath(QStringLiteral("payload.bin"))));
}

TEST_F(UpdateGuiTest, MissingChecksumEntryRefusesTheDownload) {
    QSettings settings(scratch("missing-sum-settings.ini"), QSettings::IniFormat);
    auto fake = std::make_unique<FakeSource>();
    fake->download_bytes = QByteArrayLiteral("installer payload");
    fake->sums_text = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  other.bin\n";
    UpdateChecker checker(settings, std::move(fake));

    const auto release = release_from_json();
    update::Asset asset{"payload.bin",
                        "https://github.com/aniketpratapsingh/alphashape/releases/download/"
                        "v0.3.0/payload.bin"};
    DownloadResult seen;
    checker.download_and_verify(release, asset, scratch("cache-missing"), {},
                                [&](const DownloadResult& result) { seen = result; });
    EXPECT_FALSE(seen.ok);
    EXPECT_NE(seen.error.find("does not list"), std::string::npos);
}

TEST_F(UpdateGuiTest, UpdateUrlAllowListAcceptsGitHubAndAssetCdnHosts) {
    // A GitHub release download redirects to one of two CDN hosts; both must be
    // allowed, on every hop.
    const char* const allowed[] = {
        "https://api.github.com/repos/aniketpratapsingh/alphashape/releases",
        "https://github.com/aniketpratapsingh/alphashape/releases/download/v0.3.0/SHA256SUMS",
        "https://objects.githubusercontent.com/github-production-release-asset/1/2",
        "https://release-assets.githubusercontent.com/github-production-release-asset/1/2",
        "https://github.com:443/aniketpratapsingh/alphashape",
    };
    for (const char* url : allowed) {
        EXPECT_TRUE(is_allowed_update_metadata_url(url)) << url;
    }
    // The API itself is a metadata host only, never an installer host.
    EXPECT_FALSE(
        is_allowed_update_download_url("https://api.github.com/repos/a/b/releases"));
    EXPECT_TRUE(is_allowed_update_download_url(
        "https://github.com/aniketpratapsingh/alphashape/releases/download/v0.3.0/x.dmg"));
    EXPECT_TRUE(is_allowed_update_download_url(
        "https://objects.githubusercontent.com/github-production-release-asset/1/2"));
    EXPECT_TRUE(is_allowed_update_download_url(
        "https://release-assets.githubusercontent.com/github-production-release-asset/1/2"));
}

TEST_F(UpdateGuiTest, UpdateUrlAllowListRejectsInsecureForeignAndTrickyUrls) {
    const char* const rejected[] = {
        // Not HTTPS.
        "http://github.com/aniketpratapsingh/alphashape/releases/download/v0.3.0/x.dmg",
        // Not a GitHub host.
        "https://example.com/alphashape.dmg",
        // Lookalikes: a subdomain and a prefixed host.
        "https://github.com.evil.com/alphashape.dmg",
        "https://evilgithub.com/alphashape.dmg",
        "https://objects.githubusercontent.com.evil.com/x",
        // A userinfo trick: the real host is evil.com.
        "https://github.com@evil.com/alphashape.dmg",
        // GitHub is served on 443; another port is a different server.
        "https://github.com:8443/alphashape.dmg",
        // Not a URL at all.
        "not a url",
        "",
    };
    for (const char* url : rejected) {
        EXPECT_FALSE(is_allowed_update_metadata_url(url)) << url;
        EXPECT_FALSE(is_allowed_update_download_url(url)) << url;
    }
}

TEST_F(UpdateGuiTest, DestroyingTheDialogMidDownloadCancelsWithoutCrashing) {
    QSettings settings(scratch("destroy-settings.ini"), QSettings::IniFormat);
    auto fake = std::make_unique<SlowSource>();
    SlowSource* raw = fake.get();
    UpdateChecker checker(settings, std::move(fake));

    auto dialog = std::make_unique<UpdateDialog>(available_result(), checker);
    dialog->start_download();
    ASSERT_TRUE(raw->has_pending()) << "the fake download did not start";

    dialog.reset();
    EXPECT_GE(raw->cancel_calls, 1) << "closing the dialog did not cancel the download";

    // The reply arrives after the dialog is gone: the guarded callback must do
    // nothing rather than touch freed widgets.
    raw->complete(false, "cancelled");
    EXPECT_FALSE(raw->has_pending());
}

TEST_F(UpdateGuiTest, UnsafeAssetNamesAreRefusedBeforeAnyFileIsWritten) {
    QSettings settings(scratch("unsafe-settings.ini"), QSettings::IniFormat);
    UpdateChecker checker(settings, std::make_unique<FakeSource>());
    const auto release = release_from_json();
    const QDir cache(scratch("cache-unsafe"));
    const std::vector<std::string> names = {"", "..", "../evil.bin", "a/b.bin", "a\\b.bin"};

    for (const std::string& name : names) {
        update::Asset asset{name, "https://github.com/x/download/" + name};
        DownloadResult seen;
        bool called = false;
        checker.download_and_verify(release, asset, cache.path(), {},
                                    [&](const DownloadResult& result) {
                                        seen = result;
                                        called = true;
                                    });
        ASSERT_TRUE(called) << name;
        EXPECT_FALSE(seen.ok) << name;
        EXPECT_NE(seen.error.find("unsafe"), std::string::npos) << name;
    }
    // Nothing escaped the cache directory.
    EXPECT_FALSE(QFile::exists(scratch("evil.bin")));
}

TEST_F(UpdateGuiTest, PreferencesPersistAutoCheckAndChannel) {
    QSettings settings(scratch("prefs-settings.ini"), QSettings::IniFormat);
    UpdateChecker checker(settings, std::make_unique<FakeSource>());
    PreferencesDialog dialog(checker);

    ASSERT_NE(dialog.auto_check(), nullptr);
    EXPECT_TRUE(dialog.auto_check()->isChecked());
    ASSERT_NE(dialog.channel(), nullptr);
    EXPECT_EQ(dialog.channel()->currentIndex(), 0);

    dialog.auto_check()->setChecked(false);
    EXPECT_FALSE(settings.value("updates/auto_check", true).toBool());

    dialog.channel()->setCurrentIndex(1);
    EXPECT_EQ(settings.value("updates/channel").toString(), QStringLiteral("prerelease"));
    EXPECT_EQ(static_cast<int>(checker.channel()), static_cast<int>(update::Channel::Prerelease));

    bool requested = false;
    QObject::connect(&dialog, &PreferencesDialog::check_requested,
                     [&]() { requested = true; });
    dialog.check_now()->click();
    EXPECT_TRUE(requested);
}
