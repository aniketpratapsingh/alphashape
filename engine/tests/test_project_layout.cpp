// WP-11.9: project layouts — .psz, relative frame paths, cache and writer lock.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/io/zip.hpp"
#include "base/platform/process.hpp"
#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/io/doc_xml.hpp"
#include "common/io/lock_file.hpp"
#include "common/io/project_io.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

class LayoutTempDir : public ::testing::Test {
protected:
    void SetUp() override {
        const auto base =
            fs::temp_directory_path() /
            ("alphashape_layout_" + std::to_string(platform::process_id()) + "_" +
             std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
        dir_ = base.string() + "_" + std::to_string(counter_++);
        std::error_code ec;
        fs::remove_all(dir_, ec);
        fs::create_directories(dir_, ec);
        ASSERT_FALSE(ec) << ec.message();
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    std::string path(const std::string& name) const {
        return (fs::path(dir_) / name).string();
    }

    static int counter_;
    std::string dir_;
};
int LayoutTempDir::counter_ = 0;

Document make_document(const std::vector<std::string>& photos) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    for (const std::string& photo : photos) {
        auto frame = std::make_shared<Frame>(photo, 64, 48);
        frame->set_id(chunk.allocate_id());
        chunk.frames().push_back(std::move(frame));
    }
    return document;
}

void write_text(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
}

std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), {});
}

/// Stored paths (and therefore paths resolved on load) use '/' on every
/// platform; comparisons against a native temp path go through this.
std::string slashed(std::string path) {
    std::replace(path.begin(), path.end(), '\\', '/');
    return path;
}

TEST_F(LayoutTempDir, ProjectLayout_PszRoundTrip) {
    std::error_code ec;
    fs::create_directories(path("images"), ec);
    write_text(path("images/a.jpg"), "a");
    write_text(path("images/b.jpg"), "b");
    Document document = make_document({path("images/a.jpg"), path("images/b.jpg")});

    const std::string psz = path("project.psz");
    ProjectIO::save(document, psz);
    ASSERT_TRUE(fs::exists(psz));
    EXPECT_FALSE(fs::exists(path("project.files")));

    // A .psz is one valid zip with doc.xml at its root.
    const ZipArchive archive = ZipArchive::parse(read_bytes(psz));
    EXPECT_TRUE(archive.contains(kDocXmlEntry));

    const std::shared_ptr<Document> loaded = ProjectIO::load(psz);
    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    ASSERT_EQ(loaded->workspace().chunks().front()->frames().size(), 2U);
    EXPECT_TRUE(*loaded->workspace().chunks().front() ==
                *document.workspace().chunks().front());
}

TEST_F(LayoutTempDir, ProjectLayout_DirectoryRoundTrip) {
    std::error_code ec;
    fs::create_directories(path("images"), ec);
    write_text(path("images/a.jpg"), "a");
    Document document = make_document({path("images/a.jpg")});

    const std::string psx = path("project.psx");
    ProjectIO::save(document, psx);
    ASSERT_TRUE(fs::exists(psx));
    ASSERT_TRUE(fs::exists(path("project.files/project.zip")));

    const std::shared_ptr<Document> loaded = ProjectIO::load(psx);
    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    EXPECT_TRUE(*loaded->workspace().chunks().front() ==
                *document.workspace().chunks().front());
}

TEST_F(LayoutTempDir, ProjectLayout_DirectoryLayoutStable) {
    const Document document = make_document({});
    const std::string psx = path("project.psx");
    const std::string zip = path("project.files/project.zip");

    ProjectIO::save(document, psx);
    const std::vector<std::uint8_t> zip_first = read_bytes(zip);
    const std::vector<std::uint8_t> manifest_first = read_bytes(psx);
    ASSERT_FALSE(zip_first.empty());

    ProjectIO::save(document, psx);
    EXPECT_EQ(read_bytes(zip), zip_first);
    EXPECT_EQ(read_bytes(psx), manifest_first);

    // The manifest and doc.xml are LF-only on every platform.
    const std::string xml = ZipArchive::parse(zip_first).get_text(kDocXmlEntry);
    EXPECT_EQ(xml.find('\r'), std::string::npos);
    EXPECT_EQ(std::count(manifest_first.begin(), manifest_first.end(),
                         static_cast<std::uint8_t>('\r')),
              0);
}

TEST_F(LayoutTempDir, ProjectLayout_RelativePhotosSurviveMove) {
    std::error_code ec;
    fs::create_directories(path("src/images"), ec);
    ASSERT_FALSE(ec) << ec.message();
    write_text(path("src/images/a.jpg"), "a");

    Document document = make_document({path("src/images/a.jpg")});
    ProjectIO::save(document, path("src/project.psx"));

    // The image tree and the project move together.
    fs::rename(path("src"), path("dst"), ec);
    ASSERT_FALSE(ec) << ec.message();

    const std::shared_ptr<Document> loaded = ProjectIO::load(path("dst/project.psx"));
    ASSERT_EQ(loaded->workspace().chunks().size(), 1U);
    const auto& frame = loaded->workspace().chunks().front()->frames().front();
    EXPECT_EQ(frame->path(), path("dst/images/a.jpg"));
    EXPECT_TRUE(fs::exists(frame->path()));
}

TEST_F(LayoutTempDir, ProjectLayout_MissingPhotosWarnNotFail) {
    Document document = make_document({path("images/missing.jpg")});
    ProjectIO::save(document, path("project.psx"));

    ReadReport report;
    const std::shared_ptr<Document> loaded = ProjectIO::load(path("project.psx"), &report);
    ASSERT_NE(loaded, nullptr);
    ASSERT_EQ(loaded->workspace().chunks().front()->frames().size(), 1U);

    bool warned = false;
    for (const std::string& warning : report.warnings) {
        if (warning.find("missing.jpg") != std::string::npos) {
            warned = true;
        }
    }
    EXPECT_TRUE(warned);
}

TEST_F(LayoutTempDir, ProjectLayout_AbsolutePathsFlagWritten) {
    write_text(path("a.jpg"), "a");
    Document document = make_document({path("a.jpg")});
    ProjectIO::save(document, path("project.psx"), true);

    const ZipArchive archive =
        ZipArchive::parse(read_bytes(path("project.files/project.zip")));
    const std::string xml = archive.get_text(kDocXmlEntry);
    EXPECT_NE(xml.find("Project/AbsolutePaths"), std::string::npos);
    EXPECT_NE(xml.find("value=\"true\""), std::string::npos);
    EXPECT_NE(xml.find(slashed(path("a.jpg"))), std::string::npos);

    const std::shared_ptr<Document> loaded = ProjectIO::load(path("project.psx"));
    EXPECT_EQ(loaded->workspace().chunks().front()->frames().front()->path(),
              path("a.jpg"));
}

TEST_F(LayoutTempDir, ProjectLayout_CacheClearedOnClose) {
    std::string cache;
    {
        Document document = make_document({});
        document.set_path(path("project.psx"));
        cache = document.cache_dir();
        ASSERT_FALSE(cache.empty());
        EXPECT_EQ(cache, path("project.files/cache"));

        std::error_code ec;
        fs::create_directories(cache, ec);
        ASSERT_FALSE(ec) << ec.message();
        write_text((fs::path(cache) / "spill.bin").string(), "x");
        EXPECT_TRUE(fs::exists(cache));
    }
    EXPECT_FALSE(fs::exists(cache));
}

TEST_F(LayoutTempDir, ProjectLayout_LockFileRejectsSecondWriter) {
    const std::string psx = path("project.psx");
    ProjectLock first = ProjectLock::acquire(psx);
    ASSERT_TRUE(fs::exists(ProjectLock::lock_path(psx)));

    bool rejected = false;
    try {
        ProjectLock second = ProjectLock::acquire(psx);
        (void)second;
    } catch (const ProjectIOError& error) {
        rejected = true;
        // The message names the holder, not just "locked".
        EXPECT_NE(std::string(error.what()).find(std::to_string(platform::process_id())),
                  std::string::npos);
    }
    EXPECT_TRUE(rejected);

    first.release();
    EXPECT_FALSE(fs::exists(ProjectLock::lock_path(psx)));
    const ProjectLock third = ProjectLock::acquire(psx);
    EXPECT_TRUE(third.held());
}

TEST_F(LayoutTempDir, ProjectLayout_StaleLockIsBroken) {
    const std::string psx = path("project.psx");
    const std::string lock = ProjectLock::lock_path(psx);
    std::error_code ec;
    fs::create_directories(fs::path(lock).parent_path(), ec);
    ASSERT_FALSE(ec) << ec.message();

    // Same host, a pid that cannot be running.
    write_text(lock, "pid=2147000000\nhost=" + platform::hostname() +
                         "\nstarted=0\ntoken=dead\n");
    ProjectLock local = ProjectLock::acquire(psx);
    EXPECT_TRUE(local.held());

    // Another host, older than the default 24 h maximum age.
    write_text(lock, "pid=1\nhost=some-other-host\nstarted=1\ntoken=old\n");
    const ProjectLock remote = ProjectLock::acquire(psx);
    EXPECT_TRUE(remote.held());
}

}  // namespace
}  // namespace alphashape
