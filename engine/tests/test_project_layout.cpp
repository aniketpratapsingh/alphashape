// WP-11.9: project layouts — .psz, relative frame paths, cache and writer lock.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <gtest/gtest.h>

#include "base/io/zip.hpp"
#include "base/platform/process.hpp"
#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "common/commands/command.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/io/doc_xml.hpp"
#include "common/io/lock_file.hpp"
#include "common/io/project_io.hpp"
#include "perf_gate.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

class ProjectLayout : public ::testing::Test {
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
        // A name like "images/a.jpg" appended to a native directory otherwise
        // keeps the '/' it was typed with; normalise the whole path so
        // expectations match the engine's preferred-separator output.
        fs::path full = (fs::path(dir_) / name).lexically_normal();
        full.make_preferred();
        return full.string();
    }

    static int counter_;
    std::string dir_;
};
int ProjectLayout::counter_ = 0;

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

TEST_F(ProjectLayout, PszRoundTrip) {
    std::error_code ec;
    fs::create_directories(path("images"), ec);
    write_text(path("images/a.jpg"), "a");
    write_text(path("images/b.jpg"), "b");
    Document document = make_document({path("images/a.jpg"), path("images/b.jpg")});

    // A non-frame sidecar: a depth-map set travels with the container too.
    Chunk& chunk = *document.workspace().chunks().front();
    DepthMap depth;
    depth.camera_id = chunk.frames().front()->id();
    depth.width = 4;
    depth.height = 3;
    depth.downscale = 1;
    depth.depths.assign(12U, 2.5F);
    depth.depths.front() = -1.0F;
    depth.normals.assign(36U, 0.0F);
    depth.confidence.assign(12U, 1.0F);
    depth.flags.assign(12U, static_cast<std::uint8_t>(0));
    depth.filter_mode = 2;
    chunk.depth_maps().push_back(depth);

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
    ASSERT_EQ(loaded->workspace().chunks().front()->depth_maps().size(), 1U);
    EXPECT_TRUE(loaded->workspace().chunks().front()->depth_maps().front() == depth);
    EXPECT_TRUE(*loaded->workspace().chunks().front() ==
                *document.workspace().chunks().front());

    // dir -> psz -> dir reproduces the directory archive byte for byte.
    const std::string dir = path("roundtrip.psx");
    ProjectIO::save(document, dir);
    const std::vector<std::uint8_t> dir_zip =
        read_bytes(path("roundtrip.files/project.zip"));
    const std::shared_ptr<Document> from_dir = ProjectIO::load(dir);
    ProjectIO::save(*from_dir, psz);
    const std::shared_ptr<Document> from_psz = ProjectIO::load(psz);
    ProjectIO::save(*from_psz, path("roundtrip2.psx"));
    EXPECT_EQ(read_bytes(path("roundtrip2.files/project.zip")), dir_zip);
}

TEST_F(ProjectLayout, DirectoryRoundTrip) {
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

TEST_F(ProjectLayout, DirectoryLayoutStable) {
    Document document = make_document({});
    // Insert out of order: §4.9 requires sorted meta keys on disk.
    document.meta().set("z-key", "last");
    document.meta().set("a-key", "first");
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

    // Sorted meta keys, and the default (relative) layout writes no flag.
    const std::size_t first_key = xml.find("name=\"a-key\"");
    const std::size_t last_key = xml.find("name=\"z-key\"");
    ASSERT_NE(first_key, std::string::npos);
    ASSERT_NE(last_key, std::string::npos);
    EXPECT_LT(first_key, last_key);
    EXPECT_EQ(xml.find("Project/AbsolutePaths"), std::string::npos);
}

TEST_F(ProjectLayout, RelativePhotosSurviveMove) {
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

TEST_F(ProjectLayout, MissingPhotosWarnNotFail) {
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

TEST_F(ProjectLayout, AbsolutePathsFlagWritten) {
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

TEST_F(ProjectLayout, CacheClearedOnClose) {
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

TEST_F(ProjectLayout, LockFileRejectsSecondWriter) {
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

TEST_F(ProjectLayout, StaleLockIsBroken) {
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

TEST_F(ProjectLayout, CacheDirectoryForDirectProjectZip) {
    ProjectIO::save(make_document({}), path("foo.psx"));
    const std::shared_ptr<Document> loaded = ProjectIO::load(path("foo.files/project.zip"));
    // The direct project.zip must not produce a "<...>.files.files" sibling.
    const std::string expected = path("foo.files/cache");
    EXPECT_EQ(loaded->cache_dir(), expected);

    // Scratch consumers (the undo stack spills through this) land there.
    UndoStack stack(loaded.get());
    EXPECT_EQ(stack.cache_dir(), expected);
    std::error_code ec;
    fs::create_directories(expected, ec);
    ASSERT_FALSE(ec) << ec.message();
    write_text((fs::path(expected) / "spill.bin").string(), "x");
    EXPECT_TRUE(fs::exists(fs::path(expected) / "spill.bin"));
}

TEST_F(ProjectLayout, ConcurrentStaleBreakKeepsOneWriter) {
    const std::string psx = path("project.psx");
    const std::string stale = "pid=2147000000\nhost=" + platform::hostname() +
                              "\nstarted=0\ntoken=dead\n";
    std::error_code ec;
    fs::create_directories(fs::path(ProjectLock::lock_path(psx)).parent_path(), ec);
    ASSERT_FALSE(ec) << ec.message();
    write_text(ProjectLock::lock_path(psx), stale);

    // Both callers read the same stale record; the first wins and installs a
    // fresh lock.
    ProjectLock first = ProjectLock::acquire_assuming_stale(psx, stale);
    ASSERT_TRUE(first.held());

    // The second caller still believes the OLD record is there. It must notice
    // the fresh record, put it back and report a live writer, never delete it.
    bool rejected = false;
    try {
        ProjectLock second = ProjectLock::acquire_assuming_stale(psx, stale);
        (void)second;
    } catch (const ProjectIOError&) {
        rejected = true;
    }
    EXPECT_TRUE(rejected);
    ASSERT_TRUE(fs::exists(ProjectLock::lock_path(psx)));

    // Exactly one holder: a plain acquire is refused and the first's release
    // removes the file.
    EXPECT_THROW(ProjectLock::acquire(psx), ProjectIOError);
    first.release();
    EXPECT_FALSE(fs::exists(ProjectLock::lock_path(psx)));
}

TEST_F(ProjectLayout, PszLockSitsBesideTheFile) {
    const std::string psz = path("project.psz");
    const ProjectLock lock = ProjectLock::acquire(psz);
    EXPECT_EQ(ProjectLock::lock_path(psz), path("project.psz.lock"));
    EXPECT_TRUE(fs::exists(path("project.psz.lock")));
    // A single-file project must not grow a permanent .files directory.
    EXPECT_FALSE(fs::exists(path("project.files")));
}

TEST_F(ProjectLayout, AbsolutePathsResolveRelativeInput) {
    write_text(path("images/a.jpg"), "a");
    // The model holds a relative path; absolute_paths=true must still store a
    // location a reader can open, resolved against the project directory.
    Document document = make_document({"images/a.jpg"});
    ProjectIO::save(document, path("project.psx"), true);

    const std::shared_ptr<Document> loaded = ProjectIO::load(path("project.psx"));
    const std::string resolved =
        loaded->workspace().chunks().front()->frames().front()->path();
    EXPECT_FALSE(fs::path(resolved).is_relative());
    EXPECT_EQ(resolved, path("images/a.jpg"));
}

TEST_F(ProjectLayout, RelativeProjectPathResolvesOnReload) {
    std::error_code ec;
    fs::create_directories(path("images"), ec);
    fs::create_directories(path("sub"), ec);
    ASSERT_FALSE(ec) << ec.message();
    write_text(path("images/a.jpg"), "a");

    // "sub/../proj.psx" names the same directory; the frame must resolve
    // against the normalised project directory, not against "sub/..".
    // Build it by concatenation: the fixture's path() helper would normalise.
    const std::string dotted = (fs::path(dir_) / "sub" / ".." / "proj.psx").string();
    Document document = make_document({path("images/a.jpg")});
    ProjectIO::save(document, dotted);

    const std::shared_ptr<Document> loaded = ProjectIO::load(dotted);
    EXPECT_EQ(loaded->workspace().chunks().front()->frames().front()->path(),
              path("images/a.jpg"));
}

TEST_F(ProjectLayout, ReadOnlyLockReportsReadOnly) {
#if defined(_WIN32)
    GTEST_SKIP() << "POSIX directory-permission test";
#else
    if (::geteuid() == 0) {
        GTEST_SKIP() << "root ignores directory permissions";
    }
    ProjectIO::save(make_document({}), path("project.psx"));
    const fs::path files = path("project.files");
    std::error_code ec;
    fs::permissions(files, fs::perms::owner_read | fs::perms::owner_exec,
                    fs::perm_options::replace, ec);
    ASSERT_FALSE(ec) << ec.message();

    // Not "locked by another writer": the lock cannot be written at all.
    EXPECT_THROW(ProjectLock::acquire(path("project.psx")), ProjectReadOnlyError);

    fs::permissions(files, fs::perms::owner_all, fs::perm_options::replace, ec);
    EXPECT_FALSE(ec) << ec.message();
#endif
}

TEST_F(ProjectLayout, LargeProjectSaveLoadPerf) {
    // §4.9 acceptance: a 200 MB project saves and loads in <= 6 s each on the
    // VM. 50,000,000 floats in one depth map is exactly 200 MB of sidecar.
    constexpr std::size_t kFloats = 50'000'000U;
    Document document = make_document({path("perf.jpg")});
    Chunk& chunk = *document.workspace().chunks().front();
    DepthMap depth;
    depth.camera_id = chunk.frames().front()->id();
    depth.width = 10000;
    depth.height = 5000;
    depth.downscale = 1;
    depth.depths.assign(kFloats, 0.0F);
    chunk.depth_maps().push_back(std::move(depth));

    const std::string psz = path("large.psz");
    const auto save_start = std::chrono::steady_clock::now();
    ProjectIO::save(document, psz);
    const auto save_end = std::chrono::steady_clock::now();
    const std::shared_ptr<Document> loaded = ProjectIO::load(psz);
    const auto load_end = std::chrono::steady_clock::now();

    const auto save_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(save_end - save_start)
            .count();
    const auto load_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(load_end - save_end)
            .count();
    RecordProperty("save_ms", std::to_string(save_ms));
    RecordProperty("load_ms", std::to_string(load_ms));

    ASSERT_EQ(loaded->workspace().chunks().front()->depth_maps().size(), 1U);
    EXPECT_EQ(loaded->workspace().chunks().front()->depth_maps().front().depths.size(),
              kFloats);

    if (test::perf_asserts_enabled()) {
        EXPECT_LE(save_ms, 6000);
        EXPECT_LE(load_ms, 6000);
    }
}

}  // namespace
}  // namespace alphashape
