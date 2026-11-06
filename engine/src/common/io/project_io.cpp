#include "common/io/project_io.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>

#include "base/io/zip.hpp"
#include "base/platform/env.hpp"
#include "base/utils/error.hpp"
#include "common/io/doc_xml.hpp"

namespace alphashape {

namespace fs = std::filesystem;

namespace {

constexpr const char* kManifestMagic = "alphashape-project";
constexpr const char* kFilesKey = "files=";
constexpr const char* kAbsolutePathsKey = "Project/AbsolutePaths";

std::vector<std::uint8_t> read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw ProjectIOError("cannot open '" + path.string() + "'");
    }
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), {});
}

void write_file_atomically(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
    fs::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw ProjectIOError("cannot write '" + tmp.string() + "'");
        }
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!out) {
            throw ProjectIOError("short write to '" + tmp.string() + "'");
        }
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp);
        throw ProjectIOError("cannot replace '" + path.string() + "': " + ec.message());
    }
}

/// "dir/name.psx" -> "dir/name". Only the extension of the final component is
/// stripped, so a dot in a directory name is left alone.
fs::path strip_extension(const fs::path& path) {
    fs::path out = path;
    out.replace_extension();
    return out;
}

/// The project's own directory, made absolute so a bare relative project path
/// still yields a stable base for stored frame paths.
std::string project_directory(const fs::path& project) {
    std::error_code ec;
    const fs::path absolute = fs::absolute(project, ec);
    const fs::path directory = (ec ? project : absolute).parent_path();
    return platform::path_to_utf8(directory);
}

/// `<base>.files/cache`, where `<base>` is the project path without extension.
std::string cache_directory(const fs::path& project) {
    const fs::path files(
        platform::path_from_utf8(platform::path_to_utf8(strip_extension(project)) + ".files"));
    return platform::path_to_utf8(files / "cache");
}

}  // namespace

void ProjectIO::save(const Document& document, const std::string& path,
                     bool absolute_paths) {
    const fs::path project = platform::path_from_utf8(path);
    const bool single_file = project.extension() == ".psz";
    const fs::path base = strip_extension(project);
    const fs::path files_dir =
        platform::path_from_utf8(platform::path_to_utf8(base) + ".files");
    const fs::path zip_path = files_dir / "project.zip";

    // The absolute/relative choice is recorded in doc.xml but is a property of
    // the container, not application metadata: it is written on this shallow
    // copy (the chunks are shared) and consumed on load.
    Document view = document;
    view.meta().set(kAbsolutePathsKey, absolute_paths ? "true" : "false");

    IoContext context;
    context.project_dir = project_directory(project);
    // Use the project cache only when it belongs to this document: a save-as to
    // a new path must not leave a cache directory behind that no Document will
    // ever clear. The codec then uses the platform temporary directory.
    const std::string cache_dir = cache_directory(project);
    if (cache_dir == document.cache_dir()) {
        context.scratch_dir = cache_dir;
    }
    context.absolute_paths = absolute_paths;

    ZipArchive archive;
    write_document(view, archive, Document::kFormatVersion, context);

    if (single_file) {
        // One archive carrying exactly what project.zip carries: doc.xml plus
        // every chunk payload sidecar.
        write_file_atomically(project, archive.serialise());
        return;
    }

    std::error_code ec;
    fs::create_directories(files_dir, ec);
    if (ec) {
        throw ProjectIOError("cannot create '" + files_dir.string() + "': " + ec.message());
    }
    write_file_atomically(zip_path, archive.serialise());

    std::ostringstream manifest;
    manifest << kManifestMagic << ' ' << Document::kFormatVersion << '\n'
             << kFilesKey << files_dir.filename().string() << "/project.zip\n";
    const std::string text = manifest.str();
    write_file_atomically(project, std::vector<std::uint8_t>(text.begin(), text.end()));
}

void Document::save(const std::string& path,
                    const std::vector<std::int32_t>& chunk_ids,
                    bool absolute_paths) const {
    if (chunk_ids.empty()) {
        ProjectIO::save(*this, path, absolute_paths);
        return;
    }

    // A view holding only the selected chunks. The chunks are shared, not
    // copied: the writer is read-only, and a subset save may cover a chunk
    // with tens of megabytes of assets.
    Document subset;
    subset.set_path(path);
    subset.workspace().chunks().clear();
    subset.global_crs() = global_crs_;
    subset.meta() = meta_;
    // next_chunk_id is written verbatim, so an appended copy can never collide
    // with an id the source project already used (§4.8, PRJ-06).
    subset.workspace().set_next_chunk_id(workspace().next_chunk_id());

    std::int32_t first_kept = -1;
    for (const std::int32_t id : chunk_ids) {
        const auto& all = workspace().chunks();
        const auto it = std::find_if(all.begin(), all.end(),
                                     [id](const std::shared_ptr<Chunk>& chunk) {
                                         return chunk->id() == id;
                                     });
        if (it == all.end()) {
            throw ValueError("save: no chunk with id " + std::to_string(id));
        }
        auto& kept = subset.workspace().chunks();
        if (std::find_if(kept.begin(), kept.end(),
                         [id](const std::shared_ptr<Chunk>& chunk) {
                             return chunk->id() == id;
                         }) != kept.end()) {
            throw ValueError("save: chunk " + std::to_string(id) + " is listed twice");
        }
        kept.push_back(*it);
        if (first_kept < 0) {
            first_kept = id;
        }
    }

    // The active chunk must be in the file; a subset that excludes it falls
    // back to the first kept chunk.
    const bool active_kept =
        active_chunk_id_ >= 0 &&
        std::any_of(subset.workspace().chunks().begin(),
                    subset.workspace().chunks().end(),
                    [this](const std::shared_ptr<Chunk>& chunk) {
                        return chunk->id() == active_chunk_id_;
                    });
    subset.set_active_chunk_id(active_kept ? active_chunk_id_ : first_kept);

    ProjectIO::save(subset, path, absolute_paths);
}

std::shared_ptr<Document> ProjectIO::load(const std::string& path, ReadReport* report) {
    const fs::path project = platform::path_from_utf8(path);
    ZipArchive archive;

    if (project.extension() == ".psz" || project.extension() == ".zip") {
        // .psz is the single-file project; a bare .zip is the old direct
        // project.zip form. Both parse into memory whole.
        archive = ZipArchive::parse(read_file(project));
    } else {
        const auto bytes = read_file(project);
        std::istringstream manifest(std::string(bytes.begin(), bytes.end()));
        std::string magic;
        int version = 0;
        manifest >> magic >> version;
        if (magic != kManifestMagic) {
            throw ProjectIOError("Invalid project file: '" + path + "'");
        }
        fs::path zip_path;
        std::string line;
        while (std::getline(manifest, line)) {
            if (line.rfind(kFilesKey, 0) == 0) {
                zip_path = project.parent_path() / line.substr(std::string(kFilesKey).size());
            }
        }
        if (zip_path.empty()) {
            throw ProjectIOError("Invalid project file: no files= entry in '" + path + "'");
        }
        archive = ZipArchive::parse(read_file(zip_path));
    }

    IoContext context;
    context.project_dir = project_directory(project);
    context.scratch_dir = cache_directory(project);
    auto document = read_document(archive, report, context);
    document->set_path(path);

    // The storage flag describes how the file was written. Consume it so a
    // load/save round trip does not accumulate container bookkeeping in the
    // document's own metadata.
    auto& items = document->meta().items;
    items.erase(std::remove_if(items.begin(), items.end(),
                               [](const std::pair<std::string, std::string>& item) {
                                   return item.first == kAbsolutePathsKey;
                               }),
                items.end());
    return document;
}

}  // namespace alphashape
