#include "common/io/project_io.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>

#include "base/io/zip.hpp"
#include "base/utils/error.hpp"
#include "common/io/doc_xml.hpp"

namespace alphashape {

namespace fs = std::filesystem;

namespace {

constexpr const char* kManifestMagic = "alphashape-project";
constexpr const char* kFilesKey = "files=";

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

}  // namespace

void ProjectIO::save(const Document& document, const std::string& path) {
    const fs::path project(path);
    const fs::path base = strip_extension(project);
    const fs::path files_dir = fs::path(base.string() + ".files");
    const fs::path zip_path = files_dir / "project.zip";

    ZipArchive archive;
    write_document(document, archive);  // validates before touching the disk

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
                    const std::vector<std::int32_t>& chunk_ids) const {
    if (chunk_ids.empty()) {
        ProjectIO::save(*this, path);
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

    ProjectIO::save(subset, path);
}

std::shared_ptr<Document> ProjectIO::load(const std::string& path) {
    const fs::path project(path);
    fs::path zip_path;

    if (project.extension() == ".zip") {
        zip_path = project;
    } else {
        const auto bytes = read_file(project);
        std::istringstream manifest(std::string(bytes.begin(), bytes.end()));
        std::string magic;
        int version = 0;
        manifest >> magic >> version;
        if (magic != kManifestMagic) {
            throw ProjectIOError("Invalid project file: '" + path + "'");
        }
        std::string line;
        while (std::getline(manifest, line)) {
            if (line.rfind(kFilesKey, 0) == 0) {
                zip_path = project.parent_path() / line.substr(std::string(kFilesKey).size());
            }
        }
        if (zip_path.empty()) {
            throw ProjectIOError("Invalid project file: no files= entry in '" + path + "'");
        }
    }

    auto document = read_document(ZipArchive::parse(read_file(zip_path)));
    document->set_path(path);
    return document;
}

}  // namespace alphashape
