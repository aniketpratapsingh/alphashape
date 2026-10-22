#include "common/algorithm/chunk/append_project_task.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/core/document.hpp"
#include "common/io/project_io.hpp"

namespace alphashape {
namespace {

/// Parse one chunk id from a StringList entry.
std::int32_t parse_chunk_id(const std::string& text) {
    if (text.empty()) {
        throw ValueError("AppendProject: empty chunk id");
    }
    std::size_t consumed = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &consumed);
    } catch (const std::exception&) {
        throw ValueError("AppendProject: chunk id '" + text + "' is not an integer");
    }
    if (consumed != text.size() || value < 0) {
        throw ValueError("AppendProject: chunk id '" + text + "' is not a valid id");
    }
    return static_cast<std::int32_t>(value);
}

}  // namespace

void AppendProjectTask::declare_params(Params& params) const {
    params.declare({"path", ParamType::String, std::monostate{},
                    "Path of the project to append (.psx or a project.zip)."});
    params.declare({"append_doc", ParamType::Bool, true,
                    "Also merge the source's document-level data: its global CRS "
                    "when this document has none, and metadata keys this document "
                    "does not already set."});
    params.declare({"chunk_ids", ParamType::StringList, std::vector<std::string>{},
                    "Decimal ids of the source chunks to append. Empty appends "
                    "every chunk in the source project."});
}

std::vector<Stage> AppendProjectTask::stages() const {
    return {Stage{"initialize", 0}, Stage{"append", 0}, Stage{"finalize", 0}};
}

void AppendProjectTask::apply(Document& document, Params& params,
                              const Progress& progress) {
    progress.update("initialize", 0.0);
    const std::string path = params.get_string("path");
    const std::vector<std::string> raw_ids = params.get_string_list("chunk_ids");
    const bool append_doc = params.get_bool("append_doc");

    std::shared_ptr<Document> source = ProjectIO::load(path);
    if (source == nullptr) {
        throw Error("AppendProject: '" + path + "' loaded no document");
    }

    // Select the incoming chunks before touching the destination.
    std::vector<std::shared_ptr<Chunk>> incoming;
    if (raw_ids.empty()) {
        incoming = source->workspace().chunks();
    } else {
        for (const std::string& text : raw_ids) {
            const std::int32_t id = parse_chunk_id(text);
            const auto& chunks = source->workspace().chunks();
            const auto it = std::find_if(chunks.begin(), chunks.end(),
                                         [id](const std::shared_ptr<Chunk>& chunk) {
                                             return chunk->id() == id;
                                         });
            if (it == chunks.end()) {
                throw ValueError("AppendProject: '" + path + "' has no chunk with id " +
                                 std::to_string(id));
            }
            if (std::find(incoming.begin(), incoming.end(), *it) != incoming.end()) {
                throw ValueError("AppendProject: chunk " + std::to_string(id) +
                                 " is listed twice");
            }
            incoming.push_back(*it);
        }
    }
    if (incoming.empty()) {
        throw ValueError("AppendProject: '" + path + "' has no chunks to append");
    }
    progress.update("initialize", 1.0);

    // Chunk ids are offset by the destination's next id, so every incoming
    // chunk lands on an id the destination has not used (§4.8). Chunk-local
    // ids stay inside their chunk and are preserved verbatim.
    Workspace& workspace = document.workspace();
    const std::int32_t offset = workspace.next_chunk_id();
    std::int32_t highest = -1;
    for (const auto& chunk : incoming) {
        highest = std::max(highest, chunk->id());
    }
    if (highest >= 0 &&
        offset > INT32_MAX - highest) {
        throw Error("AppendProject: chunk id space exhausted");
    }

    progress.update("append", 0.0);
    std::vector<std::shared_ptr<Chunk>> copies;
    copies.reserve(incoming.size());
    for (const auto& chunk : incoming) {
        auto copy = std::make_shared<Chunk>(*chunk);  // deep: asset payloads cloned
        copy->set_id(offset + chunk->id());
        copies.push_back(std::move(copy));
    }
    progress.update("append", 1.0);

    progress.update("finalize", 0.0);
    if (append_doc) {
        if (!document.global_crs().is_set() && source->global_crs().is_set()) {
            document.global_crs() = source->global_crs();
        }
        for (const auto& item : source->meta().items) {
            if (document.meta().find(item.first) == nullptr) {
                document.meta().set(item.first, item.second);
            }
        }
    }
    for (auto& copy : copies) {
        // restore_chunk-style insertion keeps the destination's counter ahead
        // of every appended id.
        workspace.chunks().push_back(std::move(copy));
    }
    workspace.set_next_chunk_id(offset + highest + 1);
    document.set_active_chunk_id(workspace.chunks().back()->id());
    progress.update("finalize", 1.0);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "AppendProject: " << incoming.size() << " chunk(s) from '"
                            << path << "' -> ids from " << offset;
    }
}

}  // namespace alphashape
