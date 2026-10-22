#include "common/algorithm/chunk/merge_chunks_task.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/core/chunk_ops.hpp"
#include "common/core/document.hpp"

namespace alphashape {
namespace {

/// Parse one id from a StringList entry. The CLI and the project meta both
/// spell ids as decimal strings, so an empty or non-numeric entry is an error
/// rather than a silently skipped chunk.
std::int32_t parse_chunk_id(const std::string& text) {
    if (text.empty()) {
        throw ValueError("MergeChunks: empty chunk id");
    }
    std::size_t consumed = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &consumed);
    } catch (const std::exception&) {
        throw ValueError("MergeChunks: chunk id '" + text + "' is not an integer");
    }
    if (consumed != text.size() || value < 0) {
        throw ValueError("MergeChunks: chunk id '" + text + "' is not a valid id");
    }
    return static_cast<std::int32_t>(value);
}

}  // namespace

void MergeChunksTask::declare_params(Params& params) const {
    params.declare({"chunk_ids", ParamType::StringList, std::vector<std::string>{},
                    "Decimal ids of the chunks to merge, in order. Empty merges "
                    "every chunk in the workspace except the active one when that "
                    "leaves at least two; see the task docs."});
    params.declare({"label", ParamType::String, std::string("Merged chunk"),
                    "Label for the merged chunk."});
}

std::vector<Stage> MergeChunksTask::stages() const {
    // Linear in the total content of the sources: resolve, then concatenate.
    return {Stage{"initialize", 0}, Stage{"merge", 0}, Stage{"finalize", 0}};
}

void MergeChunksTask::apply(Document& document, Params& params,
                            const Progress& progress) {
    progress.update("initialize", 0.0);

    std::vector<std::string> raw_ids = params.get_string_list("chunk_ids");
    std::vector<const Chunk*> sources;
    if (raw_ids.empty()) {
        // No list: merge the whole workspace. A single chunk is a no-op error
        // rather than a copy, so a mis-bound parameter is never silent.
        for (const auto& chunk : document.workspace().chunks()) {
            sources.push_back(chunk.get());
        }
        if (sources.size() < 2) {
            throw ValueError("MergeChunks: chunk_ids is empty and the workspace "
                             "holds fewer than two chunks");
        }
    } else {
        for (const std::string& text : raw_ids) {
            const std::int32_t id = parse_chunk_id(text);
            const Chunk* found = nullptr;
            for (const auto& chunk : document.workspace().chunks()) {
                if (chunk->id() == id) {
                    found = chunk.get();
                    break;
                }
            }
            if (found == nullptr) {
                throw ValueError("MergeChunks: no chunk with id " + std::to_string(id));
            }
            for (const Chunk* existing : sources) {
                if (existing->id() == id) {
                    throw ValueError("MergeChunks: chunk " + std::to_string(id) +
                                     " is listed twice");
                }
            }
            sources.push_back(found);
        }
        if (sources.size() < 2) {
            throw ValueError("MergeChunks: need at least two chunks to merge");
        }
    }
    progress.update("initialize", 1.0);

    const std::string label = params.get_string("label");
    const std::int32_t new_id = document.workspace().next_chunk_id();

    // Build first: a mixed-CRS refusal or a broken source leaves the document
    // untouched.
    progress.update("merge", 0.0);
    std::shared_ptr<Chunk> merged =
        std::make_shared<Chunk>(merge_chunks(sources, new_id, label));
    progress.update("merge", 1.0);

    progress.update("finalize", 0.0);
    const std::vector<std::string> validation = merged->validate();
    if (!validation.empty()) {
        throw Error("MergeChunks: merged chunk is invalid: " + validation.front());
    }
    document.workspace().chunks().push_back(std::move(merged));
    document.workspace().set_next_chunk_id(new_id + 1);
    document.set_active_chunk_id(new_id);
    progress.update("finalize", 1.0);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "MergeChunks: " << sources.size() << " chunks -> "
                            << new_id << " with "
                            << document.workspace().chunks().back()->cameras().size()
                            << " cameras";
    }
}

}  // namespace alphashape
