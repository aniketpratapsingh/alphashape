#include "common/algorithm/chunk/duplicate_chunk_task.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/core/chunk_ops.hpp"
#include "common/core/document.hpp"

namespace alphashape {

void DuplicateChunkTask::declare_params(Params& params) const {
    params.declare({"chunk_id", ParamType::Int, std::int64_t{-1},
                    "Id of the chunk to copy. -1 copies the active chunk."});
    params.declare({"label", ParamType::String, std::string{},
                    "Label for the copy. Empty selects \"<source label> copy\"."});
    params.declare({"copy_assets", ParamType::Bool, true,
                    "Copy point clouds, models, elevations, orthomosaics, tiled "
                    "models and panorama images."});
    params.declare({"copy_depth_maps", ParamType::Bool, true,
                    "Copy the dense depth-map sets."});
    params.declare({"copy_tie_points", ParamType::Bool, true,
                    "Copy keypoints, matches and tracks (the sparse tie points)."});
}

std::vector<Stage> DuplicateChunkTask::stages() const {
    return {Stage{"initialize", 0}, Stage{"duplicate", 0}, Stage{"finalize", 0}};
}

void DuplicateChunkTask::apply(Document& document, Params& params,
                               const Progress& progress) {
    progress.update("initialize", 0.0);

    // Resolve the source: an explicit id, or the active chunk for -1.
    const std::int64_t requested = params.get_int("chunk_id");
    Chunk* source = nullptr;
    if (requested < 0) {
        source = document.active_chunk();
        if (source == nullptr) {
            throw Error("DuplicateChunk needs an active chunk or an explicit chunk_id");
        }
    } else if (requested > std::int64_t{INT32_MAX}) {
        throw ValueError("DuplicateChunk: chunk_id " + std::to_string(requested) +
                         " is out of range");
    } else {
        for (const auto& chunk : document.workspace().chunks()) {
            if (chunk->id() == static_cast<std::int32_t>(requested)) {
                source = chunk.get();
                break;
            }
        }
        if (source == nullptr) {
            throw ValueError("DuplicateChunk: no chunk with id " +
                             std::to_string(requested));
        }
    }

    ChunkCopyOptions options;
    options.copy_assets = params.get_bool("copy_assets");
    options.copy_depth_maps = params.get_bool("copy_depth_maps");
    options.copy_tie_points = params.get_bool("copy_tie_points");
    std::string label = params.get_string("label");
    if (label.empty()) {
        label = source->label() + " copy";
    }
    const std::int32_t source_id = source->id();
    progress.update("initialize", 1.0);

    // Build the copy before touching the workspace, so a failure leaves the
    // document untouched.
    progress.update("duplicate", 0.0);
    const std::int32_t new_id = document.workspace().next_chunk_id();
    std::shared_ptr<Chunk> copy = std::make_shared<Chunk>(
        duplicate_chunk(*source, new_id, std::move(label), options));
    progress.update("duplicate", 1.0);

    progress.update("finalize", 0.0);
    document.workspace().chunks().push_back(std::move(copy));
    document.workspace().set_next_chunk_id(new_id + 1);
    document.set_active_chunk_id(new_id);
    progress.update("finalize", 1.0);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "DuplicateChunk: chunk " << source_id << " -> " << new_id
                            << " ('" << document.workspace().chunks().back()->label()
                            << "')";
    }
}

}  // namespace alphashape
