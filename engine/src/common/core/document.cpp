#include "common/core/document.hpp"

#include "base/utils/error.hpp"

namespace alphashape {

Chunk* Workspace::find_chunk(std::int32_t id) {
    for (auto& chunk : chunks_) {
        if (chunk->id() == id) {
            return chunk.get();
        }
    }
    return nullptr;
}

Chunk& Workspace::add_chunk(std::string label) {
    auto chunk = std::make_shared<Chunk>(std::move(label));
    chunk->set_id(next_chunk_id_++);
    chunks_.push_back(std::move(chunk));
    return *chunks_.back();
}

Chunk& Workspace::restore_chunk(std::int32_t id, std::string label) {
    if (find_chunk(id) != nullptr) {
        throw ValueError("duplicate chunk id " + std::to_string(id));
    }
    auto chunk = std::make_shared<Chunk>(std::move(label));
    chunk->set_id(id);
    chunks_.push_back(std::move(chunk));
    if (id >= next_chunk_id_) {
        next_chunk_id_ = id + 1;
    }
    return *chunks_.back();
}

Document::Document() {
    auto workspace = std::make_shared<Workspace>();
    workspace->set_id(0);
    workspaces_.push_back(std::move(workspace));
}

Document::Document(std::string path) : Document() {
    path_ = std::move(path);
}

Chunk* Document::active_chunk() {
    if (active_chunk_id_ < 0) {
        return nullptr;
    }
    for (auto& workspace : workspaces_) {
        if (Chunk* chunk = workspace->find_chunk(active_chunk_id_)) {
            return chunk;
        }
    }
    return nullptr;
}

const Chunk* Document::active_chunk() const {
    if (active_chunk_id_ < 0) {
        return nullptr;
    }
    for (const auto& workspace : workspaces_) {
        for (const auto& chunk : workspace->chunks()) {
            if (chunk != nullptr && chunk->id() == active_chunk_id_) {
                return chunk.get();
            }
        }
    }
    return nullptr;
}

// -- coordinate system -----------------------------------------------------

crs::CoordinateSystem Document::global_coordinate_system() const {
#if defined(ALPHASHAPE_USE_PROJ)
    return static_cast<crs::CoordinateSystem>(global_crs_);
#else
    throw Error("coordinate systems need a build with PROJ (ALPHASHAPE_USE_PROJ=ON)");
#endif
}

void Document::set_global_coordinate_system(const crs::CoordinateSystem& cs) {
#if defined(ALPHASHAPE_USE_PROJ)
    global_crs_ = CRS(cs);
#else
    static_cast<void>(cs);
    throw Error("coordinate systems need a build with PROJ (ALPHASHAPE_USE_PROJ=ON)");
#endif
}

}  // namespace alphashape
