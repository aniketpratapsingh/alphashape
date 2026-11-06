#include "common/core/document.hpp"

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "base/platform/env.hpp"
#include "base/utils/error.hpp"

namespace alphashape {

namespace fs = std::filesystem;

namespace {

/// Same normal form for guards and same-project checks: absolute with "." and
/// ".." resolved, so two spellings of one project share a guard.
std::string canonical_key(const std::string& path) {
    if (path.empty()) {
        return std::string();
    }
    const fs::path given = platform::path_from_utf8(path);
    std::error_code ec;
    fs::path canonical = fs::weakly_canonical(given, ec);
    if (ec) {
        canonical = given;
    }
    return platform::path_to_utf8(canonical.lexically_normal());
}

std::mutex& cache_guard_mutex() {
    static std::mutex guard_mutex;
    return guard_mutex;
}

/// One live guard per cache directory in this process. Documents that name the
/// same project share the guard, so the directory is removed exactly once, when
/// the last of them is destroyed.
std::map<std::string, std::weak_ptr<DocumentCacheGuard>>& cache_guards() {
    static std::map<std::string, std::weak_ptr<DocumentCacheGuard>> guards;
    return guards;
}

}  // namespace

std::string project_cache_directory(const std::string& project_path) {
    if (project_path.empty()) {
        return std::string();
    }
    fs::path base = platform::path_from_utf8(project_path);
    // A project.zip opened directly lives in `<base>.files`, so the cache is
    // that directory's own cache subdirectory, not a sibling of it.
    if (base.extension() == ".zip" && base.parent_path().extension() == ".files") {
        fs::path cache = base.parent_path() / "cache";
        cache.make_preferred();
        return platform::path_to_utf8(cache);
    }
    base.replace_extension();
    const fs::path files(
        platform::path_from_utf8(platform::path_to_utf8(base) + ".files"));
    fs::path cache = files / "cache";
    cache.make_preferred();
    return platform::path_to_utf8(cache);
}

/// Removes the project cache directory when the last Document that used it is
/// destroyed. Only the cache subdirectory is touched.
class DocumentCacheGuard {
public:
    explicit DocumentCacheGuard(std::string dir) : dir_(std::move(dir)) {}
    ~DocumentCacheGuard() {
        std::error_code ignored;
        fs::remove_all(platform::path_from_utf8(dir_), ignored);
    }

    DocumentCacheGuard(const DocumentCacheGuard&) = delete;
    DocumentCacheGuard& operator=(const DocumentCacheGuard&) = delete;

private:
    std::string dir_;
};

void Document::refresh_cache_guard() {
    const std::string dir = project_cache_directory(path_);
    if (dir.empty()) {
        cache_guard_.reset();
        return;
    }
    // Key on the cache directory itself: a .psx and the project.zip opened
    // directly are two spellings of one project and must share one guard.
    const std::string key = canonical_key(dir);
    const std::lock_guard<std::mutex> lock(cache_guard_mutex());
    auto& guards = cache_guards();
    for (auto it = guards.begin(); it != guards.end();) {
        if (it->second.expired()) {
            it = guards.erase(it);
        } else {
            ++it;
        }
    }
    const auto found = guards.find(key);
    if (found != guards.end()) {
        cache_guard_ = found->second.lock();
    }
    if (!cache_guard_) {
        cache_guard_ = std::make_shared<DocumentCacheGuard>(dir);
        guards[key] = cache_guard_;
    }
}

void Document::set_path(std::string path) {
    path_ = std::move(path);
    refresh_cache_guard();
}

std::string Document::cache_dir() const {
    return project_cache_directory(path_);
}

std::shared_ptr<void> Document::pin_cache_directory() const { return cache_guard_; }

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
    set_path(std::move(path));
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
