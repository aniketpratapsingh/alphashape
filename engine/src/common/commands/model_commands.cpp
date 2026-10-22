// Model commands (M11 §4.7, WP-11.5b): chunk, workspace, marker, scale-bar,
// group and camera edits. Every command derives from StagedMutation, so a
// failing apply leaves the document untouched, and apply -> revert -> apply
// reproduces exactly the state of the first apply.
//
// Staging is deliberately container-scoped rather than a whole-Chunk copy:
// copying a chunk would deep-clone every asset payload (models, DEMs), which is
// precisely the cost the undo layer must not pay for a marker edit. The
// snapshot below covers the containers these commands mutate.

#include "common/commands/model_commands.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "base/utils/error.hpp"
#include "common/core/chunk_ops.hpp"
#include "common/io/sidecars.hpp"

#if defined(_WIN32)
#include <process.h>
#define ALPHASHAPE_GETPID _getpid
#else
#include <unistd.h>
#define ALPHASHAPE_GETPID getpid
#endif

namespace alphashape {
namespace {

// --- lookup helpers --------------------------------------------------------

Chunk* find_chunk(Document& document, std::int32_t chunk_id) {
    for (auto& workspace : document.workspaces()) {
        if (Chunk* chunk = workspace->find_chunk(chunk_id)) {
            return chunk;
        }
    }
    return nullptr;
}

Chunk& require_chunk(Document& document, std::int32_t chunk_id) {
    Chunk* chunk = find_chunk(document, chunk_id);
    if (chunk == nullptr) {
        throw ValueError("no chunk with id " + std::to_string(chunk_id));
    }
    return *chunk;
}

Workspace* find_workspace(Document& document, std::int32_t chunk_id) {
    for (auto& workspace : document.workspaces()) {
        if (workspace->find_chunk(chunk_id) != nullptr) {
            return workspace.get();
        }
    }
    return nullptr;
}

template <class T>
const T* find_by_id(const std::vector<T>& items, std::int32_t id) {
    for (const T& item : items) {
        if (item.id == id) {
            return &item;
        }
    }
    return nullptr;
}

template <class T>
T* find_by_id(std::vector<T>& items, std::int32_t id) {
    for (T& item : items) {
        if (item.id == id) {
            return &item;
        }
    }
    return nullptr;
}

/// Insert by ascending id. Throws ValueError on a duplicate, so a caller that
/// handed us an already-used id fails before the chunk is touched.
template <class T>
void insert_by_id(std::vector<T>& items, T value) {
    auto it = std::lower_bound(
        items.begin(), items.end(), value.id,
        [](const T& item, std::int32_t id) { return item.id < id; });
    if (it != items.end() && it->id == value.id) {
        throw ValueError("duplicate id " + std::to_string(value.id));
    }
    items.insert(it, std::move(value));
}

/// Same as insert_by_id, for containers keyed by camera id (projections).
void insert_projection(std::vector<MarkerProjection>& projections,
                       MarkerProjection projection) {
    const std::int32_t camera_id = projection.camera_id;
    auto it = std::lower_bound(
        projections.begin(), projections.end(), camera_id,
        [](const MarkerProjection& item, std::int32_t id) {
            return item.camera_id < id;
        });
    if (it != projections.end() && it->camera_id == camera_id) {
        throw ValueError("duplicate projection for camera " +
                         std::to_string(camera_id));
    }
    projections.insert(it, std::move(projection));
}

template <class T>
bool erase_by_id(std::vector<T>& items, std::int32_t id) {
    auto it = std::remove_if(items.begin(), items.end(),
                             [id](const T& item) { return item.id == id; });
    if (it == items.end()) {
        return false;
    }
    items.erase(it, items.end());
    return true;
}

// --- validation helpers ----------------------------------------------------

bool positive_finite(const Eigen::Vector3d& value) {
    return value.allFinite() && (value.array() > 0.0).all();
}

const Camera* find_camera(const Chunk& chunk, std::int32_t camera_id) {
    for (const Camera& camera : chunk.cameras()) {
        if (camera.id == camera_id) {
            return &camera;
        }
    }
    return nullptr;
}

void require_camera(const Chunk& chunk, std::int32_t camera_id, const char* context) {
    if (find_camera(chunk, camera_id) == nullptr) {
        throw ValueError(std::string(context) + ": no camera with id " +
                         std::to_string(camera_id));
    }
}

/// Markers keep their projections sorted by camera id with unique cameras.
/// A caller-built Marker may arrive in any order (a GUI adds the projection
/// the user just clicked), so the order is normalised here rather than
/// rejected; a repeated camera or a non-finite pixel is an error.
void normalize_projections(std::vector<MarkerProjection>& projections,
                           std::int32_t marker_id) {
    for (const MarkerProjection& projection : projections) {
        if (!projection.pixel.allFinite()) {
            throw ValueError("marker " + std::to_string(marker_id) +
                             ": projection pixel is not finite");
        }
        if (projection.camera_id < 0) {
            throw ValueError("marker " + std::to_string(marker_id) +
                             ": projection has no camera");
        }
    }
    std::stable_sort(projections.begin(), projections.end(),
                     [](const MarkerProjection& a, const MarkerProjection& b) {
                         return a.camera_id < b.camera_id;
                     });
    for (std::size_t i = 1; i < projections.size(); ++i) {
        if (projections[i - 1].camera_id == projections[i].camera_id) {
            throw ValueError("marker " + std::to_string(marker_id) +
                             ": camera " +
                             std::to_string(projections[i].camera_id) +
                             " has more than one projection");
        }
    }
}

void check_scalebar_endpoint(const Chunk& chunk, const ScalebarEndpoint& endpoint,
                             std::int32_t scalebar_id) {
    const std::string context = "scalebar " + std::to_string(scalebar_id);
    if (endpoint.kind == ScalebarEndpoint::Kind::Marker) {
        if (find_by_id(chunk.markers(), endpoint.id) == nullptr) {
            throw ValueError(context + ": no marker with id " +
                             std::to_string(endpoint.id));
        }
    } else {
        require_camera(chunk, endpoint.id, context.c_str());
    }
}

void check_scalebar_length(double length_m, double accuracy_m, std::int32_t id) {
    if (!std::isfinite(length_m) || length_m < 0.0) {
        throw ValueError("scalebar " + std::to_string(id) +
                         ": reference length must be finite and non-negative");
    }
    if (!std::isfinite(accuracy_m) || !(accuracy_m > 0.0)) {
        throw ValueError("scalebar " + std::to_string(id) +
                         ": reference accuracy must be finite and positive");
    }
}

/// Chunk containers that every model command may touch. Copies are shallow
/// for payloads (assets live in shared_ptrs) and cheap for the rest.
struct ChunkSnapshot {
    std::vector<Marker> markers;
    std::vector<MarkerGroup> marker_groups;
    std::vector<Scalebar> scalebars;
    std::vector<CameraGroup> camera_groups;
    std::vector<Camera> cameras;
    std::int32_t next_id = 0;
};

/// Base for commands that edit one chunk's M11 containers.
class ChunkContainersCommand : public StagedMutation {
public:
    explicit ChunkContainersCommand(std::int32_t chunk_id) : chunk_id_(chunk_id) {}

protected:
    Chunk& target_chunk(Document& document) const {
        return require_chunk(document, chunk_id_);
    }

    void stage(Document& document) override {
        Chunk& chunk = target_chunk(document);
        snapshot_.markers = chunk.markers();
        snapshot_.marker_groups = chunk.marker_groups();
        snapshot_.scalebars = chunk.scalebars();
        snapshot_.camera_groups = chunk.camera_groups();
        snapshot_.cameras = chunk.cameras();
        snapshot_.next_id = chunk.next_id();
    }

    void restore(Document& document) override {
        Chunk& chunk = target_chunk(document);
        chunk.markers() = snapshot_.markers;
        chunk.marker_groups() = snapshot_.marker_groups;
        chunk.scalebars() = snapshot_.scalebars;
        chunk.camera_groups() = snapshot_.camera_groups;
        chunk.cameras() = snapshot_.cameras;
        chunk.set_next_id(snapshot_.next_id);
    }

    std::int32_t chunk_id() const { return chunk_id_; }

private:
    std::int32_t chunk_id_;
    ChunkSnapshot snapshot_;
};

// --- Chunk commands --------------------------------------------------------

/// Add a chunk at `insert_index` (clamped to the workspace size).
///
/// The chunk carries a fresh id allocated from the workspace counter at apply
/// time; the chunk id counter is not part of the UndoStack's document model,
/// so restoring the previous counter keeps a redo id-stable. Staging records
/// the previous counter and the previous active chunk.
class AddChunkCommand : public StagedMutation {
public:
    AddChunkCommand(Workspace& workspace, std::string label, std::int32_t insert_index)
        : workspace_(&workspace),
          label_(std::move(label)),
          insert_index_(insert_index) {}

    std::string name() const override { return "Add Chunk"; }

protected:
    void validate(Document& document) override {
        Workspace* workspace = resolve(document);
        if (workspace == nullptr) {
            throw ValueError("add chunk: the workspace is not part of the document");
        }
    }

    void stage(Document& document) override {
        Workspace* workspace = resolve(document);
        previous_next_id_ = workspace->next_chunk_id();
        previous_active_id_ = document.active_chunk_id();
        inserted_id_ = -1;
    }

    void commit(Document& document) override {
        Workspace* workspace = resolve(document);
        Chunk& chunk = workspace->add_chunk(label_);
        inserted_id_ = chunk.id();
        auto& chunks = workspace->chunks();
        const std::size_t last = chunks.size() - 1;
        std::size_t index = insert_index_ < 0 ? last : static_cast<std::size_t>(insert_index_);
        index = std::min(index, last);
        if (index != last) {
            std::shared_ptr<Chunk> added = std::move(chunks.back());
            chunks.pop_back();
            chunks.insert(chunks.begin() + static_cast<std::ptrdiff_t>(index),
                          std::move(added));
        }
        document.set_active_chunk_id(inserted_id_);
    }

    void restore(Document& document) override {
        Workspace* workspace = resolve(document);
        if (workspace == nullptr) {
            throw ValueError("add chunk: the workspace is not part of the document");
        }
        auto& chunks = workspace->chunks();
        chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
                                    [this](const std::shared_ptr<Chunk>& chunk) {
                                        return chunk->id() == inserted_id_;
                                    }),
                     chunks.end());
        workspace->set_next_chunk_id(previous_next_id_);
        document.set_active_chunk_id(previous_active_id_);
    }

    std::int32_t inserted_id() const { return inserted_id_; }

private:
    Workspace* resolve(Document& document) const {
        for (auto& workspace : document.workspaces()) {
            if (workspace.get() == workspace_) {
                return workspace.get();
            }
        }
        return nullptr;
    }

    Workspace* workspace_;
    std::string label_;
    std::int32_t insert_index_;
    std::int32_t previous_next_id_ = 0;
    std::int32_t previous_active_id_ = -1;
    std::int32_t inserted_id_ = -1;
};

/// Remove a chunk. The removed chunk is kept (shared_ptr) so revert can put
/// the very same object back at its old index.
class RemoveChunkCommand : public StagedMutation {
public:
    explicit RemoveChunkCommand(std::int32_t chunk_id) : chunk_id_(chunk_id) {}

    std::string name() const override { return "Remove Chunk"; }

protected:
    void validate(Document& document) override {
        if (find_workspace(document, chunk_id_) == nullptr) {
            throw ValueError("no chunk with id " + std::to_string(chunk_id_));
        }
        const Workspace* workspace = find_workspace(document, chunk_id_);
        if (workspace->chunks().size() <= 1) {
            throw ValueError("remove chunk: a document must keep at least one chunk");
        }
    }

    void stage(Document& document) override {
        Workspace* workspace = find_workspace(document, chunk_id_);
        previous_active_id_ = document.active_chunk_id();
        workspace_ = workspace;
        removed_.reset();
        index_ = 0;
        const auto& chunks = workspace->chunks();
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            if (chunks[i]->id() == chunk_id_) {
                removed_ = chunks[i];
                index_ = i;
                break;
            }
        }
    }

    void commit(Document& document) override {
        if (removed_ == nullptr) {
            throw ValueError("no chunk with id " + std::to_string(chunk_id_));
        }
        std::shared_ptr<Workspace> workspace = workspace_shared(document);
        auto& chunks = workspace->chunks();
        chunks.erase(chunks.begin() + static_cast<std::ptrdiff_t>(index_));
        if (document.active_chunk_id() == chunk_id_) {
            document.set_active_chunk_id(chunks.empty() ? -1 : chunks.front()->id());
        }
    }

    void restore(Document& document) override {
        std::shared_ptr<Workspace> workspace = workspace_shared(document);
        if (workspace == nullptr || removed_ == nullptr) {
            throw ValueError("no chunk with id " + std::to_string(chunk_id_));
        }
        auto& chunks = workspace->chunks();
        if (index_ > chunks.size()) {
            index_ = chunks.size();
        }
        chunks.insert(chunks.begin() + static_cast<std::ptrdiff_t>(index_), removed_);
        document.set_active_chunk_id(previous_active_id_);
    }

private:
    std::shared_ptr<Workspace> workspace_shared(Document& document) const {
        for (auto& workspace : document.workspaces()) {
            if (workspace.get() == workspace_) {
                return workspace;
            }
        }
        return nullptr;
    }

    std::int32_t chunk_id_;
    Workspace* workspace_ = nullptr;
    std::shared_ptr<Chunk> removed_;
    std::size_t index_ = 0;
    std::int32_t previous_active_id_ = -1;
};

/// Duplicate a chunk: a deep copy (Chunk's copy constructor clones asset
/// payloads) with a freshly allocated chunk id and every chunk-local id
/// remapped by core's `duplicate_chunk`, so the copy's keypoints, matches,
/// tracks and asset links point at its own cameras and assets rather than the
/// source's (WP-11.6 extended this from a verbatim copy).
class DuplicateChunkCommand : public StagedMutation {
public:
    DuplicateChunkCommand(std::int32_t chunk_id, std::string new_label)
        : chunk_id_(chunk_id), new_label_(std::move(new_label)) {}

    std::string name() const override { return "Duplicate Chunk"; }

protected:
    void validate(Document& document) override {
        if (find_workspace(document, chunk_id_) == nullptr) {
            throw ValueError("no chunk with id " + std::to_string(chunk_id_));
        }
    }

    void stage(Document& document) override {
        Workspace* workspace = find_workspace(document, chunk_id_);
        workspace_ = workspace;
        previous_next_id_ = workspace->next_chunk_id();
        previous_active_id_ = document.active_chunk_id();
        created_id_ = -1;
    }

    void commit(Document& document) override {
        std::shared_ptr<Workspace> workspace = workspace_shared(document);
        const Chunk* source = workspace->find_chunk(chunk_id_);
        if (source == nullptr) {
            throw ValueError("no chunk with id " + std::to_string(chunk_id_));
        }
        // Revert re-applies this commit, so the copy must be reproducible
        // exactly: build it into a local first, then take ownership.
        auto copy = std::make_shared<Chunk>(
            duplicate_chunk(*source, workspace->next_chunk_id(), new_label_));
        created_id_ = copy->id();
        workspace->chunks().push_back(std::move(copy));
        // add_chunk-style counter advance: next_chunk_id() must stay ahead.
        workspace->set_next_chunk_id(created_id_ + 1);
        document.set_active_chunk_id(created_id_);
    }

    void restore(Document& document) override {
        std::shared_ptr<Workspace> workspace = workspace_shared(document);
        if (workspace == nullptr) {
            throw ValueError("no chunk with id " + std::to_string(chunk_id_));
        }
        auto& chunks = workspace->chunks();
        chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
                                    [this](const std::shared_ptr<Chunk>& chunk) {
                                        return chunk->id() == created_id_;
                                    }),
                     chunks.end());
        workspace->set_next_chunk_id(previous_next_id_);
        document.set_active_chunk_id(previous_active_id_);
    }

private:
    std::shared_ptr<Workspace> workspace_shared(Document& document) const {
        for (auto& workspace : document.workspaces()) {
            if (workspace.get() == workspace_) {
                return workspace;
            }
        }
        return nullptr;
    }

    std::int32_t chunk_id_;
    std::string new_label_;
    Workspace* workspace_ = nullptr;
    std::int32_t previous_next_id_ = 0;
    std::int32_t previous_active_id_ = -1;
    std::int32_t created_id_ = -1;
};

/// Reorder the workspace's chunks to `new_order`.
///
/// `new_order` must be a permutation of the current chunk ids. Staging keeps
/// the previous vector (shared_ptrs only, so payloads are not copied).
class ReorderChunksCommand : public StagedMutation {
public:
    explicit ReorderChunksCommand(std::vector<std::int32_t> new_order)
        : new_order_(std::move(new_order)) {}

    std::string name() const override { return "Reorder Chunks"; }

protected:
    void validate(Document& document) override {
        Workspace& workspace = document.workspace();
        const auto& chunks = workspace.chunks();
        if (new_order_.size() != chunks.size()) {
            throw ValueError("reorder chunks: expected " +
                             std::to_string(chunks.size()) + " ids, got " +
                             std::to_string(new_order_.size()));
        }
        std::vector<std::int32_t> sorted = new_order_;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw ValueError("reorder chunks: the order repeats a chunk id");
        }
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            if (workspace.find_chunk(sorted[i]) == nullptr) {
                throw ValueError("reorder chunks: no chunk with id " +
                                 std::to_string(sorted[i]));
            }
        }
    }

    void stage(Document& document) override {
        previous_order_ = document.workspace().chunks();
    }

    void commit(Document& document) override {
        auto reordered = std::make_shared<std::vector<std::shared_ptr<Chunk>>>();
        reordered->reserve(new_order_.size());
        for (const std::int32_t id : new_order_) {
            Chunk* chunk = document.workspace().find_chunk(id);
            if (chunk == nullptr) {
                throw ValueError("reorder chunks: no chunk with id " +
                                 std::to_string(id));
            }
            for (const auto& candidate : document.workspace().chunks()) {
                if (candidate.get() == chunk) {
                    reordered->push_back(candidate);
                    break;
                }
            }
        }
        document.workspace().chunks() = *reordered;
    }

    void restore(Document& document) override {
        document.workspace().chunks() = previous_order_;
    }

private:
    std::vector<std::int32_t> new_order_;
    std::vector<std::shared_ptr<Chunk>> previous_order_;
};

/// Set a chunk's local->world transform and region in one undoable step.
class SetChunkTransformCommand : public StagedMutation {
public:
    SetChunkTransformCommand(std::int32_t chunk_id, Transform transform, Region region)
        : chunk_id_(chunk_id), transform_(std::move(transform)), region_(region) {}

    std::string name() const override { return "Set Chunk Transform"; }

protected:
    void validate(Document& document) override {
        require_chunk(document, chunk_id_);
        if (!transform_.allFinite() || !region_.centre.allFinite() ||
            !region_.size.allFinite()) {
            throw ValueError("set chunk transform: non-finite transform or region");
        }
        if ((region_.size.array() <= 0.0).any()) {
            throw ValueError("set chunk transform: region size must be positive");
        }
    }

    void stage(Document& document) override {
        Chunk& chunk = target_chunk(document);
        previous_transform_ = chunk.transform();
        previous_region_ = chunk.region();
    }

    void commit(Document& document) override {
        Chunk& chunk = target_chunk(document);
        chunk.transform() = transform_;
        chunk.region() = region_;
    }

    void restore(Document& document) override {
        Chunk& chunk = target_chunk(document);
        chunk.transform() = previous_transform_;
        chunk.region() = previous_region_;
    }

private:
    Chunk& target_chunk(Document& document) const {
        return require_chunk(document, chunk_id_);
    }

    std::int32_t chunk_id_;
    Transform transform_;
    Region region_;
    Transform previous_transform_ = Transform::Identity();
    Region previous_region_;
};

/// Install a fully computed chunk geometry state in one undoable step
/// (AlignChunks, §4.8). The caller (the task) owns the estimation; this
/// command owns the swap: it captures the pre-state in stage() and installs
/// the post-state in commit(), so revert restores exactly the pre-state and a
/// redo reproduces exactly the first apply.
class AlignChunksCommand : public StagedMutation {
public:
    AlignChunksCommand(std::int32_t chunk_id, ChunkGeometry after)
        : chunk_id_(chunk_id), after_(std::move(after)) {}

    std::string name() const override { return "Align Chunks"; }

    std::size_t retained_bytes() const override {
        return payload_bytes(after_) + payload_bytes(previous_);
    }

protected:
    void validate(Document& document) override {
        const Chunk& chunk = require_chunk(document, chunk_id_);
        // Structural shape must match; apply_geometry() repeats the check, but
        // failing here keeps a malformed command from ever staging.
        if (after_.camera_poses.size() != chunk.cameras().size() ||
            after_.track_points.size() != chunk.tie_points().tracks.size() ||
            after_.marker_positions.size() != chunk.markers().size() ||
            after_.point_clouds.size() != chunk.point_clouds().size() ||
            after_.models.size() != chunk.models().size()) {
            throw ValueError("align chunks: geometry snapshot does not match chunk " +
                             std::to_string(chunk_id_));
        }
        if (!after_.transform.allFinite() || !after_.region.centre.allFinite() ||
            !after_.region.size.allFinite() || (after_.region.size.array() <= 0.0).any()) {
            throw ValueError("align chunks: non-finite transform or region");
        }
    }

    void stage(Document& document) override {
        previous_ = capture_geometry(require_chunk(document, chunk_id_));
    }

    void commit(Document& document) override {
        apply_geometry(require_chunk(document, chunk_id_), after_);
    }

    void restore(Document& document) override {
        apply_geometry(require_chunk(document, chunk_id_), previous_);
    }

private:
    std::int32_t chunk_id_;
    ChunkGeometry after_;
    ChunkGeometry previous_;
};

/// Set one meta property, restoring the previous value (or absence) on revert.
class SetChunkMetaCommand : public StagedMutation {
public:
    SetChunkMetaCommand(std::int32_t chunk_id, std::string key, std::string value)
        : chunk_id_(chunk_id), key_(std::move(key)), value_(std::move(value)) {}

    std::string name() const override { return "Set Chunk Metadata"; }

protected:
    void validate(Document& document) override {
        require_chunk(document, chunk_id_);
        if (key_.empty()) {
            throw ValueError("set chunk metadata: empty key");
        }
    }

    void stage(Document& document) override {
        // Whole list, in order: Meta equality is vector equality, so restoring
        // the value in a different position would not round-trip.
        previous_items_ = require_chunk(document, chunk_id_).meta().items;
    }

    void commit(Document& document) override {
        require_chunk(document, chunk_id_).meta().set(key_, value_);
    }

    void restore(Document& document) override {
        require_chunk(document, chunk_id_).meta().items = previous_items_;
    }

private:
    std::int32_t chunk_id_;
    std::string key_;
    std::string value_;
    std::vector<std::pair<std::string, std::string>> previous_items_;
};

// --- Marker commands -------------------------------------------------------

/// Allocate a marker id from the chunk counter when the caller left it at -1.
std::int32_t marker_id_for(Chunk& chunk, const Marker& marker) {
    return marker.id >= 0 ? marker.id : chunk.allocate_id();
}

class AddMarkerCommand : public ChunkContainersCommand {
public:
    AddMarkerCommand(std::int32_t chunk_id, Marker marker)
        : ChunkContainersCommand(chunk_id), marker_(std::move(marker)) {}

    std::string name() const override { return "Add Marker"; }

protected:
    void validate(Document& document) override {
        Chunk& chunk = target_chunk(document);
        if (marker_.id >= 0 && find_by_id(chunk.markers(), marker_.id) != nullptr) {
            throw ValueError("add marker: duplicate marker id " +
                             std::to_string(marker_.id));
        }
        if (marker_.id >= chunk.next_id()) {
            throw ValueError("add marker: id " + std::to_string(marker_.id) +
                             " is not below next_id " +
                             std::to_string(chunk.next_id()));
        }
        // validate() must not mutate: normalise a copy to run the same checks
        // commit() will, so a duplicate camera fails before anything is staged.
        std::vector<MarkerProjection> projections = marker_.projections;
        normalize_projections(projections, marker_.id);
        for (const auto& projection : projections) {
            require_camera(chunk, projection.camera_id, "add marker");
        }
        if (marker_.reference.has_value() &&
            !positive_finite(marker_.reference_accuracy)) {
            throw ValueError("add marker: reference accuracy must be finite and positive");
        }
        if (marker_.reference.has_value() &&
            !marker_.reference->allFinite()) {
            throw ValueError("add marker: reference is not finite");
        }
        if (marker_.group_id >= 0 &&
            find_by_id(chunk.marker_groups(), marker_.group_id) == nullptr) {
            throw ValueError("add marker: no marker group with id " +
                             std::to_string(marker_.group_id));
        }
    }

    void commit(Document& document) override {
        Chunk& chunk = target_chunk(document);
        Marker marker = marker_;
        marker.id = marker_id_for(chunk, marker);
        normalize_projections(marker.projections, marker.id);
        insert_by_id(chunk.markers(), std::move(marker));
    }

private:
    Marker marker_;
};

class RemoveMarkerCommand : public ChunkContainersCommand {
public:
    RemoveMarkerCommand(std::int32_t chunk_id, std::int32_t marker_id)
        : ChunkContainersCommand(chunk_id), marker_id_(marker_id) {}

    std::string name() const override { return "Remove Marker"; }

protected:
    void validate(Document& document) override {
        Chunk& chunk = target_chunk(document);
        if (find_by_id(chunk.markers(), marker_id_) == nullptr) {
            throw ValueError("no marker with id " + std::to_string(marker_id_));
        }
        // A scale bar anchored at this marker would dangle after removal.
        for (const Scalebar& bar : chunk.scalebars()) {
            const bool uses_marker =
                (bar.from.kind == ScalebarEndpoint::Kind::Marker &&
                 bar.from.id == marker_id_) ||
                (bar.to.kind == ScalebarEndpoint::Kind::Marker &&
                 bar.to.id == marker_id_);
            if (uses_marker) {
                throw ValueError("remove marker: scalebar " +
                                 std::to_string(bar.id) + " references it");
            }
        }
    }

    void commit(Document& document) override {
        Chunk& chunk = target_chunk(document);
        if (!erase_by_id(chunk.markers(), marker_id_)) {
            throw ValueError("no marker with id " + std::to_string(marker_id_));
        }
    }

private:
    std::int32_t marker_id_;
};

class SetMarkerReferenceCommand : public ChunkContainersCommand {
public:
    SetMarkerReferenceCommand(std::int32_t chunk_id, std::int32_t marker_id,
                              std::optional<Eigen::Vector3d> reference,
                              Eigen::Vector3d accuracy)
        : ChunkContainersCommand(chunk_id),
          marker_id_(marker_id),
          reference_(reference),
          accuracy_(accuracy) {}

    std::string name() const override { return "Set Marker Reference"; }

protected:
    void validate(Document& document) override {
        Chunk& chunk = target_chunk(document);
        if (find_by_id(chunk.markers(), marker_id_) == nullptr) {
            throw ValueError("no marker with id " + std::to_string(marker_id_));
        }
        if (!positive_finite(accuracy_)) {
            throw ValueError("set marker reference: accuracy must be finite and positive");
        }
        if (reference_.has_value() && !reference_->allFinite()) {
            throw ValueError("set marker reference: reference is not finite");
        }
    }

    void commit(Document& document) override {
        Chunk& chunk = target_chunk(document);
        Marker* marker = find_by_id(chunk.markers(), marker_id_);
        if (marker == nullptr) {
            throw ValueError("no marker with id " + std::to_string(marker_id_));
        }
        marker->reference = reference_;
        marker->reference_accuracy = accuracy_;
    }

private:
    std::int32_t marker_id_;
    std::optional<Eigen::Vector3d> reference_;
    Eigen::Vector3d accuracy_;
};

class MoveMarkerProjectionCommand : public ChunkContainersCommand {
public:
    MoveMarkerProjectionCommand(std::int32_t chunk_id, std::int32_t marker_id,
                                std::int32_t camera_id, Eigen::Vector2d pixel,
                                bool create)
        : ChunkContainersCommand(chunk_id),
          marker_id_(marker_id),
          camera_id_(camera_id),
          pixel_(pixel),
          create_(create) {}

    std::string name() const override { return "Move Marker Projection"; }

protected:
    void validate(Document& document) override {
        Chunk& chunk = target_chunk(document);
        if (find_by_id(chunk.markers(), marker_id_) == nullptr) {
            throw ValueError("no marker with id " + std::to_string(marker_id_));
        }
        require_camera(chunk, camera_id_, "move marker projection");
        if (!pixel_.allFinite()) {
            throw ValueError("move marker projection: pixel is not finite");
        }
        const Marker* marker = find_by_id(chunk.markers(), marker_id_);
        const MarkerProjection* existing = nullptr;
        for (const auto& projection : marker->projections) {
            if (projection.camera_id == camera_id_) {
                existing = &projection;
                break;
            }
        }
        if (!create_ && existing == nullptr) {
            throw ValueError("no projection of marker " +
                             std::to_string(marker_id_) + " in camera " +
                             std::to_string(camera_id_));
        }
    }

    void commit(Document& document) override {
        Chunk& chunk = target_chunk(document);
        Marker* marker = find_by_id(chunk.markers(), marker_id_);
        if (marker == nullptr) {
            throw ValueError("no marker with id " + std::to_string(marker_id_));
        }
        for (auto& projection : marker->projections) {
            if (projection.camera_id == camera_id_) {
                projection.pixel = pixel_;
                return;
            }
        }
        if (!create_) {
            throw ValueError("no projection of marker " +
                             std::to_string(marker_id_) + " in camera " +
                             std::to_string(camera_id_));
        }
        MarkerProjection projection;
        projection.camera_id = camera_id_;
        projection.pixel = pixel_;
        insert_projection(marker->projections, projection);
    }

private:
    std::int32_t marker_id_;
    std::int32_t camera_id_;
    Eigen::Vector2d pixel_;
    bool create_;
};

class RemoveMarkerProjectionCommand : public ChunkContainersCommand {
public:
    RemoveMarkerProjectionCommand(std::int32_t chunk_id, std::int32_t marker_id,
                                  std::int32_t camera_id)
        : ChunkContainersCommand(chunk_id),
          marker_id_(marker_id),
          camera_id_(camera_id) {}

    std::string name() const override { return "Remove Marker Projection"; }

protected:
    void validate(Document& document) override {
        Chunk& chunk = target_chunk(document);
        const Marker* marker = find_by_id(chunk.markers(), marker_id_);
        if (marker == nullptr) {
            throw ValueError("no marker with id " + std::to_string(marker_id_));
        }
        for (const auto& projection : marker->projections) {
            if (projection.camera_id == camera_id_) {
                return;
            }
        }
        throw ValueError("no projection of marker " +
                         std::to_string(marker_id_) + " in camera " +
                         std::to_string(camera_id_));
    }

    void commit(Document& document) override {
        Chunk& chunk = target_chunk(document);
        Marker* marker = find_by_id(chunk.markers(), marker_id_);
        if (marker == nullptr) {
            throw ValueError("no marker with id " + std::to_string(marker_id_));
        }
        auto& projections = marker->projections;
        const auto it = std::remove_if(
            projections.begin(), projections.end(),
            [this](const MarkerProjection& projection) {
                return projection.camera_id == camera_id_;
            });
        if (it == projections.end()) {
            throw ValueError("no projection of marker " +
                             std::to_string(marker_id_) + " in camera " +
                             std::to_string(camera_id_));
        }
        projections.erase(it, projections.end());
    }

private:
    std::int32_t marker_id_;
    std::int32_t camera_id_;
};

// --- Scale-bar commands ----------------------------------------------------

class AddScalebarCommand : public ChunkContainersCommand {
public:
    AddScalebarCommand(std::int32_t chunk_id, Scalebar scalebar)
        : ChunkContainersCommand(chunk_id), scalebar_(std::move(scalebar)) {}

    std::string name() const override { return "Add Scale Bar"; }

protected:
    void validate(Document& document) override {
        Chunk& chunk = target_chunk(document);
        if (scalebar_.id >= 0 &&
            find_by_id(chunk.scalebars(), scalebar_.id) != nullptr) {
            throw ValueError("add scalebar: duplicate scalebar id " +
                             std::to_string(scalebar_.id));
        }
        if (scalebar_.id >= chunk.next_id()) {
            throw ValueError("add scalebar: id " + std::to_string(scalebar_.id) +
                             " is not below next_id " +
                             std::to_string(chunk.next_id()));
        }
        check_scalebar_endpoint(chunk, scalebar_.from, scalebar_.id);
        check_scalebar_endpoint(chunk, scalebar_.to, scalebar_.id);
        check_scalebar_length(scalebar_.reference_length_m,
                              scalebar_.reference_accuracy_m, scalebar_.id);
    }

    void commit(Document& document) override {
        Chunk& chunk = target_chunk(document);
        Scalebar bar = scalebar_;
        if (bar.id < 0) {
            bar.id = chunk.allocate_id();
        }
        insert_by_id(chunk.scalebars(), std::move(bar));
    }

private:
    Scalebar scalebar_;
};

class RemoveScalebarCommand : public ChunkContainersCommand {
public:
    RemoveScalebarCommand(std::int32_t chunk_id, std::int32_t scalebar_id)
        : ChunkContainersCommand(chunk_id), scalebar_id_(scalebar_id) {}

    std::string name() const override { return "Remove Scale Bar"; }

protected:
    void validate(Document& document) override {
        Chunk& chunk = target_chunk(document);
        if (find_by_id(chunk.scalebars(), scalebar_id_) == nullptr) {
            throw ValueError("no scalebar with id " + std::to_string(scalebar_id_));
        }
    }

    void commit(Document& document) override {
        Chunk& chunk = target_chunk(document);
        if (!erase_by_id(chunk.scalebars(), scalebar_id_)) {
            throw ValueError("no scalebar with id " + std::to_string(scalebar_id_));
        }
    }

private:
    std::int32_t scalebar_id_;
};

class SetScalebarReferenceCommand : public ChunkContainersCommand {
public:
    SetScalebarReferenceCommand(std::int32_t chunk_id, std::int32_t scalebar_id,
                                double length_m, double accuracy_m)
        : ChunkContainersCommand(chunk_id),
          scalebar_id_(scalebar_id),
          length_m_(length_m),
          accuracy_m_(accuracy_m) {}

    std::string name() const override { return "Set Scale Bar Reference"; }

protected:
    void validate(Document& document) override {
        Chunk& chunk = target_chunk(document);
        if (find_by_id(chunk.scalebars(), scalebar_id_) == nullptr) {
            throw ValueError("no scalebar with id " + std::to_string(scalebar_id_));
        }
        check_scalebar_length(length_m_, accuracy_m_, scalebar_id_);
    }

    void commit(Document& document) override {
        Chunk& chunk = target_chunk(document);
        Scalebar* bar = find_by_id(chunk.scalebars(), scalebar_id_);
        if (bar == nullptr) {
            throw ValueError("no scalebar with id " + std::to_string(scalebar_id_));
        }
        bar->reference_length_m = length_m_;
        bar->reference_accuracy_m = accuracy_m_;
    }

private:
    std::int32_t scalebar_id_;
    double length_m_;
    double accuracy_m_;
};

// --- Group commands --------------------------------------------------------

/// Marker groups live in `Chunk::marker_groups()` and are back-referenced by
/// `Marker::group_id`. Camera groups live in `Chunk::camera_groups()` and
/// carry their members only (Camera has no group field yet).
void require_supported_kind(GroupKind kind) {
    switch (kind) {
        case GroupKind::Marker:
        case GroupKind::Camera:
            return;
        case GroupKind::PointCloud:
        case GroupKind::Model:
            break;
    }
    // Point-cloud and model groups have no container on Chunk yet (the plan
    // declares PointCloudGroup/ModelGroup but no storage); see LIMITATIONS.md.
    throw ValueError("group commands support marker and camera groups only");
}

const char* group_kind_name(GroupKind kind) {
    switch (kind) {
        case GroupKind::Marker: return "marker";
        case GroupKind::PointCloud: return "point cloud";
        case GroupKind::Model: return "model";
        case GroupKind::Camera: return "camera";
    }
    return "group";
}

/// Insert `id` into a sorted, unique member list.
void add_member(std::vector<std::int32_t>& members, std::int32_t id) {
    auto it = std::lower_bound(members.begin(), members.end(), id);
    if (it == members.end() || *it != id) {
        members.insert(it, id);
    }
}

void drop_member(std::vector<std::int32_t>& members, std::int32_t id) {
    auto it = std::remove(members.begin(), members.end(), id);
    members.erase(it, members.end());
}

class AddGroupCommand : public ChunkContainersCommand {
public:
    AddGroupCommand(GroupKind kind, std::int32_t chunk_id, std::string label)
        : ChunkContainersCommand(chunk_id), kind_(kind), label_(std::move(label)) {}

    std::string name() const override { return "Add Group"; }

protected:
    void validate(Document& document) override {
        require_supported_kind(kind_);
        (void)target_chunk(document);
    }

    void commit(Document& document) override {
        Chunk& chunk = target_chunk(document);
        switch (kind_) {
            case GroupKind::Marker: {
                MarkerGroup group;
                group.id = chunk.allocate_id();
                group.label = label_.empty() ? "Marker Group" : label_;
                insert_by_id(chunk.marker_groups(), std::move(group));
                return;
            }
            case GroupKind::Camera: {
                CameraGroup group;
                group.id = chunk.allocate_id();
                group.label = label_.empty() ? "Camera Group" : label_;
                insert_by_id(chunk.camera_groups(), std::move(group));
                return;
            }
            case GroupKind::PointCloud:
            case GroupKind::Model:
                break;
        }
        throw ValueError("group commands support marker and camera groups only");
    }

private:
    GroupKind kind_;
    std::string label_;
};

class RemoveGroupCommand : public ChunkContainersCommand {
public:
    RemoveGroupCommand(GroupKind kind, std::int32_t chunk_id, std::int32_t group_id)
        : ChunkContainersCommand(chunk_id), kind_(kind), group_id_(group_id) {}

    std::string name() const override { return "Remove Group"; }

protected:
    void validate(Document& document) override {
        require_supported_kind(kind_);
        Chunk& chunk = target_chunk(document);
        const bool found =
            kind_ == GroupKind::Marker
                ? find_by_id(chunk.marker_groups(), group_id_) != nullptr
                : find_by_id(chunk.camera_groups(), group_id_) != nullptr;
        if (!found) {
            throw ValueError(std::string("no ") + group_kind_name(kind_) +
                             " group with id " + std::to_string(group_id_));
        }
    }

    void commit(Document& document) override {
        Chunk& chunk = target_chunk(document);
        if (kind_ == GroupKind::Marker) {
            if (!erase_by_id(chunk.marker_groups(), group_id_)) {
                throw ValueError("no marker group with id " +
                                 std::to_string(group_id_));
            }
            // Members keep existing but must not point at a removed group.
            for (Marker& marker : chunk.markers()) {
                if (marker.group_id == group_id_) {
                    marker.group_id = -1;
                }
            }
            return;
        }
        if (!erase_by_id(chunk.camera_groups(), group_id_)) {
            throw ValueError("no camera group with id " +
                             std::to_string(group_id_));
        }
    }

private:
    GroupKind kind_;
    std::int32_t group_id_;
};

/// Move objects into `group_id` (-1 = remove from every group).
class MoveToGroupCommand : public ChunkContainersCommand {
public:
    MoveToGroupCommand(GroupKind kind, std::int32_t chunk_id,
                       std::vector<std::int32_t> ids, std::int32_t group_id)
        : ChunkContainersCommand(chunk_id),
          kind_(kind),
          ids_(std::move(ids)),
          group_id_(group_id) {}

    std::string name() const override { return "Move To Group"; }

protected:
    void validate(Document& document) override {
        require_supported_kind(kind_);
        Chunk& chunk = target_chunk(document);
        if (group_id_ >= 0) {
            const bool found =
                kind_ == GroupKind::Marker
                    ? find_by_id(chunk.marker_groups(), group_id_) != nullptr
                    : find_by_id(chunk.camera_groups(), group_id_) != nullptr;
            if (!found) {
                throw ValueError(std::string("no ") + group_kind_name(kind_) +
                                 " group with id " + std::to_string(group_id_));
            }
        }
        for (const std::int32_t id : ids_) {
            const bool exists =
                kind_ == GroupKind::Marker ? find_by_id(chunk.markers(), id) != nullptr
                                           : find_camera(chunk, id) != nullptr;
            if (!exists) {
                throw ValueError(std::string("move to group: no ") +
                                 group_kind_name(kind_) + " with id " +
                                 std::to_string(id));
            }
        }
    }

    void commit(Document& document) override {
        Chunk& chunk = target_chunk(document);
        if (kind_ == GroupKind::Marker) {
            for (MarkerGroup& group : chunk.marker_groups()) {
                for (const std::int32_t id : ids_) {
                    drop_member(group.members, id);
                }
            }
            MarkerGroup* target =
                group_id_ >= 0 ? find_by_id(chunk.marker_groups(), group_id_) : nullptr;
            for (const std::int32_t id : ids_) {
                Marker* marker = find_by_id(chunk.markers(), id);
                if (marker == nullptr) {
                    throw ValueError("move to group: no marker with id " +
                                     std::to_string(id));
                }
                marker->group_id = target != nullptr ? target->id : -1;
                if (target != nullptr) {
                    add_member(target->members, id);
                }
            }
            return;
        }
        for (CameraGroup& group : chunk.camera_groups()) {
            for (const std::int32_t id : ids_) {
                drop_member(group.members, id);
            }
        }
        if (group_id_ < 0) {
            return;
        }
        CameraGroup* target = find_by_id(chunk.camera_groups(), group_id_);
        if (target == nullptr) {
            throw ValueError("no camera group with id " + std::to_string(group_id_));
        }
        for (const std::int32_t id : ids_) {
            add_member(target->members, id);
        }
    }

private:
    GroupKind kind_;
    std::vector<std::int32_t> ids_;
    std::int32_t group_id_;
};

// --- Camera commands -------------------------------------------------------

/// Set a flag ("enabled" or "selected") on a set of cameras in one step.
class SetCameraFlagCommand : public ChunkContainersCommand {
public:
    enum class Flag { Enabled, Selected };

    SetCameraFlagCommand(std::int32_t chunk_id, std::vector<std::int32_t> camera_ids,
                         Flag flag, bool value)
        : ChunkContainersCommand(chunk_id),
          camera_ids_(std::move(camera_ids)),
          flag_(flag),
          value_(value) {}

    std::string name() const override {
        return flag_ == Flag::Enabled ? "Set Camera Enabled"
                                      : "Set Camera Selected";
    }

protected:
    void validate(Document& document) override {
        Chunk& chunk = target_chunk(document);
        for (const std::int32_t id : camera_ids_) {
            require_camera(chunk, id, "camera flag");
        }
    }

    void commit(Document& document) override {
        Chunk& chunk = target_chunk(document);
        for (const std::int32_t id : camera_ids_) {
            Camera* camera = nullptr;
            for (Camera& candidate : chunk.cameras()) {
                if (candidate.id == id) {
                    camera = &candidate;
                    break;
                }
            }
            if (camera == nullptr) {
                throw ValueError("camera flag: no camera with id " +
                                 std::to_string(id));
            }
            if (flag_ == Flag::Enabled) {
                camera->enabled = value_;
            } else {
                camera->selected = value_;
            }
        }
    }

private:
    std::vector<std::int32_t> camera_ids_;
    Flag flag_;
    bool value_;
};

/// Append every chunk of `source` to the target document.
///
/// Incoming chunk ids are offset by the target's next_chunk_id() (§4.8); every
/// chunk-local id is preserved, because ids only need to be unique inside a
/// chunk. The chunks are deep copies, so later edits to the source never show
/// up in the target (and vice versa).
class AppendDocumentCommand : public StagedMutation {
public:
    AppendDocumentCommand(std::shared_ptr<Document> source, std::string source_path)
        : source_(std::move(source)), source_path_(std::move(source_path)) {}

    std::string name() const override { return "Append Document"; }

protected:
    void validate(Document& /*document*/) override {
        const std::string origin = describe_source();
        if (source_ == nullptr) {
            throw ValueError("append document " + origin +
                             ": source document is null");
        }
        for (const auto& chunk : source_->workspace().chunks()) {
            if (chunk == nullptr) {
                throw ValueError("append document " + origin +
                                 ": source holds a null chunk");
            }
        }
    }

    void stage(Document& document) override {
        Workspace& workspace = document.workspace();
        appended_count_ = 0;
        previous_chunks_ = workspace.chunks().size();
        previous_next_id_ = workspace.next_chunk_id();
        previous_active_id_ = document.active_chunk_id();
    }

    void commit(Document& document) override {
        Workspace& workspace = document.workspace();
        for (const auto& chunk : source_->workspace().chunks()) {
            auto copy = std::make_shared<Chunk>(*chunk);
            copy->set_id(workspace.next_chunk_id());
            workspace.chunks().push_back(std::move(copy));
            // restore_chunk-style counter advance.
            workspace.set_next_chunk_id(workspace.next_chunk_id() + 1);
            ++appended_count_;
        }
    }

    void restore(Document& document) override {
        Workspace& workspace = document.workspace();
        auto& chunks = workspace.chunks();
        while (appended_count_ > 0 && chunks.size() > previous_chunks_) {
            chunks.pop_back();
            --appended_count_;
        }
        workspace.set_next_chunk_id(previous_next_id_);
        document.set_active_chunk_id(previous_active_id_);
    }

private:
    std::string describe_source() const {
        return source_path_.empty() ? std::string("(in memory)")
                                    : "'" + source_path_ + "'";
    }

    std::shared_ptr<Document> source_;
    std::string source_path_;
    std::size_t previous_chunks_ = 0;
    std::size_t appended_count_ = 0;
    std::int32_t previous_next_id_ = 0;
    std::int32_t previous_active_id_ = -1;
};

// --- Asset commands (WP-11.5c) ---------------------------------------------
//
// A spill file name is pid + a process-wide counter: two captures in the same
// instant must not collide, so a clock is deliberately never used.
namespace {
std::string make_spill_path(const std::string& cache_dir) {
    static std::atomic<std::uint64_t> counter{0};
    const std::uint64_t n = counter.fetch_add(1, std::memory_order_relaxed);
    std::error_code ec;
    std::filesystem::path dir(cache_dir);
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        throw Error("cannot create undo spill directory '" + cache_dir +
                    "': " + ec.message());
    }
    return (dir / ("undo-model-" +
                   std::to_string(static_cast<long long>(ALPHASHAPE_GETPID())) +
                   "-" + std::to_string(static_cast<unsigned long long>(n)) +
                   ".bin"))
        .string();
}
}  // namespace
//
// Asset payloads are shared_ptrs, so a snapshot keeps the very same slot object
// instead of copying it. Only a *displaced* slot has to be retained between
// apply and revert; that is what SlotHolder does, spilling above
// kUndoAssetSpillBytes to the cache directory (§4.7, WP-11.5a policy).

/// "<Kind> <n>" with the smallest unused n. Mirrors chunk.cpp's helper, which
/// is internal to that translation unit.
template <class Slot>
std::string default_asset_label(const std::vector<Slot>& existing, const char* kind) {
    for (int n = 1;; ++n) {
        const std::string candidate = std::string(kind) + " " + std::to_string(n);
        bool taken = false;
        for (const Slot& slot : existing) {
            if (slot.label == candidate) {
                taken = true;
                break;
            }
        }
        if (!taken) {
            return candidate;
        }
    }
}

/// "<Kind> <n>" with the smallest unused n. Mirrors chunk.cpp's helper.
template <class Slot>
std::string default_asset_label(const std::vector<Slot>& existing, const char* kind);

/// One asset container, with its active-id accessors and its add operation.
/// Specialised per kind below; Panorama is the exception (a single optional
/// slot with no active id) and has its own commands.
template <class Slot>
struct AssetSlots;

template <>
struct AssetSlots<PointCloudAsset> {
    static constexpr const char* kind_name = "point cloud";
    static constexpr const char* add_name = "Add Point Cloud";
    static constexpr const char* remove_name = "Remove Point Cloud";
    static std::vector<PointCloudAsset>& get(Chunk& chunk) {
        return chunk.point_clouds();
    }
    static std::int32_t active(const Chunk& chunk) {
        return chunk.active_point_cloud_id();
    }
    static void set_active(Chunk& chunk, std::int32_t id) {
        chunk.set_active_point_cloud_id(id);
    }
    static PointCloudAsset& add(Chunk& chunk, std::string label) {
        return chunk.add_point_cloud(std::move(label));
    }
};

template <>
struct AssetSlots<ModelAsset> {
    static constexpr const char* kind_name = "model";
    static constexpr const char* add_name = "Add Model";
    static constexpr const char* remove_name = "Remove Model";
    static std::vector<ModelAsset>& get(Chunk& chunk) { return chunk.models(); }
    static std::int32_t active(const Chunk& chunk) {
        return chunk.active_model_id();
    }
    static void set_active(Chunk& chunk, std::int32_t id) {
        chunk.set_active_model_id(id);
    }
    static ModelAsset& add(Chunk& chunk, std::string label) {
        return chunk.add_model(std::move(label));
    }
};

template <>
struct AssetSlots<DepthMapsAsset> {
    static constexpr const char* kind_name = "depth-map set";
    static constexpr const char* add_name = "Add Depth Maps";
    static constexpr const char* remove_name = "Remove Depth Maps";
    static std::vector<DepthMapsAsset>& get(Chunk& chunk) {
        return chunk.depth_map_sets();
    }
    static std::int32_t active(const Chunk& chunk) {
        return chunk.active_depth_maps_id();
    }
    static void set_active(Chunk& chunk, std::int32_t id) {
        chunk.set_active_depth_maps_id(id);
    }
    static DepthMapsAsset& add(Chunk& chunk, std::string label) {
        return chunk.add_depth_maps(std::move(label));
    }
};

template <>
struct AssetSlots<ElevationAsset> {
    static constexpr const char* kind_name = "elevation";
    static constexpr const char* add_name = "Add Elevation";
    static constexpr const char* remove_name = "Remove Elevation";
    static std::vector<ElevationAsset>& get(Chunk& chunk) {
        return chunk.elevations();
    }
    static std::int32_t active(const Chunk& chunk) {
        return chunk.active_elevation_id();
    }
    static void set_active(Chunk& chunk, std::int32_t id) {
        chunk.set_active_elevation_id(id);
    }
    static ElevationAsset& add(Chunk& chunk, std::string label) {
        return chunk.add_elevation(std::move(label));
    }
};

template <>
struct AssetSlots<OrthomosaicAsset> {
    static constexpr const char* kind_name = "orthomosaic";
    static constexpr const char* add_name = "Add Orthomosaic";
    static constexpr const char* remove_name = "Remove Orthomosaic";
    static std::vector<OrthomosaicAsset>& get(Chunk& chunk) {
        return chunk.orthomosaics();
    }
    static std::int32_t active(const Chunk& chunk) {
        return chunk.active_orthomosaic_id();
    }
    static void set_active(Chunk& chunk, std::int32_t id) {
        chunk.set_active_orthomosaic_id(id);
    }
    static OrthomosaicAsset& add(Chunk& chunk, std::string label) {
        return chunk.add_orthomosaic(std::move(label));
    }
};

template <>
struct AssetSlots<TiledModelAsset> {
    static constexpr const char* kind_name = "tiled model";
    static constexpr const char* add_name = "Add Tiled Model";
    static constexpr const char* remove_name = "Remove Tiled Model";
    static std::vector<TiledModelAsset>& get(Chunk& chunk) {
        return chunk.tiled_models();
    }
    static std::int32_t active(const Chunk& chunk) {
        return chunk.active_tiled_model_id();
    }
    static void set_active(Chunk& chunk, std::int32_t id) {
        chunk.set_active_tiled_model_id(id);
    }
    /// Chunk has no add_tiled_model(): allocate, label and insert, activating.
    static TiledModelAsset& add(Chunk& chunk, std::string label) {
        TiledModelAsset slot;
        slot.id = chunk.allocate_id();
        slot.label = label.empty()
                         ? default_asset_label(chunk.tiled_models(), "Tiled Model")
                         : std::move(label);
        auto& slots = chunk.tiled_models();
        auto it = slots.begin();
        while (it != slots.end() && it->id < slot.id) {
            ++it;
        }
        TiledModelAsset& inserted = *slots.insert(it, std::move(slot));
        chunk.set_active_tiled_model_id(inserted.id);
        return inserted;
    }
};


template <class Slot>
Slot* find_slot(std::vector<Slot>& slots, std::int32_t id) {
    return find_by_id(slots, id);
}

// --- asset payload codecs ---------------------------------------------------
//
// A spilled asset must come back *exactly* as it went out, because the undo
// contract is "revert restores Chunk::operator==". The payload encoders below
// therefore reuse the project's own sidecar encoders (the same ones doc.xml v2
// writes) and wrap them with the fields those encoders do not carry (an
// elevation's source, an orthomosaic's labels/surface, a model's textures).

/// Wrap a self-describing sidecar blob with its byte count.
void put_blob(std::vector<std::uint8_t>& out, const std::vector<std::uint8_t>& blob) {
    docxml::sidecar::put_u32(out, static_cast<std::uint32_t>(blob.size()));
    out.insert(out.end(), blob.begin(), blob.end());
}

std::vector<std::uint8_t> get_blob(docxml::sidecar::ByteReader& reader) {
    const std::uint32_t count = reader.u32();
    std::vector<std::uint8_t> blob;
    reader.u8s(count, blob);
    return blob;
}

/// An 8-bit image as width, height, channels and raw pixel bytes. Raw, not
/// PNG: spilling happens on the undo path where wall-clock time is user-visible
/// and the bytes are scratch (never written to a project).
void put_image(std::vector<std::uint8_t>& out, const Image& image) {
    docxml::sidecar::put_u32(out, static_cast<std::uint32_t>(image.width()));
    docxml::sidecar::put_u32(out, static_cast<std::uint32_t>(image.height()));
    docxml::sidecar::put_u32(out, static_cast<std::uint32_t>(image.channels()));
    out.insert(out.end(), image.data(), image.data() + image.size_bytes());
}

Image get_image(docxml::sidecar::ByteReader& reader) {
    const int width = static_cast<int>(reader.u32());
    const int height = static_cast<int>(reader.u32());
    const int channels = static_cast<int>(reader.u32());
    if (width < 0 || height < 0 || (channels != 1 && channels != 3)) {
        throw ProjectIOError("undo image payload: bad header");
    }
    Image image(width, height, channels);
    std::vector<std::uint8_t> bytes;
    reader.u8s(image.size_bytes(), bytes);
    if (!bytes.empty()) {
        std::memcpy(image.data(), bytes.data(), bytes.size());
    }
    return image;
}

template <class Payload>
struct PayloadCodec;

template <>
struct PayloadCodec<PointCloud> {
    static std::size_t bytes(const PointCloud& cloud) {
        return cloud.points.size() * sizeof(Eigen::Vector3d) +
               cloud.normals.size() * sizeof(Eigen::Vector3f) +
               cloud.colours.size() + cloud.confidence.size() * sizeof(float);
    }
    static std::vector<std::uint8_t> encode(const PointCloud& cloud) {
        return docxml::sidecar::encode_point_cloud(cloud);
    }
    static PointCloud decode(const std::vector<std::uint8_t>& in) {
        return docxml::sidecar::decode_point_cloud(in);
    }
};

template <>
struct PayloadCodec<std::vector<DepthMap>> {
    static std::size_t bytes(const std::vector<DepthMap>& maps) {
        std::size_t total = 0;
        for (const DepthMap& map : maps) {
            total += map.depths.size() * sizeof(float) +
                     map.normals.size() * sizeof(float) +
                     map.confidence.size() * sizeof(float);
        }
        return total;
    }
    static std::vector<std::uint8_t> encode(const std::vector<DepthMap>& maps) {
        return docxml::sidecar::encode_depth_maps(maps);
    }
    static std::vector<DepthMap> decode(const std::vector<std::uint8_t>& in) {
        return docxml::sidecar::decode_depth_maps(in);
    }
};

template <>
struct PayloadCodec<Mesh> {
    static std::size_t bytes(const Mesh& mesh) {
        return mesh.vertices.size() * sizeof(Eigen::Vector3d) +
               mesh.faces.size() * sizeof(Eigen::Vector3i) +
               mesh.uvs.size() * sizeof(Eigen::Vector2f) +
               mesh.uv_faces.size() * sizeof(Eigen::Vector3i) +
               mesh.colours.size() + mesh.confidence.size() * sizeof(float);
    }
    static std::vector<std::uint8_t> encode(const Mesh& mesh) {
        return docxml::sidecar::encode_mesh(mesh);
    }
    static Mesh decode(const std::vector<std::uint8_t>& in) {
        return docxml::sidecar::decode_mesh(in);
    }
};

template <>
struct PayloadCodec<Elevation> {
    static std::size_t bytes(const Elevation& elevation) {
        return elevation.raster.data.size() * sizeof(float);
    }
    static std::vector<std::uint8_t> encode(const Elevation& elevation) {
        std::vector<std::uint8_t> out;
        docxml::sidecar::put_string(out, elevation.source);
        put_blob(out, docxml::sidecar::encode_elevation(elevation));
        return out;
    }
    static Elevation decode(const std::vector<std::uint8_t>& in) {
        docxml::sidecar::ByteReader reader(in, 0, "undo elevation payload");
        const std::string source = reader.string();
        Elevation elevation = docxml::sidecar::decode_elevation(get_blob(reader));
        elevation.source = source;
        return elevation;
    }
};

template <>
struct PayloadCodec<Orthomosaic> {
    static std::size_t bytes(const Orthomosaic& ortho) {
        return ortho.raster.data.size() * sizeof(std::uint8_t) +
               ortho.labels.size() * sizeof(std::int32_t);
    }
    static std::vector<std::uint8_t> encode(const Orthomosaic& ortho) {
        std::vector<std::uint8_t> out;
        docxml::sidecar::put_string(out, ortho.surface);
        docxml::sidecar::put_u32(out, static_cast<std::uint32_t>(ortho.labels.size()));
        for (const std::int32_t label : ortho.labels) {
            docxml::sidecar::put_i32(out, label);
        }
        put_blob(out, docxml::sidecar::encode_orthomosaic(ortho));
        return out;
    }
    static Orthomosaic decode(const std::vector<std::uint8_t>& in) {
        docxml::sidecar::ByteReader reader(in, 0, "undo orthomosaic payload");
        const std::string surface = reader.string();
        const std::uint32_t label_count = reader.u32();
        std::vector<std::int32_t> labels;
        labels.reserve(label_count);
        for (std::uint32_t i = 0; i < label_count; ++i) {
            labels.push_back(reader.i32());
        }
        Orthomosaic ortho = docxml::sidecar::decode_orthomosaic(get_blob(reader));
        ortho.surface = surface;
        ortho.labels = std::move(labels);
        return ortho;
    }
};

template <>
struct PayloadCodec<TiledModel> {
    static std::size_t bytes(const TiledModel& model) { return model.payload.size(); }
    static std::vector<std::uint8_t> encode(const TiledModel& model) {
        return docxml::sidecar::encode_tiled_model(model);
    }
    static TiledModel decode(const std::vector<std::uint8_t>& in) {
        return docxml::sidecar::decode_tiled_model(in);
    }
};

template <>
struct PayloadCodec<Image> {
    static std::size_t bytes(const Image& image) { return image.size_bytes(); }
    static std::vector<std::uint8_t> encode(const Image& image) {
        std::vector<std::uint8_t> out;
        put_image(out, image);
        return out;
    }
    static Image decode(const std::vector<std::uint8_t>& in) {
        docxml::sidecar::ByteReader reader(in, 0, "undo image payload");
        return get_image(reader);
    }
};

// --- retaining an asset slot -----------------------------------------------

/// Metadata half of an encoded slot: every field except the payload pointer.
template <class Slot>
std::vector<std::uint8_t> encode_slot_meta(const Slot& slot) {
    std::vector<std::uint8_t> out;
    docxml::sidecar::put_i32(out, slot.id);
    docxml::sidecar::put_string(out, slot.label);
    docxml::sidecar::put_i32(out, slot.group_id);
    out.push_back(slot.enabled ? 1U : 0U);
    docxml::sidecar::put_u32(out,
                             static_cast<std::uint32_t>(slot.meta.items.size()));
    for (const auto& item : slot.meta.items) {
        docxml::sidecar::put_string(out, item.first);
        docxml::sidecar::put_string(out, item.second);
    }
    return out;
}

template <class Slot>
void decode_slot_meta(docxml::sidecar::ByteReader& reader, Slot& slot) {
    slot.id = reader.i32();
    slot.label = reader.string();
    slot.group_id = reader.i32();
    slot.enabled = reader.u8() != 0;
    const std::uint32_t items = reader.u32();
    slot.meta.items.clear();
    for (std::uint32_t i = 0; i < items; ++i) {
        std::string key = reader.string();
        std::string value = reader.string();
        slot.meta.items.emplace_back(std::move(key), std::move(value));
    }
}

/// How one asset slot is measured, encoded and decoded when it has to leave
/// memory. The generic form covers the payload in `data`; ModelAsset adds its
/// texture pages and source link, which live outside the payload pointer.
template <class Slot>
struct AssetRetention {
    using Payload = typename Slot::value_type;

    static std::size_t bytes(const Slot& slot) {
        return PayloadCodec<Payload>::bytes(*slot.data);
    }
    static std::vector<std::uint8_t> encode(const Slot& slot) {
        std::vector<std::uint8_t> out = encode_slot_meta(slot);
        put_blob(out, PayloadCodec<Payload>::encode(*slot.data));
        return out;
    }
    static Slot decode(const std::vector<std::uint8_t>& in) {
        docxml::sidecar::ByteReader reader(in, 0, "undo asset payload");
        Slot slot;
        decode_slot_meta(reader, slot);
        slot.data = std::make_shared<Payload>(
            PayloadCodec<Payload>::decode(get_blob(reader)));
        return slot;
    }
};

template <>
struct AssetRetention<ModelAsset> {
    using Payload = Mesh;

    static std::size_t bytes(const ModelAsset& slot) {
        std::size_t total = PayloadCodec<Mesh>::bytes(*slot.data);
        for (const Image& page : slot.textures) {
            total += page.size_bytes();
        }
        return total;
    }
    static std::vector<std::uint8_t> encode(const ModelAsset& slot) {
        std::vector<std::uint8_t> out = encode_slot_meta(slot);
        docxml::sidecar::put_u32(out, static_cast<std::uint32_t>(slot.source_id));
        docxml::sidecar::put_u32(out,
                                 static_cast<std::uint32_t>(slot.textures.size()));
        for (const Image& page : slot.textures) {
            put_image(out, page);
        }
        put_blob(out, PayloadCodec<Mesh>::encode(*slot.data));
        return out;
    }
    static ModelAsset decode(const std::vector<std::uint8_t>& in) {
        docxml::sidecar::ByteReader reader(in, 0, "undo model payload");
        ModelAsset slot;
        decode_slot_meta(reader, slot);
        slot.source_id = static_cast<std::int32_t>(reader.i32());
        const std::uint32_t pages = reader.u32();
        slot.textures.reserve(pages);
        for (std::uint32_t i = 0; i < pages; ++i) {
            slot.textures.push_back(get_image(reader));
        }
        slot.data = std::make_shared<Mesh>(
            PayloadCodec<Mesh>::decode(get_blob(reader)));
        return slot;
    }
};

// --- retail/remove: retention of a whole displaced slot ---------------------

/// Retains one displaced asset slot under the §4.7 policy: the whole slot,
/// including the metadata a removal has to restore.
template <class Slot>
class SlotHolder {
public:
    void set_cache_dir(std::string cache_dir) { cache_dir_ = std::move(cache_dir); }

    /// Called at stage() time with the slot the document is about to displace.
    /// The slot is kept whole in memory when small; otherwise it is encoded
    /// and spilled, and the in-memory copy is dropped so a large remove costs
    /// disk rather than RAM.
    void hold(const Slot& slot) {
        clear();
        bytes_ = AssetRetention<Slot>::bytes(slot);
        if (bytes_ <= kUndoAssetSpillBytes) {
            slot_.emplace(slot);
            held_in_memory_ = true;
            return;
        }
        if (cache_dir_.empty()) {
            valid_ = false;
            return;
        }
        const std::vector<std::uint8_t> blob = AssetRetention<Slot>::encode(slot);
        spill_path_ = make_spill_path(cache_dir_);
        std::ofstream out(spill_path_, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw Error("cannot open undo spill file '" + spill_path_ + "'");
        }
        out.write(reinterpret_cast<const char*>(blob.data()),
                  static_cast<std::streamsize>(blob.size()));
        out.close();
        if (!out) {
            std::remove(spill_path_.c_str());
            spill_path_.clear();
            throw Error("cannot write undo spill file");
        }
    }

    /// The displaced slot: held in memory, or reloaded from the spill file.
    std::optional<Slot> release_hold() const {
        if (held_in_memory_) {
            return slot_;
        }
        if (spill_path_.empty()) {
            return std::nullopt;
        }
        std::ifstream in(spill_path_, std::ios::binary);
        if (!in) {
            throw Error("undo spill file is missing: " + spill_path_);
        }
        std::vector<std::uint8_t> blob((std::istreambuf_iterator<char>(in)),
                                       std::istreambuf_iterator<char>());
        if (!in && !in.eof()) {
            throw Error("cannot read undo spill file: " + spill_path_);
        }
        return AssetRetention<Slot>::decode(blob);
    }

    bool valid() const { return valid_; }
    bool spilled() const { return !spill_path_.empty(); }
    std::size_t bytes() const { return bytes_; }
    std::size_t retained_bytes() const {
        return spill_path_.empty() ? bytes_ : 0;
    }
    bool held_in_memory() const { return held_in_memory_; }

    void clear() {
        if (!spill_path_.empty()) {
            std::remove(spill_path_.c_str());
            spill_path_.clear();
        }
        slot_.reset();
        held_in_memory_ = false;
        bytes_ = 0;
        valid_ = true;
    }

    ~SlotHolder() { clear(); }
    SlotHolder() = default;
    SlotHolder(const SlotHolder&) = delete;
    SlotHolder& operator=(const SlotHolder&) = delete;

private:
    std::string cache_dir_;
    std::optional<Slot> slot_;
    std::string spill_path_;
    std::size_t bytes_ = 0;
    bool held_in_memory_ = false;
    bool valid_ = true;
};
// --- Asset commands --------------------------------------------------------
//
// AssetSlot::data is a shared_ptr, so a snapshot keeps the very same slot the
// document holds: a revert re-installs it byte-for-byte with no payload copy.
// A command never mutates through a pointer a snapshot also references. A slot
// that has to outlive a removal is retained whole by SlotHolder, which respects
// the §4.7 budget and spills above kUndoAssetSpillBytes.

/// Add one asset of `Slot`'s kind and activate it. With a payload the slot is
/// filled before commit() returns, so apply ends in one fully-formed asset —
/// which is exactly what wrapping a task result needs.
template <class Slot>
class AddAssetCommand : public StagedMutation {
public:
    using Payload = typename Slot::value_type;

    AddAssetCommand(std::int32_t chunk_id, std::string label)
        : chunk_id_(chunk_id), label_(std::move(label)) {}

    AddAssetCommand(std::int32_t chunk_id, std::string label, Payload payload)
        : chunk_id_(chunk_id), label_(std::move(label)), payload_(std::move(payload)),
          has_payload_(true) {}

    std::string name() const override { return AssetSlots<Slot>::add_name; }

protected:
    void validate(Document& document) override {
        (void)require_chunk(document, chunk_id_);
    }

    void stage(Document& document) override {
        Chunk& chunk = require_chunk(document, chunk_id_);
        previous_next_id_ = chunk.next_id();
        previous_active_ = AssetSlots<Slot>::active(chunk);
        inserted_id_ = -1;
    }

    void commit(Document& document) override {
        Chunk& chunk = require_chunk(document, chunk_id_);
        Slot& slot = AssetSlots<Slot>::add(chunk, label_);
        if (has_payload_) {
            slot.data = std::make_shared<Payload>(std::move(payload_));
        }
        inserted_id_ = slot.id;
    }

    void restore(Document& document) override {
        Chunk& chunk = require_chunk(document, chunk_id_);
        auto& slots = AssetSlots<Slot>::get(chunk);
        // Hand the payload back to the command, so a redo installs the same
        // bytes instead of a default-constructed one. Only the slot this
        // command installed is touched, and only once.
        for (Slot& slot : slots) {
            if (slot.id == inserted_id_ && has_payload_ && slot.data) {
                payload_ = std::move(*slot.data);
            }
        }
        slots.erase(std::remove_if(slots.begin(), slots.end(),
                                   [this](const Slot& slot) {
                                       return slot.id == inserted_id_;
                                   }),
                    slots.end());
        chunk.set_next_id(previous_next_id_);
        AssetSlots<Slot>::set_active(chunk, previous_active_);
        inserted_id_ = -1;
    }

    /// Id of the slot the last apply created, or -1 (tests and callers that
    /// need to fill the payload after execute()).
    std::int32_t inserted_id() const { return inserted_id_; }

private:
    std::int32_t chunk_id_;
    std::string label_;
    Payload payload_;
    bool has_payload_ = false;
    std::int32_t previous_next_id_ = 0;
    std::int32_t previous_active_ = -1;
    std::int32_t inserted_id_ = -1;
};

/// Add a complete model asset: the mesh payload *plus* its texture pages and
/// source link, which is what BuildTexture's output looks like.
class AddModelAssetCommand : public StagedMutation {
public:
    AddModelAssetCommand(std::int32_t chunk_id, std::string label, ModelAsset model)
        : chunk_id_(chunk_id), label_(std::move(label)), model_(std::move(model)) {}

    std::string name() const override { return AssetSlots<ModelAsset>::add_name; }

protected:
    void validate(Document& document) override {
        (void)require_chunk(document, chunk_id_);
    }

    void stage(Document& document) override {
        Chunk& chunk = require_chunk(document, chunk_id_);
        previous_next_id_ = chunk.next_id();
        previous_active_ = AssetSlots<ModelAsset>::active(chunk);
        inserted_id_ = -1;
    }

    void commit(Document& document) override {
        Chunk& chunk = require_chunk(document, chunk_id_);
        ModelAsset& slot = chunk.add_model(label_);
        slot.data = model_.data ? model_.data : std::make_shared<Mesh>();
        slot.textures = std::move(model_.textures);
        slot.source_id = model_.source_id;
        inserted_id_ = slot.id;
    }

    void restore(Document& document) override {
        Chunk& chunk = require_chunk(document, chunk_id_);
        auto& slots = chunk.models();
        // Take the payload and pages back, so a redo re-installs them rather
        // than an empty model.
        for (ModelAsset& slot : slots) {
            if (slot.id == inserted_id_) {
                model_.data = std::move(slot.data);
                model_.textures = std::move(slot.textures);
            }
        }
        slots.erase(std::remove_if(slots.begin(), slots.end(),
                                   [this](const ModelAsset& slot) {
                                       return slot.id == inserted_id_;
                                   }),
                    slots.end());
        chunk.set_next_id(previous_next_id_);
        AssetSlots<ModelAsset>::set_active(chunk, previous_active_);
        inserted_id_ = -1;
    }

private:
    std::int32_t chunk_id_;
    std::string label_;
    ModelAsset model_;
    std::int32_t previous_next_id_ = 0;
    std::int32_t previous_active_ = -1;
    std::int32_t inserted_id_ = -1;
};

/// Remove one asset. The displaced slot is held by SlotHolder (151->) so it can
/// be spilled; the active id falls back to the last remaining slot, or -1.
template <class Slot>
class RemoveAssetCommand : public StagedMutation {
public:
    RemoveAssetCommand(std::int32_t chunk_id, std::int32_t asset_id)
        : chunk_id_(chunk_id), asset_id_(asset_id) {}

    std::string name() const override { return AssetSlots<Slot>::remove_name; }

    void on_prepare(const UndoContext& context) override {
        holder_.set_cache_dir(context.cache_dir);
    }

    bool is_memory_bounded() const override { return holder_.valid(); }
    std::size_t retained_bytes() const override { return holder_.retained_bytes(); }

    bool spilled() const { return holder_.spilled(); }
    std::size_t held_bytes() const { return holder_.bytes(); }

protected:
    void validate(Document& document) override {
        Chunk& chunk = require_chunk(document, chunk_id_);
        if (find_slot(AssetSlots<Slot>::get(chunk), asset_id_) == nullptr) {
            throw ValueError(std::string("no ") + AssetSlots<Slot>::kind_name +
                             " with id " + std::to_string(asset_id_));
        }
    }

    void stage(Document& document) override {
        Chunk& chunk = require_chunk(document, chunk_id_);
        auto& slots = AssetSlots<Slot>::get(chunk);
        const auto it = std::find_if(slots.begin(), slots.end(),
                                     [this](const Slot& slot) {
                                         return slot.id == asset_id_;
                                     });
        holder_.hold(*it);
        previous_active_ = AssetSlots<Slot>::active(chunk);
        previous_next_id_ = chunk.next_id();
    }

    void commit(Document& document) override {
        if (!holder_.valid()) {
            // Too large and nowhere to spill: decline before touching the
            // document, so execute() can refuse it (§4.7, "pre-decided").
            throw Error("command too large for the undo history");
        }
        Chunk& chunk = require_chunk(document, chunk_id_);
        auto& slots = AssetSlots<Slot>::get(chunk);
        slots.erase(std::remove_if(slots.begin(), slots.end(),
                                   [this](const Slot& slot) {
                                       return slot.id == asset_id_;
                                   }),
                    slots.end());
        if (AssetSlots<Slot>::active(chunk) == asset_id_) {
            AssetSlots<Slot>::set_active(chunk, slots.empty() ? -1 : slots.back().id);
        }
    }

    void restore(Document& document) override {
        // In memory the identical slot (and its payload pointer) comes back; a
        // spilled one is decoded from the cache directory.
        std::optional<Slot> held = holder_.release_hold();
        if (held) {
            insert_slot(document, std::move(*held));
        }
    }

private:
    void insert_slot(Document& document, Slot slot) {
        Chunk& chunk = require_chunk(document, chunk_id_);
        auto& slots = AssetSlots<Slot>::get(chunk);
        insert_by_id(slots, std::move(slot));
        AssetSlots<Slot>::set_active(chunk, previous_active_);
        chunk.set_next_id(previous_next_id_);
    }

    std::int32_t chunk_id_;
    std::int32_t asset_id_;
    SlotHolder<Slot> holder_;
    std::int32_t previous_active_ = -1;
    std::int32_t previous_next_id_ = 0;
};

/// Switch the active asset of one kind (the one the legacy accessors return).
template <class Slot>
class SetActiveAssetCommand : public StagedMutation {
public:
    SetActiveAssetCommand(std::int32_t chunk_id, std::int32_t asset_id)
        : chunk_id_(chunk_id), asset_id_(asset_id) {}

    std::string name() const override { return "Set Active Asset"; }

protected:
    void validate(Document& document) override {
        Chunk& chunk = require_chunk(document, chunk_id_);
        if (find_slot(AssetSlots<Slot>::get(chunk), asset_id_) == nullptr) {
            throw ValueError(std::string("no ") + AssetSlots<Slot>::kind_name +
                             " with id " + std::to_string(asset_id_));
        }
    }

    void stage(Document& document) override {
        Chunk& chunk = require_chunk(document, chunk_id_);
        previous_active_ = AssetSlots<Slot>::active(chunk);
    }

    void commit(Document& document) override {
        AssetSlots<Slot>::set_active(require_chunk(document, chunk_id_), asset_id_);
    }

    void restore(Document& document) override {
        AssetSlots<Slot>::set_active(require_chunk(document, chunk_id_),
                                     previous_active_);
    }

private:
    std::int32_t chunk_id_;
    std::int32_t asset_id_;
    std::int32_t previous_active_ = -1;
};

/// Relabel one asset. Empty labels are rejected: a default name exists so that
/// "no label" is never a legal state.
template <class Slot>
class RenameAssetCommand : public StagedMutation {
public:
    RenameAssetCommand(std::int32_t chunk_id, std::int32_t asset_id,
                       std::string new_label)
        : chunk_id_(chunk_id), asset_id_(asset_id), new_label_(std::move(new_label)) {}

    std::string name() const override { return "Rename Asset"; }

protected:
    void validate(Document& document) override {
        Chunk& chunk = require_chunk(document, chunk_id_);
        if (find_slot(AssetSlots<Slot>::get(chunk), asset_id_) == nullptr) {
            throw ValueError(std::string("no ") + AssetSlots<Slot>::kind_name +
                             " with id " + std::to_string(asset_id_));
        }
        if (new_label_.empty()) {
            throw ValueError("rename asset: the new label is empty");
        }
    }

    void stage(Document& document) override {
        Chunk& chunk = require_chunk(document, chunk_id_);
        previous_label_ = find_slot(AssetSlots<Slot>::get(chunk), asset_id_)->label;
    }

    void commit(Document& document) override { write_label(document, new_label_); }

    void restore(Document& document) override { write_label(document, previous_label_); }

private:
    void write_label(Document& document, const std::string& label) {
        Chunk& chunk = require_chunk(document, chunk_id_);
        find_slot(AssetSlots<Slot>::get(chunk), asset_id_)->label = label;
    }

    std::int32_t chunk_id_;
    std::int32_t asset_id_;
    std::string new_label_;
    std::string previous_label_;
};

// --- panorama (the single optional slot) ------------------------------------

/// Add, replace or clear the single panorama slot. Panorama has no active id,
/// so only the optional slot itself has to be staged.
class PanoramaCommand : public StagedMutation {
public:
    enum class Mode { Add, Rename, Remove };

    PanoramaCommand(std::int32_t chunk_id, Mode mode, std::string label,
                    std::optional<Image> payload, std::int32_t asset_id = -1)
        : chunk_id_(chunk_id), mode_(mode), label_(std::move(label)),
          payload_(std::move(payload)), asset_id_(asset_id) {}

    std::string name() const override {
        switch (mode_) {
            case Mode::Add:
                return "Add Panorama";
            case Mode::Rename:
                return "Rename Asset";
            case Mode::Remove:
                return "Remove Panorama";
        }
        return "Panorama";
    }

    void on_prepare(const UndoContext& context) override {
        holder_.set_cache_dir(context.cache_dir);
    }

    bool is_memory_bounded() const override { return holder_.valid(); }
    std::size_t retained_bytes() const override { return holder_.retained_bytes(); }
    bool spilled() const { return holder_.spilled(); }

protected:
    void validate(Document& document) override {
        const Chunk& chunk = require_chunk(document, chunk_id_);
        if (mode_ == Mode::Add) {
            if (chunk.panorama().has_value()) {
                throw ValueError("add panorama: a panorama already exists");
            }
            return;
        }
        if (!chunk.panorama().has_value()) {
            throw ValueError("no panorama asset");
        }
        if (mode_ == Mode::Rename && label_.empty()) {
            throw ValueError("rename asset: the new label is empty");
        }
        if (chunk.panorama()->id != asset_id_ && mode_ != Mode::Add) {
            throw ValueError("no panorama with id " + std::to_string(asset_id_));
        }
    }

    void stage(Document& document) override {
        Chunk& chunk = require_chunk(document, chunk_id_);
        if (mode_ == Mode::Remove) {
            // Same policy as the vector kinds: in RAM when small, spilled
            // above kUndoAssetSpillBytes.
            holder_.hold(*chunk.panorama());
        } else {
            previous_ = chunk.panorama();
        }
        previous_next_id_ = chunk.next_id();
    }

    void commit(Document& document) override {
        Chunk& chunk = require_chunk(document, chunk_id_);
        if (mode_ == Mode::Add) {
            PanoramaAsset slot;
            slot.id = chunk.allocate_id();
            slot.label = label_.empty() ? "Panorama" : label_;
            if (payload_) {
                slot.data = std::make_shared<Image>(std::move(*payload_));
            }
            chunk.panorama() = std::move(slot);
            return;
        }
        if (mode_ == Mode::Remove) {
            if (!holder_.valid()) {
                throw Error("command too large for the undo history");
            }
            chunk.panorama().reset();
            return;
        }
        chunk.panorama()->label = label_;
    }

    void restore(Document& document) override {
        Chunk& chunk = require_chunk(document, chunk_id_);
        if (mode_ == Mode::Remove) {
            if (std::optional<PanoramaAsset> held = holder_.release_hold()) {
                chunk.panorama() = std::move(held);
            }
        } else {
            // Take an installed payload back, so a redo re-installs the same
            // pixels rather than a moved-from image.
            if (mode_ == Mode::Add && payload_ && chunk.panorama().has_value() &&
                chunk.panorama()->data) {
                payload_ = std::move(*chunk.panorama()->data);
            }
            chunk.panorama() = previous_;
        }
        chunk.set_next_id(previous_next_id_);
    }

private:
    std::int32_t chunk_id_;
    Mode mode_;
    std::string label_;
    std::optional<Image> payload_;
    std::optional<PanoramaAsset> previous_;
    SlotHolder<PanoramaAsset> holder_;
    std::int32_t previous_next_id_ = 0;
    std::int32_t asset_id_ = -1;
};

/// Set a chunk's CRS. The M11 plan's `make_set_chunk_crs_command` is written
/// against `crs::CoordinateSystem`, which arrives with M12 WP-12.11; until then
/// this version takes the existing `CRS` (WKT + EPSG) so the wiring exists and
/// M12 only has to widen the argument type. See LIMITATIONS.md.
class SetChunkCrsCommand : public StagedMutation {
public:
    SetChunkCrsCommand(std::int32_t chunk_id, CRS crs)
        : chunk_id_(chunk_id), crs_(std::move(crs)) {}

    std::string name() const override { return "Set Chunk CRS"; }

protected:
    void validate(Document& document) override {
        (void)require_chunk(document, chunk_id_);
        if (crs_.epsg < 0) {
            throw ValueError("set chunk CRS: EPSG code must be non-negative");
        }
        if (!crs_.is_set()) {
            throw ValueError("set chunk CRS: the CRS has neither WKT nor an EPSG code");
        }
    }

    void stage(Document& document) override {
        previous_ = require_chunk(document, chunk_id_).crs();
    }

    void commit(Document& document) override {
        require_chunk(document, chunk_id_).crs() = crs_;
    }

    void restore(Document& document) override {
        require_chunk(document, chunk_id_).crs() = previous_;
    }

private:
    std::int32_t chunk_id_;
    CRS crs_;
    CRS previous_;
};

// --- chunk factories -------------------------------------------------------

}  // namespace

std::unique_ptr<AbstractCommand> make_add_marker_command(std::int32_t chunk_id,
                                                        Marker marker) {
    return std::make_unique<AddMarkerCommand>(chunk_id, std::move(marker));
}

std::unique_ptr<AbstractCommand> make_remove_marker_command(std::int32_t chunk_id,
                                                           std::int32_t marker_id) {
    return std::make_unique<RemoveMarkerCommand>(chunk_id, marker_id);
}

std::unique_ptr<AbstractCommand> make_set_marker_reference_command(
    std::int32_t chunk_id, std::int32_t marker_id,
    std::optional<Eigen::Vector3d> reference, Eigen::Vector3d accuracy) {
    return std::make_unique<SetMarkerReferenceCommand>(
        chunk_id, marker_id, reference, accuracy);
}

std::unique_ptr<AbstractCommand> make_move_marker_projection_command(
    std::int32_t chunk_id, std::int32_t marker_id, std::int32_t camera_id,
    Eigen::Vector2d pixel) {
    return std::make_unique<MoveMarkerProjectionCommand>(
        chunk_id, marker_id, camera_id, pixel, /*create=*/true);
}

std::unique_ptr<AbstractCommand> make_remove_marker_projection_command(
    std::int32_t chunk_id, std::int32_t marker_id, std::int32_t camera_id) {
    return std::make_unique<RemoveMarkerProjectionCommand>(chunk_id, marker_id,
                                                           camera_id);
}

std::unique_ptr<AbstractCommand> make_add_scalebar_command(std::int32_t chunk_id,
                                                          Scalebar scalebar) {
    return std::make_unique<AddScalebarCommand>(chunk_id, std::move(scalebar));
}

std::unique_ptr<AbstractCommand> make_remove_scalebar_command(
    std::int32_t chunk_id, std::int32_t scalebar_id) {
    return std::make_unique<RemoveScalebarCommand>(chunk_id, scalebar_id);
}

std::unique_ptr<AbstractCommand> make_set_scalebar_reference_command(
    std::int32_t chunk_id, std::int32_t scalebar_id, double length_m,
    double accuracy_m) {
    return std::make_unique<SetScalebarReferenceCommand>(chunk_id, scalebar_id,
                                                         length_m, accuracy_m);
}

std::unique_ptr<AbstractCommand> make_add_group_command(GroupKind kind,
                                                       std::int32_t chunk_id,
                                                       std::string label) {
    return std::make_unique<AddGroupCommand>(kind, chunk_id, std::move(label));
}

std::unique_ptr<AbstractCommand> make_remove_group_command(GroupKind kind,
                                                          std::int32_t chunk_id,
                                                          std::int32_t group_id) {
    return std::make_unique<RemoveGroupCommand>(kind, chunk_id, group_id);
}

std::unique_ptr<AbstractCommand> make_move_to_group_command(
    GroupKind kind, std::int32_t chunk_id, std::vector<std::int32_t> ids,
    std::int32_t group_id) {
    return std::make_unique<MoveToGroupCommand>(kind, chunk_id, std::move(ids),
                                                group_id);
}

std::unique_ptr<AbstractCommand> make_set_camera_enabled_command(
    std::int32_t chunk_id, std::vector<std::int32_t> camera_ids, bool enabled) {
    return std::make_unique<SetCameraFlagCommand>(
        chunk_id, std::move(camera_ids), SetCameraFlagCommand::Flag::Enabled,
        enabled);
}

std::unique_ptr<AbstractCommand> make_set_camera_selected_command(
    std::int32_t chunk_id, std::vector<std::int32_t> camera_ids, bool selected) {
    return std::make_unique<SetCameraFlagCommand>(
        chunk_id, std::move(camera_ids), SetCameraFlagCommand::Flag::Selected,
        selected);
}

std::unique_ptr<AbstractCommand> make_append_document_command(
    std::shared_ptr<Document> source, std::string source_path) {
    return std::make_unique<AppendDocumentCommand>(std::move(source),
                                                   std::move(source_path));
}

std::unique_ptr<AbstractCommand> make_add_chunk_command(Workspace& workspace,
                                                        std::string label,
                                                        std::int32_t insert_index) {
    return std::make_unique<AddChunkCommand>(workspace, std::move(label),
                                             insert_index);
}

std::unique_ptr<AbstractCommand> make_remove_chunk_command(std::int32_t chunk_id) {
    return std::make_unique<RemoveChunkCommand>(chunk_id);
}

std::unique_ptr<AbstractCommand> make_duplicate_chunk_command(
    std::int32_t chunk_id, std::string new_label) {
    return std::make_unique<DuplicateChunkCommand>(chunk_id, std::move(new_label));
}

std::unique_ptr<AbstractCommand> make_reorder_chunks_command(
    std::vector<std::int32_t> new_order) {
    return std::make_unique<ReorderChunksCommand>(std::move(new_order));
}

std::unique_ptr<AbstractCommand> make_set_chunk_transform_command(
    std::int32_t chunk_id, Transform transform, Region region) {
    return std::make_unique<SetChunkTransformCommand>(chunk_id, std::move(transform),
                                                      region);
}

std::unique_ptr<AbstractCommand> make_set_chunk_crs_command(std::int32_t chunk_id,
                                                            CRS crs) {
    return std::make_unique<SetChunkCrsCommand>(chunk_id, std::move(crs));
}

std::unique_ptr<AbstractCommand> make_align_chunks_command(std::int32_t chunk_id,
                                                           ChunkGeometry after) {
    return std::make_unique<AlignChunksCommand>(chunk_id, std::move(after));
}

std::unique_ptr<AbstractCommand> make_set_chunk_meta_command(std::int32_t chunk_id,
                                                             std::string key,
                                                             std::string value) {
    return std::make_unique<SetChunkMetaCommand>(chunk_id, std::move(key),
                                                 std::move(value));
}

// --- asset factories -------------------------------------------------------

namespace {

/// Dispatch an AssetKind onto its slot type without instantiating every
/// combination in every factory. Panorama is handled separately: it is a single
/// optional slot, not a vector with an active id.
template <class Slot>
struct SlotTag {
    using type = Slot;
};

template <class Callable>
std::unique_ptr<AbstractCommand> with_asset_kind(AssetKind kind, Callable&& callable) {
    switch (kind) {
        case AssetKind::PointCloud:
            return callable(SlotTag<PointCloudAsset>{});
        case AssetKind::Model:
            return callable(SlotTag<ModelAsset>{});
        case AssetKind::DepthMaps:
            return callable(SlotTag<DepthMapsAsset>{});
        case AssetKind::Elevation:
            return callable(SlotTag<ElevationAsset>{});
        case AssetKind::Orthomosaic:
            return callable(SlotTag<OrthomosaicAsset>{});
        case AssetKind::TiledModel:
            return callable(SlotTag<TiledModelAsset>{});
        case AssetKind::Panorama:
            break;
    }
    throw ValueError("panorama assets are addressed through their own factory");
}

}  // namespace

std::unique_ptr<AbstractCommand> make_add_asset_command(AssetKind kind,
                                                        std::int32_t chunk_id,
                                                        std::string label) {
    if (kind == AssetKind::Panorama) {
        return std::make_unique<PanoramaCommand>(chunk_id, PanoramaCommand::Mode::Add,
                                                 std::move(label), std::nullopt);
    }
    return with_asset_kind(kind, [&](auto tag) -> std::unique_ptr<AbstractCommand> {
        using Slot = typename decltype(tag)::type;
        return std::make_unique<AddAssetCommand<Slot>>(chunk_id, std::move(label));
    });
}

std::unique_ptr<AbstractCommand> make_add_asset_command(std::int32_t chunk_id,
                                                        PointCloud cloud,
                                                        std::string label) {
    return std::make_unique<AddAssetCommand<PointCloudAsset>>(
        chunk_id, std::move(label), std::move(cloud));
}

std::unique_ptr<AbstractCommand> make_add_asset_command(std::int32_t chunk_id,
                                                        ModelAsset model,
                                                        std::string label) {
    return std::make_unique<AddModelAssetCommand>(chunk_id, std::move(label),
                                                  std::move(model));
}

std::unique_ptr<AbstractCommand> make_add_asset_command(
    std::int32_t chunk_id, std::vector<DepthMap> maps, std::string label) {
    return std::make_unique<AddAssetCommand<DepthMapsAsset>>(
        chunk_id, std::move(label), std::move(maps));
}

std::unique_ptr<AbstractCommand> make_add_asset_command(std::int32_t chunk_id,
                                                        Elevation elevation,
                                                        std::string label) {
    return std::make_unique<AddAssetCommand<ElevationAsset>>(
        chunk_id, std::move(label), std::move(elevation));
}

std::unique_ptr<AbstractCommand> make_add_asset_command(std::int32_t chunk_id,
                                                        Orthomosaic orthomosaic,
                                                        std::string label) {
    return std::make_unique<AddAssetCommand<OrthomosaicAsset>>(
        chunk_id, std::move(label), std::move(orthomosaic));
}

std::unique_ptr<AbstractCommand> make_add_asset_command(std::int32_t chunk_id,
                                                        TiledModel model,
                                                        std::string label) {
    return std::make_unique<AddAssetCommand<TiledModelAsset>>(
        chunk_id, std::move(label), std::move(model));
}

std::unique_ptr<AbstractCommand> make_add_asset_command(std::int32_t chunk_id,
                                                        Image panorama,
                                                        std::string label) {
    return std::make_unique<PanoramaCommand>(chunk_id, PanoramaCommand::Mode::Add,
                                             std::move(label), std::move(panorama));
}

std::unique_ptr<AbstractCommand> make_remove_asset_command(AssetKind kind,
                                                          std::int32_t chunk_id,
                                                          std::int32_t asset_id) {
    if (kind == AssetKind::Panorama) {
        return std::make_unique<PanoramaCommand>(chunk_id, PanoramaCommand::Mode::Remove,
                                                 std::string(), std::nullopt, asset_id);
    }
    return with_asset_kind(kind, [&](auto tag) -> std::unique_ptr<AbstractCommand> {
        using Slot = typename decltype(tag)::type;
        return std::make_unique<RemoveAssetCommand<Slot>>(chunk_id, asset_id);
    });
}

std::unique_ptr<AbstractCommand> make_set_active_asset_command(AssetKind kind,
                                                              std::int32_t chunk_id,
                                                              std::int32_t asset_id) {
    if (kind == AssetKind::Panorama) {
        throw ValueError("panorama assets have no active id");
    }
    return with_asset_kind(kind, [&](auto tag) -> std::unique_ptr<AbstractCommand> {
        using Slot = typename decltype(tag)::type;
        return std::make_unique<SetActiveAssetCommand<Slot>>(chunk_id, asset_id);
    });
}

std::unique_ptr<AbstractCommand> make_rename_asset_command(AssetKind kind,
                                                          std::int32_t chunk_id,
                                                          std::int32_t asset_id,
                                                          std::string new_label) {
    if (kind == AssetKind::Panorama) {
        return std::make_unique<PanoramaCommand>(chunk_id, PanoramaCommand::Mode::Rename,
                                                 std::move(new_label), std::nullopt,
                                                 asset_id);
    }
    return with_asset_kind(kind, [&](auto tag) -> std::unique_ptr<AbstractCommand> {
        using Slot = typename decltype(tag)::type;
        return std::make_unique<RenameAssetCommand<Slot>>(chunk_id, asset_id,
                                                          std::move(new_label));
    });
}

}  // namespace alphashape
