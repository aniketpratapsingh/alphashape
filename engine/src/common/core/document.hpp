#pragma once

// Document and Workspace: the top of the model hierarchy.
//
//   Document -> Workspace -> Chunk -> Frame
//
// doc.xml root carries `document` with `chunks`, `global_crs`, `meta`,
// `settings`, `next_id`, `active_id`. The Workspace level is thin
// in v0 — it exists because the model has it and because chunks are the unit
// the task layer operates on.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/core/chunk.hpp"

namespace alphashape {

/// Groups chunks. One workspace in the common case; the level is cheap to
/// carry from the start.
class Workspace {
public:
    std::int32_t id() const { return id_; }
    void set_id(std::int32_t id) { id_ = id; }

    std::vector<std::shared_ptr<Chunk>>& chunks() { return chunks_; }
    const std::vector<std::shared_ptr<Chunk>>& chunks() const { return chunks_; }

    Chunk* find_chunk(std::int32_t id);

    /// Allocate the next chunk id and append a chunk.
    Chunk& add_chunk(std::string label);

    /// Append a chunk with a known id (used when loading a project). Keeps the
    /// id counter ahead of every id seen, so later add_chunk() never collides.
    Chunk& restore_chunk(std::int32_t id, std::string label);

    std::int32_t next_chunk_id() const { return next_chunk_id_; }
    void set_next_chunk_id(std::int32_t id) { next_chunk_id_ = id; }

    Meta& meta() { return meta_; }
    const Meta& meta() const { return meta_; }

private:
    std::int32_t id_ = 0;
    std::int32_t next_chunk_id_ = 0;
    std::vector<std::shared_ptr<Chunk>> chunks_;
    Meta meta_;
};

/// The whole project. Owns workspaces and the next-id counters that keep
/// identities stable across save/load — matching the observed `next_id` /
/// `active_id` root attributes.
class Document {
public:
    Document();
    explicit Document(std::string path);

    /// Path the document was loaded from, or will be saved to. Empty for a
    /// new in-memory document.
    const std::string& path() const { return path_; }
    void set_path(std::string path) { path_ = std::move(path); }

    std::vector<std::shared_ptr<Workspace>>& workspaces() { return workspaces_; }
    const std::vector<std::shared_ptr<Workspace>>& workspaces() const {
        return workspaces_;
    }

    Workspace& workspace() { return *workspaces_.front(); }
    const Workspace& workspace() const { return *workspaces_.front(); }

    /// The active chunk, matching the observed `active_id` root attribute.
    std::int32_t active_chunk_id() const { return active_chunk_id_; }
    void set_active_chunk_id(std::int32_t id) { active_chunk_id_ = id; }

    Chunk* active_chunk();
    /// The same chunk for a document held through the shared read lock (the GUI's normal way in).
    const Chunk* active_chunk() const;

    /// Write this document to `path` with only the chunks named in `chunk_ids`
    /// (empty = every chunk). Chunk ids, their chunk-local id counters and
    /// `next_chunk_id()` are written unchanged, so the file can be appended to
    /// another project without an id collision (M11 §4.8, PRJ-06). Throws
    /// ValueError for an id the document does not hold. Defined in
    /// common/io/project_io.cpp next to the writer it wraps.
    void save(const std::string& path,
              const std::vector<std::int32_t>& chunk_ids = {}) const;

    /// Document-level CRS used for display and export defaults (M12 §4.6).
    crs::CoordinateSystem global_coordinate_system() const;
    void set_global_coordinate_system(const crs::CoordinateSystem& cs);

    /// Project-format DTO; see Chunk::crs(). Deprecated in M13.
    CRS& global_crs() { return global_crs_; }
    const CRS& global_crs() const { return global_crs_; }

    Meta& meta() { return meta_; }
    const Meta& meta() const { return meta_; }

    /// Format version written into doc.xml. Bumping this is how a future
    /// reader detects an older file. Version 2 is the M11 layout (every
    /// container, all assets, per-chunk archive paths); version 1 files are
    /// still read exactly as before (see common/io/doc_xml.cpp).
    static constexpr int kFormatVersion = 2;

private:
    std::string path_;
    std::vector<std::shared_ptr<Workspace>> workspaces_;
    std::int32_t active_chunk_id_ = -1;
    CRS global_crs_;
    Meta meta_;
};

}  // namespace alphashape
