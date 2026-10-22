#pragma once

// Chunk operations shared by the DuplicateChunk / MergeChunks / AppendProject
// task adapters (milestone 11, plan §4.8).
//
// A chunk owns a namespace of ids (cameras, sensors, frames, markers,
// scalebars, groups, components, assets, masks, tie points). Copying or
// merging chunk content therefore means *remapping* every one of those ids to a
// value the destination chunk has not used yet, and rewriting every reference
// that points at them. `duplicate_chunk` and `append_chunk` are the two
// operations that do this; `merge_chunks` is a loop over `append_chunk`.

#include <cstdint>
#include <string>
#include <vector>

#include "common/core/chunk.hpp"

namespace alphashape {

/// Which optional parts of a chunk a copy carries over. The three flags mirror
/// the DuplicateChunk task parameters; content outside them (cameras, sensors,
/// frames, markers, ...) is always copied.
struct ChunkCopyOptions {
    bool copy_assets = true;
    bool copy_depth_maps = true;
    bool copy_tie_points = true;
};

/// A copy of `source` with a fresh chunk id and every chunk-local id remapped,
/// so the result is fully independent of the source: no two ids collide and
/// every reference (camera -> sensor/frame, asset -> source camera, mask ->
/// camera, ...) still targets the right object.
///
/// `chunk_id` is the id the copy takes; it must not be used by the document
/// (the caller allocates it). Image paths and CRS are kept verbatim.
Chunk duplicate_chunk(const Chunk& source, std::int32_t chunk_id, std::string label,
                      const ChunkCopyOptions& options = {});

/// Append the content of `source` to `target`, remapping every id so nothing in
/// `target` is overwritten.*/ Existing objects are kept as they are; incoming
/// cameras/sensors/markers/... are added after them with new ids.
///
/// `next_*_id` allocators are advanced past everything written, so later
/// additions to `target` never collide.
void append_chunk(Chunk& target, const Chunk& source, const ChunkCopyOptions& options = {});

/// True when two chunks can be merged: both unreferenced (chunk-local frame),
/// or both in the same CRS. `MixedCrs` chunks are refused by MergeChunks.
bool crs_compatible(const Chunk& a, const Chunk& b);

/// Concatenate `sources` into one new chunk with the given id and label.
/// Refuses an empty list and mixed non-local CRSs (see `crs_compatible`).
Chunk merge_chunks(const std::vector<const Chunk*>& sources, std::int32_t chunk_id,
                   std::string label);

}  // namespace alphashape
