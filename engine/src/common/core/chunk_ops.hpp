#pragma once

// Chunk operations: duplicate, merge and append with id remapping (M11 §4.8).
//
// A chunk owns several id namespaces: camera ids, frame ids (both independent
// counters) and the shared chunk-local id counter behind markers, groups,
// scale bars, shapes, trajectories and every asset. Copying or merging chunk
// content therefore means *remapping* every one of those ids to values the
// destination chunk does not use yet, and rewriting every reference that points
// at them (camera -> sensor, marker -> group, asset -> source asset, mask and
// depth map -> camera, orthomosaic label -> camera, ...).
//
// This lives in core, not in the algorithms library, because the undo command
// layer (`common/commands/model_commands.cpp`) needs the same remapping and core
// must not depend on algorithms. The task adapters under
// `common/algorithm/chunk/` are thin wrappers over it.

#include <cstdint>
#include <string>
#include <vector>

#include "common/core/chunk.hpp"

namespace alphashape {

/// Which optional parts of a chunk a copy carries over. Everything else
/// (cameras, sensors, frames, markers, groups, scale bars, masks, shapes,
/// trajectories) is always copied. The three flags mirror the DuplicateChunk
/// task parameters.
struct ChunkCopyOptions {
    bool copy_assets = true;      ///< point clouds, models, DEMs, orthos, tiled models, panorama
    bool copy_depth_maps = true;  ///< depth-map sets
    bool copy_tie_points = true;  ///< keypoint sets, matches and tracks
};

/// A copy of `source` with a fresh chunk id and every chunk-local id remapped,
/// so the result is fully independent of the source: no two ids collide and
/// every reference still targets the right object.
///
/// `chunk_id` is the id the copy takes; the caller allocates it. Image paths,
/// CRS, region, transform and metadata are copied verbatim. Asset payloads are
/// deep-copied, never shared.
Chunk duplicate_chunk(const Chunk& source, std::int32_t chunk_id, std::string label,
                      const ChunkCopyOptions& options = {});

/// Append the content of `source` to `target`, remapping every id so nothing
/// already in `target` is overwritten. Sensors are deduplicated on exact
/// identity (make, model, serial, width, height, focal_length_mm, calibration);
/// cameras then point at the existing sensor. A structurally broken source
/// (dangling camera/sensor reference) throws ValueError. Callers that need
/// all-or-nothing semantics (merge, duplicate) build into a fresh chunk first.
void append_chunk(Chunk& target, const Chunk& source,
                  const ChunkCopyOptions& options = {});

/// True when two chunks can be merged: both are unreferenced (chunk-local
/// frame), or both carry the same CRS.
bool crs_compatible(const Chunk& a, const Chunk& b);

/// Concatenate `sources` into one new chunk with the given id and label.
/// Refuses an empty list, a null entry and mixed non-local CRSs. The result
/// takes its CRS, transform and metadata from the first source; the region is
/// the union of the sources' bounding boxes.
Chunk merge_chunks(const std::vector<const Chunk*>& sources, std::int32_t chunk_id,
                   std::string label);

}  // namespace alphashape
