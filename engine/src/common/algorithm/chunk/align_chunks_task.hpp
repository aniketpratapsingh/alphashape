#pragma once

// AlignChunks: marker-based alignment of the active chunk onto a reference
// chunk (M11 §4.8, WP-11.7, AL-16; point-cloud alignment is M17).
//
// Markers are matched by exact label (Metashape's semantics). From
// the matched marker positions -- in the active chunk's and in the reference
// chunk's coordinates -- a similarity dst = s * R * src + t is estimated with
// Umeyama (1991) inside a deterministic RANSAC (`sfm::estimate_similarity_ransac`,
// seed 0x5eed, threshold = 3 x median marker position sigma), and applied to
// the active chunk: camera poses, tie-point tracks, every asset payload's
// vertex data and the region. The applied similarity is recorded in both
// chunks' meta as `"AlignChunks/<other_id>/matrix"` (16 doubles, row-major).
//
// `method=reference` is the georeferenced case: both chunks must carry the same
// CRS, and the task only records the identity alignment -- the shared CRS
// already places them in one frame.
//
// The document mutation (transform, region and meta) is undoable through the
// command layer: the task applies `make_align_chunks_command`, so a caller that
// wired the task through the undo stack gets a single revertible step.

#include "common/algorithm/sfm/similarity.hpp"
#include "common/core/chunk_geometry.hpp"
#include "common/tasks/task.hpp"

namespace alphashape {

/// Apply a similarity to a chunk geometry snapshot: camera poses, triangulated
/// tie points, marker positions, shapes, trajectories, point-cloud and mesh
/// payloads, the region (axis-aligned bound of its mapped corners) and the
/// chunk transform (T' = T . S).
///
/// Public because the undo command `make_align_chunks_command` takes the
/// post-state and front ends may build it themselves; M17's point-cloud
/// alignment installs the same kind of state. Camera poses are world->camera
/// and the world side is what moves, so each pose is post-multiplied by the
/// inverse similarity: pose' = pose . S^-1.
void apply_similarity_to_geometry(ChunkGeometry& geometry,
                                  const sfm::Similarity& similarity);

/// The similarity estimate behind one marker-based alignment, exposed for tests
/// and for front ends that want the numbers without mutating a document.
struct MarkerAlignmentEstimate {
    bool ok = false;
    sfm::Similarity similarity;
    std::size_t markers_used = 0;   ///< RANSAC inliers
    std::size_t markers_matched = 0;
    double rms_m = 0.0;             ///< residual RMS over the inliers
    double threshold_m = 0.0;       ///< 3 x median marker sigma
    std::string reason;             ///< why it failed, empty when ok
};

/// Fit the similarity that maps `active`'s labelled marker positions onto
/// `reference`'s (§4.8 rules: exact label match, estimated positions only,
/// RANSAC seed 0x5eed, threshold 3 x median sigma, non-collinear, at least
/// `min_common_markers`). `accuracy_m` 0 accepts the RANSAC threshold as the
/// residual gate.
MarkerAlignmentEstimate estimate_marker_alignment(const Chunk& active,
                                                  const Chunk& reference,
                                                  std::size_t min_common_markers = 3,
                                                  double accuracy_m = 0.0);

class AlignChunksTask : public AbstractTask {
public:
    std::string name() const override { return "AlignChunks"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
