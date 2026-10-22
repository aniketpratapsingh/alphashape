#pragma once

// Mesh decimation by quadric edge collapse (Garland & Heckbert), for both
// uniform and adaptive (screen- / landmark-driven) simplification.
//
// Each vertex carries a symmetric 4x4 quadric — the sum of the squared-distance
// planes of its incident faces. Collapsing an edge to a position minimising the
// combined quadric removes two faces per interior collapse; the cost of the
// collapse is the quadric error at the chosen position.

#include <cstddef>
#include <limits>

#include "model/mesh/mesh.hpp"

namespace alphashape {

struct DecimateOptions {
    /// Stop once the mesh has this many faces or fewer. 0 means "decimate as far
    /// as max_error allows" (every admissible collapse).
    std::size_t target_faces = 0;
    /// Refuse any collapse that would touch a boundary edge.
    bool preserve_boundary = true;
    /// Upper bound on a single collapse's quadric error (a squared distance);
    /// collapses costing more than this are rejected.
    double max_error = std::numeric_limits<double>::infinity();
};

/// Garland–Heckbert quadric edge collapse. The link condition is enforced, so a
/// collapse never creates a non-manifold edge; a collapse that would flip any
/// incident face normal is rejected. The surviving vertex averages the
/// colours/confidence of the two it replaces. Returns the face count after
/// decimation.
std::size_t decimate_quadric(Mesh& mesh, const DecimateOptions& options);

}  // namespace alphashape
