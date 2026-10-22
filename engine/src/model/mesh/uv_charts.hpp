#pragma once

// UV chart construction: partition the mesh's faces into charts whose normals
// fit inside a cone, then flatten each chart to the plane.
//
// A "chart" is a set of faces that are flattened together: within one chart the
// parameterisation is continuous, so neighbouring faces share their corner
// coordinates and the texture shows no seam. Across charts the surface is cut
// (a seam), which is what lets a closed or a folded surface lie flat without
// overlapping itself.
//
// This header declares the interface only. build_charts() is a stub until the
// texturing milestone lands its implementation.

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "model/mesh/mesh.hpp"

namespace alphashape {

/// One flattened patch of the surface.
///
/// `corner_uv` is indexed in lockstep with `faces`: the 3 entries at positions
/// 3*i, 3*i+1, 3*i+2 are the flattened coordinates of faces[i]'s (v0, v1, v2)
/// corners. Coordinates are in WORLD UNITS (the same scale as the mesh's
/// positions), not normalised — the atlas packer later places and scales them
/// into the [0,1] page. A chart is flattened so that it does not overlap itself.
struct Chart {
    /// Face ids into Mesh::faces.
    std::vector<int> faces;
    /// 3 coordinates per face, same order as `faces`.
    std::vector<Eigen::Vector2d> corner_uv;
};

/// Parameters for the chart-building pass.
struct ChartOptions {
    /// Chart normal cone half-angle, in degrees. Region growing stops adding a
    /// face once the chart's normal cone would exceed this.
    double max_normal_angle_deg = 40.0;
    /// Hard cap on the faces a single chart may hold; a large flat region is
    /// split rather than flattened as one chart.
    std::size_t max_chart_faces = 50000;
};

/// Optional per-call bookkeeping, filled in by build_charts() when a non-null
/// pointer is passed. Everything here is diagnostic: the returned charts do not
/// depend on whether a caller asks for it.
struct ChartStats {
    /// How many times the flatten step found a chart folding over itself and
    /// fell back to re-growing that chart with a tighter normal cone. Zero for
    /// a surface that develops flat without overlap (a plane, a box side); a
    /// positive count says the overlap guard did real work.
    std::size_t overlap_splits = 0;
};

/// Partition `mesh`'s faces into charts by normal-cone region growing and
/// flatten each chart. When `stats` is non-null it receives this run's
/// ChartStats. Previously the overlap-split tally was published through a
/// hidden process-global (uv_charts_overlap_split_count()); this per-call
/// output replaces it and is thread-safe.
std::vector<Chart> build_charts(const Mesh& mesh, const ChartOptions& options,
                                ChartStats* stats = nullptr);

}  // namespace alphashape
