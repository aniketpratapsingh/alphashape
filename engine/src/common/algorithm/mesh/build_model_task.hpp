#pragma once

// BuildModel: turn the chunk's reconstruction into a triangle mesh (milestone 7).
//
// The adapter between the chunk's data model and the meshing stages. Depending
// on `source_data` it builds a scalar volume either by TV-L1 range-image fusion
// of the depth maps (tv_fusion.hpp) or by screened Poisson reconstruction from
// the dense point cloud (poisson.hpp), extracts the iso-surface (marching
// tetrahedra, marching_tetrahedra.hpp), cleans it up (component removal, quadric
// decimation, mesh_ops.hpp / decimate.hpp) and — when depth maps exist — colours
// its vertices from the source images.
//
// Commit. A NEW model asset (with no texture pages yet) is appended and made
// active, so an earlier model — and the texture pages addressing ITS uv layout —
// is left untouched. `label` names the new asset; an error anywhere leaves the
// chunk unchanged.

#include <string>
#include <utility>

#include <Eigen/Core>

#include "common/tasks/task.hpp"

namespace alphashape {

class Chunk;

/// The reconstruction volume's bounding box for a BuildModel `source_data`
/// value: the per-axis 2nd..98th percentile of the source points (tie-point
/// tracks for "depth_maps", the dense cloud for "point_cloud") grown by a fixed
/// margin. Exposed because the voxel size the grid ends up with — the longest
/// edge over `volume_resolution` — is the natural unit for quoting meshing
/// accuracy, and callers should not have to re-derive the box.
std::pair<Eigen::Vector3d, Eigen::Vector3d> build_model_bbox(const Chunk& chunk,
                                                             const std::string& source_data);

class BuildModelTask : public AbstractTask {
public:
    std::string name() const override { return "BuildModel"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
