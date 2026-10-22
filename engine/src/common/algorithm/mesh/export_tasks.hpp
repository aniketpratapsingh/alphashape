#pragma once

// ExportModel / ExportPointCloud: write the chunk's derived geometry to disk.
//
// These are the file-output adapters on top of the model/mesh/ply.hpp writers.
// They are deliberately thin: resolve the parameters, pick the geometry the
// chunk holds, optionally lift it into world coordinates through the chunk's
// transform, and hand it to write_ply. Nothing here computes geometry.
//
// Read-only w.r.t. the chunk. `save_colors` / `save_confidence` are honoured on
// a COPY of the geometry, never by clearing the chunk's own attribute vectors,
// so an export can never mutate the reconstruction it is writing out.

#include <string>

#include "common/tasks/task.hpp"

namespace alphashape {

/// Write one model asset as a PLY (or OBJ) file: the asset named by `asset_id`,
/// or the active model when it is -1 (the default). Throws ValueError when an
/// explicit id names no asset, and "No model" when the selected asset carries no
/// mesh.
class ExportModelTask : public AbstractTask {
public:
    std::string name() const override { return "ExportModel"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

/// Write one point-cloud asset as a PLY file: the asset named by `asset_id`, or
/// the active cloud when it is -1 (the default). Throws ValueError when an
/// explicit id names no asset, and "No point cloud" when the selected asset
/// carries no points.
class ExportPointCloudTask : public AbstractTask {
public:
    std::string name() const override { return "ExportPointCloud"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
