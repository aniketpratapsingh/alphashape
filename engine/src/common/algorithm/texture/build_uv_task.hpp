#pragma once

// BuildUV: lay the reconstructed mesh out on a texture atlas (milestone 8).
//
// The task is the adapter between the chunk's data model and the two geometry
// stages that produce a UV layout: build_charts() (uv_charts.hpp) partitions the
// faces into charts of similar normal and flattens each, and pack_atlas()
// (atlas_pack.hpp) places every chart on one square page with a shared uniform
// scale, writing the mesh's uvs / uv_faces.
//
// Commit. build_charts() and pack_atlas() run on a COPY of the ACTIVE model; the
// finished copy replaces that model's mesh in a single move, so an error anywhere
// leaves the chunk untouched. A successful run also clears the same model's
// previously baked texture page (Chunk::model_textures()): the old page
// addresses the old layout and would otherwise be shown, or exported, as if it
// still matched.
//
// Layers: this lives in `common` (it includes common/ and model/), matching
// build_model_task.hpp.

#include <string>
#include <vector>

#include "common/tasks/task.hpp"

namespace alphashape {

/// Build the chunk model's UV atlas. Requires a non-empty mesh; only the
/// "generic" mapping mode and a single texture page are implemented.
class BuildUVTask : public AbstractTask {
public:
    std::string name() const override { return "BuildUV"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
