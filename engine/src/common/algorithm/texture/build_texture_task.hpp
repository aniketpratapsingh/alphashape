#pragma once

// BuildTexture: bake the model's photographs into a texture page (milestone 8).
//
// The task is the adapter between the chunk's data model and the two stages that
// produce the texture:
//
//   * select_face_views() (texture/face_views.hpp) decides, for every mesh face,
//     which input photographs see it. Its inputs are the model and one TexView
//     per enabled camera: the camera's FULL-RESOLUTION photograph, its
//     full-resolution calibration and its world-to-camera pose.
//   * bake_texture() (texture/bake.hpp) walks the atlas texels, samples the
//     assigned views and produces one RGB page.
//
// Requires a UV layout: the mesh must carry uv_faces, i.e. a previous BuildUV
// run. A textured model therefore takes two steps — BuildUV, then BuildTexture —
// as in Metashape's two-step workflow.
//
// Commit. The page is built into a local Image and assigned to page 0 of the
// ACTIVE model's texture list (Chunk::model_textures()) in one statement, so a
// failure during view selection or baking leaves the previous page in place. The
// task never creates a model: the page always belongs to the model that was
// active when it ran, so building another model afterwards leaves this page with
// its own model. The mesh itself is not modified: the bake reads the uv layout
// BuildUV wrote.
//
// `ghosting_filter` is accepted for API compatibility and has no effect yet: the
// ghosting filter (rejecting views whose colour disagrees with the majority) is
// deferred, so the parameter is recorded and a warning is logged when it is
// left at its default of true.

#include <string>
#include <vector>

#include "common/tasks/task.hpp"

namespace alphashape {

/// Bake the chunk model's texture page from its source photographs. Requires a
/// mesh with a UV layout (run BuildUV first) and at least one enabled camera.
class BuildTextureTask : public AbstractTask {
public:
    std::string name() const override { return "BuildTexture"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
