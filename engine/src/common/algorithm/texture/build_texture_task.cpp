// BuildTexture: bake the model's photographs into a texture page (milestone 8).
//
//   * views    one TexView per ENABLED camera: the full-resolution photograph
//              (decoded once, here), the full-resolution sensor calibration and
//              the world-to-camera pose from Camera::pose. Disabled cameras are
//              skipped: their pose is identity, so they would sample noise.
//   * select   select_face_views() (texture/face_views.hpp) labels every face
//              with the views that see it and picks one best view per face.
//   * bake     bake_texture() (texture/bake.hpp) walks the atlas texels and
//              writes the page.
//
// A UV layout is a precondition, not something this task builds: the mesh must
// carry uv_faces from a previous BuildUV run ("Build UV first").
//
// Commit. The page is a local Image until the last statement of apply(), where
// it becomes page 0 of chunk.model_textures(). Any failure keeps the previous
// page. The mesh is never modified; the bake only reads the uv layout.

#include "common/algorithm/texture/build_texture_task.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/texture/bake.hpp"
#include "common/algorithm/texture/face_views.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "image/io/image_io.hpp"
#include "model/camera/calibration.hpp"
#include "model/mesh/mesh.hpp"

namespace alphashape {
namespace {

// -- tuning constants (ours) ------

/// Texture page side in texels; the same [256, 16384] window BuildUV accepts.
/// Default 4096, as in Metashape's public API.
constexpr std::int64_t kTextureSizeMin = 256;
constexpr std::int64_t kTextureSizeMax = 16384;
constexpr std::int64_t kTextureSizeDefault = 4096;

/// Gutter filled outward beyond each chart's border so bilinear filtering at a
/// chart edge does not pull in uncovered texels.
constexpr int kDilationPx = 4;

// -- validated parameter bundle ----------------------------------------------

/// Everything apply() needs, resolved once so the bake helper takes plain values
/// rather than the raw Params.
struct ResolvedOptions {
    texturing::BlendingMode blending = texturing::BlendingMode::Mosaic;
    std::int64_t texture_size = kTextureSizeDefault;
    bool fill_holes = true;
    bool ghosting_filter = true;
    bool color_correction = true;
};

ResolvedOptions resolve_options(const Params& params) {
    ResolvedOptions resolved;
    resolved.blending = texturing::parse_blending_mode(params.get_string("blending_mode"));

    resolved.texture_size = params.get_int("texture_size");
    if (resolved.texture_size < kTextureSizeMin || resolved.texture_size > kTextureSizeMax) {
        throw ValueError("BuildTexture: texture_size must be between " +
                         std::to_string(kTextureSizeMin) + " and " +
                         std::to_string(kTextureSizeMax));
    }
    resolved.fill_holes = params.get_bool("fill_holes");
    resolved.ghosting_filter = params.get_bool("ghosting_filter");
    resolved.color_correction = params.get_bool("color_correction");
    return resolved;
}

/// Resolve a frame's recorded path against the document's directory, the same
/// rule AddPhotos / BuildPointCloud / BuildModel use.
std::filesystem::path resolve_frame_path(const Document& document, const std::string& path) {
    const std::filesystem::path recorded(path);
    if (recorded.is_absolute() || document.path().empty()) {
        return recorded;
    }
    return std::filesystem::path(document.path()).parent_path() / recorded;
}

Eigen::Matrix3d pose_R(const Transform& pose) { return pose.block<3, 3>(0, 0); }
Eigen::Vector3d pose_t(const Transform& pose) { return pose.block<3, 1>(0, 3); }

/// One TexView per enabled camera, in camera order. `images` holds the decoded
/// photographs and outlives the views (TexView borrows the pointer), so the two
/// are returned together and the caller keeps both alive through the bake.
struct ViewSet {
    std::vector<Image> images;               ///< parallel to views
    std::vector<texturing::TexView> views;   ///< one per enabled camera
};

ViewSet collect_views(const Document& document, const Chunk& chunk) {
    std::unordered_map<std::int32_t, const Frame*> frame_of;
    for (const auto& frame : chunk.frames()) {
        if (frame->camera_id() >= 0) {
            frame_of.emplace(frame->camera_id(), frame.get());
        }
    }

    ViewSet set;
    for (const Camera& camera : chunk.cameras()) {
        if (!camera.enabled) {
            continue;  // unaligned: identity pose, nothing to sample
        }
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk.sensors().size())) {
            throw ValueError("BuildTexture: camera " + std::to_string(camera.id) +
                             " has no sensor");
        }
        const auto frame_it = frame_of.find(camera.id);
        if (frame_it == frame_of.end()) {
            throw ValueError("BuildTexture: camera " + std::to_string(camera.id) +
                             " has no linked frame to sample from");
        }
        set.images.push_back(read_image(resolve_frame_path(document, frame_it->second->path())
                                            .string()));

        texturing::TexView view;
        view.camera_id = camera.id;
        view.image = nullptr;  // set below, once the vector has stopped growing
        view.calibration =
            chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration;
        view.R = pose_R(camera.pose);
        view.t = pose_t(camera.pose);
        set.views.push_back(std::move(view));
    }
    // Resize above may have reallocated, so bind the image pointers last.
    for (std::size_t i = 0; i < set.views.size(); ++i) {
        set.views[i].image = &set.images[i];
    }
    return set;
}

}  // namespace

void BuildTextureTask::declare_params(Params& params) const {
    // Names follow Metashape's public Python API; the accepted values and
    // defaults are ours. `color_correction` is our own addition.
    params.declare({"blending_mode", ParamType::String, std::string("mosaic"),
                    "How views combine on a texel: \"mosaic\", \"average\" or \"disabled\"."});
    params.declare({"texture_size", ParamType::Int, kTextureSizeDefault,
                    "Texture page side in pixels."});
    params.declare({"fill_holes", ParamType::Bool, true,
                    "Fill texels no view covers from the surrounding covered texels."});
    params.declare({"ghosting_filter", ParamType::Bool, true,
                    "Accepted for compatibility. Not implemented yet: no effect."});
    params.declare({"color_correction", ParamType::Bool, true,
                    "Level each view's gain before mixing (ours; not in Metashape's API)."});
}

std::vector<Stage> BuildTextureTask::stages() const {
    return {Stage{"select_views", 0}, Stage{"bake", 0}};
}

void BuildTextureTask::apply(Document& document, Params& params, const Progress& progress) {
    // -- resolve and validate every parameter before touching the chunk ------
    progress.update("select_views", 0.0);
    const ResolvedOptions resolved = resolve_options(params);

    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("BuildTexture needs an active chunk");
    }
    // Read through a CONST view: the non-const accessor would lazily create an
    // empty model on a chunk that has none, so a rejected run would leave one.
    const Chunk& source_chunk = *chunk;
    const Mesh& mesh = source_chunk.model();
    if (mesh.empty()) {
        throw ValueError("BuildTexture: build a model first");
    }
    if (mesh.uv_faces.empty()) {
        throw ValueError("BuildTexture: the model has no texture coordinates; "
                         "run BuildUV first");
    }
    if (resolved.ghosting_filter && Log::enabled(LogLevel::Warn)) {
        log(LogLevel::Warn) << "BuildTexture: ghosting_filter is not implemented yet "
                               "and has no effect";
    }

    ViewSet set = collect_views(document, *chunk);
    if (set.views.empty()) {
        throw ValueError("BuildTexture: no enabled cameras to sample from");
    }

    // -- view selection ------------------------------------------------------
    texturing::FaceViewOptions view_options;  // defaults: downscale 2, cos 0.15, smoothing
    const texturing::FaceViews face_views = texturing::select_face_views(mesh, set.views,
                                                                         view_options);
    progress.update("select_views", 1.0);

    // -- bake ----------------------------------------------------------------
    progress.update("bake", 0.0);
    texturing::BakeOptions bake_options;
    bake_options.blending = resolved.blending;
    bake_options.fill_holes = resolved.fill_holes;
    bake_options.color_correction = resolved.color_correction;
    bake_options.dilation_px = kDilationPx;
    Image page = texturing::bake_texture(mesh, set.views, face_views,
                                         static_cast<int>(resolved.texture_size), bake_options);
    progress.update("bake", 1.0);

    // -- commit --------------------------------------------------------------
    if (chunk->model_textures().empty()) {
        chunk->model_textures().emplace_back(std::move(page));
    } else {
        chunk->model_textures()[0] = std::move(page);  // replace, keep the count
    }

    // The declared parameters are recorded into the chunk meta by run_task, so
    // apply() writes nothing here beyond the page itself.

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "BuildTexture: " << set.views.size() << " views -> "
                            << resolved.texture_size << "x" << resolved.texture_size << " "
                            << texturing::to_string(resolved.blending) << " page";
    }
}

}  // namespace alphashape
