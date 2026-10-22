// BuildUV: lay the reconstructed mesh out on a texture atlas (milestone 8).
//
// Two stages, both pure geometry on the chunk model:
//
//   * charts   build_charts() (model/mesh/uv_charts.hpp) partitions the mesh's
//              faces into charts of similar normal and flattens each one.
//   * pack     pack_atlas() (model/mesh/atlas_pack.hpp) rotates, scales and
//              places every chart on one square page with a shared uniform
//              scale, then writes mesh.uvs / mesh.uv_faces.
//
// Both run against a COPY of chunk.model(). The finished copy replaces the
// chunk's mesh with a single move, so a validation error, a chart failure or a
// pack failure leaves the chunk exactly as it was.
//
// Clearing textures. A previous BuildTexture page addresses the layout that
// used to be here. Re-running BuildUV invalidates it, so a successful run
// clears chunk.model_textures(). Baking again is a separate BuildTexture call.

#include "common/algorithm/texture/build_uv_task.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/core/document.hpp"
#include "model/mesh/atlas_pack.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/uv_charts.hpp"

namespace alphashape {
namespace {

// -- tuning constants (ours) ------

/// Texture page side in texels: [256, 16384], default 4096 (as in Metashape's
/// public Python API).
constexpr std::int64_t kTextureSizeMin = 256;
constexpr std::int64_t kTextureSizeMax = 16384;
constexpr std::int64_t kTextureSizeDefault = 4096;

/// Gap kept between charts and between a chart and the page border. Four texels
/// absorbs the bilinear bleed of a conventional 8-bit filter kernel.
constexpr int kAtlasPaddingPx = 4;

// -- validated parameter bundle ----------------------------------------------

/// The validated `mapping_mode` choice. Only Generic is implemented; the other
/// names exist so a caller gets a "not supported yet" message that names the
/// mode rather than a generic rejection.
enum class MappingMode { Generic };

/// Everything apply() needs, resolved once so the stage helpers take plain
/// values rather than the raw Params.
struct ResolvedOptions {
    MappingMode mapping = MappingMode::Generic;
    std::int64_t page_count = 1;
    std::int64_t texture_size = kTextureSizeDefault;
};

MappingMode resolve_mapping_mode(const std::string& value) {
    if (value == "generic") {
        return MappingMode::Generic;
    }
    // Metashape's API names these modes; we implement only "generic". The
    // message names the mode so a caller knows the value was understood and is
    // simply not built yet (UnsupportedTaskError is reserved for a task name
    // that is not in the registry, so a plain Error is the right type here).
    if (value == "orthophoto" || value == "spherical" || value == "camera") {
        throw Error(value + " mapping is not supported yet");
    }
    throw ValueError("BuildUV: mapping_mode must be one of \"generic\", \"orthophoto\", "
                     "\"spherical\", \"camera\" (got \"" +
                     value + "\")");
}

ResolvedOptions resolve_options(const Params& params) {
    ResolvedOptions resolved;
    resolved.mapping = resolve_mapping_mode(params.get_string("mapping_mode"));

    resolved.page_count = params.get_int("page_count");
    if (resolved.page_count < 1) {
        throw ValueError("BuildUV: page_count must be at least 1");
    }
    if (resolved.page_count > 1) {
        throw Error("page_count > 1 is not supported yet");
    }

    resolved.texture_size = params.get_int("texture_size");
    if (resolved.texture_size < kTextureSizeMin || resolved.texture_size > kTextureSizeMax) {
        throw ValueError("BuildUV: texture_size must be between " +
                         std::to_string(kTextureSizeMin) + " and " +
                         std::to_string(kTextureSizeMax));
    }
    return resolved;
}

/// Split the mesh into charts and pack them onto the page, on `mesh` in place.
PackResult pack_mesh(Mesh& mesh, const ResolvedOptions& resolved, ChartStats* stats) {
    ChartOptions chart_options;  // defaults: 40 degree cone, 50000 faces per chart
    const std::vector<Chart> charts = build_charts(mesh, chart_options, stats);
    if (charts.empty()) {
        throw Error("BuildUV: build_charts produced no charts");
    }
    PackOptions pack_options;
    pack_options.texture_size = static_cast<int>(resolved.texture_size);
    pack_options.padding_px = kAtlasPaddingPx;
    return pack_atlas(mesh, charts, pack_options);
}

}  // namespace

void BuildUVTask::declare_params(Params& params) const {
    // Names follow Metashape's public Python API; the accepted values
    // and defaults are ours.
    params.declare({"mapping_mode", ParamType::String, std::string("generic"),
                    "How the surface is unwrapped. Only \"generic\" (per-chart "
                    "flattening) is supported."});
    params.declare({"page_count", ParamType::Int, std::int64_t{1},
                    "Number of texture pages. Only 1 is supported."});
    params.declare({"texture_size", ParamType::Int, kTextureSizeDefault,
                    "Texture page side in pixels."});
}

std::vector<Stage> BuildUVTask::stages() const {
    return {Stage{"charts", 0}, Stage{"pack", 0}};
}

void BuildUVTask::apply(Document& document, Params& params, const Progress& progress) {
    // -- resolve and validate every parameter before touching the chunk ------
    progress.update("charts", 0.0);
    const ResolvedOptions resolved = resolve_options(params);

    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("BuildUV needs an active chunk");
    }
    // Read through a CONST view: the non-const accessor would lazily create an
    // empty model on a chunk that has none, so a rejected run would leave one.
    const Chunk& source_chunk = *chunk;
    if (source_chunk.model().empty()) {
        throw ValueError("BuildUV: build a model first");
    }

    // -- charts + pack, on a copy --------------------------------------------
    Mesh mesh = source_chunk.model();
    ChartStats stats;
    const PackResult pack = pack_mesh(mesh, resolved, &stats);
    progress.update("charts", 1.0);
    const std::size_t chart_uvs = mesh.uvs.size();
    const std::size_t chart_faces = mesh.uv_faces.size();
    progress.update("pack", 1.0);

    // -- commit --------------------------------------------------------------
    // The stale page addresses the previous layout; dropping it here means a
    // texture can never be shown or exported against coordinates it no longer
    // matches. Baking is a separate BuildTexture run. (The declared parameters
    // are recorded into the chunk meta by run_task itself.)
    const std::size_t stale_pages = source_chunk.model_textures().size();
    chunk->model() = std::move(mesh);
    chunk->model_textures().clear();

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "BuildUV: " << chart_faces << " faces, " << chart_uvs
                            << " uv coordinates, fill " << pack.fill_ratio << ", "
                            << stats.overlap_splits << " overlap splits"
                            << (stale_pages > 0 ? " (cleared stale texture)" : "");
    }
}

}  // namespace alphashape
