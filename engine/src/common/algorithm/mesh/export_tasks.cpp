// ExportModel / ExportPointCloud: write the chunk's derived geometry to disk.
//
// The file-output adapters on top of model/mesh/ply.hpp (PLY) and
// model/mesh/obj.hpp (OBJ). Both tasks share one parameter set and one resolve
// step; they differ only in which container they read and in the message they
// raise when it is empty.
//
// OBJ is a mesh-only format, so ExportModel additionally supports it: when the
// model carries a texture page and `save_texture` is set, the page is written
// next to the OBJ as a companion PNG and referenced from the material library.
//
// Read-only w.r.t. the chunk. `save_colors` / `save_confidence` are honoured on
// a COPY of the geometry: clearing the chunk's own attribute vectors here would
// make an export silently destructive to the reconstruction in memory.

#include "common/algorithm/mesh/export_tasks.hpp"

#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/core/assets.hpp"
#include "common/core/document.hpp"
#include "image/io/image_io.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/obj.hpp"
#include "model/mesh/ply.hpp"
#include "model/mesh/point_cloud.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

/// The resolved, validated parameters of one export invocation.
struct ExportOptions {
    std::string path;
    std::string format;
    bool save_colors = true;
    bool save_confidence = true;
    bool world_coordinates = false;
    /// ExportModel only: write the model's texture page as a companion PNG next
    /// to the OBJ and reference it from the .mtl. Ignored by ExportPointCloud
    /// and by the PLY writer.
    bool save_texture = true;
    /// Asset to export, or -1 for the active one.
    std::int64_t asset_id = -1;
};

/// Resolve `asset_id` against one of the chunk's asset vectors: -1 means the
/// active asset (nullptr when there is none, so the caller raises its own
/// "No model" / "No point cloud" error), anything else must name an existing
/// slot. Throws ValueError naming the id when an explicit id does not resolve.
template <class Slot>
const Slot* resolve_asset(const std::vector<Slot>& slots, std::int32_t active_id,
                          std::int64_t asset_id, const char* task, const char* kind) {
    if (asset_id > std::numeric_limits<std::int32_t>::max()) {
        throw ValueError(std::string(task) + ": asset_id " + std::to_string(asset_id) +
                         " is out of range");
    }
    const std::int32_t wanted =
        asset_id < 0 ? active_id : static_cast<std::int32_t>(asset_id);
    const Slot* slot = find_asset(slots, wanted);
    if (slot == nullptr && asset_id >= 0) {
        throw ValueError(std::string(task) + ": no " + kind + " asset with id " +
                         std::to_string(asset_id) + " in this chunk");
    }
    return slot;
}

/// Resolve and validate the shared export parameter set. Throws ValueError when
/// `path` is empty or `format` names a writer that does not exist. Only the mesh
/// writer can emit OBJ, so `allow_obj` gates that value per task.
ExportOptions resolve_export_options(const Params& params, bool allow_obj) {
    ExportOptions options;

    options.path = params.get_string("path");
    if (options.path.empty()) {
        throw ValueError("path is required");
    }

    options.format = params.get_string("format");
    const bool known = options.format == "ply" || (allow_obj && options.format == "obj");
    if (!known) {
        throw ValueError("Unsupported format '" + options.format + "'; valid formats: " +
                         (allow_obj ? "ply, obj" : "ply"));
    }

    options.save_colors = params.get_bool("save_colors");
    options.save_confidence = params.get_bool("save_confidence");
    options.world_coordinates = params.get_bool("world_coordinates");
    options.save_texture = params.get_bool("save_texture");
    options.asset_id = params.get_int("asset_id");
    return options;
}

/// The chunk's transform when world coordinates were requested, else nullptr —
/// what the PLY writers take to lift chunk-local geometry into world space.
const Eigen::Matrix4d* export_transform(const Chunk& chunk, const ExportOptions& options) {
    return options.world_coordinates ? &chunk.transform() : nullptr;
}

/// Declare the parameter set both export tasks share.
void declare_export_params(Params& params) {
    // The output file path. Required: an empty value is rejected at apply time.
    params.declare({"path", ParamType::String, std::string{},
                    "Destination file path for the exported geometry."});
    params.declare({"format", ParamType::String, std::string("ply"),
                    "Output file format: 'ply' or (ExportModel only) 'obj'."});
    params.declare({"save_colors", ParamType::Bool, true,
                    "Write per-vertex / per-point RGB colours when present."});
    params.declare({"save_confidence", ParamType::Bool, true,
                    "Write the per-vertex / per-point confidence when present."});
    params.declare({"world_coordinates", ParamType::Bool, false,
                    "Apply the chunk transform so positions are in world coordinates."});
    params.declare({"save_texture", ParamType::Bool, true,
                    "ExportModel + OBJ: write the model's texture page as a companion "
                    "PNG and reference it from the .mtl."});
    params.declare({"asset_id", ParamType::Int, std::int64_t{-1},
                    "Id of the asset to export. -1 selects the active one."});
}

/// Drop the attribute vectors the caller asked not to export, on the COPY being
/// written (never on the chunk).
void strip_attributes(Mesh& mesh, const ExportOptions& options) {
    if (!options.save_colors) {
        mesh.colours.clear();
    }
    if (!options.save_confidence) {
        mesh.confidence.clear();
    }
}

/// The companion PNG path for an OBJ at `obj_path`: same directory and stem with
/// a ".png" extension, matching the `<stem>.mtl` write_obj will write beside it.
std::string texture_path_for(const std::string& obj_path) {
    const fs::path obj(obj_path);
    return (obj.parent_path() / (obj.stem().string() + ".png")).string();
}

/// Export the mesh as OBJ. When the model carries a texture page and
/// `save_texture` is set, the page is written as a companion PNG and the OBJ is
/// handed that PNG's file NAME: write_obj writes the .mtl next to the .obj, so a
/// bare name resolves relative to the material library and the .obj/.mtl/.png
/// trio can be moved together. Without a page, or with save_texture cleared, the
/// OBJ is written with no material library and no texture coordinates.
void write_model_obj(const std::string& obj_path, const Mesh& mesh, const ModelAsset& asset,
                     const Chunk& chunk, const ExportOptions& options) {
    const std::vector<Image>& pages = asset.textures;
    const bool has_page = !pages.empty() && !pages[0].empty();
    std::string texture_name;
    if (options.save_texture && has_page) {
        const std::string png_path = texture_path_for(obj_path);
        write_png(png_path, pages[0]);
        texture_name = fs::path(png_path).filename().string();
    }
    write_obj(obj_path, mesh, texture_name, export_transform(chunk, options));
}

void strip_attributes(PointCloud& cloud, const ExportOptions& options) {
    if (!options.save_colors) {
        cloud.colours.clear();
    }
    if (!options.save_confidence) {
        cloud.confidence.clear();
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// ExportModel
// ---------------------------------------------------------------------------

void ExportModelTask::declare_params(Params& params) const { declare_export_params(params); }

std::vector<Stage> ExportModelTask::stages() const {
    return {Stage{"validate", 0}, Stage{"write", 0}};
}

void ExportModelTask::apply(Document& document, Params& params, const Progress& progress) {
    progress.update("validate", 0.0);
    const ExportOptions options = resolve_export_options(params, /*allow_obj=*/true);

    const Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("ExportModel needs an active chunk");
    }
    const ModelAsset* asset =
        resolve_asset(chunk->models(), chunk->active_model_id(), options.asset_id,
                      "ExportModel", "model");
    if (asset == nullptr || !asset->data) {
        throw Error("No model");
    }
    if (asset->data->empty()) {
        throw Error("No model");
    }

    // Copy BEFORE stripping: the chunk's mesh keeps its colours/confidence.
    Mesh mesh = *asset->data;
    strip_attributes(mesh, options);
    progress.update("validate", 1.0);

    progress.update("write", 0.0);
    if (options.format == "obj") {
        write_model_obj(options.path, mesh, *asset, *chunk, options);
    } else {
        write_ply(options.path, mesh, export_transform(*chunk, options));
    }
    progress.update("write", 1.0);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "ExportModel: " << mesh.faces.size() << " faces, "
                            << mesh.vertices.size() << " vertices (asset " << asset->id
                            << ") -> " << options.path << " (" << options.format << ", "
                            << (options.world_coordinates ? "world)" : "local)");
    }
}

// ---------------------------------------------------------------------------
// ExportPointCloud
// ---------------------------------------------------------------------------

void ExportPointCloudTask::declare_params(Params& params) const { declare_export_params(params); }

std::vector<Stage> ExportPointCloudTask::stages() const {
    return {Stage{"validate", 0}, Stage{"write", 0}};
}

void ExportPointCloudTask::apply(Document& document, Params& params, const Progress& progress) {
    progress.update("validate", 0.0);
    const ExportOptions options = resolve_export_options(params, /*allow_obj=*/false);

    const Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("ExportPointCloud needs an active chunk");
    }
    const PointCloudAsset* asset =
        resolve_asset(chunk->point_clouds(), chunk->active_point_cloud_id(),
                      options.asset_id, "ExportPointCloud", "point cloud");
    if (asset == nullptr || !asset->data || asset->data->points.empty()) {
        throw Error("No point cloud");
    }

    PointCloud cloud = *asset->data;
    strip_attributes(cloud, options);
    progress.update("validate", 1.0);

    progress.update("write", 0.0);
    write_ply(options.path, cloud, export_transform(*chunk, options));
    progress.update("write", 1.0);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "ExportPointCloud: " << cloud.points.size() << " points (asset "
                            << asset->id << ") -> " << options.path
                            << (options.world_coordinates ? " (world)" : "");
    }
}

}  // namespace alphashape
