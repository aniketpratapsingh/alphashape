// The built-in registration block.
//
// Every task is registered in one flat sequence at startup: one function, one
// line per task. It lives in the algorithms library,
// not core, so core never depends on concrete algorithms.

#include "common/algorithm/chunk/align_chunks_task.hpp"
#include "common/algorithm/chunk/append_project_task.hpp"
#include "common/algorithm/chunk/duplicate_chunk_task.hpp"
#include "common/algorithm/chunk/merge_chunks_task.hpp"
#include "common/algorithm/elevation/build_dem_task.hpp"
#include "common/algorithm/elevation/export_dem_task.hpp"
#include "common/algorithm/feature/detect_features_task.hpp"
#include "common/algorithm/match/match_photos_task.hpp"
#include "common/algorithm/mesh/build_model_task.hpp"
#include "common/algorithm/mesh/export_tasks.hpp"
#include "common/algorithm/ortho/build_orthomosaic_task.hpp"
#include "common/algorithm/ortho/export_orthomosaic_task.hpp"
#include "common/algorithm/photos/add_photos_task.hpp"
#if defined(ALPHASHAPE_USE_PROJ)
#include "common/algorithm/reference/export_markers_task.hpp"
#include "common/algorithm/reference/export_reference_task.hpp"
#include "common/algorithm/reference/import_markers_task.hpp"
#include "common/algorithm/reference/import_reference_task.hpp"
#endif
#include "common/algorithm/sfm/align_cameras_task.hpp"
#include "common/algorithm/sfm/optimize_cameras_task.hpp"
#include "common/algorithm/stereo/build_depth_maps_task.hpp"
#include "common/algorithm/stereo/build_point_cloud_task.hpp"
#include "common/algorithm/texture/build_texture_task.hpp"
#include "common/algorithm/texture/build_uv_task.hpp"
#include "common/tasks/registry.hpp"

namespace alphashape {

void register_builtin_tasks(TaskRegistry& registry) {
    if (registry.find("DetectFeatures") != nullptr) {
        return;  // already registered: calling twice is a no-op
    }
    registry.register_task(std::make_shared<AddPhotosTask>());
    registry.register_task(std::make_shared<DetectFeaturesTask>());
    registry.register_task(std::make_shared<MatchPhotosTask>());
    registry.register_task(std::make_shared<AlignCamerasTask>());
    registry.register_task(std::make_shared<OptimizeCamerasTask>());
    registry.register_task(std::make_shared<BuildDepthMapsTask>());
    registry.register_task(std::make_shared<BuildPointCloudTask>());
    registry.register_task(std::make_shared<BuildModelTask>());
    registry.register_task(std::make_shared<BuildUVTask>());
    registry.register_task(std::make_shared<BuildTextureTask>());
    registry.register_task(std::make_shared<BuildDemTask>());
    registry.register_task(std::make_shared<ExportDemTask>());
    registry.register_task(std::make_shared<BuildOrthomosaicTask>());
    registry.register_task(std::make_shared<ExportOrthomosaicTask>());
    registry.register_task(std::make_shared<ExportModelTask>());
    registry.register_task(std::make_shared<ExportPointCloudTask>());
    registry.register_task(std::make_shared<DuplicateChunkTask>());
    registry.register_task(std::make_shared<MergeChunksTask>());
    registry.register_task(std::make_shared<AppendProjectTask>());
    registry.register_task(std::make_shared<AlignChunksTask>());
#if defined(ALPHASHAPE_USE_PROJ)
    // WP-13.2: reference import/export depends on the CRS engine, so the tasks
    // exist only in a PROJ build (the sources are compiled in the same guard).
    registry.register_task(std::make_shared<ImportReferenceTask>());
    registry.register_task(std::make_shared<ExportReferenceTask>());
    // WP-13.4: marker projection import/export. They share the reference
    // CSV/XML helpers, so they live behind the same PROJ guard.
    registry.register_task(std::make_shared<ImportMarkersTask>());
    registry.register_task(std::make_shared<ExportMarkersTask>());
#endif
    // Later milestones append here: ...
}

TaskRegistry& builtin_registry() {
    TaskRegistry& registry = TaskRegistry::instance();
    register_builtin_tasks(registry);
    return registry;
}

}  // namespace alphashape
