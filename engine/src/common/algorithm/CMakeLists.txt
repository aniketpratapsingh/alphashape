# alphashape_algorithms — the pipeline stages, plus the built-in task
# registration block (builtin_tasks.cpp). Front ends link this target.
# v0 implements feature detection only; later milestones add match/, sfm/,
# stereo/, depth/, elevation/ (see the plan's build order).

add_library(alphashape_algorithms STATIC
    builtin_tasks.cpp
    chunk/align_chunks_task.cpp
    chunk/append_project_task.cpp
    chunk/duplicate_chunk_task.cpp
    chunk/merge_chunks_task.cpp
    photos/add_photos_task.cpp
    feature/dog_detector.cpp
    feature/mldb_descriptor.cpp
    feature/detect_features_task.cpp
    match/hamming_matcher.cpp
    match/preselection.cpp
    match/tracks.cpp
    match/match_photos_task.cpp
    sfm/five_point.cpp
    sfm/essential.cpp
    sfm/absolute_pose.cpp
    sfm/triangulation.cpp
    sfm/bundle_adjustment.cpp
    sfm/structure_builder.cpp
    sfm/align_cameras_task.cpp
    sfm/optimize_cameras_task.cpp
    sfm/georeference.cpp
    sfm/similarity.cpp
    stereo/view_selection.cpp
    stereo/patch_match.cpp
    stereo/patch_match_gpu.cpp
    stereo/depth_filter.cpp
    stereo/fusion.cpp
    stereo/build_depth_maps_task.cpp
    stereo/build_point_cloud_task.cpp
    mesh/tv_fusion.cpp
    mesh/tv_fusion_gpu.cpp
    mesh/poisson.cpp
    mesh/poisson_gpu.cpp
    mesh/build_model_task.cpp
    mesh/export_tasks.cpp
    texture/face_views.cpp
    texture/bake.cpp
    texture/build_uv_task.cpp
    texture/build_texture_task.cpp
    elevation/interpolate.cpp
    elevation/tgv_denoise.cpp
    elevation/surface_frame.cpp
    elevation/dem_builder.cpp
    elevation/build_dem_task.cpp
    elevation/export_dem_task.cpp
    ortho/orthorectify.cpp
    ortho/seamlines.cpp
    ortho/blend.cpp
    ortho/build_orthomosaic_task.cpp
    ortho/export_orthomosaic_task.cpp
)

target_include_directories(alphashape_algorithms PUBLIC ${ALPHASHAPE_INCLUDE_DIR})
target_link_libraries(alphashape_algorithms PUBLIC
    alphashape_core
    alphashape_image
    alphashape_gpu)
target_link_libraries(alphashape_algorithms PRIVATE Ceres::ceres)

# M12 §4.1: the algorithms layer links the CRS engine explicitly (it already
# reaches it through alphashape_model, but WP-12.8 makes DEM/ortho tasks call
# crs:: directly). PRIVATE: no algorithm header exposes PROJ or crs headers.
if(TARGET alphashape_crs)
    target_link_libraries(alphashape_algorithms PRIVATE alphashape_crs)
endif()
