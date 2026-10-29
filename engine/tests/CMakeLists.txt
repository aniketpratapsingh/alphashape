find_package(GTest REQUIRED)

add_executable(alphashape_tests
    test_platform.cpp
    test_params.cpp
    test_registry.cpp
    test_undo_stack.cpp
    test_model_commands.cpp
    test_data_model.cpp
    test_project_roundtrip.cpp
    test_project_v2.cpp
    test_detector.cpp
    test_image_io.cpp
    test_exif.cpp
    test_camera_model.cpp
    test_downsample.cpp
    test_add_photos.cpp
    test_two_view.cpp
    test_five_point.cpp
    test_matcher.cpp
    test_preselection.cpp
    test_tracks.cpp
    test_match_photos.cpp
    test_absolute_pose.cpp
    test_triangulation.cpp
    test_bundle_adjustment.cpp
    test_structure_builder.cpp
    test_align_cameras.cpp
    test_optimize_cameras.cpp
    test_geodesy.cpp
    test_projection.cpp
    test_similarity.cpp
    test_georeference.cpp
    test_view_selection.cpp
    test_fusion.cpp
    test_depth_filter.cpp
    test_patch_match.cpp
    test_patch_match_pyramid.cpp
    test_census.cpp
    test_patch_match_gpu.cpp
    test_build_dense.cpp
    test_voxel_grid.cpp
    test_predicates.cpp
    test_delaunay2d.cpp
    test_sibson.cpp
    test_mesh_io.cpp
    test_marching_tetrahedra.cpp
    test_mesh_ops.cpp
    test_tv_fusion.cpp
    test_tv_fusion_gpu.cpp
    test_poisson.cpp
    test_poisson_gpu.cpp
    test_decimate.cpp
    test_build_model.cpp
    test_export.cpp
    test_obj.cpp
    test_uv_charts.cpp
    test_atlas_pack.cpp
    test_rasterize.cpp
    test_face_views.cpp
    test_bake.cpp
    test_build_texture.cpp
    test_geotiff.cpp
    test_interpolate.cpp
    test_tgv_denoise.cpp
    test_aerial_fixture.cpp
    test_build_dem.cpp
    test_ortho.cpp
    test_ortho_gain.cpp
    test_build_orthomosaic.cpp
    test_chunk_ops.cpp
    test_align_chunks.cpp
    test_gpu.cpp
)

# M12: the PROJ-backed CRS suites only exist when PROJ is enabled (the CMake
# option gates the library, so the tests would have nothing to link).
if(ALPHASHAPE_USE_PROJ)
    target_sources(alphashape_tests PRIVATE
        test_crs_context.cpp
        test_crs.cpp
        test_crs_transform.cpp
        test_datum_transform.cpp
        test_local_frame.cpp
        test_geoid.cpp)
    # PROJ is linked PRIVATE by alphashape_crs, and this test includes <proj.h>
    # directly for the raw-API assertions; linking the interface target keeps
    # the include directories available without a second package lookup.
    target_link_libraries(alphashape_tests PRIVATE alphashape_proj)
endif()

target_link_libraries(alphashape_tests PRIVATE
    alphashape_algorithms
    GTest::gtest
    GTest::gtest_main)

# Where the checked-in fixtures live, for tests that compare against them.
target_compile_definitions(alphashape_tests PRIVATE
    ALPHASHAPE_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

# The GPU suite starts the real CLI once, to prove that a build with the GPU
# layer compiled in cannot fail to load when no Vulkan runtime is installed (the
# process either lists a device or explains why it cannot). Defined only when the
# CLI exists; the test skips itself otherwise.
if(ALPHASHAPE_BUILD_CLI AND TARGET alphashape-cli)
    # $<TARGET_FILE:...> expands to a forward-slash path on every platform (the
    # GUI suites bake $<TARGET_FILE:alphashape-gui> the same way), so the
    # literal has no backslashes to misread as escapes. $<SHELL_PATH:...> is
    # the wrong tool here: it produces the native cmd form, and the GitHub
    # runner's D:\a\... turned into \a and \s and failed under -Werror.
    target_compile_definitions(alphashape_tests PRIVATE
        "ALPHASHAPE_CLI_EXE=\"$<TARGET_FILE:alphashape-cli>\"")
    add_dependencies(alphashape_tests alphashape-cli)
endif()

# tests/make_v1_fixture.cpp -- regenerates tests/fixtures/v1_project.zip.
# Deliberately not part of the test binary: the fixture is checked in, and this
# target only exists so changing tests/v1_fixture.hpp can be followed by a
# rebuild of the bytes.
add_executable(make_v1_fixture EXCLUDE_FROM_ALL make_v1_fixture.cpp)
target_link_libraries(make_v1_fixture PRIVATE alphashape_algorithms)

# Under ALPHASHAPE_SANITIZE two third-party environment fixes are needed, and
# both must apply to *discovery* as well: CMake's gtest_discover_tests() runs the
# binary with --gtest_list_tests --gtest_output=json:... at build time, and
# GoogleTest's JSON printer trips ASan's container-overflow check (a libc++
# over-allocation false positive, no frame of ours — see cmake/sanitizers/).
# TEST_LAUNCHER is honored by gtest_discover_tests() for both the discovery
# command and the real test invocations, so a single cmake -E env launcher
# covers both. LeakSanitizer is unavailable on macOS arm64, hence detect_leaks=0.
if(ALPHASHAPE_SANITIZE)
    set(_alphashape_asan_supp "${PROJECT_SOURCE_DIR}/cmake/sanitizers/asan.supp")
    # ASan's option parser splits an unquoted value at whitespace, and this
    # source tree lives under a path containing a space, so the suppressions
    # path is wrapped in single quotes — ASan strips them before opening it.
    # ALPHASHAPE_SANITIZE_ASAN_OPTIONS is the leading part of ASAN_OPTIONS (default
    # keeps LeakSanitizer off); the Linux sanitizer CI job passes
    # "detect_leaks=1:abort_on_error=1". lsan.supp only matters when leak detection is on.
    set(_alphashape_lsan_supp "${PROJECT_SOURCE_DIR}/cmake/sanitizers/lsan.supp")
    set_target_properties(alphashape_tests PROPERTIES TEST_LAUNCHER
        "${CMAKE_COMMAND};-E;env;ASAN_OPTIONS=${ALPHASHAPE_SANITIZE_ASAN_OPTIONS}:suppressions='${_alphashape_asan_supp}';LSAN_OPTIONS=suppressions='${_alphashape_lsan_supp}';UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1")
    # The instrumented Debug binary takes longer than the 5 s default just to
    # start up and enumerate its tests, and the slowest tests (StructureBuilder,
    # under ASan) run for several minutes.
    set(_alphashape_discovery_args DISCOVERY_TIMEOUT 120)
    set(_alphashape_test_args PROPERTIES TIMEOUT 3600)
endif()

include(GoogleTest)

# The GPU TV-L1 fusion port landed (M10): its suite is registered like the GPU
# Poisson port's. Every TvFusionGpu test either runs the device path or skips
# with a reason (no device), and on a software device the CPU fallback path is
# asserted. The variable stays until the remaining ports are registered.
set(_alphashape_gpu_ports_filter "")

gtest_discover_tests(alphashape_tests
    ${_alphashape_discovery_args}
    ${_alphashape_test_args}
    TEST_FILTER ${_alphashape_gpu_ports_filter})
