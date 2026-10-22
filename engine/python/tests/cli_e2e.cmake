# End-to-end CLI check, run as a CTest via `cmake -P`.
#
# It exercises the whole front end the way a user would: build a project from
# two real camera JPEGs (so --add-images is proven on EXIF-bearing files, not
# just PNMs), run a task against it with a --set override, drive AddPhotos on an
# existing project with a --set list, then confirm a bad task name fails with the
# documented exit code. Nothing here inspects engine internals, so it catches
# wiring breaks the unit tests cannot.
#
# Inputs (passed with -D from python/CMakeLists.txt):
#   CLI       full path to the built alphashape-cli
#   FIXTURES  directory holding the data fixtures
#   WORK      scratch directory for the generated project

if(NOT DEFINED CLI OR NOT DEFINED FIXTURES OR NOT DEFINED WORK)
    message(FATAL_ERROR "cli_e2e.cmake requires -DCLI= -DFIXTURES= -DWORK=")
endif()

file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")
set(PROJECT "${WORK}/e2e.psx")

# 1. Create a project from two camera JPEGs. This goes through AddPhotos, so a
#    failure here means either the CLI wiring or the EXIF/sensor path is broken.
execute_process(
    COMMAND "${CLI}" --new "${PROJECT}"
            --add-images "${FIXTURES}/cam_a1.jpg" "${FIXTURES}/cam_b1.jpg"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "--new exited ${rc}\nstdout: ${out}\nstderr: ${err}")
endif()
if(NOT EXISTS "${PROJECT}")
    message(FATAL_ERROR "--new did not create ${PROJECT}")
endif()

# 1b. --list-params AddPhotos must show the list type.
execute_process(
    COMMAND "${CLI}" --list-params AddPhotos
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "--list-params AddPhotos exited ${rc}\nstderr: ${err}")
endif()
if(NOT out MATCHES "filenames[ \t]+StringList")
    message(FATAL_ERROR "--list-params did not show filenames as StringList:\n${out}")
endif()

# 2. Run a task with an override; must succeed.
execute_process(
    COMMAND "${CLI}" --project "${PROJECT}" --task DetectFeatures --set keypoint_limit=40
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "DetectFeatures exited ${rc}\nstdout: ${out}\nstderr: ${err}")
endif()

# 3. AddPhotos on the existing project, with the list bound via --set. The value
#    is a JSON array, which is the one text form a StringList takes.
execute_process(
    COMMAND "${CLI}" --project "${PROJECT}" --task AddPhotos
            --set "filenames=[\"${FIXTURES}/cam_a2.jpg\"]"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "AddPhotos via --set exited ${rc}\nstdout: ${out}\nstderr: ${err}")
endif()

# 3b. MatchPhotos on the three-view scene fixtures: create the project from the
#     scene images, then match with the fixture-appropriate keypoint cap. This
#     exercises the whole detect -> preselect -> match -> tracks path through the
#     CLI's --set override of an Int parameter.
set(SCENE_PROJECT "${WORK}/scene.psx")
execute_process(
    COMMAND "${CLI}" --new "${SCENE_PROJECT}"
            --add-images "${FIXTURES}/scene_0.jpg" "${FIXTURES}/scene_1.jpg"
                          "${FIXTURES}/scene_2.jpg"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "--new (scene) exited ${rc}\nstdout: ${out}\nstderr: ${err}")
endif()
execute_process(
    COMMAND "${CLI}" --project "${SCENE_PROJECT}" --task MatchPhotos
            --set keypoint_limit_per_mpx=10000
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "MatchPhotos exited ${rc}\nstdout: ${out}\nstderr: ${err}")
endif()

# 3c. AlignCameras on the matched scene: the whole SfM stage through the CLI.
#     This runs the incremental pipeline (initial pair, resection, triangulation,
#     bundle adjustment) and must exit 0.
execute_process(
    COMMAND "${CLI}" --project "${SCENE_PROJECT}" --task AlignCameras
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "AlignCameras exited ${rc}\nstdout: ${out}\nstderr: ${err}")
endif()

# 3d. OptimizeCameras on the aligned scene: the self-calibration stage through
#     the CLI. It refines the intrinsics, poses and points and must exit 0.
execute_process(
    COMMAND "${CLI}" --project "${SCENE_PROJECT}" --task OptimizeCameras
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "OptimizeCameras exited ${rc}\nstdout: ${out}\nstderr: ${err}")
endif()

# 3e. The milestone-9 product chain through the CLI. The aerial block is the
#     one fixture whose EXIF carries exact GPS, so AlignCameras georeferences
#     it and the DEM lands in UTM zone 32N (EPSG:32632) -- the georeferenced
#     path, not the chunk-local fallback. Steps: AddPhotos -> MatchPhotos ->
#     AlignCameras -> BuildDepthMaps -> BuildDem -> BuildOrthomosaic, then both
#     exports. Every step must exit 0 and the exports must be non-empty.
set(AERIAL_PROJECT "${WORK}/aerial.psx")
set(AERIAL_IMAGES "")
foreach(ROW RANGE 0 2)
    foreach(COL RANGE 0 3)
        list(APPEND AERIAL_IMAGES "${FIXTURES}/aerial_r${ROW}c${COL}.jpg")
    endforeach()
endforeach()

execute_process(
    COMMAND "${CLI}" --new "${AERIAL_PROJECT}" --add-images ${AERIAL_IMAGES}
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "--new (aerial) exited ${rc}\nstdout: ${out}\nstderr: ${err}")
endif()

foreach(STEP "MatchPhotos|keypoint_limit_per_mpx=10000"
              "AlignCameras|"
              "BuildDepthMaps|downscale=8"
              "BuildDem|"
              "BuildOrthomosaic|")
    # Split on "|". A trailing empty Settings field does not survive the split
    # on every CMake version -- "AlignCameras|" became a one-element list and
    # list(GET ... 1) failed with "list index: 1 out of range" (cmake 3.28) --
    # so the field is read only when the list really has two entries. CMP0007
    # pins list semantics to "empty elements count", which is what makes the
    # LENGTH check meaningful on CMP0007-unaware cmake runs.
    cmake_policy(SET CMP0007 NEW)
    string(REPLACE "|" ";" FIELDS "${STEP}")
    list(GET FIELDS 0 TASK)
    list(LENGTH FIELDS FIELD_COUNT)
    if(FIELD_COUNT GREATER 1)
        list(GET FIELDS 1 SETTING)
    else()
        set(SETTING "")
    endif()
    if(SETTING STREQUAL "")
        set(EXTRA_ARGS "")
    else()
        set(EXTRA_ARGS --set "${SETTING}")
    endif()
    execute_process(
        COMMAND "${CLI}" --project "${AERIAL_PROJECT}" --task "${TASK}" ${EXTRA_ARGS}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "${TASK} exited ${rc}\nstdout: ${out}\nstderr: ${err}")
    endif()
endforeach()

set(DEM_OUT "${WORK}/aerial_dem.tif")
set(ORTHO_OUT "${WORK}/aerial_ortho.tif")
execute_process(
    COMMAND "${CLI}" --project "${AERIAL_PROJECT}" --task ExportDem
            --set "path=${DEM_OUT}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "ExportDem exited ${rc}\nstdout: ${out}\nstderr: ${err}")
endif()

execute_process(
    COMMAND "${CLI}" --project "${AERIAL_PROJECT}" --task ExportOrthomosaic
            --set "path=${ORTHO_OUT}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "ExportOrthomosaic exited ${rc}\nstdout: ${out}\nstderr: ${err}")
endif()

foreach(OUTPUT_FILE "${DEM_OUT}" "${ORTHO_OUT}")
    if(NOT EXISTS "${OUTPUT_FILE}")
        message(FATAL_ERROR "the CLI did not write ${OUTPUT_FILE}")
    endif()
    file(SIZE "${OUTPUT_FILE}" size)
    if(size LESS 1)
        message(FATAL_ERROR "${OUTPUT_FILE} is empty")
    endif()
endforeach()

# The products are persisted in the saved project, not just in the exports.
file(GLOB AERIAL_ZIP "${WORK}/aerial.files/*")
if(NOT AERIAL_ZIP)
    message(FATAL_ERROR "the aerial project was not saved to ${WORK}/aerial.files")
endif()

# 4. An unknown task must exit 2 (UnsupportedTaskError), not 1.
execute_process(
    COMMAND "${CLI}" --project "${PROJECT}" --task NoSuchTask
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 2)
    message(FATAL_ERROR "--task NoSuchTask exited ${rc}, expected 2\nstderr: ${err}")
endif()

# 5. --list-tasks must exit 0 and name the built-in tasks.
execute_process(
    COMMAND "${CLI}" --list-tasks
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "--list-tasks exited ${rc}\nstderr: ${err}")
endif()
if(NOT out MATCHES "DetectFeatures")
    message(FATAL_ERROR "--list-tasks did not list DetectFeatures:\n${out}")
endif()
if(NOT out MATCHES "AddPhotos")
    message(FATAL_ERROR "--list-tasks did not list AddPhotos:\n${out}")
endif()
if(NOT out MATCHES "OptimizeCameras")
    message(FATAL_ERROR "--list-tasks did not list OptimizeCameras:\n${out}")
endif()

message(STATUS "cli_e2e: all checks passed")
