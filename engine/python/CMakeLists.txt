# Python bindings: the second front end onto the same task schema.
# The module dispatches through one shared parameter binder.
#
# The module is named `alphashape`. The parity test asserts that the parameter
# keys reachable here equal those written into doc.xml.

# PYBIND11_FINDPYTHON must be set BEFORE find_package(pybind11) so pybind11 uses
# FindPython (the same module we use below) instead of its own legacy probe.
# Without this the two can pick different interpreters — on a machine with more
# than one Python that produces a module linked against the wrong ABI.
set(PYBIND11_FINDPYTHON ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(alphashape_python bindings/module.cpp)

target_link_libraries(alphashape_python PRIVATE
    alphashape_algorithms
    alphashape_core
    alphashape_image)

set_target_properties(alphashape_python PROPERTIES
    OUTPUT_NAME alphashape
    # LIBRARY_OUTPUT_DIRECTORY is where a .so goes; a Windows .pyd is a DLL-like
    # module and CMake places it via RUNTIME_OUTPUT_DIRECTORY. Both are set to
    # build/python so the module is in one known place on every platform (the
    # tests below, packaging, and the installer all look there).
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)

# The .pyd/.so is loadable only by the CPython it was built against, and its
# file name carries no ABI tag off Linux. Packaging publishes this next to the
# installer so the smoke test can install the same interpreter instead of
# guessing one. Written from Python3_VERSION, which is the version pybind11 and
# the extension itself were configured with.
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dist)
file(WRITE ${CMAKE_BINARY_DIR}/dist/PYTHON-ABI.txt "${Python3_VERSION}\n")
message(STATUS "  python module ..... alphashape for Python ${Python3_VERSION}")

# ---------------------------------------------------------------------------
# Tests. TESTS=OFF (the default for this build) suppresses the C++ suite in
# tests/, but these belong to the Python front end and run here regardless.
# ---------------------------------------------------------------------------
enable_testing()

set(ALPHASHAPE_PYTHON_DIR ${CMAKE_BINARY_DIR}/python)
set(ALPHASHAPE_DATA_DIR ${PROJECT_SOURCE_DIR}/data)

# The interpreter is resolved when the test runs, not baked in here: CI
# configures the build on one runner image and unpacks it on another, and the
# Homebrew Python the configure found may not exist there. The module is per
# minor version, so the resolver insists on ${Python3_VERSION_MAJOR}.${...MINOR}.
set(ALPHASHAPE_RUN_PYTHON "${CMAKE_SOURCE_DIR}/cmake/run_python_test.cmake")
set(ALPHASHAPE_PYTHON_MINOR "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")

function(alphashape_add_python_test name script)
    cmake_parse_arguments(ARG "" "TIMEOUT" "" ${ARGN})
    add_test(
        NAME ${name}
        COMMAND ${CMAKE_COMMAND}
            "-DTEST_SCRIPT=${script}"
            "-DEXPECTED_VERSION=${ALPHASHAPE_PYTHON_MINOR}"
            "-DCONFIGURED_PYTHON=${Python3_EXECUTABLE}"
            "-DPYTHONPATH=${ALPHASHAPE_PYTHON_DIR}"
            -P "${ALPHASHAPE_RUN_PYTHON}")
    if(ARG_TIMEOUT)
        set_tests_properties(${name} PROPERTIES TIMEOUT ${ARG_TIMEOUT})
    endif()
endfunction()

# The parameter keys the module exposes must equal those written into doc.xml.
alphashape_add_python_test(python_parity ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_parity.py)

# The dense-product views (Chunk.depth_maps / Chunk.point_cloud) on the same
# three-view scene the parity test uses.
alphashape_add_python_test(python_dense ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dense.py)

# The mesh view (Chunk.model) and the ExportModel / ExportPointCloud tasks on
# the same three-view scene.
alphashape_add_python_test(python_mesh ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mesh.py)

# WP-12.7: Chunk.convertReference on the aerial block's EXIF references.
add_test(
    NAME python_reference_convert
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_reference_convert.py)
set_tests_properties(python_reference_convert PROPERTIES
    ENVIRONMENT "PYTHONPATH=${ALPHASHAPE_PYTHON_DIR}")

# The DEM / orthomosaic views (Chunk.elevation, Chunk.orthomosaic), the four
# milestone-9 tasks through the Chunk shortcuts, and project persistence of
# both products, on the georeferenced aerial block.
alphashape_add_python_test(python_dem_ortho ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dem_ortho.py
    TIMEOUT 600)

# The M11 multi-instance asset lists (Chunk.elevations and friends), the
# "builds append, last build is active" rule, the `label` parameters and
# ExportDem's `asset_id` selection, on the georeferenced aerial block.
alphashape_add_python_test(python_assets ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_assets.py
    TIMEOUT 600)

# The chunk operations (WP-11.6): Document.duplicate_chunk / merge_chunks /
# append_project and the chunk-subset form of Document.save.
alphashape_add_python_test(python_chunk_ops ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_chunk_ops.py
    TIMEOUT 600)

# Marker-based chunk alignment (WP-11.7): Document.align_chunks, the marker
# surface, and the recorded similarity.
alphashape_add_python_test(python_align_chunks ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_align_chunks.py
    TIMEOUT 600)

# The CLI end-to-end check needs the built CLI; guard on the option so a
# Python-only configure still works.
if(ALPHASHAPE_BUILD_CLI)
    add_test(
        NAME cli_e2e
        COMMAND ${CMAKE_COMMAND}
            -DCLI=$<TARGET_FILE:alphashape-cli>
            -DFIXTURES=${ALPHASHAPE_DATA_DIR}
            -DWORK=${CMAKE_CURRENT_BINARY_DIR}/cli_e2e_work
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_e2e.cmake)
endif()
