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
# tests/, but these two belong to the Python front end and run here regardless.
# ---------------------------------------------------------------------------
enable_testing()

set(ALPHASHAPE_PYTHON_DIR ${CMAKE_BINARY_DIR}/python)
set(ALPHASHAPE_DATA_DIR ${PROJECT_SOURCE_DIR}/data)

add_test(
    NAME python_parity
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_parity.py)
set_tests_properties(python_parity PROPERTIES
    # The module lives in build/python; put it first on the path.
    ENVIRONMENT "PYTHONPATH=${ALPHASHAPE_PYTHON_DIR}")

# The dense-product views (Chunk.depth_maps / Chunk.point_cloud) on the same
# three-view scene the parity test uses.
add_test(
    NAME python_dense
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dense.py)
set_tests_properties(python_dense PROPERTIES
    ENVIRONMENT "PYTHONPATH=${ALPHASHAPE_PYTHON_DIR}")

# The mesh view (Chunk.model) and the ExportModel / ExportPointCloud tasks on
# the same three-view scene.
add_test(
    NAME python_mesh
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mesh.py)
set_tests_properties(python_mesh PROPERTIES
    ENVIRONMENT "PYTHONPATH=${ALPHASHAPE_PYTHON_DIR}")

# The DEM / orthomosaic views (Chunk.elevation, Chunk.orthomosaic), the four
# milestone-9 tasks through the Chunk shortcuts, and project persistence of
# both products, on the georeferenced aerial block.
add_test(
    NAME python_dem_ortho
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dem_ortho.py)
set_tests_properties(python_dem_ortho PROPERTIES
    ENVIRONMENT "PYTHONPATH=${ALPHASHAPE_PYTHON_DIR}" TIMEOUT 600)

# The M11 multi-instance asset lists (Chunk.elevations and friends), the
# "builds append, last build is active" rule, the `label` parameters and
# ExportDem's `asset_id` selection, on the georeferenced aerial block.
add_test(
    NAME python_assets
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_assets.py)
set_tests_properties(python_assets PROPERTIES
    ENVIRONMENT "PYTHONPATH=${ALPHASHAPE_PYTHON_DIR}" TIMEOUT 600)

# The chunk operations (WP-11.6): Document.duplicate_chunk / merge_chunks /
# append_project and the chunk-subset form of Document.save.
add_test(
    NAME python_chunk_ops
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_chunk_ops.py)
set_tests_properties(python_chunk_ops PROPERTIES
    ENVIRONMENT "PYTHONPATH=${ALPHASHAPE_PYTHON_DIR}" TIMEOUT 600)

# Marker-based chunk alignment (WP-11.7): Document.align_chunks, the marker
# surface, and the recorded similarity.
add_test(
    NAME python_align_chunks
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_align_chunks.py)
set_tests_properties(python_align_chunks PROPERTIES
    ENVIRONMENT "PYTHONPATH=${ALPHASHAPE_PYTHON_DIR}" TIMEOUT 600)

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
