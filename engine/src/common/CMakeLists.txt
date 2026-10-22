# alphashape_core — the document model, task layer, command layer, project IO.
# This is the architectural heart.
#
# Core knows nothing about specific algorithms. The algorithms library depends
# on core (tasks implement AbstractTask) and owns the built-in registration
# block, so the dependency runs one way: core <- algorithms <- front ends.

add_library(alphashape_core STATIC
    core/types.cpp
    core/document.cpp
    core/chunk.cpp
    core/frame.cpp
    core/assets.cpp
    core/chunk_ops.cpp
    core/chunk_geometry.cpp
    core/validate.cpp
    tasks/registry.cpp
    tasks/params.cpp
    commands/command.cpp
    commands/command_transaction.cpp
    commands/command_json.cpp
    commands/undo_stack.cpp
    commands/model_commands.cpp
    io/project_io.cpp
    io/doc_xml.cpp
    io/doc_xml_common.cpp
    io/doc_xml_v1.cpp
    io/doc_xml_v2.cpp
    io/sidecars.cpp
)

# Core carries texture pages on Chunk (common/core/chunk.hpp includes
# image/image.hpp), so it links alphashape_image PUBLIC. image depends only on
# base, so this stays acyclic: base <- image <- model/core <- algorithms.
target_include_directories(alphashape_core PUBLIC ${ALPHASHAPE_INCLUDE_DIR})
target_link_libraries(alphashape_core PUBLIC alphashape_base alphashape_model alphashape_image)

# The CRS bridge in core/types.cpp is written against alphashape_crs; with
# -DALPHASHAPE_USE_PROJ=OFF the target does not exist and the bridge compiles
# out, leaving the DTO-only CRS that the legacy stub build expects.
if(TARGET alphashape_crs)
    target_link_libraries(alphashape_core PUBLIC alphashape_crs)
endif()

add_subdirectory(algorithm)
