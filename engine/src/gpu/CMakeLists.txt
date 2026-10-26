# alphashape_gpu — the backend-neutral GPU compute layer (M10).
#
# The public header (gpu.hpp) has no Vulkan types. The backend dlopens the
# loader at runtime; nothing here links libvulkan or MoltenVK.
include(${CMAKE_SOURCE_DIR}/cmake/alphashape_vulkan.cmake)

option(ALPHASHAPE_GPU "Build the Vulkan compute backend" ON)

if(ALPHASHAPE_GPU)
    alphashape_vulkan_available(ALPHASHAPE_VULKAN_FOUND)
    if(NOT ALPHASHAPE_VULKAN_FOUND)
        message(STATUS "alphashape: Vulkan SDK or shader tools not found in "
                       "'${ALPHASHAPE_VULKAN_SDK}'; GPU layer disabled")
        set(ALPHASHAPE_GPU OFF)
    endif()
endif()

if(ALPHASHAPE_GPU)
    set(ALPHASHAPE_GPU_HAS_VULKAN 1)
else()
    set(ALPHASHAPE_GPU_HAS_VULKAN 0)
endif()

# The validation layer manifest sits beside the ICD manifest in the SDK. With a
# distribution Vulkan install the layer lives in the standard system directory
# that the loader searches itself, so nothing is baked in and the loader must
# not set VK_LAYER_PATH.
set(ALPHASHAPE_VULKAN_LAYER_DIR "")
if(ALPHASHAPE_VULKAN_SDK)
    set(ALPHASHAPE_VULKAN_LAYER_DIR "${ALPHASHAPE_VULKAN_SDK}/share/vulkan/explicit_layer.d")
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gpu_config.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/generated/gpu/gpu_config.hpp
               @ONLY)

set(ALPHASHAPE_GPU_SOURCES gpu.cpp)

if(ALPHASHAPE_GPU)
    # Serialise glslc: see the job-pool note in cmake/alphashape_vulkan.cmake.
    alphashape_vulkan_job_pools()
    set(_shader_sources
        shaders/saxpy.comp
        shaders/reduce_sum.comp
        shaders/image_box3.comp
        shaders/chain_add.comp
        shaders/image_sample_test.comp
        shaders/pm_init.comp
        shaders/pm_refine.comp
        shaders/pm_finalize.comp
        shaders/tvl1_dual.comp
        shaders/tvl1_primal.comp
        shaders/tvl1_delta.comp
        shaders/tvl1_reduce.comp
        shaders/poisson_spmv.comp
        shaders/poisson_precond.comp
        shaders/poisson_axpy.comp)
    set(_spv_files "")
    # pm_refine -- the kernel that does all of the per-pixel work -- is
    # optimised again. pm_init is the one kernel still exempt; see the NVIDIA
    # note in alphashape_add_shader and the analysis at rng_draw() in
    # pm_math.glsl. It is a single short dispatch per colour.
    set(_unoptimised_shaders shaders/pm_init.comp)
    foreach(_shader IN LISTS _shader_sources)
        if(_shader IN_LIST _unoptimised_shaders)
            alphashape_add_shader("${CMAKE_CURRENT_SOURCE_DIR}/${_shader}" NO_OPTIMIZE)
        else()
            alphashape_add_shader("${CMAKE_CURRENT_SOURCE_DIR}/${_shader}")
        endif()
        get_filename_component(_name "${_shader}" NAME_WE)
        list(APPEND _spv_files "${CMAKE_CURRENT_BINARY_DIR}/shaders/${_name}.spv")
    endforeach()

    alphashape_embed_shaders(${CMAKE_CURRENT_BINARY_DIR}/generated/shaders_embedded.cpp
                             ${_spv_files})

    list(APPEND ALPHASHAPE_GPU_SOURCES
        vulkan/loader.cpp
        vulkan/instance.cpp
        vulkan/vma_impl.cpp
        vulkan/context.cpp
        vulkan/dispatch.cpp
        vulkan/spirv.cpp
        vulkan/backend.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/generated/shaders_embedded.cpp)
else()
    list(APPEND ALPHASHAPE_GPU_SOURCES disabled.cpp)
endif()

add_library(alphashape_gpu STATIC ${ALPHASHAPE_GPU_SOURCES})

target_include_directories(alphashape_gpu PUBLIC
    ${ALPHASHAPE_INCLUDE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/generated)

target_link_libraries(alphashape_gpu PUBLIC
    alphashape_base
    ${CMAKE_DL_LIBS})

if(ALPHASHAPE_GPU)
    target_include_directories(alphashape_gpu PRIVATE "${ALPHASHAPE_VULKAN_INCLUDE_DIR}")
    # VK_NO_PROTOTYPES: no linked loader, everything through the function table.
    target_compile_definitions(alphashape_gpu PRIVATE
        VK_NO_PROTOTYPES=1)
    # VMA's implementation triggers a long tail of warnings we do not control
    # (nullability, unused private fields, implicit conversions). It is a
    # third-party TU, so warnings are disabled for that file only, as with the
    # other vendored headers in this tree; the rest of the backend keeps -Werror.
    set_source_files_properties(vulkan/vma_impl.cpp PROPERTIES
        COMPILE_OPTIONS "${ALPHASHAPE_THIRD_PARTY_FILE_OPTIONS}")
endif()
