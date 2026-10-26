# ---------------------------------------------------------------------------
# Vulkan discovery, shader compilation and SPIR-V embedding.
#
# Three supported layouts:
#   * macOS: nothing is installed system-wide. The SDK lives in the source
#     tree's sibling tools/ directory, so the default is resolved relative to
#     the project and the tree stays relocatable.
#   * Linux: distribution packages (Vulkan headers in /usr/include, glslc and
#     spirv-val on PATH, the loader and its ICDs in the system directories).
#     Nothing is baked into the binary; the runtime loader finds the driver.
#   * Windows: vcpkg ports (vulkan-headers, vulkan-memory-allocator, shaderc,
#     spirv-tools[tools]). The headers arrive through the toolchain's
#     CMAKE_PREFIX_PATH; the tools land in <installed>/<triplet>/tools/<port>/,
#     which is not on PATH, so those directories are probed as hints.
#
# No layout links a loader. The backend dlopen()s vulkan-1.dll /
# libvulkan.so.1 / libvulkan.1.dylib at run time and resolves every entry
# point through vkGetInstanceProcAddr, so a machine without a Vulkan runtime
# starts normally and reports the CPU fallback (see src/gpu/vulkan/loader.cpp).
#
# The SDK default is therefore only consulted on Apple hosts.
# ---------------------------------------------------------------------------

set(ALPHASHAPE_VULKAN_SDK "" CACHE PATH "Vulkan SDK root (contains include/vulkan)")
if(NOT ALPHASHAPE_VULKAN_SDK AND APPLE)
    set(_alphashape_vulkan_default "${CMAKE_SOURCE_DIR}/../tools/vulkan-sdk/1.4.363.0/macOS")
    if(EXISTS "${_alphashape_vulkan_default}/include/vulkan/vulkan_core.h")
        set(ALPHASHAPE_VULKAN_SDK "${_alphashape_vulkan_default}")
    endif()
    unset(_alphashape_vulkan_default)
endif()
if(ALPHASHAPE_VULKAN_SDK)
    # Collapse "engine/../tools/..." so the baked paths read cleanly and a
    # symlinked reference tree still resolves.
    get_filename_component(ALPHASHAPE_VULKAN_SDK "${ALPHASHAPE_VULKAN_SDK}" REALPATH)
endif()

# Where a vcpkg install keeps the shader tools. The ports copy their binaries
# into <installed>/<triplet>/tools/<port>/ (vcpkg_copy_tools) rather than into
# <installed>/<triplet>/bin, and no runner puts that on PATH, so find_program()
# needs the directories as hints or the whole GPU layer silently switches off
# with "Vulkan SDK or shader tools not found in ''".
#
# Only probed when the toolchain is in play (VCPKG_INSTALLED_DIR is set by
# scripts/buildsystems/vcpkg.cmake), so a distribution install keeps using
# whatever is on PATH.
set(ALPHASHAPE_VULKAN_TOOL_HINTS "")
if(VCPKG_INSTALLED_DIR AND VCPKG_TARGET_TRIPLET)
    foreach(_port shaderc glslang spirv-tools)
        list(APPEND ALPHASHAPE_VULKAN_TOOL_HINTS
             "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/${_port}")
    endforeach()
    unset(_port)
endif()

# Derived from the SDK root, so changing ALPHASHAPE_VULKAN_SDK moves them too.
# (Individual overrides exist as cache entries only when explicitly supplied.)
#
# The clear must be an unset(), not `set(<var> "")`: under CMP0126 (NEW since
# CMake 3.21, and the policy in force at this project's minimum version) a
# plain-scope set() shadows the cache entry, so the find_program() probes below
# write the cache but the later reads still see the empty normal variable. On a
# first configure that disabled the whole GPU layer ("Vulkan SDK or shader tools
# not found in ''") even though glslc and spirv-val were on PATH. An unset()
# leaves the cache entry visible, so the probes fill it and every later read
# sees the tool. configure_file() also treats the then-undefined variables as
# empty, so gpu_config.hpp gets the same empty defaults as before.
foreach(_derived ALPHASHAPE_VULKAN_GLSLC ALPHASHAPE_VULKAN_SPIRV_VAL
        ALPHASHAPE_VULKAN_LIBRARY ALPHASHAPE_VULKAN_ICD)
    if(NOT ${_derived})
        unset(${_derived})
    endif()
endforeach()

# Tool paths are settled with probe variables: find_program() skips a name that
# is already defined, and the cache entries for the final variables always are.
if(NOT ALPHASHAPE_VULKAN_SDK)
    if(NOT ALPHASHAPE_VULKAN_GLSLC)
        # NAMES carries the bare name first, then the Windows spelling: vcpkg's
        # shaderc port installs glslc.exe.
        find_program(ALPHASHAPE_VULKAN_GLSLC_PROBE NAMES glslc glslc.exe
                     HINTS ${ALPHASHAPE_VULKAN_TOOL_HINTS})
        if(ALPHASHAPE_VULKAN_GLSLC_PROBE AND
           NOT ALPHASHAPE_VULKAN_GLSLC_PROBE MATCHES "-NOTFOUND$")
            set(ALPHASHAPE_VULKAN_GLSLC "${ALPHASHAPE_VULKAN_GLSLC_PROBE}" CACHE FILEPATH
                "glslc compiler" FORCE)
        endif()
    endif()
    if(NOT ALPHASHAPE_VULKAN_SPIRV_VAL)
        find_program(ALPHASHAPE_VULKAN_SPIRV_VAL_PROBE NAMES spirv-val spirv-val.exe
                     HINTS ${ALPHASHAPE_VULKAN_TOOL_HINTS})
        if(ALPHASHAPE_VULKAN_SPIRV_VAL_PROBE AND
           NOT ALPHASHAPE_VULKAN_SPIRV_VAL_PROBE MATCHES "-NOTFOUND$")
            set(ALPHASHAPE_VULKAN_SPIRV_VAL "${ALPHASHAPE_VULKAN_SPIRV_VAL_PROBE}" CACHE FILEPATH
                "spirv-val validator" FORCE)
        endif()
    endif()
endif()

# Shader tools: from the SDK when one is in use, otherwise from PATH.
if(NOT ALPHASHAPE_VULKAN_GLSLC)
    if(ALPHASHAPE_VULKAN_SDK)
        set(ALPHASHAPE_VULKAN_GLSLC "${ALPHASHAPE_VULKAN_SDK}/bin/glslc")
    endif()
endif()
if(NOT ALPHASHAPE_VULKAN_SPIRV_VAL)
    if(ALPHASHAPE_VULKAN_SDK)
        set(ALPHASHAPE_VULKAN_SPIRV_VAL "${ALPHASHAPE_VULKAN_SDK}/bin/spirv-val")
    endif()
endif()

# Headers: SDK include directory, or the location of the system copy.
set(ALPHASHAPE_VULKAN_INCLUDE_DIR "" CACHE PATH
    "Directory containing vulkan/vulkan_core.h")
if(ALPHASHAPE_VULKAN_INCLUDE_DIR)
    get_filename_component(ALPHASHAPE_VULKAN_INCLUDE_DIR
                           "${ALPHASHAPE_VULKAN_INCLUDE_DIR}" REALPATH)
elseif(ALPHASHAPE_VULKAN_SDK)
    set(ALPHASHAPE_VULKAN_INCLUDE_DIR "${ALPHASHAPE_VULKAN_SDK}/include")
else()
    # find_path() skips a variable that already exists in the cache, and the
    # entry above always does, so the probe uses its own name.
    #
    # HINTS names the vcpkg tree explicitly: CMAKE_PREFIX_PATH already carries
    # it, but a caller that replaces CMAKE_PREFIX_PATH (the GUI's Qt kit does)
    # would otherwise lose the headers that are installed.
    set(_alphashape_vulkan_include_hints "")
    if(VCPKG_INSTALLED_DIR AND VCPKG_TARGET_TRIPLET)
        list(APPEND _alphashape_vulkan_include_hints
             "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")
    endif()
    find_path(ALPHASHAPE_VULKAN_SYSTEM_INCLUDE_DIR vulkan/vulkan_core.h
              HINTS ${_alphashape_vulkan_include_hints})
    if(ALPHASHAPE_VULKAN_SYSTEM_INCLUDE_DIR AND
       NOT ALPHASHAPE_VULKAN_SYSTEM_INCLUDE_DIR MATCHES "-NOTFOUND$")
        set(ALPHASHAPE_VULKAN_INCLUDE_DIR "${ALPHASHAPE_VULKAN_SYSTEM_INCLUDE_DIR}"
            CACHE PATH "Directory containing vulkan/vulkan_core.h" FORCE)
    endif()
    unset(_alphashape_vulkan_include_hints)
endif()

# Loader library and ICD manifest to bake in. Only meaningful with an SDK
# (the macOS/MoltenVK case); a distribution Vulkan install is located by the
# loader at runtime through the standard soname and ICD search paths, so both
# stay empty there.
if(NOT ALPHASHAPE_VULKAN_LIBRARY AND ALPHASHAPE_VULKAN_SDK)
    if(APPLE)
        set(ALPHASHAPE_VULKAN_LIBRARY "${ALPHASHAPE_VULKAN_SDK}/lib/libvulkan.1.dylib")
    elseif(WIN32)
        set(ALPHASHAPE_VULKAN_LIBRARY "${ALPHASHAPE_VULKAN_SDK}/bin/vulkan-1.dll")
    endif()
endif()
if(NOT ALPHASHAPE_VULKAN_ICD AND ALPHASHAPE_VULKAN_SDK AND APPLE)
    set(ALPHASHAPE_VULKAN_ICD "${ALPHASHAPE_VULKAN_SDK}/share/vulkan/icd.d/MoltenVK_icd.json")
endif()

# Whether a Vulkan development environment is present. VMA may be installed
# either as <include>/vma/vk_mem_alloc.h (SDK layout) or <include>/vk_mem_alloc.h
# (distribution packages, e.g. Ubuntu's vulkan-headers).
function(alphashape_vulkan_available out_var)
    set(_ok TRUE)
    if(NOT ALPHASHAPE_VULKAN_INCLUDE_DIR)
        set(_ok FALSE)
    else()
        if(NOT EXISTS "${ALPHASHAPE_VULKAN_INCLUDE_DIR}/vulkan/vulkan_core.h")
            set(_ok FALSE)
        endif()
        if(NOT EXISTS "${ALPHASHAPE_VULKAN_INCLUDE_DIR}/vma/vk_mem_alloc.h" AND
           NOT EXISTS "${ALPHASHAPE_VULKAN_INCLUDE_DIR}/vk_mem_alloc.h")
            set(_ok FALSE)
        endif()
    endif()
    foreach(_tool "${ALPHASHAPE_VULKAN_GLSLC}" "${ALPHASHAPE_VULKAN_SPIRV_VAL}")
        if(NOT EXISTS "${_tool}")
            set(_ok FALSE)
        endif()
    endforeach()
    set(${out_var} ${_ok} PARENT_SCOPE)
endfunction()

# Compiles one .comp into a validated .spv inside the build tree.
#
# glslc's optimiser is memory-hungry: the largest kernels peak above 5 GB in a
# single process. Running a parallel build therefore needs the shader steps in
# their own job pool with one slot, or two concurrent glslc processes exhaust a
# 32 GB machine. The pool is created by the caller (alphashape_vulkan_job_pools).
function(alphashape_vulkan_job_pools)
    set_property(GLOBAL PROPERTY JOB_POOLS alphashape_shader_pool=1)
endfunction()

# alphashape_add_shader(<shader.comp> [NO_OPTIMIZE])
#
# Compiles one compute kernel to SPIR-V with glslc and validates it.
#   (default)        glslc -O   -- the performance pipeline
#   NO_OPTIMIZE      no optimiser at all
#
# First try the optimised build: if a kernel is wrong *only* optimised it is
# usually undefined behaviour the optimiser may exploit, or an assumption the
# shader makes that is not one Vulkan gives. The PatchMatch search kernels were
# on this list for a wrong random-plane normal on an NVIDIA Tesla T4, and the
# shaders' real undefined behaviour was found and fixed that way: an out-of-range
# index in a never-taken branch, aggregate `inout` search state, and a mutable RNG
# cursor whose straight-line replacement still did not change the result. Two
# kernels remain on the list, both on every platform, and the difference is not
# yet explained: on the T4 the optimised modules report normal agreement 0.92%
# (pm_init, with the depths from the same stream 100% exact) and dense agreement
# 93.4% (pm_refine), where the unoptimised modules are correct. The measurements
# and the ruled-out list are at the top of pm_math.glsl so the next diagnosis
# starts from there rather than from the beginning. The exemption cannot be keyed
# on the device: a module is compiled and embedded when the engine is built.
# NO_OPTIMIZE is therefore also the supported way to ship a kernel the optimiser
# is known to change; it costs real speed (the NV driver does not re-optimise a
# -O0 module) and that trade is made deliberately, for accuracy.
function(alphashape_add_shader shader)
    set(_opt_flag -O)
    if("NO_OPTIMIZE" IN_LIST ARGN)
        set(_opt_flag "")
    endif()
    get_filename_component(_name "${shader}" NAME_WE)
    set(_spv "${CMAKE_CURRENT_BINARY_DIR}/shaders/${_name}.spv")
    # Shared GLSL headers live beside the kernels (pm_math.glsl, pm_core.glsl);
    # glob them so editing one rebuilds every kernel that may include it.
    file(GLOB _shader_includes "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl")
    add_custom_command(
        OUTPUT "${_spv}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMAND "${ALPHASHAPE_VULKAN_GLSLC}" --target-env=vulkan1.2 ${_opt_flag}
                -fshader-stage=compute "${shader}" -o "${_spv}"
        COMMAND "${ALPHASHAPE_VULKAN_SPIRV_VAL}" --target-env vulkan1.2 "${_spv}"
        DEPENDS "${shader}" ${_shader_includes}
        COMMENT "Compiling and validating ${_name}.comp"
        JOB_POOL alphashape_shader_pool
        VERBATIM)
    set(${_name}_SPV "${_spv}" PARENT_SCOPE)
endfunction()

# Generates the embedded-SPIR-V C++ source from a list of .spv files. Runs at
# build time so editing a shader (or its .spv) regenerates the blob.
function(alphashape_embed_shaders out_cpp)
    set(_args "")
    foreach(_spv IN LISTS ARGN)
        list(APPEND _args "${_spv}")
    endforeach()
    add_custom_command(
        OUTPUT "${out_cpp}"
        COMMAND "${CMAKE_COMMAND}"
                -DALPHASHAPE_EMBED_OUTPUT=${out_cpp}
                "-DALPHASHAPE_EMBED_INPUTS=${_args}"
                -P "${CMAKE_SOURCE_DIR}/cmake/alphashape_embed_spirv.cmake"
        DEPENDS ${ARGN} "${CMAKE_SOURCE_DIR}/cmake/alphashape_embed_spirv.cmake"
        COMMENT "Embedding SPIR-V kernels"
        VERBATIM)
endfunction()
