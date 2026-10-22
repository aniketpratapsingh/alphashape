# ---------------------------------------------------------------------------
# OpenMP acquisition (M29 WP-29.1).
#
# find_package(OpenMP) already knows every platform's spelling -- -fopenmp for
# GCC/Clang, -Xclang -fopenmp for clang-cl, /openmp for MSVC -- so the only
# job here is to pick the *runtime* correctly on Windows, where a clang-cl
# build must link LLVM's libomp (shipped with the Visual Studio "C++ Clang
# tools" component and as a NuGet package) rather than MSVC's vcomp.
#
# The interface target carries the flags and the ALPHASHAPE_HAVE_OPENMP define
# so a translation unit that includes base/utils/omp.hpp sees the same state as
# the build. Nothing else in the tree reads OpenMP::OpenMP_CXX directly.
# ---------------------------------------------------------------------------

add_library(ALPHASHAPE_OMP INTERFACE)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ALPHASHAPE_OMP INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(ALPHASHAPE_OMP INTERFACE ALPHASHAPE_HAVE_OPENMP=1)

    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # clang-cl + LLVM libomp: the import library is libomp.lib, which the
        # VS component installs under the LLVM toolchain's lib directory. If
        # FindOpenMP resolved a different runtime (vcomp from the MSVC
        # toolset), the pragmas would still compile but the two runtimes must
        # not both be linked -- one thread pool per process is the contract.
        if(OpenMP_CXX_LIBRARIES MATCHES "vcomp")
            message(WARNING
                "alphashape: OpenMP resolved to vcomp for a clang-cl build; "
                "install the 'C++ Clang tools for Windows' component (LLVM libomp)")
        endif()
    endif()
else()
    # Not fatal: the engine runs single-threaded and every parallel loop is
    # compiled out. The summary reports it so a build without OpenMP is never a
    # surprise.
    message(STATUS "alphashape: OpenMP not found; parallel loops are compiled out")
endif()
