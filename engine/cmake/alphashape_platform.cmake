# ---------------------------------------------------------------------------
# Portable compiler layer (M29 WP-29.1).
#
# One interface target carries every flag that expresses *engine policy* rather
# than a platform default, so a new compiler is added in one place:
#
#   ALPHASHAPE_COMPILER_FLAGS   warnings, warning-as-error, FP contraction
#
# Compiler families:
#   GNU | Clang (Apple or upstream)   the original GCC/Clang spelling set
#   ClangCL (MSVC == true,           the same warning set with every flag
#            compiler id Clang)      routed through the /clang: prefix, because
#                                    clang-cl reinterprets bare -W... as MSVC
#   MSVC (the pre-decided fallback)  /W4 with the noisy C4xxx codes disabled,
#                                    /fp:precise /fp:no-contract
#
# Everything here is applied to in-tree targets only. Third-party packages
# (Ceres, Eigen, libtiff, Vulkan headers, the MSVC CRT) are never re-flagged;
# where a header still leaks warnings it is wrapped at the include site, as
# the tree already does for Vulkan, VMA and Ceres.
# ---------------------------------------------------------------------------

add_library(ALPHASHAPE_COMPILER_FLAGS INTERFACE)

# ---------------------------------------------------------------------------
# Defines that every Windows translation unit needs: the CRT's secure-function
# deprecation warnings are noise we do not act on, and Windows headers define
# min/max macros that break <algorithm> and Eigen.
# ---------------------------------------------------------------------------
if(WIN32)
    # _USE_MATH_DEFINES: the UCRT's <cmath> hides M_PI and friends behind it,
    # while glibc/libc++ expose them under POSIX. Defining it here keeps the
    # tests that use M_PI portable without a per-file #define.
    target_compile_definitions(ALPHASHAPE_COMPILER_FLAGS INTERFACE
        _CRT_SECURE_NO_WARNINGS
        _SCL_SECURE_NO_WARNINGS
        _USE_MATH_DEFINES
        NOMINMAX
        WIN32_LEAN_AND_MEAN)
endif()

# ---------------------------------------------------------------------------
# Warning sets
# ---------------------------------------------------------------------------
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # --- Pure MSVC (the WP-29.2 fallback if clang-cl + libomp cannot pass) ---
    # /W4 is MSVC's practical maximum; /Wall is unusable (it flags the CRT).
    # The disabled codes are the ones that fire in correct code:
    #   C4127 conditional expression is constant       (used deliberately)
    #   C4251 needs dll-interface to be used by clients (static libs here)
    #   C4324 structure padded due to alignas            (Eigen/SIMD)
    #   C4458 declaration hides class member             (matches -Wshadow in
    #         spirit but MSVC reports it inside CRT headers we include)
    target_compile_options(ALPHASHAPE_COMPILER_FLAGS INTERFACE
        /W4 /permissive-
        /wd4127 /wd4251 /wd4324 /wd4458
        /utf-8 /Zc:__cplusplus /Zc:preprocessor /EHsc)
    if(ALPHASHAPE_WARNINGS_AS_ERRORS)
        target_compile_options(ALPHASHAPE_COMPILER_FLAGS INTERFACE /WX)
    endif()
    # MSVC's /fp:precise contracts by default in VS2022; /fp:no-contract is the
    # documented off switch, and it also pins the default evaluation choices
    # that /fp:fast would otherwise change.
    set(ALPHASHAPE_FP_CONTRACT_FLAGS /fp:precise /fp:no-contract)
else()
    # --- Clang (incl. clang-cl) and GNU ---
    # The same warning set on every platform; only the *spelling* differs for
    # clang-cl.
    #
    # clang-cl parses a leading '-' argument as its MSVC counterpart when MSVC
    # has one, and 'MSVC /Wall' means literally every warning -- including
    # -Wc++98-compat and -Wpre-c++17-compat, which fire on any C++20 source and
    # are fatal under /WX. clang-cl therefore needs the clang spelling routed
    # past that table with the '/clang:' prefix. (-Wshadow etc. have no MSVC
    # counterpart and pass through either way, but keeping the whole set on one
    # spelling is what makes this branch readable.) Verified on the VS 2022
    # 17.14 toolset (LLVM 19.1.5); see docs/plan/WINDOWS_TOOLCHAIN.md.
    if(MSVC)
        set(_alphashape_warning_flags
            /clang:-Wall /clang:-Wextra /clang:-Wpedantic
            /clang:-Wshadow /clang:-Wnon-virtual-dtor /clang:-Wold-style-cast
            /clang:-Wcast-align /clang:-Wunused /clang:-Woverloaded-virtual
            /clang:-Wconversion /clang:-Wsign-conversion
            /clang:-Wdouble-promotion /clang:-Wformat=2)
        set(_alphashape_error_flag /clang:-Werror)
    else()
        set(_alphashape_warning_flags
            -Wall -Wextra -Wpedantic
            -Wshadow -Wnon-virtual-dtor -Wold-style-cast
            -Wcast-align -Wunused -Woverloaded-virtual -Wconversion
            -Wsign-conversion -Wdouble-promotion -Wformat=2)
        set(_alphashape_error_flag -Werror)
    endif()
    target_compile_options(ALPHASHAPE_COMPILER_FLAGS INTERFACE
        ${_alphashape_warning_flags})
    if(ALPHASHAPE_WARNINGS_AS_ERRORS)
        target_compile_options(ALPHASHAPE_COMPILER_FLAGS INTERFACE
            ${_alphashape_error_flag})
    endif()

    # Numeric consistency across compilers and ISAs. Clang and GCC both contract
    # a*b+c into an FMA by default, and the two engines fuse differently
    # (notably around Ceres' inner loops and the SfM accumulations), so the same
    # source produces measurably different results on ARM64 and x86-64.
    # Disabling contraction at the source is what makes the builds agree.
    #
    # clang-cl's driver does NOT accept the bare -ffp-contract spelling (it
    # answers "unknown argument ignored", fatal under /WX). The option is a cc1
    # option, and clang-cl forwards cc1 options with the /clang: prefix; the
    # -Xclang form does not survive clang-cl's argument parser. Verified against
    # the VS 2022 17.14 LLVM toolset 19.1.5: `clang-cl -### ... /clang:
    # -ffp-contract=off` shows the flag in the cc1 command line. See
    # docs/plan/WINDOWS_TOOLCHAIN.md for the probe.
    if(MSVC)
        set(ALPHASHAPE_FP_CONTRACT_FLAGS /clang:-ffp-contract=off)
    else()
        set(ALPHASHAPE_FP_CONTRACT_FLAGS -ffp-contract=off)
    endif()
endif()

target_compile_options(ALPHASHAPE_COMPILER_FLAGS INTERFACE ${ALPHASHAPE_FP_CONTRACT_FLAGS})

# ---------------------------------------------------------------------------
# Per-file "this is somebody else's code" switch, used for the vendored VMA
# implementation. The spelling differs per family.
# ---------------------------------------------------------------------------
if(MSVC)
    set(ALPHASHAPE_THIRD_PARTY_FILE_OPTIONS "/w")
else()
    set(ALPHASHAPE_THIRD_PARTY_FILE_OPTIONS "-w")
endif()
