cmake_minimum_required(VERSION 3.24)

project(alphashape
    VERSION 0.1.0
    DESCRIPTION "Photogrammetry engine"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ALPHASHAPE_BUILD_TESTS   "Build the test suite"        ON)
# M12: PROJ is the CRS engine. Turning it OFF compiles the legacy three-CRS
# stub (projection/transverse_mercator) and skips the Crs.* suites.
option(ALPHASHAPE_USE_PROJ      "Use PROJ for coordinate systems (M12)" ON)
option(ALPHASHAPE_BUILD_PYTHON  "Build the Python bindings"   ON)
option(ALPHASHAPE_BUILD_CLI     "Build the CLI application"   ON)
# The Qt 6 desktop GUI is opt-in. With it ON, a missing Qt 6.6+ (QPalette::Accent) is a
# configuration error rather than a silent downgrade.
option(ALPHASHAPE_BUILD_GUI     "Build the Qt 6 desktop GUI (requires Qt 6.6+)" OFF)
option(ALPHASHAPE_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
# Bug-hunting configuration: ASan + UBSan (abort on undefined behaviour), and
# Eigen NaN initialisation so a matrix that is declared then partially written
# surfaces as NaN rather than whatever the heap happened to hold. See TESTING.md.
option(ALPHASHAPE_SANITIZE       "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
set(ALPHASHAPE_SANITIZE_ASAN_OPTIONS "detect_leaks=0" CACHE STRING
    "ASAN_OPTIONS (without suppressions) the sanitized tests run under; detect_leaks=1 on Linux")
option(ALPHASHAPE_EIGEN_NAN_INIT "Define EIGEN_INITIALIZE_MATRICES_BY_NAN in our translation units" OFF)

# ---------------------------------------------------------------------------
# Portable compiler and platform layer (M29 WP-29.1)
# ---------------------------------------------------------------------------
# Every policy flag -- the warning set, warnings-as-errors, FP contraction and
# the Windows CRT/header defines -- lives in one interface target and is
# selected per compiler family there (GNU/Clang, clang-cl, MSVC). In-tree
# targets link ALPHASHAPE_COMPILER_FLAGS (through alphashape_warnings, which is
# kept as an alias so the per-directory CMakeLists files need no edit).
include(${CMAKE_SOURCE_DIR}/cmake/alphashape_platform.cmake)
add_library(alphashape_warnings INTERFACE)
target_link_libraries(alphashape_warnings INTERFACE ALPHASHAPE_COMPILER_FLAGS)

# ---------------------------------------------------------------------------
# Sanitizers and Eigen NaN initialisation
# ---------------------------------------------------------------------------
# One interface target every in-tree target links, so the flags reach our
# translation units (and the final link) without touching the prebuilt
# Homebrew libraries, which are not instrumented.
add_library(alphashape_sanitize INTERFACE)
if(ALPHASHAPE_SANITIZE)
    target_compile_options(alphashape_sanitize INTERFACE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -fno-sanitize-recover=undefined)
    target_link_options(alphashape_sanitize INTERFACE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        # UBSan prints the diagnostic then aborts; the flag above is what makes
        # the test fail loudly instead of silently continuing past the bug.
        -fno-sanitize-recover=undefined)
    # Marks the instrumented build so a test can widen a *timing* budget (this
    # build is Debug + ASan + UBSan, several times slower than Release). It must
    # never gate a correctness assertion.
    target_compile_definitions(alphashape_sanitize INTERFACE ALPHASHAPE_SANITIZED_BUILD=1)
endif()
if(ALPHASHAPE_EIGEN_NAN_INIT)
    # Applied only through this interface target, i.e. only to our own TUs.
    # See the ODR note in TESTING.md: Eigen objects never cross the Ceres
    # boundary by value; Ceres is driven through raw double* blocks.
    target_compile_definitions(alphashape_sanitize INTERFACE EIGEN_INITIALIZE_MATRICES_BY_NAN)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# Homebrew installs config packages under /opt/homebrew on Apple silicon, and
# libomp is keg-only (Apple clang ships no OpenMP runtime).
if(APPLE AND EXISTS "/opt/homebrew")
    list(APPEND CMAKE_PREFIX_PATH /opt/homebrew)
    if(NOT OpenMP_ROOT AND EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_ROOT /opt/homebrew/opt/libomp)
    endif()
endif()

find_package(Threads REQUIRED)
include(${CMAKE_SOURCE_DIR}/cmake/alphashape_omp.cmake)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(ZLIB REQUIRED)

# ---------------------------------------------------------------------------
# PROJ (M12) — coordinate system engine, proj.db and the grid data directory.
# Two discovery paths, because Homebrew ships a CMake config package while the
# Debian/Ubuntu packages ship only proj.pc:
#   1. find_package(PROJ 9.3 CONFIG) -> PROJ::proj
#   2. pkg_check_modules(PROJ IMPORTED_TARGET proj) -> PkgConfig::PROJ
# Either way the interface target below is what the rest of the tree links, so
# a caller never names a package variant.
# ---------------------------------------------------------------------------
if(ALPHASHAPE_USE_PROJ)
    # Test for the *target*, never for PROJ_FOUND: pkg_check_modules caches
    # PROJ_FOUND=1 as an INTERNAL entry, so a build directory that once took
    # the pkg-config path would, on a later reconfigure, read that stale 1 and
    # then link PROJ::proj, which the config package never created.
    find_package(PROJ 9.3 CONFIG QUIET)
    if(TARGET PROJ::proj)
        set(ALPHASHAPE_PROJ_TARGET PROJ::proj)
    else()
        find_package(PkgConfig QUIET)
        if(COMMAND pkg_check_modules)
            pkg_check_modules(PROJ IMPORTED_TARGET QUIET proj>=9.3)
        endif()
        if(TARGET PkgConfig::PROJ)
            set(ALPHASHAPE_PROJ_TARGET PkgConfig::PROJ)
        else()
            message(FATAL_ERROR
                "alphashape: PROJ >= 9.3 is required for the coordinate system "
                "engine.\n"
                "  macOS:  brew install proj\n"
                "  Ubuntu: sudo apt install libproj-dev proj-data\n"
                "Configure with -DALPHASHAPE_USE_PROJ=OFF to build the legacy "
                "three-CRS stub instead (the Crs.* tests are skipped).")
        endif()
    endif()
    # Runtime version probe, used by crs::proj_version() diagnostics and by the
    # Crs.ProjVersionAtLeast9_3 test. Both discovery paths define PROJ_VERSION
    # (find_package from the config's version file, pkg_check_modules from
    # proj.pc); the imported targets do not carry it.
    if(PROJ_VERSION)
        message(STATUS "  PROJ ............. ${PROJ_VERSION} (${ALPHASHAPE_PROJ_TARGET})")
    else()
        message(STATUS "  PROJ ............. >=9.3 (${ALPHASHAPE_PROJ_TARGET})")
    endif()
else()
    message(STATUS "  PROJ ............. disabled (ALPHASHAPE_USE_PROJ=OFF, legacy stub)")
endif()

# One interface target so no module names a package variant. When PROJ is
# disabled the target is empty and the legacy stub compiles instead; the model
# library checks ALPHASHAPE_USE_PROJ in its own CMakeLists.
add_library(alphashape_proj INTERFACE)
if(ALPHASHAPE_USE_PROJ)
    target_link_libraries(alphashape_proj INTERFACE ${ALPHASHAPE_PROJ_TARGET})
    target_compile_definitions(alphashape_proj INTERFACE
        ALPHASHAPE_USE_PROJ=1
        # Used by the data-directory search (step 3: <project>/../tools/proj-data).
        ALPHASHAPE_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()

# Image codecs. These are keg-only Homebrew formulae, and CMake's FindJPEG /
# FindPNG / FindTIFF modules prefer the copies buried in
# /Library/Frameworks/Mono.framework/Headers when it is on the default search
# path — which is why the framework copy of libpng (1.4.12) would otherwise
# shadow the keg's 1.6.x. Demoting framework search *for these lookups only* is
# what makes the modules pick the Homebrew headers and libraries; the version
# assertion below turns a silent regression into a configure error.
set(ALPHASHAPE_SAVED_FIND_FRAMEWORK "${CMAKE_FIND_FRAMEWORK}")
if(APPLE AND EXISTS "/opt/homebrew")
    foreach(_alphashape_keg jpeg-turbo libpng libtiff)
        if(EXISTS "/opt/homebrew/opt/${_alphashape_keg}")
            list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/${_alphashape_keg}")
        endif()
    endforeach()
    set(CMAKE_FIND_FRAMEWORK LAST)
endif()

find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(TIFF REQUIRED)

# Qt is LGPL-3: it is always linked dynamically (the official binaries are shared).
if(ALPHASHAPE_BUILD_GUI)
    find_package(Qt6 6.6 QUIET COMPONENTS Widgets Test)
    if(NOT Qt6_FOUND)
        message(FATAL_ERROR "alphashape: ALPHASHAPE_BUILD_GUI=ON but Qt 6.6 or newer (Widgets) was "
                            "not found. Point CMAKE_PREFIX_PATH at a Qt 6 install, or pass "
                            "-DALPHASHAPE_BUILD_GUI=OFF.")
    endif()
endif()

# Restore only what was there: on non-Apple hosts CMAKE_FIND_FRAMEWORK is unset,
# and assigning an empty string back leaves an invalid value that later
# find_package(Python3) calls warn about.
if(DEFINED ALPHASHAPE_SAVED_FIND_FRAMEWORK AND
   NOT "${ALPHASHAPE_SAVED_FIND_FRAMEWORK}" STREQUAL "")
    set(CMAKE_FIND_FRAMEWORK "${ALPHASHAPE_SAVED_FIND_FRAMEWORK}")
else()
    unset(CMAKE_FIND_FRAMEWORK)
endif()

# A version below 1.6 means FindPNG matched a stale framework or system copy
# rather than the Homebrew keg, which would decode differently.
if(PNG_VERSION_STRING VERSION_LESS "1.6")
    if(APPLE)
        message(FATAL_ERROR "alphashape: found libpng ${PNG_VERSION_STRING}, expected >= 1.6 from "
                            "/opt/homebrew/opt/libpng")
    else()
        message(FATAL_ERROR "alphashape: found libpng ${PNG_VERSION_STRING}, expected >= 1.6")
    endif()
endif()

# Every module includes headers relative to src/, e.g. "common/core/chunk.hpp".
set(ALPHASHAPE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
add_subdirectory(src)

# Tests are registered from more than one directory (tests/ for the C++ suite,
# python/ for the Python parity and CLI end-to-end checks), so enable testing at
# the top level unconditionally. ALPHASHAPE_BUILD_TESTS gates only the C++ suite
# below; the Python front end's own tests always register.
enable_testing()

if(ALPHASHAPE_BUILD_TESTS)
    add_subdirectory(tests)
    if(ALPHASHAPE_BUILD_GUI)
        add_subdirectory(tests/gui)
    endif()
endif()

if(ALPHASHAPE_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# ---------------------------------------------------------------------------
# Install layout, Linux packages (.deb) and macOS bundle inputs
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    include(${CMAKE_SOURCE_DIR}/cmake/alphashape_install.cmake)
elseif(APPLE)
    include(${CMAKE_SOURCE_DIR}/cmake/alphashape_macos_bundle.cmake)
endif()

# ---------------------------------------------------------------------------
# Source guards
# ---------------------------------------------------------------------------
# tools/check_rng_order.py flags seeded scenes whose draws the compiler is free
# to reorder (two draws in one argument list or initialiser) and the
# std::*_distribution uses that are not portable across libc++ and libstdc++.
# Both made the same seed build different scenes on macOS and Linux, so this
# runs with the suite instead of only when someone remembers to look.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME rng_order_check
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/check_rng_order.py
                ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tests)
    set_tests_properties(rng_order_check PROPERTIES LABELS "source-guard")
endif()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "alphashape ${PROJECT_VERSION}")
message(STATUS "  build type ....... ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenMP ........... ${OpenMP_CXX_FOUND}")
message(STATUS "  tests ............ ${ALPHASHAPE_BUILD_TESTS}")
message(STATUS "  python bindings .. ${ALPHASHAPE_BUILD_PYTHON}")
message(STATUS "  CLI .............. ${ALPHASHAPE_BUILD_CLI}")
message(STATUS "  sanitizers ....... ${ALPHASHAPE_SANITIZE} (ASan+UBSan)")
message(STATUS "  Eigen NaN init ... ${ALPHASHAPE_EIGEN_NAN_INIT}")
message(STATUS "  CRS engine ....... ${ALPHASHAPE_USE_PROJ} (PROJ)")
message(STATUS "")
