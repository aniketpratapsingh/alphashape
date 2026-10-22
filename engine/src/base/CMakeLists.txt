# alphashape_base — foundation layer.
# No photogrammetry concepts here: utils, zip IO, the platform layer.

# The platform layer is a separate target so a grep for POSIX/Win32 headers has
# exactly one directory to look in (WP-29.1 acceptance).
add_subdirectory(platform)

add_library(alphashape_base STATIC
    utils/log.cpp
    utils/error.cpp
    io/zip.cpp
    algorithm/voxel_grid.cpp
)

target_include_directories(alphashape_base PUBLIC ${ALPHASHAPE_INCLUDE_DIR})

# alphashape_sanitize is PUBLIC so the sanitizer flags and the Eigen NaN-init
# definition reach every in-tree target that transitively links base (image,
# model, core, algorithms, cli, tests) without instrumenting Homebrew libraries.
target_link_libraries(alphashape_base PUBLIC
    alphashape_warnings
    alphashape_sanitize
    alphashape_platform
    ALPHASHAPE_OMP
    Eigen3::Eigen
    ZLIB::ZLIB
    Threads::Threads)
