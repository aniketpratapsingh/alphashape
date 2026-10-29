# alphashape_base — foundation layer.
# No photogrammetry concepts here: utils, zip IO, the platform layer.

# The platform layer is a separate target so a grep for POSIX/Win32 headers has
# exactly one directory to look in (WP-29.1 acceptance).
add_subdirectory(platform)

add_library(alphashape_base STATIC
    utils/log.cpp
    utils/error.cpp
    io/zip.cpp
    io/geotiff_tags.cpp
    algorithm/predicates.cpp
    algorithm/delaunay2d.cpp
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

# io/geotiff_tags.cpp is the process's single libtiff tag registration, shared by
# the raster reader/writer (model) and the geoid grid decoder (crs). It reaches
# base because crs sits below model and both need the same one extender. PRIVATE:
# no base header exposes a libtiff type, so nothing downstream inherits it.
target_link_libraries(alphashape_base PRIVATE TIFF::TIFF)
