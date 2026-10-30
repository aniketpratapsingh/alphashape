# alphashape_crs — the M12 coordinate-system engine (PROJ-backed).
#
# The library is only built when PROJ is available; with
# -DALPHASHAPE_USE_PROJ=OFF the legacy geo::Projection stub in src/model/geo
# remains the only CRS facility and the Crs.* tests are skipped. PROJ is linked
# PRIVATE: no public header includes <proj.h> (see WP-12.1 §4.1 — pimpl), so
# nothing downstream needs PROJ's include directories or its target.
if(NOT ALPHASHAPE_USE_PROJ)
    return()
endif()

add_library(alphashape_crs STATIC
    proj_context.cpp
    coordinate_system.cpp
    builtin_crs.cpp
    transform.cpp
    datum.cpp
    local_frame.cpp
    geoid.cpp
    vertical.cpp
    euler.cpp
)

target_include_directories(alphashape_crs PUBLIC ${ALPHASHAPE_INCLUDE_DIR})

# geoid.cpp decodes its grid GeoTIFF itself (a geoid grid is one float32 band
# with two tags, and libtiff is already a configured dependency). It cannot use
# alphashape_model's GeoRaster reader: model links crs, so the reverse edge
# would be a library cycle.
target_link_libraries(alphashape_crs
    PUBLIC
        alphashape_base
    PRIVATE
        alphashape_warnings
        alphashape_proj
        TIFF::TIFF)

# The model layer's CRS bridge (types.cpp: CRS <-> CoordinateSystem) and the
# Chunk/Document accessors are written against this macro, and they live in
# alphashape_core, which must not inherit PROJ's include directory or library
# from a PRIVATE link. Publishing only the definition keeps PROJ out of every
# downstream compile while letting a TU know the engine is present.
target_compile_definitions(alphashape_crs PUBLIC ALPHASHAPE_USE_PROJ=1)

# The data-directory search resolves step 3 relative to the source tree
# (<project>/../tools/proj-data), which the interface target already defines
# through ALPHASHAPE_PROJECT_SOURCE_DIR.

# A packing step would move the staged data to <prefix>/share/proj, where the
# exe-relative search finds it; install rules arrive with M29.
