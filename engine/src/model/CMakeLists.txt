# alphashape_model — geometry: mesh, camera model, georeferenced raster.
# The camera model landed at milestone 2; the mesh data model, its operators
# and the PLY writer arrive at milestone 7; the raster follows at milestone 9.

add_library(alphashape_model STATIC
    camera/calibration.cpp
    geo/geodesy.cpp
    geo/transverse_mercator.cpp
    geo/projection.cpp
    mesh/marching_tetrahedra.cpp
    mesh/mesh_ops.cpp
    mesh/decimate.cpp
    mesh/ply.cpp
    mesh/obj.cpp
    mesh/uv_charts.cpp
    mesh/atlas_pack.cpp
    mesh/rasterize.cpp
    raster/geotiff.cpp
)

target_include_directories(alphashape_model PUBLIC ${ALPHASHAPE_INCLUDE_DIR})

target_link_libraries(alphashape_model
    PUBLIC
        alphashape_base
        Eigen3::Eigen
    PRIVATE
        alphashape_warnings
        TIFF::TIFF)

add_subdirectory(crs)

# The CRS engine is part of the model layer's public surface (later milestones
# put CoordinateSystem in Chunk), so model links it PRIVATE for now; a static
# library's private dependency still reaches the final link line.
if(TARGET alphashape_crs)
    target_link_libraries(alphashape_model PRIVATE alphashape_crs)
endif()
