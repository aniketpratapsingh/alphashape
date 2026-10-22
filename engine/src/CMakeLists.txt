# Subtree layout: one directory per layer, so that each module here has a
# single, well-defined responsibility.
#
#   base      foundation, no photogrammetry knowledge
#   image     image types and pixel operations
#   model     geometry: mesh, camera, raster
#   common    document model, tasks, commands, project IO, algorithms
#   apps      front ends

add_subdirectory(base)
add_subdirectory(image)
add_subdirectory(model)
add_subdirectory(gpu)
add_subdirectory(common)
add_subdirectory(apps)
