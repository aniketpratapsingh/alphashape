# alphashape_platform — the only place the tree talks to the operating system.
# No source file outside this directory may include <unistd.h>, <dlfcn.h> or
# <windows.h>; everything the engine needs is spelled here once.

add_library(alphashape_platform STATIC
    dynamic_library.cpp
    temp.cpp
    env.cpp
    memory.cpp
    console.cpp
    process.cpp
    paths.cpp)

target_include_directories(alphashape_platform PUBLIC ${ALPHASHAPE_INCLUDE_DIR})

# The platform layer is the foundation of the foundation: it depends on
# nothing in the engine except the compiler flags (so its own code is held to
# the same warning set). paths.cpp resolves an address to the image holding it
# with dladdr on POSIX, which lives in libdl on systems that keep it separate
# (CMAKE_DL_LIBS is "dl" there and empty on macOS and Windows).
target_link_libraries(alphashape_platform PUBLIC ALPHASHAPE_COMPILER_FLAGS ${CMAKE_DL_LIBS})
