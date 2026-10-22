# alphashape_image — image container, pixel operations and file codecs.
# v0 shipped a minimal in-memory image and PGM/PPM IO; milestone 2 adds the
# real codecs (JPEG/PNG/TIFF) and our own EXIF reader.

add_library(alphashape_image STATIC
    image.cpp
    operators/downsample.cpp
    io/pnm.cpp
    io/codec_common.cpp
    io/jpeg_codec.cpp
    io/png_codec.cpp
    io/tiff_codec.cpp
    io/image_io.cpp
    io/exif.cpp
)

target_include_directories(alphashape_image PUBLIC ${ALPHASHAPE_INCLUDE_DIR})
target_link_libraries(alphashape_image PUBLIC alphashape_base)

# The codec headers are an implementation detail: linking PRIVATE keeps
# jpeglib.h / png.h / tiffio.h from leaking into anything that consumes
# image_io.hpp. The warnings interface comes in PRIVATE too, matching base.
target_link_libraries(alphashape_image PRIVATE
    JPEG::JPEG
    PNG::PNG
    TIFF::TIFF
    alphashape_warnings)
