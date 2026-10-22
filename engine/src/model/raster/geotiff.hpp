/// GeoTIFF 1.1 (OGC 19-008r4) writer/reader for GeoRaster. Implements the
/// GeoTIFF subset directly against TIFF 6.0 via libtiff: ModelPixelScale (33550),
/// ModelTiepoint (33922), GeoKeyDirectory (34735) and GDAL_NODATA (42113).
/// No GDAL.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "model/raster/geo_raster.hpp"

namespace alphashape {

/// How a GeoRaster is laid out in the GeoTIFF container. `epsg` selects the
/// GeoKey model: 0 = unknown/local (user-defined model type), 4326/4xxx =
/// geographic (x = lon deg, y = lat deg), any other non-zero code = projected.
struct GeoTiffOptions {
    int epsg = 0;
    bool tiled = true;
    int tile_size = 256;  // multiple of 16
    bool deflate = true;  // ADOBE_DEFLATE + predictor (2 for u8, 3 for float)
    std::optional<double> nodata;  // GDAL_NODATA (42113), ASCII
};

/// Writes a single-band float32 GeoTIFF (DEM / elevation raster).
void write_geotiff(const std::string& path, const GeoRaster<float>& raster,
                   const GeoTiffOptions& options = {});

/// Writes an 8-bit GeoTIFF with 1 (grey), 3 (RGB) or 4 (RGB + unassociated
/// alpha) channels.
void write_geotiff(const std::string& path, const GeoRaster<std::uint8_t>& raster,
                   const GeoTiffOptions& options = {});

/// Everything about a GeoTIFF that can be learned without decoding pixels.
struct GeoTiffInfo {
    RasterGrid grid;
    int channels = 0;
    int bits_per_sample = 0;
    bool is_float = false;
    int epsg = 0;
    bool tiled = false;
    std::optional<double> nodata;
};

/// Reads the header/GeoTIFF tags of `path`. Throws ProjectIOError if the file
/// is not a readable single-planar-contiguous TIFF.
GeoTiffInfo read_geotiff_info(const std::string& path);

/// Reads a float32 single-band GeoTIFF. Throws ProjectIOError if the file is
/// not 32-bit IEEE float.
GeoRaster<float> read_geotiff_float(const std::string& path);

/// Reads an 8-bit GeoTIFF. Throws ProjectIOError if the file is not 8-bit
/// unsigned integer.
GeoRaster<std::uint8_t> read_geotiff_u8(const std::string& path);

/// ESRI world file (.tfw/.jgw): 6 lines = psx, 0, 0, -psy, x of TOP-LEFT PIXEL
/// CENTRE, y of top-left pixel centre.
void write_world_file(const std::string& path, const RasterGrid& grid);

}  // namespace alphashape
