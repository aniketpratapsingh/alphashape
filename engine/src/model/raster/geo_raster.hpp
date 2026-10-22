/// Georeferenced raster primitives: a north-up affine pixel grid and a
/// channel-interleaved sample buffer. Header-only. The GeoTIFF container that
/// carries these types lives in geotiff.hpp; this header stays free of any
/// codec dependency so downstream algorithms (DEM, ortho), the Python
/// bindings and tests can all use it without linking libtiff.
#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

namespace alphashape {

/// North-up affine grid in a projected/geographic plane. x grows with column
/// (east), y DEcreases with row (north-up). origin_* is the outer top-left
/// CORNER of pixel (0,0).
struct RasterGrid {
    int width = 0, height = 0;
    double origin_x = 0.0, origin_y = 0.0;
    double pixel_size_x = 0.0, pixel_size_y = 0.0;  // both > 0

    /// World coordinates of the CENTRE of pixel (col,row).
    Eigen::Vector2d pixel_center(int col, int row) const {
        return Eigen::Vector2d(origin_x + (static_cast<double>(col) + 0.5) * pixel_size_x,
                               origin_y - (static_cast<double>(row) + 0.5) * pixel_size_y);
    }

    /// Continuous pixel coords: returns (col_f,row_f) with the centre of pixel
    /// (c,r) at (c+0.5,r+0.5). Inverse of pixel_center.
    Eigen::Vector2d to_pixel(double x, double y) const {
        return Eigen::Vector2d((x - origin_x) / pixel_size_x, (origin_y - y) / pixel_size_y);
    }

    bool contains_pixel(int col, int row) const {
        return col >= 0 && row >= 0 && col < width && row < height;
    }

    bool operator==(const RasterGrid&) const = default;
};

/// A raster on a RasterGrid. `data` is row-major and channel-interleaved:
/// index = ((row * width) + col) * channels + ch. Sample type is the pixel
/// type: float for DEMs / ortho elevation, uint8_t for imagery.
template <typename T>
struct GeoRaster {
    RasterGrid grid;
    int channels = 1;
    std::vector<T> data;  // row-major, channel-interleaved

    T& at(int col, int row, int ch = 0) {
        return data[(static_cast<std::size_t>(row) * static_cast<std::size_t>(grid.width) +
                     static_cast<std::size_t>(col)) *
                        static_cast<std::size_t>(channels) +
                    static_cast<std::size_t>(ch)];
    }

    const T& at(int col, int row, int ch = 0) const {
        return data[(static_cast<std::size_t>(row) * static_cast<std::size_t>(grid.width) +
                     static_cast<std::size_t>(col)) *
                        static_cast<std::size_t>(channels) +
                    static_cast<std::size_t>(ch)];
    }

    bool empty() const { return data.empty(); }

    bool operator==(const GeoRaster&) const = default;
};

}  // namespace alphashape
