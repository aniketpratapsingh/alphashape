#pragma once

/// @file stmap.hpp
/// @brief STMap generation and TIFF export (WP-13.15b).
///
/// An STMap is a backward resampling map: for every output pixel it stores the
/// source pixel the compositor must sample. Two conventions are load-bearing.
///
/// Coordinate convention. The map covers the *undistorted* image, an ideal
/// pinhole camera with the calibration's focal length and principal point.
/// Output pixel (col, row) has its centre at
///
///     u_o = (col + 0.5) * W / width,   v_o = (row + 0.5) * H / height
///
/// in sensor pixel units, where W and H are the calibration's image size and
/// `width` / `height` are the map size. This supports a map resampled to a
/// different raster size without changing the internal geometry.
///
/// Storage convention. `s` and `t` are normalised to the source image and use
/// a bottom-left origin, matching the compositing convention:
///
///     s = u_src / W,   t = 1 - v_src / H
///
/// so (0, 0) is the bottom-left corner and (1, 1) the top-right. Undefined or
/// non-finite projections store NaN in both channels; a resampler must treat
/// such texels as holes.

#include <string>
#include <vector>

#include "model/camera/calibration.hpp"

namespace alphashape {

/// A normalised backward resampling map.
struct STMap {
    int width = 0;
    int height = 0;
    std::vector<float> s;  ///< width * height entries, row-major
    std::vector<float> t;  ///< width * height entries, row-major
};

/// Build the backward STMap for `calibration`.
///
/// `width` / `height` of zero mean the calibration's own raster size. Throws
/// ValueError for a non-positive map size, a non-positive calibration size, or
/// a focal length that is not positive. Every output pixel is covered: pixels
/// whose source projection is undefined or non-finite carry NaN (see above).
STMap compute_stmap(const Calibration& calibration, int width = 0,
                    int height = 0);

/// Write `map` as a two-sample-per-pixel 32-bit IEEE float TIFF: contiguous
/// planar configuration, min-is-black photometric, one unspecified extra
/// sample, deflate-compressed strips. Throws Error on any libtiff failure.
void write_stmap_tiff(const std::string& path, const STMap& map);

}  // namespace alphashape
