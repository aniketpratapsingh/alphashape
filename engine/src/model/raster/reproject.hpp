#pragma once

/// @file reproject.hpp
/// @brief Inverse-mapped raster reprojection between two coordinate systems
/// (M12 §4.10, WP-12.9).
///
/// The mapping is driven from the TARGET grid, never from the source: for every
/// target pixel centre the source coordinate is found by an inverse CRS
/// transform, and the source raster is resampled there. Driving from the source
/// would leave target pixels without a sample wherever the projection stretches.
///
/// Cost. A CRS transform is expensive, so the source coordinate is evaluated
/// exactly only on a COARSE LATTICE of target pixels and bilinearly interpolated
/// in between. The spacing starts at `kLatticeStep` (16 px) and is halved until
/// the interpolation error over the raster is at most `kMaxLatticeError` (0.1 px
/// in the source), measured at probe points against exact transforms.
///
/// Pixel-is-area. `RasterGrid::origin_*` is the outer CORNER of pixel (0, 0) and
/// a sample sits at the pixel CENTRE. The target extent is the transformed,
/// DENSIFIED source boundary (21 points per edge), not just its four corners.
///
/// Resampling. Floats (DEMs) are bilinear and nodata-aware: out-of-bounds,
/// `nodata_in` and non-finite taps carry no weight, the rest renormalise, and a
/// cell below `kMinValidWeight` is `nodata_out`. 8-bit rasters are ALWAYS
/// nearest: a blended colour was never photographed. An identical CRS yields a
/// bit-identical copy with no PROJ call and no arithmetic.

#include <cstdint>

#include "model/crs/coordinate_system.hpp"
#include "model/raster/geo_raster.hpp"

namespace alphashape {

/// How source samples are combined into a target cell.
enum class Resampling {
    Nearest,   ///< the nearest source pixel centre, no mixing (u8 imagery)
    Bilinear,  ///< nodata-aware bilinear (float DEMs)
};

/// The lattice spacing the refinement starts from, in target pixels.
inline constexpr int kLatticeStep = 16;
/// The largest acceptable source-coordinate error of the lattice, in source
/// pixels, before the spacing is halved.
inline constexpr double kMaxLatticeError = 0.1;
/// Bilinear taps whose total weight falls below this leave the cell nodata.
inline constexpr double kMinValidWeight = 0.5;

struct ReprojectOptions {
    /// Target cell size in TARGET CRS units. 0 (default) derives it from the
    /// source GSD measured at the raster centre.
    double resolution = 0.0;
    /// Resampling for float rasters. 8-bit rasters always use nearest.
    Resampling resampling = Resampling::Bilinear;
    /// Source sentinel meaning "no measurement" (float rasters).
    float nodata_in = -32767.0F;
    /// Written where a target cell has no usable source sample.
    float nodata_out = -32767.0F;
    /// 0 (default) refines from `kLatticeStep`; 1 is the exact per-pixel
    /// mapping; any other positive value pins that spacing.
    int lattice_step = 0;
};

/// The target grid for reprojecting `src` from `src_crs` into `dst_crs`: the
/// extent of the transformed, densified source boundary, its origin snapped
/// outward to a whole multiple of the cell, and a cell of `resolution`
/// (0 = the source GSD measured at the raster centre, so a geographic target
/// gets degrees and a projected one metres).
///
/// @throws ValueError when a CRS is unset/local (nothing to transform between),
///         when the transform does not cover the raster, or when no usable cell
///         size can be measured. @throws Error when PROJ is unavailable.
RasterGrid suggest_grid(const RasterGrid& src, const crs::CoordinateSystem& src_crs,
                        const crs::CoordinateSystem& dst_crs, double resolution = 0.0);

/// Reproject a float raster into `target` (a grid in `dst_crs`, normally from
/// `suggest_grid`) and resample it there. `src` must be non-empty and its data
/// must match the grid. An identical CRS with an identical grid returns a
/// bit-identical copy.
GeoRaster<float> reproject(const GeoRaster<float>& src, const crs::CoordinateSystem& src_crs,
                           const crs::CoordinateSystem& dst_crs, const RasterGrid& target,
                           const ReprojectOptions& options = {});

/// Reproject an 8-bit raster (an orthomosaic: 1, 3 or 4 channels) with nearest
/// resampling. Cells the source does not cover are written as zeroes, which for
/// the RGBA orthomosaic is exactly "no observation" (alpha 0).
GeoRaster<std::uint8_t> reproject(const GeoRaster<std::uint8_t>& src,
                                  const crs::CoordinateSystem& src_crs,
                                  const crs::CoordinateSystem& dst_crs,
                                  const RasterGrid& target,
                                  const ReprojectOptions& options = {});

}  // namespace alphashape
