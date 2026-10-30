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
/// cell below `kMinValidWeight` is `nodata_out`. 8-bit rasters
/// are ALWAYS nearest: a blended colour was never photographed. An identical CRS
/// yields a bit-identical copy with no PROJ call and no arithmetic; a same-CRS
/// re-cell keeps the source corner (see `resample_grid`).
///
/// Domain limits. The lattice is a lattice of one plane: a source that STRADDLES
/// the antimeridian needs its `+over`-style handling in the transform, because
/// the extent is a single box and a wrap would fold far-east samples onto the
/// west edge (refused points become nodata rather than a wrong wrap). Near a
/// pole the inverse transform is singular at the pole itself; target cells whose
/// centres the transform refuses are written as nodata. Both are out of scope
/// for the aerial/regional rasters this engine produces, and are why the lattice
/// error probe measures the mapping rather than assuming a global affinity.
///
/// @throws ValueError when a CRS is unset/local (nothing to transform between),
///         when the transform does not cover the raster, or when no usable cell
///         size can be measured. @throws Error when PROJ is unavailable.
RasterGrid suggest_grid(const RasterGrid& src, const crs::CoordinateSystem& src_crs,
                        const crs::CoordinateSystem& dst_crs, double resolution = 0.0);

/// The target grid for a RE-CELL inside one coordinate system: the source's own
/// north-west corner is kept, and the grid is sized to cover the source extent
/// at `cell`. A same-CRS resample is a pure affine mapping, so snapping the
/// origin to a multiple of `cell` (as `suggest_grid` must for a reprojection)
/// would only shift a real georeference by up to one cell and leave a blank
/// border row or column. `cell <= 0` or a cell equal to the source's returns
/// the source grid itself.
///
/// @throws ValueError when the source grid is empty or has a non-positive cell.
RasterGrid resample_grid(const RasterGrid& src, double cell);

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
