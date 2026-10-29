#pragma once

/// @file geotiff_tags.hpp
/// @brief The process-wide libtiff registration of the GeoTIFF/GDAL tags.
///
/// libtiff has no process-global slot for custom fields: they are installed on
/// each handle, and the only supported hook is a tag extender, of which libtiff
/// keeps exactly one. Two subsystems need the same tags (the model layer's
/// GeoTIFF reader/writer and the CRS layer's geoid grid decoder), and separate
/// extenders would chain in whichever order they happen to be installed — each
/// merging its own subset again. This header is the single registration, so
/// every TIFF handle in the process sees one definition of every tag.
///
/// The layer lives in base rather than in either consumer because alphashape_crs
/// is below alphashape_model (model links crs), so a shared registration cannot
/// live in model without a cycle.
///
/// Header stays free of <tiffio.h>: only tag numbers and one function leave the
/// translation unit, so nothing downstream needs libtiff's include directory.

#include <cstdint>

namespace alphashape {

// GeoTIFF 1.1 (OGC 19-008r4) tag numbers, plus GDAL's private nodata tag.
inline constexpr std::uint32_t kTagModelPixelScale = 33550;
inline constexpr std::uint32_t kTagModelTiepoint = 33922;
inline constexpr std::uint32_t kTagGeoKeyDirectory = 34735;
/// GDAL_NODATA (ASCII): the value that marks "no sample here".
inline constexpr std::uint32_t kTagGdalNodata = 42113;

/// Install the tag extender that merges ModelPixelScale, ModelTiepoint,
/// GeoKeyDirectory and GDAL_NODATA into every TIFF handle opened afterwards.
///
/// Idempotent and thread-safe: safe to call from every open path, which is
/// where it belongs because the extender must be in place before the handle is
/// created. Chains to any extender installed by another library first, so this
/// does not steal a slot the process already uses.
void install_geotiff_tags();

}  // namespace alphashape
