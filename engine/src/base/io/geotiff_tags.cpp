#include "base/io/geotiff_tags.hpp"

#include <cstddef>
#include <mutex>

#include <tiffio.h>

namespace alphashape {

namespace {

/// The extender libtiff installed before us, so the hook can chain to it.
TIFFExtendProc g_parent_extender = nullptr;

/// Merges the four tags into ONE handle. Field names must have static storage
/// because they outlive every handle that refers to them.
void merge_geotiff_fields(TIFF* tif) {
    // TIFF_VARIABLE read/write counts mean "a count is passed alongside the
    // data" (TIFFSetField(tif, tag, count, ptr)), the convention libgeotiff
    // uses. GDAL_NODATA is a count-less ASCII string, matching GDAL itself.
    static char kNamePixelScale[] = "ModelPixelScaleTag";
    static char kNameTiepoint[] = "ModelTiepointTag";
    static char kNameGeoKeys[] = "GeoKeyDirectoryTag";
    static char kNameNodata[] = "GDAL_NODATA";
    static const TIFFFieldInfo fields[] = {
        {kTagModelPixelScale, TIFF_VARIABLE, TIFF_VARIABLE, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1,
         kNamePixelScale},
        {kTagModelTiepoint, TIFF_VARIABLE, TIFF_VARIABLE, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1,
         kNameTiepoint},
        {kTagGeoKeyDirectory, TIFF_VARIABLE, TIFF_VARIABLE, TIFF_SHORT, FIELD_CUSTOM, 1, 1,
         kNameGeoKeys},
        {kTagGdalNodata, TIFF_VARIABLE, TIFF_VARIABLE, TIFF_ASCII, FIELD_CUSTOM, 1, 0,
         kNameNodata},
    };
    TIFFMergeFieldInfo(tif, fields, static_cast<std::uint32_t>(std::size(fields)));
}

/// libtiff calls this from TIFFClientOpen on EVERY handle, right after the
/// standard tags are registered. That is the only supported place to add
/// custom fields, and a one-shot merge would cover only the first file.
void geotiff_tag_extender(TIFF* tif) {
    merge_geotiff_fields(tif);
    if (g_parent_extender != nullptr) {
        g_parent_extender(tif);
    }
}

}  // namespace

void install_geotiff_tags() {
    // call_once rather than a function-local static: the guard must cover the
    // store into g_parent_extender, and both callers run it on every open.
    static std::once_flag once;
    std::call_once(once, [] { g_parent_extender = TIFFSetTagExtender(geotiff_tag_extender); });
}

}  // namespace alphashape
