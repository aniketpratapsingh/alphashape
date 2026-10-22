#pragma once

// ExportOrthomosaic: write the chunk's orthomosaic as an 8-bit GeoTIFF
// (milestone 9).
//
// The file-output adapter on top of model/raster/geotiff.hpp's u8 writer.
// Read-only w.r.t. the chunk: the RGBA raster, its grid and the recorded EPSG
// pass through unchanged. The alpha channel is optional — GeoTIFF nodata for
// imagery is an alpha channel, not a nodata tag, so `alpha = false` writes RGB
// and leaves nodata cells black.

#include <string>

#include "common/tasks/task.hpp"

namespace alphashape {

/// Write one orthomosaic asset as a 3- or 4-channel 8-bit GeoTIFF: the one named
/// by `asset_id`, or the active orthomosaic when it is -1 (the default). Throws
/// ValueError when an explicit id names no asset, and "No orthomosaic" when the
/// selected asset carries none.
class ExportOrthomosaicTask : public AbstractTask {
public:
    std::string name() const override { return "ExportOrthomosaic"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
