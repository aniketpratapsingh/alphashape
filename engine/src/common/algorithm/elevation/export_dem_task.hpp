#pragma once

// ExportDem: write the chunk's elevation model as a float GeoTIFF (milestone 9).
//
// The file-output adapter on top of model/raster/geotiff.hpp's write_geotiff().
// Read-only w.r.t. the chunk: the raster, its grid and the EPSG recorded on the
// elevation container are passed through unchanged.

#include <string>

#include "common/tasks/task.hpp"

namespace alphashape {

/// Write one elevation asset as a single-band float32 GeoTIFF: the one named by
/// `asset_id`, or the active elevation when `asset_id` is -1 (the default).
/// Throws ValueError when an explicit id names no asset, and "No elevation
/// model" when the selected asset carries no DEM.
class ExportDemTask : public AbstractTask {
public:
    std::string name() const override { return "ExportDem"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
