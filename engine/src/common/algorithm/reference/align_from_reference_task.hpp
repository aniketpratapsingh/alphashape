#pragma once

// AlignFromReference: place cameras directly from their navigation metadata.
//
// Each selected camera with an antenna position and a body attitude gets the
// measured world->camera rotation (shared with OptimizeCameras) and a centre
// derived from the antenna phase centre and the sensor's lever arm:
//
//     R_wc = reference::measured_rotation(chunk, camera, transform)
//     A    = geodetic_to_ecef(lon, lat, alt)
//     C    = A - R_wc^T * lever          (lever in the camera frame)
//     pose = [R_wc | -R_wc * C_internal]
//
// The metadata is never trusted enough to skip refinement; the task only gives
// OptimizeCameras a good starting point. Existing tie points belong to the old
// poses and are cleared.
//
// A non-georeferenced chunk gets a local ENU frame: internal = metres in the
// local ENU at the centroid of the reference positions, transform =
// enu_to_ecef(centroid), CRS 4326.

#include "common/tasks/task.hpp"

namespace alphashape {

class AlignFromReferenceTask : public AbstractTask {
public:
    std::string name() const override { return "AlignFromReference"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
