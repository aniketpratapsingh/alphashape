#pragma once

// BuildDem: build the chunk's georeferenced elevation model (milestone 9).
//
// The task adapter over elevation::build_dem(): resolve and validate the
// parameters, run the builder against the chunk, and commit the result as a NEW
// elevation asset (appended and made active), so an error anywhere leaves the
// chunk untouched and an earlier DEM is never overwritten.

#include "common/tasks/task.hpp"

namespace alphashape {

class BuildDemTask : public AbstractTask {
public:
    std::string name() const override { return "BuildDem"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
