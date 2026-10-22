#pragma once

// MergeChunks: concatenate several chunks into one new chunk (M11 §4.8).
//
// Not a Metashape task; it is ours, exposed to Python as
// `Document.merge_chunks(...)` through the ordinary task path. Every id in the
// sources is remapped, so the merged chunk's id sets are disjoint from each
// other's. Mixed non-local CRSs are refused.

#include "common/tasks/task.hpp"

namespace alphashape {

class MergeChunksTask : public AbstractTask {
public:
    std::string name() const override { return "MergeChunks"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
