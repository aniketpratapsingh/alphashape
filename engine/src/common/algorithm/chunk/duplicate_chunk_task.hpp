#pragma once

// DuplicateChunk: a deep, id-remapped copy of one chunk (M11 §4.8, PRJ-07).
//
// The copy is added to the workspace as a new chunk with a freshly allocated
// chunk id and made active. Unlike the WP-11.5b Duplicate Chunk command (which
// copies ids verbatim so the two chunks stay comparable), this task remaps
// every chunk-local id: two chunks produced by it can be merged without a
// collision, and their object ids never overlap.

#include "common/tasks/task.hpp"

namespace alphashape {

class DuplicateChunkTask : public AbstractTask {
public:
    std::string name() const override { return "DuplicateChunk"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
