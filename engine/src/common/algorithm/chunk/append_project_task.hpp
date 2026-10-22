#pragma once

// AppendProject: append another project's chunks to this document, remapping
// incoming chunk ids by offset (M11 §4.8).
//
// The other project may live on disk (`path`) or be an in-memory Document
// (`append_doc = false` leaves the path unused and expects an already-loaded
// source, which only the command layer can supply; the task itself always
// loads from `path`).

#include "common/tasks/task.hpp"

namespace alphashape {

class AppendProjectTask : public AbstractTask {
public:
    std::string name() const override { return "AppendProject"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
