#pragma once

/// @file export_stmap_task.hpp
/// @brief ExportSTMap: write a sensor's backward STMap as TIFF (WP-13.15b).
///
/// The map is generated from the active chunk's sensor calibration; the task
/// reads the document and writes one file, never mutating the chunk.

#include <string>
#include <vector>

#include "common/tasks/task.hpp"

namespace alphashape {

class ExportSTMapTask : public AbstractTask {
public:
    std::string name() const override { return "ExportSTMap"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params,
               const Progress& progress) override;
};

}  // namespace alphashape
