#pragma once

/// @file export_markers_task.hpp
/// @brief ExportMarkers: write every marker projection to CSV or XML (WP-13.4).
///
/// The CSV form is one `marker,camera,u,v` row per projection; cameras are
/// written by their exact label. The XML form matches the importer and also
/// carries a marker's chunk-local `x,y,z` when it has an estimated position.
/// Numbers are written at full double round-trip precision.

#include <string>

#include "common/tasks/task.hpp"

namespace alphashape {

class ExportMarkersTask : public AbstractTask {
public:
    std::string name() const override { return "ExportMarkers"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
