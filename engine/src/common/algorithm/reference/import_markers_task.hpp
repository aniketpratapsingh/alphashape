#pragma once

/// @file import_markers_task.hpp
/// @brief ImportMarkers: load marker projections from CSV or XML (WP-13.4).
///
/// CSV rows name a marker and a camera by exact label (`marker,camera,u,v`).
/// The XML form nests `<projection camera= u= v=/>` elements inside
/// `<marker label= [x= y= z=]/>`. Markers that the chunk does not carry are
/// created; a camera label that matches no camera is an error naming the line.
///
/// Every row is resolved and validated before anything is written, and the
/// write is one undoable command, so a malformed file leaves the chunk
/// untouched.

#include <string>

#include "common/tasks/task.hpp"

namespace alphashape {

class ImportMarkersTask : public AbstractTask {
public:
    std::string name() const override { return "ImportMarkers"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
