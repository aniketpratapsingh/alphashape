#pragma once

/// @file import_reference_task.hpp
/// @brief ImportReference: load surveyed camera/marker references (WP-13.2).
///
/// Reads a CSV, whitespace-delimited TXT or the common photogrammetry
/// reference XML and writes the positions into the chunk's camera and marker
/// reference slots. Labels are matched exactly: a camera first, then a marker.
/// An unmatched label becomes a new marker (unless `create_markers` is false,
/// in which case it is reported and skipped).
///
/// Coordinates are read in the CRS named by `crs` (empty means the chunk CRS)
/// and converted into the chunk CRS before they are stored. A camera row may
/// also carry yaw/pitch/roll or omega/phi/kappa, which are stored as
/// `CameraReference::orientation_deg` in the chunk's rotation order.
///
/// `apply` false parses and validates the whole file without touching the
/// chunk, which is what a UI preview needs.

#include <string>

#include "common/tasks/task.hpp"

namespace alphashape {

class ImportReferenceTask : public AbstractTask {
public:
    std::string name() const override { return "ImportReference"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
