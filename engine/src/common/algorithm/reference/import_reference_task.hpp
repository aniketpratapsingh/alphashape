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
/// Coordinates are read in the CRS named by `crs` (empty means no conversion)
/// and converted into each row's own slot CRS: camera rows into
/// `chunk.camera_crs()`, marker rows into `chunk.marker_crs()`. A camera row
/// may also carry yaw/pitch/roll or omega/phi/kappa, which are stored as
/// `CameraReference::orientation_deg` in the chunk's rotation order.
///
/// Every row is resolved and validated before any is written, and the write is
/// one undoable command, so a malformed row leaves the chunk untouched.
/// `apply` false parses and validates the whole file without touching the
/// chunk, which is what a UI preview needs.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "common/algorithm/reference/reference_table.hpp"
#include "common/tasks/task.hpp"

namespace alphashape {

class AbstractCommand;

namespace reference {

/// One validated import row, ready to apply. `position`, `accuracy` and
/// `orientation_deg` are already expressed in the row's target slot CRS and
/// (for the angles) in the chunk's configured rotation order.
struct ImportRowPlan {
    enum class Target { Camera, Marker, NewMarker };

    Target target = Target::Marker;
    /// Index into `chunk.cameras()` / `chunk.markers()`; unused for NewMarker.
    std::size_t index = 0;
    /// Label for a newly created marker.
    std::string label;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d accuracy = Eigen::Vector3d::Constant(0.0);
    std::optional<Eigen::Vector3d> orientation_deg;
    bool enabled = true;
    /// Marker role to write. nullopt leaves an existing marker's role
    /// unchanged; a newly created marker with none reads as gcp.
    std::optional<MarkerReferenceRole> role;
};

/// Apply a validated import plan as one undo entry. The camera and marker
/// containers and the id counter are snapshotted on apply, so undo restores
/// them exactly, including removing markers the plan created.
std::unique_ptr<AbstractCommand> make_import_reference_command(
    std::int32_t chunk_id, std::vector<ImportRowPlan> rows);

}  // namespace reference

class ImportReferenceTask : public AbstractTask {
public:
    std::string name() const override { return "ImportReference"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
