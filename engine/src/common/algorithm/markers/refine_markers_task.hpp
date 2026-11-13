#pragma once

/// @file refine_markers_task.hpp
/// @brief RefineMarkers: multi-view consensus refinement of marker projections
/// (WP-13.9).
///
/// The decision unit is the marker, not an individual 2-D patch. The task
/// triangulates the marker from its projections, searches a depth-normalised
/// range for the physical target plane, scores every view against a reference
/// view with zero-mean normalised cross-correlation, and only then moves the
/// inlier projections. The printed pattern decides which refiners may move a
/// projection: a circular ring/disc is refined from its boundary, a checker
/// cross from its saddle centre. A marker with no recorded target type runs
/// both and keeps the better-supported answer; a saddle found near the click
/// is specific evidence for the checker model.
///
/// A projection is NEVER lost to an unconfident refinement. It is moved only
/// by a bounded amount (six full-resolution pixels, or half the apparent
/// target radius when the target is larger) relative to the pixel the
/// surveyor supplied, and otherwise kept exactly where it was measured.
/// Deletion is off by default; when explicitly enabled, only a measured pixel
/// that the refined candidate contradicts by more than one target radius is
/// removed. A flat patch, a low correlation or a patch that left the image
/// says the target model does not fit this view, not that the click was
/// wrong.
///
/// The marker's estimated chunk-local position is re-triangulated from the
/// accepted refined projections; its geodetic reference is never touched.
/// Markers with a single projection fall back to 2-D-only refinement. The
/// whole run is one snapshot-backed undo step, and the per-reason counts and
/// shift distribution are published on the chunk meta under `refine_markers.`
/// for a validation protocol or the GUI to read.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/commands/command.hpp"
#include "common/tasks/task.hpp"

namespace alphashape {

/// Why a projection left its marker. An unconfident refinement is never a
/// removal reason: such a projection is kept unchanged. Only an explicit
/// `remove_invalid` run records the geometric-gate reason.
enum class RefineRemovalReason : std::uint8_t {
    None = 0,
    /// The candidate reprojects farther from the measured pixel than the
    /// one-target-radius tolerance: a contradicted measurement.
    GeometricGate = 1,
    /// The physical patch has no gradient energy, so NCC carries no evidence.
    FlatPatch = 2,
    /// NCC is below `min_correlation`.
    LowCorrelation = 3,
    /// NCC is farther below the best other view than `correlation_margin`.
    CorrelationMargin = 4,
};

/// Why a whole marker was left untouched.
enum class RefineFailureReason : std::uint8_t {
    None = 0,
    NoViews = 1,
    /// No view shows a measurable printed target, so no physical scale exists.
    NoScale = 2,
    /// The initial (or refined) triangulation did not converge.
    Triangulation = 3,
    /// The depth search or per-view consensus found no two agreeing views.
    Consensus = 4,
};

/// Counts reported by one RefineMarkers run. Every selected marker is counted
/// exactly once: refined, unchanged, failed or skipped.
struct RefineMarkersStats {
    std::size_t refined = 0;
    std::size_t unchanged = 0;
    std::size_t failed = 0;
    std::size_t skipped = 0;
    /// Implicit camera lists skip an unreadable image and count it here; an
    /// explicit list fails instead.
    std::size_t unreadable_cameras = 0;
    std::size_t removed_projections = 0;

    // -- per-projection diagnostics --------------------------------------
    //
    // These exist so a validation protocol can tell "the target model does
    // not fit" from "the consensus rejects good measurements". Every
    // projection that was not moved is accounted for exactly once:
    //   evaluated = moved_projections + kept_unrefined + removed_projections.
    std::size_t evaluated_projections = 0;
    std::size_t moved_projections = 0;
    /// An evaluated projection that was left at its measured position
    /// because refinement could not be trusted, or because it was not an
    /// inlier and deletions were not requested. Never a failure.
    std::size_t kept_unrefined = 0;
    std::size_t removed_geometric_gate = 0;
    std::size_t removed_flat_patch = 0;
    std::size_t removed_low_correlation = 0;
    std::size_t removed_correlation_margin = 0;
    /// Shift magnitude over every moved projection, full-resolution pixels.
    double shift_sum_px = 0.0;
    double shift_max_px = 0.0;
    std::size_t shifts_over_2px = 0;
    std::size_t shifts_over_5px = 0;
    std::size_t shifts_over_10px = 0;

    // -- marker-level failure diagnostics --------------------------------
    std::size_t failed_no_views = 0;
    std::size_t failed_no_scale = 0;
    std::size_t failed_triangulation = 0;
    std::size_t failed_consensus = 0;

    /// Estimated peak resident bytes of decoded views plus their gradients.
    std::size_t peak_bytes = 0;
};

/// Everything the task reads from `Params`, resolved and converted.
struct RefineMarkersOptions {
    std::vector<std::string> markers;
    std::vector<std::string> cameras;
    int patch_size = 25;
    int downscale = 1;
    double min_correlation = 0.5;
    double correlation_margin = 0.15;
    int search_steps = 10;
    int max_iterations = 20;
    double tolerance_px = 0.01;
    /// Delete a projection whose measured pixel the refined candidate
    /// contradicts by more than one target radius. Off by default: on a real
    /// survey the projections are the observations, and an unconfident
    /// refinement must leave them where the surveyor clicked. Only the
    /// geometric contradiction can ever be removed; a flat patch or a low
    /// correlation describes the target model, not the measurement.
    bool remove_invalid = false;
};

/// The staged result of one refinement pass.
struct RefineMarkersPlan {
    std::unique_ptr<AbstractCommand> command;
    RefineMarkersStats stats;
};

/// Refine and stage a command for `chunk_id`. Pure with respect to the
/// document: it reads frames, the chunk and its point cloud but never applies
/// `command`. The caller owns the command: the task applies it directly, while
/// a host or test may pass it to an UndoStack to make the run one undo step.
RefineMarkersPlan plan_refine_markers(const Document& document,
                                      std::int32_t chunk_id,
                                      const RefineMarkersOptions& options);

class RefineMarkersTask : public AbstractTask {
public:
    std::string name() const override { return "RefineMarkers"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
