#pragma once

/// @file refine_markers_task.hpp
/// @brief RefineMarkers: multi-view consensus refinement of marker projections
/// (WP-13.9).
///
/// The decision unit is the marker, not an individual 2-D patch. The task
/// triangulates the marker from its projections, searches a depth-normalised
/// range for the physical target plane, scores every view against a reference
/// view with zero-mean normalised cross-correlation, and only then moves the
/// inlier projections. A projection whose measured position disagrees with the
/// refined candidate by more than about one apparent target radius is a
/// contradiction, not noise: it is removed rather than repaired. Markers with
/// a single projection fall back to 2-D-only refinement. The whole run is one
/// snapshot-backed undo step.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/commands/command.hpp"
#include "common/tasks/task.hpp"

namespace alphashape {

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
    bool remove_invalid = true;
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
