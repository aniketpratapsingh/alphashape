#pragma once

/// @file refine_markers_task.hpp
/// @brief RefineMarkers: multi-view consensus refinement of marker projections
/// (WP-13.9).
///
/// The decision unit is the marker, not an individual 2-D patch. The task
/// triangulates the marker from its projections, searches a depth-normalised
/// range for the physical target plane, scores every view against a reference
/// view with zero-mean normalised cross-correlation, and only then moves the
/// inlier projections. Markers with a single projection fall back to 2-D-only
/// refinement.
///
/// Only local windows are kept: every camera image is decoded once, the crops
/// needed by the markers projected in it are cut and gradient-transformed, and
/// the image is released before the next camera. A projection whose candidate
/// patch leaves a window is a non-contributor, not a contradiction. A projection
/// displaced by more than about one apparent target radius from the consensus is
/// removed, never repaired. Everything is staged as one command named
/// "Refine Markers", which snapshots the marker container once.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/commands/command.hpp"
#include "common/tasks/task.hpp"

namespace alphashape {

/// Counts reported by one RefineMarkers run.
struct RefineMarkersStats {
    std::size_t refined = 0;
    std::size_t unchanged = 0;
    std::size_t failed = 0;
    /// No usable projection (all projections disabled, or the camera could not
    /// be loaded). Every selected marker is counted in exactly one of refined,
    /// unchanged, failed or skipped.
    std::size_t skipped = 0;
    std::size_t removed_projections = 0;
    /// Total refinement rounds executed across all markers (at most 3 each).
    std::size_t rounds = 0;
    /// Estimated peak bytes: retained windows plus one image per concurrently
    /// decoded camera.
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
