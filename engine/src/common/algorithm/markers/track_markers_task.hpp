#pragma once

/// @file track_markers_task.hpp
/// @brief TrackMarkers: pyramidal Lucas-Kanade projection tracking with a
/// forward-backward consistency check (WP-13.10).
///
/// Every marker is followed through the ordered camera sequence starting at
/// the last camera (in sequence order) that already carries a projection.
/// Tracking uses a pyramidal translation-only Lucas-Kanade alignment of a
/// square window, accepted only when the forward and backward displacements
/// agree within `max_error_px`, the zero-mean NCC between the two windows is
/// at least 0.6, and neither window is flat. That triple is how occlusion is
/// detected. Existing projections are never overwritten: they are anchors.
/// A failed step may redetect the printed target in a local window. All
/// added projections form ONE CommandTransaction("Track Markers").

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/commands/command.hpp"
#include "common/tasks/task.hpp"

namespace alphashape {

/// Counts reported by one TrackMarkers run. Every selected marker is counted
/// exactly once in markers_extended / markers_unchanged / markers_skipped.
struct TrackMarkersStats {
    std::size_t tracked = 0;         ///< projections added by Lucas-Kanade
    std::size_t redetected = 0;      ///< projections added by redetection
    std::size_t misses = 0;          ///< failed tracking attempts
    std::size_t stopped_markers = 0;
    std::size_t markers_extended = 0;
    std::size_t markers_unchanged = 0;
    std::size_t markers_skipped = 0;
    std::size_t unreadable_cameras = 0;
    std::size_t peak_bytes = 0;
};

/// Everything the task reads from `Params`, resolved and converted.
struct TrackMarkersOptions {
    std::vector<std::string> cameras;
    std::string direction = "forward";
    double max_error_px = 1.0;
    int window = 21;
    int pyramid_levels = 3;
    bool redetect = true;
    std::string target_type;
    int max_gap = 3;
};

/// The staged result of one tracking pass.
struct TrackMarkersPlan {
    std::unique_ptr<AbstractCommand> command;
    TrackMarkersStats stats;
};

/// Track and stage a command for `chunk_id`. Pure with respect to the
/// document: it reads frames and the chunk but never applies `command`.
TrackMarkersPlan plan_track_markers(const Document& document,
                                    std::int32_t chunk_id,
                                    const TrackMarkersOptions& options);

class TrackMarkersTask : public AbstractTask {
public:
    std::string name() const override { return "TrackMarkers"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
